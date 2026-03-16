"""Intervention nodes: LCSColorIntervene, LCSColorBatch, and LCSToneAdjust."""

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ..core.lcs_data import LCSData
from ..core.patchify import patchify, unpatchify
from ..core.timestep import get_alpha_beta, get_alpha_beta_t50, normalize_to_t50, denormalize_from_t50
from ..core.color_space import hex_to_hsl, encode_hsl_to_lcs, decode_lcs_to_hsl

LCS_DATA = io.Custom("LCS_DATA")

# FLUX VAE constants
SCALE_FACTOR = 0.3611
SHIFT_FACTOR = 0.1159


def _build_post_cfg_fn(lcs_data, target_colors_hsl, strength, mode, start_step, end_step, mask):
    """Build the post_cfg_function closure for color intervention.

    target_colors_hsl: list of (h, s, l) tuples, one per batch item (or one for all).
    """
    def post_cfg_fn(args):
        """Post-CFG hook: project to LCS, apply color intervention, reconstruct."""
        denoised = args["denoised"]  # [B, 16, H, W] in process_in space
        sigma = args["sigma"]

        # Determine current step
        sigma_val = float(sigma.flatten()[0])

        # Get step index from sample_sigmas
        sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
        step_index = None
        matched = (sigmas == sigma_val).nonzero()
        if len(matched) > 0:
            step_index = matched[0].item()
        else:
            for i in range(len(sigmas) - 1):
                if (sigmas[i] - sigma_val) * (sigmas[i + 1] - sigma_val) <= 0:
                    step_index = i
                    break
        if step_index is None:
            return denoised

        # Check if in intervention range
        if step_index < start_step or step_index > end_step:
            return denoised

        device = denoised.device
        dtype = denoised.dtype

        # Move LCS data to device
        ld = lcs_data.to(device, dtype)
        B_mat = ld.basis        # [64, 3]
        mu = ld.mean            # [64]
        anchor_lcs = ld.anchor_lcs    # [8, 3]
        anchor_angles = ld.anchor_angles  # [6]

        # Convert from process_in to raw VAE space
        raw = denoised / SCALE_FACTOR + SHIFT_FACTOR  # [B, 16, H, W]

        # Patchify
        patches, h_len, w_len = patchify(raw)  # [B, L, 64]

        # Project to LCS
        projection = (patches - mu) @ B_mat  # [B, L, 3]

        # Compute residual (61D orthogonal complement)
        reconstruction = projection @ B_mat.T + mu  # [B, L, 64]
        residual = patches - reconstruction  # [B, L, 64]

        # Get timestep statistics
        alpha_t, beta_t = get_alpha_beta(sigma_val, device=device)
        alpha_t, beta_t = alpha_t.to(dtype), beta_t.to(dtype)
        alpha_50, beta_50 = get_alpha_beta_t50(device=device)
        alpha_50, beta_50 = alpha_50.to(dtype), beta_50.to(dtype)

        # Normalize to t=50
        c_norm = normalize_to_t50(projection, alpha_t, beta_t, alpha_50, beta_50)  # [B, L, 3]

        # Apply intervention per batch item
        B_size = c_norm.shape[0]
        new_c_norm = c_norm.clone()

        for b in range(B_size):
            color_idx = b if b < len(target_colors_hsl) else 0
            t_h, t_s, t_l = target_colors_hsl[color_idx]

            # Encode target color to LCS at t=50
            t_h_t = torch.tensor(t_h, device=device, dtype=dtype)
            t_s_t = torch.tensor(t_s, device=device, dtype=dtype)
            t_l_t = torch.tensor(t_l, device=device, dtype=dtype)
            target_lcs = encode_hsl_to_lcs(t_h_t, t_s_t, t_l_t, anchor_lcs, anchor_angles)  # [3]

            c_b = c_norm[b]  # [L, 3]

            if mode == "type_i":
                # Type I: direct LCS translation
                shift = target_lcs - c_b.mean(dim=0)
                new_c_norm[b] = c_b + strength * shift

            elif mode == "type_ii":
                # Type II: decode → shift in HSL → re-encode
                h_cur, s_cur, l_cur = decode_lcs_to_hsl(c_b, anchor_lcs, anchor_angles)
                # Shift towards target HSL
                h_new = t_h_t.expand_as(h_cur)
                s_new = t_s_t.expand_as(s_cur)
                l_new = t_l_t.expand_as(l_cur)
                # Interpolate in HSL
                h_interp = _hue_lerp(h_cur, h_new, strength)
                s_interp = s_cur + strength * (s_new - s_cur)
                l_interp = l_cur + strength * (l_new - l_cur)
                new_c_norm[b] = encode_hsl_to_lcs(h_interp, s_interp.clamp(0, 1),
                                                   l_interp.clamp(0, 1),
                                                   anchor_lcs, anchor_angles)

            else:  # interpolated (default)
                # gamma_t = sigma (high sigma → Type I, low sigma → Type II)
                gamma = sigma_val

                # Type I
                shift = target_lcs - c_b.mean(dim=0)
                c_type_i = c_b + strength * shift

                # Type II
                h_cur, s_cur, l_cur = decode_lcs_to_hsl(c_b, anchor_lcs, anchor_angles)
                h_new = t_h_t.expand_as(h_cur)
                s_new = t_s_t.expand_as(s_cur)
                l_new = t_l_t.expand_as(l_cur)
                h_interp = _hue_lerp(h_cur, h_new, strength)
                s_interp = s_cur + strength * (s_new - s_cur)
                l_interp = l_cur + strength * (l_new - l_cur)
                c_type_ii = encode_hsl_to_lcs(h_interp, s_interp.clamp(0, 1),
                                               l_interp.clamp(0, 1),
                                               anchor_lcs, anchor_angles)

                # Interpolate: gamma * Type_I + (1-gamma) * Type_II
                new_c_norm[b] = gamma * c_type_i + (1.0 - gamma) * c_type_ii

        # Apply mask if provided
        if mask is not None:
            mask_dev = mask.to(device=device, dtype=dtype)
            if mask_dev.ndim == 3:
                mask_dev = mask_dev[:1]  # Use first mask for all batch items
            # Downsample mask to patch grid: → [B, 1, h_len, w_len]
            if mask_dev.ndim == 2:
                mask_4d = mask_dev.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif mask_dev.ndim == 3:
                mask_4d = mask_dev.unsqueeze(1)  # [B, 1, H, W]
            else:
                mask_4d = mask_dev  # assume already 4D
            mask_resized = F.interpolate(
                mask_4d, size=(h_len, w_len), mode="bilinear", align_corners=False
            )
            # Flatten to [1, L, 1]
            mask_flat = mask_resized.reshape(1, -1, 1)
            if mask_flat.shape[1] != new_c_norm.shape[1]:
                mask_flat = mask_flat[:, :new_c_norm.shape[1], :]
            # Blend: masked areas get intervention, unmasked keep original
            new_c_norm = c_norm + mask_flat * (new_c_norm - c_norm)

        # Denormalize back to timestep t
        new_projection = denormalize_from_t50(new_c_norm, alpha_t, beta_t, alpha_50, beta_50)

        # Reconstruct patches
        patches_new = new_projection @ B_mat.T + mu + residual  # [B, L, 64]

        # Unpatchify
        raw_new = unpatchify(patches_new, h_len, w_len)  # [B, 16, H, W]

        # Convert back to process_in space
        modified = (raw_new - SHIFT_FACTOR) * SCALE_FACTOR

        return modified.to(dtype)

    return post_cfg_fn


def _hue_lerp(h1, h2, t):
    """Lerp hues on the circle [0,1], taking the shortest path."""
    diff = h2 - h1
    # Wrap to [-0.5, 0.5]
    diff = diff - (diff > 0.5).float() + (diff < -0.5).float()
    result = h1 + t * diff
    return result % 1.0


class LCSColorIntervene(io.ComfyNode):
    """Steer colors during FLUX generation via the Latent Color Subspace.

    Installs a post-CFG hook that projects the denoised prediction into the
    3D LCS, shifts it toward the target color (Type I, Type II, or interpolated),
    preserves the 61D residual, and writes the modified prediction back.
    Active only during [start_step, end_step].
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define inputs (MODEL, LCS_DATA, color, strength, mode, steps, mask) and MODEL output."""
        return io.Schema(
            node_id="LCSColorIntervene",
            display_name="LCS Color Intervene",
            category="LCS/intervention",
            description="Steer colors during FLUX generation via Latent Color Subspace",
            inputs=[
                io.Model.Input("model"),
                LCS_DATA.Input("lcs_data", tooltip="Calibration data from LCSCalibrate or LCSLoadData"),
                io.Color.Input("color", default="#FF0000", tooltip="Target color"),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="Intervention strength (1.0 = full, 0.0 = none)"),
                io.Combo.Input("mode", options=["interpolated", "type_i", "type_ii"],
                               default="interpolated",
                               tooltip="Interpolated blends Type I (LCS shift) and Type II (HSL shift)"),
                io.Int.Input("start_step", default=8, min=0, max=50,
                             tooltip="First step to apply intervention (paper optimal: 8)"),
                io.Int.Input("end_step", default=10, min=0, max=50,
                             tooltip="Last step to apply intervention (paper optimal: 10)"),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional mask for localized color control"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, lcs_data, color, strength, mode, start_step, end_step,
                mask=None) -> io.NodeOutput:
        """Clone model, attach LCS color intervention hook. Returns patched MODEL."""
        m = model.clone()
        h, s, l = hex_to_hsl(color)
        hook = _build_post_cfg_fn(lcs_data, [(h, s, l)], strength, mode, start_step, end_step, mask)
        m.set_model_sampler_post_cfg_function(hook)
        return io.NodeOutput(m)


class LCSColorBatch(io.ComfyNode):
    """Apply different target colors to each batch item for multi-color generation.

    Parses comma-separated hex colors and installs a post-CFG hook that applies
    a distinct color target per batch index. Also outputs batch_size (INT) for
    connecting to EmptyLatentImage.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define inputs (MODEL, LCS_DATA, colors string, strength, mode, steps, mask) and (MODEL, INT) outputs."""
        return io.Schema(
            node_id="LCSColorBatch",
            display_name="LCS Color Batch",
            category="LCS/intervention",
            description="Apply different target colors to each batch item for multi-color generation",
            inputs=[
                io.Model.Input("model"),
                LCS_DATA.Input("lcs_data", tooltip="Calibration data from LCSCalibrate or LCSLoadData"),
                io.String.Input("colors", default="#FF0000,#00FF00,#0000FF",
                                tooltip="Comma-separated hex colors, one per batch item"),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Combo.Input("mode", options=["interpolated", "type_i", "type_ii"],
                               default="interpolated"),
                io.Int.Input("start_step", default=8, min=0, max=50),
                io.Int.Input("end_step", default=10, min=0, max=50),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Int.Output(display_name="batch_size"),
            ],
        )

    @classmethod
    def execute(cls, model, lcs_data, colors, strength, mode, start_step, end_step,
                mask=None) -> io.NodeOutput:
        """Clone model, attach per-batch color hooks. Returns (MODEL, batch_size INT)."""
        m = model.clone()

        # Parse comma-separated hex colors
        color_list = [c.strip() for c in colors.split(",") if c.strip()]
        target_hsl = [hex_to_hsl(c) for c in color_list]
        batch_size = len(target_hsl)

        hook = _build_post_cfg_fn(lcs_data, target_hsl, strength, mode, start_step, end_step, mask)
        m.set_model_sampler_post_cfg_function(hook)
        return io.NodeOutput(m, batch_size)


def _build_tone_fn(lcs_data, contrast, brightness, saturation, color_temperature,
                   start_step, end_step, mask):
    """Build the post_cfg_function closure for tone adjustment (contrast/brightness/saturation/temperature).

    Operates directly in 3D LCS space by decomposing into lightness (projection
    onto achromatic axis) and chroma (perpendicular residual). No HSL round-trip.
    """
    def post_cfg_fn(args):
        """Post-CFG hook: project to LCS, adjust contrast/brightness/saturation, reconstruct."""
        denoised = args["denoised"]  # [B, 16, H, W] in process_in space
        sigma = args["sigma"]

        # Determine current step
        sigma_val = float(sigma.flatten()[0])

        sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
        step_index = None
        matched = (sigmas == sigma_val).nonzero()
        if len(matched) > 0:
            step_index = matched[0].item()
        else:
            for i in range(len(sigmas) - 1):
                if (sigmas[i] - sigma_val) * (sigmas[i + 1] - sigma_val) <= 0:
                    step_index = i
                    break
        if step_index is None:
            return denoised

        if step_index < start_step or step_index > end_step:
            return denoised

        device = denoised.device
        dtype = denoised.dtype

        ld = lcs_data.to(device, dtype)
        B_mat = ld.basis        # [64, 3]
        mu = ld.mean            # [64]
        anchor_lcs = ld.anchor_lcs  # [8, 3]

        # Convert from process_in to raw VAE space
        raw = denoised / SCALE_FACTOR + SHIFT_FACTOR  # [B, 16, H, W]

        # Patchify
        patches, h_len, w_len = patchify(raw)  # [B, L, 64]

        # Project to LCS
        projection = (patches - mu) @ B_mat  # [B, L, 3]

        # Compute residual (61D orthogonal complement)
        reconstruction = projection @ B_mat.T + mu  # [B, L, 64]
        residual = patches - reconstruction  # [B, L, 64]

        # Get timestep statistics
        alpha_t, beta_t = get_alpha_beta(sigma_val, device=device)
        alpha_t, beta_t = alpha_t.to(dtype), beta_t.to(dtype)
        alpha_50, beta_50 = get_alpha_beta_t50(device=device)
        alpha_50, beta_50 = alpha_50.to(dtype), beta_50.to(dtype)

        # Normalize to t=50
        c_norm = normalize_to_t50(projection, alpha_t, beta_t, alpha_50, beta_50)  # [B, L, 3]

        # Achromatic axis: black → white in LCS anchor space
        black = anchor_lcs[6]  # [3]
        white = anchor_lcs[7]  # [3]
        a = white - black      # [3]
        a_sq = (a * a).sum()   # ||a||²

        # Decompose into lightness + chroma
        # l_scalar: scalar projection along achromatic axis, [B, L]
        l_scalar = ((c_norm - black) * a).sum(dim=-1) / a_sq
        # c_L: point on achromatic axis, [B, L, 3]
        c_L = black + l_scalar.unsqueeze(-1) * a
        # chroma: perpendicular component, [B, L, 3]
        chroma = c_norm - c_L

        # Adjust lightness: contrast around per-image mean + brightness shift
        # No clamp: LCS coords are naturally unbounded during denoising (same as
        # Type I intervention), and clamping destroys highlight/shadow detail that
        # the user wants to enhance. The no-op skip in execute() handles defaults.
        l_mean = l_scalar.mean(dim=-1, keepdim=True)  # [B, 1]
        l_new = (l_scalar - l_mean) * contrast + l_mean + brightness

        # Adjust color temperature: shift chroma along warm↔cool axis
        if color_temperature != 0.0:
            # Compute chromatic projections of the 6 hue anchors (indices 0-5)
            # Red=0, Blue=1, Green=2, Magenta=3, Cyan=4, Yellow=5
            def _anchor_chroma(idx):
                anc = anchor_lcs[idx]  # [3]
                l_a = ((anc - black) * a).sum() / a_sq
                return anc - (black + l_a * a)

            warm_center = (_anchor_chroma(0) + _anchor_chroma(5)) / 2  # Red + Yellow
            cool_center = (_anchor_chroma(1) + _anchor_chroma(4)) / 2  # Blue + Cyan
            warm_dir = warm_center - cool_center
            warm_dir = warm_dir / warm_dir.norm()  # unit vector
            chroma = chroma + color_temperature * warm_dir

        # Adjust saturation
        chroma_new = chroma * saturation

        # Reconstruct in normalized LCS space
        new_c_norm = black + l_new.unsqueeze(-1) * a + chroma_new  # [B, L, 3]

        # Apply mask if provided
        if mask is not None:
            mask_dev = mask.to(device=device, dtype=dtype)
            if mask_dev.ndim == 3:
                mask_dev = mask_dev[:1]
            if mask_dev.ndim == 2:
                mask_4d = mask_dev.unsqueeze(0).unsqueeze(0)
            elif mask_dev.ndim == 3:
                mask_4d = mask_dev.unsqueeze(1)
            else:
                mask_4d = mask_dev
            mask_resized = F.interpolate(
                mask_4d, size=(h_len, w_len), mode="bilinear", align_corners=False
            )
            mask_flat = mask_resized.reshape(1, -1, 1)
            if mask_flat.shape[1] != new_c_norm.shape[1]:
                mask_flat = mask_flat[:, :new_c_norm.shape[1], :]
            new_c_norm = c_norm + mask_flat * (new_c_norm - c_norm)

        # Denormalize back to timestep t
        new_projection = denormalize_from_t50(new_c_norm, alpha_t, beta_t, alpha_50, beta_50)

        # Reconstruct patches
        patches_new = new_projection @ B_mat.T + mu + residual  # [B, L, 64]

        # Unpatchify
        raw_new = unpatchify(patches_new, h_len, w_len)  # [B, 16, H, W]

        # Convert back to process_in space
        modified = (raw_new - SHIFT_FACTOR) * SCALE_FACTOR

        return modified.to(dtype)

    return post_cfg_fn


TONE_PRESETS = {
    "Cinematic":    {"contrast": 1.20, "brightness": -0.05, "saturation": 0.90, "color_temperature": 0.15},
    "HDR":          {"contrast": 1.40, "brightness":  0.0,  "saturation": 1.20, "color_temperature": 0.0},
    "Vivid":        {"contrast": 1.10, "brightness":  0.0,  "saturation": 1.50, "color_temperature": 0.0},
    "Dramatic":     {"contrast": 1.50, "brightness": -0.10, "saturation": 0.85, "color_temperature": 0.0},
    "Low Key":      {"contrast": 1.30, "brightness": -0.20, "saturation": 0.80, "color_temperature": 0.0},
    "High Key":     {"contrast": 0.80, "brightness":  0.20, "saturation": 0.90, "color_temperature": 0.0},
    "Warm":         {"contrast": 1.0,  "brightness":  0.0,  "saturation": 1.0,  "color_temperature": 0.50},
    "Cool":         {"contrast": 1.0,  "brightness":  0.0,  "saturation": 1.0,  "color_temperature": -0.50},
    "Desaturated":  {"contrast": 1.0,  "brightness":  0.0,  "saturation": 0.40, "color_temperature": 0.0},
    "Custom":       None,
}


class LCSToneAdjust(io.ComfyNode):
    """Adjust tone (contrast, brightness, saturation, color temperature) in the Latent Color Subspace.

    Decomposes each patch into lightness (projection onto black→white axis)
    and chroma (perpendicular residual). Contrast scales lightness around its
    mean, brightness shifts it, saturation scales the chroma magnitude, and
    color temperature shifts chroma along the warm↔cool axis.
    All math is done directly in 3D LCS — no HSL round-trip needed.
    Select a preset for one-click tonal styles, or use Custom to set sliders manually.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define inputs and MODEL output for tone adjustment."""
        return io.Schema(
            node_id="LCSToneAdjust",
            display_name="LCS Tone Adjust",
            category="LCS/intervention",
            description="Adjust tone (contrast, brightness, saturation, color temperature) via Latent Color Subspace",
            inputs=[
                io.Model.Input("model"),
                LCS_DATA.Input("lcs_data", tooltip="Calibration data from LCSCalibrate or LCSLoadData"),
                io.Combo.Input("preset", options=list(TONE_PRESETS.keys()), default="Custom",
                               tooltip="Select a tonal preset or Custom to use the sliders below"),
                io.Float.Input("contrast", default=1.0, min=0.0, max=3.0, step=0.05, optional=True,
                               tooltip="[Custom only] Lightness contrast multiplier (>1 = more contrast, <1 = less, 1 = no change)"),
                io.Float.Input("brightness", default=0.0, min=-1.0, max=1.0, step=0.05, optional=True,
                               tooltip="[Custom only] Lightness shift (>0 = brighter, <0 = darker)"),
                io.Float.Input("saturation", default=1.0, min=0.0, max=3.0, step=0.05, optional=True,
                               tooltip="[Custom only] Saturation multiplier (>1 = more vivid, <1 = more muted, 0 = grayscale)"),
                io.Float.Input("color_temperature", default=0.0, min=-2.0, max=2.0, step=0.05, optional=True,
                               tooltip="[Custom only] Color temperature shift (>0 = warmer/amber, <0 = cooler/blue)"),
                io.Int.Input("start_step", default=5, min=0, max=50,
                             tooltip="First step to apply adjustment"),
                io.Int.Input("end_step", default=15, min=0, max=50,
                             tooltip="Last step to apply adjustment"),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional mask for localized adjustment"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, lcs_data, preset, start_step, end_step,
                contrast=None, brightness=None, saturation=None, color_temperature=None,
                mask=None) -> io.NodeOutput:
        """Clone model, attach LCS tone adjustment hook. Returns patched MODEL."""
        if preset != "Custom":
            p = TONE_PRESETS[preset]
            contrast = p["contrast"]
            brightness = p["brightness"]
            saturation = p["saturation"]
            color_temperature = p["color_temperature"]
        else:
            contrast = contrast if contrast is not None else 1.0
            brightness = brightness if brightness is not None else 0.0
            saturation = saturation if saturation is not None else 1.0
            color_temperature = color_temperature if color_temperature is not None else 0.0

        m = model.clone()
        # Skip hook entirely when all parameters are at default (true no-op)
        if contrast != 1.0 or brightness != 0.0 or saturation != 1.0 or color_temperature != 0.0:
            hook = _build_tone_fn(lcs_data, contrast, brightness, saturation,
                                  color_temperature, start_step, end_step, mask)
            m.set_model_sampler_post_cfg_function(hook)
        return io.NodeOutput(m)
