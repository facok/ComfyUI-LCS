"""Color anchor node: correct color drift during sampling.

Adaptive version — all scheduling and filtering parameters are derived
from runtime signals (sigma schedule, local color statistics, robust
outlier detection). User controls: mode + intensity.
"""

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ..core.adaptive import compute_step_phases, compute_strength_envelope, estimate_intensity
from ..core.bilateral import bilateral_filter_lcs, estimate_bilateral_params
from ..core.relationships import (
    compute_local_relationships,
    detect_anomalies_adaptive,
    infer_color_from_neighbors,
)
from ..core.patchify import patchify, unpatchify
from ..core.sampling import (
    find_step_index,
    denoised_to_raw,
    raw_to_denoised,
    unpack_video_if_needed,
    repack_video_if_needed,
    downsample_mask,
)
from ..core.timestep import get_alpha_beta, get_alpha_beta_t50, normalize_to_t50, denormalize_from_t50

LCS_DATA = io.Custom("LCS_DATA")


def _encode_reference_to_lcs(reference_image, vae, lcs_data):
    """VAE-encode reference image to LCS coordinates.

    reference_image: [B, H, W, 3] (BHWC ComfyUI format)
    Returns (c_ref [1, L, 3], h_len, w_len) in t=50 space.
    """
    latent = vae.encode(reference_image[:1, :, :, :3])
    patches, h_len, w_len, _ = patchify(latent)
    if patches is None:
        return None, 0, 0

    device = patches.device
    dtype = patches.dtype
    ld = lcs_data.to(device, dtype)
    c_ref = (patches - ld.mean) @ ld.basis  # [1, L, 3]
    return c_ref, h_len, w_len


def _resize_color_field(c, src_h, src_w, dst_h, dst_w):
    """Bilinear resize of [B, L, 3] color field for resolution mismatch."""
    if src_h == dst_h and src_w == dst_w:
        return c
    B = c.shape[0]
    grid = c.reshape(B, src_h, src_w, 3).permute(0, 3, 1, 2)
    resized = F.interpolate(grid, size=(dst_h, dst_w), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1).reshape(B, -1, 3)


def _build_adaptive_anchor_fn(lcs_data, mode, intensity, mask,
                               c_ref=None, ref_h=0, ref_w=0, r_ref=None,
                               auto_intensity=False):
    """Build unified post_cfg_function for all anchor modes.

    Phase assignment and strength scheduling are derived from the sigma
    schedule on the first hook call. All filter/threshold parameters are
    estimated from the data at each step.

    Closure state auto-resets per graph execution (new closure = new dict).
    """
    state = {
        "phases": None,
        "envelope": None,
        "correction_index": 0,
        "r_ema": None,
        "c_ema": None,
        "prev_c_mean": None,
        "drift_samples": [],
        "auto_intensity_val": None,
    }

    def post_cfg_fn(args):
        denoised = args["denoised"]
        sigma = args["sigma"]
        model = args["model"]

        # --- Lazy init: compute phases and envelope from sigma schedule ---
        if state["phases"] is None:
            sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
            state["phases"] = compute_step_phases(sigmas, mode)
            n_correct = sum(1 for p in state["phases"] if p == "correct")
            state["envelope"] = compute_strength_envelope(n_correct)
            state["correction_index"] = 0

        # Find current step index
        sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
        step_index = find_step_index(sigma, sigmas)

        # Look up phase (guard against out-of-range)
        if step_index >= len(state["phases"]):
            return denoised
        phase = state["phases"][step_index]

        # Skip phase — return unchanged
        if phase == "skip":
            return denoised

        # --- Common pipeline: unpack → raw → patchify → project → normalize ---
        working, pack_info = unpack_video_if_needed(denoised, args)

        sigma_val = float(sigma.flatten()[0])
        device = working.device
        dtype = working.dtype

        ld = lcs_data.to(device, dtype)
        B_mat = ld.basis
        mu = ld.mean

        raw = denoised_to_raw(working, model)
        patches, h_len, w_len, extra_shape = patchify(raw)
        if patches is None:
            return denoised

        projection = (patches - mu) @ B_mat  # [B, L, 3]
        reconstruction = projection @ B_mat.T + mu
        residual = patches - reconstruction

        alpha_t, beta_t = get_alpha_beta(sigma_val, device=device)
        alpha_t, beta_t = alpha_t.to(dtype), beta_t.to(dtype)
        alpha_50, beta_50 = get_alpha_beta_t50(device=device)
        alpha_50, beta_50 = alpha_50.to(dtype), beta_50.to(dtype)

        c_norm = normalize_to_t50(projection, alpha_t, beta_t, alpha_50, beta_50)

        # --- Observe phase (self_anchor warmup): update EMA, return unchanged ---
        if phase == "observe":
            r_current = compute_local_relationships(c_norm, h_len, w_len)
            decay = 0.8
            if state["r_ema"] is None:
                state["r_ema"] = r_current.detach().clone()
                state["c_ema"] = c_norm.detach().clone()
            else:
                state["r_ema"] = decay * state["r_ema"] + (1 - decay) * r_current.detach()
                state["c_ema"] = decay * state["c_ema"] + (1 - decay) * c_norm.detach()

            # Collect step-to-step drift for auto_intensity (self_anchor)
            c_mean_now = c_norm.detach().mean(dim=1, keepdim=True)
            if auto_intensity and state["prev_c_mean"] is not None:
                drift = (c_mean_now - state["prev_c_mean"]).abs().mean().item()
                state["drift_samples"].append(drift)
            state["prev_c_mean"] = c_mean_now
            return denoised

        # --- Correct phase ---
        # Auto-intensity: compute on first correction step, cache for rest
        effective_intensity = intensity
        if auto_intensity:
            if state["auto_intensity_val"] is None:
                if mode == "self_anchor" and state["drift_samples"]:
                    drift_signal = sum(state["drift_samples"]) / len(state["drift_samples"])
                elif mode == "reference":
                    c_ref_dev = c_ref.to(device=device, dtype=dtype)
                    if ref_h != h_len or ref_w != w_len:
                        c_ref_meas = _resize_color_field(c_ref_dev, ref_h, ref_w, h_len, w_len)
                    else:
                        c_ref_meas = c_ref_dev
                    drift_signal = (c_norm - c_ref_meas).abs().mean().item()
                elif mode == "smooth":
                    sigma_s, sigma_c = estimate_bilateral_params(c_norm, h_len, w_len)
                    c_filt = bilateral_filter_lcs(c_norm, h_len, w_len, sigma_s, sigma_c)
                    drift_signal = (c_filt - c_norm).abs().mean().item()
                else:
                    drift_signal = 0.2  # fallback
                state["auto_intensity_val"] = estimate_intensity(drift_signal)
            effective_intensity = state["auto_intensity_val"]

        # Compute step strength from envelope
        ci = state["correction_index"]
        envelope = state["envelope"]
        if ci < len(envelope):
            step_strength = effective_intensity * float(envelope[ci])
        else:
            step_strength = effective_intensity
        state["correction_index"] = ci + 1

        # Self-anchor convergence damping
        if mode == "self_anchor" and state["prev_c_mean"] is not None:
            c_mean_now = c_norm.detach().mean(dim=1, keepdim=True)
            delta = (c_mean_now - state["prev_c_mean"]).abs().mean().item()
            step_strength *= min(delta / 0.1, 1.0)

        # Mode-specific correction
        if mode == "smooth":
            sigma_s, sigma_c = estimate_bilateral_params(c_norm, h_len, w_len)
            c_filtered = bilateral_filter_lcs(c_norm, h_len, w_len, sigma_s, sigma_c)
            new_c_norm = c_norm + step_strength * (c_filtered - c_norm)

        elif mode == "reference":
            c_ref_dev = c_ref.to(device=device, dtype=dtype)
            r_ref_dev = r_ref.to(device=device, dtype=dtype)
            if ref_h != h_len or ref_w != w_len:
                c_ref_resized = _resize_color_field(c_ref_dev, ref_h, ref_w, h_len, w_len)
                r_ref_resized = compute_local_relationships(c_ref_resized, h_len, w_len)
            else:
                c_ref_resized = c_ref_dev
                r_ref_resized = r_ref_dev

            B_size = c_norm.shape[0]
            c_ref_exp = c_ref_resized.expand(B_size, -1, -1)
            r_ref_exp = r_ref_resized.expand(B_size, -1, -1)

            r_current = compute_local_relationships(c_norm, h_len, w_len)
            anomaly_mag = detect_anomalies_adaptive(r_current, r_ref_exp)

            correction = c_ref_exp - c_norm
            new_c_norm = c_norm + step_strength * anomaly_mag * correction

        else:  # self_anchor
            r_current = compute_local_relationships(c_norm, h_len, w_len)

            if state["r_ema"] is None:
                # No warmup data yet — seed EMA and skip
                state["r_ema"] = r_current.detach().clone()
                state["c_ema"] = c_norm.detach().clone()
                state["prev_c_mean"] = c_norm.detach().mean(dim=1, keepdim=True)
                return denoised

            anomaly_mag = detect_anomalies_adaptive(r_current, state["r_ema"])
            c_corrected = infer_color_from_neighbors(
                c_norm, state["r_ema"], anomaly_mag, h_len, w_len
            )
            new_c_norm = c_norm + step_strength * (c_corrected - c_norm)

            # Update EMA (slow decay during correction)
            decay = 0.95
            state["r_ema"] = decay * state["r_ema"] + (1 - decay) * r_current.detach()
            state["c_ema"] = decay * state["c_ema"] + (1 - decay) * c_norm.detach()
            state["prev_c_mean"] = c_norm.detach().mean(dim=1, keepdim=True)

        # --- Apply mask ---
        if mask is not None:
            mask_flat = downsample_mask(mask, h_len, w_len, device, dtype)
            if mask_flat.shape[1] != new_c_norm.shape[1]:
                mask_flat = mask_flat[:, :new_c_norm.shape[1], :]
            new_c_norm = c_norm + mask_flat * (new_c_norm - c_norm)

        # --- Denormalize → reconstruct → unpatchify → repack ---
        new_projection = denormalize_from_t50(new_c_norm, alpha_t, beta_t, alpha_50, beta_50)
        patches_new = new_projection @ B_mat.T + mu + residual
        raw_new = unpatchify(patches_new, h_len, w_len, extra_shape)
        modified = raw_to_denoised(raw_new, model).to(dtype)
        return repack_video_if_needed(modified, pack_info)

    return post_cfg_fn


class LCSColorAnchor(io.ComfyNode):
    """Correct color drift during sampling by anchoring local color relationships.

    Four modes:
    - auto: Infer mode from connected inputs and intensity from drift signals
    - smooth: Bilateral filter smooths color discontinuities (inpainting boundaries)
    - reference: Anchor to a reference image's color relationships
    - self_anchor: Build internal color model during warmup, then correct drift

    All scheduling and filter parameters are derived adaptively from the sigma
    schedule and image content. In auto mode, intensity is also derived automatically.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LCSColorAnchor",
            display_name="LCS Color Anchor",
            category="LCS/intervention",
            description="Correct color drift during sampling by anchoring local color relationships",
            inputs=[
                io.Model.Input("model"),
                LCS_DATA.Input("lcs_data", tooltip="Calibration data from LCSLoadData"),
                io.Combo.Input("mode", options=["auto", "smooth", "reference", "self_anchor"],
                               default="auto",
                               tooltip="auto: infer mode and intensity from connected inputs; smooth: bilateral filter; reference: anchor to image; self_anchor: auto-detect drift"),
                io.Float.Input("intensity", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Correction intensity (0 = none, 1 = full)"),
                io.Vae.Input("vae", optional=True,
                             tooltip="Required for reference mode (VAE-encodes reference image)"),
                io.Image.Input("reference_image", optional=True,
                               tooltip="Reference image for reference mode"),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional mask for localized correction"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, lcs_data, mode, intensity,
                vae=None, reference_image=None, mask=None) -> io.NodeOutput:
        """Clone model, attach adaptive color anchor hook."""
        m = model.clone()

        # Resolve auto mode based on connected inputs
        auto_intensity = False
        if mode == "auto":
            auto_intensity = True
            if reference_image is not None and vae is not None:
                mode = "reference"
            elif mask is not None:
                mode = "smooth"
            else:
                mode = "self_anchor"

        if not auto_intensity and intensity < 1e-6:
            return io.NodeOutput(m)

        c_ref = None
        ref_h = 0
        ref_w = 0
        r_ref = None

        if mode == "reference":
            if vae is None or reference_image is None:
                print("[LCS Color Anchor] Reference mode requires vae and reference_image — skipping.")
                return io.NodeOutput(m)
            c_ref, ref_h, ref_w = _encode_reference_to_lcs(reference_image, vae, lcs_data)
            if c_ref is None:
                print("[LCS Color Anchor] Failed to encode reference image — skipping.")
                return io.NodeOutput(m)
            r_ref = compute_local_relationships(c_ref, ref_h, ref_w)

        hook = _build_adaptive_anchor_fn(
            lcs_data, mode, intensity, mask,
            c_ref=c_ref, ref_h=ref_h, ref_w=ref_w, r_ref=r_ref,
            auto_intensity=auto_intensity,
        )
        m.set_model_sampler_post_cfg_function(hook)
        return io.NodeOutput(m)
