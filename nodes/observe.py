"""Observation nodes: LCSPreviewColors and LCSStepObserver."""

import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image as PILImage
from comfy_api.latest import io
import folder_paths

from ..core.lcs_data import LCSData
from ..core.patchify import patchify
from ..core.timestep import get_alpha_beta, get_alpha_beta_t50, normalize_to_t50
from ..core.color_space import decode_lcs_to_hsl, hsl_to_rgb

LCS_DATA = io.Custom("LCS_DATA")

# FLUX VAE constants
SCALE_FACTOR = 0.3611
SHIFT_FACTOR = 0.1159


def _latent_to_color_preview(samples, lcs_data, sigma, upscale=8):
    """Convert latent tensor to LCS color preview image.

    samples: [B, 16, H, W] in process_in space
    Returns: [B, H_up, W_up, 3] float32 in [0,1]
    """
    device = samples.device
    dtype = samples.dtype
    ld = lcs_data.to(device, dtype)

    raw = samples / SCALE_FACTOR + SHIFT_FACTOR
    patches, h_len, w_len, _ = patchify(raw)
    projection = (patches - ld.mean) @ ld.basis

    alpha_t, beta_t = get_alpha_beta(sigma, device=device)
    alpha_t, beta_t = alpha_t.to(dtype), beta_t.to(dtype)
    alpha_50, beta_50 = get_alpha_beta_t50(device=device)
    alpha_50, beta_50 = alpha_50.to(dtype), beta_50.to(dtype)
    c_norm = normalize_to_t50(projection, alpha_t, beta_t, alpha_50, beta_50)

    B = c_norm.shape[0]
    images = []
    for b in range(B):
        c_b = c_norm[b]
        h_vals, s_vals, l_vals = decode_lcs_to_hsl(c_b, ld.anchor_lcs, ld.anchor_angles)
        r, g, b_ch = hsl_to_rgb(h_vals, s_vals, l_vals)
        rgb = torch.stack([r, g, b_ch], dim=-1).reshape(h_len, w_len, 3)
        if upscale > 1:
            rgb = F.interpolate(
                rgb.permute(2, 0, 1).unsqueeze(0),
                scale_factor=upscale, mode="nearest"
            ).squeeze(0).permute(1, 2, 0)
        images.append(rgb)

    return torch.stack(images, dim=0).clamp(0, 1).cpu().float()


class LCSPreviewColors(io.ComfyNode):
    """Visualize latent colors without VAE decoding — pure math color preview from LCS.

    Projects latent patches into the 3D LCS, normalizes to t=50, decodes to HSL,
    converts to RGB, and upscales 8x to pixel resolution. Produces a [B, H, W, 3] IMAGE.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define inputs (LATENT, LCS_DATA, sigma) and IMAGE output."""
        return io.Schema(
            node_id="LCSPreviewColors",
            display_name="LCS Preview Colors",
            category="LCS/observe",
            description="Visualize latent colors without VAE decoding — pure math color preview from LCS",
            inputs=[
                io.Latent.Input("latent", tooltip="Latent from KSampler or similar"),
                LCS_DATA.Input("lcs_data"),
                io.Float.Input("sigma", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Sigma for normalization (0.0 = final/clean, use sigma from sampler)"),
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
            ],
        )

    @classmethod
    def execute(cls, latent, lcs_data, sigma) -> io.NodeOutput:
        """Decode latent to LCS color preview. Returns IMAGE [B, H, W, 3]."""
        samples = latent["samples"]
        result = _latent_to_color_preview(samples, lcs_data, sigma, upscale=8)
        return io.NodeOutput(result)


class LCSStepObserver(io.ComfyNode):
    """Patches model to save per-step LCS color previews to ComfyUI's temp directory.

    Installs a post-CFG hook that generates a color preview image for the first
    batch item at each sampling step. Images are saved as lcs_step_NNN_sX.XXX.png.
    Does not modify the denoised prediction.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define inputs (MODEL, LCS_DATA) and MODEL output."""
        return io.Schema(
            node_id="LCSStepObserver",
            display_name="LCS Step Observer",
            category="LCS/observe",
            description="Patches model to save per-step LCS color previews to temp directory",
            inputs=[
                io.Model.Input("model"),
                LCS_DATA.Input("lcs_data"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, lcs_data) -> io.NodeOutput:
        """Clone model, attach step observer hook. Returns patched MODEL."""
        m = model.clone()
        step_counter = [0]

        def observer_fn(args):
            """Post-CFG hook: generate color preview and save to temp directory."""
            denoised = args["denoised"]
            sigma = args["sigma"]
            sigma_val = float(sigma.flatten()[0])

            # Generate color preview for first batch item
            preview = _latent_to_color_preview(
                denoised[:1], lcs_data, sigma_val, upscale=4
            )

            # Save to temp directory
            temp_dir = folder_paths.get_temp_directory()
            img_np = (preview[0].numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = PILImage.fromarray(img_np)
            filename = f"lcs_step_{step_counter[0]:03d}_s{sigma_val:.3f}.png"
            pil_img.save(os.path.join(temp_dir, filename))
            step_counter[0] += 1

            return denoised  # Don't modify

        m.set_model_sampler_post_cfg_function(observer_fn)
        return io.NodeOutput(m)
