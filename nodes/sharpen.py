"""Sharpness nodes: LCSSharpnessCalibrate and LCSSharpnessIntervene."""

import os

import torch
import torch.nn.functional as F
from comfy_api.latest import io
from safetensors.torch import save_file, load_file

from ..core.sharpness import SharpnessData, calibrate_sharpness
from ..core.calibration import vae_fingerprint
from ..core.patchify import patchify, unpatchify
from ..core.sampling import SCALE_FACTOR, SHIFT_FACTOR, find_step_index

SHARPNESS_DATA = io.Custom("SHARPNESS_DATA")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _save_sharpness(data: SharpnessData, path: str):
    """Save SharpnessData to safetensors file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({
        "basis": data.basis.contiguous(),
        "mean": data.mean.contiguous(),
        "pc1_std": torch.tensor([data.pc1_std]),
        "sign": torch.tensor([data.sign]),
    }, path)


def _load_sharpness(path: str) -> SharpnessData:
    """Load SharpnessData from safetensors file."""
    d = load_file(path)
    return SharpnessData(
        basis=d["basis"],
        mean=d["mean"],
        pc1_std=float(d["pc1_std"].item()),
        sign=float(d["sign"].item()),
    )


class LCSSharpnessCalibrate(io.ComfyNode):
    """Calibrate the sharpness subspace for a VAE.

    Generates blur stimuli at varying sigma levels, VAE-encodes them,
    and runs PCA to find the sharpness direction in 64D patch space.
    Result is cached per-VAE fingerprint.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LCSSharpnessCalibrate",
            display_name="LCS Sharpness Calibrate",
            category="LCS/calibration",
            description="Auto-calibrate and cache sharpness subspace data per-VAE",
            inputs=[
                io.Vae.Input("vae", tooltip="VAE model (calibration is cached per-VAE)"),
            ],
            outputs=[
                SHARPNESS_DATA.Output(display_name="sharpness_data"),
            ],
        )

    @classmethod
    def execute(cls, vae) -> io.NodeOutput:
        fp = vae_fingerprint(vae)
        cache_path = os.path.join(DATA_DIR, f"sharpness_{fp}.safetensors")

        if os.path.exists(cache_path):
            data = _load_sharpness(cache_path)
        else:
            data = calibrate_sharpness(vae)
            _save_sharpness(data, cache_path)

        return io.NodeOutput(data)


def _downsample_mask(mask, h_len, w_len, device, dtype):
    """Downsample a mask to patch grid and flatten to [1, L, 1]."""
    mask_dev = mask.to(device=device, dtype=dtype)
    if mask_dev.ndim == 3:
        mask_dev = mask_dev[:1]
    if mask_dev.ndim == 2:
        mask_4d = mask_dev.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif mask_dev.ndim == 3:
        mask_4d = mask_dev.unsqueeze(1)  # [B, 1, H, W]
    else:
        mask_4d = mask_dev
    mask_resized = F.interpolate(
        mask_4d, size=(h_len, w_len), mode="bilinear", align_corners=False
    )
    return mask_resized.reshape(1, -1, 1)  # [1, L, 1]


def _build_sharpness_fn(sharpness_data, strength, start_step, end_step, mask):
    """Build the post_cfg_function closure for sharpness intervention.

    Projects to sharpness subspace, shifts PC1, preserves residual (all other
    dimensions including color). This ensures sharpness edits don't affect color.
    """
    def post_cfg_fn(args):
        denoised = args["denoised"]  # [B, 16, H, W] in process_in space
        sigma = args["sigma"]

        # Step gating
        sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
        step_index = find_step_index(sigma, sigmas)

        if step_index < start_step or step_index > end_step:
            return denoised

        device = denoised.device
        dtype = denoised.dtype

        shd = sharpness_data.to(device, dtype)
        B_mat = shd.basis  # [64, K]
        mu = shd.mean      # [64]

        # Convert from process_in to raw VAE space
        raw = denoised / SCALE_FACTOR + SHIFT_FACTOR  # [B, 16, H, W]

        # Patchify
        patches, h_len, w_len = patchify(raw)  # [B, L, 64]

        # Project to sharpness subspace
        projection = (patches - mu) @ B_mat  # [B, L, K]

        # Compute residual (all dimensions NOT in sharpness subspace)
        reconstruction = projection @ B_mat.T + mu  # [B, L, 64]
        residual = patches - reconstruction  # [B, L, 64]

        # Edit PC1: shift by strength * sign * pc1_std
        delta = strength * shd.sign * shd.pc1_std
        new_projection = projection.clone()
        new_projection[..., 0] = new_projection[..., 0] + delta

        # Apply mask if provided
        if mask is not None:
            mask_flat = _downsample_mask(mask, h_len, w_len, device, dtype)
            if mask_flat.shape[1] != new_projection.shape[1]:
                mask_flat = mask_flat[:, :new_projection.shape[1], :]
            # Blend: masked areas get intervention, unmasked keep original
            new_projection = projection + mask_flat * (new_projection - projection)

        # Reconstruct with residual preservation
        patches_new = new_projection @ B_mat.T + mu + residual  # [B, L, 64]

        # Unpatchify
        raw_new = unpatchify(patches_new, h_len, w_len)  # [B, 16, H, W]

        # Convert back to process_in space
        return ((raw_new - SHIFT_FACTOR) * SCALE_FACTOR).to(dtype)

    return post_cfg_fn


class LCSSharpnessIntervene(io.ComfyNode):
    """Control sharpness during FLUX generation via the sharpness subspace.

    Installs a post-CFG hook that projects to the sharpness subspace, shifts
    PC1, and preserves the residual (all other latent dimensions including color).
    Positive strength = sharper, negative = blurrier.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LCSSharpnessIntervene",
            display_name="LCS Sharpness Intervene",
            category="LCS/intervention",
            description="Control sharpness during FLUX generation (positive = sharper, negative = blurrier)",
            inputs=[
                io.Model.Input("model"),
                SHARPNESS_DATA.Input("sharpness_data", tooltip="Calibration data from LCSSharpnessCalibrate"),
                io.Float.Input("strength", default=0.0, min=-2.0, max=2.0, step=0.05,
                               tooltip="Sharpness strength (>0 = sharper, <0 = blurrier, 0 = no change)"),
                io.Int.Input("start_step", default=5, min=0, max=50,
                             tooltip="First step to apply sharpness intervention"),
                io.Int.Input("end_step", default=15, min=0, max=50,
                             tooltip="Last step to apply sharpness intervention"),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional mask for localized sharpness control"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, sharpness_data, strength, start_step, end_step,
                mask=None) -> io.NodeOutput:
        m = model.clone()
        # Skip hook when strength is zero (true no-op)
        if abs(strength) > 1e-6:
            hook = _build_sharpness_fn(sharpness_data, strength, start_step, end_step, mask)
            m.set_model_sampler_post_cfg_function(hook)
        return io.NodeOutput(m)
