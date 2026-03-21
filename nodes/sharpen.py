"""Sharpness nodes: LCSSharpnessCalibrate and LCSSharpnessIntervene."""

import os

import torch
from comfy_api.latest import io
from safetensors.torch import save_file, load_file

from ..core.sharpness import SharpnessData, calibrate_sharpness
from ..core.calibration import vae_fingerprint
from ..core.patchify import patchify, unpatchify
from ..core.sampling import find_step_index, denoised_to_raw, raw_to_denoised, unpack_video_if_needed, repack_video_if_needed, downsample_mask

SHARPNESS_DATA = io.Custom("SHARPNESS_DATA")
LCS_DATA = io.Custom("LCS_DATA")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _save_sharpness(data: SharpnessData, path: str):
    """Save SharpnessData to safetensors file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({
        "basis": data.basis.contiguous(),
        "mean": data.mean.contiguous(),
        "sign": torch.tensor([data.sign]),
    }, path)


def _load_sharpness(path: str) -> SharpnessData:
    """Load SharpnessData from safetensors file."""
    d = load_file(path)
    return SharpnessData(
        basis=d["basis"],
        mean=d["mean"],
        sign=float(d["sign"].item()),
    )


class LCSSharpnessCalibrate(io.ComfyNode):
    """Calibrate the sharpness subspace for a VAE.

    Generates blur stimuli at varying sigma levels, VAE-encodes them,
    and runs PCA to find the sharpness direction in 64D patch space.
    Result is cached per-VAE fingerprint.

    When lcs_data is provided, the color component is removed during calibration,
    ensuring the sharpness PC1 is orthogonal to the color subspace.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LCSSharpnessCalibrate",
            display_name="LCS Sharpness Calibrate",
            category="LCS/calibration",
            description="Auto-calibrate and cache sharpness subspace data per-VAE. Connect lcs_data to ensure sharpness edits don't affect color.",
            inputs=[
                io.Vae.Input("vae", tooltip="VAE model (calibration is cached per-VAE)"),
                LCS_DATA.Input("lcs_data", optional=True, tooltip="Optional: remove color component to prevent color shifts"),
            ],
            outputs=[
                SHARPNESS_DATA.Output(display_name="sharpness_data"),
            ],
        )

    @classmethod
    def execute(cls, vae, lcs_data=None) -> io.NodeOutput:
        fp = vae_fingerprint(vae)
        suffix = "_lcs" if lcs_data is not None else ""
        cache_path = os.path.join(DATA_DIR, f"sharpness_{fp}{suffix}.safetensors")

        if os.path.exists(cache_path):
            data = _load_sharpness(cache_path)
        else:
            data = calibrate_sharpness(vae, lcs_data=lcs_data)
            _save_sharpness(data, cache_path)

        return io.NodeOutput(data)


def _build_sharpness_fn(sharpness_data, strength, start_step, end_step, mask):
    """Build the post_cfg_function closure for sharpness intervention.

    Simple and correct approach: patches_new = patches + edit_vec.
    Adding a vector along one direction automatically preserves all other
    dimensions (residual preservation by construction). No need for explicit
    projection/residual/reconstruction.

    The sharpness basis is calibrated with LCS color removal (if lcs_data was
    provided at calibration time), so pc1_dir is already orthogonal to color.
    At intervention time, we just add delta along that direction.
    """
    # Precompute constant edit vector once (not per-step).
    # Remove DC component from pc1_dir to prevent brightness shift.
    pc1_dir = sharpness_data.basis[:, 0].clone()
    pc1_dir = pc1_dir - pc1_dir.mean()
    edit_vec = (strength * sharpness_data.sign) * pc1_dir  # [64], on CPU

    def post_cfg_fn(args):
        denoised = args["denoised"]
        sigma = args["sigma"]
        model = args["model"]

        # Step gating
        sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
        step_index = find_step_index(sigma, sigmas)

        if step_index < start_step or step_index > end_step:
            return denoised

        # Unpack LTXAV packed format if needed
        working, pack_info = unpack_video_if_needed(denoised, args)

        device = working.device
        dtype = working.dtype

        # Move edit vector to device/dtype (short-circuits if already there)
        ev = edit_vec.to(device=device, dtype=dtype)

        # Convert from process_in to raw VAE space
        raw = denoised_to_raw(working, model)

        # Patchify
        patches, h_len, w_len, extra_shape = patchify(raw)
        if patches is None:
            return denoised  # Incompatible latent format

        # Apply sharpness edit
        if mask is not None:
            mask_flat = downsample_mask(mask, h_len, w_len, device, dtype)
            if mask_flat.shape[1] != patches.shape[1]:
                mask_flat = mask_flat[:, :patches.shape[1], :]
            patches_new = patches + mask_flat * ev
        else:
            patches_new = patches + ev

        # Unpatchify
        raw_new = unpatchify(patches_new, h_len, w_len, extra_shape)

        # Convert back to process_in space
        modified = raw_to_denoised(raw_new, model).to(dtype)

        # Repack if LTXAV
        return repack_video_if_needed(modified, pack_info)

    return post_cfg_fn


class LCSSharpnessIntervene(io.ComfyNode):
    """Control sharpness during FLUX generation via the sharpness subspace.

    Installs a post-CFG hook that adds a scaled shift along the sharpness
    PC1 direction. When calibrated with lcs_data, the sharpness direction
    is orthogonal to color, so color is preserved by construction.
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
                io.Float.Input("strength", default=0.0, min=-5.0, max=5.0, step=0.1,
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
