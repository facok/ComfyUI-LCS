"""Calibration node: LCSLoadData with automatic per-VAE caching."""

import os
import torch
from comfy_api.latest import io
from safetensors.torch import save_file, load_file

from ..core.calibration import calibrate, vae_fingerprint
from ..core.lcs_data import LCSData

LCS_DATA = io.Custom("LCS_DATA")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _save_lcs(lcs_data: LCSData, path: str):
    """Save LCSData to safetensors file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({
        "basis": lcs_data.basis.contiguous(),
        "mean": lcs_data.mean.contiguous(),
        "anchor_lcs": lcs_data.anchor_lcs.contiguous(),
        "anchor_angles": lcs_data.anchor_angles.contiguous(),
    }, path)


def _load_lcs(path: str) -> LCSData:
    """Load LCSData from safetensors file."""
    data = load_file(path)
    return LCSData(
        basis=data["basis"],
        mean=data["mean"],
        anchor_lcs=data["anchor_lcs"],
        anchor_angles=data["anchor_angles"],
    )


class LCSLoadData(io.ComfyNode):
    """Load or auto-compute LCS calibration data for a VAE.

    Computes a fingerprint of the VAE weights and checks for a cached
    calibration file. On cache miss, runs PCA calibration automatically
    and saves the result for future reuse.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LCSLoadData",
            display_name="LCS Load Data",
            category="LCS/calibration",
            description="Auto-calibrate and cache LCS data per-VAE",
            inputs=[
                io.Vae.Input("vae", tooltip="VAE model (calibration is cached per-VAE)"),
            ],
            outputs=[
                LCS_DATA.Output(display_name="lcs_data"),
            ],
        )

    @classmethod
    def execute(cls, vae) -> io.NodeOutput:
        fp = vae_fingerprint(vae)
        cache_path = os.path.join(DATA_DIR, f"lcs_{fp}.safetensors")

        if os.path.exists(cache_path):
            lcs_data = _load_lcs(cache_path)
        else:
            lcs_data = calibrate(vae, num_colors=512, image_size=512)
            _save_lcs(lcs_data, cache_path)

        return io.NodeOutput(lcs_data)
