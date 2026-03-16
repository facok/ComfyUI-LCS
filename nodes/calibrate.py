"""Calibration nodes: LCSCalibrate and LCSLoadData."""

import os
import torch
from comfy_api.latest import io
from safetensors.torch import save_file, load_file

from ..core.calibration import calibrate
from ..core.lcs_data import LCSData

LCS_DATA = io.Custom("LCS_DATA")

# Directory for cached calibration files
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class LCSCalibrate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LCSCalibrate",
            display_name="LCS Calibrate",
            category="LCS/calibration",
            description="Compute LCS basis and anchors from FLUX VAE via PCA on solid-color images",
            inputs=[
                io.Vae.Input("vae", tooltip="FLUX VAE model"),
                io.Int.Input("num_colors", default=512, min=64, max=2048, step=64,
                             tooltip="Number of solid-color samples for PCA"),
                io.Int.Input("image_size", default=512, min=256, max=1024, step=128,
                             tooltip="Size of solid-color calibration images"),
            ],
            outputs=[
                LCS_DATA.Output(display_name="lcs_data"),
            ],
        )

    @classmethod
    def execute(cls, vae, num_colors, image_size) -> io.NodeOutput:
        lcs_data = calibrate(vae, num_colors=num_colors, image_size=image_size)

        # Auto-save to data/ directory
        os.makedirs(DATA_DIR, exist_ok=True)
        save_path = os.path.join(DATA_DIR, "lcs_calibration.safetensors")
        save_file({
            "basis": lcs_data.basis,
            "mean": lcs_data.mean,
            "anchor_lcs": lcs_data.anchor_lcs,
            "anchor_angles": lcs_data.anchor_angles,
        }, save_path)

        return io.NodeOutput(lcs_data)


class LCSLoadData(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        # Scan data/ dir for available calibration files
        files = ["auto"]
        if os.path.isdir(DATA_DIR):
            for f in sorted(os.listdir(DATA_DIR)):
                if f.endswith(".safetensors"):
                    files.append(f)

        return io.Schema(
            node_id="LCSLoadData",
            display_name="LCS Load Data",
            category="LCS/calibration",
            description="Load cached LCS calibration data, or auto-calibrate if VAE is provided",
            inputs=[
                io.Combo.Input("calibration_file", options=files, default="auto",
                               tooltip="Select calibration file or 'auto' to use latest/auto-calibrate"),
                io.Vae.Input("vae", optional=True,
                             tooltip="VAE for auto-calibration if no cached file exists"),
            ],
            outputs=[
                LCS_DATA.Output(display_name="lcs_data"),
            ],
        )

    @classmethod
    def execute(cls, calibration_file, vae=None) -> io.NodeOutput:
        loaded = False

        if calibration_file != "auto":
            path = os.path.join(DATA_DIR, calibration_file)
            if os.path.exists(path):
                data = load_file(path)
                lcs_data = LCSData(
                    basis=data["basis"],
                    mean=data["mean"],
                    anchor_lcs=data["anchor_lcs"],
                    anchor_angles=data["anchor_angles"],
                )
                loaded = True

        if not loaded:
            # Try auto-loading the default calibration file
            default_path = os.path.join(DATA_DIR, "lcs_calibration.safetensors")
            if os.path.exists(default_path):
                data = load_file(default_path)
                lcs_data = LCSData(
                    basis=data["basis"],
                    mean=data["mean"],
                    anchor_lcs=data["anchor_lcs"],
                    anchor_angles=data["anchor_angles"],
                )
                loaded = True

        if not loaded:
            if vae is None:
                raise ValueError(
                    "No calibration file found and no VAE provided. "
                    "Either run LCSCalibrate first or provide a VAE for auto-calibration."
                )
            # Auto-calibrate
            lcs_data = calibrate(vae, num_colors=512, image_size=512)
            os.makedirs(DATA_DIR, exist_ok=True)
            save_path = os.path.join(DATA_DIR, "lcs_calibration.safetensors")
            save_file({
                "basis": lcs_data.basis,
                "mean": lcs_data.mean,
                "anchor_lcs": lcs_data.anchor_lcs,
                "anchor_angles": lcs_data.anchor_angles,
            }, save_path)

        return io.NodeOutput(lcs_data)
