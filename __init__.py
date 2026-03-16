"""ComfyUI-LCS: The Latent Color Subspace — training-free color control for FLUX.

Paper: "The Latent Color Subspace" (arXiv:2603.12261v1, ICML 2026)
"""

# V3 ComfyExtension entry point
from comfy_api.latest import ComfyExtension, io
from .nodes.calibrate import LCSLoadData
from .nodes.intervene import LCSColorIntervene, LCSColorBatch, LCSToneAdjust
from .nodes.observe import LCSPreviewColors, LCSStepObserver


class LCSExtension(ComfyExtension):
    """V3 ComfyExtension providing all LCS nodes to ComfyUI."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return all LCS node classes."""
        return [
            LCSLoadData,
            LCSColorIntervene,
            LCSColorBatch,
            LCSToneAdjust,
            LCSPreviewColors,
            LCSStepObserver,
        ]


async def comfy_entrypoint() -> LCSExtension:
    """V3 async entry point called by ComfyUI on startup."""
    return LCSExtension()


# V2 backward compatibility
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "LCSExtension",
    "comfy_entrypoint",
]
