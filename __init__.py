"""ComfyUI-LCS: The Latent Color Subspace — training-free color control for FLUX.

Paper: "The Latent Color Subspace" (arXiv:2603.12261v1, ICML 2026)
"""

# V3 ComfyExtension entry point
from comfy_api.latest import ComfyExtension, io
from .nodes.calibrate import LCSCalibrate, LCSLoadData
from .nodes.intervene import LCSColorIntervene, LCSColorBatch
from .nodes.observe import LCSPreviewColors, LCSStepObserver


class LCSExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LCSCalibrate,
            LCSLoadData,
            LCSColorIntervene,
            LCSColorBatch,
            LCSPreviewColors,
            LCSStepObserver,
        ]


async def comfy_entrypoint() -> LCSExtension:
    return LCSExtension()


# V2 backward compatibility
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "LCSExtension",
    "comfy_entrypoint",
]
