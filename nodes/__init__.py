"""V2 backward compatibility exports for all LCS nodes."""

from .calibrate import LCSLoadData
from .intervene import LCSColorIntervene, LCSColorBatch, LCSToneAdjust
from .observe import LCSPreviewColors, LCSStepObserver
from .sharpen import LCSSharpnessCalibrate, LCSSharpnessIntervene

NODE_CLASS_MAPPINGS = {
    "LCSLoadData": LCSLoadData,
    "LCSColorIntervene": LCSColorIntervene,
    "LCSColorBatch": LCSColorBatch,
    "LCSToneAdjust": LCSToneAdjust,
    "LCSPreviewColors": LCSPreviewColors,
    "LCSStepObserver": LCSStepObserver,
    "LCSSharpnessCalibrate": LCSSharpnessCalibrate,
    "LCSSharpnessIntervene": LCSSharpnessIntervene,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCSLoadData": "LCS Load Data",
    "LCSColorIntervene": "LCS Color Intervene",
    "LCSColorBatch": "LCS Color Batch",
    "LCSToneAdjust": "LCS Tone Adjust",
    "LCSPreviewColors": "LCS Preview Colors",
    "LCSStepObserver": "LCS Step Observer",
    "LCSSharpnessCalibrate": "LCS Sharpness Calibrate",
    "LCSSharpnessIntervene": "LCS Sharpness Intervene",
}
