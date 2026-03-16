"""V2 backward compatibility exports for all LCS nodes."""

from .calibrate import LCSCalibrate, LCSLoadData
from .intervene import LCSColorIntervene, LCSColorBatch
from .observe import LCSPreviewColors, LCSStepObserver

NODE_CLASS_MAPPINGS = {
    "LCSCalibrate": LCSCalibrate,
    "LCSLoadData": LCSLoadData,
    "LCSColorIntervene": LCSColorIntervene,
    "LCSColorBatch": LCSColorBatch,
    "LCSPreviewColors": LCSPreviewColors,
    "LCSStepObserver": LCSStepObserver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCSCalibrate": "LCS Calibrate",
    "LCSLoadData": "LCS Load Data",
    "LCSColorIntervene": "LCS Color Intervene",
    "LCSColorBatch": "LCS Color Batch",
    "LCSPreviewColors": "LCS Preview Colors",
    "LCSStepObserver": "LCS Step Observer",
}
