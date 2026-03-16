from dataclasses import dataclass
import torch


@dataclass
class LCSData:
    basis: torch.Tensor        # [64, 3] PCA basis B (orthonormal columns)
    mean: torch.Tensor         # [64] PCA mean mu
    anchor_lcs: torch.Tensor   # [8, 3] LCS coords of 8 anchor colors [R,B,G,M,C,Y,Black,White]
    anchor_angles: torch.Tensor  # [6] hue angles (radians) of the 6 chromatic anchors

    def to(self, device, dtype=None):
        """Move all tensors to device/dtype."""
        kw = {"device": device}
        if dtype is not None:
            kw["dtype"] = dtype
        return LCSData(
            basis=self.basis.to(**kw),
            mean=self.mean.to(**kw),
            anchor_lcs=self.anchor_lcs.to(**kw),
            anchor_angles=self.anchor_angles.to(**kw),
        )
