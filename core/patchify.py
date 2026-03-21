"""Patchify/unpatchify for latent tensors (patch_size=2, auto-detect channels).

Handles 3D, 4D, and 5D inputs. Pads odd spatial dims to even before patchifying.
"""

from einops import rearrange
import torch.nn.functional as F


def patchify(x):
    """Convert latent [C, H, W], [B, C, H, W], or [B, C, T, H, W] → patch sequence [B, L, C*4].

    Handles three input formats:
    - 3D [C, H, W]: adds batch dim, extra_shape="unbatched"
    - 4D [B, C, H, W]: standard path, extra_shape=None
    - 5D [B, C, T, H, W]: video VAE, merges T into batch, extra_shape=(B, C, T)

    Pads odd H/W to even before patchifying. The pad amounts are stored
    in the returned extra_shape for unpatchify to crop back.

    L = (H_padded/2) * (W_padded/2), d = C * 2 * 2.
    """
    extra_shape = None
    pad_h = 0
    pad_w = 0

    if x.ndim == 3:
        extra_shape = "unbatched"
        x = x.unsqueeze(0)
    elif x.ndim == 5:
        B_orig, C, T, H, W = x.shape
        extra_shape = (B_orig, C, T)
        x = x.permute(0, 2, 1, 3, 4).reshape(B_orig * T, C, H, W)

    B, C, H, W = x.shape
    if H < 1 or W < 1:
        return None, None, None, None

    # Pad odd dimensions to even (replicate last row/col)
    if H % 2 != 0:
        pad_h = 1
    if W % 2 != 0:
        pad_w = 1
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    H_p, W_p = x.shape[2], x.shape[3]
    h_len = H_p // 2
    w_len = W_p // 2
    patches = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    # Bundle pad info with extra_shape
    if pad_h or pad_w:
        extra_shape = {"orig_extra": extra_shape, "pad_h": pad_h, "pad_w": pad_w}

    return patches, h_len, w_len, extra_shape


def unpatchify(patches, h_len, w_len, extra_shape=None):
    """Convert patch sequence [B, L, C*4] → latent, restoring original shape.

    Auto-detects channel count from patch dimension: C = D / 4.
    Handles padding removal and 3D/5D restoration based on extra_shape.
    """
    D = patches.shape[-1]
    C = D // 4  # patch_size=2×2=4
    x = rearrange(patches, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                  h=h_len, w=w_len, c=C, ph=2, pw=2)

    # Unwrap pad info if present
    pad_h = 0
    pad_w = 0
    orig_extra = extra_shape
    if isinstance(extra_shape, dict):
        pad_h = extra_shape["pad_h"]
        pad_w = extra_shape["pad_w"]
        orig_extra = extra_shape["orig_extra"]

    # Remove padding
    if pad_h:
        x = x[:, :, :-pad_h, :]
    if pad_w:
        x = x[:, :, :, :-pad_w]

    # Restore original format
    if orig_extra == "unbatched":
        x = x.squeeze(0)
    elif orig_extra is not None:
        B_orig, C_orig, T = orig_extra
        H, W = x.shape[2], x.shape[3]
        x = x.reshape(B_orig, T, C_orig, H, W).permute(0, 2, 1, 3, 4)

    return x
