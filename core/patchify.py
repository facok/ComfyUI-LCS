"""Patchify/unpatchify for FLUX-family latent tensors (patch_size=2, auto-detect channels)."""

from einops import rearrange


def patchify(x):
    """Convert latent [B, C, H, W] or [B, C, T, H, W] → patch sequence [B, L, C*4].

    For video VAEs (5D input with T frames), merges T into the batch dimension
    so all frames are patchified together. The original shape is returned as
    extra_shape for unpatchify to restore.

    L = (H/2) * (W/2), d = C * 2 * 2.
    Returns (patches, h_len, w_len, extra_shape) where extra_shape is None for 4D
    or (B_orig, C, T) for 5D.
    """
    extra_shape = None
    if x.ndim == 5:
        B_orig, C, T, H, W = x.shape
        extra_shape = (B_orig, C, T)
        # Merge B and T: [B*T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B_orig * T, C, H, W)
    B, C, H, W = x.shape
    h_len = H // 2
    w_len = W // 2
    patches = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return patches, h_len, w_len, extra_shape


def unpatchify(patches, h_len, w_len, extra_shape=None):
    """Convert patch sequence [B, L, C*4] → latent [B, C, H, W] or [B, C, T, H, W].

    Auto-detects channel count from patch dimension: C = D / 4.
    If extra_shape is provided, restores the 5D video format.
    """
    D = patches.shape[-1]
    C = D // 4  # patch_size=2×2=4
    x = rearrange(patches, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                  h=h_len, w=w_len, c=C, ph=2, pw=2)
    if extra_shape is not None:
        B_orig, C_orig, T = extra_shape
        # Unmerge B and T: [B_orig*T, C, H, W] → [B_orig, C, T, H, W]
        H, W = x.shape[2], x.shape[3]
        x = x.reshape(B_orig, T, C, H, W).permute(0, 2, 1, 3, 4)
    return x
