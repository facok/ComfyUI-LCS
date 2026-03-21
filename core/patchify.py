"""Patchify/unpatchify for FLUX-family latent tensors (patch_size=2, auto-detect channels)."""

from einops import rearrange


def patchify(x):
    """Convert latent [C, H, W], [B, C, H, W], or [B, C, T, H, W] → patch sequence [B, L, C*4].

    Handles three input formats:
    - 3D [C, H, W]: adds batch dim, extra_shape="unbatched"
    - 4D [B, C, H, W]: standard path, extra_shape=None
    - 5D [B, C, T, H, W]: video VAE, merges T into batch, extra_shape=(B, C, T)

    L = (H/2) * (W/2), d = C * 2 * 2.
    """
    extra_shape = None
    if x.ndim == 3:
        # No batch dimension (e.g. LTXAV): [C, H, W] → [1, C, H, W]
        extra_shape = "unbatched"
        x = x.unsqueeze(0)
    elif x.ndim == 5:
        B_orig, C, T, H, W = x.shape
        extra_shape = (B_orig, C, T)
        # Merge B and T: [B*T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B_orig * T, C, H, W)
    B, C, H, W = x.shape
    if H < 2 or W < 2:
        # Incompatible latent format (e.g. LTXAV uses flattened 1D layout)
        return None, None, None, None
    h_len = H // 2
    w_len = W // 2
    patches = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return patches, h_len, w_len, extra_shape


def unpatchify(patches, h_len, w_len, extra_shape=None):
    """Convert patch sequence [B, L, C*4] → latent, restoring original shape.

    Auto-detects channel count from patch dimension: C = D / 4.
    Restores 3D/5D format based on extra_shape from patchify.
    """
    D = patches.shape[-1]
    C = D // 4  # patch_size=2×2=4
    x = rearrange(patches, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                  h=h_len, w=w_len, c=C, ph=2, pw=2)
    if extra_shape == "unbatched":
        # Restore [1, C, H, W] → [C, H, W]
        x = x.squeeze(0)
    elif extra_shape is not None:
        B_orig, C_orig, T = extra_shape
        # Unmerge B and T: [B_orig*T, C, H, W] → [B_orig, C, T, H, W]
        H, W = x.shape[2], x.shape[3]
        x = x.reshape(B_orig, T, C, H, W).permute(0, 2, 1, 3, 4)
    return x
