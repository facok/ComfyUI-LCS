"""Patchify/unpatchify for FLUX-family latent tensors (patch_size=2, auto-detect channels)."""

from einops import rearrange


def patchify(x):
    """Convert latent [B, C, H, W] or [B, C, T, H, W] → patch sequence [B, L, C*4].

    For video VAEs (5D input), squeezes the temporal dimension (T must be 1).
    L = (H/2) * (W/2), d = C * 2 * 2.
    Returns (patches, h_len, w_len) where h_len=H/2, w_len=W/2.
    """
    if x.ndim == 5:
        # Video VAE: [B, C, T, H, W] → squeeze T
        x = x.squeeze(2)
    B, C, H, W = x.shape
    h_len = H // 2
    w_len = W // 2
    patches = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return patches, h_len, w_len


def unpatchify(patches, h_len, w_len):
    """Convert patch sequence [B, L, C*4] → latent [B, C, H, W].

    Auto-detects channel count from patch dimension: C = D / 4.
    h_len, w_len from patchify output.
    """
    D = patches.shape[-1]
    C = D // 4  # patch_size=2×2=4
    return rearrange(patches, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                     h=h_len, w=w_len, c=C, ph=2, pw=2)
