"""Patchify/unpatchify for FLUX latent tensors (patch_size=2, 16 channels → 64-dim patches)."""

from einops import rearrange


def patchify(x):
    """Convert latent [B, 16, H, W] → patch sequence [B, L, 64].

    L = (H/2) * (W/2), d = 16 * 2 * 2 = 64.
    Returns (patches, h_len, w_len) where h_len=H/2, w_len=W/2.
    """
    B, C, H, W = x.shape
    h_len = H // 2
    w_len = W // 2
    patches = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return patches, h_len, w_len


def unpatchify(patches, h_len, w_len):
    """Convert patch sequence [B, L, 64] → latent [B, 16, H, W].

    h_len, w_len from patchify output.
    """
    return rearrange(patches, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                     h=h_len, w=w_len, c=16, ph=2, pw=2)
