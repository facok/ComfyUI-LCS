"""PCA calibration from FLUX VAE: compute LCS basis, mean, and anchor positions."""

import hashlib
import math
import torch
import comfy.utils
from .patchify import patchify
from .lcs_data import LCSData
from .color_space import _chromatic_plane_basis


def vae_fingerprint(vae) -> str:
    """8-char hex fingerprint from VAE decoder weights.

    Used to cache calibration data per-VAE so different VAE models
    get separate calibration files automatically.
    """
    sd = vae.get_sd()
    # Use first decoder weight tensor as fingerprint source
    for key in sorted(sd.keys()):
        if "decoder" in key and "weight" in key:
            w = sd[key]
            return hashlib.sha256(w.cpu().float().numpy().tobytes()).hexdigest()[:8]
    # Fallback: hash first weight found
    first_key = sorted(sd.keys())[0]
    w = sd[first_key]
    return hashlib.sha256(w.cpu().float().numpy().tobytes()).hexdigest()[:8]


# 8 anchor colors: R, B, G, M, C, Y, Black, White
ANCHOR_COLORS_RGB = [
    (1.0, 0.0, 0.0),   # Red
    (0.0, 0.0, 1.0),   # Blue
    (0.0, 1.0, 0.0),   # Green
    (1.0, 0.0, 1.0),   # Magenta
    (0.0, 1.0, 1.0),   # Cyan
    (1.0, 1.0, 0.0),   # Yellow
    (0.0, 0.0, 0.0),   # Black
    (1.0, 1.0, 1.0),   # White
]


def calibrate(vae, num_colors=512, image_size=512, batch_size=8):
    """Compute LCS data (PCA basis, mean, anchors) from FLUX VAE.

    1. Sample num_colors solid-color images uniformly from HSV
    2. VAE encode each → latent
    3. Patchify → average patches per image → vector in R^64
    4. PCA on all vectors → basis B [64,3], mean μ [64]
    5. Encode 8 anchor colors → compute LCS coords + hue angles

    Returns: LCSData
    """
    device = comfy.model_management.intermediate_device()

    # Step 1: Sample colors uniformly from HSV (full saturation, full value for diversity)
    colors = []
    for i in range(num_colors):
        # Uniform sampling in HSV
        h = (i * 137.508) % 360.0 / 360.0  # Golden angle for uniform coverage
        s = 0.3 + 0.7 * ((i * 73) % 100) / 100.0  # Vary saturation 0.3-1.0
        v = 0.3 + 0.7 * ((i * 47) % 100) / 100.0  # Vary value 0.3-1.0
        # HSV to RGB
        r, g, b = _hsv_to_rgb(h, s, v)
        colors.append((r, g, b))

    # Step 2+3: Encode and average patches
    vectors = []
    pbar = comfy.utils.ProgressBar(num_colors)

    for batch_start in range(0, num_colors, batch_size):
        batch_end = min(batch_start + batch_size, num_colors)
        batch_colors = colors[batch_start:batch_end]
        actual_batch = len(batch_colors)

        # Create solid color images [B, H, W, 3] (BHWC format for ComfyUI VAE)
        imgs = torch.zeros(actual_batch, image_size, image_size, 3, dtype=torch.float32, device="cpu")
        for j, (r, g, b) in enumerate(batch_colors):
            imgs[j, :, :, 0] = r
            imgs[j, :, :, 1] = g
            imgs[j, :, :, 2] = b

        # VAE encode → [B, 16, H/8, W/8]
        latent = vae.encode(imgs[:, :, :, :3])

        # Patchify → [B, L, 64]
        patches, _, _ = patchify(latent)

        # Average across patches → [B, 64]
        avg = patches.mean(dim=1).cpu()
        for j in range(actual_batch):
            vectors.append(avg[j])

        pbar.update(actual_batch)

    # Stack all vectors: [N, 64]
    X = torch.stack(vectors, dim=0).float()

    # Step 4: PCA
    mean = X.mean(dim=0)  # [64]
    X_centered = X - mean
    # SVD for PCA
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    # Top 3 components: B = V[:, :3] (columns are principal directions)
    basis = Vh[:3].T  # [64, 3] (Vh rows are right singular vectors)

    # Step 5: Encode 8 anchor colors → LCS coords
    anchor_lcs_list = []
    for r, g, b in ANCHOR_COLORS_RGB:
        img = torch.zeros(1, image_size, image_size, 3, dtype=torch.float32, device="cpu")
        img[0, :, :, 0] = r
        img[0, :, :, 1] = g
        img[0, :, :, 2] = b
        latent = vae.encode(img[:, :, :, :3])
        patches, _, _ = patchify(latent)
        avg = patches.mean(dim=1).cpu().squeeze(0)  # [64]
        # Project to LCS
        lcs_coord = (avg - mean) @ basis  # [3]
        anchor_lcs_list.append(lcs_coord)

    anchor_lcs = torch.stack(anchor_lcs_list, dim=0)  # [8, 3]

    # Compute hue angles for 6 chromatic anchors
    anchor_angles = _compute_anchor_angles(anchor_lcs, basis, mean)

    return LCSData(
        basis=basis,
        mean=mean,
        anchor_lcs=anchor_lcs,
        anchor_angles=anchor_angles,
    )


def _compute_anchor_angles(anchor_lcs, basis, mean):
    """Compute hue angles of the 6 chromatic anchors in the chromatic plane.

    The chromatic plane is perpendicular to the achromatic axis (black→white).
    Returns [6] tensor of angles in radians.
    """
    black = anchor_lcs[6]  # [3]
    white = anchor_lcs[7]  # [3]
    chromatic = anchor_lcs[:6]  # [6, 3]

    # Achromatic axis
    a = white - black
    a_unit, e1, e2 = _chromatic_plane_basis(a)

    # Project each chromatic anchor onto the plane and compute angle
    angles = []
    for i in range(6):
        c = chromatic[i]
        # Project onto achromatic axis
        c_proj = black + ((c - black) * a).sum() / ((a * a).sum() + 1e-10) * a
        # Chromatic residual
        chroma = c - c_proj
        x = (chroma * e1).sum()
        y = (chroma * e2).sum()
        angle = torch.atan2(y, x) % (2 * math.pi)
        angles.append(angle)

    return torch.stack(angles)  # [6]


def _hsv_to_rgb(h, s, v):
    """Convert HSV to RGB (scalars in [0,1])."""
    if s < 1e-10:
        return v, v, v
    h6 = h * 6.0
    i = int(h6) % 6
    f = h6 - int(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    return v, p, q
