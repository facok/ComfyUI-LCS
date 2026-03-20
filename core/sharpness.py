"""Sharpness subspace calibration: PCA on blur stimuli in FLUX VAE patch space."""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import comfy.utils

from .patchify import patchify
from .lcs_data import LCSData


@dataclass
class SharpnessData:
    """Calibration data for the sharpness subspace.

    Produced by PCA on FLUX VAE-encoded images at varying blur levels.
    PC1 captures ~93% of sharpness/blur variance.
    """

    basis: torch.Tensor   # [64, K] PCA basis (columns), K typically 1-2
    mean: torch.Tensor    # [64] PCA mean (in color-removed space if lcs_data was used)
    pc1_std: float        # Standard deviation along PC1 (for normalizing strength)
    sign: float           # +1 or -1: ensures positive strength = sharper

    def to(self, device, dtype=None):
        """Move all tensors to device/dtype."""
        kw = {"device": device}
        if dtype is not None:
            kw["dtype"] = dtype
        return SharpnessData(
            basis=self.basis.to(**kw),
            mean=self.mean.to(**kw),
            pc1_std=self.pc1_std,
            sign=self.sign,
        )


# Cache for Gaussian kernels to avoid recomputation
_gaussian_kernel_cache: Dict[Tuple[int, float, torch.dtype, torch.device], torch.Tensor] = {}


def _gaussian_kernel_1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 1D Gaussian kernel with caching."""
    cache_key = (kernel_size, sigma, dtype, device)
    if cache_key in _gaussian_kernel_cache:
        return _gaussian_kernel_cache[cache_key]

    x = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    _gaussian_kernel_cache[cache_key] = gauss
    return gauss


def _apply_gaussian_blur(images: torch.Tensor, blur_sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to a batch of images [B, C, H, W].

    Uses separable convolution for kernel_size > 15 (O(2k) vs O(k²)).

    Returns blurred images on same device/dtype as input.
    blur_sigma=0 returns input unchanged.
    """
    if blur_sigma < 1e-6:
        return images

    # Kernel size: 6*sigma rounded up to odd
    kernel_size = int(math.ceil(blur_sigma * 6)) | 1
    kernel_size = max(kernel_size, 3)

    B, C, H, W = images.shape
    device, dtype = images.device, images.dtype

    if kernel_size <= 15:
        # Direct 2D convolution for small kernels
        kernel_1d = _gaussian_kernel_1d(kernel_size, blur_sigma, device, dtype)
        kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]  # outer product
        # [C, 1, K, K] for depthwise conv with groups=C
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(C, 1, -1, -1).contiguous()
        pad = kernel_size // 2
        blurred = F.conv2d(images, kernel, padding=pad, groups=C)
        return blurred
    else:
        # Separable convolution: apply 1D Gaussian in X then Y direction
        # This reduces O(k²) to O(2k) operations
        gauss_1d = _gaussian_kernel_1d(kernel_size, blur_sigma, device, dtype)
        pad = kernel_size // 2

        # Horizontal pass: convolve along W dimension
        kernel_h = gauss_1d.view(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size).contiguous()
        blurred = F.conv2d(images, kernel_h, padding=(0, pad), groups=C)

        # Vertical pass: convolve along H dimension
        kernel_v = gauss_1d.view(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1).contiguous()
        blurred = F.conv2d(blurred, kernel_v, padding=(pad, 0), groups=C)

        return blurred


def calibrate_sharpness(vae, num_samples: int = 64, image_size: int = 512,
                        blur_levels: Tuple[float, ...] = (0, 0.5, 1, 2, 4, 8),
                        batch_size: int = 8,
                        lcs_data: LCSData = None) -> SharpnessData:
    """Compute sharpness subspace data (PCA basis, mean, sign) from FLUX VAE.

    1. Generate num_samples random noise images (spatial detail needed for blur)
    2. For each blur level, apply Gaussian blur to the SAME images
    3. VAE encode → patchify → average patches → [64] vector
    4. Optionally remove LCS color component (ensures sharpness PC1 is orthogonal to color)
    5. PCA on all vectors → extract PC1 (+ PC2)
    6. Determine sign: positive strength = sharper
    7. Compute pc1_std from spread of PC1 scores

    Args:
        vae: ComfyUI VAE object
        num_samples: Number of base images to generate
        image_size: Size of generated images
        blur_levels: Blur sigma levels to apply
        batch_size: Batch size for VAE encoding
        lcs_data: Optional LCS data for removing color component during calibration.
                  When provided, the sharpness PC1 will be orthogonal to the color subspace,
                  preventing color shifts during intervention.

    Returns: SharpnessData
    """
    n_levels = len(blur_levels)
    total_images = num_samples * n_levels

    print(f"\n[LCS Sharpness Calibration] Starting: {num_samples} images × {n_levels} blur levels = {total_images} samples")
    print(f"[LCS Sharpness Calibration] Blur sigmas: {list(blur_levels)}")

    # Deterministic seed for reproducibility
    rng = torch.Generator().manual_seed(42)

    # Step 1: Generate all base images upfront [num_samples, 3, H, W]
    # Use GRAYSCALE noise (same value across RGB) so that blur affects all channels
    # identically. RGB noise introduces inter-channel variance that PCA captures
    # as a secondary component unrelated to sharpness.
    print(f"[LCS Sharpness Calibration] Generating {num_samples} grayscale noise images...")
    gray_noise = torch.rand(num_samples, 1, image_size, image_size, generator=rng)
    base_images = gray_noise.expand(num_samples, 3, image_size, image_size).contiguous()

    # Step 2+3: For each blur level, apply blur to ALL base images, then encode
    vectors = []
    blur_labels = []  # track blur sigma per vector for sign determination
    pbar = comfy.utils.ProgressBar(total_images)

    for blur_sigma in blur_levels:
        print(f"[LCS Sharpness Calibration] Processing blur σ={blur_sigma}...")

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch = base_images[batch_start:batch_end]  # [B, 3, H, W]
            actual_batch = batch.shape[0]

            # Apply blur (no-op for sigma=0)
            blurred = _apply_gaussian_blur(batch, blur_sigma)

            # Convert BCHW → BHWC for ComfyUI VAE
            imgs_bhwc = blurred.permute(0, 2, 3, 1).contiguous().cpu()

            # VAE encode → [B, 16, H/8, W/8]
            latent = vae.encode(imgs_bhwc)

            # Patchify → [B, L, 64], average across patches → [B, 64]
            patches, _, _ = patchify(latent)
            avg = patches.mean(dim=1).cpu()

            vectors.extend(avg.unbind(0))
            blur_labels.extend([blur_sigma] * actual_batch)

            pbar.update(actual_batch)

    # Stack all vectors: [N, 64]
    X = torch.stack(vectors, dim=0).float()
    blur_labels_t = torch.tensor(blur_labels, dtype=torch.float32)
    print(f"[LCS Sharpness Calibration] Collected {X.shape[0]} vectors of dimension {X.shape[1]}")

    # Remove per-vector mean BEFORE PCA.
    # VAE encoding of blurred images shifts the latent mean (non-linear VAE effect).
    # Without this, PCA captures brightness drift as the dominant component.
    # Per-vector zero-mean forces PCA to find patterns in the relative channel
    # structure, not in the absolute level — isolating true sharpness.
    X = X - X.mean(dim=1, keepdim=True)

    # Optionally remove LCS color component to ensure sharpness PC1 is orthogonal to color
    if lcs_data is not None:
        print("[LCS Sharpness Calibration] Removing LCS color component...")
        lcs_mean = lcs_data.mean.to(X.device, X.dtype)
        lcs_basis = lcs_data.basis.to(X.device, X.dtype)
        centered = X - lcs_mean
        lcs_coords = centered @ lcs_basis  # [N, 3]
        color_reconstruction = lcs_coords @ lcs_basis.T + lcs_mean
        X = X - (color_reconstruction - lcs_mean)
        print("[LCS Sharpness Calibration] Color component removed")

    # Step 4: PCA
    print("[LCS Sharpness Calibration] Computing PCA...")
    mean = X.mean(dim=0)  # [64]
    X_centered = X - mean
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    # Top 2 components
    basis = Vh[:2].T  # [64, 2]

    # Variance explained
    total_var = (S ** 2).sum()
    explained = (S[:2] ** 2) / total_var
    print(f"[LCS Sharpness Calibration] PC1: {explained[0]:.1%}, PC2: {explained[1]:.1%} ({(explained[0]+explained[1]):.1%} total)")

    # Step 5: Determine sign convention
    # Project all vectors onto PC1
    pc1_scores = X_centered @ basis[:, 0]  # [N]

    # Correlate PC1 score with blur sigma
    # If positive correlation (more blur = higher score), flip sign
    correlation = torch.corrcoef(torch.stack([pc1_scores, blur_labels_t]))[0, 1]
    sign = -1.0 if correlation > 0 else 1.0
    print(f"[LCS Sharpness Calibration] PC1-blur correlation: {correlation:.3f} → sign = {sign:+.0f}")

    # Step 6: Compute pc1_std
    pc1_std = float(pc1_scores.std())
    print(f"[LCS Sharpness Calibration] PC1 std: {pc1_std:.4f}")
    print(f"[LCS Sharpness Calibration] Complete! Basis shape: {basis.shape}")

    return SharpnessData(
        basis=basis,
        mean=mean,
        pc1_std=pc1_std,
        sign=sign,
    )
