"""Sharpness subspace calibration: PCA on blur stimuli in FLUX VAE patch space."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils

from .patchify import patchify


@dataclass
class SharpnessData:
    """Calibration data for the sharpness subspace.

    Produced by PCA on FLUX VAE-encoded images at varying blur levels.
    PC1 captures ~93% of sharpness/blur variance.
    """

    basis: torch.Tensor   # [64, K] PCA basis (columns), K typically 1-2
    mean: torch.Tensor    # [64] PCA mean
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


def _gaussian_kernel_2d(kernel_size, sigma):
    """Create a 2D Gaussian kernel [1, 1, K, K] for F.conv2d."""
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)  # outer product
    return kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]


def _apply_gaussian_blur(images, blur_sigma):
    """Apply Gaussian blur to a batch of images [B, C, H, W].

    Returns blurred images on same device/dtype as input.
    blur_sigma=0 returns input unchanged.
    """
    if blur_sigma < 1e-6:
        return images

    # Kernel size: 6*sigma rounded up to odd
    kernel_size = int(math.ceil(blur_sigma * 6)) | 1  # ensure odd
    kernel_size = max(kernel_size, 3)

    kernel = _gaussian_kernel_2d(kernel_size, blur_sigma)
    kernel = kernel.to(device=images.device, dtype=images.dtype)

    pad = kernel_size // 2
    B, C, H, W = images.shape
    # Apply per-channel via groups
    images_grouped = images.reshape(B * C, 1, H, W)
    blurred = F.conv2d(images_grouped, kernel, padding=pad)
    return blurred.reshape(B, C, H, W)


def calibrate_sharpness(vae, num_samples=64, image_size=512,
                        blur_levels=(0, 1, 2, 4, 8, 16), batch_size=8):
    """Compute sharpness subspace data (PCA basis, mean, sign) from FLUX VAE.

    1. Generate num_samples random grayscale solid-color images
    2. For each blur level, apply Gaussian blur
    3. VAE encode → patchify → average patches → [64] vector
    4. PCA on all vectors → extract PC1 (+ PC2)
    5. Determine sign: positive strength = sharper
    6. Compute pc1_std from spread of PC1 scores

    Returns: SharpnessData
    """
    device = comfy.model_management.intermediate_device()
    n_levels = len(blur_levels)
    total_images = num_samples * n_levels

    print(f"\n[LCS Sharpness Calibration] Starting: {num_samples} images × {n_levels} blur levels = {total_images} samples")
    print(f"[LCS Sharpness Calibration] Blur sigmas: {list(blur_levels)}")

    # Step 1: Generate random grayscale base images [num_samples, 3, H, W]
    # Use deterministic seed for reproducibility across runs
    rng = torch.Generator().manual_seed(42)
    gray_values = torch.rand(num_samples, 1, 1, 1, generator=rng)
    # [num_samples, 3, H, W] in BCHW for blur, will convert to BHWC for VAE
    base_images = gray_values.expand(num_samples, 3, image_size, image_size).contiguous()

    # Step 2+3: For each blur level, blur all images, VAE encode, collect patch vectors
    vectors = []
    blur_labels = []  # track blur sigma per vector for sign determination
    pbar = comfy.utils.ProgressBar(total_images)

    for blur_sigma in blur_levels:
        # Apply blur
        blurred = _apply_gaussian_blur(base_images, blur_sigma)

        # Encode in batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch = blurred[batch_start:batch_end]  # [B, 3, H, W] BCHW
            actual_batch = batch.shape[0]

            # Convert BCHW → BHWC for ComfyUI VAE
            imgs_bhwc = batch.permute(0, 2, 3, 1).contiguous().cpu()

            # VAE encode → [B, 16, H/8, W/8]
            latent = vae.encode(imgs_bhwc[:, :, :, :3])

            # Patchify → [B, L, 64], average across patches → [B, 64]
            patches, _, _ = patchify(latent)
            avg = patches.mean(dim=1).cpu()

            for j in range(actual_batch):
                vectors.append(avg[j])
                blur_labels.append(blur_sigma)

            pbar.update(actual_batch)

    # Stack all vectors: [N, 64]
    X = torch.stack(vectors, dim=0).float()
    blur_labels_t = torch.tensor(blur_labels, dtype=torch.float32)
    print(f"[LCS Sharpness Calibration] Collected {X.shape[0]} vectors of dimension {X.shape[1]}")

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
