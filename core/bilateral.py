"""Bilateral filter in LCS space for smooth color anchoring."""

import math

import torch
import torch.nn.functional as F


def estimate_bilateral_params(c, h_len, w_len):
    """Estimate bilateral filter parameters from local color statistics.

    Computes per-channel spatial std of c across the grid, takes the median
    to derive sigma_color. sigma_spatial is fixed at 1.5 (5x5 kernel is small).

    c: [B, L, 3] LCS coordinates
    Returns: (sigma_spatial, sigma_color) floats
    """
    B = c.shape[0]
    grid = c.reshape(B, h_len, w_len, 3)  # [B, H, W, 3]
    # Per-channel std across spatial dims → [B, 3]
    channel_std = grid.reshape(B, -1, 3).std(dim=1)  # [B, 3]
    # Median across batch and channels
    median_std = float(channel_std.median())
    sigma_color = max(0.05, min(3.0, 0.75 * median_std))
    sigma_spatial = 1.5
    return sigma_spatial, sigma_color


def bilateral_filter_lcs(c, h_len, w_len, sigma_spatial, sigma_color, kernel_radius=2):
    """Bilateral filter on [B, L, 3] LCS coordinates arranged on h_len x w_len grid.

    Uses spatial distance + LCS color distance as joint weights.
    kernel_radius=2 -> 5x5 neighborhood (25 lookups per patch).
    Returns [B, L, 3] filtered coordinates.
    """
    B = c.shape[0]
    # Reshape to spatial grid
    grid = c.reshape(B, h_len, w_len, 3)  # [B, H, W, 3]

    # Pad by kernel_radius (replicate) — pad last two spatial dims
    # F.pad on [B, H, W, 3]: need to pad dims -3 and -2 (H and W)
    # Permute to [B, 3, H, W] for F.pad, then back
    grid_chw = grid.permute(0, 3, 1, 2)  # [B, 3, H, W]
    r = kernel_radius
    padded = F.pad(grid_chw, (r, r, r, r), mode="replicate")  # [B, 3, H+2r, W+2r]

    # Precompute spatial Gaussian weights for each offset in kernel
    inv_2ss = -0.5 / (sigma_spatial * sigma_spatial)
    inv_2sc = -0.5 / (sigma_color * sigma_color)

    # Accumulate weighted sum
    weight_sum = torch.zeros(B, 1, h_len, w_len, device=c.device, dtype=c.dtype)
    value_sum = torch.zeros(B, 3, h_len, w_len, device=c.device, dtype=c.dtype)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            # Spatial weight (constant per offset)
            spatial_dist_sq = float(dy * dy + dx * dx)
            w_spatial = math.exp(spatial_dist_sq * inv_2ss)

            # Extract neighbor values from padded grid
            y_start = r + dy
            x_start = r + dx
            neighbor = padded[:, :, y_start:y_start + h_len, x_start:x_start + w_len]  # [B, 3, H, W]

            # Color distance weight (per-pixel)
            diff = neighbor - grid_chw  # [B, 3, H, W]
            color_dist_sq = (diff * diff).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            w_color = torch.exp(color_dist_sq * inv_2sc)  # [B, 1, H, W]

            w = w_spatial * w_color
            weight_sum.add_(w)
            value_sum.add_(w * neighbor)

    # Normalize
    result = value_sum / weight_sum.clamp(min=1e-8)  # [B, 3, H, W]

    # Back to [B, L, 3]
    return result.permute(0, 2, 3, 1).reshape(B, -1, 3)
