"""Sigma ↔ paper timestep conversion and α_t/β_t interpolation."""

import torch
from .defaults import get_alpha_table, get_beta_table


def sigma_to_paper_t(sigma):
    """Convert FLUX sigma ∈ [0,1] to paper timestep t ∈ [0,50].

    sigma=1 → noise → t=0, sigma=0 → clean → t=50.
    """
    if isinstance(sigma, torch.Tensor):
        return 50.0 * (1.0 - sigma.clamp(0.0, 1.0))
    return 50.0 * (1.0 - max(0.0, min(1.0, sigma)))


def get_alpha_beta(sigma, device=None):
    """Get interpolated α_t and β_t [3] vectors for a given sigma.

    Returns (alpha_t, beta_t) as tensors on the specified device.
    """
    t = sigma_to_paper_t(sigma)
    if isinstance(t, torch.Tensor):
        t = t.item()

    alpha_table = get_alpha_table()  # [51, 3]
    beta_table = get_beta_table()    # [51, 3]

    t = max(0.0, min(50.0, t))
    t_low = int(t)
    t_high = min(t_low + 1, 50)
    frac = t - t_low

    alpha = (1.0 - frac) * alpha_table[t_low] + frac * alpha_table[t_high]
    beta = (1.0 - frac) * beta_table[t_low] + frac * beta_table[t_high]

    if device is not None:
        alpha = alpha.to(device)
        beta = beta.to(device)
    return alpha, beta


def get_alpha_beta_t50(device=None):
    """Get α_50 and β_50 (reference timestep t=50, clean image)."""
    alpha_table = get_alpha_table()
    beta_table = get_beta_table()
    alpha_50 = alpha_table[50]
    beta_50 = beta_table[50]
    if device is not None:
        alpha_50 = alpha_50.to(device)
        beta_50 = beta_50.to(device)
    return alpha_50, beta_50


def normalize_to_t50(c, alpha_t, beta_t, alpha_50, beta_50):
    """Normalize LCS coords from timestep t to reference t=50.

    ĉ = (c - α_t) / β_t * β_50 + α_50
    c: [..., 3], alpha_t/beta_t/alpha_50/beta_50: [3]
    """
    beta_t_safe = beta_t.clone()
    beta_t_safe = torch.where(beta_t_safe.abs() < 1e-6,
                              torch.full_like(beta_t_safe, 1e-6), beta_t_safe)
    return (c - alpha_t) / beta_t_safe * beta_50 + alpha_50


def denormalize_from_t50(c_hat, alpha_t, beta_t, alpha_50, beta_50):
    """Denormalize LCS coords from reference t=50 back to timestep t.

    c = (ĉ - α_50) / β_50 * β_t + α_t
    """
    beta_50_safe = beta_50.clone()
    beta_50_safe = torch.where(beta_50_safe.abs() < 1e-6,
                               torch.full_like(beta_50_safe, 1e-6), beta_50_safe)
    return (c_hat - alpha_50) / beta_50_safe * beta_t + alpha_t
