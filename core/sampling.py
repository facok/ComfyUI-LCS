"""Shared sampling utilities for LCS intervention hooks."""

import torch

# FLUX VAE process_in ↔ raw space conversion constants
SCALE_FACTOR = 0.3611
SHIFT_FACTOR = 0.1159


def find_step_index(sigma, sigmas):
    """Find the step index for a given sigma value in the sigma schedule.

    Uses torch.isclose for robust matching across dtype differences (e.g. bfloat16
    sigma vs float32 sample_sigmas), with argmin fallback for edge cases.
    """
    sigma_val = sigma.flatten()[0].float()
    sigmas_f = sigmas.float()
    matched = torch.isclose(sigmas_f, sigma_val, rtol=1e-3, atol=1e-5).nonzero()
    if len(matched) > 0:
        return matched[0].item()
    return (sigmas_f - sigma_val).abs().argmin().item()
