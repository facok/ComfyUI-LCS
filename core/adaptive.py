"""Schedule-aware adaptive logic for LCS color anchoring.

Derives intervention windows, strength envelopes, and phase assignments
from the sigma schedule's amplification factor (beta_50 / beta_t), replacing
all manually-tuned step/strength parameters with data-driven decisions.
"""

import math
import torch
from .defaults import get_beta_table


def compute_amplification(sigma_val, device=None):
    """Compute amplification factor A = max_k(beta_50[k] / beta_t(sigma)[k]).

    The amplification factor measures how much the normalization step inflates
    noise relative to signal. High A means corrections are dangerous (amplified
    noise dominates), low A means corrections are safe.

    sigma_val: float in [0, 1] (FLUX sigma, 1=noise, 0=clean)
    Returns: float amplification factor
    """
    beta_table = get_beta_table()  # [51, 3]
    beta_50 = beta_table[50]  # [3]

    # Convert sigma to paper timestep
    t = 50.0 * (1.0 - max(0.0, min(1.0, sigma_val)))
    t = max(0.0, min(50.0, t))
    t_low = int(t)
    t_high = min(t_low + 1, 50)
    frac = t - t_low

    beta_t = (1.0 - frac) * beta_table[t_low] + frac * beta_table[t_high]

    # Per-component ratio, take max
    beta_t_safe = beta_t.clamp(min=1e-8)
    ratios = beta_50 / beta_t_safe  # [3]
    return ratios.max().item()


def compute_step_phases(sigmas, mode):
    """Assign a phase to each sampling step based on amplification factor.

    Physics-derived constants (not empirical):
      A_MAX = 10.0   — above: normalization amplifies noise >10x → skip
      A_WARMUP = 5.0  — self_anchor only: observe phase for EMA buildup
      SIGMA_MIN = 0.15 — below: final detail refinement → skip

    sigmas: 1D tensor of sigma values for each step (length N+1, last is 0)
    mode: "smooth", "reference", or "self_anchor"

    Returns: list of N strings, each "skip" / "observe" / "correct"
    """
    A_MAX = 10.0
    A_WARMUP = 5.0
    SIGMA_MIN = 0.15

    n_steps = len(sigmas) - 1  # last sigma is terminal (0)
    phases = []

    for i in range(n_steps):
        sigma_val = float(sigmas[i])

        # Final refinement — skip
        if sigma_val < SIGMA_MIN:
            phases.append("skip")
            continue

        amp = compute_amplification(sigma_val)

        # Too noisy — skip
        if amp > A_MAX:
            phases.append("skip")
            continue

        # Self-anchor warmup zone
        if mode == "self_anchor" and amp > A_WARMUP:
            phases.append("observe")
            continue

        phases.append("correct")

    return phases


def estimate_intensity(drift_signal):
    """Map drift magnitude to intensity in [0.15, 0.6]."""
    DRIFT_SCALE = 0.2
    INTENSITY_MIN = 0.15
    INTENSITY_MAX = 0.6
    return max(INTENSITY_MIN, min(INTENSITY_MAX, drift_signal / DRIFT_SCALE))


def compute_strength_envelope(n_correction_steps):
    """Sinusoidal bell envelope over correction steps.

    sin(pi * i / (n-1)) for i in 0..n-1
    Prevents abrupt on/off at phase boundaries.
    Single step returns [1.0].

    Returns: 1D tensor of length n_correction_steps
    """
    if n_correction_steps <= 0:
        return torch.zeros(0)
    if n_correction_steps == 1:
        return torch.ones(1)
    n = n_correction_steps
    indices = torch.arange(n, dtype=torch.float32)
    return torch.sin(math.pi * indices / (n - 1))
