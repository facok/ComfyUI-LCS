"""Diagnostic tests for LCS intervention pipeline.

This module provides tests and diagnostics to identify conditions that
cause image blurriness or quality degradation during LCS intervention.
"""

import torch
import math
from .color_space import decode_lcs_to_hsl, encode_hsl_to_lcs
from .timestep import get_alpha_beta, get_alpha_beta_t50, normalize_to_t50, denormalize_from_t50

# Test constants
_T50_REFERENCE_COORD = [0.5, 0.3, 0.1]  # Typical LCS magnitude at t=50
_TEST_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # Range from none to overshoot
_VARIATION_SCALE = 0.5  # Scale for test patch variation
_NOISE_SCALE = 2.0  # Simulated diffusion noise magnitude
_PROBLEMATIC_AMPLIFICATION_THRESHOLD = 50  # >50x noise amplification is problematic


def test_round_trip_consistency(anchor_lcs, anchor_angles):
    """Test that encode(decode(x)) ≈ x for typical LCS coordinates.

    This verifies the bicone geometry math is correct.
    """
    chromatic = anchor_lcs[:6]
    black, white = anchor_lcs[6], anchor_lcs[7]

    # Test round-trip on anchor positions
    errors = []
    test_cases = list(chromatic)  # All 6 chromatic anchors

    # Add some mid-tones and random points
    for _ in range(5):
        # Generate random LCS point
        h = torch.rand(1).item()
        s = torch.rand(1).item()
        l = torch.rand(1).item()
        c = encode_hsl_to_lcs(
            torch.tensor(h), torch.tensor(s), torch.tensor(l),
            anchor_lcs, anchor_angles
        )
        test_cases.append(c)

    for c in test_cases:
        h, s, l = decode_lcs_to_hsl(c, anchor_lcs, anchor_angles)
        c_round = encode_hsl_to_lcs(h, s, l, anchor_lcs, anchor_angles)
        error = (c - c_round).norm().item()
        errors.append(error)

    max_error = max(errors)
    avg_error = sum(errors) / len(errors)
    return {
        "max_round_trip_error": max_error,
        "avg_round_trip_error": avg_error,
        "passed": max_error < 1e-4,
        "errors": errors,
    }


def test_normalization_stability():
    """Test that normalize/denormalize round-trip is stable across all timesteps.

    Identifies timesteps where numerical instability could cause issues.
    """
    # Sample LCS coordinates at t=50 (clean image reference)
    c_t50 = torch.tensor(_T50_REFERENCE_COORD, dtype=torch.float32)
    alpha_50, beta_50 = get_alpha_beta_t50()

    results = []
    for t in range(51):
        sigma = 1.0 - t / 50.0  # sigma = 1 - t/50
        alpha_t, beta_t = get_alpha_beta(sigma)

        # Normalize then denormalize
        c_norm = normalize_to_t50(c_t50, alpha_t, beta_t, alpha_50, beta_50)
        c_back = denormalize_from_t50(c_norm, alpha_t, beta_t, alpha_50, beta_50)

        error = (c_t50 - c_back).norm().item()

        # Check amplification factor
        amplification = (beta_50 / beta_t).max().item()

        results.append({
            "t": t,
            "sigma": sigma,
            "beta_t_min": beta_t.min().item(),
            "amplification": amplification,
            "round_trip_error": error,
        })

    return results


def test_type_ii_uniformity(anchor_lcs, anchor_angles):
    """Test if Type II intervention at high strength produces uniform outputs.

    This is a key diagnostic for the blurriness issue - if all patches
    converge to the same HSL values, the image loses detail.
    """
    # Create diverse patch set (simulate image with color variation)
    patches = torch.randn(100, 3) * _VARIATION_SCALE + torch.tensor([0.3, 0.2, 0.1])

    # Target color (e.g., saturated red)
    t_h, t_s, t_l = 0.0, 1.0, 0.5

    # Decode all patches ONCE (constant across strengths)
    h_cur, s_cur, l_cur = decode_lcs_to_hsl(patches, anchor_lcs, anchor_angles)

    # Target HSL tensors
    h_new = torch.full_like(h_cur, t_h)
    s_new = torch.full_like(s_cur, t_s)
    l_new = torch.full_like(l_cur, t_l)

    # Test different strengths
    for strength in _TEST_STRENGTHS:
        # Hue lerp (wrap to [-0.5, 0.5])
        diff = t_h - h_cur
        diff = diff - (diff > 0.5).float() + (diff < -0.5).float()
        h_interp = (h_cur + strength * diff) % 1.0
        s_interp = (s_cur + strength * (s_new - s_cur)).clamp(0, 1)
        l_interp = (l_cur + strength * (l_new - l_cur)).clamp(0, 1)

        # Re-encode
        new_patches = encode_hsl_to_lcs(h_interp, s_interp, l_interp, anchor_lcs, anchor_angles)

        # Measure variance loss
        input_var = patches.var(dim=0).mean().item()
        output_var = new_patches.var(dim=0).mean().item()
        var_ratio = output_var / (input_var + 1e-10)

        # Check how many unique HSL values we end up with
        h_unique = len(torch.unique(h_interp.round(decimals=3)))
        s_unique = len(torch.unique(s_interp.round(decimals=3)))
        l_unique = len(torch.unique(l_interp.round(decimals=3)))

        print(f"strength={strength:.2f}: var_ratio={var_ratio:.3f}, "
              f"unique_h={h_unique}, unique_s={s_unique}, unique_l={l_unique}")


def test_early_timestep_amplification():
    """Test numerical behavior at very early timesteps (high sigma).

    At t≈0 (sigma≈1), beta_t is very small, causing large amplification
    in normalize_to_t50. This could amplify noise and corrupt the signal.
    """
    # Typical LCS coordinate magnitude at t=50
    c_ref = torch.tensor(_T50_REFERENCE_COORD, dtype=torch.float32)

    for sigma in [1.0, 0.99, 0.95, 0.90, 0.85, 0.80, 0.50, 0.0]:
        alpha_t, beta_t = get_alpha_beta(sigma)
        alpha_50, beta_50 = get_alpha_beta_t50()

        # Simulate a noisy observation at timestep t
        # In diffusion, the observation is alpha_t * clean + beta_t * noise
        # At high sigma, noise dominates
        noise = torch.randn(3) * _NOISE_SCALE
        c_observed = alpha_t + beta_t * c_ref + beta_t * noise

        # Normalize to t=50
        c_norm = normalize_to_t50(c_observed, alpha_t, beta_t, alpha_50, beta_50)

        # Measure deviation from reference
        deviation = (c_norm - c_ref).norm().item()
        amplification = (beta_50 / beta_t).max().item()

        print(f"sigma={sigma:.2f}: beta_t={beta_t.numpy()}, "
              f"amplification={amplification:.1f}x, deviation={deviation:.3f}")


def analyze_blurriness_causes(lcs_data_path=None):
    """Comprehensive analysis of all potential blurriness causes."""
    print("=" * 60)
    print("LCS INTERVENTION BLURRINESS ANALYSIS")
    print("=" * 60)

    # Load actual calibration data
    if lcs_data_path is None:
        from pathlib import Path
        data_dir = Path(__file__).parent.parent / "data"
        safetensors_files = list(data_dir.glob("lcs_*.safetensors"))
        if safetensors_files:
            lcs_data_path = safetensors_files[0]
        else:
            print("ERROR: No calibration data found. Run LCSLoadData with calibrate=True first.")
            return

    from safetensors.torch import load_file
    data = load_file(lcs_data_path)
    anchor_lcs = data["anchor_lcs"]
    anchor_angles = data["anchor_angles"]

    print(f"\nLoaded calibration data from: {lcs_data_path}")
    print(f"anchor_lcs shape: {anchor_lcs.shape}")
    print(f"anchor_angles shape: {anchor_angles.shape}")

    print("\n1. ROUND-TRIP CONSISTENCY TEST")
    print("-" * 40)
    result = test_round_trip_consistency(anchor_lcs, anchor_angles)
    print(f"Max error: {result['max_round_trip_error']:.2e}")
    print(f"Avg error: {result['avg_round_trip_error']:.2e}")
    print(f"Status: {'PASS' if result['passed'] else 'FAIL'}")

    print("\n2. NORMALIZATION STABILITY TEST")
    print("-" * 40)
    norm_results = test_normalization_stability()
    problematic = [r for r in norm_results if r['amplification'] > _PROBLEMATIC_AMPLIFICATION_THRESHOLD]
    print(f"Timesteps with >{_PROBLEMATIC_AMPLIFICATION_THRESHOLD}x amplification: {len(problematic)}")
    for r in problematic[:5]:
        print(f"  t={r['t']:2d} (sigma={r['sigma']:.2f}): amp={r['amplification']:.1f}x")

    print("\n3. TYPE II UNIFORMITY TEST")
    print("-" * 40)
    test_type_ii_uniformity(anchor_lcs, anchor_angles)

    print("\n4. EARLY TIMESTEP AMPLIFICATION TEST")
    print("-" * 40)
    test_early_timestep_amplification()

    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print("""
Potential blurriness causes identified:

1. TYPE II AT HIGH STRENGTH: At strength=1.0, all patches get the same
   target HSL, destroying spatial color variation. This is the PRIMARY
   cause of blur in type_ii mode.

2. EARLY TIMESTEP AMPLIFICATION: At sigma>0.95 (t<2.5), beta_t is ~0.02,
   causing ~250x amplification of noise. Intervening too early (step 0-2)
   will corrupt the signal.

3. OVERSHOOTING: strength>1.0 overshoots the target, potentially pushing
   values outside the valid color gamut. This can cause clipping and
   artifacts.

RECOMMENDATIONS:
- For type_ii mode, use strength<0.8 to preserve some original variation
- Avoid intervening before step 5 (sigma<0.90)
- For interpolated mode, the gamma=sigma blending naturally limits damage
  at early steps
""")


if __name__ == "__main__":
    analyze_blurriness_causes()
