# ComfyUI-LCS

Training-free color control via the **Latent Color Subspace**, plus **sharpness control** via a discovered sharpness subspace.

> **Note:** This is an unofficial community implementation. For the official code, see [ExplainableML/LCS](https://github.com/ExplainableML/LCS).

Based on ["The Latent Color Subspace"](https://arxiv.org/abs/2603.12261v1) (ICML 2026): color in diffusion model latent patch spaces lives in a **3D subspace** (PCA captures 100% color variance), while the remaining 61 dimensions encode structure and detail orthogonally.

This plugin steers colors directly in the 3D LCS during diffusion sampling — no training, no LoRA, no post-processing.

> [中文版 README](README_zh.md)

## LCS vs Traditional Post-Processing

LCS operates **during** diffusion sampling, not after — this is the key difference from traditional color grading (Photoshop, filters, etc.).

| | Traditional Post-Processing | LCS |
|---|---|---|
| **When** | After VAE decode, in pixel space | During sampling, in latent space |
| **Mechanism** | Color filter on the final image | Modifies 3D color subspace mid-generation |
| **Model awareness** | None — structure already locked | Model adapts to color shifts in subsequent steps |
| **Result** | Colors can look "painted on" | Colors look naturally intended by the model |

For example: to get a warm orange sunset, post-processing tints everything orange (muddying shadows and skin tones), while LCS nudges the color subspace early in sampling so clouds, lighting, and reflections are *coherently* warm.

The core insight: color and structure are **orthogonal** in the latent patch space — you can steer one without disturbing the other.

## Tested Models

| Model | Status |
|-------|--------|
| FLUX | Tested |
| FLUX2.klein | Tested |
| z-image | Tested |
| z-image-turbo | Tested |
| Wan (qwen-image) | Tested |
| LTX2.3 | Tested |


LCS calibrates per-VAE, so it should work with any model using a compatible VAE. Feel free to report results with other models.

## Features

- **Color Steering** — Push colors toward any target color
- **Batch Multi-Color** — Different colors per batch item
- **Tone Adjustment** — Contrast, brightness, saturation, temperature with one-click presets
- **Color Anchor** — Zero-config color drift correction: self-anchor, reference-based, or spatial smoothing with auto mode
- **Sharpness Control** — Sharpen or blur during generation via a discovered sharpness subspace (PC1 explains ~97% variance)
- **Localized Control** — Optional mask for region-specific changes
- **Latent Color Preview** — Visualize color structure without VAE decoding
- **Step Observer** — Per-step color previews for debugging

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/facok/ComfyUI-LCS.git
```

Dependencies (usually already present in ComfyUI):

```bash
pip install einops safetensors
```

## Quick Start

### Basic Color Control

```
LCS Load Data → LCS Color Intervene → KSampler
                       ↑
                  (pick a color)
```

1. **LCS Load Data** — connect your VAE (auto-calibrates on first run)
2. **LCS Color Intervene** — connect MODEL and LCS_DATA, pick a target color
3. Connect the output MODEL to KSampler

### Tone Adjustment

```
LCS Load Data → LCS Tone Adjust → KSampler
```

1. **LCS Load Data** → **LCS Tone Adjust**
2. Select a preset (e.g., "Cinematic") or adjust sliders manually

![3d3c82eb0e89ed1608e40ac7a8cc3408](https://github.com/user-attachments/assets/62868e2d-0275-4801-a9bd-606bfea3ce2f)
![42541357](https://github.com/user-attachments/assets/fe22f09e-98ac-4281-ae40-f58232c7700f)

### Sharpness Control

```
LCS Load Data ──→ LCS Sharpness Calibrate → LCS Sharpness Intervene → KSampler
                        ↑ lcs_data
```

1. **LCS Sharpness Calibrate** — connect VAE (auto-calibrates and caches). Optionally connect `lcs_data` from LCS Load Data to ensure sharpness edits don't affect color.
2. **LCS Sharpness Intervene** — connect MODEL and SHARPNESS_DATA, set strength
   - Positive strength → sharper
   - Negative strength → blurrier
   - 0 → no change
![89814728](https://github.com/user-attachments/assets/62f036e9-0bea-4cc0-9220-af4c2fb8fa76)
### Multi-Color Batch

```
LCS Load Data → LCS Color Batch → KSampler
                      ↓
                  batch_size → EmptyLatentImage
```

Enter comma-separated hex colors (e.g., `#FF0000,#00FF00,#0000FF`). Each color applies to one batch item.

### Color Anchor (Zero-Config Drift Correction)

```
LCS Load Data → LCS Color Anchor → KSampler
```

1. **LCS Load Data** → **LCS Color Anchor** — connect MODEL and LCS_DATA
2. Set mode to **auto** (default) and leave intensity at default
3. Connect the output MODEL to KSampler

That's it. In `auto` mode, the node automatically selects the correction strategy based on which optional inputs are connected:

| Connected Inputs | Resolved Mode | Behavior |
|---|---|---|
| Nothing | self_anchor | Learns the image's color patterns early on, then prevents sudden color shifts |
| reference_image + vae | reference | Keeps generated colors close to your reference image |
| mask (no reference) | smooth | Smooths out color seams (great for inpainting) |

Intensity is also derived automatically from measured drift — no manual tuning needed.

> **When to use manual mode:** If you want full control, set mode to `smooth`, `reference`, or `self_anchor` explicitly and adjust the `intensity` slider (0–1). Auto mode is designed for zero-config "just works" usage.

## Nodes

### Calibration

| Node | Description |
|------|-------------|
| **LCS Load Data** | Auto-calibrate and cache LCS color data per-VAE. Fingerprints VAE weights for automatic cache management. |
| **LCS Sharpness Calibrate** | Discover sharpness subspace via PCA on blur stimuli. Optionally connect `lcs_data` for color-orthogonal sharpness. |

Calibration runs once per VAE and caches automatically. Subsequent runs load instantly.

### Intervention

| Node | Description |
|------|-------------|
| **LCS Color Intervene** | Steer colors toward a target. Supports Type I (LCS shift), Type II (HSL shift), or interpolated mode. |
| **LCS Color Batch** | Different target colors per batch item. Outputs `batch_size` for EmptyLatentImage. |
| **LCS Tone Adjust** | Contrast, brightness, saturation, temperature. Preset dropdown with real-time slider sync. |
| **LCS Color Anchor** | Correct color drift during sampling. Auto mode infers strategy and intensity from connected inputs. |
| **LCS Sharpness Intervene** | Control sharpness during generation. Positive = sharper, negative = blurrier. |

### Observation

| Node | Description |
|------|-------------|
| **LCS Preview Colors** | Decode latent colors to RGB preview without VAE decoding. |
| **LCS Step Observer** | Save per-step color preview PNGs to ComfyUI temp directory. |

## Intervention Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **interpolated** (default) | Blends Type I and Type II using sigma | General use |
| **type_i** | Direct translation in 3D LCS space | Strong global color shifts |
| **type_ii** | Per-patch HSL interpolation via bicone geometry | Precise local color control |

## Key Parameters

### Color Intervention
- **strength** (0.0–2.0): Intervention intensity. 1.0 = full, 0.0 = none.
- **start_step / end_step**: Step range for intervention. Paper optimal: steps 8–10 of 50.
- **mask**: Optional. Downsampled to patch grid for localized control.

### Sharpness Intervention
- **strength** (-5.0–5.0): Positive = sharper, negative = blurrier, 0 = no change.
- **start_step / end_step**: Step range (default 5–15).
- **mask**: Optional. Localized sharpness control.

> **Tip for distilled models**: Step-distilled models (e.g., z-image-turbo) use far fewer steps, so intervention should start earlier — even from step 0.

### Color Anchor

Sometimes diffusion models produce unexpected color shifts during sampling — a blue sky suddenly turns purple, or inpainting leaves visible color seams. The Color Anchor node fixes these problems by monitoring and correcting colors as the image is being generated.

**Modes:**

| Mode | What it does | When to use |
|------|-------------|----------|
| **auto** (default) | Looks at what you connected and picks the best strategy for you | Just want it to work, no config needed |
| **self_anchor** | Watches how colors evolve in early steps, then prevents sudden color jumps in later steps | General color stability, no reference needed |
| **reference** | Keeps the generated image's colors close to a reference image you provide | "Make it look like this photo's color palette" |
| **smooth** | Smooths out abrupt color boundaries between regions | Fixing visible seams after inpainting |

**How auto mode picks for you:**

1. **Which strategy?** Based on what you plugged in:
   - Connected a reference image + VAE → uses `reference`
   - Connected a mask (but no reference) → uses `smooth`
   - Connected nothing extra → uses `self_anchor`
2. **How strong?** The node measures how much color drift is actually happening, then sets the correction strength accordingly. Big drift → stronger fix. Small drift → gentle touch. The range is 0.15–0.6, so it never over-corrects or does nothing.

**What happens during sampling:**

The node runs at every sampling step but doesn't always intervene. It automatically figures out which steps are safe to correct:

1. **Early steps** (image is mostly noise) — Too early to fix colors without creating artifacts. Skipped. In self_anchor mode, the node uses these steps to *learn* the image's color patterns.
2. **Middle steps** (image is taking shape) — The sweet spot. The node applies corrections here, ramping smoothly in and out to avoid sudden changes.
3. **Late steps** (fine details) — Corrections would disturb fine detail. Skipped.

Only colors are modified — structure, texture, and detail are never touched.

**Parameters:**

- **mode**: `auto`, `smooth`, `reference`, or `self_anchor`
- **intensity** (0.0–1.0): How strong the correction is. In `auto` mode this is determined automatically. Set to 0 to disable the node entirely.
- **vae** (optional): Needed for `reference` mode to encode the reference image
- **reference_image** (optional): The image whose colors you want to match
- **mask** (optional): Only correct colors inside the masked area

## Tone Presets

Select a preset — sliders update in real-time. Tweak after selecting for fine-tuning. Select **Custom** to set values manually.

| Preset | Contrast | Brightness | Saturation | Temperature |
|--------|----------|------------|------------|-------------|
| Base | 1.0 | 0.0 | 1.0 | 0.0 |
| Cinematic | 1.20 | -0.05 | 0.90 | 0.05 |
| HDR | 1.40 | 0.0 | 1.20 | 0.0 |
| Vivid | 1.10 | 0.0 | 1.50 | 0.0 |
| Dramatic | 1.50 | -0.10 | 0.85 | 0.0 |
| Low Key | 1.30 | -0.20 | 0.80 | 0.0 |
| High Key | 0.80 | 0.20 | 0.90 | 0.0 |
| Warm | 1.05 | 0.03 | 1.10 | 0.30 |
| Cool | 1.05 | 0.0 | 1.05 | -0.30 |
| Desaturated | 1.0 | 0.0 | 0.40 | 0.0 |

## How It Works

### Color (LCS)

1. **Project** — Convert denoised prediction to 64D patch space, project onto 3D LCS basis
2. **Decompose** — Separate 3D color coordinates from the 61D structural residual
3. **Normalize** — Transform to reference timestep (t=50) using learned alpha/beta statistics
4. **Manipulate** — Shift colors, adjust tone, or apply other transformations in 3D LCS
5. **Reconstruct** — Denormalize, add back the preserved 61D residual, convert to latent space

The 61D residual (structure, texture, detail) is never modified — only the 3D color subspace is touched.

### Sharpness

Sharpness lives in a separate subspace orthogonal to color:

1. **Calibrate** — Generate grayscale noise images at multiple blur levels, VAE-encode, PCA on color-removed patch vectors. PC1 captures ~97% of sharpness variance.
2. **Intervene** — Add `strength * pc1_direction` to each patch. Since pc1_direction is orthogonal to color (calibrated with LCS removal) and DC-free (per-vector zero-mean before PCA), this modifies only spatial frequency content without affecting color or brightness.

### Color Anchor

The Color Anchor stabilizes colors without pushing them toward a specific target — it prevents drift from what the model is already generating:

1. **Decide when to act** — The node checks each sampling step: is the image still mostly noise (too early), taking shape (good time to correct), or nearly finished (too late)? It only corrects during the safe middle window.
2. **Learn the color pattern** (self_anchor) — During early noisy steps, the node watches how colors relate to their neighbors and builds a running average of these relationships. This is more reliable than tracking absolute colors, which shift naturally as the image forms.
3. **Measure drift** — On the first correction step, the node measures how much the colors have actually drifted (varies by mode: step-to-step jumps, distance from reference, or spatial roughness). This sets the correction strength in auto mode.
4. **Apply gentle corrections** — Corrections ramp smoothly in and out (no sudden jumps). Each mode corrects differently: self_anchor fixes patches that deviate from learned patterns, reference pulls toward the reference image's colors, smooth blurs out sharp color boundaries.
5. **Preserve everything else** — As with all LCS operations, only the 3D color coordinates change. Structure, texture, and detail are untouched.

## File Structure

```
ComfyUI-LCS/
├── __init__.py           # Entry point (V3 + V2 compat)
├── requirements.txt
├── core/
│   ├── adaptive.py       # Adaptive scheduling (phases, envelopes, drift estimation)
│   ├── bilateral.py      # Bilateral filter for LCS color smoothing
│   ├── calibration.py    # PCA calibration pipeline (color)
│   ├── color_space.py    # Bicone LCS ↔ HSL mapping
│   ├── defaults.py       # Alpha/beta tables from paper
│   ├── lcs_data.py       # LCSData dataclass
│   ├── patchify.py       # Patch ↔ latent conversion
│   ├── relationships.py  # Local color relationship analysis & anomaly detection
│   ├── sampling.py       # Shared constants & step utilities
│   ├── sharpness.py      # Sharpness subspace calibration
│   └── timestep.py       # Sigma/timestep utilities
├── nodes/
│   ├── anchor.py         # LCSColorAnchor (adaptive color drift correction)
│   ├── calibrate.py      # LCSLoadData (auto-calibrate + cache)
│   ├── intervene.py      # LCSColorIntervene, LCSColorBatch, LCSToneAdjust
│   ├── observe.py        # LCSPreviewColors, LCSStepObserver
│   └── sharpen.py        # LCSSharpnessCalibrate, LCSSharpnessIntervene
├── data/                 # Cached calibration files
└── web/js/
    └── tone_preset.js    # Frontend preset sync
```

## Changelog

### 2026-03-21
- **Color Anchor: auto mode** — New `auto` mode that infers correction strategy (self_anchor / reference / smooth) from connected inputs and derives intensity from measured drift. Zero-config usage.
- **Color Anchor: adaptive scheduling** — Phase assignment (observe/correct/skip) and strength envelope are derived from the sigma schedule at runtime.

### 2026-03-20
- **Sharpness Control** — New sharpness subspace discovered via PCA on blur stimuli. `LCS Sharpness Calibrate` + `LCS Sharpness Intervene` nodes. PC1 explains ~97% variance, orthogonal to color.
- **Color-orthogonal sharpness** — Optional `lcs_data` input removes color component during sharpness calibration, preventing color shift.

### 2026-03-19
- **Video VAE support (Wan)** — Handle 5D video latents in patchify/unpatchify. Per-image VAE encoding fallback for video VAEs.
- **LTXV compatibility** — Pad odd spatial dims in patchify, handle 3D tensors, skip gracefully for incompatible formats.
- **FLUX2 support** — Auto-detect 128-channel latents in unpatchify.
- **Universal latent format** — Use model's `latent_format` for space conversion instead of hardcoded FLUX constants.

### 2026-03-18
- **Tone Adjust** — `LCS Tone Adjust` node with contrast, brightness, saturation, temperature sliders. 10 presets with frontend real-time sync.
- **Color temperature** — Warm/cool shift along LCS blue-yellow axis.
- **Bicone HSL geometry** — Correct Type II intervention via bicone LCS-to-HSL mapping.

### 2026-03-17
- **Initial release** — Color steering (Type I + Type II + interpolated), batch multi-color, localized mask control, latent color preview, step observer. Per-VAE auto-calibration with caching.

## Citation

Official repository: [ExplainableML/LCS](https://github.com/ExplainableML/LCS)

```bibtex
@article{pach2026latentcolorsubspace,
  title={The Latent Color Subspace: Emergent Order in High-Dimensional Chaos},
  author={Mateusz Pach and Jessica Bader and Quentin Bouniot and Serge Belongie and Zeynep Akata},
  journal={arxiv},
  year={2026}
}
```

## Acknowledgments

Thanks to Mateusz Pach, Jessica Bader, Quentin Bouniot, Serge Belongie, and Zeynep Akata for their research making training-free color control possible.

## License

MIT
