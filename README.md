# ComfyUI-LCS

Training-free color control via the **Latent Color Subspace**.

Based on the paper ["The Latent Color Subspace"](https://arxiv.org/abs/2603.12261v1) (ICML 2026), which discovers that color in diffusion model latent patch spaces lives in a **3D subspace** (found via PCA with 100% color variance). The remaining 61 dimensions encode structure and detail, orthogonal to color.

This plugin manipulates colors directly in the 3D LCS during diffusion sampling — no model training, no LoRA, no post-processing.

### Tested Models

| Model | Status |
|-------|--------|
| FLUX | Tested |
| z-image | Tested |
| z-image-turbo | Tested |

The LCS is calibrated per-VAE, so it should work with any model that uses a compatible VAE architecture. If you test with other models, feel free to report your results.

> [中文版 README](README_zh.md)

## Features

- **Color Steering** — Push generated image colors toward any target color
- **Batch Multi-Color** — Apply different colors to each image in a batch
- **Tone Adjustment** — Contrast, brightness, saturation, color temperature with one-click presets
- **Localized Control** — Optional mask input for region-specific color changes
- **Latent Color Preview** — Visualize color structure without VAE decoding
- **Step Observer** — Save per-step color previews to inspect the diffusion process

## Installation

Clone into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-LCS.git
```

Install dependencies (usually already present in ComfyUI):

```bash
pip install einops safetensors
```

## Nodes

### Calibration

| Node | Description |
|------|-------------|
| **LCS Load Data** | Auto-calibrate and cache LCS data per-VAE. Fingerprints VAE weights for automatic cache management — just connect your VAE. |

Calibration runs once per VAE and is cached automatically. Subsequent runs load instantly from cache.

### Intervention

| Node | Description |
|------|-------------|
| **LCS Color Intervene** | Steer colors toward a target during generation. Supports Type I (LCS shift), Type II (HSL shift), or interpolated mode. |
| **LCS Color Batch** | Apply different target colors per batch item. Outputs `batch_size` for connecting to EmptyLatentImage. |
| **LCS Tone Adjust** | Adjust contrast, brightness, saturation, and color temperature. Includes preset dropdown with real-time slider sync. |

### Observation

| Node | Description |
|------|-------------|
| **LCS Preview Colors** | Decode latent colors to an RGB preview image without VAE decoding. |
| **LCS Step Observer** | Save per-step color preview PNGs to ComfyUI's temp directory for debugging. |

## Tone Presets

Select a preset from the dropdown — sliders update in real-time. Tweak sliders after selecting a preset for fine-tuning. Select **Custom** to set values manually without preset interference.

| Preset | Contrast | Brightness | Saturation | Temperature |
|--------|----------|------------|------------|-------------|
| Base | 1.0 | 0.0 | 1.0 | 0.0 |
| Cinematic | 1.20 | -0.05 | 0.90 | 0.05 |
| HDR | 1.40 | 0.0 | 1.20 | 0.0 |
| Vivid | 1.10 | 0.0 | 1.50 | 0.0 |
| Dramatic | 1.50 | -0.10 | 0.85 | 0.0 |
| Low Key | 1.30 | -0.20 | 0.80 | 0.0 |
| High Key | 0.80 | 0.20 | 0.90 | 0.0 |
| Warm | 1.0 | 0.0 | 1.0 | 0.15 |
| Cool | 1.0 | 0.0 | 1.0 | -0.15 |
| Desaturated | 1.0 | 0.0 | 0.40 | 0.0 |

## Quick Start

### Basic Color Control

```
LCS Load Data → LCS Color Intervene → KSampler
                       ↑
                  (pick a color)
```

1. Add **LCS Load Data** — connect your VAE (first run only, calibrates automatically)
2. Add **LCS Color Intervene** — connect MODEL and LCS_DATA
3. Pick a target color, set strength (default 1.0)
4. Connect the output MODEL to your KSampler

### Tone Adjustment

```
LCS Load Data → LCS Tone Adjust → KSampler
                      ↑
               (select preset or
                adjust sliders)
```

1. Add **LCS Load Data** → **LCS Tone Adjust**
2. Select a preset (e.g., "Cinematic") or use Custom mode
3. Fine-tune sliders as needed

### Multi-Color Batch

```
LCS Load Data → LCS Color Batch → KSampler
                      ↓
                  batch_size → EmptyLatentImage
```

Enter comma-separated hex colors (e.g., `#FF0000,#00FF00,#0000FF`). Each color applies to one batch item.

## Intervention Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **interpolated** (default) | Blends Type I and Type II using sigma as weight | General use |
| **type_i** | Direct translation in 3D LCS space | Strong global color shifts |
| **type_ii** | Per-patch HSL interpolation via bicone geometry | Precise local color control |

## Key Parameters

- **strength** (0.0–2.0): Intervention intensity. 1.0 = full, 0.0 = none.
- **start_step / end_step**: Step range for intervention. Paper optimal: 8–10 of 50 steps.
- **mask**: Optional. Bilinearly downsampled to patch grid for localized control.

## LCS vs Post-Processing

LCS operates **during** diffusion sampling, not after — this is the key difference from traditional color grading.

| | Post-Processing | LCS |
|---|---|---|
| **When** | After VAE decode, in pixel space | During sampling, in latent space |
| **Mechanism** | Color filter on the final image | Modifies 3D color subspace mid-generation |
| **Model awareness** | None — structure is already locked | Model adapts to color shifts in subsequent steps |
| **Result** | Colors can look "painted on" — shadows/skin tones may shift unnaturally | Colors look like the model intended them — content harmonizes naturally |

Example: for a warm orange sunset, post-processing tints everything orange (muddying shadows), while LCS nudges colors early in sampling so the model generates clouds, lighting, and reflections that are *coherent* with warm tones.

The paper's core insight: color and structure are **orthogonal** in the latent patch space, so you can steer one without disturbing the other — impossible in pixel space where they are entangled.

## How It Works

1. **Project**: Convert denoised prediction to 64D patch space, project onto 3D LCS basis
2. **Decompose**: Separate the 3D color coordinates from the 61D structural residual
3. **Normalize**: Transform to the reference timestep (t=50) using learned alpha/beta statistics
4. **Manipulate**: Shift colors, adjust tone, or apply other transformations in 3D LCS
5. **Reconstruct**: Denormalize, add back the preserved 61D residual, convert to latent space

The 61D residual (structure, texture, detail) is never modified — only the 3D color subspace is touched.

## File Structure

```
ComfyUI-LCS/
├── __init__.py           # Entry point (V3 + V2 compat)
├── requirements.txt
├── core/
│   ├── calibration.py    # PCA calibration pipeline
│   ├── color_space.py    # Bicone LCS ↔ HSL mapping
│   ├── defaults.py       # Alpha/beta tables from paper
│   ├── lcs_data.py       # LCSData dataclass
│   ├── patchify.py       # Patch ↔ latent conversion
│   └── timestep.py       # Sigma/timestep utilities
├── nodes/
│   ├── calibrate.py      # LCSLoadData (auto-calibrate + cache)
│   ├── intervene.py      # LCSColorIntervene, LCSColorBatch, LCSToneAdjust
│   └── observe.py        # LCSPreviewColors, LCSStepObserver
├── data/                 # Cached calibration files
└── web/js/
    └── tone_preset.js    # Frontend preset sync
```

## Citation

```bibtex
@inproceedings{lcs2026,
  title={The Latent Color Subspace},
  author={...},
  booktitle={ICML},
  year={2026},
  note={arXiv:2603.12261v1}
}
```

## License

MIT
