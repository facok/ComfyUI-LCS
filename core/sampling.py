"""Shared sampling utilities for LCS intervention hooks."""

import torch

# Legacy FLUX constants — kept for backward compatibility with observe.py preview
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


def denoised_to_raw(denoised, model):
    """Convert denoised tensor from process_in space to raw VAE space.

    Uses the model's latent_format.process_out (inverse of process_in).
    Works for any model: FLUX (scale+shift), LTXV (identity), SD (scale), etc.
    """
    return model.latent_format.process_out(denoised)


def raw_to_denoised(raw, model):
    """Convert raw VAE space tensor back to process_in space.

    Uses the model's latent_format.process_in.
    """
    return model.latent_format.process_in(raw)


def unpack_video_if_needed(denoised, args):
    """Unpack LTXAV-style packed latents if detected.

    LTXAV packs video [B,128,F,H,W] + audio [B,ch,T,freq] into [B,1,flat].
    Returns (tensor_to_process, pack_info) where pack_info is None for
    non-packed formats or a dict for repacking.
    """
    # Detect packed format: shape [B, 1, flat] with very large last dim
    if denoised.ndim == 3 and denoised.shape[1] == 1:
        # Try to find latent_shapes from cond data
        cond = args.get("cond")
        latent_shapes = _extract_latent_shapes(cond)
        if latent_shapes is not None and len(latent_shapes) > 1:
            import comfy.utils
            tensors = comfy.utils.unpack_latents(denoised, latent_shapes)
            # tensors[0] = video [B, 128, F, H, W], tensors[1] = audio [B, ch, T, freq]
            return tensors[0], {"packed": True, "latent_shapes": latent_shapes,
                                "other_tensors": tensors[1:], "original": denoised}
    return denoised, None


def repack_video_if_needed(modified, original_denoised, pack_info):
    """Repack video tensor back into LTXAV packed format if it was unpacked.

    modified: the video tensor after intervention [B, 128, F, H, W]
    original_denoised: the original packed tensor (for audio portion)
    pack_info: from unpack_video_if_needed
    """
    if pack_info is None:
        return modified
    import comfy.utils
    all_tensors = [modified] + pack_info["other_tensors"]
    packed, _ = comfy.utils.pack_latents(all_tensors)
    return packed


def _extract_latent_shapes(cond):
    """Try to extract latent_shapes from conditioning data.

    After convert_cond, cond is a list of dicts with 'model_conds' containing
    CONDConstant-wrapped values like 'latent_shapes'.
    """
    if cond is None:
        return None
    for c in cond:
        if isinstance(c, dict):
            model_conds = c.get('model_conds', {})
            if 'latent_shapes' in model_conds:
                ls = model_conds['latent_shapes']
                # CONDConstant wraps the value in .cond
                if hasattr(ls, 'cond'):
                    return ls.cond
                return ls
    return None
