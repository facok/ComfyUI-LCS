"""Local color relationship analysis for drift detection and correction."""

import torch
import torch.nn.functional as F


def compute_local_relationships(c, h_len, w_len, kernel_radius=2):
    """Compute per-patch relationship vector from 5x5 neighborhood.

    For each patch, cosine similarity with each of up to 24 neighbors.
    Returns [B, L, N_neighbors] relationship vectors where N_neighbors = (2*r+1)^2 - 1.
    """
    B = c.shape[0]
    r = kernel_radius
    k_size = 2 * r + 1
    n_neighbors = k_size * k_size - 1  # 24 for r=2

    # Reshape to spatial grid
    grid = c.reshape(B, h_len, w_len, 3)  # [B, H, W, 3]

    # Permute to [B, 3, H, W] for padding
    grid_chw = grid.permute(0, 3, 1, 2)  # [B, 3, H, W]
    padded = F.pad(grid_chw, (r, r, r, r), mode="replicate")  # [B, 3, H+2r, W+2r]

    # Center values — normalize for cosine similarity
    center_norm = grid_chw / grid_chw.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Pre-normalize padded tensor once (avoids per-neighbor normalization in loop)
    padded_norm = padded / padded.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Collect cosine similarities with each neighbor
    similarities = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            y_start = r + dy
            x_start = r + dx
            neighbor_norm = padded_norm[:, :, y_start:y_start + h_len, x_start:x_start + w_len]
            # Cosine similarity per pixel
            sim = (center_norm * neighbor_norm).sum(dim=1)  # [B, H, W]
            similarities.append(sim)

    # Stack to [B, H, W, N_neighbors] -> [B, L, N_neighbors]
    rel = torch.stack(similarities, dim=-1)  # [B, H, W, N_neighbors]
    return rel.reshape(B, -1, n_neighbors)


def detect_anomalies_adaptive(r_current, r_reference):
    """Compare current vs reference relationships with adaptive threshold.

    Uses per-batch robust outlier detection: threshold = median + 3.0 * 1.4826 * MAD.
    Returns anomaly_magnitude [B, L, 1] in [0, 1].
    """
    # Mean absolute difference across neighbor relationships
    diff = (r_current - r_reference).abs().mean(dim=-1)  # [B, L]

    # Per-batch robust statistics
    median = diff.median(dim=-1, keepdim=True).values  # [B, 1]
    mad = (diff - median).abs().median(dim=-1, keepdim=True).values  # [B, 1]
    threshold = median + 3.0 * 1.4826 * mad  # [B, 1]

    # Soft ramp above threshold, normalized to [0, 1]
    anomaly = (diff - threshold).clamp(min=0.0)  # [B, L]
    # Normalize per-batch: max anomaly → 1.0
    amax = anomaly.amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
    anomaly = anomaly / amax

    return anomaly.unsqueeze(-1)  # [B, L, 1]


def infer_color_from_neighbors(c, anomaly_mag, h_len, w_len, kernel_radius=2):
    """For anomalous patches, infer correct color from non-anomalous neighbors.

    Uses inverse-anomaly weighting: patches with low anomaly contribute more.
    Returns [B, L, 3] corrected colors (blended: anomalous patches get
    neighbor-inferred values, non-anomalous patches keep their original).
    """
    B = c.shape[0]
    r = kernel_radius

    # Reshape to spatial grid
    grid = c.reshape(B, h_len, w_len, 3)
    anom_grid = anomaly_mag.reshape(B, h_len, w_len, 1)

    # Pad both grid and anomaly
    grid_chw = grid.permute(0, 3, 1, 2)  # [B, 3, H, W]
    anom_chw = anom_grid.permute(0, 3, 1, 2)  # [B, 1, H, W]
    padded_c = F.pad(grid_chw, (r, r, r, r), mode="replicate")
    padded_a = F.pad(anom_chw, (r, r, r, r), mode="replicate")

    # Weight neighbors by how non-anomalous they are
    weight_sum = torch.zeros(B, 1, h_len, w_len, device=c.device, dtype=c.dtype)
    value_sum = torch.zeros(B, 3, h_len, w_len, device=c.device, dtype=c.dtype)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            y_start = r + dy
            x_start = r + dx
            neighbor_c = padded_c[:, :, y_start:y_start + h_len, x_start:x_start + w_len]
            neighbor_a = padded_a[:, :, y_start:y_start + h_len, x_start:x_start + w_len]

            # Weight: 1 - anomaly (non-anomalous neighbors get high weight)
            w = (1.0 - neighbor_a).clamp(min=0.01)  # [B, 1, H, W]
            weight_sum = weight_sum + w
            value_sum = value_sum + w * neighbor_c

    # Inferred color from neighbors
    inferred = value_sum / weight_sum.clamp(min=1e-8)  # [B, 3, H, W]
    inferred = inferred.permute(0, 2, 3, 1).reshape(B, -1, 3)  # [B, L, 3]

    # Blend: anomalous patches use inferred, non-anomalous keep original
    # anomaly_mag is [B, L, 1], range [0, ~1]
    blend = anomaly_mag.clamp(0, 1)
    return c * (1.0 - blend) + inferred * blend
