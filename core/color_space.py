"""Bicone LCS ↔ HSL mapping using 8 anchor colors.

Anchors are indexed as: [Red, Blue, Green, Magenta, Cyan, Yellow, Black, White]
Indices: 0=R, 1=B, 2=G, 3=M, 4=C, 5=Y, 6=Black, 7=White
"""

import math
import torch

# Standard HSL hue for each anchor: R=0, B=4/6, G=2/6, M=5/6, C=3/6, Y=1/6
_ANCHOR_HUES = (0.0, 4.0/6.0, 2.0/6.0, 5.0/6.0, 3.0/6.0, 1.0/6.0)


def _bicone_factor(l, clamp_min=None):
    """Compute bicone scaling factor: 1 - |2L - 1|.

    At l=0.5 (equator), factor=1 (full radius).
    At l=0 or l=1 (poles), factor=0 (zero radius).

    Args:
        l: Lightness tensor [...]
        clamp_min: Optional minimum value for numerical stability

    Returns:
        Bicone factor tensor [...]
    """
    factor = 1.0 - (2.0 * l - 1.0).abs()
    if clamp_min is not None:
        factor = factor.clamp(min=clamp_min)
    return factor


def _wrap_hue_diff(diff):
    """Wrap hue differences to the shortest path on the unit circle [-0.5, 0.5]."""
    return diff - (diff > 0.5).float() + (diff < -0.5).float()


def _hue_lerp(h1, h2, t):
    """Lerp hues on the circle [0,1], taking the shortest path."""
    return (h1 + t * _wrap_hue_diff(h2 - h1)) % 1.0


def _chromatic_plane_basis(a):
    """Build orthonormal basis (a_unit, e1, e2) for the chromatic plane perpendicular to a."""
    a_unit = a / (a.norm() + 1e-10)
    arb = torch.zeros(3, device=a.device, dtype=a.dtype)
    arb[0] = 1.0
    if a_unit[0].abs() > 0.9:
        arb[0] = 0.0
        arb[1] = 1.0
    e1 = arb - (arb * a_unit).sum() * a_unit
    e1 = e1 / (e1.norm() + 1e-10)
    e2 = torch.linalg.cross(a_unit, e1)
    return a_unit, e1, e2


def hex_to_hsl(hex_str):
    """Convert "#RRGGBB" to (h, s, l) where h∈[0,1], s∈[0,1], l∈[0,1]."""
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return rgb_to_hsl(r, g, b)


def rgb_to_hsl(r, g, b):
    """Convert RGB [0,1] to HSL [0,1]."""
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    l = (cmax + cmin) / 2.0

    if delta < 1e-10:
        return 0.0, 0.0, l

    s = delta / (1.0 - abs(2.0 * l - 1.0)) if abs(2.0 * l - 1.0) < 1.0 else 0.0

    if cmax == r:
        h = ((g - b) / delta) % 6.0
    elif cmax == g:
        h = (b - r) / delta + 2.0
    else:
        h = (r - g) / delta + 4.0
    h = h / 6.0
    if h < 0:
        h += 1.0

    return h, max(0.0, min(1.0, s)), max(0.0, min(1.0, l))


def hsl_to_rgb(h, s, l):
    """Convert HSL [0,1] to RGB [0,1]. Works with scalars or tensors."""
    if isinstance(h, torch.Tensor):
        return _hsl_to_rgb_tensor(h, s, l)

    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
    m = l - c / 2.0

    h6 = h * 6.0
    if h6 < 1:
        r, g, b = c, x, 0
    elif h6 < 2:
        r, g, b = x, c, 0
    elif h6 < 3:
        r, g, b = 0, c, x
    elif h6 < 4:
        r, g, b = 0, x, c
    elif h6 < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return r + m, g + m, b + m


def _hsl_to_rgb_tensor(h, s, l):
    """Vectorized HSL→RGB for tensors."""
    c = _bicone_factor(l) * s
    h6 = h * 6.0
    x = c * (1.0 - ((h6 % 2.0) - 1.0).abs())
    m = l - c / 2.0

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    mask0 = h6 < 1
    mask1 = (h6 >= 1) & (h6 < 2)
    mask2 = (h6 >= 2) & (h6 < 3)
    mask3 = (h6 >= 3) & (h6 < 4)
    mask4 = (h6 >= 4) & (h6 < 5)
    mask5 = h6 >= 5

    r[mask0] = c[mask0]; g[mask0] = x[mask0]
    r[mask1] = x[mask1]; g[mask1] = c[mask1]
    g[mask2] = c[mask2]; b[mask2] = x[mask2]
    g[mask3] = x[mask3]; b[mask3] = c[mask3]
    r[mask4] = x[mask4]; b[mask4] = c[mask4]
    r[mask5] = c[mask5]; b[mask5] = x[mask5]

    return (r + m).clamp(0, 1), (g + m).clamp(0, 1), (b + m).clamp(0, 1)


def decode_lcs_to_hsl(c, anchor_lcs, anchor_angles):
    """Decode LCS coordinates to HSL using bicone geometry.

    c: [..., 3] LCS coordinates (normalized to t=50)
    anchor_lcs: [8, 3] anchor positions [R,B,G,M,C,Y,Black,White]
    anchor_angles: [6] hue angles of chromatic anchors in radians

    Returns: (h, s, l) each [...] in [0,1]
    """
    black = anchor_lcs[6]  # [3]
    white = anchor_lcs[7]  # [3]
    chromatic = anchor_lcs[:6]  # [6, 3]

    # Achromatic axis
    a = white - black  # [3]
    a_norm_sq = (a * a).sum() + 1e-10

    # Lightness: project onto achromatic axis
    diff = c - black  # [..., 3]
    l = (diff * a).sum(dim=-1) / a_norm_sq  # [...]
    l = l.clamp(0.0, 1.0)

    # Point on achromatic axis
    c_L = black + l.unsqueeze(-1) * a  # [..., 3]

    # Chromatic residual
    chroma_vec = c - c_L  # [..., 3]
    chroma_dist = chroma_vec.norm(dim=-1) + 1e-10  # [...]

    # Compute hue angle in chromatic plane
    a_unit, e1, e2 = _chromatic_plane_basis(a)

    # Project chromatic vector to 2D
    x_coord = (chroma_vec * e1).sum(dim=-1)  # [...]
    y_coord = (chroma_vec * e2).sum(dim=-1)  # [...]
    angle = torch.atan2(y_coord, x_coord)  # [...] radians
    angle = angle % (2 * math.pi)

    # Map angle to hue [0,1] using sorted anchor angles
    # anchor_angles are the angles of [R,B,G,M,C,Y] in the same coordinate system
    # Standard HSL hue: R=0, Y=1/6, G=2/6, C=3/6, B=4/6, M=5/6
    # But anchors may not be in that order in angle-space, so we interpolate
    sorted_angles, sort_idx = anchor_angles.sort()
    anchor_hues = torch.tensor(_ANCHOR_HUES, device=c.device, dtype=c.dtype)
    sorted_hues = anchor_hues[sort_idx]

    # Piecewise linear interpolation around the circle
    h = _angle_to_hue(angle, sorted_angles, sorted_hues)

    # Saturation: distance to achromatic axis normalized by max distance
    # Max distance at this hue and lightness
    bicone_factor = _bicone_factor(l, clamp_min=1e-10)

    # Find the chroma boundary at this hue (perpendicular to achromatic axis)
    chroma_boundary = _hue_to_chroma_vector(h, chromatic, anchor_angles, a_unit, e1, e2, black, a)
    max_radius = chroma_boundary.norm(dim=-1) + 1e-10
    s = chroma_dist / (max_radius * bicone_factor)
    s = s.clamp(0.0, 1.0)

    return h, s, l


def encode_hsl_to_lcs(h, s, l, anchor_lcs, anchor_angles):
    """Encode HSL to LCS coordinates using bicone geometry.

    h, s, l: [...] in [0,1]
    anchor_lcs: [8, 3]
    anchor_angles: [6] radians

    Returns: c [..., 3] LCS coordinates
    """
    black = anchor_lcs[6]  # [3]
    white = anchor_lcs[7]  # [3]
    chromatic = anchor_lcs[:6]  # [6, 3]

    a = white - black
    a_unit, e1, e2 = _chromatic_plane_basis(a)

    # Lightness point on achromatic axis
    c_L = black + l.unsqueeze(-1) * a  # [..., 3]

    # Chroma direction vector (equatorial radius at this hue)
    chroma_dir = _hue_to_chroma_vector(h, chromatic, anchor_angles, a_unit, e1, e2, black, a)

    # Combine: c = c_L + s * (1 - |2l-1|) * chroma_dir
    bicone_factor = _bicone_factor(l)
    c = c_L + (s * bicone_factor).unsqueeze(-1) * chroma_dir

    return c


def _angle_to_hue(angle, sorted_angles, sorted_hues):
    """Map an angle [...] to hue [0,1] via piecewise linear interpolation on anchor angles."""
    n = len(sorted_angles)
    h = torch.zeros_like(angle)

    for i in range(n):
        j = (i + 1) % n
        a_start = sorted_angles[i]
        a_end = sorted_angles[j]
        h_start = sorted_hues[i]
        h_end = sorted_hues[j]

        # Handle wraparound
        if a_end < a_start:
            a_end = a_end + 2 * math.pi
        span = a_end - a_start
        if span < 1e-10:
            continue

        # Check which angles fall in this segment
        if a_end > 2 * math.pi:
            # Wraparound segment
            mask = (angle >= a_start) | (angle < (a_end - 2 * math.pi))
            angle_shifted = torch.where(angle < a_start, angle + 2 * math.pi, angle)
        else:
            mask = (angle >= a_start) & (angle < a_end)
            angle_shifted = angle

        frac = ((angle_shifted - a_start) / span).clamp(0, 1)

        # Interpolate hue (handling hue wraparound)
        h_diff = h_end - h_start
        if abs(h_diff) > 0.5:
            if h_diff > 0:
                h_diff -= 1.0
            else:
                h_diff += 1.0
        interp = h_start + frac * h_diff
        interp = interp % 1.0

        h = torch.where(mask, interp, h)

    return h


def _hue_to_chroma_vector(h, chromatic, anchor_angles, a_unit, e1, e2, black, a):
    """Map hue values [...] to EQUATORIAL chroma direction vectors.

    Returns vectors in 3D LCS space that lie in the chromatic plane (perpendicular to a_unit)
    with magnitude equal to the equatorial chroma radius at that hue (i.e., the radius at l=0.5).

    The equatorial radius is computed by normalizing each anchor's chroma radius by its
    bicone factor (1 - |2L - 1|), where L is the anchor's lightness. This ensures proper
    round-trip encoding/decoding across the bicone.

    chromatic: [6, 3] anchor LCS positions
    anchor_angles: [6] calibrated angles of chromatic anchors (radians)
    a_unit: [3] unit vector along achromatic axis
    e1, e2: [3] orthonormal basis for chromatic plane
    black: [3] black anchor position
    a: [3] full achromatic axis vector (white - black)
    """
    # Compute each anchor's lightness (scalar projection onto achromatic axis)
    a_sq = (a * a).sum() + 1e-10
    anchor_diff = chromatic - black  # [6, 3]
    anchor_l = (anchor_diff * a).sum(dim=-1) / a_sq  # [6] lightness values

    # Project anchors onto chromatic plane to get chroma vectors
    anchor_on_axis = black + anchor_l.unsqueeze(-1) * a  # [6, 3]
    anchor_chroma = chromatic - anchor_on_axis  # [6, 3] chroma vectors
    anchor_r = anchor_chroma.norm(dim=-1)  # [6] radii at anchor lightness

    # Normalize to equatorial radii (radius at l=0.5 where bicone_factor=1)
    bicone_factors = _bicone_factor(anchor_l, clamp_min=1e-6)  # [6]
    equatorial_r = anchor_r / bicone_factors  # [6] equatorial radii

    anchor_hues = torch.tensor(_ANCHOR_HUES, device=chromatic.device, dtype=chromatic.dtype)

    # Sort by ANGLE (same as _angle_to_hue) to match segment structure
    sorted_angles, sort_idx = anchor_angles.sort()
    sorted_hues = anchor_hues[sort_idx]
    sorted_radii = equatorial_r[sort_idx]  # [6] equatorial radii

    # Iterate segments in angle order (same as _angle_to_hue)
    n = 6
    result = torch.empty(h.shape + (3,), device=chromatic.device, dtype=chromatic.dtype)

    for i in range(n):
        j = (i + 1) % n
        h_start = sorted_hues[i]
        h_end = sorted_hues[j]

        # Hue span with wraparound (same logic as _angle_to_hue)
        h_diff = h_end - h_start
        if abs(h_diff) > 0.5:
            if h_diff > 0:
                h_diff -= 1.0
            else:
                h_diff += 1.0

        if abs(h_diff) < 1e-10:
            continue

        # Determine hue range for this segment
        h_end_unwrapped = h_start + h_diff

        # Build mask for which input hues fall in this segment
        if h_diff > 0:
            if h_end_unwrapped > 1.0:
                mask = (h >= h_start) | (h < (h_end_unwrapped - 1.0))
                h_shifted = torch.where(h < h_start, h + 1.0, h)
            else:
                mask = (h >= h_start) & (h < h_end_unwrapped)
                h_shifted = h
        else:
            # Hue decreases
            if h_end_unwrapped < 0.0:
                mask = (h <= h_start) | (h > (h_end_unwrapped + 1.0))
                h_shifted = torch.where(h > h_start, h - 1.0, h)
            else:
                mask = (h <= h_start) & (h > h_end_unwrapped)
                h_shifted = h

        frac = ((h_shifted - h_start) / h_diff).clamp(0, 1)

        # Interpolate radius
        interp_r = sorted_radii[i] + frac * (sorted_radii[j] - sorted_radii[i])

        # Interpolate angle
        a_start = sorted_angles[i]
        a_end = sorted_angles[j]
        a_span = a_end - a_start
        if a_span < 0:
            a_span += 2 * math.pi
        interp_angle = (a_start + frac * a_span) % (2 * math.pi)

        # Reconstruct 3D chroma vector
        interp_vec = interp_r.unsqueeze(-1) * (
            torch.cos(interp_angle).unsqueeze(-1) * e1
            + torch.sin(interp_angle).unsqueeze(-1) * e2
        )

        result = torch.where(mask.unsqueeze(-1), interp_vec, result)

    return result
