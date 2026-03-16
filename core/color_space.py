"""Bicone LCS ↔ HSL mapping using 8 anchor colors.

Anchors are indexed as: [Red, Blue, Green, Magenta, Cyan, Yellow, Black, White]
Indices: 0=R, 1=B, 2=G, 3=M, 4=C, 5=Y, 6=Black, 7=White
"""

import math
import torch


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
    c = (1.0 - (2.0 * l - 1.0).abs()) * s
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
    # Build 2 orthonormal basis vectors in the plane perpendicular to a
    a_unit = a / a.norm()
    # Pick an arbitrary vector not parallel to a
    arb = torch.zeros(3, device=a.device, dtype=a.dtype)
    arb[0] = 1.0
    if (a_unit[0].abs() > 0.9):
        arb[0] = 0.0
        arb[1] = 1.0
    e1 = arb - (arb * a_unit).sum() * a_unit
    e1 = e1 / (e1.norm() + 1e-10)
    e2 = torch.linalg.cross(a_unit, e1)

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
    # Standard hue for each anchor: R=0/6, B=4/6, G=2/6, M=5/6, C=3/6, Y=1/6
    anchor_hues = torch.tensor([0.0, 4.0/6.0, 2.0/6.0, 5.0/6.0, 3.0/6.0, 1.0/6.0],
                               device=c.device, dtype=c.dtype)
    sorted_hues = anchor_hues[sort_idx]

    # Piecewise linear interpolation around the circle
    h = _angle_to_hue(angle, sorted_angles, sorted_hues)

    # Saturation: distance to achromatic axis normalized by max distance
    # Max distance at this hue and lightness
    bicone_factor = 1.0 - (2.0 * l - 1.0).abs()  # [...]
    bicone_factor = bicone_factor.clamp(min=1e-10)

    # Find the hue point on the anchor polygon
    c_H = _hue_to_polygon_point(h, chromatic, anchor_angles, a_unit, e1, e2)  # [..., 3]
    c_H_dist = (c_H - c_L).norm(dim=-1) + 1e-10
    s = chroma_dist / (c_H_dist * bicone_factor)
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
    a_unit = a / a.norm()

    # Build chromatic plane basis
    arb = torch.zeros(3, device=a.device, dtype=a.dtype)
    arb[0] = 1.0
    if (a_unit[0].abs() > 0.9):
        arb[0] = 0.0
        arb[1] = 1.0
    e1 = arb - (arb * a_unit).sum() * a_unit
    e1 = e1 / (e1.norm() + 1e-10)
    e2 = torch.linalg.cross(a_unit, e1)

    # Lightness point on achromatic axis
    c_L = black + l.unsqueeze(-1) * a  # [..., 3]

    # Hue point on chromatic polygon
    c_H = _hue_to_polygon_point(h, chromatic, anchor_angles, a_unit, e1, e2)  # [..., 3]

    # Combine: c = c_L + s * (1 - |2l-1|) * (c_H - c_L)
    bicone_factor = 1.0 - (2.0 * l - 1.0).abs()  # [...]
    c = c_L + (s * bicone_factor).unsqueeze(-1) * (c_H - c_L)

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
        angle_shifted = angle.clone()
        if a_end > 2 * math.pi:
            # Wraparound segment
            mask = (angle >= a_start) | (angle < (a_end - 2 * math.pi))
            angle_shifted = torch.where(angle < a_start, angle + 2 * math.pi, angle)
        else:
            mask = (angle >= a_start) & (angle < a_end)

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


def _hue_to_polygon_point(h, chromatic, anchor_angles, a_unit, e1, e2):
    """Map hue values [...] to points on the chromatic anchor polygon in 3D LCS space.

    chromatic: [6, 3] anchor LCS positions
    """
    # Standard hue for each anchor: R=0, B=4/6, G=2/6, M=5/6, C=3/6, Y=1/6
    anchor_hues = torch.tensor([0.0, 4.0/6.0, 2.0/6.0, 5.0/6.0, 3.0/6.0, 1.0/6.0],
                               device=chromatic.device, dtype=chromatic.dtype)

    # Sort anchors by hue for polygon interpolation
    sorted_hues, sort_idx = anchor_hues.sort()
    sorted_chromatic = chromatic[sort_idx]  # [6, 3]

    # For each input hue, find which polygon segment it falls in and interpolate
    result = torch.zeros(h.shape + (3,), device=chromatic.device, dtype=chromatic.dtype)

    for i in range(6):
        j = (i + 1) % 6
        h_start = sorted_hues[i]
        h_end = sorted_hues[j]

        if j == 0:
            h_end = h_end + 1.0  # wraparound

        span = h_end - h_start
        if span < 1e-10:
            continue

        if j == 0:
            mask = (h >= h_start) | (h < sorted_hues[0])
            h_shifted = torch.where(h < h_start, h + 1.0, h)
        else:
            mask = (h >= h_start) & (h < h_end)
            h_shifted = h

        frac = ((h_shifted - h_start) / span).clamp(0, 1)

        interp = sorted_chromatic[i] + frac.unsqueeze(-1) * (sorted_chromatic[j] - sorted_chromatic[i])
        result = torch.where(mask.unsqueeze(-1), interp, result)

    return result
