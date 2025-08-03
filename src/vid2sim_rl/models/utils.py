import torch
import torch.nn as nn


def adjust_hue(images, hue_shift):
    """
    Adjusts the hue of images.

    Args:
        images (torch.Tensor): Batch of images (N, C, H, W) in [0, 1].
        hue_shift (torch.Tensor): Hue shift values in [-0.1, 0.1] (N, 1, 1, 1).

    Returns:
        torch.Tensor: Hue-adjusted images.
    """
    # Convert RGB to HSV
    hsv = rgb_to_hsv(images)
    # Shift hue
    hsv[:, 0:1, :, :] = (hsv[:, 0:1, :, :] + hue_shift) % 1.0
    # Convert back to RGB
    rgb = hsv_to_rgb(hsv)
    return rgb

def rgb_to_hsv(images):
    """
    Converts RGB images to HSV.

    Args:
        images (torch.Tensor): Batch of images (N, C, H, W) in [0, 1].

    Returns:
        torch.Tensor: HSV images (N, C, H, W).
    """
    r, g, b = images[:, 0, :, :], images[:, 1, :, :], images[:, 2, :, :]
    maxc, _ = images.max(dim=1)
    minc, _ = images.min(dim=1)
    delta = maxc - minc + 1e-8

    # Hue calculation
    hue = torch.zeros_like(maxc)
    mask = (maxc == r)
    hue[mask] = (60 * (g[mask] - b[mask]) / delta[mask]) % 360
    mask = (maxc == g)
    hue[mask] = (60 * (b[mask] - r[mask]) / delta[mask] + 120) % 360
    mask = (maxc == b)
    hue[mask] = (60 * (r[mask] - g[mask]) / delta[mask] + 240) % 360
    hue = hue / 360.0  # Normalize to [0,1]

    # Saturation calculation
    saturation = delta / (maxc + 1e-8)

    # Value calculation
    value = maxc

    hsv = torch.stack([hue, saturation, value], dim=1)
    return hsv

def hsv_to_rgb(hsv):
    """
    Converts HSV images to RGB.

    Args:
        hsv (torch.Tensor): HSV images (N, C, H, W).

    Returns:
        torch.Tensor: RGB images (N, C, H, W).
    """
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    h = h * 360.0
    c = v * s
    x = c * (1 - torch.abs((h / 60.0) % 2 - 1))
    m = v - c

    zeros = torch.zeros_like(h)

    cond = (h >= 0) & (h < 60)
    r = torch.where(cond, c, zeros)
    g = torch.where(cond, x, zeros)
    b = zeros.clone()

    cond = (h >= 60) & (h < 120)
    r = torch.where(cond, x, r)
    g = torch.where(cond, c, g)
    b = torch.where(cond, zeros, b)

    cond = (h >= 120) & (h < 180)
    r = torch.where(cond, zeros, r)
    g = torch.where(cond, c, g)
    b = torch.where(cond, x, b)

    cond = (h >= 180) & (h < 240)
    r = torch.where(cond, zeros, r)
    g = torch.where(cond, x, g)
    b = torch.where(cond, c, b)

    cond = (h >= 240) & (h < 300)
    r = torch.where(cond, x, r)
    g = torch.where(cond, zeros, g)
    b = torch.where(cond, c, b)

    cond = (h >= 300) & (h < 360)
    r = torch.where(cond, c, r)
    g = torch.where(cond, zeros, g)
    b = torch.where(cond, x, b)

    r = (r + m).unsqueeze(1)
    g = (g + m).unsqueeze(1)
    b = (b + m).unsqueeze(1)

    rgb = torch.cat([r, g, b], dim=1)
    return rgb