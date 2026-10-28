"""Pure-PyTorch point-cloud rasterisation.

The original implementation depended on pytorch3d's ``PointsRasterizer`` and
``AlphaCompositor``.  pytorch3d has no wheels for current Colab Python/Torch
versions, so this module provides a small z-buffered point splatter with the
same input/output contract.  It is used to warp the input view into the target
views (the ``warpped_target_rgbs`` conditioning image).
"""

import torch

from src.utils.typing import *


def _splat_single_view(
    points: torch.Tensor,
    colors: torch.Tensor,
    c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    image_width: int,
    image_height: int,
    point_radius_ndc: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render one view.

    Args:
        points: (N, 3) world-space points.
        colors: (N, 3) RGB values in [0, 1].
        c2w: (4, 4) camera-to-world transform.
        intrinsic: (3, 3) OpenCV camera matrix at the target resolution.
        point_radius_ndc: splat radius in normalised device coordinates
            (pytorch3d convention: 1.0 spans half the image width).

    Returns:
        rgb: (3, H, W) float tensor in [0, 1].
        depth: (1, H, W) camera-space depth, zero where no point projects.
    """
    device = points.device
    dtype = points.dtype
    height, width = image_height, image_width

    w2c = torch.inverse(c2w.to(dtype))
    rotation, translation = w2c[:3, :3], w2c[:3, 3]
    cam_points = points @ rotation.transpose(0, 1) + translation

    z = cam_points[:, 2]
    valid = z > 1e-4

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    safe_z = torch.where(valid, z, torch.ones_like(z))
    u = fx * cam_points[:, 0] / safe_z + cx
    v = fy * cam_points[:, 1] / safe_z + cy
    ui = torch.round(u).long()
    vi = torch.round(v).long()

    radius_px = max(1.0, float(point_radius_ndc) * width / 2.0)
    radius_int = int(torch.ceil(torch.tensor(radius_px)).item())
    radius_sq = radius_px * radius_px

    offsets = []
    for dy in range(-radius_int, radius_int + 1):
        for dx in range(-radius_int, radius_int + 1):
            dist_sq = float(dx * dx + dy * dy)
            if dist_sq <= radius_sq:
                offsets.append((dx, dy, max(1e-3, 1.0 - dist_sq / radius_sq)))

    flat_depth = torch.full((height * width,), float("inf"), device=device, dtype=dtype)

    # Pass 1: nearest-surface z-buffer.
    for dx, dy, _ in offsets:
        px = ui + dx
        py = vi + dy
        inside = valid & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        if not inside.any():
            continue
        flat_idx = py[inside] * width + px[inside]
        flat_depth.scatter_reduce_(0, flat_idx, z[inside], reduce="amin", include_self=True)

    # Pass 2: average colours of the points lying on the nearest surface.
    accum_rgb = torch.zeros((height * width, 3), device=device, dtype=dtype)
    accum_weight = torch.zeros((height * width,), device=device, dtype=dtype)
    for dx, dy, weight in offsets:
        px = ui + dx
        py = vi + dy
        inside = valid & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        if not inside.any():
            continue
        flat_idx = py[inside] * width + px[inside]
        point_depth = z[inside]
        # Accept points within a small tolerance of the nearest surface to
        # soften the splat without blending across depth discontinuities.
        near = point_depth <= flat_depth[flat_idx] * 1.02 + 1e-3
        if not near.any():
            continue
        selected_idx = flat_idx[near]
        selected_weight = torch.full_like(point_depth[near], weight)
        accum_rgb.index_add_(0, selected_idx, colors[inside][near] * selected_weight[:, None])
        accum_weight.index_add_(0, selected_idx, selected_weight)

    rgb = accum_rgb / accum_weight.clamp_min(1e-6)[:, None]
    rgb = rgb.reshape(height, width, 3).permute(2, 0, 1).contiguous()

    depth = flat_depth.reshape(height, width)
    depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
    return rgb, depth.unsqueeze(0)


def torch3d_rasterize_points(
    cv_cam_poses_c2w: Float[Tensor, "B 4 4"],
    in_pointcloud: Float[Tensor, "N 6"],
    intrinsic: Float[Tensor, "3 3"],
    image_width: int = 256,
    image_height: int = 256,
    point_radius: float = 0.01,
    device: str = "cuda",
) -> Tuple[Float[Tensor, "B 3 H W"], Float[Tensor, "B 1 H W"]]:
    """Rasterise a coloured point cloud into a batch of camera views.

    Drop-in replacement for the previous pytorch3d implementation.  Returns
    RGB in [0, 1] and camera-space depth maps.
    """
    device = torch.device(device)
    c2w = cv_cam_poses_c2w.to(device=device, dtype=torch.float32)
    points = in_pointcloud[:, :3].to(device=device, dtype=torch.float32)
    colors = in_pointcloud[:, 3:6].to(device=device, dtype=torch.float32)
    intrinsic = intrinsic.to(device=device, dtype=torch.float32)

    rendered_rgbs = []
    rendered_depths = []
    for idx in range(c2w.shape[0]):
        rgb, depth = _splat_single_view(
            points=points,
            colors=colors,
            c2w=c2w[idx],
            intrinsic=intrinsic,
            image_width=image_width,
            image_height=image_height,
            point_radius_ndc=point_radius,
        )
        rendered_rgbs.append(rgb)
        rendered_depths.append(depth)

    if len(rendered_rgbs) == 0:
        empty_rgb = torch.zeros((0, 3, image_height, image_width), device=device)
        empty_depth = torch.zeros((0, 1, image_height, image_width), device=device)
        return empty_rgb, empty_depth

    return torch.stack(rendered_rgbs, dim=0), torch.stack(rendered_depths, dim=0)


def render_rgb_and_depth_from_ply(
    input_pointcloud: Float[Tensor, "N 6"],
    cv_cam_poses_c2w: Float[Tensor, "B 4 4"],
    intrinsic: Float[Tensor, "B 3 3"],
    image_width: int = 256,
    image_height: int = 256,
    point_radius: float = 0.01,
    device: str = "cuda",
):
    rendered_tar_rgbs, rendered_tar_depths = torch3d_rasterize_points(
        cv_cam_poses_c2w=cv_cam_poses_c2w,
        in_pointcloud=input_pointcloud,
        intrinsic=intrinsic,
        image_height=image_height,
        image_width=image_width,
        point_radius=point_radius,
        device=device,
    )
    rendered_tar_depths: Float[Tensor, "B H W 1"] = rendered_tar_depths.permute(0, 2, 3, 1)
    rendered_masks = (rendered_tar_rgbs.mean(1, keepdim=True) > 0).float().clamp(0, 1)
    rendered_masks: Float[Tensor, "B H W 1"] = rendered_masks.permute(0, 2, 3, 1)
    rendered_tar_rgbs: Float[Tensor, "B H W 3"] = rendered_tar_rgbs.permute(0, 2, 3, 1)
    return rendered_tar_rgbs, rendered_tar_depths, rendered_masks
