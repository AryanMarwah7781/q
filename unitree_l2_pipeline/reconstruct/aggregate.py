"""Aggregate per-frame clouds into one reconstructed point cloud.

Given frames and their world poses, transform every frame into the world frame,
merge, then clean up: voxel downsample (deduplicate / uniform density) and
statistical outlier removal. The result is the reconstructed scene that gets
exported to Isaac Sim.

numpy-only; Open3D is used opportunistically for the outlier filter when present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..formats import LidarFrame, apply_transform

try:
    import open3d as o3d  # type: ignore

    _HAVE_O3D = True
except Exception:
    _HAVE_O3D = False


@dataclass
class ReconstructionConfig:
    voxel_size: float = 0.03          # metres; 0 disables downsampling
    remove_outliers: bool = True
    sor_neighbors: int = 16
    sor_std_ratio: float = 2.0
    max_range: float = 0.0            # drop points farther than this (0 = keep all)


@dataclass
class Reconstruction:
    xyz: np.ndarray            # (N,3) world points
    intensity: np.ndarray      # (N,)
    poses: list[np.ndarray]    # per-frame world poses used

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.xyz.min(0), self.xyz.max(0)


def voxel_downsample(xyz: np.ndarray, voxel: float,
                     values: np.ndarray | None = None):
    """Average points (and optional per-point ``values``) within each voxel.

    Returns downsampled xyz, or ``(xyz, values)`` if ``values`` is given.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if voxel <= 0 or xyz.shape[0] == 0:
        return (xyz.astype(np.float32), values) if values is not None else xyz.astype(np.float32)
    keys = np.floor(xyz / voxel).astype(np.int64)
    # unique voxel id via lexicographic ordering
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ks = keys[order]
    xs = xyz[order]
    new = np.ones(ks.shape[0], dtype=bool)
    new[1:] = np.any(ks[1:] != ks[:-1], axis=1)
    group = np.cumsum(new) - 1
    ngroups = int(group[-1]) + 1
    sums = np.zeros((ngroups, 3))
    counts = np.zeros(ngroups)
    np.add.at(sums, group, xs)
    np.add.at(counts, group, 1.0)
    centroids = (sums / counts[:, None]).astype(np.float32)
    if values is None:
        return centroids
    vals = np.asarray(values, dtype=np.float64)[order]
    vsum = np.zeros(ngroups)
    np.add.at(vsum, group, vals)
    return centroids, (vsum / counts).astype(np.float32)


def _grid_mean_neighbor_distance(xyz: np.ndarray, k: int) -> np.ndarray:
    """Mean distance to up to ``k`` nearest neighbours via a spatial hash grid.

    O(N * points-per-cell) rather than O(N^2): points are bucketed into cells of
    side ``cell``; for each point only its own and the 26 adjacent cells are
    searched. Robust to clouds with hundreds of thousands of points.
    """
    n = xyz.shape[0]
    xyz64 = xyz.astype(np.float64)
    lo = xyz64.min(0)
    extent = np.maximum(xyz64.max(0) - lo, 1e-6)
    # target a few points per cell -> cell ~ density^(-1/3)
    density = n / float(np.prod(extent))
    cell = max((1.0 / max(density, 1e-9)) ** (1.0 / 3.0), 1e-3)

    cidx = np.floor((xyz64 - lo) / cell).astype(np.int64)
    dims = cidx.max(0) + 3  # pad
    flat = (cidx[:, 0] * dims[1] + cidx[:, 1]) * dims[2] + cidx[:, 2]
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    # map cell-id -> slice of sorted indices
    from collections import defaultdict

    cell_to_rows: dict[int, np.ndarray] = {}
    boundaries = np.flatnonzero(np.diff(flat_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    for s, e in zip(starts, ends):
        cell_to_rows[int(flat_sorted[s])] = order[s:e]

    mean_d = np.full(n, np.inf)
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for cflat, rows in cell_to_rows.items():
        # gather candidate neighbour indices from the 27-cell neighbourhood
        c0 = cidx[rows[0]]
        cand = []
        for dx, dy, dz in offsets:
            key = int(((c0[0] + dx) * dims[1] + (c0[1] + dy)) * dims[2] + (c0[2] + dz))
            nb = cell_to_rows.get(key)
            if nb is not None:
                cand.append(nb)
        cand_idx = np.concatenate(cand)
        cand_pts = xyz64[cand_idx]
        pts = xyz64[rows]
        d2 = (np.sum(pts * pts, axis=1)[:, None]
              - 2.0 * (pts @ cand_pts.T)
              + np.sum(cand_pts * cand_pts, axis=1)[None, :])
        kk = min(k + 1, d2.shape[1])
        nn = np.partition(d2, kk - 1, axis=1)[:, :kk]
        # drop the self-distance (0) then average the rest
        nn = np.sort(nn, axis=1)[:, 1:kk]
        mean_d[rows] = np.sqrt(np.maximum(nn, 0)).mean(axis=1)
    return mean_d


def statistical_outlier_removal(xyz: np.ndarray, k: int = 16,
                                std_ratio: float = 2.0,
                                values: np.ndarray | None = None):
    """Remove points whose mean distance to k neighbours is an outlier."""
    n = xyz.shape[0]
    if n <= k + 1:
        return (xyz, values) if values is not None else xyz
    if _HAVE_O3D:  # pragma: no cover
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(xyz, np.float64))
        _, keep_idx = pc.remove_statistical_outlier(k, std_ratio)
        keep = np.zeros(n, dtype=bool)
        keep[keep_idx] = True
    else:
        mean_d = _grid_mean_neighbor_distance(xyz, k)
        finite = np.isfinite(mean_d)
        thresh = mean_d[finite].mean() + std_ratio * mean_d[finite].std()
        keep = finite & (mean_d < thresh)
    if values is not None:
        return xyz[keep], values[keep]
    return xyz[keep]


def aggregate(
    frames: list[LidarFrame],
    poses: list[np.ndarray],
    cfg: ReconstructionConfig | None = None,
) -> Reconstruction:
    """Merge frames into a single cleaned world-frame point cloud."""
    cfg = cfg or ReconstructionConfig()
    world_xyz = []
    world_i = []
    for frame, T in zip(frames, poses):
        xyz = frame.xyz
        if cfg.max_range > 0:
            r = np.linalg.norm(xyz, axis=1)
            m = r <= cfg.max_range
            xyz = xyz[m]
            inten = frame.intensity[m]
        else:
            inten = frame.intensity
        world_xyz.append(apply_transform(T, xyz))
        world_i.append(inten)

    xyz = np.concatenate(world_xyz) if world_xyz else np.zeros((0, 3), np.float32)
    inten = np.concatenate(world_i) if world_i else np.zeros((0,), np.float32)

    if cfg.voxel_size > 0 and xyz.shape[0]:
        xyz, inten = voxel_downsample(xyz, cfg.voxel_size, inten)
    if cfg.remove_outliers and xyz.shape[0]:
        xyz, inten = statistical_outlier_removal(
            xyz, cfg.sor_neighbors, cfg.sor_std_ratio, inten
        )
    return Reconstruction(xyz=xyz.astype(np.float32),
                          intensity=inten.astype(np.float32),
                          poses=list(poses))
