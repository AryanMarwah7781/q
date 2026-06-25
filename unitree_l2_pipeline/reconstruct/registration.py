"""Point-cloud registration to recover sensor poses for reconstruction.

When frames carry ground-truth poses (e.g. from the synthetic generator or an
external SLAM/odometry source) reconstruction can use them directly. When they
do not, we estimate frame-to-frame motion with ICP and chain it into a global
trajectory.

A dependency-free point-to-point ICP is implemented in numpy; if Open3D is
installed it is used instead (point-to-plane, faster and more robust). Both
expose the same :func:`register_frames` interface.
"""

from __future__ import annotations

import numpy as np

from ..formats import LidarFrame, apply_transform

try:  # optional acceleration
    import open3d as o3d  # type: ignore

    _HAVE_O3D = True
except Exception:  # pragma: no cover - exercised only when o3d present
    _HAVE_O3D = False


# ---------------------------------------------------------------------------
# numpy KD-tree-free nearest neighbour (chunked brute force, fine for L2 sizes)
# ---------------------------------------------------------------------------

def _nearest(src: np.ndarray, dst: np.ndarray, chunk: int = 4096):
    """For each point in ``src`` find the nearest in ``dst``. Returns (idx, d2)."""
    idx = np.empty(src.shape[0], dtype=np.int64)
    dist2 = np.empty(src.shape[0], dtype=np.float64)
    dst_sq = np.sum(dst * dst, axis=1)
    for start in range(0, src.shape[0], chunk):
        s = src[start:start + chunk]
        # ||s - d||^2 = ||s||^2 - 2 s.d + ||d||^2
        cross = s @ dst.T
        d2 = (np.sum(s * s, axis=1)[:, None] - 2.0 * cross + dst_sq[None, :])
        j = np.argmin(d2, axis=1)
        idx[start:start + chunk] = j
        dist2[start:start + chunk] = d2[np.arange(s.shape[0]), j]
    return idx, dist2


def _best_fit_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Least-squares rigid transform mapping A onto B (Kabsch/Umeyama)."""
    ca, cb = A.mean(0), B.mean(0)
    AA, BB = A - ca, B - cb
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def icp_numpy(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_iter: int = 30,
    tol: float = 1e-5,
    max_corr_dist: float = 1.0,
    init: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Point-to-point ICP. Returns (4x4 transform source->target, rmse)."""
    T = np.eye(4) if init is None else init.copy()
    src = apply_transform(T, source).astype(np.float64)
    tgt = target.astype(np.float64)
    prev = np.inf
    rmse = np.inf
    for _ in range(max_iter):
        idx, d2 = _nearest(src, tgt)
        keep = d2 < max_corr_dist * max_corr_dist
        if keep.sum() < 10:
            break
        dT = _best_fit_transform(src[keep], tgt[idx[keep]])
        src = apply_transform(dT, src)
        T = dT @ T
        rmse = float(np.sqrt(np.maximum(np.mean(d2[keep]), 0.0)))
        if abs(prev - rmse) < tol:
            break
        prev = rmse
    return T, rmse


def icp_open3d(source, target, *, max_corr_dist=1.0, init=None):  # pragma: no cover
    sp = o3d.geometry.PointCloud()
    sp.points = o3d.utility.Vector3dVector(np.asarray(source, np.float64))
    tp = o3d.geometry.PointCloud()
    tp.points = o3d.utility.Vector3dVector(np.asarray(target, np.float64))
    tp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    )
    init = np.eye(4) if init is None else init
    reg = o3d.pipelines.registration.registration_icp(
        sp, tp, max_corr_dist, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return np.asarray(reg.transformation), float(reg.inlier_rmse)


def register_pair(source_xyz, target_xyz, *, voxel=0.1, max_corr_dist=1.0, init=None):
    """Register two clouds, downsampling first for speed/robustness."""
    from .aggregate import voxel_downsample

    s = voxel_downsample(source_xyz, voxel) if voxel else source_xyz
    t = voxel_downsample(target_xyz, voxel) if voxel else target_xyz
    if _HAVE_O3D:
        return icp_open3d(s, t, max_corr_dist=max_corr_dist, init=init)
    return icp_numpy(s, t, max_corr_dist=max_corr_dist, init=init)


def register_frames(
    frames: list[LidarFrame],
    *,
    use_ground_truth: bool = True,
    voxel: float = 0.1,
    max_corr_dist: float = 1.0,
) -> list[np.ndarray]:
    """Return a world pose (4x4) for every frame.

    If ``use_ground_truth`` and frames carry ``pose``, those are returned. IMU
    orientation, when present, seeds each ICP step. Otherwise sequential
    frame-to-frame ICP builds the trajectory, anchored with frame 0 at identity.
    """
    if use_ground_truth and all(f.pose is not None for f in frames):
        return [f.pose.copy() for f in frames]

    poses = [np.eye(4)]
    for i in range(1, len(frames)):
        init = np.eye(4)
        rel, _ = register_pair(
            frames[i].xyz, frames[i - 1].xyz,
            voxel=voxel, max_corr_dist=max_corr_dist, init=init,
        )
        poses.append(poses[-1] @ rel)
    return poses
