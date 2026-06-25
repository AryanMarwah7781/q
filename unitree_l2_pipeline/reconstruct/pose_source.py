"""External pose sources for drift-free reconstruction.

Frame-to-frame ICP drifts. The robust way to place L2 scans in a common world
frame is to use a trajectory from a SLAM / odometry system. This module imports
such a trajectory and resolves a world pose for every LiDAR scan.

The first-class source is **cuVSLAM** (NVIDIA's GPU stereo-visual-inertial SLAM):

* via Isaac ROS — ``isaac_ros_visual_slam`` publishes ``nav_msgs/Odometry`` on
  ``/visual_slam/tracking/odometry`` (the ``odom_frame -> base_link`` transform).
  Record it alongside ``/unilidar/cloud`` and import it straight from the bag.
* via the standalone ``PyCuVSLAM`` library — export its per-frame poses to a TUM
  trajectory file (``timestamp tx ty tz qx qy qz qw``) and import that.

cuVSLAM tracks the **robot base** (cameras), not the LiDAR, so we compose its
trajectory with a static LiDAR->base extrinsic::

    T_world_lidar(t) = T_world_base(t) @ T_base_lidar

``T_world_base(t)`` is interpolated (SLERP + lerp) to each scan's timestamp, so
LiDAR and camera clocks need not be sample-aligned, only synchronized.

numpy-only; no ROS, no CUDA required to *consume* a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..formats import LidarFrame, make_transform, quaternion_to_matrix


# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z)
# ---------------------------------------------------------------------------

def _normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two ``(w,x,y,z)`` quaternions."""
    q0 = _normalize(np.asarray(q0, float))
    q1 = _normalize(np.asarray(q1, float))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take the shorter arc
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # nearly parallel -> linear
        return _normalize(q0 + t * (q1 - q0))
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta0 * t
    sin0 = np.sin(theta0)
    s0 = np.sin(theta0 - theta) / sin0
    s1 = np.sin(theta) / sin0
    return _normalize(s0 * q0 + s1 * q1)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """A time-ordered set of world poses, queryable at arbitrary timestamps."""

    stamps: np.ndarray            # (N,) seconds, ascending
    positions: np.ndarray         # (N, 3)
    quaternions: np.ndarray       # (N, 4) w,x,y,z

    def __post_init__(self):
        order = np.argsort(self.stamps)
        self.stamps = np.asarray(self.stamps, float)[order]
        self.positions = np.asarray(self.positions, float)[order]
        self.quaternions = np.asarray(self.quaternions, float)[order]

    def __len__(self) -> int:
        return int(self.stamps.shape[0])

    @property
    def duration(self) -> float:
        return float(self.stamps[-1] - self.stamps[0]) if len(self) else 0.0

    def pose_at(self, t: float) -> np.ndarray:
        """Interpolated 4x4 world pose at time ``t`` (clamped at the ends)."""
        st = self.stamps
        if t <= st[0]:
            return make_transform(quaternion_to_matrix(self.quaternions[0]),
                                  self.positions[0])
        if t >= st[-1]:
            return make_transform(quaternion_to_matrix(self.quaternions[-1]),
                                  self.positions[-1])
        j = int(np.searchsorted(st, t))
        t0, t1 = st[j - 1], st[j]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        pos = (1 - a) * self.positions[j - 1] + a * self.positions[j]
        quat = slerp(self.quaternions[j - 1], self.quaternions[j], a)
        return make_transform(quaternion_to_matrix(quat), pos)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tum(path: str | Path) -> Trajectory:
    """Load a TUM trajectory file: ``timestamp tx ty tz qx qy qz qw`` per line."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append([float(x) for x in parts[:8]])
    arr = np.asarray(rows, float)
    if arr.size == 0:
        raise ValueError(f"no poses parsed from {path}")
    stamps = arr[:, 0]
    pos = arr[:, 1:4]
    # TUM stores qx,qy,qz,qw -> reorder to w,x,y,z
    quat = arr[:, [7, 4, 5, 6]]
    return Trajectory(stamps, pos, quat)


def save_tum(path: str | Path, trajectory: Trajectory) -> None:
    """Write a Trajectory to a TUM file (handy for round-tripping / debugging)."""
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, p, q in zip(trajectory.stamps, trajectory.positions,
                           trajectory.quaternions):
            w, x, y, z = q
            f.write(f"{t:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{x:.8f} {y:.8f} {z:.8f} {w:.8f}\n")


def load_cuvslam_bag(path: str | Path,
                     topic: str = "/visual_slam/tracking/odometry") -> Trajectory:
    """Load a cuVSLAM / Isaac ROS odometry topic from a ROS1 bag."""
    from ..ingest.rosbag import iter_bag_odometry

    stamps, pos, quat = [], [], []
    for stamp, position, quaternion_wxyz in iter_bag_odometry(path, topic):
        stamps.append(stamp)
        pos.append(position)
        quat.append(quaternion_wxyz)
    if not stamps:
        raise ValueError(f"no odometry on topic {topic!r} in {path}")
    return Trajectory(np.array(stamps), np.array(pos), np.array(quat))


# ---------------------------------------------------------------------------
# Extrinsics + per-frame pose resolution
# ---------------------------------------------------------------------------

def parse_extrinsic(spec) -> np.ndarray:
    """Build a 4x4 LiDAR->base extrinsic from several conveniences.

    Accepts: None / "identity" (identity), a 16-list (row-major 4x4), or a
    7-list ``x y z qx qy qz qw``.
    """
    if spec is None or (isinstance(spec, str) and spec.strip() in ("", "identity")):
        return np.eye(4)
    if isinstance(spec, str):
        spec = [float(v) for v in spec.replace(",", " ").split()]
    spec = np.asarray(spec, float).reshape(-1)
    if spec.size == 16:
        return spec.reshape(4, 4)
    if spec.size == 7:
        x, y, z, qx, qy, qz, qw = spec
        R = quaternion_to_matrix([qw, qx, qy, qz])
        return make_transform(R, [x, y, z])
    raise ValueError("extrinsic must be 'identity', 7 values (x y z qx qy qz qw), "
                     "or 16 values (row-major 4x4)")


def poses_for_frames(
    frames: list[LidarFrame],
    trajectory: Trajectory,
    extrinsic: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Resolve a world pose per frame: ``T_world_base(stamp) @ T_base_lidar``."""
    T_base_lidar = np.eye(4) if extrinsic is None else np.asarray(extrinsic, float)
    out = []
    for f in frames:
        T_world_base = trajectory.pose_at(f.stamp)
        out.append(T_world_base @ T_base_lidar)
    return out
