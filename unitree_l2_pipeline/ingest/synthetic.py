"""Synthetic Unitree L2 data sample generator.

The official ``unilidar_sdk2`` ships **no** recorded sample data, so to make this
pipeline runnable end-to-end without hardware we simulate the L2 sensor: its
18-ring dome scan pattern is ray-cast against a simple indoor scene (a room box
with a few obstacles) while the sensor moves along a trajectory. Each scan is
emitted as a :class:`~unitree_l2_pipeline.formats.LidarFrame` carrying realistic
``(x, y, z, intensity, time, ring)`` points, ground-truth poses and IMU.

Because every frame sees the scene from a different viewpoint, aggregating the
frames (using the poses, or poses estimated by registration) yields a dense,
gap-filled reconstruction of the room — exactly what the downstream stages do.

This module only depends on numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..formats import (
    L2_NUM_RINGS,
    L2_RANGE_MAX,
    L2_RANGE_MIN,
    ImuSample,
    LidarFrame,
    make_transform,
)


@dataclass
class Box:
    """Axis-aligned box with a per-surface reflectivity."""

    lo: np.ndarray
    hi: np.ndarray
    reflectivity: float = 0.6

    @staticmethod
    def of(lo, hi, reflectivity: float = 0.6) -> "Box":
        return Box(np.asarray(lo, float), np.asarray(hi, float), reflectivity)


@dataclass
class SceneConfig:
    """A room (the enclosing box) plus interior obstacle boxes."""

    room: Box = field(
        default_factory=lambda: Box.of([-5, -4, 0.0], [5, 4, 3.0], 0.5)
    )
    obstacles: list[Box] = field(
        default_factory=lambda: [
            Box.of([1.0, 1.0, 0.0], [2.0, 2.0, 1.0], 0.8),     # a crate
            Box.of([-3.0, -2.0, 0.0], [-2.2, -1.2, 1.6], 0.7),  # a cabinet
            Box.of([-1.0, 2.5, 0.0], [0.5, 3.0, 0.8], 0.65),    # a low bench
            Box.of([2.5, -3.0, 0.0], [3.0, -1.0, 2.2], 0.75),   # a pillar
        ]
    )


@dataclass
class L2ScanConfig:
    """Geometry of the L2 dome scan."""

    num_rings: int = L2_NUM_RINGS
    elev_min_deg: float = -25.0
    elev_max_deg: float = 65.0
    azimuth_step_deg: float = 0.4
    range_min: float = L2_RANGE_MIN
    range_max: float = L2_RANGE_MAX
    range_noise_m: float = 0.01
    scan_period_s: float = 0.1


# ---------------------------------------------------------------------------
# Vectorized ray casting against axis-aligned boxes
# ---------------------------------------------------------------------------

def _ray_box(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Slab ray/box intersection.

    Returns ``(t_enter, t_exit, valid)`` for rays ``o + t d``. ``o`` is ``(3,)``,
    ``d`` is ``(N,3)`` (need not be normalized). Faces are inferred by the caller.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv  # (N,3)
        t2 = (hi - o) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    t_enter = np.max(tmin, axis=1)
    t_exit = np.min(tmax, axis=1)
    valid = t_exit >= np.maximum(t_enter, 0.0)
    return t_enter, t_exit, valid


def _surface_normal(point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Approximate outward normal of the box face nearest to ``point`` (N,3)."""
    # distance to each of the 6 faces; smallest -> that face's axis/sign
    d_lo = np.abs(point - lo)
    d_hi = np.abs(point - hi)
    stacked = np.stack(
        [d_lo[:, 0], d_hi[:, 0], d_lo[:, 1], d_hi[:, 1], d_lo[:, 2], d_hi[:, 2]],
        axis=1,
    )
    face = np.argmin(stacked, axis=1)
    normals = np.zeros_like(point)
    for f, axis, sign in [
        (0, 0, -1), (1, 0, 1), (2, 1, -1), (3, 1, 1), (4, 2, -1), (5, 2, 1)
    ]:
        m = face == f
        normals[m, axis] = sign
    return normals


def cast_scene(origin: np.ndarray, dirs: np.ndarray, scene: SceneConfig):
    """Cast normalized rays ``dirs`` (N,3) from ``origin`` against the scene.

    Returns ``(t, hit, reflectivity, normal)`` where ``t`` is the range to the
    first surface, ``hit`` a boolean mask, and reflectivity/normal describe the
    struck surface (used for intensity).
    """
    n = dirs.shape[0]
    best_t = np.full(n, np.inf)
    best_refl = np.zeros(n)
    best_normal = np.zeros((n, 3))

    # Room: the sensor is inside, so the relevant hit is the *exit* (t_exit) on
    # the inner wall surface.
    te, tx, valid = _ray_box(origin, dirs, scene.room.lo, scene.room.hi)
    room_t = np.where(valid & (tx > 0), tx, np.inf)
    upd = room_t < best_t
    best_t = np.where(upd, room_t, best_t)
    if np.any(upd):
        pts = origin + dirs[upd] * room_t[upd][:, None]
        # inner wall -> inward normal (negate outward)
        best_normal[upd] = -_surface_normal(pts, scene.room.lo, scene.room.hi)
        best_refl[upd] = scene.room.reflectivity

    # Obstacles: sensor outside, relevant hit is the *entry* (t_enter > 0).
    for box in scene.obstacles:
        te, tx, valid = _ray_box(origin, dirs, box.lo, box.hi)
        obs_t = np.where(valid & (te > 0), te, np.inf)
        upd = obs_t < best_t
        best_t = np.where(upd, obs_t, best_t)
        if np.any(upd):
            pts = origin + dirs[upd] * obs_t[upd][:, None]
            best_normal[upd] = _surface_normal(pts, box.lo, box.hi)
            best_refl[upd] = box.reflectivity

    hit = np.isfinite(best_t)
    return best_t, hit, best_refl, best_normal


# ---------------------------------------------------------------------------
# Scan pattern + trajectory
# ---------------------------------------------------------------------------

def _ring_directions(cfg: L2ScanConfig, az_offset_deg: float):
    """Build (dirs, ring_idx, az) for one dome sweep in the sensor frame."""
    elevs = np.deg2rad(
        np.linspace(cfg.elev_min_deg, cfg.elev_max_deg, cfg.num_rings)
    )
    az = np.deg2rad(
        np.arange(0.0, 360.0, cfg.azimuth_step_deg) + az_offset_deg
    )
    AZ, EL = np.meshgrid(az, elevs)  # (rings, n_az)
    ce = np.cos(EL)
    dirs = np.stack([ce * np.cos(AZ), ce * np.sin(AZ), np.sin(EL)], axis=-1)
    dirs = dirs.reshape(-1, 3)
    ring_idx = np.repeat(np.arange(cfg.num_rings), az.shape[0]).astype(np.uint32)
    az_flat = np.tile(az, cfg.num_rings)
    return dirs, ring_idx, az_flat


def default_trajectory(num_frames: int) -> list[np.ndarray]:
    """A gentle elliptical walk through the room with yaw following the path."""
    poses = []
    for i in range(num_frames):
        s = i / max(num_frames - 1, 1)
        ang = 2.0 * np.pi * s
        x = 2.2 * np.cos(ang)
        y = 1.6 * np.sin(ang)
        z = 0.9 + 0.05 * np.sin(2 * ang)  # sensor mounted ~0.9 m up
        yaw = ang + np.pi / 2.0           # face along the direction of travel
        c, sn = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        poses.append(make_transform(R, [x, y, z]))
    return poses


def generate_frames(
    num_frames: int = 24,
    scene: SceneConfig | None = None,
    scan: L2ScanConfig | None = None,
    poses: list[np.ndarray] | None = None,
    seed: int = 0,
) -> Iterator[LidarFrame]:
    """Yield ``num_frames`` synthetic L2 :class:`LidarFrame` objects."""
    scene = scene or SceneConfig()
    scan = scan or L2ScanConfig()
    poses = poses or default_trajectory(num_frames)
    rng = np.random.default_rng(seed)

    for i in range(num_frames):
        T = poses[i]
        R, t = T[:3, :3], T[:3, 3]
        az_offset = (i * 13.7) % 360.0  # non-repetitive sweep, like the real L2
        dirs_s, ring_idx, az = _ring_directions(scan, az_offset)
        # rays in world frame
        dirs_w = dirs_s @ R.T
        rng_t, hit, refl, normal = cast_scene(t, dirs_w, scene)

        m = hit & (rng_t >= scan.range_min) & (rng_t <= scan.range_max)
        rng_t = rng_t[m]
        rng_t = rng_t + rng.normal(0.0, scan.range_noise_m, size=rng_t.shape)
        dirs_w_m = dirs_w[m]
        ring_m = ring_idx[m]
        az_m = az[m]

        # points in the *sensor* frame (that is what the L2 reports)
        pts_world = t + dirs_w_m * rng_t[:, None]
        pts_sensor = (pts_world - t) @ R  # R^T @ (p - t)

        # intensity ~ reflectivity * cos(incidence), attenuated with range
        cos_inc = np.abs(np.sum(-dirs_w_m * normal[m], axis=1))
        atten = np.clip(1.0 - rng_t / scan.range_max, 0.05, 1.0)
        intensity = (refl[m] * np.clip(cos_inc, 0.0, 1.0) * atten * 255.0)
        intensity += rng.normal(0.0, 2.0, size=intensity.shape)
        intensity = np.clip(intensity, 0.0, 255.0)

        # per-point time within the scan, proportional to azimuth
        ptime = (np.mod(az_m, 2 * np.pi) / (2 * np.pi)) * scan.scan_period_s

        stamp = i * scan.scan_period_s
        imu = ImuSample(
            stamp=stamp,
            quaternion=_matrix_to_quat(R),
            angular_velocity=np.zeros(3),
            linear_acceleration=np.array([0.0, 0.0, 9.81]),
        )
        frame = LidarFrame.from_arrays(
            pts_sensor,
            intensity=intensity,
            time=ptime,
            ring=ring_m,
            stamp=stamp,
            frame_id=i,
            imu=imu,
            pose=T,
        )
        yield frame


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (w, x, y, z) quaternion."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)
