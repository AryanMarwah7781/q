"""Core data structures and (de)serialization for Unitree 4D LiDAR L2 data.

The field layout mirrors the structs exposed by the official ``unilidar_sdk2``
(``PointUnitree`` / ``ScanUnitree`` / ``IMUUnitree``) so that anything produced
or consumed here is byte-for-byte compatible with the vendor SDK callbacks.

Reference (unitree_lidar_sdk.h)::

    typedef struct {
        float x, y, z;       // metres, sensor frame (right-handed, origin at
                             // the centre of the bottom mounting surface)
        float intensity;     // reflectivity / signal strength
        float time;          // seconds relative to the start of the scan
        uint32_t ring;       // laser channel index, 0 .. NUM_RINGS-1
    } PointUnitree;

    typedef struct {
        double stamp;        // scan timestamp (seconds)
        uint32_t id;         // monotonically increasing frame id
        uint32_t validPointsNum;
        PointUnitree points[];
    } ScanUnitree;

    typedef struct {
        double stamp;
        uint32_t id;
        float quaternion[4];          // w, x, y, z
        float angular_velocity[3];    // rad/s
        float linear_acceleration[3]; // m/s^2
    } IMUUnitree;
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Hardware constants for the Unitree 4D LiDAR L2
# ---------------------------------------------------------------------------

#: Number of laser rings (channels) on the L2.
L2_NUM_RINGS = 18
#: Manufacturer range limits, metres.
L2_RANGE_MIN = 0.05
L2_RANGE_MAX = 30.0

#: numpy structured dtype matching ``PointUnitree`` (little endian, packed).
POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "<f4"),
        ("time", "<f4"),
        ("ring", "<u4"),
    ]
)

#: Field order used for the plain ``Nx6`` float view of a cloud.
POINT_FIELDS = ("x", "y", "z", "intensity", "time", "ring")


@dataclass
class ImuSample:
    """A single IMU reading, mirroring ``IMUUnitree``."""

    stamp: float
    quaternion: np.ndarray  # (4,) w, x, y, z
    angular_velocity: np.ndarray  # (3,)
    linear_acceleration: np.ndarray  # (3,)

    @staticmethod
    def identity(stamp: float = 0.0) -> "ImuSample":
        return ImuSample(
            stamp=stamp,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            linear_acceleration=np.array([0.0, 0.0, 9.81], dtype=np.float64),
        )


@dataclass
class LidarFrame:
    """One scan of the L2: a structured point cloud plus metadata.

    ``points`` is a structured numpy array with :data:`POINT_DTYPE`. Helper
    accessors return contiguous views/copies suitable for geometry maths.
    """

    points: np.ndarray
    stamp: float = 0.0
    frame_id: int = 0
    imu: ImuSample | None = None
    #: Optional 4x4 sensor->world pose for this frame (ground truth or estimated)
    pose: np.ndarray | None = None

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        xyz: np.ndarray,
        intensity: np.ndarray | None = None,
        time: np.ndarray | None = None,
        ring: np.ndarray | None = None,
        *,
        stamp: float = 0.0,
        frame_id: int = 0,
        imu: ImuSample | None = None,
        pose: np.ndarray | None = None,
    ) -> "LidarFrame":
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        n = xyz.shape[0]
        pts = np.zeros(n, dtype=POINT_DTYPE)
        pts["x"], pts["y"], pts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        if intensity is not None:
            pts["intensity"] = np.asarray(intensity, dtype=np.float32).reshape(n)
        if time is not None:
            pts["time"] = np.asarray(time, dtype=np.float32).reshape(n)
        if ring is not None:
            pts["ring"] = np.asarray(ring, dtype=np.uint32).reshape(n)
        return cls(points=pts, stamp=stamp, frame_id=frame_id, imu=imu, pose=pose)

    # -- accessors -----------------------------------------------------------
    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def xyz(self) -> np.ndarray:
        """``(N, 3)`` float32 array of point coordinates (sensor frame)."""
        return np.stack(
            [self.points["x"], self.points["y"], self.points["z"]], axis=-1
        ).astype(np.float32)

    @property
    def intensity(self) -> np.ndarray:
        return self.points["intensity"].astype(np.float32)

    @property
    def ring(self) -> np.ndarray:
        return self.points["ring"].astype(np.uint32)

    def transformed_xyz(self) -> np.ndarray:
        """Point coordinates in world frame using :attr:`pose` (identity if None)."""
        xyz = self.xyz
        if self.pose is None:
            return xyz
        return apply_transform(self.pose, xyz)


# ---------------------------------------------------------------------------
# Rigid transform helpers
# ---------------------------------------------------------------------------

def apply_transform(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to ``(N, 3)`` points."""
    xyz = np.asarray(xyz, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return (xyz @ R.T + t).astype(np.float32)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


# ---------------------------------------------------------------------------
# Recorded-frame I/O (.npz canonical container)
# ---------------------------------------------------------------------------

def save_frame_npz(path: str | Path, frame: LidarFrame) -> None:
    """Persist a :class:`LidarFrame` losslessly to a ``.npz`` file."""
    path = Path(path)
    payload = {
        "points": frame.points,
        "stamp": np.float64(frame.stamp),
        "frame_id": np.uint32(frame.frame_id),
    }
    if frame.pose is not None:
        payload["pose"] = frame.pose.astype(np.float64)
    if frame.imu is not None:
        payload["imu_stamp"] = np.float64(frame.imu.stamp)
        payload["imu_quat"] = frame.imu.quaternion.astype(np.float64)
        payload["imu_gyro"] = frame.imu.angular_velocity.astype(np.float64)
        payload["imu_acc"] = frame.imu.linear_acceleration.astype(np.float64)
    np.savez_compressed(path, **payload)


def load_frame_npz(path: str | Path) -> LidarFrame:
    data = np.load(path, allow_pickle=False)
    imu = None
    if "imu_quat" in data:
        imu = ImuSample(
            stamp=float(data["imu_stamp"]),
            quaternion=data["imu_quat"],
            angular_velocity=data["imu_gyro"],
            linear_acceleration=data["imu_acc"],
        )
    pose = data["pose"] if "pose" in data else None
    return LidarFrame(
        points=data["points"],
        stamp=float(data["stamp"]),
        frame_id=int(data["frame_id"]),
        imu=imu,
        pose=pose,
    )


# ---------------------------------------------------------------------------
# PCD I/O (PCL-compatible, ascii + binary)
# ---------------------------------------------------------------------------

def save_pcd(path: str | Path, xyz: np.ndarray, intensity: np.ndarray | None = None,
             *, binary: bool = True) -> None:
    """Write an ``(N,3)`` cloud (optionally with intensity) to a ``.pcd`` file."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = xyz.shape[0]
    has_i = intensity is not None
    fields = "x y z intensity" if has_i else "x y z"
    size = "4 4 4 4" if has_i else "4 4 4"
    typ = "F F F F" if has_i else "F F F"
    count = "1 1 1 1" if has_i else "1 1 1"
    header = (
        "# .PCD v0.7 - Unitree L2 pipeline\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {size}\n"
        f"TYPE {typ}\n"
        f"COUNT {count}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA {'binary' if binary else 'ascii'}\n"
    )
    path = Path(path)
    if binary:
        if has_i:
            arr = np.empty(n, dtype=np.dtype([("x", "<f4"), ("y", "<f4"),
                                              ("z", "<f4"), ("intensity", "<f4")]))
            arr["intensity"] = intensity.astype(np.float32).reshape(n)
        else:
            arr = np.empty(n, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]))
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(arr.tobytes())
    else:
        with open(path, "w") as f:
            f.write(header)
            if has_i:
                inten = intensity.astype(np.float32).reshape(n)
                for i in range(n):
                    f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} {inten[i]}\n")
            else:
                for i in range(n):
                    f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]}\n")


def load_pcd(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an ``(N,3)`` cloud and optional intensity from a ``.pcd`` file."""
    path = Path(path)
    with open(path, "rb") as f:
        fields: list[str] = []
        data_type = "ascii"
        npoints = 0
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("POINTS"):
                npoints = int(line.split()[1])
            elif line.startswith("DATA"):
                data_type = line.split()[1]
                break
        idx = {name: i for i, name in enumerate(fields)}
        if data_type == "ascii":
            rows = np.loadtxt(f).reshape(npoints, len(fields))
        else:
            dt = np.dtype([(name, "<f4") for name in fields])
            rows_struct = np.frombuffer(f.read(npoints * dt.itemsize), dtype=dt)
            rows = np.stack([rows_struct[name] for name in fields], axis=-1)
    xyz = rows[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    inten = rows[:, idx["intensity"]].astype(np.float32) if "intensity" in idx else None
    return xyz, inten


# ---------------------------------------------------------------------------
# PLY I/O (ascii) -- handy for meshlab / open3d / blender inspection
# ---------------------------------------------------------------------------

def save_ply(path: str | Path, xyz: np.ndarray, colors: np.ndarray | None = None) -> None:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = xyz.shape[0]
    has_c = colors is not None
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_c:
        colors = np.asarray(colors).reshape(-1, 3)
        if colors.dtype != np.uint8:
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    lines.append("end_header")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        if has_c:
            for i in range(n):
                f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} "
                        f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")
        else:
            for i in range(n):
                f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]}\n")


# ---------------------------------------------------------------------------
# Concatenation helper
# ---------------------------------------------------------------------------

def concat_frames(frames: Iterable[LidarFrame]) -> np.ndarray:
    """Stack the structured point arrays of several frames into one."""
    arrays = [f.points for f in frames]
    if not arrays:
        return np.zeros(0, dtype=POINT_DTYPE)
    return np.concatenate(arrays)
