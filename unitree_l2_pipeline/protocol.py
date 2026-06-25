"""Unitree 4D LiDAR L2 network/serial protocol constants and helpers.

The L2 streams point-cloud and IMU data over UDP (or USB serial). These are the
factory defaults documented in the *Unitree 4D LiDAR L2 User Manual* and the
``unilidar_sdk2`` README:

============================  =================
Parameter                     Default
============================  =================
LiDAR IP                      192.168.1.62
Host / server IP              192.168.1.2
LiDAR data UDP port           6101  (lidar -> host)
Host receive UDP port         6201  (host listens here)
Command UDP port              6101  (host -> lidar)
Serial device                 /dev/ttyACM0
============================  =================

The raw on-wire framing is a Unitree-proprietary binary protocol (handled inside
the closed SDK). For an open pipeline we treat the *decoded* point as the unit of
exchange and provide a simple, well-documented UDP payload format used by
:mod:`unitree_l2_pipeline.ingest.udp_capture` and the synthetic streamer so the
ingest path is exercisable without the vendor binary. When you have the real SDK,
point ingest at its callback instead — everything downstream is identical because
both produce :class:`~unitree_l2_pipeline.formats.LidarFrame` objects.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .formats import POINT_DTYPE, LidarFrame

# ---------------------------------------------------------------------------
# Factory network defaults
# ---------------------------------------------------------------------------

LIDAR_IP = "192.168.1.62"
HOST_IP = "192.168.1.2"
LIDAR_PORT = 6101          # lidar -> host (data) and host -> lidar (commands)
HOST_RECV_PORT = 6201      # host listens here for point/imu datagrams
SERIAL_DEVICE = "/dev/ttyACM0"
SERIAL_BAUD = 4_000_000

# ---------------------------------------------------------------------------
# Open UDP payload format used by this pipeline's capture/replay path.
#
#   magic   : 4 bytes  b"UL2P"
#   version : uint8    = 1
#   kind     : uint8    0 = pointcloud, 1 = imu
#   reserved: uint16
#   stamp   : float64  (seconds)
#   frame_id: uint32
#   count   : uint32   number of points (kind=0) else 0
#   --- kind 0: count * PointUnitree records (24 bytes each) ---
#   --- kind 1: quaternion[4] gyro[3] acc[3] as float32 ---
# ---------------------------------------------------------------------------

MAGIC = b"UL2P"
VERSION = 1
KIND_POINTCLOUD = 0
KIND_IMU = 1
_HEADER = struct.Struct("<4sBBHdII")  # 24 bytes
#: Conservative MTU-safe payload split; points per datagram.
MAX_POINTS_PER_DATAGRAM = 1400 // POINT_DTYPE.itemsize


@dataclass
class PacketHeader:
    kind: int
    stamp: float
    frame_id: int
    count: int


def encode_pointcloud_datagrams(frame: LidarFrame) -> list[bytes]:
    """Serialize a frame's points into one or more UDP datagrams.

    Splitting keeps each datagram below a typical 1500-byte MTU. All datagrams
    of a frame share ``frame_id``; the receiver reassembles by ``frame_id``.
    """
    pts = frame.points
    out: list[bytes] = []
    n = pts.shape[0]
    for start in range(0, max(n, 1), MAX_POINTS_PER_DATAGRAM):
        chunk = pts[start:start + MAX_POINTS_PER_DATAGRAM]
        head = _HEADER.pack(
            MAGIC, VERSION, KIND_POINTCLOUD, 0,
            float(frame.stamp), int(frame.frame_id) & 0xFFFFFFFF, chunk.shape[0],
        )
        out.append(head + chunk.tobytes())
        if n == 0:
            break
    return out


def encode_imu_datagram(stamp: float, frame_id: int, quat, gyro, acc) -> bytes:
    head = _HEADER.pack(MAGIC, VERSION, KIND_IMU, 0, float(stamp),
                        int(frame_id) & 0xFFFFFFFF, 0)
    body = np.concatenate([
        np.asarray(quat, dtype="<f4").reshape(4),
        np.asarray(gyro, dtype="<f4").reshape(3),
        np.asarray(acc, dtype="<f4").reshape(3),
    ]).tobytes()
    return head + body


def decode_datagram(buf: bytes) -> tuple[PacketHeader, np.ndarray]:
    """Decode a datagram into a header and a payload array.

    For point clouds the payload is a structured :data:`POINT_DTYPE` array; for
    IMU it is a length-10 float32 array (quat[4], gyro[3], acc[3]).
    """
    if len(buf) < _HEADER.size:
        raise ValueError("datagram shorter than header")
    magic, ver, kind, _res, stamp, frame_id, count = _HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if ver != VERSION:
        raise ValueError(f"unsupported version {ver}")
    body = buf[_HEADER.size:]
    header = PacketHeader(kind=kind, stamp=stamp, frame_id=frame_id, count=count)
    if kind == KIND_POINTCLOUD:
        payload = np.frombuffer(body[:count * POINT_DTYPE.itemsize], dtype=POINT_DTYPE)
    elif kind == KIND_IMU:
        payload = np.frombuffer(body[:10 * 4], dtype="<f4")
    else:
        raise ValueError(f"unknown kind {kind}")
    return header, payload
