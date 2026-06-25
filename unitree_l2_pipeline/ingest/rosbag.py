"""Read Unitree L2 ROS1 ``.bag`` recordings without ROS installed.

Unitree publishes real L2 recordings as ROS1 bags (used by ``point_lio_unilidar``):

* L2 Indoor — https://oss-global-cdn.unitree.com/static/L2%20Indoor%20Point%20Cloud%20Data.bag
* L2 Park   — https://oss-global-cdn.unitree.com/static/L2%20Park%20Point%20Cloud%20Data.bag

They contain two topics::

    /unilidar/cloud   sensor_msgs/PointCloud2
    /unilidar/imu     sensor_msgs/Imu

This module is a small, dependency-free parser for the ROS1 bag v2.0 container
and for ``PointCloud2`` / ``Imu`` message serialization. It decodes the
``PointCloud2`` field layout *dynamically* (reading each field's name, offset and
datatype from the message), so it works regardless of the exact point struct the
firmware emits — it maps whatever ``x, y, z, intensity, ring`` and time-like
field (``time`` / ``t`` / ``timestamp``) are present onto a
:class:`~unitree_l2_pipeline.formats.LidarFrame`.

Compression: ``none`` and ``bz2`` chunks are handled with the standard library.
``lz4`` chunks need the optional ``lz4`` package; a clear error is raised if it
is missing.
"""

from __future__ import annotations

import bz2
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import formats
from ..formats import ImuSample, LidarFrame

CLOUD_TOPIC = "/unilidar/cloud"
IMU_TOPIC = "/unilidar/imu"

# ROS PointField datatype enum -> numpy dtype
_PF_DTYPE = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}

# bag record op codes
_OP_MSG_DATA = 0x02
_OP_BAG_HEADER = 0x03
_OP_INDEX_DATA = 0x04
_OP_CHUNK = 0x05
_OP_CHUNK_INFO = 0x06
_OP_CONNECTION = 0x07

_MAGIC = b"#ROSBAG V2.0\n"


# ---------------------------------------------------------------------------
# Low-level record parsing
# ---------------------------------------------------------------------------

def _read_header_fields(buf: bytes) -> dict[str, bytes]:
    """Parse a bag record header (sequence of ``len``-prefixed ``name=value``)."""
    fields: dict[str, bytes] = {}
    i = 0
    n = len(buf)
    while i + 4 <= n:
        (flen,) = struct.unpack_from("<I", buf, i)
        i += 4
        field = buf[i:i + flen]
        i += flen
        eq = field.index(b"=")
        fields[field[:eq].decode("ascii")] = field[eq + 1:]
    return fields


def _iter_records(buf: bytes, start: int = 0):
    """Yield ``(header_fields, data_bytes)`` for records in ``buf`` from ``start``."""
    i = start
    n = len(buf)
    while i + 4 <= n:
        (hlen,) = struct.unpack_from("<I", buf, i)
        i += 4
        header = buf[i:i + hlen]
        i += hlen
        (dlen,) = struct.unpack_from("<I", buf, i)
        i += 4
        data = buf[i:i + dlen]
        i += dlen
        yield _read_header_fields(header), data


def _decompress_chunk(compression: bytes, data: bytes, uncompressed_size: int) -> bytes:
    comp = compression.decode("ascii")
    if comp == "none":
        return data
    if comp == "bz2":
        return bz2.decompress(data)
    if comp == "lz4":
        try:
            import lz4.frame  # type: ignore

            return lz4.frame.decompress(data)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "bag uses lz4 compression; install the 'lz4' package "
                "(pip install lz4) to read it"
            ) from exc
    raise RuntimeError(f"unsupported bag compression: {comp}")


# ---------------------------------------------------------------------------
# Message deserialization
# ---------------------------------------------------------------------------

def _read_string(buf: bytes, i: int) -> tuple[str, int]:
    (slen,) = struct.unpack_from("<I", buf, i)
    i += 4
    return buf[i:i + slen].decode("utf-8", "replace"), i + slen


def _read_header(buf: bytes, i: int) -> tuple[float, int]:
    """std_msgs/Header -> (stamp_seconds, new_offset)."""
    i += 4  # seq uint32
    secs, nsecs = struct.unpack_from("<II", buf, i)
    i += 8
    _frame_id, i = _read_string(buf, i)
    return secs + nsecs * 1e-9, i


def _decode_pointcloud2(buf: bytes, frame_id: int) -> LidarFrame:
    i = 0
    stamp, i = _read_header(buf, i)
    height, width = struct.unpack_from("<II", buf, i)
    i += 8
    (nfields,) = struct.unpack_from("<I", buf, i)
    i += 4
    fields = []
    for _ in range(nfields):
        name, i = _read_string(buf, i)
        offset, datatype, count = struct.unpack_from("<IBI", buf, i)
        i += 9
        fields.append((name, offset, datatype, count))
    (is_bigendian,) = struct.unpack_from("<B", buf, i)
    i += 1
    point_step, row_step = struct.unpack_from("<II", buf, i)
    i += 8
    (data_len,) = struct.unpack_from("<I", buf, i)
    i += 4
    raw = buf[i:i + data_len]
    npoints = width * height

    # Build a structured view over the raw point buffer using the declared layout.
    by_name = {f[0]: f for f in fields}
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(npoints, point_step)

    def col(name: str):
        f = by_name.get(name)
        if f is None:
            return None
        _n, offset, datatype, _c = f
        dt = np.dtype(_PF_DTYPE[datatype]).newbyteorder(">" if is_bigendian else "<")
        sub = arr[:, offset:offset + dt.itemsize]
        return np.frombuffer(np.ascontiguousarray(sub).tobytes(), dtype=dt)

    x, y, z = col("x"), col("y"), col("z")
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    intensity = col("intensity")
    ring = col("ring")
    # time-like field varies by driver: 'time', 't', or 'timestamp'
    tcol = None
    for cand in ("time", "t", "timestamp"):
        if cand in by_name:
            tcol = col(cand)
            break

    return LidarFrame.from_arrays(
        xyz,
        intensity=intensity.astype(np.float32) if intensity is not None else None,
        time=tcol.astype(np.float32) if tcol is not None else None,
        ring=ring.astype(np.uint32) if ring is not None else None,
        stamp=stamp,
        frame_id=frame_id,
    )


def _decode_imu(buf: bytes) -> ImuSample:
    i = 0
    stamp, i = _read_header(buf, i)
    qx, qy, qz, qw = struct.unpack_from("<dddd", buf, i)
    i += 32
    i += 72  # orientation_covariance[9]
    wx, wy, wz = struct.unpack_from("<ddd", buf, i)
    i += 24
    i += 72  # angular_velocity_covariance[9]
    ax, ay, az = struct.unpack_from("<ddd", buf, i)
    return ImuSample(
        stamp=stamp,
        quaternion=np.array([qw, qx, qy, qz], dtype=np.float64),
        angular_velocity=np.array([wx, wy, wz], dtype=np.float64),
        linear_acceleration=np.array([ax, ay, az], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def iter_bag_frames(
    path: str | Path,
    *,
    cloud_topic: str = CLOUD_TOPIC,
    imu_topic: str = IMU_TOPIC,
    max_frames: int | None = None,
    attach_imu: bool = True,
) -> Iterator[LidarFrame]:
    """Yield :class:`LidarFrame` objects from a Unitree L2 ROS1 bag.

    The most recent IMU sample before each cloud is attached (``attach_imu``).
    """
    path = Path(path)
    data = path.read_bytes()
    if not data.startswith(_MAGIC):
        raise ValueError(f"{path} is not a ROS1 bag (bad magic)")

    conn_topic: dict[int, str] = {}
    produced = 0
    last_imu: ImuSample | None = None
    frame_id = 0

    def handle(fields: dict[str, bytes], rec_data: bytes):
        nonlocal produced, last_imu, frame_id
        op = fields.get("op", b"\x00")[0]
        if op == _OP_CONNECTION:
            conn = struct.unpack("<I", fields["conn"])[0]
            topic = fields.get("topic", b"").decode("ascii", "replace")
            if not topic:  # topic also lives in the connection-header data
                ch = _read_header_fields(rec_data)
                topic = ch.get("topic", b"").decode("ascii", "replace")
            conn_topic[conn] = topic
        elif op == _OP_MSG_DATA:
            conn = struct.unpack("<I", fields["conn"])[0]
            topic = conn_topic.get(conn)
            if topic == imu_topic and attach_imu:
                last_imu = _decode_imu(rec_data)
            elif topic == cloud_topic:
                frame = _decode_pointcloud2(rec_data, frame_id)
                frame_id += 1
                if attach_imu:
                    frame.imu = last_imu
                yield_frames.append(frame)

    # ROS1 bag: after the magic + bag-header record, payload is a series of
    # chunk records (op=5). Connections/messages live inside chunks (and at the
    # top level for older bags). Walk both.
    yield_frames: list[LidarFrame] = []
    for fields, rec_data in _iter_records(data, start=len(_MAGIC)):
        op = fields.get("op", b"\x00")[0]
        if op == _OP_CHUNK:
            chunk = _decompress_chunk(
                fields.get("compression", b"none"),
                rec_data,
                int(struct.unpack("<I", fields["size"])[0]) if "size" in fields else len(rec_data),
            )
            for cf, cd in _iter_records(chunk, 0):
                handle(cf, cd)
                while yield_frames:
                    yield yield_frames.pop(0)
                    produced += 1
                    if max_frames is not None and produced >= max_frames:
                        return
        elif op in (_OP_CONNECTION, _OP_MSG_DATA):
            handle(fields, rec_data)
            while yield_frames:
                yield yield_frames.pop(0)
                produced += 1
                if max_frames is not None and produced >= max_frames:
                    return


def bag_info(path: str | Path) -> dict:
    """Lightweight ``rosbag info``-style summary (topics, message counts)."""
    path = Path(path)
    data = path.read_bytes()
    if not data.startswith(_MAGIC):
        raise ValueError(f"{path} is not a ROS1 bag")
    conn_topic: dict[int, str] = {}
    conn_type: dict[int, str] = {}
    counts: dict[str, int] = {}
    compression: set[str] = set()
    for fields, rec_data in _iter_records(data, start=len(_MAGIC)):
        op = fields.get("op", b"\x00")[0]
        if op == _OP_CHUNK:
            compression.add(fields.get("compression", b"none").decode())
            chunk = _decompress_chunk(
                fields.get("compression", b"none"), rec_data,
                int(struct.unpack("<I", fields["size"])[0]) if "size" in fields else len(rec_data),
            )
            for cf, cd in _iter_records(chunk, 0):
                cop = cf.get("op", b"\x00")[0]
                if cop == _OP_CONNECTION:
                    conn = struct.unpack("<I", cf["conn"])[0]
                    ch = _read_header_fields(cd)
                    conn_topic[conn] = ch.get("topic", cf.get("topic", b"")).decode("ascii", "replace")
                    conn_type[conn] = ch.get("type", b"").decode("ascii", "replace")
                elif cop == _OP_MSG_DATA:
                    conn = struct.unpack("<I", cf["conn"])[0]
                    t = conn_topic.get(conn, f"conn{conn}")
                    counts[t] = counts.get(t, 0) + 1
    return {
        "path": str(path),
        "size_bytes": len(data),
        "compression": sorted(compression) or ["none"],
        "topics": {t: {"type": conn_type.get(c, "?"), "messages": counts.get(t, 0)}
                   for c, t in conn_topic.items()},
    }
