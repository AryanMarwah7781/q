"""Parser for the Unitree L2's **native** on-wire protocol (``0x55AA050A`` framing).

This lets you process a *raw* capture straight off the device — a ``.pcap``, a raw
UDP-payload dump, or a live socket — without the vendor SDK or ROS. Pure stdlib +
numpy, so it runs anywhere (incl. Windows).

What is reliable vs. approximate
--------------------------------
* **Framing (reliable, tested).** Every packet is ``FrameHeader(12) + body + FrameTail(12)``::

      FrameHeader: uint8 magic[4]={0x55,0xAA,0x05,0x0A}; uint32 packet_type; uint32 packet_size;
      FrameTail:   uint32 crc32; uint32 msg_type_check; uint8 reserve[2]; uint8 tail[2]={0x00,0xFF};

  packet_type: 102=3D points, 103=2D points, 104=IMU, 105=version. The demux below
  splits a byte stream into typed packets and resynchronizes on the magic.

* **Body fields (faithful where documented).** For the 102 point packet (1012-byte
  body) the layout is pinned by arithmetic (16-byte DataInfo + 60 opaque
  state/param bytes + eight float params + point_num + ``uint16 ranges[300]`` in mm
  + ``uint8 intensities[300]``). Timestamps, ranges and intensities decode exactly.

* **Cartesian XYZ (APPROXIMATE).** The exact polar->Cartesian conversion uses the
  device's factory ``LidarCalibParam`` baked into the closed SDK. We reconstruct a
  spherical estimate from the packet's documented angle fields (azimuth =
  ``com_horizontal_angle_start + i*step``; elevation = ``angle_min + i*increment``).
  Good for previewing/registration; for metric-accurate clouds use the SDK or ROS
  driver (record a bag) — that path in this repo is exact.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .formats import ImuSample, LidarFrame

# ---------------------------------------------------------------------------
# Framing constants
# ---------------------------------------------------------------------------

NATIVE_MAGIC = bytes([0x55, 0xAA, 0x05, 0x0A])
TAIL_MAGIC = bytes([0x00, 0xFF])
FRAME_HEADER = struct.Struct("<4sII")     # magic, packet_type, packet_size
FRAME_TAIL = struct.Struct("<II2s2s")     # crc32, msg_type_check, reserve, tail
HEADER_SIZE = FRAME_HEADER.size           # 12
TAIL_SIZE = FRAME_TAIL.size               # 12

PT_POINT = 102        # LIDAR_POINT_DATA_PACKET_TYPE (3D)
PT_2D_POINT = 103     # LIDAR_2D_POINT_DATA_PACKET_TYPE
PT_IMU = 104          # LIDAR_IMU_DATA_PACKET_TYPE
PT_VERSION = 105      # LIDAR_VERSION_PACKET_TYPE

# 102 point body layout (bytes), pinned by 112 + 300*2 + 300*1 == 1012
POINT_BODY_SIZE = 1012
_PB_PARAM_OFFSET = 76      # after DataInfo(16) + opaque state/param(60)
_PB_POINTNUM_OFFSET = 108
_PB_RANGES_OFFSET = 112
_PB_MAX_POINTS = 300
_PB_INTENS_OFFSET = _PB_RANGES_OFFSET + _PB_MAX_POINTS * 2   # 712


@dataclass
class NativePacket:
    packet_type: int
    body: bytes
    crc32: int
    crc_ok: bool | None      # None when not checked


@dataclass
class PointScanChunk:
    """One 102 packet decoded into raw polar fields + an approximate XYZ."""

    stamp: float
    seq: int
    point_num: int
    ranges_m: np.ndarray       # (n,) metres
    intensities: np.ndarray    # (n,)
    azimuth: np.ndarray        # (n,) rad
    elevation: np.ndarray      # (n,) rad
    times: np.ndarray          # (n,) s, relative to stamp
    scan_period: float
    range_min: float
    range_max: float

    def xyz_approx(self) -> np.ndarray:
        ce = np.cos(self.elevation)
        return np.stack([
            self.ranges_m * ce * np.cos(self.azimuth),
            self.ranges_m * ce * np.sin(self.azimuth),
            self.ranges_m * np.sin(self.elevation),
        ], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# CRC
# ---------------------------------------------------------------------------

def crc32_of(header: bytes, body: bytes) -> int:
    """CRC32 over header+body (best-effort; the exact vendor coverage is undocumented)."""
    return zlib.crc32(header + body) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Single-frame encode / decode (the encoder defines our canonical layout; the
# decoder + roundtrip tests validate the framing contract)
# ---------------------------------------------------------------------------

def encode_frame(packet_type: int, body: bytes) -> bytes:
    header = FRAME_HEADER.pack(NATIVE_MAGIC, packet_type, len(body))
    crc = crc32_of(header, body)
    tail = FRAME_TAIL.pack(crc, packet_type, b"\x00\x00", TAIL_MAGIC)
    return header + body + tail


def decode_frame(buf: bytes, offset: int = 0, *, check_crc: bool = False) -> NativePacket:
    magic, ptype, psize = FRAME_HEADER.unpack_from(buf, offset)
    if magic != NATIVE_MAGIC:
        raise ValueError(f"bad frame magic {magic!r}")
    b0 = offset + HEADER_SIZE
    body = buf[b0:b0 + psize]
    crc, _mtc, _res, tail = FRAME_TAIL.unpack_from(buf, b0 + psize)
    crc_ok = None
    if check_crc:
        crc_ok = (crc == crc32_of(buf[offset:b0], body))
    return NativePacket(packet_type=ptype, body=body, crc32=crc, crc_ok=crc_ok)


def iter_frames_from_stream(buf: bytes, *, check_crc: bool = False):
    """Demux a byte stream of concatenated native frames, resyncing on the magic.

    Robust to unknown ``packet_size`` semantics: it trusts the size field when the
    tail lands correctly, otherwise scans forward to the next magic.
    """
    n = len(buf)
    i = buf.find(NATIVE_MAGIC, 0)
    while i != -1 and i + HEADER_SIZE + TAIL_SIZE <= n:
        _magic, ptype, psize = FRAME_HEADER.unpack_from(buf, i)
        end = i + HEADER_SIZE + psize + TAIL_SIZE
        ok = (end <= n
              and buf[i + HEADER_SIZE + psize + TAIL_SIZE - 2:
                      i + HEADER_SIZE + psize + TAIL_SIZE] == TAIL_MAGIC)
        if ok:
            yield decode_frame(buf, i, check_crc=check_crc)
            nxt = buf.find(NATIVE_MAGIC, end)
            i = nxt if nxt != -1 else end
        else:
            # size field didn't validate -> resync to the next magic
            i = buf.find(NATIVE_MAGIC, i + 1)


# ---------------------------------------------------------------------------
# Body decoders
# ---------------------------------------------------------------------------

def decode_point_body(body: bytes) -> PointScanChunk:
    seq, _payload, sec, nsec = struct.unpack_from("<IIII", body, 0)
    stamp = sec + nsec * 1e-9
    (com_h_start, com_h_step, scan_period, range_min, range_max,
     angle_min, angle_increment, time_increment) = struct.unpack_from(
        "<8f", body, _PB_PARAM_OFFSET)
    (point_num,) = struct.unpack_from("<I", body, _PB_POINTNUM_OFFSET)
    n = int(min(max(point_num, 0), _PB_MAX_POINTS))
    ranges = np.frombuffer(body[_PB_RANGES_OFFSET:_PB_INTENS_OFFSET],
                           dtype="<u2")[:n].astype(np.float32) / 1000.0  # mm -> m
    intens = np.frombuffer(body[_PB_INTENS_OFFSET:_PB_INTENS_OFFSET + _PB_MAX_POINTS],
                           dtype=np.uint8)[:n].astype(np.float32)
    i = np.arange(n, dtype=np.float32)
    return PointScanChunk(
        stamp=stamp, seq=int(seq), point_num=n,
        ranges_m=ranges, intensities=intens,
        azimuth=com_h_start + i * com_h_step,
        elevation=angle_min + i * angle_increment,
        times=i * time_increment,
        scan_period=scan_period, range_min=range_min, range_max=range_max,
    )


def encode_point_body(chunk_fields: dict) -> bytes:
    """Encode a 102 body from fields (used by tests and synthetic raw capture)."""
    body = bytearray(POINT_BODY_SIZE)
    struct.pack_into("<IIII", body, 0,
                     chunk_fields.get("seq", 0), POINT_BODY_SIZE,
                     chunk_fields.get("sec", 0), chunk_fields.get("nsec", 0))
    struct.pack_into("<8f", body, _PB_PARAM_OFFSET,
                     chunk_fields.get("com_h_start", 0.0),
                     chunk_fields.get("com_h_step", 0.0),
                     chunk_fields.get("scan_period", 0.1),
                     chunk_fields.get("range_min", 0.05),
                     chunk_fields.get("range_max", 30.0),
                     chunk_fields.get("angle_min", 0.0),
                     chunk_fields.get("angle_increment", 0.0),
                     chunk_fields.get("time_increment", 0.0))
    ranges = np.asarray(chunk_fields.get("ranges_mm", []), dtype="<u2")
    intens = np.asarray(chunk_fields.get("intensities", []), dtype=np.uint8)
    struct.pack_into("<I", body, _PB_POINTNUM_OFFSET, len(ranges))
    body[_PB_RANGES_OFFSET:_PB_RANGES_OFFSET + ranges.nbytes] = ranges.tobytes()
    body[_PB_INTENS_OFFSET:_PB_INTENS_OFFSET + intens.nbytes] = intens.tobytes()
    return bytes(body)


def decode_imu_body(body: bytes) -> ImuSample:
    """Decode a 104 IMU body.

    DataInfo timestamp is at the front; the ten motion floats
    (quaternion[4] w,x,y,z; angular_velocity[3]; linear_acceleration[3]) are read
    from the END of the body. The trailing-offset assumption is documented because
    the body carries undocumented reserved bytes; adjust if your firmware differs.
    """
    _seq, _payload, sec, nsec = struct.unpack_from("<IIII", body, 0)
    stamp = sec + nsec * 1e-9
    floats = struct.unpack_from("<10f", body, len(body) - 40)
    quat = np.array(floats[0:4], dtype=np.float64)
    gyro = np.array(floats[4:7], dtype=np.float64)
    acc = np.array(floats[7:10], dtype=np.float64)
    return ImuSample(stamp=stamp, quaternion=quat,
                     angular_velocity=gyro, linear_acceleration=acc)


def encode_imu_body(stamp_sec: int, stamp_nsec: int, quat, gyro, acc,
                    *, body_size: int = 132) -> bytes:
    body = bytearray(body_size)
    struct.pack_into("<IIII", body, 0, 0, body_size, stamp_sec, stamp_nsec)
    motion = list(quat) + list(gyro) + list(acc)
    struct.pack_into("<10f", body, body_size - 40, *motion)
    return bytes(body)


# ---------------------------------------------------------------------------
# Assemble packets -> LidarFrame(s)
# ---------------------------------------------------------------------------

def assemble_frames(packets, *, packets_per_frame: int = 72,
                    attach_imu: bool = True):
    """Group decoded native packets into :class:`LidarFrame` scans.

    A single 102 packet is only ~300 points (one short scan segment), so we
    accumulate ``packets_per_frame`` of them into a frame (72*300 ~= 21.6k points,
    roughly one L2 revolution). The boundary is heuristic and configurable; tune it
    to your firmware's packets-per-revolution if needed.
    """
    chunks: list[PointScanChunk] = []
    last_imu: ImuSample | None = None
    frame_id = 0

    def flush():
        nonlocal frame_id
        if not chunks:
            return None
        xyz = np.concatenate([c.xyz_approx() for c in chunks])
        inten = np.concatenate([c.intensities for c in chunks])
        tim = np.concatenate([c.times for c in chunks])
        r = np.concatenate([c.ranges_m for c in chunks])
        rmin = min(c.range_min for c in chunks)
        rmax = max(c.range_max for c in chunks)
        keep = (r >= rmin) & (r <= rmax) & (r > 0)
        f = LidarFrame.from_arrays(
            xyz[keep], intensity=inten[keep], time=tim[keep],
            ring=np.zeros(int(keep.sum()), dtype=np.uint32),
            stamp=chunks[0].stamp, frame_id=frame_id,
            imu=last_imu if attach_imu else None,
        )
        frame_id += 1
        chunks.clear()
        return f

    for pkt in packets:
        if pkt.packet_type == PT_POINT and len(pkt.body) >= POINT_BODY_SIZE:
            chunks.append(decode_point_body(pkt.body))
            if len(chunks) >= packets_per_frame:
                out = flush()
                if out is not None:
                    yield out
        elif pkt.packet_type == PT_IMU and attach_imu:
            try:
                last_imu = decode_imu_body(pkt.body)
            except struct.error:
                pass
    tail = flush()
    if tail is not None:
        yield tail
