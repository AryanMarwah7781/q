"""Tests for the dependency-free ROS1 bag reader.

We synthesize a *minimal but valid* ROS1 bag v2.0 (uncompressed) containing one
``/unilidar/cloud`` ``PointCloud2`` and one ``/unilidar/imu`` ``Imu`` message,
using the exact L2 field layout (x,y,z f32; intensity f32; ring u16; time f32;
point_step=32), then read it back through the public API. This mirrors the real
Unitree L2 bags without needing the ~520 MB download.
"""

from __future__ import annotations

import struct

import numpy as np

from unitree_l2_pipeline.ingest.rosbag import bag_info, iter_bag_frames


def _field(name: bytes, value: bytes) -> bytes:
    f = name + b"=" + value
    return struct.pack("<I", len(f)) + f


def _record(header_fields: bytes, data: bytes) -> bytes:
    return (struct.pack("<I", len(header_fields)) + header_fields
            + struct.pack("<I", len(data)) + data)


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<I", len(b)) + b


def _std_header(stamp_s=1.0) -> bytes:
    secs = int(stamp_s)
    nsecs = int((stamp_s - secs) * 1e9)
    return struct.pack("<I", 0) + struct.pack("<II", secs, nsecs) + _string("unilidar")


def _pointcloud2_msg(points) -> bytes:
    n = len(points)
    out = bytearray()
    out += _std_header(1.0)
    out += struct.pack("<II", 1, n)  # height, width
    fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7),
              ("intensity", 16, 7), ("ring", 20, 4), ("time", 24, 7)]
    out += struct.pack("<I", len(fields))
    for name, offset, dt in fields:
        out += _string(name) + struct.pack("<IBI", offset, dt, 1)
    point_step = 32
    out += struct.pack("<B", 0)               # is_bigendian
    out += struct.pack("<II", point_step, point_step * n)
    raw = bytearray(point_step * n)
    for i, (x, y, z, inten, ring, t) in enumerate(points):
        base = i * point_step
        struct.pack_into("<fff", raw, base, x, y, z)
        struct.pack_into("<f", raw, base + 16, inten)
        struct.pack_into("<H", raw, base + 20, ring)
        struct.pack_into("<f", raw, base + 24, t)
    out += struct.pack("<I", len(raw)) + bytes(raw)
    out += struct.pack("<B", 1)               # is_dense
    return bytes(out)


def _imu_msg() -> bytes:
    out = bytearray()
    out += _std_header(1.0)
    out += struct.pack("<dddd", 0.0, 0.0, 0.0, 1.0)   # orientation x,y,z,w
    out += struct.pack("<9d", *([0.0] * 9))
    out += struct.pack("<ddd", 0.1, 0.2, 0.3)         # angular velocity
    out += struct.pack("<9d", *([0.0] * 9))
    out += struct.pack("<ddd", 0.0, 0.0, 9.81)        # linear acceleration
    out += struct.pack("<9d", *([0.0] * 9))
    return bytes(out)


def _build_bag() -> bytes:
    bag = bytearray(b"#ROSBAG V2.0\n")

    # bag header record (op=3) -- minimal
    bag += _record(_field(b"op", b"\x03"), b"\x00" * 8)

    # build a chunk containing two connections + two messages
    chunk = bytearray()
    # connection 0 -> cloud
    conn0_hdr = _field(b"op", b"\x07") + _field(b"conn", struct.pack("<I", 0)) \
        + _field(b"topic", b"/unilidar/cloud")
    conn0_data = _field(b"topic", b"/unilidar/cloud") + _field(b"type", b"sensor_msgs/PointCloud2")
    chunk += _record(conn0_hdr, conn0_data)
    # connection 1 -> imu
    conn1_hdr = _field(b"op", b"\x07") + _field(b"conn", struct.pack("<I", 1)) \
        + _field(b"topic", b"/unilidar/imu")
    conn1_data = _field(b"topic", b"/unilidar/imu") + _field(b"type", b"sensor_msgs/Imu")
    chunk += _record(conn1_hdr, conn1_data)
    # imu message (op=2) before cloud so it gets attached
    imu_hdr = _field(b"op", b"\x02") + _field(b"conn", struct.pack("<I", 1)) \
        + _field(b"time", struct.pack("<II", 1, 0))
    chunk += _record(imu_hdr, _imu_msg())
    # cloud message
    pts = [(1.0, 2.0, 3.0, 100.0, 0, 0.0),
           (1.1, 2.1, 3.1, 110.0, 3, 0.01),
           (1.2, 2.2, 3.2, 120.0, 7, 0.02)]
    cloud_hdr = _field(b"op", b"\x02") + _field(b"conn", struct.pack("<I", 0)) \
        + _field(b"time", struct.pack("<II", 1, 0))
    chunk += _record(cloud_hdr, _pointcloud2_msg(pts))

    chunk_hdr = _field(b"op", b"\x05") + _field(b"compression", b"none") \
        + _field(b"size", struct.pack("<I", len(chunk)))
    bag += _record(chunk_hdr, bytes(chunk))
    return bytes(bag)


def test_bag_info(tmp_path):
    p = tmp_path / "mini.bag"
    p.write_bytes(_build_bag())
    info = bag_info(p)
    assert info["topics"]["/unilidar/cloud"]["type"] == "sensor_msgs/PointCloud2"
    assert info["topics"]["/unilidar/cloud"]["messages"] == 1
    assert info["topics"]["/unilidar/imu"]["type"] == "sensor_msgs/Imu"


def test_iter_bag_frames_decodes_l2_layout(tmp_path):
    p = tmp_path / "mini.bag"
    p.write_bytes(_build_bag())
    frames = list(iter_bag_frames(p))
    assert len(frames) == 1
    f = frames[0]
    assert len(f) == 3
    np.testing.assert_allclose(f.xyz[0], [1.0, 2.0, 3.0], atol=1e-5)
    np.testing.assert_allclose(f.intensity, [100.0, 110.0, 120.0], atol=1e-4)
    assert f.ring.tolist() == [0, 3, 7]
    np.testing.assert_allclose(f.points["time"], [0.0, 0.01, 0.02], atol=1e-5)
    # IMU attached from the preceding /unilidar/imu message
    assert f.imu is not None
    np.testing.assert_allclose(f.imu.angular_velocity, [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(f.imu.linear_acceleration, [0.0, 0.0, 9.81], atol=1e-6)


def test_bag_drives_full_pipeline(tmp_path):
    from unitree_l2_pipeline.pipeline import PipelineConfig, run_pipeline
    from unitree_l2_pipeline.reconstruct.aggregate import ReconstructionConfig

    # a bag with several cloud messages so reconstruction has something to merge
    p = tmp_path / "multi.bag"
    p.write_bytes(_build_multi_frame_bag(6))
    cfg = PipelineConfig(source=str(p), out_dir=str(tmp_path / "out"),
                         recon=ReconstructionConfig(voxel_size=0.0,
                                                    remove_outliers=False))
    result = run_pipeline(cfg, log=lambda *a, **k: None)
    assert result.usd_path.exists()
    assert len(result.reconstruction) > 0


def _build_multi_frame_bag(num_clouds: int) -> bytes:
    bag = bytearray(b"#ROSBAG V2.0\n")
    bag += _record(_field(b"op", b"\x03"), b"\x00" * 8)
    chunk = bytearray()
    conn_hdr = _field(b"op", b"\x07") + _field(b"conn", struct.pack("<I", 0)) \
        + _field(b"topic", b"/unilidar/cloud")
    conn_data = _field(b"topic", b"/unilidar/cloud") + _field(b"type", b"sensor_msgs/PointCloud2")
    chunk += _record(conn_hdr, conn_data)
    rng = np.random.default_rng(0)
    for _ in range(num_clouds):
        base = rng.normal(size=3)
        pts = [(float(base[0] + dx), float(base[1] + dy), float(base[2] + dz),
                200.0, j % 18, 0.001 * j)
               for j, (dx, dy, dz) in enumerate(rng.normal(scale=0.2, size=(50, 3)))]
        hdr = _field(b"op", b"\x02") + _field(b"conn", struct.pack("<I", 0)) \
            + _field(b"time", struct.pack("<II", 1, 0))
        chunk += _record(hdr, _pointcloud2_msg(pts))
    chunk_hdr = _field(b"op", b"\x05") + _field(b"compression", b"none") \
        + _field(b"size", struct.pack("<I", len(chunk)))
    bag += _record(chunk_hdr, bytes(chunk))
    return bytes(bag)
