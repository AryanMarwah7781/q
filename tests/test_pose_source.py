"""Tests for external pose sources (cuVSLAM / Isaac ROS odometry + TUM)."""

from __future__ import annotations

import struct

import numpy as np

from unitree_l2_pipeline.formats import make_transform, quaternion_to_matrix
from unitree_l2_pipeline.reconstruct.pose_source import (
    Trajectory,
    load_cuvslam_bag,
    load_tum,
    parse_extrinsic,
    poses_for_frames,
    save_tum,
    slerp,
)

# reuse the in-memory bag builders
from tests.test_rosbag import _field, _record, _std_header, _string


def test_slerp_endpoints_and_midpoint():
    q0 = np.array([1.0, 0, 0, 0])           # identity
    q1 = np.array([0.0, 0, 0, 1.0])         # 180 deg about z
    np.testing.assert_allclose(slerp(q0, q1, 0.0), q0, atol=1e-9)
    np.testing.assert_allclose(np.abs(slerp(q0, q1, 1.0)), np.abs(q1), atol=1e-9)
    # halfway = 90 deg about z -> (cos45, 0,0, sin45)
    mid = slerp(q0, q1, 0.5)
    np.testing.assert_allclose(mid, [0.70710678, 0, 0, 0.70710678], atol=1e-6)


def test_trajectory_interpolation_clamps_and_lerps():
    stamps = np.array([0.0, 1.0])
    pos = np.array([[0, 0, 0], [2.0, 0, 0]])
    quat = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
    tr = Trajectory(stamps, pos, quat)
    # clamp before start / after end
    np.testing.assert_allclose(tr.pose_at(-1.0)[:3, 3], [0, 0, 0])
    np.testing.assert_allclose(tr.pose_at(5.0)[:3, 3], [2, 0, 0])
    # midpoint translation lerps
    np.testing.assert_allclose(tr.pose_at(0.5)[:3, 3], [1, 0, 0], atol=1e-9)


def test_tum_roundtrip(tmp_path):
    stamps = np.array([0.0, 0.5, 1.0])
    pos = np.random.default_rng(0).normal(size=(3, 3))
    quat = np.tile([1.0, 0, 0, 0], (3, 1))
    tr = Trajectory(stamps, pos, quat)
    p = tmp_path / "traj.tum"
    save_tum(p, tr)
    back = load_tum(p)
    np.testing.assert_allclose(back.positions, tr.positions, atol=1e-5)
    np.testing.assert_allclose(back.stamps, tr.stamps, atol=1e-6)


def test_parse_extrinsic_variants():
    np.testing.assert_allclose(parse_extrinsic(None), np.eye(4))
    np.testing.assert_allclose(parse_extrinsic("identity"), np.eye(4))
    np.testing.assert_allclose(parse_extrinsic("0 0 0 0 0 0 1"), np.eye(4))
    T = parse_extrinsic("1 2 3 0 0 0 1")
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
    flat = list(np.eye(4).reshape(-1))
    np.testing.assert_allclose(parse_extrinsic(flat), np.eye(4))


def test_poses_for_frames_applies_extrinsic():
    from unitree_l2_pipeline.formats import LidarFrame

    tr = Trajectory(np.array([0.0, 1.0]),
                    np.array([[0, 0, 0], [0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [1, 0, 0, 0]]))
    frame = LidarFrame.from_arrays(np.zeros((1, 3)), stamp=0.5)
    extr = parse_extrinsic("1 0 0 0 0 0 1")  # lidar 1m ahead of base in x
    poses = poses_for_frames([frame], tr, extr)
    # base at origin, lidar offset by +x -> world lidar pose translation = [1,0,0]
    np.testing.assert_allclose(poses[0][:3, 3], [1, 0, 0], atol=1e-9)


# --------------------------------------------------------------------------- bag odometry

def _odom_msg(stamp_s, pos, quat_xyzw) -> bytes:
    out = bytearray()
    out += _std_header(stamp_s)              # header
    out += _string("base_link")              # child_frame_id
    out += struct.pack("<ddd", *pos)         # position
    out += struct.pack("<dddd", *quat_xyzw)  # orientation x,y,z,w
    out += struct.pack("<36d", *([0.0] * 36))  # pose covariance
    out += struct.pack("<ddd", 0, 0, 0)      # twist linear
    out += struct.pack("<ddd", 0, 0, 0)      # twist angular
    out += struct.pack("<36d", *([0.0] * 36))  # twist covariance
    return bytes(out)


def _build_odom_bag(poses) -> bytes:
    bag = bytearray(b"#ROSBAG V2.0\n")
    bag += _record(_field(b"op", b"\x03"), b"\x00" * 8)
    chunk = bytearray()
    conn_hdr = _field(b"op", b"\x07") + _field(b"conn", struct.pack("<I", 0)) \
        + _field(b"topic", b"/visual_slam/tracking/odometry")
    conn_data = _field(b"topic", b"/visual_slam/tracking/odometry") \
        + _field(b"type", b"nav_msgs/Odometry")
    chunk += _record(conn_hdr, conn_data)
    for stamp_s, pos, quat_xyzw in poses:
        secs = int(stamp_s)
        nsecs = int((stamp_s - secs) * 1e9)
        hdr = _field(b"op", b"\x02") + _field(b"conn", struct.pack("<I", 0)) \
            + _field(b"time", struct.pack("<II", secs, nsecs))
        chunk += _record(hdr, _odom_msg(stamp_s, pos, quat_xyzw))
    chunk_hdr = _field(b"op", b"\x05") + _field(b"compression", b"none") \
        + _field(b"size", struct.pack("<I", len(chunk)))
    bag += _record(chunk_hdr, bytes(chunk))
    return bytes(bag)


def test_load_cuvslam_bag(tmp_path):
    poses = [(1.0, (0.0, 0.0, 0.0), (0, 0, 0, 1)),
             (2.0, (1.0, 0.0, 0.0), (0, 0, 0, 1)),
             (3.0, (2.0, 0.0, 0.0), (0, 0, 0, 1))]
    p = tmp_path / "odom.bag"
    p.write_bytes(_build_odom_bag(poses))
    tr = load_cuvslam_bag(p)
    assert len(tr) == 3
    np.testing.assert_allclose(tr.pose_at(2.0)[:3, 3], [1, 0, 0], atol=1e-6)
    # quaternion stored as w,x,y,z; identity here
    np.testing.assert_allclose(tr.quaternions[0], [1, 0, 0, 0], atol=1e-6)


def test_full_pipeline_with_external_poses_is_driftfree(tmp_path):
    """Synthetic frames + their true trajectory as TUM -> recon matches ground truth."""
    from unitree_l2_pipeline.ingest.synthetic import _matrix_to_quat, generate_frames
    from unitree_l2_pipeline.pipeline import PipelineConfig, run_pipeline
    from unitree_l2_pipeline.reconstruct.aggregate import ReconstructionConfig

    frames = list(generate_frames(num_frames=8, seed=0))
    tr = Trajectory(
        np.array([f.stamp for f in frames]),
        np.array([f.pose[:3, 3] for f in frames]),
        np.array([_matrix_to_quat(f.pose[:3, :3]) for f in frames]),
    )
    tum = tmp_path / "traj.tum"
    save_tum(tum, tr)

    cfg = PipelineConfig(source="synthetic", num_frames=8, seed=0,
                         pose_source=str(tum), extrinsic="identity",
                         out_dir=str(tmp_path / "out"),
                         recon=ReconstructionConfig(voxel_size=0.05))
    result = run_pipeline(cfg, log=lambda *a, **k: None)
    lo, hi = result.reconstruction.bounds
    # drift-free: spans the true 10x8x3 room, not a drifted blob
    assert hi[0] - lo[0] < 10.5 and hi[0] - lo[0] > 9.5
    assert hi[1] - lo[1] < 8.5 and hi[1] - lo[1] > 7.5
    assert hi[2] - lo[2] < 3.5 and hi[2] - lo[2] > 2.5
