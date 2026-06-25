"""End-to-end and unit tests for the Unitree L2 pipeline (numpy-only)."""

from __future__ import annotations

import numpy as np
import pytest

from unitree_l2_pipeline.formats import (
    POINT_DTYPE,
    LidarFrame,
    apply_transform,
    concat_frames,
    load_frame_npz,
    load_pcd,
    make_transform,
    quaternion_to_matrix,
    save_frame_npz,
    save_pcd,
    save_ply,
)
from unitree_l2_pipeline.ingest.synthetic import generate_frames
from unitree_l2_pipeline.protocol import (
    KIND_POINTCLOUD,
    decode_datagram,
    encode_pointcloud_datagrams,
)
from unitree_l2_pipeline.reconstruct.aggregate import (
    ReconstructionConfig,
    aggregate,
    statistical_outlier_removal,
    voxel_downsample,
)
from unitree_l2_pipeline.reconstruct.registration import icp_numpy, register_frames


# --------------------------------------------------------------------------- formats

def test_lidarframe_fields_roundtrip():
    xyz = np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32)
    f = LidarFrame.from_arrays(xyz, intensity=np.arange(100),
                               ring=np.arange(100) % 18)
    assert len(f) == 100
    assert f.points.dtype == POINT_DTYPE
    np.testing.assert_allclose(f.xyz, xyz, atol=1e-5)
    assert f.ring.max() < 18


def test_npz_roundtrip(tmp_path):
    frame = next(generate_frames(num_frames=1, seed=1))
    p = tmp_path / "f.npz"
    save_frame_npz(p, frame)
    back = load_frame_npz(p)
    assert len(back) == len(frame)
    np.testing.assert_allclose(back.xyz, frame.xyz, atol=1e-5)
    assert back.pose is not None
    np.testing.assert_allclose(back.pose, frame.pose, atol=1e-9)
    assert back.imu is not None


@pytest.mark.parametrize("binary", [True, False])
def test_pcd_roundtrip(tmp_path, binary):
    xyz = np.random.default_rng(2).normal(size=(50, 3)).astype(np.float32)
    inten = np.random.default_rng(3).random(50).astype(np.float32)
    p = tmp_path / "c.pcd"
    save_pcd(p, xyz, inten, binary=binary)
    bx, bi = load_pcd(p)
    np.testing.assert_allclose(bx, xyz, atol=1e-4)
    np.testing.assert_allclose(bi, inten, atol=1e-4)


def test_ply_writes(tmp_path):
    xyz = np.zeros((5, 3), np.float32)
    save_ply(tmp_path / "p.ply", xyz, colors=np.ones((5, 3)))
    assert (tmp_path / "p.ply").read_text().startswith("ply")


def test_quaternion_matrix_identity():
    R = quaternion_to_matrix([1, 0, 0, 0])
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)


def test_apply_transform():
    T = make_transform(np.eye(3), [1, 2, 3])
    out = apply_transform(T, np.zeros((1, 3)))
    np.testing.assert_allclose(out[0], [1, 2, 3])


# --------------------------------------------------------------------------- synthetic

def test_synthetic_frames_are_in_range_and_have_rings():
    frames = list(generate_frames(num_frames=3, seed=0))
    assert len(frames) == 3
    for f in frames:
        assert len(f) > 1000
        r = np.linalg.norm(f.xyz, axis=1)
        assert r.max() <= 30.0 + 0.1
        assert set(np.unique(f.ring).tolist()).issubset(set(range(18)))
        assert f.pose is not None


# --------------------------------------------------------------------------- protocol

def test_udp_encode_decode_roundtrip():
    frame = next(generate_frames(num_frames=1, seed=5))
    dgrams = encode_pointcloud_datagrams(frame)
    assert len(dgrams) >= 1
    total = []
    for dg in dgrams:
        header, payload = decode_datagram(dg)
        assert header.kind == KIND_POINTCLOUD
        assert header.frame_id == frame.frame_id
        total.append(payload)
    merged = concat_frames([LidarFrame(points=np.concatenate(total))])
    assert merged.shape[0] == len(frame)
    np.testing.assert_allclose(merged["x"], frame.points["x"], atol=1e-5)


def test_frame_assembler_capture_via_loopback():
    """Stream frames over UDP loopback and reassemble them."""
    import threading

    from unitree_l2_pipeline.ingest.udp_capture import capture_udp, stream_frames_udp

    frames = list(generate_frames(num_frames=3, seed=7))
    captured = []

    def _recv():
        for f in capture_udp(host="127.0.0.1", port=0, max_frames=3, timeout_s=3.0):
            captured.append(f)

    # bind an ephemeral port by capturing first; instead use a fixed test port
    port = 6299
    t = threading.Thread(
        target=lambda: captured.extend(
            capture_udp(host="127.0.0.1", port=port, max_frames=3, timeout_s=3.0)
        )
    )
    t.start()
    import time
    time.sleep(0.3)
    stream_frames_udp(frames, dst_ip="127.0.0.1", dst_port=port,
                      inter_packet_s=0.0005)
    t.join(timeout=5)
    # last frame only flushes when the next id arrives; expect >=2 of 3
    assert len(captured) >= 2
    assert len(captured[0]) > 0


# --------------------------------------------------------------------------- reconstruct

def test_voxel_downsample_reduces_count():
    xyz = np.random.default_rng(0).random((10000, 3))
    out = voxel_downsample(xyz, 0.1)
    assert out.shape[0] < xyz.shape[0]
    assert out.shape[1] == 3


def test_voxel_downsample_with_values():
    xyz = np.random.default_rng(0).random((1000, 3))
    vals = np.random.default_rng(1).random(1000)
    dxyz, dvals = voxel_downsample(xyz, 0.2, vals)
    assert dxyz.shape[0] == dvals.shape[0]


def test_statistical_outlier_removal_drops_flyers():
    rng = np.random.default_rng(0)
    inliers = rng.normal(0, 0.1, size=(2000, 3))
    flyers = rng.normal(0, 0.1, size=(20, 3)) + np.array([10.0, 10.0, 10.0])
    xyz = np.vstack([inliers, flyers]).astype(np.float32)
    kept = statistical_outlier_removal(xyz, k=16, std_ratio=2.0)
    # most flyers should be gone, most inliers retained
    assert kept.shape[0] < xyz.shape[0]
    assert kept.shape[0] >= 1900


def test_icp_recovers_known_translation():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(500, 3))
    T_true = make_transform(np.eye(3), [0.2, -0.1, 0.05])
    tgt = apply_transform(T_true, src)
    T_est, rmse = icp_numpy(src, tgt, max_iter=40, max_corr_dist=2.0)
    np.testing.assert_allclose(T_est[:3, 3], [0.2, -0.1, 0.05], atol=0.02)
    assert rmse < 0.05


def test_register_frames_ground_truth_passthrough():
    frames = list(generate_frames(num_frames=4, seed=0))
    poses = register_frames(frames, use_ground_truth=True)
    assert len(poses) == 4
    np.testing.assert_allclose(poses[1], frames[1].pose)


# --------------------------------------------------------------------------- full pipeline

def test_full_pipeline_synthetic(tmp_path):
    from unitree_l2_pipeline.pipeline import PipelineConfig, run_pipeline

    cfg = PipelineConfig(
        source="synthetic",
        num_frames=6,
        out_dir=str(tmp_path / "out"),
        recon=ReconstructionConfig(voxel_size=0.05, remove_outliers=True),
    )
    result = run_pipeline(cfg, log=lambda *a, **k: None)
    assert result.usd_path.exists()
    assert len(result.reconstruction) > 1000
    # reconstruction should roughly span the 10x8x3 room
    lo, hi = result.reconstruction.bounds
    assert hi[0] - lo[0] > 8.0
    assert hi[2] - lo[2] > 2.0
    text = result.usd_path.read_text()
    assert 'def Points "UnitreeL2Cloud"' in text
    assert "upAxis = \"Z\"" in text


def test_full_pipeline_dataset_roundtrip(tmp_path):
    """synth -> disk -> reader -> reconstruct -> USD."""
    from unitree_l2_pipeline.formats import save_frame_npz
    from unitree_l2_pipeline.pipeline import PipelineConfig, run_pipeline

    ds = tmp_path / "ds"
    ds.mkdir()
    for f in generate_frames(num_frames=4, seed=2):
        save_frame_npz(ds / f"frame_{f.frame_id:05d}.npz", f)

    cfg = PipelineConfig(source=str(ds), num_frames=4,
                         out_dir=str(tmp_path / "out2"),
                         recon=ReconstructionConfig(voxel_size=0.05))
    result = run_pipeline(cfg, log=lambda *a, **k: None)
    assert result.usd_path.exists()
    assert len(result.reconstruction) > 1000
