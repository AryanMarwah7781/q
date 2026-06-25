"""Tests for the native 0x55AA050A protocol parser and the pcap/raw ingest.

The framing and documented body fields are validated by encode->decode roundtrip
(we control the bytes). The Cartesian XYZ is an explicit approximation and is only
checked for self-consistency, not metric accuracy.
"""

from __future__ import annotations

import struct

import numpy as np

from unitree_l2_pipeline.raw_protocol import (
    NATIVE_MAGIC,
    PT_IMU,
    PT_POINT,
    assemble_frames,
    decode_frame,
    decode_imu_body,
    decode_point_body,
    encode_frame,
    encode_imu_body,
    encode_point_body,
    iter_frames_from_stream,
)


def _make_point_body(n=300, seq=0, sec=1, nsec=0):
    rng = np.random.default_rng(seq)
    ranges_mm = rng.integers(100, 30000, size=n).astype(np.uint16)
    intens = rng.integers(0, 255, size=n).astype(np.uint8)
    return encode_point_body({
        "seq": seq, "sec": sec, "nsec": nsec,
        "com_h_start": 0.1, "com_h_step": 0.001,
        "scan_period": 0.1, "range_min": 0.05, "range_max": 30.0,
        "angle_min": -0.2, "angle_increment": 0.0005, "time_increment": 1e-5,
        "ranges_mm": ranges_mm, "intensities": intens,
    }), ranges_mm, intens


def test_frame_roundtrip():
    body = b"hello-unitree-body" * 4
    frame = encode_frame(PT_POINT, body)
    assert frame[:4] == NATIVE_MAGIC
    pkt = decode_frame(frame, check_crc=True)
    assert pkt.packet_type == PT_POINT
    assert pkt.body == body
    assert pkt.crc_ok is True


def test_stream_demux_multiple_frames_with_noise():
    body1, _, _ = _make_point_body(seq=1)
    body2, _, _ = _make_point_body(seq=2)
    imu = encode_imu_body(1, 0, [1, 0, 0, 0], [0.1, 0.2, 0.3], [0, 0, 9.81])
    stream = (b"\x00\x11garbage"
              + encode_frame(PT_POINT, body1)
              + encode_frame(PT_IMU, imu)
              + b"\xde\xad"               # stray bytes between frames
              + encode_frame(PT_POINT, body2))
    pkts = list(iter_frames_from_stream(stream, check_crc=True))
    types = [p.packet_type for p in pkts]
    assert types == [PT_POINT, PT_IMU, PT_POINT]
    assert all(p.crc_ok for p in pkts)


def test_point_body_decode_fields():
    body, ranges_mm, intens = _make_point_body(n=300, seq=5, sec=7, nsec=500_000_000)
    chunk = decode_point_body(body)
    assert chunk.seq == 5
    assert abs(chunk.stamp - 7.5) < 1e-6
    assert chunk.point_num == 300
    np.testing.assert_allclose(chunk.ranges_m, ranges_mm.astype(np.float32) / 1000.0,
                               atol=1e-6)
    np.testing.assert_allclose(chunk.intensities, intens.astype(np.float32))
    # azimuth/elevation follow the documented linear model
    np.testing.assert_allclose(chunk.azimuth[1] - chunk.azimuth[0], 0.001, atol=1e-7)
    np.testing.assert_allclose(chunk.elevation[1] - chunk.elevation[0], 0.0005, atol=1e-7)
    # xyz norm equals range (spherical consistency)
    xyz = chunk.xyz_approx()
    np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), chunk.ranges_m, atol=1e-3)


def test_imu_body_roundtrip():
    body = encode_imu_body(3, 250_000_000, [0.0, 1.0, 0.0, 0.0],
                           [0.5, -0.5, 0.25], [0.1, 0.2, 9.8])
    imu = decode_imu_body(body)
    assert abs(imu.stamp - 3.25) < 1e-6
    np.testing.assert_allclose(imu.quaternion, [0.0, 1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(imu.angular_velocity, [0.5, -0.5, 0.25], atol=1e-6)
    np.testing.assert_allclose(imu.linear_acceleration, [0.1, 0.2, 9.8], atol=1e-6)


def test_assemble_frames_groups_packets_and_attaches_imu():
    pkts = []
    imu = encode_imu_body(1, 0, [1, 0, 0, 0], [0, 0, 0], [0, 0, 9.81])
    pkts.append(decode_frame(encode_frame(PT_IMU, imu)))
    for s in range(6):
        body, _, _ = _make_point_body(seq=s)
        pkts.append(decode_frame(encode_frame(PT_POINT, body)))
    frames = list(assemble_frames(pkts, packets_per_frame=3))
    assert len(frames) == 2                 # 6 point packets / 3
    assert frames[0].imu is not None
    assert len(frames[0]) > 0


# --------------------------------------------------------------------------- pcap

def _wrap_eth_ipv4_udp(payload: bytes) -> bytes:
    eth = b"\x00" * 12 + b"\x08\x00"                       # dst,src, ethertype IPv4
    udp = struct.pack(">HHHH", 6101, 6201, 8 + len(payload), 0) + payload
    ihl_ver = 0x45
    total = 20 + len(udp)
    ip = struct.pack(">BBHHHBBH4s4s", ihl_ver, 0, total, 0, 0, 64, 17, 0,
                     b"\xc0\xa8\x01\x3e", b"\xc0\xa8\x01\x02")  # 192.168.1.62 -> .2
    return eth + ip + udp


def _build_pcap(payloads) -> bytes:
    out = bytearray()
    out += struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)  # ethernet
    for p in payloads:
        pkt = _wrap_eth_ipv4_udp(p)
        out += struct.pack("<IIII", 0, 0, len(pkt), len(pkt)) + pkt
    return bytes(out)


def test_pcap_ingest_end_to_end(tmp_path):
    from unitree_l2_pipeline.ingest.pcap import iter_frames_from_pcap

    payloads = []
    for s in range(4):
        body, _, _ = _make_point_body(seq=s)
        payloads.append(encode_frame(PT_POINT, body))
    imu = encode_imu_body(1, 0, [1, 0, 0, 0], [0, 0, 0], [0, 0, 9.81])
    payloads.insert(0, encode_frame(PT_IMU, imu))

    p = tmp_path / "cap.pcap"
    p.write_bytes(_build_pcap(payloads))
    frames = list(iter_frames_from_pcap(p, packets_per_frame=2))
    assert len(frames) == 2
    assert all(len(f) > 0 for f in frames)
    assert frames[0].imu is not None


def test_pcap_drives_full_pipeline(tmp_path):
    from unitree_l2_pipeline.ingest.pcap import iter_frames_from_pcap
    from unitree_l2_pipeline.pipeline import PipelineConfig, run_pipeline
    from unitree_l2_pipeline.reconstruct.aggregate import ReconstructionConfig

    payloads = [encode_frame(PT_POINT, _make_point_body(seq=s)[0]) for s in range(8)]
    p = tmp_path / "cap.pcap"
    p.write_bytes(_build_pcap(payloads))
    # sanity: reader recognizes the .pcap source
    frames = list(iter_frames_from_pcap(p, packets_per_frame=4))
    assert len(frames) == 2

    cfg = PipelineConfig(source=str(p), out_dir=str(tmp_path / "out"),
                         recon=ReconstructionConfig(voxel_size=0.0,
                                                    remove_outliers=False))
    result = run_pipeline(cfg, log=lambda *a, **k: None)
    assert result.usd_path.exists()
    assert len(result.reconstruction) > 0
