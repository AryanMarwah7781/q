"""Live UDP ingest for the Unitree L2.

Two entry points:

* :func:`capture_udp` — bind the host receive port (default 6201), reassemble
  per-frame datagrams, and yield :class:`LidarFrame` objects. This consumes the
  open payload format defined in :mod:`unitree_l2_pipeline.protocol`, which is
  what :func:`stream_frames_udp` emits. Point the same reassembly logic at the
  vendor SDK output when you have hardware.
* :func:`stream_frames_udp` — replay a list of frames over UDP (useful to test
  the capture path, or to feed another machine on the network).

The vendor SDK's proprietary framing is decoded inside the closed binary; once
decoded into points the path here is identical. To bridge the real SDK, call its
C++/Python callback and hand the points to :meth:`FrameAssembler.add_points`.
"""

from __future__ import annotations

import socket
import time
from typing import Iterator

import numpy as np

from ..formats import POINT_DTYPE, ImuSample, LidarFrame
from ..protocol import (
    HOST_RECV_PORT,
    KIND_IMU,
    KIND_POINTCLOUD,
    LIDAR_IP,
    LIDAR_PORT,
    decode_datagram,
    encode_imu_datagram,
    encode_pointcloud_datagrams,
)


class FrameAssembler:
    """Reassemble point datagrams that share a ``frame_id`` into a frame."""

    def __init__(self) -> None:
        self._fid: int | None = None
        self._stamp = 0.0
        self._chunks: list[np.ndarray] = []
        self._imu: ImuSample | None = None

    def add_points(self, frame_id: int, stamp: float, pts: np.ndarray):
        """Add a point chunk; returns a completed frame when ``frame_id`` rolls."""
        completed = None
        if self._fid is None:
            self._fid, self._stamp = frame_id, stamp
        elif frame_id != self._fid:
            completed = self._flush()
            self._fid, self._stamp = frame_id, stamp
        self._chunks.append(pts)
        return completed

    def add_imu(self, payload: np.ndarray):
        self._imu = ImuSample(
            stamp=self._stamp,
            quaternion=payload[0:4].astype(np.float64),
            angular_velocity=payload[4:7].astype(np.float64),
            linear_acceleration=payload[7:10].astype(np.float64),
        )

    def _flush(self) -> LidarFrame | None:
        if not self._chunks:
            return None
        pts = np.concatenate(self._chunks) if len(self._chunks) > 1 else self._chunks[0]
        frame = LidarFrame(points=pts.copy(), stamp=self._stamp,
                           frame_id=int(self._fid or 0), imu=self._imu)
        self._chunks = []
        self._imu = None
        return frame

    def finish(self) -> LidarFrame | None:
        return self._flush()


def capture_udp(
    host: str = "0.0.0.0",
    port: int = HOST_RECV_PORT,
    *,
    max_frames: int | None = None,
    timeout_s: float | None = 5.0,
) -> Iterator[LidarFrame]:
    """Bind ``host:port`` and yield reassembled frames from the L2 stream."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    sock.bind((host, port))
    if timeout_s is not None:
        sock.settimeout(timeout_s)
    asm = FrameAssembler()
    produced = 0
    try:
        while True:
            try:
                buf, _ = sock.recvfrom(65535)
            except socket.timeout:
                break
            try:
                header, payload = decode_datagram(buf)
            except ValueError:
                continue
            if header.kind == KIND_POINTCLOUD:
                done = asm.add_points(header.frame_id, header.stamp, payload)
                if done is not None:
                    yield done
                    produced += 1
                    if max_frames is not None and produced >= max_frames:
                        return
            elif header.kind == KIND_IMU:
                asm.add_imu(payload)
        tail = asm.finish()
        if tail is not None and len(tail) > 0:
            yield tail
    finally:
        sock.close()


def stream_frames_udp(
    frames,
    dst_ip: str = "127.0.0.1",
    dst_port: int = HOST_RECV_PORT,
    *,
    realtime: bool = False,
    inter_packet_s: float = 0.0,
) -> int:
    """Emit frames as UDP datagrams to ``dst_ip:dst_port``. Returns datagram count."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0
    try:
        for frame in frames:
            if frame.imu is not None:
                sock.sendto(
                    encode_imu_datagram(
                        frame.stamp, frame.frame_id,
                        frame.imu.quaternion, frame.imu.angular_velocity,
                        frame.imu.linear_acceleration,
                    ),
                    (dst_ip, dst_port),
                )
                sent += 1
            for dg in encode_pointcloud_datagrams(frame):
                sock.sendto(dg, (dst_ip, dst_port))
                sent += 1
                if inter_packet_s:
                    time.sleep(inter_packet_s)
            if realtime:
                time.sleep(0.1)
    finally:
        sock.close()
    return sent
