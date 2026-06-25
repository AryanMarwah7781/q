"""Ingest a *raw* Unitree L2 capture: ``.pcap``, a raw frame dump, or a live socket.

Pure stdlib + numpy (runs on Windows). Three entry points, all yielding
:class:`~unitree_l2_pipeline.formats.LidarFrame` via the native-protocol parser in
:mod:`unitree_l2_pipeline.raw_protocol`:

* :func:`iter_frames_from_pcap` — classic ``.pcap`` (Wireshark/tcpdump), strips
  Ethernet/IPv4/UDP to recover datagrams, then demuxes native frames.
* :func:`iter_frames_from_raw` — a raw dump of concatenated native frames / UDP
  payloads (``.bin``/``.raw``).
* :func:`capture_native_udp` — bind a UDP port and parse native frames live.

See ``raw_protocol`` for the accuracy caveats (ranges/intensities/IMU decode
faithfully; Cartesian XYZ is an approximate, calibration-free reconstruction).
"""

from __future__ import annotations

import socket
import struct
from pathlib import Path
from typing import Iterator

from ..formats import LidarFrame
from ..raw_protocol import (
    NATIVE_MAGIC,
    assemble_frames,
    decode_frame,
    iter_frames_from_stream,
)

# pcap link-layer types we understand
_LINKTYPE_ETHERNET = 1
_LINKTYPE_RAW_IP = 101


def _iter_pcap_records(data: bytes) -> Iterator[tuple[int, bytes]]:
    """Yield ``(linktype, packet_bytes)`` from a classic pcap file."""
    if len(data) < 24:
        raise ValueError("file too short to be a pcap")
    magic = data[:4]
    # A little-endian writer packs 0xA1B2C3D4 -> file bytes D4 C3 B2 A1, so those
    # bytes mean "read little-endian". (0xA1B23C4D is the nanosecond variant.)
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        endian = ">"
    else:
        raise ValueError("not a classic .pcap (bad magic; pcapng is unsupported)")
    (network,) = struct.unpack_from(endian + "I", data, 20)
    i = 24
    rec = struct.Struct(endian + "IIII")
    n = len(data)
    while i + 16 <= n:
        _ts, _us, incl, _orig = rec.unpack_from(data, i)
        i += 16
        if i + incl > n:
            break
        yield network, data[i:i + incl]
        i += incl


def _udp_payload(linktype: int, pkt: bytes) -> bytes | None:
    """Strip link/IP/UDP layers, return the UDP payload (or None if not IPv4/UDP)."""
    off = 0
    if linktype == _LINKTYPE_ETHERNET:
        if len(pkt) < 14:
            return None
        ethertype = pkt[12] << 8 | pkt[13]
        off = 14
        if ethertype == 0x8100:        # 802.1Q VLAN tag
            ethertype = pkt[16] << 8 | pkt[17]
            off = 18
        if ethertype != 0x0800:        # not IPv4
            return None
    elif linktype == _LINKTYPE_RAW_IP:
        off = 0
    else:
        return None
    if len(pkt) < off + 20:
        return None
    ihl = (pkt[off] & 0x0F) * 4
    proto = pkt[off + 9]
    if proto != 17:                    # not UDP
        return None
    udp = off + ihl
    if len(pkt) < udp + 8:
        return None
    return pkt[udp + 8:]


def iter_frames_from_pcap(path: str | Path, *, packets_per_frame: int = 72,
                          check_crc: bool = False) -> Iterator[LidarFrame]:
    """Yield LidarFrames from a classic ``.pcap`` capture of the L2 UDP stream."""
    data = Path(path).read_bytes()

    def packets():
        for linktype, pkt in _iter_pcap_records(data):
            payload = _udp_payload(linktype, pkt)
            if not payload or NATIVE_MAGIC not in payload:
                continue
            # one datagram usually carries exactly one native frame, but demux to
            # be safe in case several are coalesced.
            yield from iter_frames_from_stream(payload, check_crc=check_crc)

    yield from assemble_frames(packets(), packets_per_frame=packets_per_frame)


def iter_frames_from_raw(path: str | Path, *, packets_per_frame: int = 72,
                         check_crc: bool = False) -> Iterator[LidarFrame]:
    """Yield LidarFrames from a raw dump of concatenated native frames."""
    data = Path(path).read_bytes()
    yield from assemble_frames(
        iter_frames_from_stream(data, check_crc=check_crc),
        packets_per_frame=packets_per_frame,
    )


def capture_native_udp(host: str = "0.0.0.0", port: int = 6201, *,
                       max_frames: int | None = None, timeout_s: float | None = 5.0,
                       packets_per_frame: int = 72) -> Iterator[LidarFrame]:
    """Bind a UDP port and parse the L2's native frames live (one frame/datagram)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    sock.bind((host, port))
    if timeout_s is not None:
        sock.settimeout(timeout_s)

    def packets():
        while True:
            try:
                buf, _ = sock.recvfrom(65535)
            except socket.timeout:
                return
            if NATIVE_MAGIC in buf:
                yield from iter_frames_from_stream(buf)

    produced = 0
    try:
        for frame in assemble_frames(packets(), packets_per_frame=packets_per_frame):
            yield frame
            produced += 1
            if max_frames is not None and produced >= max_frames:
                return
    finally:
        sock.close()
