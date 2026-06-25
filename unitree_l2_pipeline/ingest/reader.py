"""Read recorded LiDAR frames back from disk.

Supports the canonical ``.npz`` frame container (lossless, with pose + IMU) and
plain ``.pcd`` clouds. A *dataset* is simply a directory of per-frame files,
loaded in sorted filename order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..formats import LidarFrame, load_frame_npz, load_pcd


def iter_frames(path: str | Path, *, max_frames: int | None = None) -> Iterator[LidarFrame]:
    """Yield frames from a dataset directory, a single frame file, or a ROS bag."""
    path = Path(path)
    if path.is_file():
        if path.suffix == ".bag":
            from .rosbag import iter_bag_frames

            yield from iter_bag_frames(path, max_frames=max_frames)
            return
        if path.suffix in (".pcap", ".pcapng"):
            from .pcap import iter_frames_from_pcap

            it = iter_frames_from_pcap(path)
            for n, fr in enumerate(it):
                if max_frames is not None and n >= max_frames:
                    return
                yield fr
            return
        if path.suffix in (".bin", ".raw"):
            from .pcap import iter_frames_from_raw

            for n, fr in enumerate(iter_frames_from_raw(path)):
                if max_frames is not None and n >= max_frames:
                    return
                yield fr
            return
        yield _load_one(path)
        return
    files = sorted(
        [p for p in path.iterdir() if p.suffix in (".npz", ".pcd")]
    )
    if not files:
        raise FileNotFoundError(f"no .npz/.pcd frames found in {path}")
    for i, p in enumerate(files):
        frame = _load_one(p)
        if frame.frame_id == 0 and i != 0:
            frame.frame_id = i
        yield frame


def load_dataset(path: str | Path, *, max_frames: int | None = None) -> list[LidarFrame]:
    """Eagerly load every frame in a dataset directory (or ROS bag)."""
    return list(iter_frames(path, max_frames=max_frames))


def _load_one(p: Path) -> LidarFrame:
    if p.suffix == ".npz":
        return load_frame_npz(p)
    if p.suffix == ".pcd":
        xyz, inten = load_pcd(p)
        return LidarFrame.from_arrays(xyz, intensity=inten)
    raise ValueError(f"unsupported frame file: {p}")
