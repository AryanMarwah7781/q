"""Orchestrate the full Unitree L2 -> Isaac Sim pipeline.

    ingest  ->  reconstruct  ->  export

Each stage is independently usable; this module wires them together with a
single config so the CLI (and tests) can drive the whole thing in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .formats import LidarFrame, save_frame_npz, save_pcd, save_ply
from .reconstruct.aggregate import Reconstruction, ReconstructionConfig, aggregate
from .reconstruct.registration import register_frames


@dataclass
class PipelineConfig:
    # ingest
    source: str = "synthetic"          # synthetic | <dataset dir or .bag> | udp
    num_frames: int = 24               # for synthetic / udp
    max_frames: int | None = None      # cap frames read from a dataset / ROS bag
    seed: int = 0
    # reconstruct
    use_ground_truth_poses: bool = True
    recon: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    register_voxel: float = 0.1
    # export
    out_dir: str = "output"
    usd_name: str = "reconstruction.usda"
    point_width: float = 0.02
    save_intermediate: bool = True     # write merged .pcd/.ply alongside USD


@dataclass
class PipelineResult:
    frames: list[LidarFrame]
    poses: list[np.ndarray]
    reconstruction: Reconstruction
    usd_path: Path
    out_dir: Path


def _load_frames(cfg: PipelineConfig) -> list[LidarFrame]:
    if cfg.source == "synthetic":
        from .ingest.synthetic import generate_frames

        return list(generate_frames(num_frames=cfg.num_frames, seed=cfg.seed))
    if cfg.source == "udp":
        from .ingest.udp_capture import capture_udp

        return list(capture_udp(max_frames=cfg.num_frames))
    # otherwise treat as a dataset directory / file / ROS bag
    from .ingest.reader import load_dataset

    return load_dataset(cfg.source, max_frames=cfg.max_frames)


def run_pipeline(cfg: PipelineConfig | None = None,
                 log=print) -> PipelineResult:
    cfg = cfg or PipelineConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. ingest -----------------------------------------------------------
    log(f"[ingest] source={cfg.source}")
    frames = _load_frames(cfg)
    total_pts = sum(len(f) for f in frames)
    log(f"[ingest] {len(frames)} frames, {total_pts} raw points")
    if not frames:
        raise RuntimeError("no frames ingested")

    # 2. reconstruct ------------------------------------------------------
    use_gt = cfg.use_ground_truth_poses and all(f.pose is not None for f in frames)
    if cfg.use_ground_truth_poses and not use_gt:
        log("[reconstruct] no stored poses (e.g. real bag) -> falling back to ICP")
    log("[reconstruct] estimating poses"
        + (" (ground truth)" if use_gt else " (ICP)"))
    poses = register_frames(
        frames,
        use_ground_truth=use_gt,
        voxel=cfg.register_voxel,
    )
    log("[reconstruct] aggregating frames")
    recon = aggregate(frames, poses, cfg.recon)
    lo, hi = recon.bounds if len(recon) else (np.zeros(3), np.zeros(3))
    log(f"[reconstruct] {len(recon)} points; "
        f"bounds {np.round(lo,2)} -> {np.round(hi,2)} m")

    # 3. export -----------------------------------------------------------
    from .export.usd_export import export_usd

    usd_path = out_dir / cfg.usd_name
    export_usd(recon, usd_path, point_width=cfg.point_width)
    log(f"[export] wrote USD -> {usd_path}")

    if cfg.save_intermediate and len(recon):
        save_pcd(out_dir / "reconstruction.pcd", recon.xyz, recon.intensity)
        from .export.usd_export import intensity_to_color

        save_ply(out_dir / "reconstruction.ply", recon.xyz,
                 intensity_to_color(recon.intensity))
        log(f"[export] wrote merged .pcd/.ply -> {out_dir}")

    return PipelineResult(frames=frames, poses=poses, reconstruction=recon,
                          usd_path=usd_path, out_dir=out_dir)
