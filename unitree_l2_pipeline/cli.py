"""Command line interface for the Unitree L2 -> Isaac Sim pipeline.

Subcommands::

    unitree-l2 synth     generate a synthetic L2 dataset to disk
    unitree-l2 capture   capture a live L2 UDP stream to disk
    unitree-l2 replay    replay a recorded dataset over UDP (for testing capture)
    unitree-l2 run       full pipeline: ingest -> reconstruct -> export USD
    unitree-l2 export    reconstruct an existing dataset and export USD

Run ``unitree-l2 <cmd> -h`` for per-command options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .formats import save_frame_npz
from .pipeline import PipelineConfig, run_pipeline
from .reconstruct.aggregate import ReconstructionConfig


def _cmd_synth(args):
    from .ingest.synthetic import generate_frames

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for frame in generate_frames(num_frames=args.frames, seed=args.seed):
        save_frame_npz(out / f"frame_{frame.frame_id:05d}.npz", frame)
        n += 1
    print(f"wrote {n} synthetic L2 frames -> {out}")


def _cmd_capture(args):
    from .ingest.udp_capture import capture_udp

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for frame in capture_udp(host=args.host, port=args.port,
                             max_frames=args.frames, timeout_s=args.timeout):
        save_frame_npz(out / f"frame_{frame.frame_id:05d}.npz", frame)
        n += 1
        print(f"  captured frame {frame.frame_id} ({len(frame)} pts)")
    print(f"captured {n} frames -> {out}")


def _cmd_replay(args):
    from .ingest.reader import load_dataset
    from .ingest.udp_capture import stream_frames_udp

    frames = load_dataset(args.dataset)
    sent = stream_frames_udp(frames, dst_ip=args.host, dst_port=args.port,
                             realtime=args.realtime)
    print(f"replayed {len(frames)} frames as {sent} datagrams "
          f"-> {args.host}:{args.port}")


def _build_cfg(args) -> PipelineConfig:
    recon = ReconstructionConfig(
        voxel_size=args.voxel,
        remove_outliers=not args.no_outlier_removal,
        max_range=args.max_range,
    )
    return PipelineConfig(
        source=args.source,
        num_frames=args.frames,
        seed=args.seed,
        use_ground_truth_poses=not args.icp,
        recon=recon,
        out_dir=args.out,
        usd_name=args.usd_name,
        point_width=args.point_width,
        save_intermediate=not args.no_intermediate,
    )


def _cmd_run(args):
    run_pipeline(_build_cfg(args))


def _cmd_export(args):
    args.source = args.dataset
    run_pipeline(_build_cfg(args))


def _add_recon_export_args(p):
    p.add_argument("--frames", type=int, default=24,
                   help="synthetic/capture frame count")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--voxel", type=float, default=0.03,
                   help="reconstruction voxel size (m); 0 disables")
    p.add_argument("--max-range", type=float, default=0.0,
                   help="drop points beyond this range (m); 0 keeps all")
    p.add_argument("--no-outlier-removal", action="store_true")
    p.add_argument("--icp", action="store_true",
                   help="estimate poses with ICP instead of stored poses")
    p.add_argument("--out", default="output", help="output directory")
    p.add_argument("--usd-name", default="reconstruction.usda")
    p.add_argument("--point-width", type=float, default=0.02)
    p.add_argument("--no-intermediate", action="store_true",
                   help="skip writing merged .pcd/.ply")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unitree-l2",
        description="Unitree 4D LiDAR L2 -> point-cloud reconstruction -> Isaac Sim USD",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("synth", help="generate a synthetic L2 dataset")
    ps.add_argument("--out", default="data/samples/synthetic_room")
    ps.add_argument("--frames", type=int, default=24)
    ps.add_argument("--seed", type=int, default=0)
    ps.set_defaults(func=_cmd_synth)

    pc = sub.add_parser("capture", help="capture a live L2 UDP stream")
    pc.add_argument("--out", default="data/captures/live")
    pc.add_argument("--host", default="0.0.0.0")
    pc.add_argument("--port", type=int, default=6201)
    pc.add_argument("--frames", type=int, default=None)
    pc.add_argument("--timeout", type=float, default=5.0)
    pc.set_defaults(func=_cmd_capture)

    pr = sub.add_parser("replay", help="replay a dataset over UDP")
    pr.add_argument("dataset")
    pr.add_argument("--host", default="127.0.0.1")
    pr.add_argument("--port", type=int, default=6201)
    pr.add_argument("--realtime", action="store_true")
    pr.set_defaults(func=_cmd_replay)

    prun = sub.add_parser("run", help="full pipeline from a source")
    prun.add_argument("--source", default="synthetic",
                      help="'synthetic', 'udp', or a dataset directory")
    _add_recon_export_args(prun)
    prun.set_defaults(func=_cmd_run)

    pex = sub.add_parser("export", help="reconstruct a dataset dir and export USD")
    pex.add_argument("dataset", help="dataset directory of recorded frames")
    _add_recon_export_args(pex)
    pex.set_defaults(func=_cmd_export)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
