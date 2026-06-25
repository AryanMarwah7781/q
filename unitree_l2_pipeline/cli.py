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
    if args.native:
        from .ingest.pcap import capture_native_udp as capture
        cap = capture(host=args.host, port=args.port, max_frames=args.frames,
                      timeout_s=args.timeout)
    else:
        from .ingest.udp_capture import capture_udp
        cap = capture_udp(host=args.host, port=args.port,
                          max_frames=args.frames, timeout_s=args.timeout)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for frame in cap:
        save_frame_npz(out / f"frame_{frame.frame_id:05d}.npz", frame)
        n += 1
        print(f"  captured frame {frame.frame_id} ({len(frame)} pts)")
    print(f"captured {n} frames -> {out}")


def _cmd_raw_info(args):
    from .ingest.pcap import _iter_pcap_records, _udp_payload
    from .raw_protocol import NATIVE_MAGIC, iter_frames_from_stream

    path = Path(args.file)
    data = path.read_bytes()
    counts: dict[int, int] = {}
    if path.suffix in (".pcap", ".pcapng"):
        streams = (_udp_payload(lt, p) for lt, p in _iter_pcap_records(data))
        streams = (s for s in streams if s and NATIVE_MAGIC in s)
        blob = b"".join(streams)
    else:
        blob = data
    crc_ok = crc_total = 0
    for pkt in iter_frames_from_stream(blob, check_crc=True):
        counts[pkt.packet_type] = counts.get(pkt.packet_type, 0) + 1
        if pkt.crc_ok is not None:
            crc_total += 1
            crc_ok += int(pkt.crc_ok)
    names = {102: "3D points (102)", 103: "2D points (103)",
             104: "IMU (104)", 105: "version (105)"}
    print(f"file: {path}  ({len(data)/1e6:.1f} MB)")
    print("native packets:")
    for t, c in sorted(counts.items()):
        print(f"  {names.get(t, f'type {t}'):20s} {c}")
    if crc_total:
        print(f"crc32 matched (zlib): {crc_ok}/{crc_total} "
              f"(informational; vendor CRC coverage is undocumented)")


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
        max_frames=args.max_frames_cap,
        seed=args.seed,
        use_ground_truth_poses=not args.icp,
        pose_source=args.poses,
        pose_topic=args.poses_topic,
        extrinsic=args.extrinsic,
        recon=recon,
        out_dir=args.out,
        usd_name=args.usd_name,
        point_width=args.point_width,
        save_intermediate=not args.no_intermediate,
    )


def _cmd_bag_info(args):
    from .ingest.rosbag import bag_info

    info = bag_info(args.bag)
    print(f"bag: {info['path']}")
    print(f"size: {info['size_bytes'] / 1e6:.1f} MB   compression: {info['compression']}")
    print("topics:")
    for topic, meta in info["topics"].items():
        print(f"  {topic:24s} {meta['type']:28s} {meta['messages']} msgs")


def _cmd_run(args):
    run_pipeline(_build_cfg(args))


def _cmd_export(args):
    args.source = args.dataset
    run_pipeline(_build_cfg(args))


def _add_recon_export_args(p):
    p.add_argument("--frames", type=int, default=24,
                   help="synthetic/capture frame count")
    p.add_argument("--max-frames-cap", type=int, default=None,
                   help="cap frames read from a dataset/ROS bag (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--voxel", type=float, default=0.03,
                   help="reconstruction voxel size (m); 0 disables")
    p.add_argument("--max-range", type=float, default=0.0,
                   help="drop points beyond this range (m); 0 keeps all")
    p.add_argument("--no-outlier-removal", action="store_true")
    p.add_argument("--icp", action="store_true",
                   help="estimate poses with ICP instead of stored poses")
    p.add_argument("--poses", default=None,
                   help="external trajectory for drift-free poses: a TUM file "
                        "(timestamp tx ty tz qx qy qz qw) or a .bag with a "
                        "cuVSLAM/Isaac ROS odometry topic")
    p.add_argument("--poses-topic", default="/visual_slam/tracking/odometry",
                   help="odometry topic name when --poses is a .bag")
    p.add_argument("--extrinsic", default=None,
                   help="LiDAR->base extrinsic: 'identity', 7 values "
                        "'x y z qx qy qz qw', or 16 row-major 4x4 values")
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
    pc.add_argument("--native", action="store_true",
                    help="parse the device's native 0x55AA050A framing instead of "
                         "the open replay payload")
    pc.set_defaults(func=_cmd_capture)

    pbi = sub.add_parser("bag-info", help="summarize a Unitree L2 ROS1 .bag")
    pbi.add_argument("bag")
    pbi.set_defaults(func=_cmd_bag_info)

    pri = sub.add_parser("raw-info",
                         help="summarize a native L2 capture (.pcap / raw dump)")
    pri.add_argument("file")
    pri.set_defaults(func=_cmd_raw_info)

    pr = sub.add_parser("replay", help="replay a dataset over UDP")
    pr.add_argument("dataset")
    pr.add_argument("--host", default="127.0.0.1")
    pr.add_argument("--port", type=int, default=6201)
    pr.add_argument("--realtime", action="store_true")
    pr.set_defaults(func=_cmd_replay)

    prun = sub.add_parser("run", help="full pipeline from a source")
    prun.add_argument("--source", default="synthetic",
                      help="'synthetic', 'udp', a dataset dir, a .bag, or a "
                           "native .pcap / .bin raw capture")
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
