# Unitree 4D LiDAR L2 → Point-Cloud Reconstruction → Isaac Sim

A complete, runnable pipeline that ingests **Unitree 4D LiDAR L2** data, reconstructs
a dense point cloud from a sequence of scans, and exports it to **NVIDIA Isaac Sim**
as an OpenUSD stage.

```
   ┌──────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
   │  INGEST  │ ──▶ │  RECONSTRUCT │ ──▶ │    EXPORT     │ ──▶ │  ISAAC SIM  │
   │ udp /    │     │ register +   │     │ point cloud → │     │ open .usda  │
   │ synthetic│     │ aggregate +  │     │ UsdGeom.Points│     │ (Z-up, m)   │
   │ / record │     │ filter       │     │  (.usda/.usd) │     │             │
   └──────────┘     └──────────────┘     └───────────────┘     └─────────────┘
```

The whole pipeline **runs on `numpy` alone** — no ROS, no Open3D, no `pxr` required.
Open3D and the USD bindings are used automatically *if present* for extra speed, but
are entirely optional.

---

## Why this exists

The official Unitree SDK for the L2 — [`unitreerobotics/unilidar_sdk2`](https://github.com/unitreerobotics/unilidar_sdk2)
— gives you C++/ROS interfaces to pull live `(x, y, z, intensity, time, ring)` points
off the sensor, but it ships **no recorded sample data** and stops at the single-scan
level. There is no published, end-to-end path from "L2 scans" to "a reconstructed scene
loaded in Isaac Sim".

Unitree *does* publish real L2 recordings — as ROS1 bags used by
[`point_lio_unilidar`](https://github.com/unitreerobotics/point_lio_unilidar):

* [L2 Indoor Point Cloud Data.bag](https://oss-global-cdn.unitree.com/static/L2%20Indoor%20Point%20Cloud%20Data.bag) (~520 MB)
* [L2 Park Point Cloud Data.bag](https://oss-global-cdn.unitree.com/static/L2%20Park%20Point%20Cloud%20Data.bag)

This repo wires those in directly (no ROS install needed) and fills the rest of the gap:

1. A faithful re-implementation of the L2 **data model and network protocol**
   (18 rings; UDP ports `6101`/`6201`; IPs `192.168.1.62`/`192.168.1.2`).
2. A **dependency-free ROS1 `.bag` reader** for Unitree's real L2 recordings
   (`/unilidar/cloud` + `/unilidar/imu`), plus a **synthetic L2 generator** so you can
   run the whole pipeline with no hardware *and* no download. Small samples of **both**
   (real L2 frames extracted from the indoor bag, and a synthetic room) are committed
   under [`data/samples/`](data/samples).
3. Multi-frame **reconstruction** (pose registration via stored poses / IMU / ICP,
   then aggregation, voxel downsampling and outlier removal).
4. **Export to Isaac Sim** as `UsdGeom.Points` in a ready-to-open `.usda` stage, plus a
   standalone Isaac Sim loader script.

---

## Quick start

```bash
pip install -e .            # installs numpy + the `unitree-l2` CLI
# or: pip install -r requirements.txt

# 1) Run the whole thing on built-in synthetic data and write a USD:
unitree-l2 run --source synthetic --frames 24 --out output
#   -> output/reconstruction.usda   (open this in Isaac Sim)
#   -> output/reconstruction.pcd / .ply

# 2) Or reconstruct a committed sample dataset (real L2 frames, or synthetic):
unitree-l2 export data/samples/l2_indoor_real --out output   # real Unitree L2 data
unitree-l2 export data/samples/synthetic_room --out output

# 3) Or use a full real Unitree L2 ROS bag (downloads ~520 MB):
scripts/fetch_l2_bag.sh indoor                  # -> data/captures/L2_Indoor.bag
unitree-l2 bag-info data/captures/L2_Indoor.bag
unitree-l2 run --source data/captures/L2_Indoor.bag --max-frames-cap 60 --out output

# 4) Then load the result into Isaac Sim (inside Isaac's python env):
./python.sh unitree_l2_pipeline/export/isaac_loader.py \
    --usd output/reconstruction.usda --as-instancer
```

A pre-built example output is committed at
[`data/samples/example_output/reconstruction.usda`](data/samples/example_output) so you
can open it in Isaac Sim immediately.

---

## Using it with a real Unitree L2

The pipeline produces and consumes `LidarFrame` objects, so there are several ways to
feed it real data:

**A. A recorded ROS bag (easiest).** Use Unitree's published L2 bags (or your own
recording of `/unilidar/cloud` + `/unilidar/imu`). No ROS install required:

```bash
scripts/fetch_l2_bag.sh indoor                  # or: park
unitree-l2 run --source data/captures/L2_Indoor.bag --max-frames-cap 60 --out output
```

**B. A raw capture — `.pcap`, raw dump, or live native UDP.** When you only have raw bytes
off the wire (no SDK, no ROS), the pipeline parses the device's native `0x55AA050A`
framing directly. Pure stdlib + numpy, so it runs on Windows:

```bash
unitree-l2 raw-info capture.pcap                 # what's inside the capture
unitree-l2 run --source capture.pcap --out output
unitree-l2 capture --native --port 6201 --out data/captures/live   # live
```

> Faithful for ranges/intensities/IMU/timestamps; **Cartesian XYZ is an approximate,
> calibration-free reconstruction** (the exact polar→XYZ calibration lives in the closed
> SDK). For metric-accurate clouds, prefer a `.bag` (option **A**). See
> [`docs/raw_capture.md`](docs/raw_capture.md).

**C. Bridge the vendor SDK live.** For exact points without recording, build
`unilidar_sdk2` and feed its point-cloud callback into a `FrameAssembler` (see
[`unitree_l2_pipeline/ingest/udp_capture.py`](unitree_l2_pipeline/ingest/udp_capture.py)).
Everything downstream is identical.

> **Poses for real data.** Real bags carry IMU but no ground-truth poses, so the pipeline
> falls back to frame-to-frame ICP, which drifts over long sequences. For drift-free maps,
> supply an external trajectory with `--poses`:
>
> ```bash
> # cuVSLAM / Isaac ROS odometry recorded in the bag (GPU visual-inertial SLAM):
> unitree-l2 run --source run.bag --poses run.bag \
>     --poses-topic /visual_slam/tracking/odometry \
>     --extrinsic "0.10 0 0.05 0 0 0 1" --out output
>
> # ...or a TUM trajectory file from PyCuVSLAM / point_lio_unilidar / FAST-LIO:
> unitree-l2 run --source run.bag --poses traj.tum --extrinsic identity --out output
> ```
>
> See [`docs/cuvslam.md`](docs/cuvslam.md) for the full cuVSLAM workflow (topics, the
> LiDAR↔base extrinsic, and the `odom` vs `map` frames).

---

## CLI reference

| Command | Purpose |
|---|---|
| `unitree-l2 synth --out DIR --frames N` | Generate a synthetic L2 dataset to disk |
| `unitree-l2 bag-info BAG` | Summarize a Unitree L2 ROS1 `.bag` (topics, counts) |
| `unitree-l2 raw-info FILE` | Summarize a native capture (`.pcap` / raw dump) |
| `unitree-l2 capture [--native] --out DIR` | Capture a live L2 UDP stream (`0.0.0.0:6201`) |
| `unitree-l2 replay DATASET` | Replay a recorded dataset over UDP (test capture) |
| `unitree-l2 run --source synthetic\|udp\|DIR\|BAG\|PCAP` | Full pipeline → USD |
| `unitree-l2 export DATASET\|BAG` | Reconstruct a recorded dataset/bag → USD |

Key reconstruction/export flags (`run`/`export`):

```
--voxel 0.03            voxel size (m) for downsampling; 0 disables
--max-range 0.0         drop points beyond this range (m); 0 keeps all
--no-outlier-removal    skip statistical outlier removal
--icp                   estimate poses with ICP instead of stored poses
--poses PATH            external trajectory (TUM file or .bag with odometry) for
                        drift-free poses, e.g. cuVSLAM / Isaac ROS, point_lio
--poses-topic TOPIC     odometry topic when --poses is a .bag
                        (default /visual_slam/tracking/odometry)
--extrinsic SPEC        LiDAR->base extrinsic: 'identity', 'x y z qx qy qz qw',
                        or 16 row-major 4x4 values
--point-width 0.02      rendered point size in USD
--usd-name NAME.usda    output stage filename
```

---

## How reconstruction works

* **Poses.** Each scan only sees part of the scene; to reconstruct we place every scan in
  a common world frame. Best results come from an external trajectory via `--poses`:
  **cuVSLAM / Isaac ROS** odometry (GPU visual-inertial SLAM — see
  [`docs/cuvslam.md`](docs/cuvslam.md)),
  [`point_lio_unilidar`](https://github.com/unitreerobotics/point_lio_unilidar), or FAST-LIO,
  as a TUM file or a `nav_msgs/Odometry` bag topic; poses are interpolated (SLERP) to each
  scan's timestamp and composed with the LiDAR↔base `--extrinsic`. If no external poses are
  given and frames carry their own (the synthetic generator and `.npz` records do), those
  are used. Otherwise `--icp` runs frame-to-frame ICP (numpy, or point-to-plane via Open3D)
  and chains the result — simplest, but drifts without loop closure.
* **Aggregate.** All scans are transformed to world, merged, voxel-downsampled (uniform
  density + dedup), and cleaned with statistical outlier removal (a spatial-hash-grid
  implementation that scales to hundreds of thousands of points without a KD-tree).
* **Export.** The cleaned cloud becomes a `UsdGeom.Points` prim with per-point
  `displayColor` derived from intensity, plus a `BasisCurves` showing the sensor
  trajectory. The stage is authored **Z-up, metres** to match the L2 convention and Isaac
  Sim defaults.

See [`docs/data_format.md`](docs/data_format.md) and
[`docs/isaac_sim.md`](docs/isaac_sim.md) for details.

---

## Project layout

```
unitree_l2_pipeline/
├── formats.py            L2 point struct, LidarFrame, PCD/PLY/NPZ I/O, transforms
├── protocol.py           open UDP replay payload (de)serialization
├── raw_protocol.py       native 0x55AA050A device framing + body decoders
├── ingest/
│   ├── synthetic.py      synthetic L2 sample generator (ray-cast room scene)
│   ├── rosbag.py         dependency-free ROS1 .bag reader (real Unitree L2 bags)
│   ├── pcap.py           native capture ingest (.pcap / raw dump / live UDP)
│   ├── udp_capture.py    open-format UDP capture + frame reassembly + replay
│   └── reader.py         load recorded sources (.npz / .pcd / .bag / .pcap / .bin)
├── reconstruct/
│   ├── registration.py   ICP (numpy + optional Open3D), pose chaining
│   ├── pose_source.py    external trajectories (cuVSLAM/Isaac ROS odom, TUM) + extrinsic
│   └── aggregate.py      transform/merge/voxel-downsample/outlier-removal
├── export/
│   ├── usd_export.py     point cloud → .usda (no deps) / .usd (pxr)
│   └── isaac_loader.py   standalone Isaac Sim loader script
├── pipeline.py           orchestrates ingest → reconstruct → export
└── cli.py                `unitree-l2` command line
data/samples/             real L2 frames (from the indoor bag) + synthetic room + example USD
scripts/fetch_l2_bag.sh   download Unitree's full real L2 ROS bags (indoor/park)
tests/                    pytest suite (numpy-only, runs in CI)
```

## Testing

```bash
pip install pytest
pytest -q
```

## References

* Unitree L2 SDK — <https://github.com/unitreerobotics/unilidar_sdk2>
* Unitree L1 SDK — <https://github.com/unitreerobotics/unilidar_sdk>
* Point-LIO for Unitree LiDAR — <https://github.com/unitreerobotics/point_lio_unilidar>
* cuVSLAM / Isaac ROS Visual SLAM — <https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam>
* PyCuVSLAM (standalone Python cuVSLAM) — <https://github.com/nvidia-isaac/PyCuVSLAM>
* Unitree 4D LiDAR L2 User Manual — <https://oss-global-cdn.unitree.com/static/Unitree%204D%20LiDAR%20L2%20User%20Manual.pdf>
* NVIDIA Isaac Sim — <https://developer.nvidia.com/isaac/sim>
* OpenUSD — <https://openusd.org>

## License

BSD-3-Clause (matching the upstream Unitree SDK license).
