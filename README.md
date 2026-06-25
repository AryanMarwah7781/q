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

This repo fills that gap:

1. A faithful re-implementation of the L2 **data model and network protocol**
   (18 rings; UDP ports `6101`/`6201`; IPs `192.168.1.62`/`192.168.1.2`).
2. A **synthetic L2 data sample generator** so you can run the entire pipeline today
   with no hardware — and a small recorded sample committed under
   [`data/samples/`](data/samples).
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

# 2) Or reconstruct the committed sample dataset:
unitree-l2 export data/samples/synthetic_room --out output

# 3) Then load it into Isaac Sim (inside Isaac's python env):
./python.sh unitree_l2_pipeline/export/isaac_loader.py \
    --usd output/reconstruction.usda --as-instancer
```

A pre-built example output is committed at
[`data/samples/example_output/reconstruction.usda`](data/samples/example_output) so you
can open it in Isaac Sim immediately.

---

## Using it with a real Unitree L2

The pipeline produces and consumes `LidarFrame` objects, so there are two ways to feed
it real hardware:

**A. Bridge the vendor SDK (recommended).** Build `unilidar_sdk2`, and in its point-cloud
callback hand the points to a `FrameAssembler` (see
[`unitree_l2_pipeline/ingest/udp_capture.py`](unitree_l2_pipeline/ingest/udp_capture.py)).
Everything downstream is identical.

**B. UDP capture.** The L2 streams to host `192.168.1.2:6201` by default. Configure your
NIC to that subnet and capture:

```bash
unitree-l2 capture --out data/captures/live --frames 100
unitree-l2 export data/captures/live --out output
```

> The capture path consumes the open, documented UDP payload format in
> [`protocol.py`](unitree_l2_pipeline/protocol.py). The vendor's on-wire framing is
> proprietary and decoded inside the closed SDK; once decoded to points the path is the
> same. Use option **A** to bridge it, or `unitree-l2 replay <dataset>` to exercise the
> capture path end-to-end over loopback.

---

## CLI reference

| Command | Purpose |
|---|---|
| `unitree-l2 synth --out DIR --frames N` | Generate a synthetic L2 dataset to disk |
| `unitree-l2 capture --out DIR` | Capture a live L2 UDP stream (`0.0.0.0:6201`) |
| `unitree-l2 replay DATASET` | Replay a recorded dataset over UDP (test capture) |
| `unitree-l2 run --source synthetic\|udp\|DIR` | Full pipeline → USD |
| `unitree-l2 export DATASET` | Reconstruct a recorded dataset → USD |

Key reconstruction/export flags (`run`/`export`):

```
--voxel 0.03            voxel size (m) for downsampling; 0 disables
--max-range 0.0         drop points beyond this range (m); 0 keeps all
--no-outlier-removal    skip statistical outlier removal
--icp                   estimate poses with ICP instead of stored/odometry poses
--point-width 0.02      rendered point size in USD
--usd-name NAME.usda    output stage filename
```

---

## How reconstruction works

* **Poses.** Each scan only sees part of the scene; to reconstruct we place every scan in
  a common world frame. Best results come from poses supplied by odometry/SLAM (e.g.
  [`point_lio_unilidar`](https://github.com/unitreerobotics/point_lio_unilidar) or
  FAST-LIO). If frames carry poses (the synthetic generator and `.npz` records do), they
  are used directly. Otherwise `--icp` runs frame-to-frame point-to-point ICP (numpy) or
  point-to-plane ICP (Open3D, if installed) and chains the result into a trajectory.
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
├── protocol.py           L2 UDP/serial constants + datagram (de)serialization
├── ingest/
│   ├── synthetic.py      synthetic L2 sample generator (ray-cast room scene)
│   ├── udp_capture.py    live UDP capture + frame reassembly + replay
│   └── reader.py         load recorded datasets (.npz / .pcd)
├── reconstruct/
│   ├── registration.py   ICP (numpy + optional Open3D), pose chaining
│   └── aggregate.py      transform/merge/voxel-downsample/outlier-removal
├── export/
│   ├── usd_export.py     point cloud → .usda (no deps) / .usd (pxr)
│   └── isaac_loader.py   standalone Isaac Sim loader script
├── pipeline.py           orchestrates ingest → reconstruct → export
└── cli.py                `unitree-l2` command line
data/samples/             committed L2 sample dataset + example USD output
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
* Unitree 4D LiDAR L2 User Manual — <https://oss-global-cdn.unitree.com/static/Unitree%204D%20LiDAR%20L2%20User%20Manual.pdf>
* NVIDIA Isaac Sim — <https://developer.nvidia.com/isaac/sim>
* OpenUSD — <https://openusd.org>

## License

BSD-3-Clause (matching the upstream Unitree SDK license).
