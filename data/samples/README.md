# Sample Unitree L2 data

This directory provides self-contained samples so the pipeline is runnable out of the box
— both **real** Unitree L2 data and a **synthetic** scene.

## `l2_indoor_real/`

24 **real** L2 scans extracted from Unitree's official
[L2 Indoor Point Cloud Data.bag](https://oss-global-cdn.unitree.com/static/L2%20Indoor%20Point%20Cloud%20Data.bag)
(used by [`point_lio_unilidar`](https://github.com/unitreerobotics/point_lio_unilidar)).
Each `.npz` holds the real `(x, y, z, intensity, time, ring)` points plus the IMU sample
for that scan. Reconstruct them with:

```bash
unitree-l2 export data/samples/l2_indoor_real --out output
```

> Real bags have no ground-truth poses, so reconstruction falls back to ICP (which drifts
> over long runs). For accurate maps, get per-scan poses from `point_lio_unilidar`. The
> full 520 MB bag (2807 scans) is **not** committed — fetch it with
> `scripts/fetch_l2_bag.sh indoor`.

## `synthetic_room/`

8 synthetic L2 scans (`frame_00000.npz` … `frame_00007.npz`) of a 10 × 8 × 3 m room with a
few obstacles, captured along a looping sensor trajectory. Each frame is a lossless,
compressed record containing the `(x, y, z, intensity, time, ring)` points plus the
ground-truth sensor pose and an IMU sample — exactly the data model the real L2 produces.

Regenerate (or make a denser one) with:

```bash
unitree-l2 synth --out data/samples/synthetic_room --frames 8 --seed 0
```

## `example_output/`

A pre-built reconstruction of `synthetic_room/`:

* `reconstruction.usda` — open this directly in Isaac Sim.
* `reconstruction.pcd` — the merged cloud (PCL-compatible).

Regenerate with:

```bash
unitree-l2 export data/samples/synthetic_room --voxel 0.05 --out data/samples/example_output
```
