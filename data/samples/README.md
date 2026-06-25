# Sample Unitree L2 data

The official Unitree L2 SDK ships no recorded sample data, so this directory provides a
self-contained sample so the pipeline is runnable out of the box.

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
