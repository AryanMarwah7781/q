# Loading the reconstruction into Isaac Sim

The export stage writes an OpenUSD file (`reconstruction.usda`) containing:

* `def Points "UnitreeL2Cloud"` — a `UsdGeom.Points` prim with the reconstructed cloud,
  per-point `widths`, and `displayColor` derived from intensity.
* `def BasisCurves "SensorTrajectory"` — the sensor path, for context.

The stage metadata is **`upAxis = "Z"`, `metersPerUnit = 1`**, matching the L2 frame and
Isaac Sim's default stage, so it drops in at correct scale and orientation.

## Option 1 — open it in the GUI

1. Launch Isaac Sim.
2. **File ▸ Open** `output/reconstruction.usda` (or **Create ▸ Add ▸ Reference** to drop it
   into an existing stage).
3. The point cloud appears immediately. Frame the viewport (press `F`).

## Option 2 — the standalone loader script

Run inside Isaac Sim's Python environment (it boots a `SimulationApp` for you):

```bash
# from your Isaac Sim install directory
./python.sh /path/to/repo/unitree_l2_pipeline/export/isaac_loader.py \
    --usd /path/to/output/reconstruction.usda \
    --as-instancer            # also instance a small sphere per point (visible & pickable)
    # --headless              # no GUI (e.g. for batch USD post-processing)
```

`--as-instancer` builds a `UsdGeom.PointInstancer` of spheres on top of the `Points` prim.
`Points` prims render fast but can be hard to select; the instancer makes every point a
real, pickable, collidable prim — handy for annotation or physics.

## Option 3 — Script Editor

Paste the contents of `isaac_loader.py` into **Window ▸ Script Editor**, set `USD_PATH` at
the top of the file, and run.

## Generating a binary `.usd`

Isaac Sim ships its own `pxr` (OpenUSD) build. Inside that environment you can author a
binary `.usd` directly:

```python
from unitree_l2_pipeline.export.usd_export import export_usd_binary
export_usd_binary(reconstruction, "output/reconstruction.usd")
```

Outside Isaac Sim, the dependency-free `.usda` writer is used automatically and Isaac Sim
opens it natively.

## Where to go next

The loader is a deliberate hook point for follow-on work in Isaac Sim:

* **Meshing** — run Poisson/ball-pivoting (Open3D) on the cloud, export a `UsdGeom.Mesh`,
  and assign collision for robot navigation tests.
* **RTX LiDAR replay** — drive an Isaac RTX-LiDAR sensor through the reconstructed scene to
  compare simulated vs. real L2 returns (sim-to-real).
* **Semantics / domain randomization** — label the instanced points or segment planes
  (floor/walls) for synthetic-data generation.
