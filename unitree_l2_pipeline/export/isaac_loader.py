"""Standalone Isaac Sim loader for the reconstructed L2 point cloud.

Run this *inside* Isaac Sim's python environment, e.g.::

    ./python.sh /path/to/isaac_loader.py --usd /path/to/reconstruction.usda

or from the Isaac Sim *Script Editor* by setting ``USD_PATH`` below. It opens (or
references) the exported USD stage and, for large clouds, can additionally
instance the points as a ``UsdGeom.PointInstancer`` of small spheres so they are
visible and pickable in the viewport.

The exported ``.usda`` is already a valid Isaac Sim stage (Z-up, metres) — you
can also simply *File > Open* or *Add > Reference* it in the GUI. This script is
the programmatic path and a place to hook follow-on work (collision meshing,
semantic labels, RTX-lidar replay, domain randomization, etc.).
"""

from __future__ import annotations

import argparse
import sys

# Default used when run from the Script Editor (edit to taste).
USD_PATH = "/tmp/reconstruction.usda"


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Load L2 reconstruction into Isaac Sim")
    p.add_argument("--usd", default=USD_PATH, help="path to exported .usd/.usda")
    p.add_argument("--headless", action="store_true", help="run without GUI")
    p.add_argument("--as-instancer", action="store_true",
                   help="also create a PointInstancer of spheres for visibility")
    p.add_argument("--sphere-radius", type=float, default=0.02)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv or sys.argv[1:])

    # Booting the kit application must happen before importing omni/pxr modules.
    try:
        from isaacsim import SimulationApp  # Isaac Sim 4.x
    except Exception:  # pragma: no cover - older layout
        from omni.isaac.kit import SimulationApp  # type: ignore

    sim_app = SimulationApp({"headless": args.headless})

    import numpy as np
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom

    # Open the exported stage directly so the Points prim + trajectory show up.
    ctx = omni.usd.get_context()
    ctx.open_stage(args.usd)
    stage = ctx.get_stage()
    print(f"[unitree-l2] opened stage: {args.usd}")

    if args.as_instancer:
        _add_point_instancer(stage, args.sphere_radius, Gf, Sdf, Usd, UsdGeom, np)

    # Spin a few frames so the viewport populates, then idle.
    for _ in range(60):
        sim_app.update()

    if not args.headless:
        print("[unitree-l2] stage loaded; close the window to exit.")
        while sim_app.is_running():
            sim_app.update()
    sim_app.close()


def _add_point_instancer(stage, radius, Gf, Sdf, Usd, UsdGeom, np):
    """Instance a small sphere at every point of the first Points prim."""
    points_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Points):
            points_prim = UsdGeom.Points(prim)
            break
    if points_prim is None:
        print("[unitree-l2] no Points prim found to instance")
        return
    positions = np.asarray(points_prim.GetPointsAttr().Get())
    print(f"[unitree-l2] instancing {len(positions)} points as spheres")

    inst = UsdGeom.PointInstancer.Define(stage, "/World/CloudInstancer")
    proto = UsdGeom.Sphere.Define(stage, "/World/CloudInstancer/protoSphere")
    proto.GetRadiusAttr().Set(float(radius))
    inst.CreatePrototypesRel().SetTargets([proto.GetPath()])
    inst.CreatePositionsAttr([Gf.Vec3f(*p) for p in positions])
    inst.CreateProtoIndicesAttr([0] * len(positions))


if __name__ == "__main__":
    main()
