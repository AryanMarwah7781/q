"""Export a reconstructed point cloud to USD for Isaac Sim.

Isaac Sim is built on OpenUSD, so the portable hand-off is a ``.usd``/``.usda``
stage containing a ``UsdGeom.Points`` prim. We write USD ASCII (``.usda``)
*directly* — no ``pxr`` dependency required — so the export works in any
environment. If the ``pxr`` USD bindings happen to be available (e.g. inside
Isaac Sim's Python) a binary ``.usd`` writer is used instead.

The stage is authored Z-up, metres, matching the L2 coordinate convention and
Isaac Sim's default stage settings, so the cloud drops in at correct scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..reconstruct.aggregate import Reconstruction

try:
    from pxr import Gf, Usd, UsdGeom, Vt  # type: ignore

    _HAVE_PXR = True
except Exception:
    _HAVE_PXR = False


# ---------------------------------------------------------------------------
# intensity -> RGB colormap (turbo-ish, no matplotlib dependency)
# ---------------------------------------------------------------------------

def intensity_to_color(intensity: np.ndarray) -> np.ndarray:
    """Map intensity (any range) to ``(N,3)`` float RGB in [0,1]."""
    i = np.asarray(intensity, dtype=np.float64)
    if i.size == 0:
        return np.zeros((0, 3), np.float32)
    lo, hi = np.percentile(i, 2), np.percentile(i, 98)
    t = np.clip((i - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    # simple blue->green->red ramp
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _fmt_vec3_array(arr: np.ndarray) -> str:
    """Format an ``(N,3)`` array as a USD bracketed tuple list."""
    arr = np.asarray(arr, dtype=np.float64)
    rows = [f"({x:.5g}, {y:.5g}, {z:.5g})" for x, y, z in arr]
    return "[" + ", ".join(rows) + "]"


def _fmt_float_array(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    return "[" + ", ".join(f"{v:.5g}" for v in arr) + "]"


def export_usda(
    recon: Reconstruction,
    path: str | Path,
    *,
    prim_name: str = "UnitreeL2Cloud",
    point_width: float = 0.02,
    with_color: bool = True,
) -> Path:
    """Write the reconstruction to a ``.usda`` file (ASCII, no deps)."""
    path = Path(path)
    xyz = recon.xyz
    n = xyz.shape[0]
    widths = np.full(n, point_width)

    header = (
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "World"\n'
        '    upAxis = "Z"\n'
        "    metersPerUnit = 1\n"
        '    doc = "Reconstructed Unitree 4D LiDAR L2 point cloud"\n'
        ")\n\n"
        'def Xform "World"\n{\n'
    )
    body = [header]
    body.append(f'    def Points "{prim_name}"\n    {{\n')
    body.append(f"        point3f[] points = {_fmt_vec3_array(xyz)}\n")
    body.append(f"        float[] widths = {_fmt_float_array(widths)}\n")
    if with_color and recon.intensity.size == n and n > 0:
        colors = intensity_to_color(recon.intensity)
        body.append(
            "        color3f[] primvars:displayColor = "
            f"{_fmt_vec3_array(colors)} (interpolation = \"vertex\")\n"
        )
    body.append("        uniform token[] xformOpOrder = []\n")
    body.append("    }\n")

    # also drop the sensor trajectory in as a thin reference (BasisCurves)
    if recon.poses:
        traj = np.array([T[:3, 3] for T in recon.poses])
        body.append('    def BasisCurves "SensorTrajectory"\n    {\n')
        body.append('        uniform token type = "linear"\n')
        body.append(f"        int[] curveVertexCounts = [{len(traj)}]\n")
        body.append(f"        point3f[] points = {_fmt_vec3_array(traj)}\n")
        body.append(f"        float[] widths = {_fmt_float_array(np.full(len(traj), 0.03))}\n")
        body.append(
            "        color3f[] primvars:displayColor = [(1, 0.9, 0)] "
            '(interpolation = "constant")\n'
        )
        body.append("    }\n")

    body.append("}\n")
    path.write_text("".join(body))
    return path


def export_usd_binary(recon: Reconstruction, path: str | Path,
                      *, prim_name: str = "UnitreeL2Cloud",
                      point_width: float = 0.02) -> Path:  # pragma: no cover
    """Write a binary ``.usd`` using the pxr bindings (Isaac Sim python)."""
    if not _HAVE_PXR:
        raise RuntimeError("pxr (OpenUSD) not available; use export_usda instead")
    path = Path(path)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    pts = UsdGeom.Points.Define(stage, f"/World/{prim_name}")
    pts.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(recon.xyz.astype(np.float32)))
    pts.CreateWidthsAttr(
        Vt.FloatArray.FromNumpy(np.full(len(recon), point_width, np.float32))
    )
    if recon.intensity.size == len(recon) and len(recon) > 0:
        colors = intensity_to_color(recon.intensity)
        pts.CreateDisplayColorPrimvar("vertex").Set(
            Vt.Vec3fArray.FromNumpy(colors)
        )
    stage.GetRootLayer().Save()
    return path


def export_usd(recon: Reconstruction, path: str | Path, **kw) -> Path:
    """Export to USD, choosing the best available backend by extension."""
    path = Path(path)
    if path.suffix == ".usd" and _HAVE_PXR:
        return export_usd_binary(recon, path, **{k: v for k, v in kw.items()
                                                 if k in ("prim_name", "point_width")})
    if path.suffix == ".usd":
        path = path.with_suffix(".usda")
    return export_usda(recon, path, **kw)
