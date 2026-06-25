"""End-to-end pipeline for Unitree 4D LiDAR L2 data.

Ingest (synthetic / UDP / recorded) -> reconstruct (register + aggregate) ->
export to USD for Isaac Sim.
"""

from .formats import (
    L2_NUM_RINGS,
    POINT_DTYPE,
    ImuSample,
    LidarFrame,
)
from .pipeline import PipelineConfig, run_pipeline

__all__ = [
    "L2_NUM_RINGS",
    "POINT_DTYPE",
    "ImuSample",
    "LidarFrame",
    "PipelineConfig",
    "run_pipeline",
]

__version__ = "0.1.0"
