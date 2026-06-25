# Drift-free poses with cuVSLAM (NVIDIA Isaac ROS Visual SLAM)

Frame-to-frame ICP drifts over long sequences. The robust, GPU-native way to place L2
scans in a consistent world frame is to use a trajectory from **cuVSLAM** — NVIDIA's
CUDA-accelerated stereo visual-inertial SLAM — and transform each scan by the
corresponding pose. This is also the natural fit for an Isaac Sim workflow.

## Why a camera SLAM for a LiDAR map?

cuVSLAM tracks the **robot base** from stereo cameras (+ IMU), not the LiDAR. The L2 is
rigidly bolted to the same body, so a single static extrinsic relates them. The pipeline
composes them per scan:

```
T_world_lidar(t) = T_world_base(t) @ T_base_lidar
```

`T_world_base(t)` is the cuVSLAM pose interpolated (SLERP + lerp) to the scan's timestamp,
so the LiDAR and camera clocks only need to be *synchronized*, not sample-aligned.
`T_base_lidar` is your LiDAR→base mounting transform.

## Two ways to feed cuVSLAM poses in

### A. Isaac ROS (recommended on a robot)

[`isaac_ros_visual_slam`](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam) runs
cuVSLAM as a ROS 2 node. It publishes:

| Topic | Type | Meaning |
|---|---|---|
| `/visual_slam/tracking/odometry` | `nav_msgs/Odometry` | `odom_frame -> base_link` (visual odometry) |
| `/visual_slam/tracking/vo_pose` | `geometry_msgs/PoseStamped` | latest VO pose |
| `/visual_slam/tracking/slam_path` | `nav_msgs/Path` | `map_frame -> base_link` (loop-closed SLAM) |

Record the odometry **alongside** the LiDAR cloud, then import it straight from the bag:

```bash
# record both streams together
ros2 bag record /unilidar/cloud /unilidar/imu /visual_slam/tracking/odometry

# reconstruct using cuVSLAM odometry as the pose source
unitree-l2 run --source run.bag \
    --poses run.bag \
    --poses-topic /visual_slam/tracking/odometry \
    --extrinsic "0.10 0 0.05  0 0 0 1" \
    --out output
```

> `odom` is continuous but drifts slowly; `map` (SLAM, with loop closure) is globally
> consistent but can jump on loop closures. For offline reconstruction prefer the
> loop-closed trajectory if you have it.

### B. Standalone PyCuVSLAM (no ROS)

[`PyCuVSLAM`](https://github.com/nvidia-isaac/PyCuVSLAM) is NVIDIA's official Python wrapper
for cuVSLAM (mono / stereo / RGB-D / multi-cam / visual-inertial). Run it on your recorded
camera frames, export the per-frame poses to a **TUM** trajectory file, and import that:

```python
# sketch — see the PyCuVSLAM docs for the exact Tracker/Rig API
from unitree_l2_pipeline.reconstruct.pose_source import Trajectory, save_tum
import numpy as np

stamps, positions, quats_wxyz = [], [], []
for t, left, right in stereo_stream:           # your synchronized frames
    pose = tracker.track(t, [left, right])     # cuVSLAM pose (base in world)
    stamps.append(t)
    positions.append(pose.translation)         # (x, y, z)
    q = pose.rotation_quaternion               # (x, y, z, w) or (w, x, y, z)
    quats_wxyz.append([q.w, q.x, q.y, q.z])

save_tum("cuvslam.tum", Trajectory(np.array(stamps), np.array(positions),
                                   np.array(quats_wxyz)))
```

```bash
unitree-l2 run --source run.bag --poses cuvslam.tum --extrinsic "0.10 0 0.05 0 0 0 1" --out output
```

TUM format is one pose per line: `timestamp tx ty tz qx qy qz qw`.

## The extrinsic

`--extrinsic` accepts:

* `identity` (or omitted) — LiDAR frame == base frame.
* seven values `x y z qx qy qz qw` — translation + quaternion of the LiDAR in the base frame.
* sixteen values — a row-major 4×4 matrix.

Get it from your robot's URDF/TF (`base_link` → LiDAR link) or a LiDAR–camera calibration.
A wrong extrinsic shows up as a consistent "ghosting" offset between overlapping scans.

## How it works internally

* `reconstruct/pose_source.py` — `Trajectory` (SLERP/lerp interpolation), `load_tum`,
  `load_cuvslam_bag` (decodes `nav_msgs/Odometry` from a bag), `parse_extrinsic`,
  `poses_for_frames`.
* `ingest/rosbag.py` — `iter_bag_odometry()` decodes the odometry topic with no ROS install.
* `pipeline.py` — when `--poses` is set, it interpolates a world pose per scan instead of
  running ICP, so reconstruction is drift-free.

## Verifying

The test suite generates synthetic scans with a known trajectory, writes it as TUM, and
checks that `--poses` reconstruction recovers the true room dimensions (drift-free) — in
contrast to the `--icp` path, which drifts. See `tests/test_pose_source.py`.
