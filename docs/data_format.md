# Unitree 4D LiDAR L2 — data format & protocol

This document describes the data model this pipeline uses and how it maps to the official
[`unilidar_sdk2`](https://github.com/unitreerobotics/unilidar_sdk2).

## Point structure

Each L2 point matches the SDK's `PointUnitree`:

| Field       | Type       | Units / meaning                                            |
|-------------|------------|------------------------------------------------------------|
| `x, y, z`   | `float32`  | Cartesian coordinates, **metres**, sensor frame            |
| `intensity` | `float32`  | Reflectivity / return signal strength                      |
| `time`      | `float32`  | Seconds relative to the start of the scan                  |
| `ring`      | `uint32`   | Laser channel index, `0 .. 17` (the L2 has **18 rings**)   |

In code this is `unitree_l2_pipeline.formats.POINT_DTYPE`, a packed little-endian numpy
structured dtype byte-compatible with the SDK struct (24 bytes/point).

### Coordinate frame

Right-handed; the origin is the **centre of the bottom mounting surface** of the LiDAR.
`+Z` is up. The pipeline authors USD **Z-up, metres**, so reconstructed clouds land in
Isaac Sim at correct scale and orientation.

## Scan / frame

A scan matches `ScanUnitree`: a timestamp (`stamp`, seconds), a monotonically increasing
`id`, a valid-point count, and the point array. The pipeline wraps this as
`formats.LidarFrame` (points + `stamp` + `frame_id` + optional `imu` + optional `pose`).

## IMU

The L2 carries an IMU (`IMUUnitree`): `stamp`, `id`, `quaternion[w,x,y,z]`,
`angular_velocity[3]` (rad/s), `linear_acceleration[3]` (m/s²). Carried as
`formats.ImuSample` and used to seed registration when available.

## Network protocol (factory defaults)

| Parameter            | Default        |
|----------------------|----------------|
| LiDAR IP             | `192.168.1.62` |
| Host / server IP     | `192.168.1.2`  |
| LiDAR data UDP port   | `6101`         |
| Host receive UDP port | `6201`         |
| Serial device        | `/dev/ttyACM0` |

The vendor's on-wire framing is proprietary and decoded inside the closed SDK. For an
open, testable ingest path this pipeline defines a small documented UDP payload
(`protocol.py`): a 24-byte header (`magic`, `version`, `kind`, `stamp`, `frame_id`,
`count`) followed by either `count` × `PointUnitree` records or an IMU record. Datagrams
are split to stay under a typical 1500-byte MTU and reassembled by `frame_id`.

To ingest from real hardware, bridge the SDK callback into
`ingest.udp_capture.FrameAssembler.add_points(...)` — everything downstream is identical.

## On-disk formats

| Format  | Writer/Reader                         | Notes                                        |
|---------|---------------------------------------|----------------------------------------------|
| `.npz`  | `save_frame_npz` / `load_frame_npz`   | Canonical, lossless: points + pose + IMU (compressed) |
| `.pcd`  | `save_pcd` / `load_pcd`               | PCL-compatible, ascii or binary, `x y z intensity` |
| `.ply`  | `save_ply`                            | ascii, optional per-vertex RGB, for MeshLab/Blender |
| `.usda` | `export.usd_export`                   | OpenUSD `Points` prim for Isaac Sim          |

A *dataset* is just a directory of per-frame `.npz` (or `.pcd`) files loaded in sorted
filename order.

## Real Unitree L2 ROS bags

Unitree publishes real L2 recordings as ROS1 bags (`point_lio_unilidar`):

| Dataset   | URL |
|-----------|-----|
| L2 Indoor | `https://oss-global-cdn.unitree.com/static/L2%20Indoor%20Point%20Cloud%20Data.bag` |
| L2 Park   | `https://oss-global-cdn.unitree.com/static/L2%20Park%20Point%20Cloud%20Data.bag` |

Each bag contains two topics:

| Topic              | Type                      | Notes |
|--------------------|---------------------------|-------|
| `/unilidar/cloud`  | `sensor_msgs/PointCloud2` | `point_step` 32: `x,y,z` f32; `intensity` f32 @16; `ring` u16 @20; `time` f32 @24 |
| `/unilidar/imu`    | `sensor_msgs/Imu`         | orientation, angular velocity, linear acceleration |

`ingest/rosbag.py` reads these **without ROS installed**: it parses the ROS1 bag v2.0
container (uncompressed / bz2 / lz4 chunks) and decodes the `PointCloud2` layout
*dynamically* from the message's own `PointField` descriptors, so it adapts to firmware
variations. (Note: in the published indoor bag the driver emits a constant `ring=1` and
`intensity=255`; `time` is a valid per-point offset. The reader passes through whatever
the bag actually contains.) Use `unitree-l2 bag-info <bag>` for a quick summary, and
`scripts/fetch_l2_bag.sh` to download the bags (they are large, so not committed).
