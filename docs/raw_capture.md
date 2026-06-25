# Processing a raw L2 capture (native `0x55AA050A` framing)

When you only have a **raw capture** of the sensor — a `.pcap` from Wireshark/tcpdump, a
raw UDP-payload dump, or a live socket — and not a ROS bag or the SDK, this pipeline can
still ingest it. The parser is pure stdlib + numpy, so it runs on Windows.

```powershell
# inspect what's in the capture
unitree-l2 raw-info capture.pcap

# reconstruct straight from it
unitree-l2 run --source capture.pcap --out output

# or capture live from the device's native stream
unitree-l2 capture --native --port 6201 --out data\captures\live
unitree-l2 export data\captures\live --out output
```

Accepted raw sources: `.pcap` / `.pcapng*`, `.bin` / `.raw` (concatenated native
frames), and `--native` live UDP. (*classic `.pcap`; `.pcapng` containers aren't parsed —
re-save as `.pcap` in Wireshark.)

## How to capture a `.pcap`

The L2 streams UDP from `192.168.1.62` to the host `192.168.1.2:6201`. With the sensor on
your network:

```bash
# Linux/macOS
sudo tcpdump -i <iface> host 192.168.1.62 -w capture.pcap
```

On Windows use **Wireshark** (capture the L2 interface, filter `ip.addr == 192.168.1.62`,
**File ▸ Save As ▸ .pcap**).

## What is reliable vs. approximate

This matters — read it before trusting the geometry.

| Field | Status |
|---|---|
| Frame demux (type 102/103/104/105, sizes, `0x00FF` tail) | **Exact** — documented 12-byte header/tail |
| Timestamps, `point_num`, scan params | **Exact** |
| `ranges` (mm → m), `intensities` (0–255) | **Exact** |
| IMU quaternion / angular velocity / linear acceleration | **Decoded** (trailing-offset assumption; see below) |
| Cartesian **XYZ** | **Approximate** — see below |

**Why XYZ is approximate.** The L2 is a prism/nutating 4D LiDAR; the exact polar→Cartesian
conversion uses the per-device `LidarCalibParam` baked into Unitree's closed SDK. From a
raw capture we only have the documented angle fields, so we reconstruct a spherical
estimate:

```
azimuth_i   = com_horizontal_angle_start + i * com_horizontal_angle_step
elevation_i = angle_min                  + i * angle_increment
x = r·cos(el)·cos(az),  y = r·cos(el)·sin(az),  z = r·sin(el)
```

This is great for previewing, debugging, and coarse registration, but it is **not
calibration-accurate**. For metric-accurate clouds, get points from the SDK/ROS driver —
record a `.bag` and use `unitree-l2 run --source run.bag`, which is exact.

**IMU offset assumption.** The 104 body carries undocumented reserved bytes; we read the
ten motion floats from the **end** of the body and the timestamp from the front. If your
firmware lays them out differently, adjust `decode_imu_body` in `raw_protocol.py`.

**CRC.** `raw-info` reports how many packets match a `zlib.crc32(header+body)`. The vendor's
exact CRC coverage/polynomial is undocumented, so this is informational only — packets are
never rejected on a CRC mismatch.

## Frame assembly

A single 102 packet is ~300 points (one short scan segment), so the parser accumulates
`packets_per_frame` packets (default 72 ≈ one revolution ≈ ~21.6k points) into each
`LidarFrame`, attaching the most recent IMU. This boundary is heuristic; tune it to your
firmware's packets-per-revolution if scans look split or merged.

## Internals

* `raw_protocol.py` — native framing (`encode_frame`/`decode_frame`/`iter_frames_from_stream`),
  body decoders (`decode_point_body`, `decode_imu_body`), and `assemble_frames`.
* `ingest/pcap.py` — `iter_frames_from_pcap` (Ethernet/IPv4/UDP demux),
  `iter_frames_from_raw`, and `capture_native_udp`.

Validated by encode→decode roundtrip tests (framing, body fields, IMU) and an end-to-end
synthetic-pcap→USD test; see `tests/test_raw_protocol.py`.
