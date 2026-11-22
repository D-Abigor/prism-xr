# Path Finder - Track 03 (XR Hackathon 2026)

Estimates a 6-DoF trajectory (position + orientation) from a **degraded IMU stream** and
dead-reckons the **5 s after the IMU fails**. Single thread, offline, streaming (memory does not grow with file length).

| | |
|---|---|
| Main version | `main.cpp` (C++11, standard library only) |
| Same algorithm in Python | `main.py` (standard library only, no numpy) - identical output |
| Input | EuRoC `imu0/data.csv` (`#timestamp [ns], w_x,w_y,w_z [rad/s], a_x,a_y,a_z [m/s^2]`) |
| Output | `estimated_trajectory.txt` - TUM format: `timestamp[s] tx ty tz qx qy qz qw`, one pose per IMU row |

## Build and run

```bash
make                                     # needs g++ (any C++11 compiler)
./pathfinder <imu0/data.csv | folder containing imu0/data.csv> estimated_trajectory.txt
# Python alternative (same result):
python3 main.py <imu0/data.csv | folder> estimated_trajectory.txt
```
`make static` builds a statically linked binary (about 0.8 MB resident). A `Dockerfile` is included (untested, see its header).
Prints one summary line on stderr (rows, dropouts found, peak memory); `--quiet` hides it.

## How it works

State: orientation quaternion `q` (body to world), velocity `v`, position `p`, gyro bias. Per IMU row:

1. **Start-up** (first ~100 rows): if the sensor is still, average the gyro (= gyro bias) and the accelerometer.
   The accelerometer direction is "up", so a rotation taking it onto world +z gives the initial roll/pitch (this removes the
   *unknown gravity alignment*; yaw starts at 0). The measured gravity magnitude is used for `g`.
2. **Orientation**: integrate the bias-corrected gyro, `q <- q (x) exp(w dt)` (trapezoid rule), renormalise.
3. **Position**: rotate the accelerometer into the world frame, subtract gravity, integrate twice
   (`p += v dt + a dt^2 / 2`, `v += a dt`).
4. **Drift control**
   * *Tilt correction* - only in quasi-steady flight (gyro < 0.1 rad/s and estimated linear acceleration < 0.15 m/s^2) the
     accelerometer is trusted as "up" and roll/pitch are nudged toward it (gain 0.002/sample). During turns/acceleration it is
     off, because the accelerometer then does not measure gravity alone.
   * *Zero-velocity update* - if a 0.2 s block looks perfectly still **and** speed is already < 0.3 m/s, velocity is reset
     to 0 and the gyro bias is refreshed. (A moving aircraft at constant speed also looks "still" to an IMU, hence the speed gate.)
5. **Dropouts** (missing rows, timestamp gaps): gaps up to 2.5 s are bridged by linearly interpolating the two good samples on
   either side; longer gaps are coasted. Corrupted rows (NaN/empty/garbage, all-zero, 10+ identical rows, out-of-range values)
   are treated as missing.
6. **Coasting / dead reckoning** (no usable data): velocity decays as `exp(-0.1 t)`, angular rate as `exp(-0.5 t)`.
   After the **last row** the program appends poses at the nominal IMU rate until 5 s of dead reckoning have been written
   (time already spent in trailing corrupted rows counts towards the 5 s, so the output never has surplus rows).

Timestamps are converted ns -> s with integer arithmetic (`%lld.%09lld`), so no precision is lost.

## Parameters

All have defaults; override with `--name value` (e.g. `--decay-v 0.3`). Tune these on the real data.

| Option | Default | Meaning |
|---|---|---|
| `--tail` | 5 | seconds of dead reckoning to guarantee after the IMU stops (0 = off) |
| `--decay-v` / `--decay-w` | 0.10 / 0.50 | 1/s decay of velocity / angular rate while coasting |
| `--gap-interp` | 2.5 | s; gaps up to this are interpolated, longer ones coasted |
| `--alpha` | 0.002 | tilt-correction gain per sample (0 = gyro only) |
| `--gate-lin` / `--gate-w` | 0.15 / 0.10 | m/s^2, rad/s limits for allowing tilt correction |
| `--zupt-vmax` | 0.30 | m/s; zero-velocity updates only below this speed |
| `--vmax` | 80 | m/s safety clamp |
| `--init-rows` | 100 | rows used for start-up (10..512) |

## Measured performance (authoring sandbox: Intel Xeon 2.1 GHz, 1 core)

| | C++ (`-O2`) | Python 3.12 |
|---|---|---|
| Compute per 1 s of IMU data (limit 20 ms) | 0.44 ms | 2.2 ms |
| Peak resident memory of the process (limit 8 MB) | 2.2 MB (0.8 MB static) | 11.1 MB (interpreter alone is ~9 MB) |

Memory is flat from 9 thousand to 283 thousand rows (1500 s of data). If the 8 MB limit is applied to the whole process, use the C++ version.
`tools/bench.py` reproduces these numbers and writes `metadata.json`.

## Self-test on simulated data (NOT real accuracy)

`tools/make_synthetic.py` simulates a 50 s flight with known ground truth, a 2 s stationary start, then growing noise,
bias drift, random dropped rows, a 0.3 s and a 2 s signal loss, and a failure at 50 s followed by 5 s blind.
`tools/evaluate.py` scores a trajectory (SE(3)-aligned ATE RMSE like `evo`, plus unaligned error).

| Simulated level | ATE, flight part | ATE, blind 5 s part |
|---|---|---|
| clean (no noise, no dropouts) | 0.10 m | 5.2 m |
| mild degradation | 7.3 m | 33 m |
| hostile degradation | 19.9 m | 86 m |

The clean case proves the maths is right (the same code integrating noise-free data gets ~1 mm with the tilt correction off).
The other rows show how errors grow with sensor degradation; they say nothing about EuRoC/TUM-VI accuracy, which we could not
test without the data. Reproduce:

```bash
python3 tools/make_synthetic.py /tmp/mild --level mild            # also: clean, hostile; --fail-mode nan|zeros|freeze|truncate
./pathfinder /tmp/mild /tmp/est.txt
python3 tools/evaluate.py /tmp/est.txt /tmp/mild/gt.txt --fail-time 50
```
(`make_synthetic.py` and `evaluate.py` need numpy; the estimator itself does not.)

## Assumptions and limits

* **IMU-only integration always drifts** - position error grows roughly with t^2. This code limits the drift, it cannot remove it.
* Output has exactly one pose per input row (including rows that were corrupted); rows missing from the file are not invented.
  If the IMU file simply ends at the failure, 5 s of extra rows are appended at the median sample interval. Use `--tail 0` if the
  organisers' ground truth has no such rows.
* The mission checkpoints are not given as an input in the brief; the program does not need them.
* The world frame starts at the origin with yaw 0. If scoring does not align trajectories first, the ground-truth start
  orientation must be matched by the evaluation tooling.
* After failure the aircraft is assumed to keep its last velocity (slowly decaying) and hold its last attitude trend. A model that
  knows the vehicle type (e.g. fixed-wing flies along its body axis) could do better; try it with the real data.
# prism-xr
