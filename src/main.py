#!/usr/bin/env python3
"""IMU-only 6-DoF trajectory estimator for Track 3.

The program streams the input more than once and keeps only fixed-size
calibration buffers and the current state. Output uses TUM format and has one
pose for every data row, including duplicate timestamps. Missing timestamps are
integrated across internally; they do not create output rows.
"""

import argparse
import csv
import math
import os
import statistics
import sys

G = 9.80665
NS = 1_000_000_000
MAX_GYRO = 35.0
MAX_ACCEL = 160.0
# Apply a gentle roll/pitch correction only when acceleration is close to
# gravity and angular motion is modest. This limits gyro integration drift
# without treating strong translational acceleration as a gravity direction.
GRAVITY_CORRECTION_GAIN = 0.2
GRAVITY_CORRECTION_ACCEL_BAND = 1.0
GRAVITY_CORRECTION_MAX_GYRO = 2.0


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def qnormalize(q):
    n = math.sqrt(sum(x * x for x in q))
    if n <= 1e-15:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(x / n for x in q)


def qfrom_rotvec(r):
    theta = norm(r)
    if theta < 1e-10:
        return qnormalize((1.0, 0.5 * r[0], 0.5 * r[1], 0.5 * r[2]))
    scale = math.sin(0.5 * theta) / theta
    return (math.cos(0.5 * theta), r[0] * scale, r[1] * scale, r[2] * scale)


def qrotate(q, v):
    w, x, y, z = q
    u = (x, y, z)
    t = (2.0 * (u[1] * v[2] - u[2] * v[1]),
         2.0 * (u[2] * v[0] - u[0] * v[2]),
         2.0 * (u[0] * v[1] - u[1] * v[0]))
    c = (u[1] * t[2] - u[2] * t[1],
         u[2] * t[0] - u[0] * t[2],
         u[0] * t[1] - u[1] * t[0])
    return (v[0] + w * t[0] + c[0],
            v[1] + w * t[1] + c[1],
            v[2] + w * t[2] + c[2])


def clean_name(value):
    return "".join(ch.lower() for ch in value if ch.isalnum())


def find_header_indices(header):
    if not header:
        return None
    names = [clean_name(x.lstrip("#")) for x in header]
    gyro = [-1, -1, -1]
    accel = [-1, -1, -1]
    for i, name in enumerate(names):
        for axis, letter in enumerate("xyz"):
            if ("wrss" + letter) in name or name in ("gyro" + letter, "g" + letter):
                gyro[axis] = i
            if ("arss" + letter) in name or ("accel" + letter) in name or name in ("acc" + letter, "a" + letter):
                accel[axis] = i
    if min(gyro + accel) >= 0:
        return tuple(gyro), tuple(accel)
    return None


def first_header_and_rows(path):
    """Return the optional header and first numeric record without retaining rows."""
    header = None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row or not any(x.strip() for x in row):
                continue
            token = row[0].strip()
            if token.startswith("#") or _not_float(token):
                if len(row) >= 7:
                    header = row
                continue
            return header, row
    raise ValueError("input CSV contains no records")


def _not_float(value):
    try:
        float(value)
        return False
    except (TypeError, ValueError):
        return True


def determine_layout(path, order_override, accel_unit_override, timestamp_unit_override, gyro_degrees):
    header, first = first_header_and_rows(path)
    indices = find_header_indices(header)
    if order_override == "gyro-first":
        indices = ((1, 2, 3), (4, 5, 6))
    elif order_override == "accel-first":
        indices = ((4, 5, 6), (1, 2, 3))

    sample = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if _not_float(row[0]):
                continue
            if indices is not None:
                try:
                    g = tuple(float(row[i]) for i in indices[0])
                    a = tuple(float(row[i]) for i in indices[1])
                except (IndexError, ValueError):
                    continue
            else:
                try:
                    left = tuple(float(row[i]) for i in (1, 2, 3))
                    right = tuple(float(row[i]) for i in (4, 5, 6))
                except (IndexError, ValueError):
                    continue
                sample.append((left, right))
                if len(sample) >= 256:
                    break
            if indices is not None:
                sample.append((g, a))
                if len(sample) >= 256:
                    break
    if not sample:
        raise ValueError("could not infer CSV column order or read numeric rows")

    if indices is None:
        med_left = statistics.median(norm(x[0]) for x in sample)
        med_right = statistics.median(norm(x[1]) for x in sample)
        # The acceleration triplet is normally near 9.8 m/s^2 or 1 g, while
        # angular velocity is usually much smaller. Ambiguous headerless input
        # follows the EuRoC order from the Track 3 specification.
        accel_left = (0.5 <= med_left <= 20.0 and med_left > 1.5 * max(med_right, 1e-6))
        indices = ((4, 5, 6), (1, 2, 3)) if accel_left else ((1, 2, 3), (4, 5, 6))
        if accel_left:
            sample = [(right, left) for left, right in sample]

    timestamp_unit = timestamp_unit_override
    if timestamp_unit == "auto":
        header_text = " ".join(header or []).lower()
        if "ns" in header_text or (first and abs(float(first[0])) >= 1e14):
            timestamp_unit = "ns"
        elif "us" in header_text:
            timestamp_unit = "us"
        elif "ms" in header_text or (first and abs(float(first[0])) >= 1e11):
            timestamp_unit = "ms"
        else:
            timestamp_unit = "s"
    ts_scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[timestamp_unit]

    accel_unit = accel_unit_override
    if accel_unit == "auto":
        norms = [norm(x[1]) for x in sample]
        med_norm = statistics.median(norms)
        accel_unit = "g" if 0.5 <= med_norm <= 2.0 else "ms2"
    accel_scale = G if accel_unit == "g" else 1.0
    gyro_scale = math.pi / 180.0 if gyro_degrees else 1.0
    return header, indices, ts_scale, accel_scale, gyro_scale, timestamp_unit, accel_unit


def timestamp_seconds(token, scale, unit):
    if unit == "ns" and token.strip().lstrip("+-").isdigit():
        n = int(token)
        sec = (n // NS) + (n % NS) * 1e-9
        return sec, "%d.%09d" % (n // NS, n % NS)
    sec = float(token) * scale
    return sec, "%.9f" % sec


def iter_samples(path, layout):
    header, indices, ts_scale, accel_scale, gyro_scale, ts_unit, _ = layout
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if _not_float(row[0]):
                continue
            if len(row) < 7:
                raise ValueError("expected at least 7 CSV columns")
            try:
                ts, ts_text = timestamp_seconds(row[0], ts_scale, ts_unit)
                gyro = tuple(float(row[i]) * gyro_scale for i in indices[0])
                accel = tuple(float(row[i]) * accel_scale for i in indices[1])
                finite = all(math.isfinite(x) for x in gyro + accel + (ts,))
            except (ValueError, IndexError, OverflowError) as exc:
                raise ValueError("malformed numeric CSV row") from exc
            valid = finite and norm(gyro) <= MAX_GYRO and norm(accel) <= MAX_ACCEL
            valid = valid and not (norm(gyro) < 1e-12 and norm(accel) < 1e-12)
            yield ts, ts_text, gyro, accel, valid


def inspect_stream(path, layout):
    dt_samples = []
    gyro_sum = [0.0, 0.0, 0.0]
    gyro_count = 0
    startup = []
    total_rows = 0
    previous_ts = None
    for ts, _, gyro, accel, valid in iter_samples(path, layout):
        total_rows += 1
        if previous_ts is not None:
            dt = ts - previous_ts
            if 0.0 < dt < 10.0 and len(dt_samples) < 4096:
                dt_samples.append(dt)
        previous_ts = ts
        if valid:
            for i in range(3):
                gyro_sum[i] += gyro[i]
            gyro_count += 1
            if len(startup) < 200:
                startup.append((gyro, accel))
    if not total_rows or not gyro_count:
        raise ValueError("input has no usable IMU samples")
    nominal_dt = statistics.median(dt_samples) if dt_samples else 0.005
    global_bias = tuple(x / gyro_count for x in gyro_sum)

    startup_stationary = False
    startup_bias = global_bias
    if len(startup) == 200:
        gm = tuple(sum(s[0][k] for s in startup) / 200.0 for k in range(3))
        am = tuple(sum(s[1][k] for s in startup) / 200.0 for k in range(3))
        gvar = max(0.0, sum(sum(x * x for x in s[0]) for s in startup) / 200.0 - sum(x*x for x in gm))
        avar = max(0.0, sum(sum(x * x for x in s[1]) for s in startup) / 200.0 - sum(x*x for x in am))
        first_half = tuple(sum(startup[i][1][k] for i in range(100)) / 100.0 for k in range(3))
        second_half = tuple(sum(startup[i][1][k] for i in range(100, 200)) / 100.0 for k in range(3))
        avg_norm = sum(norm(s[1]) for s in startup) / 200.0
        startup_stationary = (gvar < 0.0025 and avar < 0.02 and abs(avg_norm - G) < 0.6
                              and norm(sub(first_half, second_half)) < 0.03)
        if startup_stationary:
            startup_bias = gm
    return nominal_dt, (startup_bias if startup_stationary else global_bias), startup_stationary, total_rows


def estimate_gravity(path, layout, nominal_dt, gyro_bias, accel_bias, startup_stationary, total_rows):
    q = (1.0, 0.0, 0.0, 0.0)
    prev = None
    force_sum = [0.0, 0.0, 0.0]
    count = 0
    startup_force = [0.0, 0.0, 0.0]
    startup_count = 0
    stride = max(1, int(math.ceil(total_rows / 2048.0)))
    robust_forces = []
    row_index = 0
    for ts, _, gyro, accel, valid in iter_samples(path, layout):
        sample_this_row = row_index % stride == 0 and len(robust_forces) < 2048
        row_index += 1
        if not valid:
            prev = None
            continue
        if prev is not None:
            dt = ts - prev[0]
            if dt > 0.0 and dt <= 10.0:
                w = mul(add(prev[1], gyro), 0.5)
                w = sub(w, gyro_bias)
                q = qnormalize(qmul(q, qfrom_rotvec(mul(w, dt))))
        f_world = qrotate(q, sub(accel, accel_bias))
        if sample_this_row:
            # A bounded, deterministic sample gives a robust fallback when
            # the sequence has no detectable stationary interval.
            robust_forces.append(f_world)
        for k in range(3):
            force_sum[k] += f_world[k]
        count += 1
        if startup_count < 200:
            for k in range(3):
                startup_force[k] += f_world[k]
            startup_count += 1
        prev = (ts, gyro, accel)
    if not count:
        return (0.0, 0.0, -G)
    mean_force = (tuple(x / startup_count for x in startup_force)
                  if startup_stationary and startup_count else
                  (tuple(statistics.median(force[k] for force in robust_forces)
                         for k in range(3)) if robust_forces else tuple(x / count for x in force_sum)))
    magnitude = norm(mean_force)
    return mul(mean_force, -G / magnitude) if magnitude > 1.0 else (0.0, 0.0, -G)


class Integrator:
    def __init__(self, out, gravity, gyro_bias, accel_bias, nominal_dt, max_interp):
        self.out = out
        self.gravity = gravity
        self.gyro_bias = gyro_bias
        self.accel_bias = accel_bias
        self.nominal_dt = nominal_dt
        self.max_interp = max_interp
        self.position = (0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0, 0.0)
        self.orientation = (1.0, 0.0, 0.0, 0.0)
        self.previous = None
        self.last_accel_world = (0.0, 0.0, 0.0)
        self.last_gyro = (0.0, 0.0, 0.0)
        self.coast_time = 0.0
        self.invalid_duration = 0.0
        self.output_rows = 0
        self.coast_steps = 0
        self.filled_steps = 0
        self.max_gap = 0.0

    def write_pose(self, ts_text):
        q = self.orientation
        if q[0] < 0.0:
            q = tuple(-x for x in q)
        self.out.write("%s %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" %
                       (ts_text, self.position[0], self.position[1], self.position[2],
                        q[1], q[2], q[3], q[0]))
        self.output_rows += 1

    def integrate_step(self, h, gyro, accel):
        w = sub(gyro, self.gyro_bias)
        f = sub(accel, self.accel_bias)
        qmid = qnormalize(qmul(self.orientation, qfrom_rotvec(mul(w, 0.5 * h))))
        self.orientation = qnormalize(qmul(self.orientation, qfrom_rotvec(mul(w, h))))
        force_norm = norm(f)
        if (abs(force_norm - G) <= GRAVITY_CORRECTION_ACCEL_BAND
                and norm(w) <= GRAVITY_CORRECTION_MAX_GYRO):
            # At low dynamics, specific force points opposite gravity. Use
            # its world-frame direction as a weak attitude observation. The
            # correction is applied in world coordinates to avoid changing
            # yaw, which an accelerometer cannot observe.
            measured_up = mul(qrotate(qmid, f), 1.0 / force_norm)
            target_up = mul(self.gravity, -1.0 / norm(self.gravity))
            error_axis = (measured_up[1] * target_up[2] - measured_up[2] * target_up[1],
                          measured_up[2] * target_up[0] - measured_up[0] * target_up[2],
                          measured_up[0] * target_up[1] - measured_up[1] * target_up[0])
            correction = qfrom_rotvec(mul(error_axis, GRAVITY_CORRECTION_GAIN * h))
            self.orientation = qnormalize(qmul(correction, self.orientation))
        aw = add(qrotate(qmid, f), self.gravity)
        self.position = add(self.position, add(mul(self.velocity, h), mul(aw, 0.5 * h * h)))
        self.velocity = add(self.velocity, mul(aw, h))
        speed = norm(self.velocity)
        if speed > 100.0:
            self.velocity = mul(self.velocity, 100.0 / speed)
        self.last_gyro = w
        self.last_accel_world = aw
        self.coast_time = 0.0

    def coast(self, h):
        if h <= 0.0:
            return
        decay_a = math.exp(-0.8 * self.coast_time)
        aw = mul(self.last_accel_world, decay_a)
        self.position = add(self.position, add(mul(self.velocity, h), mul(aw, 0.5 * h * h)))
        self.velocity = add(self.velocity, mul(aw, h))
        speed = norm(self.velocity)
        if speed > 100.0:
            self.velocity = mul(self.velocity, 100.0 / speed)
        w = mul(self.last_gyro, math.exp(-0.5 * self.coast_time))
        self.orientation = qnormalize(qmul(self.orientation, qfrom_rotvec(mul(w, h))))
        self.coast_time += h
        self.coast_steps += 1

    def process(self, sample):
        ts, ts_text, gyro, accel, valid = sample
        if self.previous is None:
            self.previous = sample
            self.write_pose(ts_text)
            return
        pts, _, pgyro, paccel, pvalid = self.previous
        dt = ts - pts
        if dt <= 0.0:
            # Preserve one output row per input record, including duplicate stamps.
            self.write_pose(ts_text)
            if dt == 0.0:
                self.previous = sample
            return
        self.max_gap = max(self.max_gap, dt)
        if valid and pvalid and dt <= self.max_interp:
            self.invalid_duration = 0.0
            steps = max(1, int(math.floor(dt / self.nominal_dt + 0.5)))
            steps = min(steps, 20000)
            h = dt / steps
            for i in range(steps):
                u0 = i / steps
                u1 = (i + 1) / steps
                g0 = add(mul(pgyro, 1.0 - u0), mul(gyro, u0))
                g1 = add(mul(pgyro, 1.0 - u1), mul(gyro, u1))
                a0 = add(mul(paccel, 1.0 - u0), mul(accel, u0))
                a1 = add(mul(paccel, 1.0 - u1), mul(accel, u1))
                self.integrate_step(h, mul(add(g0, g1), 0.5), mul(add(a0, a1), 0.5))
            self.filled_steps += max(0, steps - 1)
        else:
            self.coast(dt)
            # Use a valid resumed sample as the next integration endpoint.
            if valid:
                self.last_gyro = sub(gyro, self.gyro_bias)
                self.last_accel_world = add(qrotate(self.orientation, sub(accel, self.accel_bias)), self.gravity)
                self.coast_time = 0.0
                self.invalid_duration = 0.0
            else:
                self.invalid_duration += dt
                # Sustained unusable rows mark sensor failure. Continue at
                # constant velocity instead of extending stale acceleration.
                if self.invalid_duration >= 0.1:
                    self.last_accel_world = (0.0, 0.0, 0.0)
                    self.last_gyro = (0.0, 0.0, 0.0)
                    self.coast_time = 0.0
        self.write_pose(ts_text)
        self.previous = sample


def resolve_input(path):
    if os.path.isdir(path):
        candidates = (os.path.join(path, "imu0", "data.csv"),
                      os.path.join(path, "mav0", "imu0", "data.csv"))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    return path


def parse_vec(text):
    values = tuple(float(x) for x in text.split(","))
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise argparse.ArgumentTypeError("expected three comma-separated finite values")
    return values


def main():
    ap = argparse.ArgumentParser(description="Estimate one TUM pose per degraded IMU CSV row.")
    ap.add_argument("input", help="imu0/data.csv or a dataset folder")
    ap.add_argument("output", nargs="?", default="estimated_trajectory.txt")
    ap.add_argument("--order", choices=("auto", "gyro-first", "accel-first"), default="auto",
                    help="channel order when the CSV has no recognizable header")
    ap.add_argument("--timestamp-unit", choices=("auto", "s", "ms", "us", "ns"), default="auto")
    ap.add_argument("--accel-unit", choices=("auto", "ms2", "g"), default="auto")
    ap.add_argument("--gyro-deg", action="store_true", help="gyro values are in degrees per second")
    ap.add_argument("--gyro-bias", type=parse_vec, help="known gyro bias as x,y,z in rad/s")
    ap.add_argument("--accel-bias", type=parse_vec, help="known accel bias as x,y,z in m/s^2")
    ap.add_argument("--max-interp", type=float, default=2.5,
                    help="largest timestamp gap to interpolate internally (default 2.5 s)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.max_interp <= 0.0:
        ap.error("--max-interp must be positive")

    try:
        path = resolve_input(args.input)
        layout = determine_layout(path, args.order, args.accel_unit, args.timestamp_unit, args.gyro_deg)
        nominal_dt, estimated_gyro_bias, startup_stationary, rows = inspect_stream(path, layout)
        gyro_bias = args.gyro_bias or estimated_gyro_bias
        accel_bias = args.accel_bias or (0.0, 0.0, 0.0)
        gravity = estimate_gravity(path, layout, nominal_dt, gyro_bias, accel_bias,
                                   startup_stationary, rows)
        with open(args.output, "w", encoding="ascii", newline="\n") as out:
            integrator = Integrator(out, gravity, gyro_bias, accel_bias, nominal_dt, args.max_interp)
            for sample in iter_samples(path, layout):
                integrator.process(sample)
        if integrator.output_rows != rows:
            raise RuntimeError("output row count does not match input data-row count")
        if not args.quiet:
            print("Input:", path, file=sys.stderr)
            print("Output:", args.output, file=sys.stderr)
            print("Input rows:", rows, "output poses:", integrator.output_rows, file=sys.stderr)
            print("Nominal dt: %.9f s; largest timestamp gap: %.6f s" %
                  (nominal_dt, integrator.max_gap), file=sys.stderr)
            print("Coasted intervals:", integrator.coast_steps,
                  "internal gap steps:", integrator.filled_steps, file=sys.stderr)
            print("Startup stationary calibration:", "yes" if startup_stationary else "no", file=sys.stderr)
            print("Estimated gravity:", "%.6f %.6f %.6f" % gravity, file=sys.stderr)
    except (OSError, ValueError, RuntimeError) as exc:
        print("Error:", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
