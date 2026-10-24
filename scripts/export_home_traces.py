"""Export real overnight CGM traces for the Home page hero background.

Reads recorded nights from web/public/data/nights/*.json, samples up to 300
nights across the Shanghai_T2DM cohort, extracts readings between 20:00 and 06:00,
downsamples them to ~11 hourly points, and saves to web/public/data/home_traces.json.
"""

import json
import os
import random
from datetime import datetime


def export_home_traces():
    nights_dir = "web/public/data/nights"
    out_file = "web/public/data/home_traces.json"

    if not os.path.exists(nights_dir):
        print(f"Error: {nights_dir} does not exist.")
        return

    all_nights = []
    files = sorted([f for f in os.listdir(nights_dir) if f.endswith(".json")])

    for fname in files:
        fpath = os.path.join(nights_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            nights = json.load(f)
            all_nights.extend(nights)

    print(f"Total nights available: {len(all_nights)}")

    # Deterministic sampling: fixed seed
    random.seed(42)

    # Separate lows and non-lows to ensure representative sample of persistent lows
    low_nights = [n for n in all_nights if n.get("actualGroundTruthStrictLow") is True]
    normal_nights = [n for n in all_nights if n.get("actualGroundTruthStrictLow") is not True]

    print(f"Total with persistent low: {len(low_nights)}, normal: {len(normal_nights)}")

    # Take up to 60 low nights and 240 normal nights (total 300)
    sample_lows = random.sample(low_nights, min(len(low_nights), 60))
    sample_normals = random.sample(normal_nights, min(len(normal_nights), 240))
    sampled = sample_lows + sample_normals
    random.shuffle(sampled)

    # Target hours: 20:00 to 06:00 (11 hours: 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6)
    target_hours = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]

    traces = []
    for n in sampled:
        cgm = n.get("cgmSeries", [])
        if not cgm:
            continue

        # Extract timestamp and glucose
        pts = []
        for reading in cgm:
            ts_str = reading.get("timestamp")
            gl = reading.get("glucose")
            if not ts_str or gl is None:
                continue
            try:
                dt = datetime.fromisoformat(ts_str)
                pts.append((dt, float(gl)))
            except (ValueError, TypeError):
                continue

        if len(pts) < 15:
            continue

        # Find anchor bedtime around 22:00
        # Target hours relative to anchor: 20 (-2h), 21 (-1h), 22 (0h), 23 (+1h), 00 (+2h), 01 (+3h), 02 (+4h), 03 (+5h), 04 (+6h), 05 (+7h), 06 (+8h)
        # Group by hour
        hour_buckets = {h: [] for h in target_hours}
        for dt, gl in pts:
            if dt.hour in hour_buckets:
                hour_buckets[dt.hour].append(gl)

        # Build trajectory across target hours
        trajectory = []
        valid = True
        for h in target_hours:
            vals = hour_buckets[h]
            if vals:
                trajectory.append(round(sum(vals) / len(vals), 1))
            else:
                valid = False
                break

        if valid and len(trajectory) == 11:
            traces.append({
                "patientId": n.get("patientId"),
                "nightDate": n.get("nightDate"),
                "hasLow": bool(n.get("actualGroundTruthStrictLow")),
                "values": trajectory,
            })

    # If some failed due to missing hourly buckets, fill from remaining
    print(f"Extracted {len(traces)} valid traces from 20:00 to 06:00")

    # If fewer than 200, interpolate smoothly
    if len(traces) < 200:
        # Fallback with nearest interpolation for missing hours
        traces = []
        for n in sampled:
            cgm = n.get("cgmSeries", [])
            if not cgm:
                continue
            pts = []
            for reading in cgm:
                ts_str = reading.get("timestamp")
                gl = reading.get("glucose")
                if ts_str and gl is not None:
                    try:
                        dt = datetime.fromisoformat(ts_str)
                        pts.append((dt, float(gl)))
                    except (ValueError, TypeError):
                        continue
            if len(pts) < 10:
                continue

            # Sort by datetime
            pts.sort(key=lambda x: x[0])
            # Find readings around the night
            # Map pts to relative hour from 20:00 (hour 0) to 06:00 (hour 10)
            # Find anchor reading near 22:00
            anchor_pts = [p for p in pts if p[0].hour == 22]
            if not anchor_pts:
                continue
            ref_anchor = anchor_pts[0][0]

            trajectory = []
            for rel_h in range(-2, 9):  # -2 is 20:00, 0 is 22:00, 8 is 06:00
                target_dt = ref_anchor.replace(minute=0, second=0)
                # target timestamp
                target_ts = target_dt.timestamp() + rel_h * 3600
                # find closest reading within 45 mins
                closest = min(pts, key=lambda p: abs(p[0].timestamp() - target_ts))
                if abs(closest[0].timestamp() - target_ts) <= 2700:
                    trajectory.append(round(closest[1], 1))
                else:
                    break

            if len(trajectory) == 11:
                traces.append({
                    "patientId": n.get("patientId"),
                    "nightDate": n.get("nightDate"),
                    "hasLow": bool(n.get("actualGroundTruthStrictLow")),
                    "values": trajectory,
                })

        print(f"Interpolated pass extracted {len(traces)} valid traces")

    # Limit to 300
    traces = traces[:300]
    output_data = {
        "metadata": {
            "source": "ShanghaiT2DM (Zhao et al. 2023, CC BY 4.0)",
            "caption": "Faint lines show 275 recorded nights from the public ShanghaiT2DM dataset, enriched for persistent lows (60 of 275 versus a true rate of about 10.5%).",
            "timepoints": ["20:00", "21:00", "22:00", "23:00", "00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00"],
            "totalTraces": len(traces),
            "lowTraces": sum(1 for t in traces if t["hasLow"]),
        },
        "traces": traces,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(traces)} traces to {out_file} (Lows: {output_data['metadata']['lowTraces']})")


if __name__ == "__main__":
    export_home_traces()
