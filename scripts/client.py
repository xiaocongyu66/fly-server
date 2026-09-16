#!/usr/bin/env python3
"""fly-server visual client: video frames -> ME neuron current injection.

The sensory encoder lives client-side: ffmpeg decodes frames to an 8x8
grayscale grid, buckets of ME neurons (sorted by root id) receive current
proportional to brightness. 30 fps video -> 66 ticks (33 ms) per frame.

Usage:
  client.py --video FILE --label A [--invert] [--seconds 6] [--port 8321]
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request

GRID = 8  # 8x8 buckets
LAYERS = 4  # brightness layers, one observe request per layer
FPS = 30
TICKS_PER_FRAME = 66  # 33 ms at dt=0.5 ms
DT_MS = 0.5


def api(base, method, path, body=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None),
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def frames(path, seconds, invert):
    """Yield per-frame 64-int grayscale lists via ffmpeg rawvideo."""
    w = h = GRID
    cmd = [
        "ffmpeg", "-loglevel", "error", "-t", str(seconds), "-i", path,
        "-vf", f"scale={w}:{h}", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = w * h
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        if invert:
            yield [255 - b for b in buf]
        else:
            yield list(buf)
    proc.terminate()


def encode_frame(frame, me_ids):
    """Population-rate code: per bucket, inject a fraction of neurons equal
    to brightness (dark=0%, bright=100%). Spike count tracks the frame's
    luminance distribution, so inverted video yields a different response."""
    n = len(me_ids)
    per = n // (GRID * GRID)
    buckets = [me_ids[i * per:(i + 1) * per] for i in range(GRID * GRID)]
    # group 16 buckets (one video quadrant) per request to bound request count
    reqs = []
    for q in range(4):
        ids = []
        avg = 0
        for j in range(16):
            b = frame[q * 16 + j]
            avg += b
            k = int(len(buckets[q * 16 + j]) * b / 255)
            ids.extend(buckets[q * 16 + j][:k])
        if ids and avg / 16 >= 20:
            reqs.append({"modality": "visual", "target": {"ids": ids}, "current": 35.0})
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8321")
    ap.add_argument("--video", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--seconds", type=int, default=6)
    args = ap.parse_args()

    models = api(args.base, "GET", "/v1/models")
    substrate_id = models["data"][0]["id"]

    sess = api(args.base, "POST", "/v1/sessions", {
        "substrate": substrate_id,
        "metadata": {"client": args.label, "video": args.video, "invert": str(args.invert)},
    })
    sid = sess["id"]

    # ME region neurons sorted by root id -> pseudo-retinotopic buckets
    # (server has no region-list endpoint yet; ask via a probe observe is
    # wasteful, so request the full ME set once through a big selector and
    # read the count from the item body is not possible either. Instead we
    # keep a server-side trick: ids selection works on root ids, so we need
    # the list. We fetch it from the dataset directly.)
    import gzip
    me_ids = []
    with gzip.open("data/neurons.csv.gz", "rt") as f:
        header = f.readline().strip().split(",")
        i_id, i_region = header.index("Root ID"), header.index("Top in/out region")
        for line in f:
            cols = line.rstrip("\n").split(",")
            if len(cols) > i_region and cols[i_region] == "ME":
                me_ids.append(int(cols[i_id]))
    me_ids.sort()
    if not me_ids:
        sys.exit("no ME neurons found in data/neurons.csv.gz")

    spikes_per_frame = []
    t0 = time.time()
    n_frames = 0
    r = None
    for frame in frames(args.video, args.seconds, args.invert):
        for req in encode_frame(frame, me_ids):
            api(args.base, "POST", f"/v1/sessions/{sid}/observe", req)
        # tick 1 captures the injection response (n_spikes is the *last*
        # tick's count server-side); remaining ticks advance the timeline.
        r = api(args.base, "POST", f"/v1/sessions/{sid}/step", {"steps": 1})
        spikes_per_frame.append(r["n_spikes"])
        if TICKS_PER_FRAME > 1:
            r = api(args.base, "POST", f"/v1/sessions/{sid}/step", {"steps": TICKS_PER_FRAME - 1})
        n_frames += 1
    wall = time.time() - t0

    result = {
        "label": args.label,
        "session_id": sid,
        "invert": args.invert,
        "frames": n_frames,
        "ticks": r["tick"] if r else 0,
        "wall_s": round(wall, 2),
        "spikes_per_frame": spikes_per_frame,
        "total_spikes": sum(spikes_per_frame),
        "mean_spikes": round(sum(spikes_per_frame) / max(1, n_frames), 1),
        "usage": r["usage"] if r else {},
    }
    print(json.dumps(result, ensure_ascii=False))
    with open(f"/tmp/flyexp_{args.label}.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
