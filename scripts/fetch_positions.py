#!/usr/bin/env python3
"""Fetch soma positions from neuPrint (malecns) -> positions.csv (root_id,x,y,z).

Requires a neuPrint auth token (neuprint.janelia.org -> account -> token).
Usage: NEUPRINT_TOKEN=xxx python3 scripts/fetch_positions.py > data/positions.csv
"""
import json, os, sys, urllib.request

TOKEN = os.environ.get("NEUPRINT_TOKEN")
if not TOKEN:
    sys.exit("set NEUPRINT_TOKEN (neuprint.janelia.org -> account page)")

DATASET = os.environ.get("NEUPRINT_DATASET", "malecns")
URL = "https://neuprint.janelia.org/api/cypher"

QUERY = """
MATCH (n:Neuron) WHERE n.somaLocation IS NOT NULL
RETURN n.bodyId AS bodyId, n.somaLocation AS loc
"""

def main():
    body = {"cypher": QUERY, "dataset": DATASET}
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    rows = d.get("data", [])
    print("root_id,x,y,z", flush=True)
    n = 0
    for body_id, loc in rows:
        if not loc or len(loc) < 3:
            continue
        x, y, z = loc[0], loc[1], loc[2]
        print(f"{body_id},{x},{y},{z}", flush=True)
        n += 1
    print(f"fetched {n} positions", file=sys.stderr)

if __name__ == "__main__":
    main()
