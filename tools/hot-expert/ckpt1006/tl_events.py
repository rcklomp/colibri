#!/usr/bin/env python3
"""tl_events.py -- list the events of one layer of one chunk in a --timeline CSV (glm5 timeline, GLM5.md section 14), or every event that ENDS inside a window.
Times are printed in ms from the start of layer --origin-layer (default 2) of the chunk on card 0, so the rows of different cards line up.
Used by record §L5-PF12 point 3 (what layer 3 of a staged chunk waits behind).
  python3 -I tl_events.py CSV --chunk 4 --layer 3 [--classes stage_copy,ring_wait,own_ready_wait]
  python3 -I tl_events.py CSV --chunk 4 --ending 20560 20660          (all chunks, window in ms from the origin)
"""
import argparse, sys

def load(path):
    rows = []
    for ln in open(path, errors="replace"):
        if ln.startswith("#") or ln.startswith("chunk,"):
            continue
        f = ln.rstrip("\n").split(",")
        rows.append(dict(chunk=int(f[0]), layer=int(f[1]), card=int(f[2]), cls=f[3], t0=float(f[4]), t1=float(f[5]),
                         owner=int(f[6]), bytes=int(f[7]), region=f[8], batch=f[9], stream=f[10]))
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv"); ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--layer", type=int); ap.add_argument("--classes", default="")
    ap.add_argument("--origin-layer", type=int, default=2)
    ap.add_argument("--ending", type=float, nargs=2, metavar=("FROM_MS", "TO_MS"))
    a = ap.parse_args()
    rows = load(a.csv)
    org = [r["t0"] for r in rows if r["chunk"] == a.chunk and r["layer"] == a.origin_layer and r["card"] == 0]
    if not org:
        sys.exit("no card-0 event for chunk %d layer %d" % (a.chunk, a.origin_layer))
    base = min(org); T = lambda x: (x - base) / 1e3
    keep = set(c for c in a.classes.split(",") if c)
    if a.ending:
        sel = [r for r in rows if a.ending[0] <= T(r["t1"]) <= a.ending[1]]
    elif a.layer is not None:
        sel = [r for r in rows if r["chunk"] == a.chunk and r["layer"] == a.layer]
    else:
        sys.exit("give --layer or --ending")
    if keep:
        sel = [r for r in sel if r["cls"] in keep]
    for r in sorted(sel, key=lambda r: (r["t0"], r["card"])):
        print("chunk%d L%-2d card%d %-16s owner%d %10.1f..%10.1f ms (%8.1f) bytes=%d region=%s batch=%s stream=%s" % (
            r["chunk"], r["layer"], r["card"], r["cls"], r["owner"], T(r["t0"]), T(r["t1"]), (r["t1"] - r["t0"]) / 1e3,
            r["bytes"], r["region"], r["batch"], r["stream"]))

if __name__ == "__main__":
    main()
