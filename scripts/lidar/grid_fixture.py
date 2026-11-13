"""The app's wall-crossing test needs the real walking grid: write a twin folder's occupancy as the test fixture it reads.

    python scripts/lidar/grid_fixture.py <twin folder> app/test/fixtures/lidar-office-8b --source <url of the twin>

Reads occupancy.json and occupancy.png (and scene.json, places.json and the walk check inside occupancy.json when the
twin has them) and writes <stem>.json and <stem>.bin. The .bin is the grid exactly as the viewer reads it: one byte a
cell, 255 free, 0 blocked, 127 unknown, row-major, row = +z, column = +x, top-left cell at the meta's origin. The .json
carries the meta, the spawn and camera path in xz, the places in xz, the counts, and the SHA-256 of the
bytes, so the test can prove it is walking the grid the twin published and not a copy that drifted.

    node --experimental-strip-types app/test/walk-lidar.test.mts     # walks it
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def fixture(folder: Path, out_stem: Path, source: str = "") -> dict:
    meta = json.loads((folder / "occupancy.json").read_text())
    grid = cv2.imread(str(folder / meta.get("image", "occupancy.png")), cv2.IMREAD_GRAYSCALE)
    if grid is None:
        raise SystemExit(f"{folder}: no {meta.get('image', 'occupancy.png')}")
    if grid.shape != (meta["height"], meta["width"]):
        raise SystemExit(f"{folder}: the image is {grid.shape}, the meta says {meta['height']} x {meta['width']}")
    bad = set(np.unique(grid).tolist()) - {0, 127, 255}
    if bad:
        raise SystemExit(f"{folder}: the image holds values outside 0/127/255: {sorted(bad)}")
    scene = json.loads((folder / "scene.json").read_text()) if (folder / "scene.json").exists() else {}
    places = []
    if (folder / "places.json").exists():
        places = [pl["p"][::2] for pl in json.loads((folder / "places.json").read_text()).get("places", []) if pl.get("p")]
    spawn = (scene.get("spawn") or {}).get("position") or [0, 0, 0]
    path = [[p[0], p[2]] for p in scene.get("cameraPath") or []]
    raw = grid.tobytes()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_stem.with_suffix(".bin").write_bytes(raw)
    doc = {"scene": out_stem.name, "source": source, "written": datetime.date.today().isoformat(),
           "cell": meta["cellSize"], "origin": meta["origin"], "width": meta["width"], "height": meta["height"],
           "values": meta["values"], "counts": meta["stats"], "walk": meta.get("walk"),
           "spawn": [round(float(spawn[0]), 3), round(float(spawn[2]), 3)], "cameraPath": path, "places": places,
           "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    out_stem.with_suffix(".json").write_text(json.dumps(doc, indent=1))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--source", default="")
    a = ap.parse_args(argv)
    doc = fixture(a.folder, a.out, a.source)
    print(json.dumps({k: doc[k] for k in ("cell", "width", "height", "counts", "sha256", "bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
