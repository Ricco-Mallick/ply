# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Compare a rendered pose with the photograph taken there: where is the model wrong, and by how much.

  python scripts/compare.py <room-dir> <pose-index>

Prints a coarse per-region difference (3x3 grid, mean colour distance 0..1) and the worst region, so the next
modelling pass has one concrete thing to fix. Not a beauty metric: a low score means the geometry and colours sit
where the photograph has them.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load(path: Path, size=(320, 180)) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(size), dtype=np.float32) / 255.0


def main() -> int:
    room = Path(sys.argv[1])
    index = sys.argv[2] if len(sys.argv) > 2 else "0"
    poses = json.loads((room / "evidence" / "poses.json").read_text())["frames"]
    pose = poses[int(index)]
    photo = room / "evidence" / "frames" / Path(pose["file"]).name
    render = room / "compare" / f"pose-{int(index):03d}-render.png"
    if not photo.exists() or not render.exists():
        print(f"missing {photo if not photo.exists() else render}")
        return 2
    a, b = load(photo), load(render)
    diff = np.abs(a - b).mean(axis=2)
    H, W = diff.shape
    rows, cols = 3, 3
    grid = [[float(diff[r * H // rows:(r + 1) * H // rows, c * W // cols:(c + 1) * W // cols].mean()) for c in range(cols)] for r in range(rows)]
    print(f"pose {index}: overall difference {diff.mean():.3f}  (0 = identical framing and colour)")
    names = ["top", "middle", "bottom"]
    sides = ["left", "centre", "right"]
    worst = max(((grid[r][c], names[r], sides[c]) for r in range(rows) for c in range(cols)))
    for r in range(rows):
        print("  " + "  ".join(f"{grid[r][c]:.2f}" for c in range(cols)) + f"   {names[r]}")
    print(f"  worst region: {worst[1]} {worst[2]} ({worst[0]:.2f}) — check that part of the model against the photo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
