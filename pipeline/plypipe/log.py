"""JSON-lines progress log: one object per line on stdout and in <out>/log.jsonl.

A worker can tail either stream. Fields: t (ISO time), stage, event
(start | progress | info | warn | done | skip | error), msg, plus any extra data.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


class Log:
    def __init__(self, out_dir: Path | None = None, quiet: bool = False):
        self.path = Path(out_dir) / "log.jsonl" if out_dir else None
        self.quiet = quiet
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, event: str, msg: str = "", **data: Any) -> None:
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": stage, "event": event, "msg": msg, **data}
        line = json.dumps(rec, default=_default)
        if not self.quiet:
            print(line, flush=True)
        if self.path:
            with self.path.open("a") as f:
                f.write(line + "\n")


def _default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
