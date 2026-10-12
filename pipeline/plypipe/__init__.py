"""Ply capture pipeline.

Stages (each resumable, see plypipe.cli):
    frames -> sfm -> align -> occupancy -> train -> export

Output frame ("scene frame"), shared with the web viewer:
    right-handed, +Y up, floor plane at y = 0, units = metres.
"""

__version__ = "0.1.0"

STAGES = ("frames", "sfm", "align", "occupancy", "train", "export")
