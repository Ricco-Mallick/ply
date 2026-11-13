import os
import pathlib

import pytest

from plypipe.paths import import_root


def test_a_checkout_is_usable(tmp_path):
    (tmp_path / "plypipe").mkdir()
    assert import_root(tmp_path) is True


def test_a_directory_without_the_marker_is_not(tmp_path):
    assert import_root(tmp_path) is False


def test_missing_and_unreadable_paths_do_not_raise(tmp_path):
    assert import_root(pathlib.Path("/definitely/not/here")) is False
    if os.geteuid() == 0:
        pytest.skip("root can traverse anything, the case cannot be built")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "plypipe").mkdir()
    locked.chmod(0o000)
    try:
        assert import_root(locked) is False, "an unreadable parent must read as not usable, not raise"
    finally:
        locked.chmod(0o700)
