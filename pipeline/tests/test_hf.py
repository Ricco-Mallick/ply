# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Results storage on Hugging Face without the network: a stand-in huggingface_hub that records what it is asked."""

from __future__ import annotations

import sys
import types

import pytest

from plypipe import hf


@pytest.fixture
def hub(monkeypatch):
    calls: list = []

    class HfApi:
        def __init__(self, token=None):
            calls.append(("token", token))

        def create_commit(self, repo_id, operations, *, commit_message, repo_type=None):
            calls.append(("commit", repo_id, repo_type, [(o.path_in_repo, o.path_or_fileobj) for o in operations], commit_message))

        def upload_file(self, **kw):
            calls.append(("upload_file", kw["path_in_repo"]))

    class CommitOperationAdd:
        def __init__(self, path_in_repo, path_or_fileobj):
            self.path_in_repo, self.path_or_fileobj = path_in_repo, path_or_fileobj

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=HfApi, CommitOperationAdd=CommitOperationAdd))
    monkeypatch.setenv("PLY_HF_REPO", "owner/ply-data")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    return calls


def test_upload_many_is_one_commit_for_every_file(hub, tmp_path):
    files = [(f"scenes/j1/evidence/frames/{i:03d}.jpg", tmp_path / f"{i}.jpg") for i in range(48)]
    files.append(("scenes/j1/evidence/poses.json", tmp_path / "poses.json"))
    hf.upload_many(files)
    commits = [c for c in hub if c[0] != "token"]
    assert len(commits) == 1 and commits[0][:3] == ("commit", "owner/ply-data", "dataset")
    assert commits[0][3] == [(key, str(path)) for key, path in files]
    assert commits[0][4] == "add 49 files"


def test_upload_many_names_the_commit(hub, tmp_path):
    hf.upload_many([("scenes/j1/a.json", tmp_path / "a.json")], message="evidence for j1")
    assert hub[-1][4] == "evidence for j1"


def test_upload_many_without_configuration_says_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("PLY_HF_REPO", raising=False)
    with pytest.raises(hf.Missing):
        hf.upload_many([("scenes/j1/a.json", tmp_path / "a.json")])


def test_upload_many_without_the_library_says_missing(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)   # an import of it raises ImportError
    monkeypatch.setenv("PLY_HF_REPO", "owner/ply-data")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    with pytest.raises(hf.Missing):
        hf.upload_many([("scenes/j1/a.json", tmp_path / "a.json")])
