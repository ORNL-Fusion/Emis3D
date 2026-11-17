"""
tests/test_raddist_file_order.py

radDist files are loaded in the order of their sorted paths. They used to
come in the order the file system listed them (os.walk), which differs
between machines and file systems, so the same library could number its
radDists differently on a laptop and on a cluster, and a tie between two
equally good radDists could go either way.
"""

import os

from emis3d import util
from emis3d.emis3D import Emis3D
from test_calibration import write_workspace


real_walk = os.walk


def scrambled_walk(top):
    """os.walk, but with every listing reversed: an order a file system may well give."""
    for root, dirs, files in real_walk(top):
        dirs[:] = sorted(dirs, reverse=True)
        yield root, dirs, sorted(files, reverse=True)


def make_tree(tmp_path):
    for name in ("b/R_2.json", "a/R_10.json", "a/R_1.json", "c.json", "a/.hidden.json", "a/config.yaml", "b/sub/R_0.json"):
        f = tmp_path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}")
    return [str(tmp_path / n) for n in ("a/R_1.json", "a/R_10.json", "b/R_2.json", "b/sub/R_0.json", "c.json")]


def test_files_come_back_sorted_whatever_the_file_system_lists(tmp_path, monkeypatch):
    expected = make_tree(tmp_path)
    assert util.get_filenames_in_directory(tmp_path) == expected               # hidden and yaml files left out, as before
    monkeypatch.setattr(util.os, "walk", scrambled_walk)
    assert util.get_filenames_in_directory(tmp_path) == expected


def test_raddists_are_numbered_in_sorted_order(tmp_path, monkeypatch):
    run = write_workspace(tmp_path, tstart=10.0)
    monkeypatch.setattr(util.os, "walk", scrambled_walk)
    e = Emis3D(runConfigName=str(run))
    loaded = [os.path.basename(e.data["synthetic"]["locIndependent"][k].info["radDistPath"]) for k in sorted(e.data["synthetic"]["locIndependent"])]
    assert loaded == ["R_0.json", "R_1.json", "R_2.json"]
