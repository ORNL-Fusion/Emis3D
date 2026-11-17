"""
tests/test_save_speed.py

Results files are written by the standard pickler when it can write them,
and by dill only when it cannot. dill's pickler runs in Python and takes
about a second per time on a machine with a couple of hundred channels,
which the per-time checkpoints paid after every time. Either way the file
is one that _load_bestFits (dill.load) reads.
"""

import pickle
import pickletools

import dill
import numpy as np
import pytest

from emis3d.emis3D import Emis3D
from emis3d.mock import CameraSpec, mock_workspace
from emis3d.tokamak import Tokamak


def big_tokamak(tmp_path):
    cams = [CameraSpec(f"C{i:02d}", 360.0 * i / 6, n_channels=16) for i in range(6)]
    return Tokamak(tokamakName="MOCK", mode="Build", loadBolometers=True, workspace=mock_workspace(tmp_path, cameras=cams))


def uses_dill(path):
    """Whether the file needs dill to be read: it refers to dill's own helpers."""
    with open(path, "rb") as f:
        return any(op.name in ("GLOBAL", "STACK_GLOBAL", "SHORT_BINUNICODE", "BINUNICODE") and isinstance(arg, str) and arg.startswith("dill")
                   for op, arg, _ in pickletools.genops(f))


def test_plain_results_are_written_by_the_standard_pickler(tmp_path):
    data = {"fit_data": {10.0: {"observed": [[1.0, 2.0]]}}, "bestFits": {10.0: {"radDist": big_tokamak(tmp_path), "tpf": np.float64(1.5)}}}
    path = tmp_path / "1_bestFits_10.0000.dill"
    Emis3D._dump_atomic(path, data)
    assert not uses_dill(path)
    with open(path, "rb") as f:
        assert len(pickle.load(f)["bestFits"][10.0]["radDist"].bolometers) == 6   # the standard loader reads it
    e = Emis3D(initialize=False)
    e._load_bestFits(path)                                                        # and so does Emis3D (dill.load)
    assert e.bestFits[10.0]["tpf"] == 1.5 and len(e.bestFits[10.0]["radDist"].bolometers) == 6
    assert e.fitData[10.0]["observed"] == [[1.0, 2.0]]
    assert [p.name for p in tmp_path.glob(".*.tmp")] == []


def test_what_the_standard_pickler_refuses_is_still_saved(tmp_path):
    def local_function(x):                                                        # a function defined inside another: dill only
        return 2 * x

    data = {"fit_data": {}, "bestFits": {10.0: {"scale": lambda phi: 3.0 * phi, "local": local_function}}}
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        pickle.dumps(data)
    path = tmp_path / "1_bestFits_10.0000.dill"
    Emis3D._dump_atomic(path, data)
    with open(path, "rb") as f:
        back = dill.load(f)["bestFits"][10.0]
    assert back["scale"](2.0) == 6.0 and back["local"](2.0) == 4.0
    assert [p.name for p in tmp_path.glob(".*.tmp")] == []


def test_the_standard_pickler_is_much_faster_on_a_tokamak(tmp_path):
    """The reason for the change, with a wide margin so it does not depend on the machine running the test."""
    import time

    tok = big_tokamak(tmp_path)
    t0 = time.perf_counter()
    plain = pickle.dumps(tok, protocol=pickle.HIGHEST_PROTOCOL)
    t_pickle = time.perf_counter() - t0
    t0 = time.perf_counter()
    dilled = dill.dumps(tok)
    t_dill = time.perf_counter() - t0
    assert t_pickle < 0.5 * t_dill
    assert len(dill.loads(plain).bolometers) == len(dill.loads(dilled).bolometers) == 6
