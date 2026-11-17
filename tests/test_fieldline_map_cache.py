"""
tests/test_fieldline_map_cache.py

The field-line mapping cache (fieldline_map.get_or_build). A mapping's file
name holds the matrix's name, the equilibrium, the source plane's angle and
the transits, but not substeps, psi_max or the matrix's grid. A cached
mapping built with other values of those used to be returned as it was, so
changing substeps or psi_max in a run config, or rebuilding the matrix
under the same name, silently had no effect. It is now rebuilt.

The tracing itself (FieldLineMap.build) is replaced by a stub that records
what it was asked for.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from emis3d import fieldline_map as FM
from emis3d import mock
from emis3d.geometry import GridSpec
from emis3d.workspace import Workspace

SETTINGS = dict(machine="MOCK", matrix_name="low", equilibrium=mock.EQ_FILE, phi0_deg=225.0, transits=1)


def matrix(n_phi=8):
    grid = GridSpec(np.linspace(1.0, 2.4, 8), np.linspace(-1.1, 1.1, 12), n_phi)
    return SimpleNamespace(grid=grid, machine="MOCK")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """(workspace, list of the build calls): get_or_build with the tracing stubbed out."""
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    calls = []

    def build(G, gfile, phi0, matrix_name, equilibrium, transits=1, substeps=2, psi_norm=None, psi_max=None, n_tor=None, progress=None,
              mirror_helicity=False):
        calls.append({"substeps": substeps, "psi_max": psi_max, "n_phi": G.grid.n_phi})
        g = G.grid
        return FM.FieldLineMap(machine=G.machine, matrix_name=matrix_name, equilibrium=str(equilibrium), phi0=float(phi0), transits=int(transits),
                               substeps=int(substeps), n_voxels=g.n_voxels, n_phi=g.n_phi, phi_centers=g.phi_centers,
                               sector_of_voxel=np.zeros(g.n_voxels, dtype=int), ir=np.zeros(1, dtype=int), iz=np.zeros(1, dtype=int),
                               r_centers=g.r_centers, z_centers=g.z_centers, passes=[], provenance={"psi_max": psi_max})

    monkeypatch.setattr(FM.FieldLineMap, "build", staticmethod(build))
    monkeypatch.setattr(FM, "_LOADED", {})
    return ws, calls


def get(ws, G=None, **kw):
    return FM.get_or_build(ws, G or matrix(), **SETTINGS, **kw)


def test_same_settings_reuse_the_cached_mapping(cache):
    ws, calls = cache
    first, path = get(ws, substeps=2)
    again, path2 = get(ws, substeps=2)
    assert len(calls) == 1 and again is first and path2 == path and path.is_file()
    FM._LOADED.clear()                                                        # a new process: read from the file
    from_file, _ = get(ws, substeps=2)
    assert len(calls) == 1 and from_file.substeps == 2


def test_other_substeps_rebuild_the_mapping(cache):
    ws, calls = cache
    _, path = get(ws, substeps=1)
    fmap, path2 = get(ws, substeps=3)
    assert [c["substeps"] for c in calls] == [1, 3] and fmap.substeps == 3 and path2 == path
    assert FM.FieldLineMap.load(path).substeps == 3                           # the file was replaced
    assert FM.cached_map(path) is fmap                                        # and so was the copy held in memory
    get(ws, substeps=3)
    assert len(calls) == 2                                                    # the rebuilt one is reused


def test_other_psi_max_rebuilds_the_mapping(cache):
    ws, calls = cache
    get(ws)
    get(ws, psi_max=1.05)
    get(ws, psi_max=1.05)
    get(ws, psi_max=1.1)
    get(ws)
    assert [c["psi_max"] for c in calls] == [None, 1.05, 1.1, None]


def test_a_rebuilt_matrix_rebuilds_the_mapping(cache):
    ws, calls = cache
    get(ws, matrix(n_phi=8))
    fmap, _ = get(ws, matrix(n_phi=16))                                        # same name, another grid
    assert [c["n_phi"] for c in calls] == [8, 16] and fmap.n_phi == 16


def test_the_rebuild_is_reported(cache):
    ws, _ = cache
    get(ws, substeps=1)
    lines = []
    get(ws, substeps=2, progress=lines.append)
    assert any("built with substeps 1, not 2" in line and "rebuilding" in line for line in lines)


def test_the_rebuild_is_a_warning_and_a_reused_mapping_is_silent(cache, caplog):
    """Without logging set up only warnings are shown, and a rebuild is minutes of silence otherwise."""
    ws, _ = cache
    with caplog.at_level("WARNING", logger="emis3d.fieldline_map"):
        get(ws, substeps=1)                                                   # built for the first time: not a warning
        get(ws, substeps=1)                                                   # reused
        assert not caplog.records
        get(ws, substeps=2)
    (record,) = caplog.records
    assert record.levelname == "WARNING" and "built with substeps 1, not 2" in record.getMessage() and "minutes" in record.getMessage()


def test_stale_reason_for_a_matching_mapping_is_none(cache):
    ws, _ = cache
    fmap, _ = get(ws, psi_max=1.05)
    assert FM.stale_reason(fmap, matrix(), np.deg2rad(225.0), 2, 1.05) is None
    assert "source plane" in FM.stale_reason(fmap, matrix(), np.deg2rad(225.0000004), 2, 1.05)
