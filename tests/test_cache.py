import copy
from dataclasses import asdict

import numpy as np
import pytest

from wiloc.cache import cache_key, cached_maps
from wiloc.environment import FloorPlan
from wiloc.radio import RadioConfig

FAST = dict(spacing=2.0, samples=5, map_method="model")


@pytest.fixture
def fp():
    return FloorPlan.builtin("office")


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("WILOC_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_office_radio_section_matches_defaults(fp):
    # the venue file is now the source of truth; for the office it keeps the
    # values every earlier result was produced with
    assert RadioConfig.from_dict(fp.radio) == RadioConfig()


def test_from_dict_overrides_and_rejects_typos():
    cfg = RadioConfig.from_dict({"seed": 3, "shadowing_sigma_db": 4}, seed=None, temporal_sigma_db=2.0)
    assert (cfg.seed, cfg.shadowing_sigma_db, cfg.temporal_sigma_db) == (3, 4.0, 2.0)
    assert isinstance(cfg.shadowing_sigma_db, float)
    with pytest.raises(ValueError, match="shadowing_sigma"):
        RadioConfig.from_dict({"shadowing_sigma": 4.0})


def test_cache_hit_returns_the_same_maps(fp, tmp_cache):
    cfg = RadioConfig()
    r1, d1, hit1 = cached_maps(fp, cfg, map_method="gp", spacing=2.0, samples=5)
    r2, d2, hit2 = cached_maps(fp, cfg, map_method="gp", spacing=2.0, samples=5)
    assert (hit1, hit2) == (False, True)
    assert len(list(tmp_cache.glob("*.npz"))) == 1
    for a, b in ((r1.mean, r2.mean), (r1.positions, r2.positions), (d1.mean, d2.mean), (d1.std, d2.std)):
        np.testing.assert_array_equal(a, b)
    assert r2.ap_ids == r1.ap_ids and d2.ap_ids == d1.ap_ids
    assert (d2.res, d2.width, d2.height) == (d1.res, d1.width, d1.height)
    pts = np.array([[3.0, 3.0], [12.5, 6.0]])
    np.testing.assert_allclose(d1.lookup(pts)[0], d2.lookup(pts)[0])


def test_key_changes_with_everything_the_maps_depend_on(fp):
    cfg = RadioConfig()
    base = cache_key(fp, cfg, **FAST)
    moved = copy.deepcopy(fp)
    moved.access_points[0].x += 0.5
    walls = copy.deepcopy(fp)
    walls.wall_materials[5] = "concrete"
    assert cache_key(moved, cfg, **FAST) != base
    assert cache_key(walls, cfg, **FAST) != base
    assert cache_key(fp, RadioConfig(**{**asdict(cfg), "seed": 8}), **FAST) != base
    assert cache_key(fp, cfg, **{**FAST, "samples": 6}) != base
    assert cache_key(copy.deepcopy(fp), RadioConfig(), **FAST) == base


def test_unreadable_cache_file_is_rebuilt(fp, tmp_cache):
    cached_maps(fp, RadioConfig(), **FAST)
    (path,) = tmp_cache.glob("*.npz")
    path.write_bytes(b"not a zip file")
    rmap, dense, hit = cached_maps(fp, RadioConfig(), **FAST)
    assert not hit and rmap.n_aps == len(fp.access_points)
    _, _, hit = cached_maps(fp, RadioConfig(), **FAST)
    assert hit
