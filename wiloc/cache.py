"""On-disk cache of the localiser's radio maps (survey + dense GP map).

Building the maps means a simulated site survey plus one Gaussian-process fit
per access point: a few seconds for the office, longer for bigger venues.
The result depends only on the venue, its access points, the radio settings
and the survey / map parameters, so it is cached under a hash of exactly
those.  Changing any of them (or ``CACHE_VERSION``, when the map code
changes) gives a new key and a fresh build; a stale map is never reused.

    rmap, dense, hit = cached_maps(fp, cfg)

The cache lives in ``$WILOC_CACHE_DIR`` (default ``~/.cache/wiloc``).  Files
are written atomically, and an unreadable file is rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .environment import FloorPlan
from .fingerprint import DenseRadioMap, RadioMap
from .radio import RadioConfig, RadioSimulator

CACHE_VERSION = 1


def cache_dir() -> Path:
    return Path(os.environ.get("WILOC_CACHE_DIR") or Path.home() / ".cache" / "wiloc")


def cache_key(fp: FloorPlan, cfg: RadioConfig, **params) -> str:
    """Hash of everything the radio maps depend on."""
    desc = {
        "version": CACHE_VERSION,
        "size": [fp.width, fp.height],
        "walls": np.round(np.asarray(fp.walls, dtype=float), 4).tolist(),
        "wall_materials": list(fp.wall_materials),
        "losses": {m: float(v) for m, v in sorted(fp.materials.items())},
        "aps": [[ap.id, ap.x, ap.y, ap.tx_power, ap.channel] for ap in fp.access_points],
        "radio": asdict(cfg),
        "params": params,
    }
    return hashlib.sha1(json.dumps(desc, sort_keys=True).encode()).hexdigest()


def build_maps(fp, cfg, spacing=1.0, samples=20, map_method="gp", res=0.25, min_std=1.5, survey_seed=123):
    sim = RadioSimulator(fp, cfg)
    rmap = RadioMap.survey(sim, fp.grid(float(spacing)), int(samples), np.random.default_rng(survey_seed))
    dense = DenseRadioMap.from_radio_map(rmap, fp, res=res, method=map_method, min_std=min_std)
    return rmap, dense


def _save(path: Path, rmap: RadioMap, dense: DenseRadioMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(
            tmp,
            positions=rmap.positions,
            mean=rmap.mean,
            std=rmap.std,
            detect_rate=rmap.detect_rate,
            ap_ids=np.array(rmap.ap_ids),
            ap_positions=rmap.ap_positions,
            dense_res=np.array(dense.res),
            dense_mean=dense.mean,
            dense_std=dense.std,
            dense_size=np.array([dense.width, dense.height]),
        )
        os.replace(tmp, path)  # atomic: a reader never sees half a file
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _load(path: Path):
    with np.load(path, allow_pickle=False) as d:
        ap_ids = [str(s) for s in d["ap_ids"]]
        rmap = RadioMap(d["positions"], d["mean"], d["std"], d["detect_rate"], ap_ids, d["ap_positions"])
        w, h = (float(v) for v in d["dense_size"])
        dense = DenseRadioMap(float(d["dense_res"]), d["dense_mean"], d["dense_std"], list(ap_ids), w, h)
    return rmap, dense


def cached_maps(
    fp: FloorPlan,
    cfg: RadioConfig,
    spacing: float = 1.0,
    samples: int = 20,
    map_method: str = "gp",
    res: float = 0.25,
    min_std: float = 1.5,
    survey_seed: int = 123,
    use_cache: bool = True,
):
    """(RadioMap, DenseRadioMap, cache_hit) for this venue and radio config."""
    params = dict(spacing=float(spacing), samples=int(samples), map_method=map_method, res=float(res),
                  min_std=float(min_std), survey_seed=int(survey_seed))
    path = cache_dir() / f"{fp.name}-{cache_key(fp, cfg, **params)[:16]}.npz"
    if use_cache and path.exists():
        try:
            rmap, dense = _load(path)
            return rmap, dense, True
        except Exception:  # truncated or from an incompatible numpy: rebuild it
            pass
    rmap, dense = build_maps(fp, cfg, **params)
    if use_cache:
        try:
            _save(path, rmap, dense)
        except OSError:
            pass  # read-only home: run without a cache
    return rmap, dense, False
