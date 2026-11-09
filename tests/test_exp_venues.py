"""Venue experiment: one short run end to end (people blocking Wi-Fi, one kidnap)."""

from wiloc.exp_venues import METHODS, run, summarise


def test_short_run_in_the_home():
    r = run(("home", "installed", 0, 70.0, 45.0))
    assert set(r["median"]) == set(METHODS)
    assert r["median"]["pf"] < 1.5
    rows = summarise([r])
    assert rows[0]["venue"] == "home" and rows[0]["runs"] == 1
