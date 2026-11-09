"""Story layer: confidence state with hysteresis, recovery clock, attract-mode schedule."""

from wiloc.story import LOCKED, SEARCHING, UNCERTAIN, AttractMode, ConfidenceState, RecoveryClock, merge_modes, tour_stops


def run(cs, trace, dt=0.5):
    out = []
    for i, (spread, lost, w) in enumerate(trace):
        out.append(cs.update(i * dt, spread, lost, w))
    return out


def test_confidence_needs_to_hold_before_switching():
    cs = ConfidenceState(hold=2.0, hold_search=0.5)
    cs.reset(LOCKED)
    # one noisy scan with a wide spread does not flip the pill
    states = run(cs, [(0.3, False, 1.0)] * 4 + [(2.0, False, 0.5)] + [(0.3, False, 1.0)] * 4)
    assert set(states) == {LOCKED}
    # a sustained two-room ambiguity becomes UNCERTAIN after 2 s
    cs.reset(LOCKED)
    states = run(cs, [(2.0, False, 0.5)] * 8)
    assert states[3] == LOCKED and states[-1] == UNCERTAIN


def test_kidnap_shows_searching_quickly_then_locks():
    cs = ConfidenceState()
    cs.reset(LOCKED)
    trace = [(0.3, False, 1.0)] * 4 + [(0.4, True, 1.0)] * 6 + [(2.5, False, 0.5)] * 6 + [(0.4, False, 0.9)] * 8
    states = run(cs, trace)
    assert states[5] == SEARCHING  # 0.5 s after the alarm
    assert UNCERTAIN in states[10:16]
    assert states[-1] == LOCKED


def test_recovery_clock_reports_once():
    rc = RecoveryClock(threshold=0.8, hold=2.0, grace=1.0)
    assert rc.update(0.0, 0.2) is None  # no kidnap yet
    rc.kidnap(10.0)
    assert rc.update(10.2, 0.1) is None  # stale error inside the grace period
    got = []
    for i in range(60):
        t = 11.0 + 0.5 * i
        err = 6.0 if t < 20.0 else (2.0 if t < 21.0 else 0.4)
        got.append(rc.update(t, err))
    hits = [g for g in got if g is not None]
    assert hits == [11.0]  # good from t=21 s, confirmed 2 s later
    assert not rc.waiting


def test_recovery_clock_resets_on_relapse():
    rc = RecoveryClock(threshold=0.8, hold=2.0, grace=0.0)
    rc.kidnap(0.0)
    errs = [0.5, 0.5, 3.0, 0.5, 0.5, 0.5, 0.5, 0.5]  # good, relapse, good for good
    hits = [rc.update(i * 1.0, e) for i, e in enumerate(errs)]
    assert [h for h in hits if h is not None] == [3.0]


def test_attract_schedule():
    a = AttractMode(idle=60, kidnap_every=150, rotate_every=360, first_kidnap=45)
    a.touch(0.0)
    assert a.due(30.0) == []
    assert a.due(60.0) == ["start"]
    assert a.due(100.0) == []
    assert a.due(105.0) == ["kidnap"]
    assert a.due(254.0) == []
    assert a.due(255.0) == ["kidnap"]
    assert a.due(420.0) == ["rotate"]
    a.venue_loaded(430.0)
    assert a.due(470.0) == []
    assert a.due(475.0) == ["kidnap"]
    a.touch(480.0)  # a visitor takes over
    assert a.due(500.0) == []
    assert not a.active
    a.enabled = False
    assert a.due(1000.0) == []


def test_tour_stops_cover_every_layout():
    stops = tour_stops({"office": ["installed", "corridor"], "home": ["installed"], "lab": ["a", "b", "c"]})
    assert stops[:3] == [("office", "installed"), ("home", "installed"), ("lab", "a")]
    assert len(stops) == 6 and ("lab", "c") in stops


def test_merge_modes_folds_neighbours():
    modes = [(5.0, 5.0, 0.54), (5.8, 5.6, 0.31), (15.0, 5.0, 0.15)]
    got = merge_modes(modes, radius=1.5)
    assert len(got) == 2
    assert got[0][:2] == (5.0, 5.0) and abs(got[0][2] - 0.85) < 1e-9
    assert merge_modes([]) == []
