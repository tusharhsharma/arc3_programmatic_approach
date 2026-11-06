"""The claim ledger: a prior with a track record (agent/core/claims.py)."""
from agent.core import claims as C


def test_silence_after_strikes_outnumber_holds():
    led = C.reset()
    assert led.on("x") is True
    led.begin(); led.on("x"); led.step(False)
    assert led.on("x") is True            # one strike is noise
    led.begin(); led.on("x"); led.step(False)
    assert led.on("x") is False           # two strikes, no holds: silenced
    assert any("silenced" in l for l in led.log)


def test_this_level_outranks_the_carried():
    led = C.reset(prior={"x": [0, 5]})     # carried: struck five times elsewhere
    assert led.on("x") is False            # the prior decides while this level has no record
    led.begin(); led.on("x"); led.step(True)
    assert led.on("x") is True             # one hold here outranks the carried strikes
    out = led.export()
    assert out["x"] == [1, 5]


def test_directed_retest_finds_the_culprit():
    led = C.reset()
    world = {"a": 1, "b": 2}

    def predict():
        # a prediction that is wrong while claim 'b' holds, right when it is silenced
        return (1 if led.on("a") else 0) + (10 if led.on("b") else 2)
    cands = led.consulted_by(predict)
    assert cands == {"a", "b"}
    observed = 3
    culprit = next((n for n in sorted(cands) if led.with_off(n, predict) == observed), None)
    assert culprit == "b"
    led.strike(culprit, 2)
    assert led.on("b") is False and led.on("a") is True
    assert led.forced_off == set()        # the trial silencing is undone


def test_frozen_and_off_switches(monkeypatch):
    led = C.reset()
    monkeypatch.setenv("CLAIMS_OFF", "x,y")
    assert led.on("x") is False and led.on("z") is True
    monkeypatch.delenv("CLAIMS_OFF")
    monkeypatch.setenv("CLAIMS_FROZEN", "1")
    led.begin(); led.on("x"); led.step(False); led.begin(); led.on("x"); led.step(False)
    assert led.on("x") is True            # frozen: the ledger records but never silences
