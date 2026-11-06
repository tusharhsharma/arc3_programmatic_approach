"""THE CLAIM LEDGER — a prior is a claim with a track record (Tushar,
2026-10-06: "you seem to be creating hard rules ... the entire idea is to
be conditionally self-programmable based on what worked"; 2026-10-15:
"the claim ledger first").

Every standing presumption the core makes at a decision point — "a pump at
its limit refuses for itself", "a kind the keys moved is no panel", "a
thing in the helper's hands is not a second body" — is a CLAIM: named,
consulted through `on(name)`, and answerable.  A plan is built consulting
claims; when a step of the plan holds in the world every claim it consulted
is credited, and when the world leaves the plan every claim it consulted is
struck.  A claim struck more than it was credited on THIS level, at least
twice, is SILENCED for the rest of the level: the decision point takes the
other branch, and the next plan is the directed re-test.  The record is
carried by game as a prior — consulted only for claims this level has not
yet judged (what this level shows outranks what was carried).

Nothing is decided by hand here: the ledger says which priors held where,
and the ablation table (every claim on/off) reads off the same records.

    from agent.core import claims as _claims
    if _claims.on("pump_limit_refuses_self"): ...     # the prior, unless silenced
    _claims.begin()                                   # before a plan is built
    _claims.step(ok=True)                             # after each executed step
    _claims.export() / _claims.seed(prior)            # the carry

Switches for the ablation: CLAIMS_OFF=name1,name2 silences claims by hand
(the organ x level table); CLAIMS_FROZEN=1 keeps every claim at its default
(the ledger records but never silences) — the pre-ledger core.
"""
import os
import threading
from collections import defaultdict

_tl = threading.local()
DEFAULTS = {}            # name -> default truth (True: the prior applies unless silenced)
SILENCE_AFTER = 2        # strikes needed before a claim is silenced (hand-chosen: one contradiction may be noise; two that outnumber the holds are a record)


class Ledger:
    def __init__(self):
        self.rec = defaultdict(lambda: [0, 0])     # name -> [holds, strikes] on this level
        self.prior = {}                            # name -> [holds, strikes] carried
        self.touched = set()
        self.silenced_now = set()
        self.forced_off = set()                    # claims silenced for a trial (the directed re-test)
        self.log = []

    def silenced(self, name):
        if name in self.forced_off:
            return True
        off = os.environ.get("CLAIMS_OFF", "")
        if off and name in {x.strip() for x in off.split(",")}:
            return True
        if os.environ.get("CLAIMS_FROZEN"):
            return False
        h, s = self.rec.get(name, (0, 0))
        if h or s:
            return s >= SILENCE_AFTER and s > h
        ph, ps = self.prior.get(name, (0, 0))
        return ps >= SILENCE_AFTER and ps > ph

    def on(self, name, default=True):
        DEFAULTS.setdefault(name, default)
        self.touched.add(name)
        if self.silenced(name):
            if name not in self.silenced_now:
                self.silenced_now.add(name)
                self.log.append(f"claim '{name}' silenced: struck {self.rec[name][1]} times, held {self.rec[name][0]}" if (self.rec[name][0] or self.rec[name][1])
                                else f"claim '{name}' silenced by the carried record {self.prior.get(name)}")
            return not default
        return default

    def begin(self):
        self.touched = set()

    def step(self, ok):
        for n in list(self.touched):
            self.rec[n][1 if not ok else 0] += 1

    def strike(self, name, n=1):
        self.rec[name][1] += n

    def credit(self, name, n=1):
        self.rec[name][0] += n

    def consulted_by(self, fn):
        """The claims one call of `fn` consults (its touched set, apart)."""
        saved = self.touched; self.touched = set()
        try:
            fn()
        except Exception:
            pass
        got = self.touched; self.touched = saved
        return got

    def with_off(self, name, fn):
        """Run `fn` with claim `name` silenced for the call: the directed re-test."""
        self.forced_off.add(name)
        try:
            return fn()
        finally:
            self.forced_off.discard(name)

    def export(self):
        out = {}
        for n, (h, s) in self.rec.items():
            ph, ps = self.prior.get(n, (0, 0))
            out[n] = [h + ph, s + ps]
        for n, v in self.prior.items():
            out.setdefault(n, list(v))
        return out

    def seed(self, prior):
        self.prior = {n: list(v) for n, v in (prior or {}).items()}

    def report(self):
        judged = {n: v for n, v in self.rec.items() if v[0] or v[1]}
        if not judged and not self.silenced_now:
            return ""
        parts = [f"{n}: held {v[0]}, struck {v[1]}{' — SILENCED' if n in self.silenced_now else ''}" for n, v in sorted(judged.items())]
        return "claims this level: " + "; ".join(parts)


def current():
    led = getattr(_tl, "ledger", None)
    if led is None:
        led = Ledger(); _tl.ledger = led
    return led


def reset(prior=None):
    led = Ledger(); led.seed(prior); _tl.ledger = led
    return led


def on(name, default=True):
    return current().on(name, default)


def begin():
    current().begin()


def step(ok):
    current().step(ok)


def export():
    return current().export()


def report():
    return current().report()
