"""THE READER SEAM: who says what goes where, in one language.

The core's organs execute: the scene tracker names things, the fiddler
learns laws by press, the planner carries a READING out, the ledger
checks the plan against the world.  What none of them does for a game
never met is make the reading — "crates go onto the pen", "the orange
one is helping", "that line is a wall I cannot pass but the boxes can".
That is the reader's job, and this module makes it a seam with two
sides:

  * the SCENE, as text any reader can take (`scene_text`): the things
    the tracker names (id, colour, size, box, tags, materials, agents)
    and the frame itself as a grid; with, after a lost attempt, the
    ledger's failure report;
  * the READING, in the relation language the planner already takes
    (`READING_SCHEMA` -> `to_hypotheses`), and LAWS BY KIND the model
    already holds (`apply_laws`): agents, what passes what, what refuses
    what, the carry law's target kind, hazards.

Two readers implement it: `RelationReader` (today's first-frame
relations, agent/core/relations.py) and `LLMReader`, which formats the
scene for a language model and parses its answer — the model call
itself is a function handed in (`ask(prompt) -> str`), so the same
reader runs under any model, on Kaggle or elsewhere.  Nothing here
names a game.
"""
import json, re
from types import SimpleNamespace
from collections import Counter

from .goals import GoalHypothesis

KINDS = ("align", "cover", "equal")

READING_SCHEMA = """Answer with ONE JSON object and nothing else:
{
  "controlled": "S<id> or null",            // the thing the arrow keys move, if you can tell
  "readings": [                              // what goes where, best first (at most 4)
    {"kind": "align" | "cover" | "equal",    // align: the mover's centre onto the target's centre (into a socket, onto a marker);
                                             // cover: the mover's cells onto the target's cells (fit the shape, dock the marks);
                                             // equal: make the target look like the mover (a picture to copy)
     "mover": "S<id>",                       // the thing that must be moved (or, for a carried thing, the thing carried)
     "target": "S<id>",                      // the thing it goes to
     "why": "one sentence"}
  ],
  "laws": {                                  // only what you are fairly sure of; omit keys you are not
    "agents": [colour, ...],                 // colours of things that move on their own (an ally, a thief)
    "helpers": [colour, ...],                // agents that advance the goal (leave their work to them)
    "foes": [colour, ...],                   // agents that undo it
    "passes": [[mover colour, material colour], ...],   // this kind can stand on / move through that kind
    "refuses": [[mover colour, material colour], ...],  // this kind cannot enter that kind
    "hazards": [colour, ...],                // touching or entering kills or resets
    "carry": {"key": "ACTION5", "kind": colour, "target": colour},   // a key that binds a kind to the body; where that kind goes
    "fiddle_first": ["S<id>", ...]           // things to press against / click first, if nothing above is sure
  }
}
Colours are the integers 0-15 shown in the table; a material (a hatched or brick texture) is listed with its own kind number (colour + 16)."""


# ---------------------------------------------------------------- the scene as text
def scene_text(frame, things, model=None, report=None, max_things=40):
    """The scene a reader is given: the things the tracker names, the
    laws already known, the frame as a grid, and — after a lost attempt —
    the ledger's failure report."""
    import numpy as np
    f = np.asarray(frame)
    H, W = f.shape
    cnt = Counter(int(v) for v in f.ravel())
    bg = cnt.most_common(1)[0][0]
    lines = [f"FRAME {H}x{W}, background colour {bg}. Rows top to bottom, columns left to right; each cell one hex digit (colour 0-15).", ""]
    for r in range(H):
        lines.append("%2d " % r + "".join("%x" % int(f[r, c]) for c in range(W)))
    lines.append("")
    lines.append("THINGS (the scene tracker's reading; S<id> names a thing; box is r<top>-<bottom> c<left>-<right>):")
    whole = [s for s in things if s.sid < 1000]
    for s in sorted(whole, key=lambda s: (-s.size, s.sid))[:max_things]:
        r0, c0, r1, c1 = s.bbox
        kd = s.body + 16 if "texture" in s.tags else s.body
        tags = [t for t in sorted(s.tags) if t in ("hud", "terrain", "texture", "bar")]
        parts = [p for p in things if p.sid >= 1000 and p.sid // 1000 == s.sid]
        lines.append(f"S{s.sid}: colour {s.body}" + (f" (material, kind {kd})" if kd != s.body else "")
                     + f", {s.size} cells, box r{r0}-{r1} c{c0}-{c1}"
                     + (f", tags {tags}" if tags else "")
                     + (f", marks inside: {sorted({p.body for p in parts})}" if parts else ""))
    if model is not None:
        known = []
        try:
            for k in model.fixed.keys():
                mv = model.fixed.moving(k)
                if mv:
                    known.append(f"{k}: " + ", ".join(f"S{sid} {e[0]} {tuple(e[1]) if isinstance(e[1], (tuple, list)) else e[1]}" for sid, e in list(mv.items())[:4]))
        except Exception:
            pass
        if known:
            lines += ["", "LAWS LEARNED BY PRESSING (key: what moved and how):"] + known
        ag = sorted(getattr(model, "agents", ()) or ())
        if ag:
            lines.append(f"Things of colour {ag} travel on their own.")
        ps = sorted(getattr(model, "passable", ()) or ())
        if ps:
            lines.append(f"Seen to pass through / stand on: {ps[:8]}")
    if report:
        lines += ["", "THE LAST ATTEMPT WAS LOST. The agent's own report:", report]
    return "\n".join(lines)


# ---------------------------------------------------------------- the reading, in the planner's language
def _thing_like(s):
    cells = set(s.cells) if not isinstance(s.cells, dict) else set(s.cells.keys())
    r0, c0, r1, c1 = s.bbox
    return SimpleNamespace(color=s.body, size=s.size, centre=((r0 + r1) / 2.0, (c0 + c1) / 2.0), cells=frozenset(cells), bbox=s.bbox, sid=s.sid)


def parse_answer(text):
    """The first JSON object in a model's answer, or None."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    raw = m.group(0)
    for cand in (raw, re.sub(r"//[^\n]*", "", raw), re.sub(r",\s*([}\]])", r"\1", re.sub(r"//[^\n]*", "", raw))):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def _sid(x):
    if x is None:
        return None
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else None


def to_hypotheses(answer, things, base_prio=8.0):
    """Readings from a parsed answer -> GoalHypothesis objects the planner
    takes (best first, prio above the first-frame relations' so they are
    pursued first — and judged by the ledger like any other)."""
    out = []
    if not answer:
        return out
    by = {s.sid: s for s in things if s.sid < 1000}
    for i, rd in enumerate(answer.get("readings", []) or []):
        kind = (rd.get("kind") or "align").lower()
        if kind not in KINDS:
            continue
        m = by.get(_sid(rd.get("mover"))); t = by.get(_sid(rd.get("target")))
        if m is None or t is None or m.sid == t.sid:
            continue
        mv = _thing_like(m); tg = _thing_like(t)
        target = tg.centre if kind == "align" else tg.cells
        note = f"reader: {rd.get('why', '')[:60]} C{t.body}[{t.size}@{t.bbox}]"
        h = GoalHypothesis(kind, mv, target, note, base_prio - 0.1 * i, tg.size)
        h.t_cells = frozenset(tg.cells)
        h.t_color = t.body
        h.evidence = ("reader",)
        out.append(h)
    return out


def apply_laws(answer, model, things=(), note=None):
    """Laws by kind from a parsed answer into the model's own fields —
    as CLAIMS: each is also written to `model.reader_claims` so the
    ledger can hold the reader to them."""
    if not answer:
        return []
    laws = answer.get("laws") or {}
    claims = []

    def _col(x):
        try:
            return int(x)
        except Exception:
            return None
    for c in laws.get("agents", []) or []:
        k = _col(c)
        if k is not None:
            model.agents.add(k); claims.append(("agent", k))
    for c in laws.get("helpers", []) or []:
        k = _col(c)
        if k is not None:
            model.agents.add(k)
            sg = getattr(model, "agent_sign", None)
            if sg is None:
                sg = model.agent_sign = {}
            sg[k] = max(sg.get(k, 0), 1); claims.append(("helper", k))
    for c in laws.get("foes", []) or []:
        k = _col(c)
        if k is not None:
            model.agents.add(k)
            sg = getattr(model, "agent_sign", None)
            if sg is None:
                sg = model.agent_sign = {}
            sg[k] = min(sg.get(k, 0), -1); claims.append(("foe", k))
    for pr in laws.get("passes", []) or []:
        if isinstance(pr, (list, tuple)) and len(pr) == 2 and _col(pr[0]) is not None and _col(pr[1]) is not None:
            model.passable.add((_col(pr[0]), _col(pr[1]))); claims.append(("passes", (_col(pr[0]), _col(pr[1]))))
    for pr in laws.get("refuses", []) or []:
        if isinstance(pr, (list, tuple)) and len(pr) == 2 and _col(pr[1]) is not None:
            model.ground_refused[_col(pr[1])] = max(1, model.ground_refused.get(_col(pr[1]), 0)); claims.append(("refuses", (_col(pr[0]), _col(pr[1]))))
    for c in laws.get("hazards", []) or []:
        k = _col(c)
        if k is not None:
            model.death_touch[k] = model.death_touch.get(k, 0) + 1; claims.append(("hazard", k))
    cr = laws.get("carry") or {}
    if cr and _col(cr.get("kind")) is not None and cr.get("key"):
        law = {"key": str(cr["key"]), "kinds": {_col(cr["kind"])}, "target_kind": _col(cr.get("target")), "held": None, "offset": None, "seeded": True}
        # (the form — dims, core, the held colour — is read off a thing of the kind when one stands here)
        for s in things:
            if s.sid < 1000 and s.body == _col(cr["kind"]):
                law["dims"] = (s.bbox[2] - s.bbox[0] + 1, s.bbox[3] - s.bbox[1] + 1)
                parts = [p for p in things if p.sid >= 1000 and p.sid // 1000 == s.sid]
                if parts:
                    p = parts[0]
                    law["core"] = (p.body, p.bbox[0] - s.bbox[0], p.bbox[1] - s.bbox[1], p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1])
                break
        if law.get("dims") and getattr(model, "carry", None) is None:
            model.carry = law; claims.append(("carry", (law["key"], _col(cr["kind"]), law["target_kind"])))
    model.reader_claims = getattr(model, "reader_claims", []) + claims
    if note and claims:
        note(f"reader's laws taken as claims: {claims}")
    return claims


# ---------------------------------------------------------------- the readers
class RelationReader:
    """Today's reader: first-frame relations (agent/core/relations.py)."""
    name = "relations"

    def read(self, objs, frame, things, model=None, report=None, **kw):
        from .core.relations import read_frame_goals
        return read_frame_goals(objs, frame.shape[0], frame.shape[1], playfield=kw.get("playfield"), hud_cells=kw.get("hud_cells", frozenset()), weights=kw.get("weights")), {}


class LLMReader:
    """A language model as the reader.  `ask(prompt) -> str` is the model
    call (Kaggle: a local quantized model; elsewhere: whatever is at
    hand).  Every answer is kept (`self.log`) for the trial's record."""
    name = "llm"
    PREAMBLE = ("You are reading one screen of an unknown puzzle game for an agent that can move things with keys "
                "(ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right, ACTION5 and others unknown) and click. "
                "The level is won when some arrangement is reached. Say what goes where, as a person would on first look: "
                "like shapes fit their frames or sockets, marks go onto like marks, a picture is copied, things of a kind "
                "are brought to the pen of their colour, two like movers meet. Name only things from the table.\n\n")

    def __init__(self, ask, max_things=40):
        self.ask = ask
        self.max_things = max_things
        self.log = []

    def prompt(self, frame, things, model=None, report=None):
        return self.PREAMBLE + scene_text(frame, things, model, report, self.max_things) + "\n\n" + READING_SCHEMA

    def read(self, objs, frame, things, model=None, report=None, **kw):
        p = self.prompt(frame, things, model, report)
        text = self.ask(p)
        ans = parse_answer(text)
        self.log.append({"prompt_chars": len(p), "answer": text, "parsed": ans})
        return to_hypotheses(ans, things), (ans or {})


class ScriptedReader:
    """Answers recorded elsewhere (the Kaggle trial's reader_answers.json,
    keyed "<game>:<level>") replayed into the agent here, where there is
    no GPU: the trial's readings are tested by play, not only by eye."""
    name = "scripted"

    def __init__(self, answers, game=None):
        self.answers = answers
        self.game = game
        self.level = 0
        self.log = []

    def read(self, objs, frame, things, model=None, report=None, **kw):
        self.level += 1
        key = f"{self.game}:{self.level}"
        a = self.answers.get(key)
        text = a.get("answer") if isinstance(a, dict) else a
        ans = parse_answer(text)
        self.log.append({"key": key, "parsed": ans})
        return to_hypotheses(ans, things), (ans or {})
