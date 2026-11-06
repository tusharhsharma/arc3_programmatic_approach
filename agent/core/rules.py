"""RULES: effect templates over entities, fitted to transitions.

A transition is (action, click point, sprites before, sprites after,
events).  Every in-field event is a WITNESS for a template

    action  when CONDITION  ->  EFFECT of CLASS (by MAGNITUDE)

where the class is the kind of sprite affected (its colours), and the
condition is drawn from a small closed set: always, click_on(kind),
click_near(kind), selected(kind) [the sprite wore a livery], blocked
[a move that worked elsewhere did nothing here].  A template is adopted
on one witness (provisional), confirmed by a second, and CONTESTED when
the same (action, class) sometimes produced the effect and sometimes
nothing — then the condition that separates the witnesses is searched,
in order of simplicity (the atom's law rule), and the open hypothesis
"unknown condition" is kept when none does.

ROLES fall out of the rules: controlled = translated by named keys;
mover = moved/grown by clicks, pushes or buttons; button = a clicked
kind whose click changed something else; selector = a clicked kind whose
click changed a livery and, after it, a different kind answered the
keys; budget = a bar that shrinks at nearly every action; wall = the
kind in the way of a refused move; collectible = a kind that vanished
under the controlled; hazard = the kind beside the controlled at a
game over.  Nothing here knows a game.
"""
from collections import Counter, defaultdict


def kind_of(s):
    """A sprite's kind for reporting: its colours, body first."""
    cols = s.colors
    body = cols.most_common(1)[0][0]
    rest = tuple(sorted(c for c in cols if c != body))
    return (body,) + rest


def _unit(v):
    r, c = v
    return ((r > 0) - (r < 0), (c > 0) - (c < 0))


class Witness:
    __slots__ = ("action", "effect", "kind", "cond", "mag", "n")

    def __init__(self, action, effect, kind, cond, mag, n):
        self.action, self.effect, self.kind, self.cond, self.mag, self.n = action, effect, kind, cond, mag, n

    def __repr__(self):
        return f"{self.action} {self.cond} -> {self.effect}{self.kind} {self.mag}"


class RuleInducer:
    def __init__(self):
        self.witnesses = []             # every effect seen, with its context
        self.presses = Counter()        # action -> times pressed
        self.nothing = Counter()        # action -> times nothing in-field happened
        self.nothing_ctx = defaultdict(list)   # action -> [context of a nothing]
        self.click_kind = {}            # n -> kind under the click (or None)
        self.n = 0
        self.key_moved = defaultdict(Counter)  # action -> kind -> times translated by it
        self.selected_before = {}       # n -> set of kinds wearing a livery before the action
        self.deaths = []                # contexts at game over
        self.co_moves = Counter()       # (kind, kind) moved with equal vectors, same step
        self.blocked_by = Counter()     # kind standing in the way of a refused move
        self.changed_per_action = Counter()   # kind -> times it changed (any action): budgets
        self.known_vec = {}             # (action, kind) -> vec (last translate seen)

    # ------------------------------------------------------------ observe

    def observe(self, action, click, before, after, events, game_over=False):
        self.n += 1
        n = self.n
        self.presses[action] += 1
        b_by = {s.sid: s for s in before}
        a_by = {s.sid: s for s in after}
        field_b = [s for s in before if "bar" not in s.tags and "hud" not in s.tags]
        clicked = None
        if click is not None:
            hit = [s for s in field_b if s.bbox[0] <= click[0] <= s.bbox[2] and s.bbox[1] <= click[1] <= s.bbox[3]
                   and (click in s.cells or s.size <= 60)]
            if hit:
                clicked = min(hit, key=lambda s: s.size)
        self.click_kind[n] = kind_of(clicked) if clicked is not None else None
        selected = {kind_of(s) for s in field_b if s.livery and s.size >= 4}
        self.selected_before[n] = selected
        moved_kinds = []
        infield = False
        for ev in events:
            s_after = a_by.get(ev.sid)
            s_before = b_by.get(ev.sid)
            ref = s_after or s_before
            if ref is None or "bar" in ref.tags or "hud" in ref.tags:
                continue
            if "terrain" in ref.tags and ev.kind in ("grew", "shrank", "split", "merged", "recoloured"):
                continue    # ground deforms when things move over it
            if ev.kind in ("moved", "rotated", "grew", "shrank", "recoloured", "vanished", "appeared", "merged", "split"):
                infield = True
            kind = kind_of(ref)
            cond = "always"
            if click is not None:
                if clicked is not None and clicked.sid == ev.sid:
                    cond = f"click_on(self)"
                elif clicked is not None:
                    cond = f"click_on({kind_of(clicked)})"
                else:
                    d = abs(click[0] - ref.centre[0]) + abs(click[1] - ref.centre[1])
                    cond = "click_near" if d <= 12 else "click_far"
            elif kind in selected:
                cond = "selected"
            mag = None
            if ev.kind == "moved":
                v = ev.data["vec"]
                mag = (int(round(v[0])), int(round(v[1])))
                if click is not None and s_after is not None:
                    # a jump to the click point is a teleport, not a translation
                    if abs(s_after.centre[0] - click[0]) + abs(s_after.centre[1] - click[1]) <= 2.5 and abs(mag[0]) + abs(mag[1]) >= 4:
                        self.witnesses.append(Witness(action, "teleport", kind, cond, None, n))
                        continue
                if click is None:
                    moved_kinds.append((kind, mag, ev.sid))
            elif ev.kind in ("grew", "shrank"):
                mag = ev.data.get("delta")
            ekind = ev.kind
            if ev.kind == "recoloured" and ev.data.get("n", 9) <= 3:
                ekind = "select"          # a livery changed hands
            self.witnesses.append(Witness(action, ekind, kind, cond, mag, n))
        # co-movement: kinds that moved together this step (rigid or pushed)
        for i in range(len(moved_kinds)):
            for j in range(i + 1, len(moved_kinds)):
                if moved_kinds[i][1] == moved_kinds[j][1]:
                    self.co_moves[(moved_kinds[i][0], moved_kinds[j][0])] += 1
        for kind, mag, sid in moved_kinds:
            self.key_moved[action][kind] += 1
        for kind, mag, sid in moved_kinds:
            self.known_vec[(action, kind)] = mag
        for ev in events:
            ref = a_by.get(ev.sid) or b_by.get(ev.sid)
            if ref is not None and ev.kind in ("grew", "shrank", "moved") and "terrain" not in ref.tags:
                self.changed_per_action[kind_of(ref)] += 1
        if not infield:
            self.nothing[action] += 1
            self.nothing_ctx[action].append({"click_kind": self.click_kind[n], "selected": selected, "n": n})
            # who was in the way?  the kind this key is known to move,
            # displaced by its vector, overlaps some sprite
            for (act, kind), vec in list(self.known_vec.items()):
                if act != action:
                    continue
                for s in field_b:
                    if kind_of(s) != kind:
                        continue
                    dest = {(r + vec[0], c + vec[1]) for r, c in s.cells}
                    for t in field_b:
                        if t.sid != s.sid and dest & set(t.cells):
                            self.blocked_by[kind_of(t)] += 1
        if game_over:
            self.deaths.append(n)

    # -------------------------------------------------------------- rules

    def rules(self):
        """Templates with support; contested ones carry the condition that
        separates their witnesses from the 'nothing' presses."""
        table = defaultdict(list)
        for w in self.witnesses:
            table[(w.action, w.effect, w.kind)].append(w)
        out = []
        for (action, effect, kind), ws in table.items():
            conds = Counter(w.cond for w in ws)
            cond, c_n = conds.most_common(1)[0]
            mags = Counter(w.mag for w in ws if w.mag is not None)
            mag = mags.most_common(1)[0][0] if mags else None
            support = len(ws)
            contested = self.nothing[action] > 0 and effect in ("moved", "grew", "shrank", "rotated", "teleport")
            if action.startswith("ACTION") and action != "ACTION6" and contested:
                # was the move refused when the kind was NOT selected?
                sel_when_moved = all(kind in self.selected_before[w.n] for w in ws)
                sel_when_nothing = any(kind in c["selected"] for c in self.nothing_ctx[action])
                if sel_when_moved and not sel_when_nothing and any(self.selected_before[w.n] for w in ws):
                    cond = "selected"
                else:
                    cond = "blocked_sometimes"
            out.append({"action": action, "effect": {"moved": "translate", "grew": "grow", "shrank": "shrink", "rotated": "rotate", "recoloured": "recolour", "vanished": "vanish", "appeared": "appear", "merged": "merge"}.get(effect, effect),
                        "kind": kind, "of": list(kind), "when": cond, "by": mag,
                        "support": support, "contested": contested})
        for action, n in self.nothing.items():
            if n >= 1:
                only = not any(r["action"] == action for r in out)
                out.append({"action": action, "effect": "nothing", "kind": None, "of": [],
                            "when": "always" if only else "elsewhere",
                            "by": None, "support": n, "contested": False})
        # BLOCKED: a key that translates a kind did nothing while something
        # stood where the kind would have gone -> that something is a wall
        for kind, n in self.blocked_by.items():
            out.append({"action": "*", "effect": "blocked", "kind": kind, "of": list(kind), "when": "adjacent",
                        "by": None, "support": n, "contested": False})
        return out

    # -------------------------------------------------------------- roles

    def roles(self, sprites):
        """kind -> role, from the rules and the sprite inventory."""
        role = {}
        rules = self.rules()
        keys = [a for a in self.presses if a != "ACTION6"]
        # controlled: translated by the keys, most often
        moved_by_keys = Counter()
        for a in keys:
            for kind, n in self.key_moved[a].items():
                moved_by_keys[kind] += n
        # a kind that only ever moved TOGETHER with another (a rigid part or
        # a pushed thing) is not the controlled one
        if moved_by_keys:
            top, _ = moved_by_keys.most_common(1)[0]
            role[top] = "controlled"
            for kind, n in moved_by_keys.items():
                if kind != top:
                    role.setdefault(kind, "mover")
        for r in rules:
            k = r["kind"]
            if k is None:
                continue
            if r["action"] == "ACTION6":
                if r["effect"] in ("translate", "grow", "shrink", "rotate", "teleport") and role.get(k) != "controlled":
                    role.setdefault(k, "mover")
                    if r["when"].startswith("click_on(") and "self" not in r["when"]:
                        bk = r["when"][len("click_on("):-1]
                        try:
                            bkind = tuple(int(x) for x in bk.strip("()").split(",") if x.strip())
                        except ValueError:
                            bkind = None
                        if bkind and role.get(bkind) not in ("controlled",):
                            role[bkind] = "button"
                if r["effect"] == "recoloured" and r["when"] == "click_on(self)":
                    role[k] = "selector" if role.get(k) not in ("controlled",) else role[k]
        for kind, n in self.blocked_by.items():
            if role.get(kind) not in ("controlled", "mover"):
                role[kind] = "wall"
        presses = sum(self.presses.values())
        for s in sprites:
            k = kind_of(s)
            if "bar" in s.tags or (presses >= 6 and self.changed_per_action[k] >= 0.6 * presses
                                   and role.get(k) not in ("controlled", "mover")):
                role[k] = "budget"
            elif "terrain" in s.tags:
                role.setdefault(k, "floor")
        return role

    def report(self, sprites):
        role = self.roles(sprites)
        ents = []
        seen = set()
        for k, r in role.items():
            key = (r, tuple(sorted(set(k))))
            if key not in seen:
                seen.add(key)
                ents.append({"role": r, "colors": sorted(set(k))})
        rules = []
        for r in self.rules():
            rules.append({"action": r["action"], "effect": r["effect"], "of": sorted(set(r["of"])),
                          "when": r["when"], "by": r["by"], "support": r["support"]})
        return {"entities": ents, "rules": rules, "goal": None}
