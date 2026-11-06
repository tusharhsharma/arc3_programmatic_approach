"""Micromouse-style world model v2, camera-aware.

All positions are WORLD coordinates: field-local coords + accumulated camera
offset, living on a fixed canvas with an origin margin so scrolling games fit.

Learns: avatar identity, per-action motion vectors (possibly multi-cell
slides), occupancy (FREE/WALL/UNKNOWN), a color->terrain generalization, and
a persistent color map of everything ever seen (survives off-screen scroll).
Planning is optimistic flood-fill over the canvas.
"""
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

UNKNOWN, FREE, WALL = 0, 1, 2
CANVAS = 224          # world canvas size
ORIGIN = 80           # world (0,0) maps to canvas[ORIGIN, ORIGIN]


def wc(pos):
    """world -> canvas index"""
    return (pos[0] + ORIGIN, pos[1] + ORIGIN)


@dataclass
class MotionBelief:
    votes: dict

    def best(self):
        """VERIFICATION BY USE: a single UNCONTESTED observation is
        provisionally true — planning with it makes every purposeful step a
        free confirmation trial (the second vote arrives during travel, not
        during a paid probe). Only a CONTESTED belief (two different
        outcomes observed) demands the classic two-vote consistency, since
        one of the samples was noise by construction. The attribution
        filters (global-motion, autonomous movers, timers, avatar
        consistency) do the pre-cleaning that makes provisional trust safe."""
        if not self.votes:
            return None
        vec, n = max(self.votes.items(), key=lambda kv: kv[1])
        if n >= 2:
            return vec
        # single vote: provisional ONLY if nothing contradicts it
        return vec if len(self.votes) == 1 else None

    def plausible(self):
        """All outcome vectors with meaningful support. Net motion is
        context-dependent (drift blocked at walls, slides of varying
        length), so planning must consider every credible outcome."""
        total = sum(self.votes.values())
        if not total:
            return []
        return [v for v, n in self.votes.items()
                if n >= max(1, 0.25 * total) and v != (0, 0)]


@dataclass
class Law:
    """CONDITIONED LAW — the atom's first coined relation form.

    A belief ("ACTION1 moves up", "color 5 is floor") that is CONTESTED
    by observation is not discarded and not averaged: the contradiction
    is explained by the simplest CONTEXT predicate that separates the
    successes from the failures (a decision stump over a few context
    features).  group = what the law is about — ("action", name) or
    ("color", c); the law says the move is REFUSED exactly when
    (feature == value) is `blocks_when`.  Adopted on perfect separation
    with >=2 witnesses per side; refuted by the first counter-example.
    Blind-set motivation (sx05: 'a move on the same axis as the previous
    move is ignored'; sx04: 'gate cells block on even action counts')."""
    group: tuple
    feature: str
    value: object
    blocks_when: bool = True
    outcome: dict = None        # CONDITIONED OUTCOME: {True: vec, False: vec}

    def refuses(self, ctx):
        if self.outcome is not None:
            return False
        return (ctx.get(self.feature) == self.value) == self.blocks_when

    def outcome_for(self, ctx):
        return self.outcome[ctx.get(self.feature) == self.value]


CTX_FEATURES = ("parity", "prev_axis", "prev_action")
# per-(position, action) features: not part of the product state, computed
# where the trial happened and again per planner state
POS_FEATURES = ("at_edge",)


def _axis(vec):
    if vec is None or vec == (0, 0):
        return None
    return "v" if abs(vec[0]) >= abs(vec[1]) else "h"


def unit(vec):
    """Collapse a slide like (0, 5) to its direction (0, 1)."""
    dr, dc = vec
    ur = 0 if dr == 0 else (1 if dr > 0 else -1)
    uc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return (ur, uc)


class MicromouseModel:
    def __init__(self):
        self.occ = np.full((CANVAS, CANVAS), UNKNOWN, dtype=np.int8)
        self.colors = np.full((CANVAS, CANVAS), -1, dtype=np.int16)  # seen colors
        self.motion = defaultdict(lambda: MotionBelief(votes=defaultdict(int)))
        self.slide_lengths = defaultdict(list)   # action -> observed magnitudes
        self.avatar_shape_key = None
        self.avatar_pos = None                   # world coords
        self.color_stats = defaultdict(lambda: [0, 0])  # color -> [free, wall]
        self._noted_cells = set()   # (cell, label) pairs already counted
        self.inert_actions = set()               # probed to exhaustion, no effect
        self.shape_op_actions = {}               # action -> dihedral op name (rotate/flip)
        # --- conditioned laws (the atom, v0) ---
        self.ctx = {"parity": 0, "prev_axis": None, "prev_action": None}
        self.trials = []            # (action, moved, dest_color, dest_pos, ctx_before)
        self.laws = {}              # group -> Law
        self.law_events = []        # ("adopt"|"refute"|"unknown", Law, support) — drained by the agent
        self._unknown_logged = set()
        self.pushable = set()       # colors that moved with the avatar on contact (confirmed)
        self.last_stay_explained = False
        self.prev_colors = None
        self._pend_terrain = {}     # cell -> color it showed before the avatar stood on it
        # --- causal click laws: click(color X) -> avatar lands on X ---
        self.click_trials = []      # (color, obj_tl, av_before, av_after)
        self.click_laws = {}        # color -> "teleport"
        self.last_objs = []
        self.avatar_colors = set()  # every color identity has worn this level

    # ---------- color knowledge ----------

    def _note_color(self, color, label):
        if color is None or color < 0:
            return
        self.color_stats[int(color)][0 if label == FREE else 1] += 1

    def wall_colors(self):
        return {c for c, (f, w) in self.color_stats.items()
                if w >= 1 and w > 4 * f}

    def update_view(self, field_colors: np.ndarray, cam):
        """Paint the currently visible playfield into the world color map."""
        h, w = field_colors.shape
        r0, c0 = wc((cam[0], cam[1]))
        self.prev_colors = self.colors.copy()
        self.colors[r0:r0 + h, c0:c0 + w] = field_colors
        # terrain testimony settles when a vacated cell shows its old color
        # again (a collected item leaves background instead)
        for cell, pc in list(self._pend_terrain.items()):
            cur = int(self.colors[cell])
            if self.avatar_pos is not None and wc(self.avatar_pos) == cell:
                continue
            del self._pend_terrain[cell]
            if cur == pc and (cell, "pre") not in self._noted_cells:
                self._noted_cells.add((cell, "pre"))
                self._note_color(pc, FREE)

    def _mark(self, world_pos, label, note=True):
        r, c = wc(world_pos)
        if not (0 <= r < CANVAS and 0 <= c < CANVAS):
            return
        self.occ[r, c] = label
        # color generalization counts DISTINCT CELLS, not events: bumping the
        # same blocked corner 200 times must not cast 200 wall votes for the
        # floor color (observed in dc22: 2 cells produced 356/367 votes)
        # note=False: occupancy knowledge WITHOUT color testimony — a cell
        # merely PASSED OVER during a multi-cell move is proven passable,
        # but its color must not be condemned as walked-on floor (a slide
        # crossing a goal cell classified the GOAL COLOR as terrain and
        # goal-seek went blind — observed st80)
        if note and ((r, c), label) not in self._noted_cells:
            self._noted_cells.add(((r, c), label))
            col = self.colors[r, c]
            self._note_color(int(col) if col >= 0 else None, label)
            # THE TERRAIN OF A CELL IS WHAT IT SHOWED BEFORE YOU STOOD ON
            # IT: after a one-cell step the destination now wears the
            # avatar's color, so the floor color earned no free testimony
            # and a handful of refused moves condemned it as a wall
            # (observed sx05: floor 0 -> wall_colors, every later
            # refusal on floor discounted as "explained by a wall")
            # ...but only once the cell REVERTS to that color after the
            # avatar leaves: a collectible's color is also "what the cell
            # showed before" and noting it as terrain excluded the goal
            # class from goal-seek (st21 collect3: L1 29 -> 55, L2 lost)
            if (label == FREE and self.prev_colors is not None
                    and ((r, c), "pre") not in self._noted_cells):
                pc = self.prev_colors[r, c]
                if pc >= 0 and pc != col:
                    self._pend_terrain[(r, c)] = int(pc)

    # ---------- learning ----------

    def observe_transition(self, action_name, diff, objs_after_world, blocked=None):
        """objs_after_world: [(shape_key, color, size, bbox_world)] of current frame.
        blocked: explicit no-effect verdict from the agent (which can tell
        avatar-local animation noise from real effects); defaults to the
        raw no-world-change test."""
        ctx_before = dict(self.ctx)
        self.last_stay_explained = False
        self.last_objs = list(objs_after_world)
        if self.avatar_shape_key is not None:
            self.avatar_colors.add(self.avatar_shape_key[0])
        if action_name == "RESET":
            # the level restarts: so does every history-dependent context
            self.ctx = {"parity": 0, "prev_axis": None, "prev_action": None}
        else:
            self.ctx["parity"] ^= 1
            self.ctx["prev_action"] = action_name
        if action_name in self.shape_op_actions:
            return   # a rotate/flip: no translation to learn from
        if blocked is None:
            blocked = not diff.world_changed
        if action_name == "ACTION6":
            # clicks are not translations and confer no avatar identity:
            # a clicked cell that "moves" must not become the player
            return
        pos_before = self.avatar_pos
        self._observe_translation(action_name, diff, objs_after_world, blocked)
        # the trial is judged once the avatar has been re-located
        # (finish_transition, called by the agent after locate_avatar)
        self._pending = (action_name, diff, blocked, pos_before, ctx_before)

    def finish_transition(self):
        p = getattr(self, "_pending", None)
        self._pending = None
        if p is not None:
            self._record_trial(*p)

    # ---------- conditioned laws ----------

    def _record_trial(self, action_name, diff, blocked, pos_before, ctx_before):
        """One line in the trial ledger per translation attempt, then the
        stump search.  Only attempts with a known avatar and a believed
        direction are evidence (an unexplained no-op is nothing yet)."""
        if action_name == "RESET" or pos_before is None:
            return
        moved = bool(diff.moved) and self.avatar_pos != pos_before
        ctx_before = dict(ctx_before)
        bv = self.motion[action_name].best()
        ctx_before["at_edge"] = self.at_edge(pos_before, bv) if bv else None
        obs_vec = None
        if moved:
            vec = (self.avatar_pos[0] - pos_before[0],
                   self.avatar_pos[1] - pos_before[1])
            obs_vec = vec
            self.ctx["prev_axis"] = _axis(vec)
            dest = self.avatar_pos
        else:
            if not blocked:
                return          # something else happened; not a refusal
            if self.drift_estimate() != (0, 0):
                return          # under a current a no-op may be flow cancelling
            vec = self.motion[action_name].best()
            if not vec:
                return
            u = unit(vec)
            dest = (pos_before[0] + u[0], pos_before[1] + u[1])
            # a refusal with a KNOWN wall anywhere on the step's path is
            # explained already (mz02: a 2-cell step blocked by the wall
            # two cells out was logged against the floor cell adjacent)
            wcs = self.wall_colors()
            n = max(abs(vec[0]), abs(vec[1]))
            for i in range(1, n + 1):
                pr = pos_before[0] + round(vec[0] * i / n)
                pc = pos_before[1] + round(vec[1] * i / n)
                r_, c_ = wc((pr, pc))
                if not (0 <= r_ < CANVAS and 0 <= c_ < CANVAS):
                    return
                if self.occ[r_, c_] == WALL and (pr, pc) != dest:
                    return
                if self.colors[r_, c_] in wcs:
                    return
        rr, cc = wc(dest)
        src = self.prev_colors if self.prev_colors is not None else self.colors
        if not (0 <= rr < CANVAS and 0 <= cc < CANVAS) or src[rr, cc] < 0:
            return
        dcol = int(src[rr, cc])
        if len(self.trials) < 600:
            self.trials.append((action_name, moved, dcol, dest, ctx_before, obs_vec))
        # a stay the adopted law predicted is not a wall
        if not moved:
            for g in (("action", action_name), ("color", dcol)):
                law = self.laws.get(g)
                if law and law.refuses(ctx_before):
                    self.last_stay_explained = True
                    self.occ[rr, cc] = UNKNOWN if self.occ[rr, cc] == WALL else self.occ[rr, cc]
        self._condition_search(action_name, dcol)
        self._outcome_search(action_name)

    def at_edge(self, pos, vec):
        """The cell one unit step from pos in vec's direction has never
        been seen (outside the playfield): where wrap-arounds and
        teleports live."""
        if vec is None or vec == (0, 0):
            return None
        u = unit(vec)
        r_, c_ = wc((pos[0] + u[0], pos[1] + u[1]))
        if not (0 <= r_ < CANVAS and 0 <= c_ < CANVAS):
            return True
        return bool(self.colors[r_, c_] < 0)

    def _outcome_search(self, action_name):
        """CONDITIONED OUTCOMES: an action whose observed vectors are
        contested (two well-supported vectors, not a slide, no current)
        is explained by the context stump that separates them (sx02:
        ACTION4 is (0,1) except at the east edge, where it is (0,-15)).
        Evidence bar: both outcomes >= 2 witnesses, the minority from >= 2
        distinct cells; refuted by one contradicting move."""
        if self.is_slide(action_name) or self.drift_estimate() != (0, 0):
            return
        g = ("action", action_name)
        rows = [t for t in self.trials if t[0] == action_name and t[1] and t[5]]
        cur = self.laws.get(g)
        if cur is not None and cur.outcome is not None:
            if any(cur.outcome_for(t[4]) != t[5] for t in rows):
                self.law_events.append(("refute", cur, len(rows)))
                del self.laws[g]
            return
        if cur is not None or len(rows) < 4:
            return
        from collections import Counter as _Co
        cnt = _Co(t[5] for t in rows)
        if len(cnt) < 2:
            return
        (va, na), (vb, nb) = cnt.most_common(2)
        # the minority outcome needs two cells or three witnesses (a
        # re-test of the same anomaly happens at the same cell)
        if nb < 2 or (len({t[3] for t in rows if t[5] == vb}) < 2 and nb < 3):
            return
        if sum(cnt.values()) != na + nb:
            return          # three-way contest: not a stump
        for f in CTX_FEATURES + POS_FEATURES:
            vals = sorted({t[4].get(f) for t in rows}, key=lambda v: (v is None, str(v)))
            for v in vals:
                if all((t[4].get(f) == v) == (t[5] == vb) for t in rows):
                    law = Law(g, f, v, True, outcome={True: vb, False: va})
                    self.laws[g] = law
                    self.law_events.append(("adopt", law, len(rows)))
                    return
        if g not in self._unknown_logged:
            self._unknown_logged.add(g)
            self.law_events.append(("unknown", Law(g, "outcome", None), len(rows)))

    def _condition_search(self, action_name, dcol):
        """Contradiction-driven conditioning.  For each group the trial
        touches, if the group is CONTESTED (both moved and refused, with
        the refusals not explained by known walls), look for one context
        stump that separates them perfectly (>=2 witnesses each side).
        First-found is adopted; an existing law that the ledger now
        contradicts is refuted (and the search may re-adopt another)."""
        wcs = self.wall_colors()
        for g in (("action", action_name), ("color", dcol)):
            if g[0] == "action":
                rows = [t for t in self.trials if t[0] == g[1]
                        and (t[1] or t[2] not in wcs)]
            else:
                if dcol in wcs:
                    continue
                rows = [t for t in self.trials if t[2] == g[1]]
            moved = [t for t in rows if t[1]]
            stayed = [t for t in rows if not t[1]]
            cur = self.laws.get(g)
            if cur is not None and cur.outcome is not None:
                continue        # an outcome law owns this group
            if cur is not None:
                ok = all(cur.refuses(t[4]) == (not t[1]) for t in rows)
                if not ok:
                    self.law_events.append(("refute", cur, len(rows)))
                    del self.laws[g]
                    cur = None
            # EVIDENCE BAR: three witnesses a side, and the refusals must
            # come from two different cells (one wall bumped twice with
            # the same previous action is the classic false stump)
            if (cur is not None or len(moved) < 3 or len(stayed) < 3
                    or len({t[3] for t in stayed}) < 2):
                continue
            found = None
            # deterministic, simplest-first: a POSITIVE stump ("refused
            # when f == v") before its complement, values in sorted order
            for f in CTX_FEATURES:
                vals = sorted({t[4].get(f) for t in rows}, key=lambda v: (v is None, str(v)))
                for bw in (True, False):
                    for v in vals:
                        cand = Law(g, f, v, bw)
                        if all(cand.refuses(t[4]) == (not t[1]) for t in rows):
                            found = cand
                            break
                    if found:
                        break
                if found:
                    break
            if not found and g not in self._unknown_logged:
                # THE OPEN HYPOTHESIS: contested, and no context the
                # program can name separates the outcomes — "unknown
                # mechanic here", logged, never rounded to a wall
                self._unknown_logged.add(g)
                self.law_events.append(("unknown", Law(g, "refusal", None), len(rows)))
            if found:
                self.laws[g] = found
                self.law_events.append(("adopt", found, len(rows)))
                # walls the law explains were never walls
                for t in stayed:
                    if found.refuses(t[4]):
                        r_, c_ = wc(t[3])
                        if 0 <= r_ < CANVAS and 0 <= c_ < CANVAS \
                                and self.occ[r_, c_] == WALL:
                            self.occ[r_, c_] = UNKNOWN

    def record_click(self, color, obj_tl, av_before, av_after, av_col=None):
        """CAUSAL CLICK LAW (grammar: click(X) -> delta(Y)). A click on an
        object of color X after which the avatar sits exactly where that
        object was is a teleport witness; two witnesses adopt the law,
        one click on X that leaves the avatar put refutes it.  (Blind
        set 1, sx06: click-swap with an anchor; hidden games are click-
        heavy, and a coined teleport edge is worth a whole level.)"""
        if color is None or av_before is None or av_after is None:
            return False
        # THE BODY MUST BE DISTINCTIVE for a click to testify about it:
        # among many same-colored pieces (r11l's rope) identity hops onto
        # whatever was clicked and every hop reads as a "teleport" (a
        # false law adopted at step 16 there, refuted at 38, and the
        # trajectory it bent cost 46 -> 202 actions)
        if av_col is None or int(av_col) == int(color):
            return False
        if sum(1 for o in self.last_objs if o[1] == av_col) > 2:
            return False
        # judge by APPEARANCE, not by the tracker's identity, which
        # may LAG a swap by a few steps (position-first
        # tracking), so "the avatar" is any color identity has worn this
        # level — but never the clicked object's own color
        cols = set(self.avatar_colors)
        if av_col is not None:
            cols.add(av_col)
        if self.avatar_shape_key is not None:
            cols.add(self.avatar_shape_key[0])
        cols.discard(int(color))
        if cols:
            landed = any(o[1] in cols and (o[3][0], o[3][1]) == tuple(obj_tl)
                         for o in self.last_objs)
            if landed:
                av_after = tuple(obj_tl)
            elif tuple(av_after) == tuple(obj_tl):
                av_after = tuple(av_before)
        tele = (tuple(av_after) == tuple(obj_tl)
                and tuple(av_before) != tuple(obj_tl))
        self.click_trials.append((int(color), tuple(obj_tl),
                                  tuple(av_before), tuple(av_after)))
        rows = [t for t in self.click_trials if t[0] == int(color)]
        hits = [t for t in rows if t[3] == t[1] and t[2] != t[1]]
        miss = [t for t in rows if t[3] == t[2]]
        law = self.click_laws.get(int(color))
        if law and not tele and av_after == av_before:
            del self.click_laws[int(color)]
            self.law_events.append(("refute", Law(("click", int(color)),
                                                  "teleport", True), len(rows)))
        elif not law and len(hits) >= 2 and not miss:
            self.click_laws[int(color)] = "teleport"
            self.law_events.append(("adopt", Law(("click", int(color)),
                                                 "teleport", True), len(rows)))
        if tele and int(color) in self.click_laws:
            # IDENTITY FOLLOWS THE TELEPORT once the law stands: the body
            # is the avatar-colored object now standing where the clicked
            # object was — do not wait four stale steps for the tracker
            # to notice. (A single witness does not move identity: r11l's
            # rope clicks produce one-off landings.)
            for o in self.last_objs:
                if o[1] in cols and (o[3][0], o[3][1]) == tuple(obj_tl):
                    self.avatar_shape_key = o[0]
                    self.avatar_pos = (o[3][0], o[3][1])
                    return True
        return False

    def teleport_targets(self):
        """Anchors the planner may click; a law over a CROWDED color
        (dozens of instances) would multiply the branching factor and
        is not an anchor class anyway — bounded to scarce classes."""
        out = []
        from collections import Counter as _Ct
        cnt = _Ct(o[1] for o in self.last_objs)
        for o in self.last_objs:
            if o[1] in self.click_laws and cnt[o[1]] <= 6:
                out.append((o[1], (o[3][0], o[3][1])))
        return out

    def law_refuses(self, action_name, dest_color, ctx):
        law = self.laws.get(("action", action_name))
        if law and law.refuses(ctx):
            return True
        law = self.laws.get(("color", dest_color))
        return bool(law and law.refuses(ctx))

    def _observe_translation(self, action_name, diff, objs_after_world, blocked):
        if diff.moved:
            # WAKE PHANTOMS BELONG TO A CROWD: an avatar gliding over
            # background patches of its own scale makes the just-covered
            # patch "vanish" and the just-uncovered one "appear" — paired
            # by the differ as a mover going the OPPOSITE way, and often
            # bigger than the body it shadows (observed tu93: 9px color-0
            # wake outsized the 8px body, inverting every motion vote and
            # stealing identity). A real body is distinctive; a wake's
            # (color, size) class has many static instances elsewhere in
            # the frame. Movers from crowded classes neither vote nor
            # anchor identity.
            from collections import Counter as _CW
            _colcnt = _CW(o[1] for o in objs_after_world)
            _cand = [m for m in diff.moved if _colcnt.get(m[0], 0) < 4]
            if not _cand:
                _cand = diff.moved
            # identity anchors on the LARGEST controlled mover ("the
            # biggest thing that moves under your control is you"):
            # vote-count anchoring self-reinforced a rigid body's tiny
            # attached parts (cn04's prong dots outvoted the plug and
            # the plug was never adopted)
            color, size, (dr, dc) = min(
                _cand,
                key=lambda m: (-m[1],
                               -self.motion[action_name].votes.get(m[2], 0)))
            # vote on the FULL net vector: under drift/current an action's
            # true effect is action + flow (e.g. (0,2)), and planning with
            # collapsed unit directions provably diverges.
            # ACTION6 is a coordinate click, never a translation: letting it
            # into the motion model poisons planning with phantom moves.
            if action_name not in ("ACTION6", "RESET"):
                self.motion[action_name].votes[(dr, dc)] += 1
                self.slide_lengths[action_name].append(max(abs(dr), abs(dc)))
            cand = [o for o in objs_after_world
                    if o[1] == color and o[2] == size]
            # adopt an avatar only once this action's vector is consistent —
            # a single wobble-move must not crown a panel cell as the player.
            # And an avatar is DISTINCTIVE: a class with many identical
            # instances (grid cells, tiles) can never be the player.
            if (cand and len(cand) <= 2
                    and self.motion[action_name].votes[(dr, dc)] >= 2):
                o = min(cand, key=lambda o: 0 if self.avatar_pos is None else
                        abs(o[3][0] - self.avatar_pos[0]) +
                        abs(o[3][1] - self.avatar_pos[1]))
                self.avatar_shape_key = o[0]
                old = self.avatar_pos
                new = (o[3][0], o[3][1])
                if old is not None:
                    steps = max(abs(new[0] - old[0]), abs(new[1] - old[1]))
                    for i in range(steps + 1):
                        r = old[0] + round(i * (new[0] - old[0]) / max(steps, 1))
                        c = old[1] + round(i * (new[1] - old[1]) / max(steps, 1))
                        # multi-cell moves: cells crossed by a
                        # CLASSIFIED slide give no color testimony (a
                        # slide passing over a goal cell condemned the
                        # goal color as floor); everything else keeps
                        # full testimony — withholding it broadly
                        # regressed cn04 (step-3 terrain starved and the
                        # fragile rotation-detection sequence diverged)
                        self._mark((r, c), FREE,
                                   note=(steps <= 1 or (r, c) == new
                                         or not self.is_slide(action_name)))
                self.avatar_pos = new
                self._mark(new, FREE)
                # FREE WALL PER SLIDE (Law 3): a slide stops only where
                # the next cell is blocked — every observed endpoint
                # donates one wall for nothing. Slide games otherwise
                # starve the wall map (blocked-verdicts barely fire).
                if self.is_slide(action_name) and (dr, dc) != (0, 0):
                    u = unit((dr, dc))
                    self._mark((new[0] + u[0], new[1] + u[1]), WALL)
        elif blocked:
            vec = self.motion[action_name].best()
            if vec and self.is_slide(action_name):
                vec = unit(vec)   # a blocked slide names the ADJACENT wall
            if vec and self.avatar_pos is not None:
                tr = self.avatar_pos[0] + vec[0]
                tc = self.avatar_pos[1] + vec[1]
                rr, cc = wc((tr, tc))
                if not (0 <= rr < CANVAS and 0 <= cc < CANVAS
                        and self.colors[rr, cc] >= 0):
                    # the full-vector target lies OUTSIDE the seen
                    # playfield (a long not-yet-classified slide blocked
                    # at a border marked phantom walls in the letterbox
                    # while the REAL wall's color went unlearned): the
                    # only cell a blocked move certifies universally is
                    # the adjacent one
                    u = unit(vec)
                    tr = self.avatar_pos[0] + u[0]
                    tc = self.avatar_pos[1] + u[1]
                self._mark((tr, tc), WALL)

    def forget_terrain(self):
        """THE WORLD CAN BE REWRITTEN: a layout switch (clicked button
        that swaps platform groups) invalidates every learned wall and
        occupancy fact — but not motions, identity, or color statistics
        for colors that survive (kept coarse: stats cleared too, they
        re-accumulate in a few steps)."""
        self.occ[:] = UNKNOWN
        self._noted_cells.clear()
        self.color_stats.clear()

    def mark_wall_from_failure(self, fail_pos_world, vec):
        self._mark((fail_pos_world[0] + vec[0], fail_pos_world[1] + vec[1]), WALL)

    def locate_avatar(self, objs_world):
        if self.avatar_shape_key is None:
            return None
        for o in objs_world:
            if o[0] == self.avatar_shape_key:
                self.avatar_pos = (o[3][0], o[3][1])
                return self.avatar_pos
        return self.avatar_pos

    @property
    def known_motions(self):
        out = {}
        for name, bel in self.motion.items():
            v = bel.best()
            if v:
                out[name] = v
        return out

    def typical_slide(self, action_name):
        ls = self.slide_lengths.get(action_name, [])
        return max(ls) if ls else 1

    def is_slide(self, action_name):
        """SLIDE CONCEPT (simulator-discovered, sg03/sg53/sg54): an
        action whose observed vectors share ONE direction but vary in
        magnitude moves-until-blocked (ice physics). Electing any single
        observed vector mispredicts almost everywhere and the blocked-
        verdict machinery then learns phantom walls."""
        votes = {v: n for v, n in self.motion[action_name].votes.items()
                 if v != (0, 0)}
        if len(votes) < 2:
            return False
        # SLIDES AND CURRENTS BOTH VARY MAGNITUDE — for different reasons.
        # A slide runs until blocked; a current ADDS a constant vector to
        # every action, so a step-1 move under a (0,1) current lands at
        # 2 in free flow and 1 when the flow is blocked (observed mz02:
        # ACTION4 misread as a slide, plans went move-until-blocked, L1
        # took 130 actions instead of v11's 68). Remove the estimated
        # drift before judging magnitudes: what remains of a step is one
        # magnitude; what remains of a slide still varies.
        drift = self.drift_estimate() if len(self.motion) >= 2 else (0, 0)
        if drift != (0, 0):
            votes = {(v[0] - drift[0], v[1] - drift[1]): n
                     for v, n in votes.items()}
            votes = {v: n for v, n in votes.items() if v != (0, 0)}
            if len(votes) < 2:
                return False
        units = {unit(v) for v in votes}
        if len(units) != 1:
            return False
        mag_votes = {}
        for v, n in votes.items():
            m = max(abs(v[0]), abs(v[1]))
            mag_votes[m] = mag_votes.get(m, 0) + n
        # a STEP action with attribution noise has ONE well-supported
        # magnitude (cn04: step-3 with a stray single-vote 6); a real
        # slide accumulates SEVERAL magnitudes with real support — even
        # when one corridor dominates by frequency (the dominance-ratio
        # version misread a much-used lane as a step of 11)
        supported = [m for m, n in mag_votes.items() if n >= 2]
        # DISPERSION AS EVIDENCE (belief audit, 2026-09-04) was tried:
        # "three distinct magnitudes along one direction is a slide"
        # raised slide-belief accuracy on tuning (0.67 -> 0.76) but the
        # games disagreed (st72 0 -> 2, st32 2 -> 0, holdout L1 18 ->
        # 17): classifying a slide earlier only helps once the planner
        # can act on a slide whose walls are unknown. Kept as a negative
        # result; the slide rule waits for conditioned OUTCOMES.
        return len(supported) >= 2

    @property
    def motion_options(self):
        """{action: [plausible outcome vectors]} for planning."""
        out = {}
        for name, bel in self.motion.items():
            vs = bel.plausible()
            if vs:
                out[name] = vs
        return out

    # ---------- drift decomposition (flow/current physics) ----------

    def full_vec(self, name):
        """The action's maximal observed net vector: blocked contexts only
        ever shrink motion, so the largest credible sample is the free-space
        effect (action delta + drift)."""
        votes = self.motion[name].votes
        if not votes:
            return None
        # max L1 extent among REPEATED vectors: blocked contexts shrink the
        # net vector (so prefer the largest), but a one-off teleport/glitch
        # sample must never win — require support >= 2
        repeated = [(v, n) for v, n in votes.items() if n >= 2]
        pool = repeated or list(votes.items())
        return max(pool, key=lambda kv: (abs(kv[0][0]) + abs(kv[0][1]), kv[1]))[0]

    def drift_estimate(self):
        """Global per-step drift = componentwise median of the actions'
        free-space net vectors (the common component all actions share)."""
        vecs = [self.full_vec(n) for n in self.motion if self.full_vec(n)]
        if len(vecs) < 2:
            return (0, 0)

        def modal(xs):
            """The shared component must have real consensus (>=60% of
            actions), else there is no drift on that axis — a median can
            hallucinate drift from junk votes (observed)."""
            from collections import Counter as C
            v, n = C(xs).most_common(1)[0]
            return v if n >= max(2, 0.6 * len(xs)) else 0
        return (modal([v[0] for v in vecs]), modal([v[1] for v in vecs]))

    def action_deltas(self):
        """{action: own delta (net minus drift)}. An action that repeatedly
        produced NO visible change while drift is active must be exactly
        cancelling it: its delta is -drift (how a 'lost' left-arrow under an
        eastward current is recovered by logic alone)."""
        drift = self.drift_estimate()
        out = {}
        for name in self.motion:
            fv = self.full_vec(name)
            if fv is None:
                continue
            out[name] = (fv[0] - drift[0], fv[1] - drift[1])
        if drift != (0, 0):
            for name in self.inert_actions:
                if name not in out:
                    out[name] = (-drift[0], -drift[1])
        for name in list(out):
            if self.is_slide(name):
                # slide planning delta = direction x arena: _sim_walk
                # stops at the first known wall, which IS the slide rule
                u = unit(out[name]) if out[name] != (0, 0) else None
                if u:
                    out[name] = (u[0] * 31, u[1] * 31)
        return {n: d for n, d in out.items() if d != (0, 0) or drift != (0, 0)}

    @staticmethod
    def _sim_walk(posc, delta, passable, atomic=False):
        """Walk unit cells toward delta, stopping before an impassable
        cell. MOTION ATOMICITY: a step-action's displacement is a single
        transaction — it completes fully or not at all (partial-progress
        semantics fabricated off-lattice states the planner chased into
        walls forever; only SLIDES genuinely move partially)."""
        n = max(abs(delta[0]), abs(delta[1]))
        if n == 0:
            return posc
        if atomic:
            # a step-action is a TRANSACTION evaluated at its DESTINATION
            # (bounding-box engines teleport the full step; requiring
            # passable intermediates blocked cn04's decor-colored
            # corridors, and partial progress fabricated off-lattice
            # states — both wrong for different reasons)
            nr, nc = posc[0] + delta[0], posc[1] + delta[1]
            if (0 <= nr < CANVAS and 0 <= nc < CANVAS
                    and passable[nr, nc]):
                return (nr, nc)
            return posc
        best = posc
        for i in range(1, n + 1):
            nr = posc[0] + round(delta[0] * i / n)
            nc = posc[1] + round(delta[1] * i / n)
            if not (0 <= nr < CANVAS and 0 <= nc < CANVAS) or not passable[nr, nc]:
                break
            best = (nr, nc)
        return best

    def outcomes(self, posc, passable, deltas=None, drift=None):
        """Simulated next positions per action: move (clipped), then drift
        (clipped) — position-dependent physics, the walls we know included."""
        deltas = deltas if deltas is not None else self.action_deltas()
        drift = drift if drift is not None else self.drift_estimate()
        outs = {}
        for name, delta in deltas.items():
            p1 = self._sim_walk(posc, delta, passable,
                                atomic=not self.is_slide(name))
            p2 = self._sim_walk(p1, drift, passable) if drift != (0, 0) else p1
            if p2 != posc:
                outs[name] = p2
        return outs

    # ---------- planning ----------

    def _passable(self):
        mask = self.occ != WALL
        wcs = self.wall_colors()
        if wcs:
            predicted = np.isin(self.colors, list(wcs)) & (self.occ == UNKNOWN)
            mask &= ~predicted
        return mask

    # ---------- planning under conditioned laws ----------

    def _law_features(self):
        return tuple(sorted({law.feature for law in self.laws.values()
                             if law.feature in CTX_FEATURES}))

    def _ctx_key(self, ctx, feats):
        return tuple(ctx.get(f) for f in feats)

    def _advance_ctx(self, ctx, feats, action, moved, vec):
        if not feats:
            return ctx
        n = dict(zip(feats, ctx))
        if "parity" in n:
            n["parity"] ^= 1
        if "prev_action" in n:
            n["prev_action"] = action
        if "prev_axis" in n and moved:
            n["prev_axis"] = _axis(vec)
        return tuple(n[f] for f in feats)

    def _expand(self, cur, deltas, drift, passable, feats, slides=None):
        """(pos, ctx) -> {action: (pos', ctx')}.  With no laws adopted this
        is exactly outcomes(): the product state collapses to the cell.
        A move a law refuses is a NO-OP edge that still advances the
        context (a bump that flips parity is a legitimate plan step)."""
        posc, ctx = cur
        outs = {}
        ctxd = dict(zip(feats, ctx)) if feats else self.ctx
        if slides is None:
            slides = {name: self.is_slide(name) for name in deltas}
        for name, delta in deltas.items():
            law_o = self.laws.get(("action", name)) if self.laws else None
            if law_o is not None and law_o.outcome is not None:
                pos_w = (posc[0] - ORIGIN, posc[1] - ORIGIN)
                cx = dict(ctxd)
                cx["at_edge"] = self.at_edge(pos_w, law_o.outcome[False])
                delta = law_o.outcome_for(cx)
            p1 = self._sim_walk(posc, delta, passable,
                                atomic=not slides[name])
            p2 = self._sim_walk(p1, drift, passable) if drift != (0, 0) else p1
            refused = False
            if self.laws:
                dcol = int(self.colors[p2]) if p2 != posc else -1
                refused = self.law_refuses(name, dcol, ctxd)
            if refused or p2 == posc:
                if self.laws and feats:
                    nctx = self._advance_ctx(ctx, feats, name, False, None)
                    if nctx != ctx:
                        outs[name] = (posc, nctx)
                continue
            vec = (p2[0] - posc[0], p2[1] - posc[1])
            outs[name] = (p2, self._advance_ctx(ctx, feats, name, True, vec))
        if self.click_laws:
            for color, tl in self.teleport_targets():
                tc = wc(tl)
                if not (0 <= tc[0] < CANVAS and 0 <= tc[1] < CANVAS) or tc == posc:
                    continue
                key = ("ACTION6", tl)
                vec = (tc[0] - posc[0], tc[1] - posc[1])
                outs[key] = (tc, self._advance_ctx(ctx, feats, "ACTION6", True, vec))
        return outs

    def _passable_under_laws(self, passable):
        """Cells of a color governed by a law are gated per state, not
        by the static mask (their WALL marks came from law refusals)."""
        for g in self.laws:
            if g[0] == "color":
                passable = passable | ((self.colors == g[1]) & (self.colors >= 0))
        return passable

    def plan_to(self, target_world, tol=False):
        if self.avatar_pos is None:
            return None
        deltas = self.action_deltas()
        drift = self.drift_estimate()
        if not deltas:
            return None
        passable = self._passable() & (self.colors >= 0)  # plan within seen world
        if self.laws:
            passable = self._passable_under_laws(passable)
        feats = self._law_features()
        slides = {name: self.is_slide(name) for name in deltas}   # once per plan
        start = (wc(self.avatar_pos), self._ctx_key(self.ctx, feats))
        target = wc(tuple(target_world))
        if not (0 <= target[0] < CANVAS and 0 <= target[1] < CANVAS):
            return None
        q = deque([start])
        came = {start: None}
        goal_node = None
        while q:
            cur = q.popleft()
            if cur[0] == target:
                goal_node = cur
                break
            for name, nxt in self._expand(cur, deltas, drift, passable, feats, slides).items():
                if nxt not in came:
                    came[nxt] = (cur, name)
                    q.append(nxt)
        if goal_node is None:
            # UNREACHABLE IS INFORMATION (v11 semantics restored after the
            # 0.21 study): an exact target the model cannot reach means
            # the model is INCOMPLETE — the right response is more
            # learning, not approximation. Session 19's "go to the nearest
            # cell within tolerance" committed the agent to plans built on
            # two half-learned, drift-contaminated vectors (lockstep
            # replay vs v11 on mz02: identical beliefs at step 9, v11
            # declined and kept probing -> 68 actions; tolerance planned
            # anyway -> 130). Off-lattice perception targets that
            # motivated tolerance are the caller's job to snap — or to
            # request explicitly (tol=True, docking-class games only).
            if not tol:
                return None
            tolv = max((max(abs(d[0]), abs(d[1]))
                        for d in deltas.values()), default=1)
            best = min(came, key=lambda p: (abs(p[0][0] - target[0])
                                            + abs(p[0][1] - target[1]),
                                            p[0][0], p[0][1]))
            bd = abs(best[0][0] - target[0]) + abs(best[0][1] - target[1])
            if bd > tolv or best[0] == start[0]:
                return None
            goal_node = best
        acts, node = [], goal_node
        while came[node] is not None:
            node, name = came[node]
            acts.append(name)
        return list(reversed(acts))

    def _predicted_free(self, color):
        """A FRONTIER IS WHERE PASSABILITY IS UNPREDICTED, not merely
        unvisited: once a floor color has three free witnesses and no
        wall testimony, walking every cell of it teaches nothing (sx06:
        180 of 306 actions spent visiting an open room cell by cell
        while the one informative click waited)."""
        # (a few stray wall notes on a floor color are attribution noise
        # — a swap-teleport's interpolated path, an arrival bump; the
        # ratio rule survives them; tu93's edge-walls give the floor
        # color a wall share far above it)
        f, w = self.color_stats.get(color, (0, 0))
        return f >= 3 and w * 4 <= f

    def nearest_frontier(self, strict=True):
        """Closest reachable UNKNOWN-occupancy, seen-or-unseen cell that is
        adjacent to visited space and not a predicted wall.  strict: skip
        cells whose passability the color statistics already predict
        (tier 1); strict=False is the exhaustive tier 2 walk."""
        if self.avatar_pos is None:
            return None
        deltas = self.action_deltas()
        drift = self.drift_estimate()
        if not deltas:
            return None
        passable = self._passable() & (self.colors >= 0)
        if self.laws:
            passable = self._passable_under_laws(passable)
        feats = self._law_features()
        slides = {name: self.is_slide(name) for name in deltas}
        start = (wc(self.avatar_pos), self._ctx_key(self.ctx, feats))
        q = deque([start])
        seen = {start}
        seen_pos = {start[0]}
        while q:
            cur = q.popleft()
            for name, nxt in self._expand(cur, deltas, drift, passable, feats, slides).items():
                if nxt in seen:
                    continue
                if nxt[0] not in seen_pos:
                    if (self.occ[nxt[0]] == UNKNOWN and self.colors[nxt[0]] >= 0
                            and not (strict and
                                     self._predicted_free(int(self.colors[nxt[0]])))):
                        return (nxt[0][0] - ORIGIN, nxt[0][1] - ORIGIN)
                    seen_pos.add(nxt[0])
                seen.add(nxt)
                q.append(nxt)
        return None
