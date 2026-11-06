"""FIDDLING, AND WHAT IT TEACHES.

Push, select, pull and teleport are not words the program is given;
they are what a person learns by fiddling: press the keys, click on a
thing and press the keys again, click beside a thing, click far from
everything, walk into things.  The FIDDLER is that repertoire of
moves.  The GENERAL MODEL is an effect language wide enough to hold
what the fiddling shows, without naming it in advance:

  fixed        key -> component transform (translate / extend / retract)
  selected     key -> 'the thing that wears the mark' moves by v; a
               click on a thing moves the mark to it
  relative     a click at offset o from a thing moves it by f(o): the
               whole offset (it lands on the click) or the offset
               clipped per axis to a cap (it is pulled toward the
               click), within a reach
  contact      when a thing moves by v into another and that other
               moves by v too, contact transmits motion (a push)

A key that never moved anything under the fiddling may still be a
mode key; that is left to the selection record.  Nothing here knows a
game: the words above are the program's own summaries of what it saw.
"""
import re
import os
_ENV = dict(os.environ)      # (the switches are read once: a lookup in os.environ is a Mapping call, and one stood in the planner's successor loop — 20 million calls, 60 s of a 300 s level on bp35 L2)
import numpy as np
from collections import defaultdict, Counter

from .planner import _sweep, KeyModel, effect_of, apply as apply_eff, _shift, _bbox
from . import claims as _claims


def kind(s):
    # (a material — a textured thing — is a kind of its own, apart from a
    # solid fill of the same colour: hatching refuses what the fill lets by)
    if "texture" in getattr(s, "tags", ()):
        return s.body + 16
    return s.body


def _centre(cells):
    r0, c0, r1, c1 = _bbox(cells)
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def worn_colours(s, comps, small=16):
    """The colours of the marks a thing wears: its parts, and the small
    things of another colour lying against it (a piece's docking marks,
    however the frame was cut into things)."""
    out = set()
    sb = s.bbox
    for o in comps:
        if o.sid == s.sid or "hud" in o.tags or "terrain" in o.tags:
            continue
        if o.sid >= 1000:
            if o.sid // 1000 == s.sid:
                out.add(kind(o))
            continue
        if o.size > small or o.body == s.body:
            continue
        ob = o.bbox
        if max(sb[0] - ob[2], ob[0] - sb[2], sb[1] - ob[3], ob[1] - sb[3]) > 1:
            continue                        # (boxes of touching things are one apart)
        if any(abs(r - hr) + abs(c - hc) == 1 for r, c in o.cells for hr, hc in s.cells):
            out.add(kind(o))
    return out


def _bbox_cells(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def _eff(p, s):
    """What happened to a thing between two frames — the tracker's reading
    of its whole motion first (a thing moved and redrawn within is the
    thing translated), the cell-level transform otherwise."""
    e = effect_of(p, s)
    mv = getattr(s, "moved_by", None)
    if mv is not None and p.size == s.size and e[0] in ("rotate", "translate", "morph") and e != ("translate", mv):
        return ("translate", (int(mv[0]), int(mv[1])))
    return e


def livery_host(comp, comps):
    """A tiny component lying inside a larger one's box and touching it
    marks that larger one (a selection mark).  Returns the host or None."""
    own0 = set(comp.cells)
    dotted = comp.size > 3 and all(((r + dr, c + dc) not in own0) for r, c in own0 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    # (a mark is small AGAINST ITS HOST, not small outright: the centre of
    # a block twice the size is a mark twice the size — a patch of up to a
    # sixth of the host, enclosed by it, marks it)
    if comp.size > 16 and not dotted:
        return None                 # (a pattern of lone dots, one per cell of its host, is a mark too)
    r0, c0, r1, c1 = comp.bbox
    best = None
    for h in comps:
        if h.sid == comp.sid or h.size <= comp.size or "terrain" in h.tags or "hud" in h.tags or "texture" in h.tags:
            continue
        if dotted and h.size < 4 * comp.size:
            continue
        if not dotted and comp.size > 3 and 6 * comp.size > h.size:
            continue
        hb = h.bbox
        if hb[0] <= r0 and hb[1] <= c0 and hb[2] >= r1 and hb[3] >= c1:
            # ENCLOSED: every 4-neighbour of the mark is host or mark (a
            # dot riding the END of a bar is a tip, not a mark)
            own = set(comp.cells)
            if all(((r + dr, c + dc) in h.cells or (r + dr, c + dc) in own)
                   for r, c in own for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                if best is None or h.size < best.size:
                    best = h
    return best


def part_host(comp, comps):
    """A tiny component inside a larger one's box and touching it is a
    PART of it (a facing dot in a ring, a tip in a bar) — whether or not
    it is enclosed.  Returns the host or None."""
    if comp.size > 3:
        return None
    r0, c0, r1, c1 = comp.bbox
    best = None
    for h in comps:
        if h.sid == comp.sid or h.size <= comp.size or "terrain" in h.tags or "hud" in h.tags or "texture" in h.tags:
            continue
        hb = h.bbox
        if hb[0] <= r0 and hb[1] <= c0 and hb[2] >= r1 and hb[3] >= c1:
            if any((r + dr, c + dc) in h.cells for r, c in comp.cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                if best is None or h.size < best.size:
                    best = h
    return best


def selected_hosts(comps):
    """{host sid: mark colour} for every mark in the frame."""
    out = {}
    for c in comps:
        h = livery_host(c, comps)
        if h is not None:
            out[h.sid] = c.body
    return out



def _won_colour(h):
    """The colour of the thing a winning reading was about (from the reading,
    or from its note "... C7[5@...]" when the reading did not record it)."""
    c = getattr(h, "t_color", None)
    if c is not None:
        return c
    m = re.search(r"\bC(\d+)\[", getattr(h, "note", "") or "")
    return int(m.group(1)) if m else None

class Fiddler:
    """The repertoire, as an ordered list of moves within a budget."""

    def moves(self, sprites, comps, keys, clickable, budget=40, first=()):
        """`first`: colours whose things are clicked before the rest (what a
        win before was read as: a click on a thing of that colour)."""
        plan = []
        # every key twice before anything else: a first press may be
        # swallowed (a level's opening), and one observation is no law
        plan += [k for k in keys] + [k for k in keys]
        things = [s for s in sprites if "terrain" not in s.tags and "bar" not in s.tags
                  and "hud" not in s.tags and s.size <= 160]
        # THE DISTINCT THINGS FIRST, THE BACKGROUND-LIKE LAST (Tushar, lf52):
        # a thing of a kind few things share stands out; a kind strewn many
        # times over the board is texture, fiddled with after, and one or
        # two of a kind stand for it (its things answer alike).  Within a
        # kind the larger first, and the extremes (the ends) before the
        # middle
        n_kind = Counter((s.body, s.size, s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1]) for s in things)
        n_col = Counter(s.body for s in things)

        def kind_(s):
            return (s.body, s.size, s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1])
        things.sort(key=lambda s: (s.body not in first, n_kind[kind_(s)] >= 8, n_col[s.body] >= 8, n_kind[kind_(s)], -s.size))
        picked = []; per_kind = Counter()
        for s in things:
            if per_kind[kind_(s)] >= 2:
                continue
            per_kind[kind_(s)] += 1
            picked.append(s)
        things = picked + [s for s in things if s not in picked]
        if clickable:
            for s in things[:10]:
                c = click_point(s)
                plan.append(("click", c))
                # a click may hand the keys to this thing: press them again
                # (only if the click did anything: an "if" move)
                plan += [("if", k) for k in keys[:4]]
                # the parts of a panel
                if s.parts >= 2 and s.size >= 12:
                    for p in part_points(s)[1:]:
                        plan.append(("click", p, s.sid))         # (a part's click: owed to its thing — not made once the thing is gone)
            # BESIDE A SMALL THING: does a click there act on the thing?  The
            # sides probed are read off the scene, not the screen: toward
            # the nearest like thing, at twice its offset (a leap's landing,
            # lf52), and the side with the most room — never "right and up"
            # (the invariance gauge: mirrored, the leap was never found)
            for s in things[:6]:
                if s.size <= 30:
                    for q in beside_points(s, sprites, [t for t in things if kind_(t) == kind_(s)]):
                        if ("click", q) not in plan:
                            plan.append(("click", q))
            # FAR FROM EVERYTHING: the centre, and the emptiest cell of the
            # frame (the cell farthest from any thing) — a click there tells
            # whether the board itself answers clicks
            plan.append(("click", (32, 32)))
            occupied = set()
            for s_ in sprites:
                if "terrain" not in s_.tags and s_.sid < 1000:
                    occupied |= set(s_.cells)
            best_q, best_d = (32, 32), -1
            for r_ in range(2, 64, 4):
                for c_ in range(2, 64, 4):
                    d0 = min((abs(r_ - q[0]) + abs(c_ - q[1]) for q in occupied), default=64)
                    if d0 > best_d:
                        best_q, best_d = (r_, c_), d0
            if ("click", best_q) not in plan:
                plan.append(("click", best_q))
        plan += [k for k in keys]
        return plan[:budget]


def click_point(s):
    """Where to click a thing: its own cell nearest its centre (a hollow
    thing's centre is not on it)."""
    r, c = s.centre
    return min(s.cells, key=lambda q: abs(q[0] - r) + abs(q[1] - c))

def beside_points(s, sprites, like=(), step=5):
    """Where to click BESIDE a small thing, read off the scene, not the
    screen: toward the nearest like thing at twice its offset (a leap's
    landing — lf52), and the side with the most room.  Never "right and
    up": the invariance gauge found the leap unfound in a mirrored game."""
    occupied = set()
    for s_ in sprites:
        if "terrain" not in s_.tags and s_.sid < 1000:
            occupied |= set(s_.cells)
    r, c = click_point(s)
    probes = []
    like = sorted([t for t in like if t is not s], key=lambda t: abs(t.centre[0] - s.centre[0]) + abs(t.centre[1] - s.centre[1]))
    if like:
        # beyond a like neighbour, where nothing stands: a leap's landing
        # (the nearest neighbour whose beyond is free; a landing on a thing
        # is a click on that thing, not a probe of the place)
        landing = None
        for t in like[:4]:
            v = (t.centre[0] - s.centre[0], t.centre[1] - s.centre[1])
            q = (int(round(r + 2 * v[0])), int(round(c + 2 * v[1])))
            if q not in occupied:
                landing = q; break
        if landing is not None:
            probes.append(landing)
        probes.append(click_point(like[0]))                                      # (the neighbour itself)

    def room(d):
        n_ = 0
        for k_ in range(1, 12):
            q = (r + d[0] * k_, c + d[1] * k_)
            if not (0 <= q[0] < 64 and 0 <= q[1] < 64) or q in occupied:
                break
            n_ += 1
        return n_
    d_ = max(((0, 1), (0, -1), (1, 0), (-1, 0)), key=room)
    probes.append((r + d_[0] * step, c + d_[1] * step))
    return [q for q in probes if 0 <= q[0] < 64 and 0 <= q[1] < 64]



def part_points(s):
    pts = [click_point(s)]
    by_col = defaultdict(list)
    for p, col in s.cells.items():
        by_col[col].append(p)
    for col, cells in by_col.items():
        if len(cells) > s.size // 2:
            continue
        cs = set(cells)
        while cs:
            start = cs.pop(); comp = {start}; st = [start]
            while st:
                r, c = st.pop()
                for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if q in cs:
                        cs.discard(q); comp.add(q); st.append(q)
            rs = [r for r, _ in comp]; ccs = [c for _, c in comp]
            pts.append((int(round(sum(rs) / len(rs))), int(round(sum(ccs) / len(ccs)))))
    return pts


class GeneralModel:
    def __init__(self):
        self.fixed = KeyModel()
        # selection
        self.sel_moves = defaultdict(list)     # key -> [(host sid, v)] moves of a marked thing
        self.key_selected = {}                  # key -> v : the key moves the marked thing
        self.select_click = Counter()           # kind -> times a click on it drew the mark
        self.select_kinds = set()
        # A LAW LEARNED ON KINDS IS PRESUMED OF LIKE-FORMED THINGS: what
        # every selectable thing wore last level (its marks' colour) is the
        # selectable FORM; a thing of a new kind wearing it is presumed
        # selectable until a click says otherwise
        self.select_form = set()                # mark colours every selectable thing wore
        self.select_presumed = set()            # kinds presumed selectable by their form (untested)
        self.select_refused = set()             # kinds a click did not select
        # relative clicks
        self.rel = defaultdict(list)            # kind -> [(offset, displacement)]
        self.centroid_of = {}                   # sid -> (handle sids): a thing that sits at the centroid of others (a disc hung between its handles)
        self.centroid_kinds = {}                # kind -> (handle size, handle dims): the same law, carried by form
        self.rel_miss = defaultdict(list)       # kind -> [|offset| of clicks that did nothing]
        # contact
        self.pushable = set()                   # kinds that moved when walked into
        self.passable = set()                   # (kind, kind): things seen to stand on each other's cells — not solid to one another
        self.slid_over = Counter()              # colours a shoved thing was seen to slide over (a pit the pusher may not enter)
        self.dock_colour = None                 # the colour a docked pair of marks wears (learned at the first docking)
        self.legend_map = {}                    # glyph colour -> "same"/"other" than the card's centre (the legend within)
        self.bound = set()                      # (sid, sid): pieces joined by a docking — one body until seen apart
        self.push_vec = {}                      # (kind, dir r, dir c) -> (displacement when pushed, pusher moved too)
        self.push_stopped = {}                  # (kind, dir r, dir c) -> the shove came to rest AGAINST something (a slide until stopped)
        self.push_step = {}                     # (kind, dir r, dir c) -> the pusher's own step that press
        self.last_selected_pos = None
        self.walls = set()                      # kinds that refused a walk
        self.wall_kinds = Counter()             # kind -> times a thing of it refused the walker
        self.marked_before = {}
        self.select_color = None                # the mark colour that changes hands
        self.mark_votes = Counter()             # mark colour -> times its wearer moved under keys
        self.cycle_keys = {}                    # key -> [(from host, to host)] the mark moved under a key
        self.rel_keys = set()                   # click keys explained by a relative law (not buttons)
        self.last_sel = None
        self.flat = set()                       # kinds a mover has lain over (not walls)
        self.flat_size = {}                     # kind -> the largest thing of it lain over (a block of the mark's colour is not flat)
        self.ground_ok = set()                  # colours a move landed on
        self.ground_size = {}                   # colour -> the most cells of it one move landed on (a mark's colour is ground only as a mark)
        self.ground_refused = Counter()         # colour -> refusals of a move onto it
        self.stayed = Counter()                 # (key, sid) -> presses where others moved and it did not
        self.contact_seen = {}                  # (key, sid) -> (kind, dir r, dir c, displacement) of a contact
        self.pushed_count = Counter()           # (key, sid) -> presses on which the thing moved with the mover against it (pushed, not its own law)
        self.refused_at = set()                 # (kind, r0, c0, v): a move refused from that place
        self.wall_patches = set()               # cells of a walkable colour's PATCH that refused the body (a hatched fence of the floor's colour)
        self.silent_click_kinds = Counter()     # kind -> clicks on it that changed nothing (carried: two and the kind is not clicked again to see)
        self.uncovered = Counter()              # colour -> cells of it shown where the active mover stood (the ground under it)
        self.refused_seen = {}                  # entry -> what the frame showed where the move would have gone
        self.kind_seen = {}                     # sid -> body colour, for things no longer in view
        self.kinds_worn = defaultdict(set)      # sid -> every body colour it wore
        self.presses_total = 0
        self.change_presses = Counter()         # sid -> presses at which it changed
        self.change_keys = defaultdict(set)     # sid -> keys under which it changed
        self.counters = {}                      # sid -> (kind, cells per press): the game's clock
        self.triggers = defaultdict(list)       # kind arrived on -> [(display sid, key)]
        self.displays = Counter()               # sid -> times it changed on an arrival
        self.tried_nudges = set()               # (sid, direction, lane) experiments already made
        self.undo_keys = set()                  # keys that take the last action back
        self.livery_votes = Counter()           # frozenset of (dr, dc, colour) from the body's corner -> sights of an outline that moves with it
        self.slide_votes = Counter()            # direction -> sights of the controlled thing sliding on until stopped
        self.gravity_votes = Counter()          # direction -> falls seen within a press's frames (THE PATH WITHIN A PRESS)
        self.gravity = None                     # the way the body falls after every press, when nothing holds it
        self.paths = {}                         # key -> the last path of the body within the press (cell steps per frame)
        self.support = Counter()                # colour -> times the body came to rest on it (what holds the body up under gravity)
        self.slide_recent = []                  # the last few sights for/against the slide (recency outweighs tally)
        self.slide_flips = []                   # (old direction, new direction, kinds touched before): the slide's law changed
        self.slide_against = Counter()          # direction -> moves after which it could have slid and did not
        self.move_keys = defaultdict(set)       # (kind, vector) -> keys under which a thing of the kind moved so
        self.selfmovers = set()                 # kinds that move on their own (a clockwork)
        self.own_ways = defaultdict(lambda: defaultdict(list))   # kind -> sid -> vectors it moved by on its own (not the body's, not pushed)
        self.agents = set()                     # kinds that travel on their own, choosing their way (an ally, a foe)
        from .agents import AgentRecord
        self.agent_record = AgentRecord()       # THE PARALLEL RECORD: each agent's steps, want, acts, sign and next step
        self.in_hands = {}                      # sid -> (agent sid, bbox): a thing last seen going with an agent
        self.last_way = {}                      # agent sid -> its last own vector
        self.stalled = {}                       # agent sid -> presses it has stood still with our body in its way
        self.rests = defaultdict(list)          # kind -> [bbox]: where things an agent carried came to rest (what the helper does with them)
        self.pos_evidence = defaultdict(list)   # key -> presses that looked like a track (moved dicts by kind)
        self.undo_votes = defaultdict(set)      # key -> previous keys after which it restored the frame
        self._last_key = None
        self.pos_eff = defaultdict(dict)        # key -> (kind, r0, c0) -> displacement seen from that slot
        self.pos_bad = defaultdict(set)         # key -> slots with two different answers
        self.pos_kinds = defaultdict(set)       # key -> kinds of the things it moved by slot
        self.positional = set()                 # keys whose effect depends on where the thing is
        self._frames = []                       # the two frames before the current one
        self.comove = Counter()                 # (sid, sid) -> presses they moved together
        self.moved_count = Counter()            # sid -> presses it moved
        self.body_moves = defaultdict(list)     # key -> [(body, v)]
        self.active = None                      # the body the keys moved last
        self.ever_active = set()
        self.pending_cycle = None               # (key, active before) of a no-effect press
        self.pending_select = None              # (sid, kind, click key): a click recoloured a thing; did it select it?
        self.select_by_colour = {}              # sid -> kind: things a click selected by recolouring them
        self.selected_colour = None             # the colour a thing wears while selected (selection shown by colour)
        self.key_rotate = {}                    # key -> quarter turns: the key turns the selected body
        self.cycle_bodies = {}                  # key -> [(body before, body after)]
        self.paints = defaultdict(list)         # key -> [(sid, cells recoloured, colour)]: a key that recolours a thing in place
        self.colour_clicks = []                 # [(click, clicked sid, host sid, recoloured sid, colour, cells from its corner)]: a click that hands a colour to a thing
        self.camera = (0, 0)                    # where the screen is in the world (moved by scrolls)
        self.stops = set()                      # world cells of the wall's colour the body has rested against (along the slide)
        self.click_gone = Counter()             # kind -> clicks after which the clicked thing was gone (a breakable)
        self.click_clears = {}                  # kind -> (up, left, down, right): how far beyond its box the removal clears
        self.click_marks = Counter()            # kind -> removals that left a dotted mark of the kind (a toggle: the mark clicked is the thing again)
        self.click_gone_size = {}               # kind -> the smallest thing of the kind a click took away (a mark of it is not it)
        self.click_grows = {}                   # kind -> (mark size, (up, left, down, right)): a click on a mark of the kind raises the thing (a toggle back)
        self.death_touch = Counter()            # kind -> times the body touched it on a press that lost the level
        self.touched_alive = Counter()          # kind -> times the body touched it and lived
        self.death_enter = Counter()            # kind -> times the body STEPPED ONTO it on a press that lost the level or put it back
        self.entered_alive = Counter()          # kind -> times the body stood on it and lived
        self.prior_death_touch = Counter()      # the same four, carried from earlier levels: priors, outranked by this level's own counts
        self.prior_touched_alive = Counter()
        self.prior_death_enter = Counter()
        self.prior_entered_alive = Counter()
        self.world = {}                         # (world row, world col) -> colour: the board as remembered across scrolls (micromouse)
        self.click_stays = Counter()            # kind -> clicks after which it stayed
        self.click_under = {}                   # click key -> kind of the smallest thing under it (a BUTTON is known by its kind, and carries)
        self.click_under_form = {}              # click key -> (box h, box w, cells) of that thing
        self.journeys = []                      # [(key, {sid: [steps]}, frame before)]: a press sent a thing on a journey and put it back (a launch)
        self.flips = {}                         # colour -> colour: a click on a small mark recolours it in place (a toggle)
        self.toggle_cells = set()               # cells of marks a click has flipped
        self.leap = None                        # {kind, pitch, ...}: select a thing, point two pitches away over its neighbour — it leaps there, the neighbour is gone
        self._last_click = None
        self.carry = None                       # {key, kinds, held, offset, toggle}: a key pressed against a thing binds it to the body; it comes along
        self.colour_targets = Counter()         # sid -> times a click recoloured it
        self.clock = None                       # the counter every press spends: the time left
        self.clock_colour = None                # the colour it spends (None: it shrinks)
        self.clock_rate = 0                     # cells it loses per press
        self.clock_full = 0                     # the most it has held
        self.clock_box = None                   # where it is (its identity may churn)
        self.clock_votes = Counter()            # (sid, colour, cells per press) -> presses that spent it so
        self.refills = Counter()                # kind -> arrivals on it that refilled the clock (the visit uses it up)
        self.refills_here = Counter()           # the same, this level only (what this level shows outranks what was carried)
        self.drains = Counter()                 # kind -> arrivals on it that SPENT the clock beyond its rate (a drain: the refill's twin)
        self.deaths = 0                         # times the clock ran out
        self.spawn = {}                         # sid -> bbox where the controlled thing is put back
        self.portals = {}                       # (kind, r0, c0, key) -> (r0, c0): a press from that place lands the thing there
        from .thinglaws import ThingLaws
        self.thing_laws = ThingLaws()           # laws BETWEEN things (an image is a shape mirrored across a line)

    # ------------------------------------------------------------ observe
    def scroll_memories(self, u):
        """The screen moved by u under the world: every place remembered is
        now at place + u."""
        dr, dc = int(u[0]), int(u[1])
        self.refused_at = {(kd, r0 + dr, c0 + dc, v) for (kd, r0, c0, v) in self.refused_at}
        self.refused_seen = {(kd, r0 + dr, c0 + dc, v): frozenset(((q[0] + dr, q[1] + dc), col) for q, col in seen)
                             for (kd, r0, c0, v), seen in self.refused_seen.items()}
        self.portals = {(kd, r0 + dr, c0 + dc, k): (r1 + dr, c1 + dc) for (kd, r0, c0, k), (r1, c1) in self.portals.items()}
        self.spawn = {sid: (b[0] + dr, b[1] + dc, b[2] + dr, b[3] + dc) for sid, b in self.spawn.items()}
        if self.clock_box is not None:
            b = self.clock_box; self.clock_box = (b[0] + dr, b[1] + dc, b[2] + dr, b[3] + dc)
        for k_ in list(self.pos_eff):
            self.pos_eff[k_] = {(r0 + dr, c0 + dc): v for (r0, c0), v in self.pos_eff[k_].items()}
        for k_ in list(self.pos_bad):
            self.pos_bad[k_] = {(r0 + dr, c0 + dc) for (r0, c0) in self.pos_bad[k_]}
        if self.last_selected_pos is not None and isinstance(self.last_selected_pos, tuple) and len(self.last_selected_pos) == 2:
            self.last_selected_pos = (self.last_selected_pos[0] + dr, self.last_selected_pos[1] + dc)
        self.camera = (getattr(self, "camera", (0, 0))[0] - dr, getattr(self, "camera", (0, 0))[1] - dc)

    def observe(self, key, click, before, after, frame=None, frame_after=None, scroll=None):
        # THE TWO WERE TWO ALL ALONG: when this press split a thing (the
        # tracker: a piece born with split_from and the vector it moved by —
        # set on self._splits by the game loop), the frame before is read
        # with the piece as a thing of its own where it stood.  The push
        # record is then the piece's (it moved by the vector, hit by the
        # mover); the parent, which only lost our misreading, is given no
        # law and no displacement from this press.  (tw01: read as the
        # parent's, the press taught 'the key retracts the block' and a push
        # vector off the parent's shifted centre.)
        splits_ = list(getattr(self, "_splits", ()) or ())
        if splits_ and _ENV.get("SPLIT_READ"):      # (experiment, off: see session 37p — the press that split tw01's pair was the HELPER's push, and the tracker then swapped the halves)
            from .entities import Sprite as _Sp
            a_by0_ = {s_.sid: s_ for s_ in after}
            bef_ = list(before)
            for par_, piece_, vec_ in splits_:
                pc_ = a_by0_.get(piece_)
                pi_ = next((i_ for i_, s_ in enumerate(bef_) if s_.sid == par_), None)
                if pc_ is None or pi_ is None:
                    continue
                old_ = {(r - vec_[0], c - vec_[1]): col for (r, c), col in pc_.cells.items()}
                par0_ = bef_[pi_]
                if not set(old_) <= set(par0_.cells):
                    continue
                kept_ = {q_: col for q_, col in par0_.cells.items() if q_ not in old_}
                if not kept_:
                    continue
                np_ = _Sp(sid=par0_.sid, cells=kept_, parts=par0_.parts, tags=set(par0_.tags))
                pp_ = _Sp(sid=pc_.sid, cells=old_, parts=1, tags=set(pc_.tags))
                bef_[pi_] = np_
                bef_.append(pp_)
            before = bef_
        # THE SCREEN MOVED, NOT THE WORLD: what is seen after a scroll is
        # read back in the frame before it (the walker's motion is its
        # motion in the world; what only scrolled did not move); the places
        # remembered are shifted after
        after_screen = after
        if scroll is not None:
            u = (int(scroll[0]), int(scroll[1]))
            from .entities import Sprite as _Sp
            aft = []
            for s_ in after:
                s2 = _Sp(sid=s_.sid, cells={(r - u[0], c - u[1]): col for (r, c), col in s_.cells.items()}, parts=s_.parts, tags=set(s_.tags))
                mv_ = getattr(s_, "moved_by", None)
                s2.moved_by = (mv_[0] - u[0], mv_[1] - u[1]) if mv_ is not None else None
                if s2.moved_by == (0, 0):
                    s2.moved_by = None
                for a_ in ("source", "derived"):
                    if hasattr(s_, a_):
                        setattr(s2, a_, getattr(s_, a_))
                aft.append(s2)
            after = aft
            frame_after_screen = frame_after
            if frame_after is not None:
                fa = np.full(frame_after.shape, -1, dtype=frame_after.dtype)
                H_, W_ = frame_after.shape
                r0s, r1s = max(0, u[0]), min(H_, H_ + u[0])
                c0s, c1s = max(0, u[1]), min(W_, W_ + u[1])
                fa[r0s - u[0]:r1s - u[0], c0s - u[1]:c1s - u[1]] = frame_after[r0s:r1s, c0s:c1s]
                frame_after = fa
            if _ENV.get("MOVE_DEBUG"):
                print(f"   scrolled by {u}: read back in the old frame")
            self._scroll_now = True
            try:
                return self._observe(key, click, before, after, frame=frame, frame_after=frame_after)
            finally:
                self._scroll_now = False
                self.scroll_memories(u)
                self.remember(frame_after_screen)
                self.note_rest(after_screen, frame_after_screen)
        self._scroll_now = False
        try:
            return self._observe(key, click, before, after, frame=frame, frame_after=frame_after)
        finally:
            self.remember(frame_after)
            self.note_rest(after_screen, frame_after)

    def note_rest(self, comps, frame_after):
        """WHAT THE BODY RESTS AGAINST IS A STOP: with a slide known, the
        body comes to rest after every press; the wall-coloured cells just
        beyond it in the slide's direction held it — a line the eye might
        read as a drawing is, there, a wall.  Kept in the world's frame."""
        d_ = self.slide()
        if d_ is None or frame_after is None or not self.active:
            return
        body = set()
        for s_ in comps:
            if s_.sid in self.active:
                body |= set(s_.cells)
        if not body:
            return
        H_, W_ = frame_after.shape
        blocking = self.blocking()
        beyond = {(r + d_[0], c + d_[1]) for (r, c) in body} - body
        cam = self.camera
        for (r, c) in beyond:
            if 0 <= r < H_ and 0 <= c < W_ and int(frame_after[r, c]) in blocking:
                self.stops.add((r + cam[0], c + cam[1]))

    def is_ground(self, kd, size=None):
        """A colour a move landed on is ground — but a THING of that colour
        that is also known solid in another form (a block of the mark's
        colour: removed by clicks, or raised) is ground only when it is no
        bigger than what the body ever stood on."""
        if kd not in self.ground_ok:
            return False
        if size is None:
            return True
        two_forms = kd in self.click_gone_size or kd in self.click_grows or kd in self.wall_kinds or kd in self.blocking()
        if two_forms and size > 3 * max(1, self.ground_size.get(kd, 0)):
            return False
        return True

    def is_flat(self, kd, size=None):
        """A kind the body has lain over is flat — a thing of the kind no
        larger than twice what it lay over; a block raised in a mark's
        colour is not the mark."""
        if kd not in self.flat:
            return False
        if size is None:
            return True
        return size <= 2 * self.flat_size.get(kd, 10 ** 6)

    def refuse(self, entry, frame=None, size=(6, 6)):
        """A MOVE REFUSED FROM A PLACE is remembered with what stood where
        it would have gone; when that changes (the block became a mark,
        the door opened) the refusal lapses — the way may be open now."""
        self.refused_at.add(entry)
        if frame is None:
            frame = getattr(self, "_frame_now", None)
        if frame is None:
            return
        kd, r0, c0, v = entry
        H_, W_ = frame.shape
        h_, w_ = size
        dest = {(r0 + v[0] + i, c0 + v[1] + j) for i in range(h_) for j in range(w_)} - {(r0 + i, c0 + j) for i in range(h_) for j in range(w_)}
        self.refused_seen[entry] = frozenset((q, int(frame[q[0], q[1]])) for q in dest if 0 <= q[0] < H_ and 0 <= q[1] < W_)

    def refused_now(self, frame):
        """The refusals still standing: those whose destination looks as it
        did when the move was refused."""
        if frame is None:
            return set(self.refused_at)
        H_, W_ = frame.shape
        out = set()
        for e in self.refused_at:
            seen = self.refused_seen.get(e)
            if seen is None or all(int(frame[q[0], q[1]]) == col for q, col in seen if 0 <= q[0] < H_ and 0 <= q[1] < W_):
                out.add(e)
        return out

    def _touching_kinds(self, comps):
        """Kinds of the things (not ground, not the body) that touch the
        controlled body now — 8-neighbours or overlap."""
        if not self.active:
            return set()
        body = set()
        for s_ in comps:
            if s_.sid in self.active:
                body |= set(s_.cells)
        ring = {(r + dr, c + dc) for (r, c) in body for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        out = set()
        for s_ in comps:
            if s_.sid in self.active or "terrain" in s_.tags or "hud" in s_.tags or s_.sid >= 1000:
                continue
            if kind(s_) in self.ground_ok or kind(s_) in self.flat:
                continue
            if ring & set(s_.cells):
                out.add(kind(s_))
        return out

    def _entered_kinds(self, comps, body=None):
        """Kinds of the things the controlled body STANDS ON now (its cells
        on theirs) — what it entered, as against what it merely touches."""
        if not self.active:
            return set()
        if body is None:
            body = set()
            for s_ in comps:
                if s_.sid in self.active:
                    body |= set(s_.cells)
        out = set()
        for s_ in comps:
            if s_.sid in self.active or "terrain" in s_.tags or "hud" in s_.tags or s_.sid >= 1000:
                continue
            if kind(s_) in self.ground_ok:
                continue
            if body & set(s_.cells):
                out.add(kind(s_))
        return out

    def carried_kinds(self):
        """The kinds a carried thing wears (the carry law's kinds, its core,
        the colour it wears when held): a thing of these moving with the
        body is carried by it — what IT passes says nothing of the body's
        way (wa30 L3: the block set into the fence made the fence flat,
        and the body walked into it)."""
        cr_ = getattr(self, "carry", None)
        if not cr_:
            return set()
        out = set(cr_.get("kinds", ()) or ())
        if cr_.get("held_colour") is not None:
            out.add(cr_["held_colour"])
        if cr_.get("core"):
            out.add(cr_["core"][0])
        return out

    def passable_kinds(self, kd):
        """The kinds a thing of kind `kd` has been seen to pass through."""
        return {b for (a, b) in self.passable if a == kd}

    def note_alive(self, comps):
        for kd in self._touching_kinds(comps):
            self.touched_alive[kd] += 1
        for kd in self._entered_kinds(comps):
            self.entered_alive[kd] += 1
        self._kin_comps = comps

    def hazard_kin(self, comps=None):
        """THE WHOLE OF A HAZARD: a kind every thing of which touches a
        hazard (the blades above the spike's base) is of the hazard — a
        thing is all its colours, and what kills is the whole thing."""
        comps = comps if comps is not None else getattr(self, "_kin_comps", None)
        base = self._deadly_kinds(self.death_touch, self.touched_alive, self.prior_death_touch, self.prior_touched_alive)
        if not comps or not base:
            return set()
        things = [s_ for s_ in comps if s_.sid < 1000 and "terrain" not in s_.tags and "hud" not in s_.tags
                  and s_.sid not in (self.active or ()) and kind(s_) not in self.ground_ok and kind(s_) not in self.flat]
        haz_cells = set()
        for s_ in things:
            if kind(s_) in base:
                haz_cells |= {(r + dr, c + dc) for (r, c) in s_.cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        out = set()
        by_kind = defaultdict(list)
        for s_ in things:
            if kind(s_) not in base:
                by_kind[kind(s_)].append(s_)
        for kd, ss in by_kind.items():
            if kd in self.removable() or self.touched_alive.get(kd, 0) + self.prior_touched_alive.get(kd, 0) > 2 * max(1, len(ss)):
                continue                        # (a kind the body has often touched and lived is not of it)
            if all(set(s_.cells) & haz_cells for s_ in ss):
                out.add(kd)
        return out

    def note_death(self, comps, before=None, key=None, put_back=False, frame=None):
        """THE LEVEL WAS LOST ON THIS PRESS: what the body touched, and had
        never touched and lived, is a hazard (the hunter's wall, a spike).
        When the body is gone from the frame, its last place says: what
        touched it there, and the first thing along its slide (a fall onto
        spikes) or along the key's move.  `put_back`: the body was not lost
        but set back to its start — the same lesson, without the loss (the
        body's place now says nothing: it is where it began).
        Apart from what was touched, what the step ENTERED — the body's
        cells after the key's move on a thing's cells — is counted on its
        own: a cell one may stand beside but not on (a hazard strip a
        corridor runs along)."""
        kinds_ = set() if put_back else set(self._touching_kinds(comps))
        entered_ = set()
        if before is not None and self.active and key is not None and isinstance(key, str):
            e_ = self.fixed.moving(key)
            for s_ in before:
                if s_.sid not in self.active:
                    continue
                law_ = e_.get(s_.sid)
                if law_ is None or law_[0] != "translate":
                    continue
                dest_ = {(r + law_[1][0], c + law_[1][1]) for (r, c) in s_.cells}
                entered_ |= self._entered_kinds(before, body=dest_)
                if _ENV.get("DEATH_DEBUG"):
                    print(f"   [death] put_back {put_back} key {key} body S{s_.sid} {s_.bbox} law {law_} dest {_bbox_cells(dest_)} near {[(x.sid, kind(x), x.bbox, sorted(x.tags)) for x in before if x.sid != s_.sid and x.sid < 1000 and max(x.bbox[0] - _bbox_cells(dest_)[2], _bbox_cells(dest_)[0] - x.bbox[2], x.bbox[1] - _bbox_cells(dest_)[3], _bbox_cells(dest_)[1] - x.bbox[3]) <= 1]}")
        for kd in entered_:
            self.death_enter[kd] += 1
        if before is not None and self.active and not put_back:
            body = set()
            for s_ in before:
                if s_.sid in self.active:
                    body |= set(s_.cells)
            if body:
                kinds_ |= self._touching_kinds(before)
                # (the first thing along the key's move — what it walked
                # into — and the first along the slide from where the move
                # put it: what it fell onto)
                rays_ = []
                v_ = None
                if isinstance(key, str):
                    e_ = self.fixed.moving(key)
                    sid_v = next((sid_ for sid_, e in e_.items() if sid_ in self.active and e[0] == "translate"), None)
                    v_ = e_[sid_v][1] if sid_v is not None else None
                    if v_ is not None:
                        # (a step walks into what lies within one step; only a
                        # slide goes on — the blame reaches no farther than the
                        # move does: the thing far down the row is not what
                        # killed a walker — st20, the collectible blamed)
                        reach_ = 64 if (key, sid_v) in getattr(self.fixed, "slides", ()) else max(abs(v_[0]), abs(v_[1]))
                        rays_.append((body, ((v_[0] > 0) - (v_[0] < 0), (v_[1] > 0) - (v_[1] < 0)), reach_))
                d_ = self.slide()
                if d_ is not None:
                    from_ = {(r + v_[0], c + v_[1]) for (r, c) in body} if v_ is not None else body
                    rays_.append((from_, d_, 64))
                things = [s_ for s_ in before if s_.sid not in self.active and "terrain" not in s_.tags and "hud" not in s_.tags
                          and s_.sid < 1000 and kind(s_) not in self.ground_ok and kind(s_) not in self.flat and kind(s_) not in self.removable()]
                for from_, d_, reach_ in rays_:
                    if reach_ >= 64 and frame is not None:
                        # A SLIDE STOPS WHERE THE WAY IS BLOCKED: the body's
                        # landing is found by walking the ray to the first
                        # cell of a blocking colour or of a solid thing (or the
                        # frame's edge); what the landing stands on is ENTERED
                        # (a hazard one may cross but not rest on); what lies
                        # just beyond is touched
                        H_, W_ = frame.shape
                        blk_ = self.blocking()
                        solid_ = set()
                        for s_ in before:
                            if s_.sid not in self.active and "hud" not in s_.tags and s_.sid < 1000 \
                                    and kind(s_) not in self.ground_ok and kind(s_) not in self.flat and kind(s_) not in self.passable_kinds(kind(next((x for x in before if x.sid in self.active), s_))):
                                solid_ |= set(s_.cells)
                        land_ = set(from_)
                        for k_ in range(1, 64):
                            nxt_ = {(r + d_[0], c + d_[1]) for (r, c) in land_}
                            if any(not (0 <= r < H_ and 0 <= c < W_) or int(frame[r, c]) in blk_ or (r, c) in solid_ for (r, c) in nxt_ - land_):
                                break
                            land_ = nxt_
                        stood_ = [s_ for s_ in before if s_.sid not in self.active and s_.sid < 1000 and "terrain" not in s_.tags and "hud" not in s_.tags
                                  and set(s_.cells) & land_]
                        for s_ in stood_:
                            entered_.add(kind(s_)); self.death_enter[kind(s_)] += 1
                        beyond_ = {(r + d_[0], c + d_[1]) for (r, c) in land_} - land_
                        kinds_ |= {kind(s_) for s_ in things if set(s_.cells) & beyond_}
                        continue
                    for k_ in range(1, max(2, int(reach_) + 1)):
                        ray = {(r + d_[0] * k_, c + d_[1] * k_) for (r, c) in from_}
                        hit = [s_ for s_ in things if ray & set(s_.cells)]
                        if hit:
                            kinds_ |= {kind(s_) for s_ in hit}
                            break
                if _ENV.get("DEATH_DEBUG"):
                    print(f"   [death] key {key} body {_bbox(body)} rays {[(_bbox(f_), d_, r_) for f_, d_, r_ in rays_]} touching-now {sorted(self._touching_kinds(comps))} kinds {sorted(kinds_)} comps-now {len(comps)} things {[(s_.sid, kind(s_), s_.bbox) for s_ in things][:12]}")
        found = []
        for kd in kinds_:
            self.death_touch[kd] += 1
            if self._deadly(kd, self.death_touch, self.touched_alive, self.prior_death_touch, self.prior_touched_alive):
                found.append(kd)
        for kd in entered_:
            if self._deadly(kd, self.death_enter, self.entered_alive, self.prior_death_enter, self.prior_entered_alive) and kd not in found:
                found.append(kd)
        return found

    @staticmethod
    def _deadly(kd, death, alive, pdeath, palive):
        """A kind is deadly by this level's own count where it has one; the
        carried counts decide only for kinds this level has not judged."""
        if death.get(kd, 0) or alive.get(kd, 0):
            return death.get(kd, 0) > alive.get(kd, 0)
        return pdeath.get(kd, 0) > palive.get(kd, 0)

    def _deadly_kinds(self, death, alive, pdeath, palive):
        return {kd for kd in set(death) | set(alive) | set(pdeath) | set(palive) if self._deadly(kd, death, alive, pdeath, palive)}

    def hazard_touch(self):
        """Kinds the body may not so much as touch."""
        return self._deadly_kinds(self.death_touch, self.touched_alive, self.prior_death_touch, self.prior_touched_alive) | self.hazard_kin()

    def hazard_enter(self):
        """Kinds the body may not stand on (beside them is fine) — and the
        drains of the clock, unless this level has shown the kind refilling."""
        return self._deadly_kinds(self.death_enter, self.entered_alive, self.prior_death_enter, self.prior_entered_alive) \
            | {kd for kd, n_ in self.drains.items() if n_ and not self.refills_here.get(kd)}

    def hazards(self):
        return self.hazard_touch() | self.hazard_enter()

    def remember(self, frame_after):
        """THE BOARD REMEMBERED: every cell seen, kept in the world's frame
        (screen + camera) so what scrolled off is still known — a
        micromouse's map.  A cell seen again is overwritten."""
        if frame_after is None:
            return
        cam = self.camera
        H_, W_ = frame_after.shape
        w = self.world
        for r_ in range(H_):
            row_ = frame_after[r_]
            wr = r_ + cam[0]
            for c_ in range(W_):
                v_ = int(row_[c_])
                if v_ >= 0:
                    w[(wr, c_ + cam[1])] = v_

    def _observe(self, key, click, before, after, frame=None, frame_after=None):
        # UNDO: a key after which the frame is exactly the frame before
        # the previous action — it takes back the last action; nothing
        # it "did" is an effect of its own
        if frame is not None and frame_after is not None and not isinstance(key, tuple):
            # (the HUD — a bar, a strip, a column of energy, a counter — is
            # not the world: masked cell by cell, not by row)
            keep = np.ones(frame_after.shape, dtype=bool)
            for s_ in after + before:
                if "hud" in s_.tags or "bar" in s_.tags or s_.sid in self.counters or s_.sid == self.clock:
                    for (r_, c_) in s_.cells:
                        if 0 <= r_ < keep.shape[0] and 0 <= c_ < keep.shape[1]:
                            keep[r_, c_] = False
            # THE STATE TAKEN BACK is the world before the last press that
            # changed it (a silent press in between does not count)
            fr_ = getattr(self, "_before_effective", None)
            restored = fr_ is not None and frame_after.shape == fr_.shape and (frame_after[keep] == fr_[keep]).all()
            if _ENV.get("UNDO_DEBUG") and not isinstance(key, tuple):
                print(f"   undo? {key} after {getattr(self, '_last_effective', None)}: restored {restored} changed {not (frame_after[keep] == frame[keep]).all()} votes {dict(self.undo_votes)}")
            if restored and not (frame_after[keep] == frame[keep]).all():
                # an inverse key (down after up) also restores the frame:
                # an undo restores it after DIFFERENT previous changes (two
                # keys that made the same change — left, and an undo of a
                # right — are one vote)
                sig_ = getattr(self, "_last_sig", getattr(self, "_last_effective", self._last_key))
                # (a press that moved everything exactly back over the last
                # press's steps is the opposite arrow walking back, whatever
                # the laws say so far: no vote for an undo)
                b_by_ = {s_.sid: s_ for s_ in before}
                this_vs_ = frozenset((s_.bbox[0] - b_by_[s_.sid].bbox[0], s_.bbox[1] - b_by_[s_.sid].bbox[1]) for s_ in after
                                     if s_.sid in b_by_ and s_.bbox[:2] != b_by_[s_.sid].bbox[:2] and "hud" not in s_.tags)
                walked_back_ = isinstance(sig_, tuple) and len(sig_) == 2 and sig_[0] == "moved" and this_vs_ and \
                    this_vs_ == frozenset((-u[0], -u[1]) for u in sig_[1])
                if sig_ is not None and not walked_back_:
                    self.undo_votes[key].add(sig_)
                # (an arrow that walks the body back over its own step is the
                # opposite arrow, not an undo: the votes must name something
                # other than the exact reverse of this key's own translate)
                own_ = {e[1] for e in self.fixed.moving(key).values() if e[0] == "translate"}
                own_ |= {self.key_selected[key]} if key in self.key_selected else set()
                inverse_only = bool(own_) and all(isinstance(v_, tuple) and v_[0] == "moved" and all((-u[0], -u[1]) in own_ for u in v_[1]) for v_ in self.undo_votes[key])
                if _ENV.get("UNDO_DEBUG"):
                    print(f"   undo vote {key}: votes {self.undo_votes[key]} own {own_} inverse_only {inverse_only}")
                if len(self.undo_votes[key]) >= 2 and not inverse_only:
                    self.undo_keys.add(key)
            self._frames.append(frame)
            self._frames = self._frames[-4:]
        elif frame is not None:
            self._frames.append(frame)
            self._frames = self._frames[-4:]
        self._last_key = key
        if frame is not None and frame_after is not None and not isinstance(key, tuple):
            keep_ = np.ones(frame_after.shape, dtype=bool)
            for s_ in after + before:
                if "hud" in s_.tags or "bar" in s_.tags or s_.sid in self.counters or s_.sid == self.clock:
                    for (r_, c_) in s_.cells:
                        if 0 <= r_ < keep_.shape[0] and 0 <= c_ < keep_.shape[1]:
                            keep_[r_, c_] = False
            if not (frame_after[keep_] == frame[keep_]).all():
                self._last_effective = key      # the last press that changed the world (the HUD is not the world)
                self._before_effective = frame
                # THE CHANGE'S SIGNATURE, by what moved and how (not by where):
                # an undo restores after changes of different kinds (a left,
                # a right); an inverse key restores only after its inverse,
                # wherever that was pressed
                b_s = {s_.sid: s_ for s_ in before}
                mv_ = frozenset(_eff(b_s[s_.sid], s_)[1] for s_ in after if s_.sid in b_s and "hud" not in s_.tags
                                and "terrain" not in s_.tags and kind(s_) not in self.flat
                                and (not self.active or s_.sid in self.active)
                                and _eff(b_s[s_.sid], s_)[0] == "translate")
                if not mv_:
                    # (whatever moved names the change, the body or another
                    # thing: a step is a step, its taking-back an inverse)
                    mv_ = frozenset(_eff(b_s[s_.sid], s_)[1] for s_ in after if s_.sid in b_s and "hud" not in s_.tags
                                    and "terrain" not in s_.tags and _eff(b_s[s_.sid], s_)[0] == "translate")
                if mv_:
                    self._last_sig = ("moved", mv_)
                elif not self.active:
                    diff_ = np.argwhere((frame_after != frame) & keep_)
                    self._last_sig = hash((tuple(map(tuple, diff_.tolist())), tuple(int(frame[r, c]) for r, c in diff_.tolist())))
                else:
                    self._last_sig = None       # (the body did not move: no change to be taken back that a vote can name)
        if key in self.undo_keys:
            self.fixed.presses[key] += 1
            return
        b_by0 = {s.sid: s for s in before}
        a_by0 = {s.sid: s for s in after}
        # A DEATH: the clock ran out and refilled, and the controlled thing
        # did not move as the key moves it (it was put back): a life spent.
        # Nothing about this press is a law of the key
        cb_ = self.clock_thing(before); ca_ = self.clock_thing(after)
        self._clock_pair = (cb_, ca_)
        # (a clock spent to nothing may not be read at all the press before)
        val_b = self.clock_value(cb_) if cb_ is not None else (0 if ca_ is not None and self.clock_full and self.clock_value(ca_) >= 0.9 * self.clock_full else None)
        if val_b is not None and ca_ is not None and self.active \
                and val_b <= self.clock_rate and self.clock_value(ca_) > val_b:
            law = self.fixed.moving(key) if not isinstance(key, tuple) else {}
            jumped = False
            for sid in self.active:
                if sid in b_by0 and sid in a_by0:
                    e = effect_of(b_by0[sid], a_by0[sid])
                    if e[0] == "translate" and law.get(sid) != e:
                        jumped = True
            if jumped:
                self.deaths += 1
                self.spawn = {sid: a_by0[sid].bbox for sid in self.active if sid in a_by0}
                self.clock_full = max(self.clock_full, self.clock_value(ca_))
                self.presses_total += 1
                self._frame = frame
                return
        # (the selection the press was made under: the host wearing the
        # selection mark — one host, or the one of the selection colour)
        sel_b = None
        mb_ = selected_hosts(before) if self.select_color is not None else {}     # (a mark is a selection once the keys have shown it: a ring's centre is no mark)
        if mb_:
            if self.select_color is not None and any(c_ == self.select_color for c_ in mb_.values()):
                sel_b = next(h_ for h_, c_ in mb_.items() if c_ == self.select_color)
            elif len(mb_) == 1:
                sel_b = next(iter(mb_))
        if sel_b is None and (self.selected_colour is not None or self.select_kinds):
            sel_b = self.selected_now(before)       # (selection shown by the thing's colour, or by which thing answers)
        live_bars = set(selected_hosts(before)) | set(selected_hosts(after))
        for x in before + after:
            src = getattr(x, "source", None)
            if src is not None:
                live_bars.add(src[1])           # a mirror's axis
        # TEXTURE: lone cells of a colour strewn over the frame (the dots on
        # a wall) are a drawing, not things with laws
        cnt1_ = Counter(s_.body for s_ in before if s_.size == 1 and s_.sid < 1000)
        self.texture = {c_ for c_, n_ in cnt1_.items() if n_ >= 20}
        skip_ = {s_.sid for s_ in before + after if s_.size == 1 and s_.body in self.texture}
        if getattr(self, "_scroll_now", False) and self.active:
            # (when the screen moved, only the walker's motion is read: what
            # else "moved" is a twin taken for a thing that left the screen)
            skip_ |= {s_.sid for s_ in before + after if s_.sid not in self.active and (s_.sid // 1000 if s_.sid >= 1000 else s_.sid) not in self.active}
        # (a thing bound to the body — carried — moves by the body's law,
        # not by one of its own: not read under the keys)
        if getattr(self, "carry", None) and self.bound:
            carried_ = {o_ for (m_, o_) in self.bound}
            skip_ |= carried_ | {s_.sid for s_ in before + after if s_.sid >= 1000 and s_.sid // 1000 in carried_}
        if getattr(self, "_put_back_now", False) and self.active and _ENV.get("PUT_BACK_SKIP"):
            # (WITHDRAWN TO A SWITCH: m0r0 L1 41 -> 44 and L2 lost with it — the
            # put-back press's reading of the keys is part of what that level
            # lives by, in a way not yet understood.  PUT_BACK_SKIP=1)
            # (the controlled things were set back to the start: no law of the key)
            skip_ |= {m_ for m_ in self.active} | {s_.sid for s_ in before + after if s_.sid >= 1000 and s_.sid // 1000 in self.active}
        if isinstance(key, tuple) and self.active and self.slide_votes:
            # (a click after which the body slid along the slide's axis: the
            # slide's doing, not the click's law on the body)
            d_ = self.slide_votes.most_common(1)[0][0]
            b_s = {s_.sid: s_ for s_ in before}
            for s_ in after:
                if s_.sid in self.active and s_.sid in b_s:
                    e_ = _eff(b_s[s_.sid], s_)
                    if e_[0] == "translate" and ((d_[0] == 0) == (e_[1][0] == 0)) and ((d_[1] == 0) == (e_[1][1] == 0)):
                        skip_ |= {m_ for m_ in self.active} | {m_ * 1000 + c_ for m_ in self.active for c_ in range(16)}
        # A THING'S MOTION IS ITS BODY'S: members of one body (things that
        # always move together — a body and its dark front row) whose own
        # vectors disagree (the front row swung to another side: a turn)
        # moved by what the body as a whole moved, when the body kept its
        # size (wa30: a blocked press is a turn; the sprite steps by four)
        ovr_ = {}
        try:
            b_by0 = {s_.sid: s_ for s_ in before}
            a_by0 = {s_.sid: s_ for s_ in after}
            for grp in self.bodies():
                mem = [sid for sid in grp if sid < 1000 and sid in b_by0 and sid in a_by0]
                if len(mem) < 2:
                    continue
                vs_ = {}
                for sid in mem:
                    e_ = _eff(b_by0[sid], a_by0[sid])
                    vs_[sid] = e_[1] if e_[0] == "translate" else None
                if len(set(vs_.values())) < 2:
                    continue
                cb = set(); ca = set()
                for sid in mem:
                    cb |= set(b_by0[sid].cells); ca |= set(a_by0[sid].cells)
                bb = _bbox_cells(cb); ba = _bbox_cells(ca)
                if (bb[2] - bb[0], bb[3] - bb[1]) == (ba[2] - ba[0], ba[3] - ba[1]) and len(cb) == len(ca):
                    v_body = (ba[0] - bb[0], ba[1] - bb[1])
                    for sid in mem:
                        ovr_[sid] = ("translate", v_body) if v_body != (0, 0) else ("none", (0, 0))
        except Exception:
            ovr_ = {}
        # THE KEY'S OWN STEP IS THE PATH BEFORE THE FALL: under a known
        # gravity, what the press moved the body by is the path's steps up
        # to the fall, not the net of the press (a hop off a ledge is the
        # hop; the fall is gravity's — tw05, the jump read as "down 28")
        try:
            own_ = getattr(self, "path_own", {}).get(key) if getattr(self, "gravity", None) is not None else None
            if own_ is not None and self.active:
                a_by0 = {s_.sid: s_ for s_ in after}; b_by0 = {s_.sid: s_ for s_ in before}
                for sid in self.active:
                    if sid in a_by0 and sid in b_by0 and sid not in ovr_ and len(a_by0[sid].cells) == len(b_by0[sid].cells):
                        e0_ = _eff(b_by0[sid], a_by0[sid])
                        if e0_[0] == "translate" and tuple(e0_[1]) != tuple(own_):
                            ovr_[sid] = ("translate", tuple(own_)) if tuple(own_) != (0, 0) else ("none", (0, 0))
        except Exception:
            pass
        try:
            # (an arc, once seen, is the key's way with the thing: the flag
            # stays — a press against a wall shows no path and unsays nothing)
            if getattr(self, "path_arc", {}).get(key, False):
                for sid in (self.active or ()):
                    self.fixed.arcs.add((key, sid))
        except Exception:
            pass
        # (a refusal with something in the way is the way's, not the law's:
        # the cells a thing's recorded step would take it to, read in the
        # frame before the press — anything there but the background)
        blocked_ = set()
        try:
            if frame is not None and not isinstance(key, tuple):
                fr0_ = np.asarray(frame)
                bg0_ = int(np.bincount(fr0_.ravel()).argmax())
                H0_, W0_ = fr0_.shape
                for s0_ in before:
                    e0_ = self.fixed.eff.get(key, {}).get(s0_.sid)
                    if e0_ is None or e0_[0] != "translate" or e0_[1] == (0, 0):
                        continue
                    own0_ = set(s0_.cells)
                    ahead0_ = {(r + e0_[1][0], c + e0_[1][1]) for (r, c) in own0_} - own0_
                    if any(not (0 <= r < H0_ and 0 <= c < W0_) or int(fr0_[r, c]) != bg0_ for (r, c) in ahead0_):
                        blocked_.add(s0_.sid)
        except Exception:
            blocked_ = set()
        self.fixed.observe(key, before, after, sel=self.sel_ctx(sel_b), live_bars=live_bars, skip=skip_, override=ovr_, blocked=frozenset(blocked_))
        self.thing_laws.observe(after)
        self.read_materials(after)
        self._frame = frame
        for s_ in before + after:
            self.kind_seen[s_.sid] = kind(s_)
            self.kinds_worn[s_.sid].add(kind(s_))     # a thing's colour may be a state (selected = black)
        b_by = {s.sid: s for s in before}
        a_by = {s.sid: s for s in after}
        marks_b = selected_hosts(before)
        marks_a = selected_hosts(after)
        moved = {}
        for s in after:
            p = b_by.get(s.sid)
            if p is None or "hud" in s.tags or ("bar" in s.tags and "derived" not in s.tags and s.sid not in marks_b and s.sid not in marks_a
                                                and not any(getattr(x, "source", (None, None))[1] == s.sid for x in after)):
                continue                # (a bar that is selected, or a mirror's axis, moves like anything)
            if s.sid in skip_:
                continue
            e = _eff(p, s)
            if e[0] == "translate":
                moved[s.sid] = e[1]
        for sid_, e_ in ovr_.items():
            if e_[0] == "translate":
                moved[sid_] = e_[1]
            else:
                moved.pop(sid_, None)
        # A TURN OF THE ACTIVE BODY: a key under which a member of the body
        # the keys move turned is a turning key (for whichever body is
        # selected)
        if not isinstance(key, tuple) and self.active:
            turns = Counter()
            for s in after:
                p = b_by.get(s.sid)
                if p is not None and s.sid in self.active:
                    e = _eff(p, s)
                    if e[0] == "rotate":
                        turns[e[1]] += 1
            if turns:
                self.key_rotate[key] = turns.most_common(1)[0][0]
        # A COUNTER: a thing that changes the same way under every key, at
        # (nearly) every press — a step bar, a life strip, a clock — is
        # the game's sense of time, not an effect of any key.  It leaves
        # the key records and is remembered with its rate
        self.presses_total += 1
        changed_now = set()
        # (the thing under a click that changes is a button's feedback, not
        # the clock: it does not count toward being a counter)
        pressed_sids = set()
        if click is not None:
            under_ = [s for s in before if s.bbox[0] <= click[0] <= s.bbox[2] and s.bbox[1] <= click[1] <= s.bbox[3]
                      and "terrain" not in s.tags and "hud" not in s.tags]
            for u_ in under_:
                pressed_sids.add(u_.sid); pressed_sids.add(u_.sid // 1000 if u_.sid >= 1000 else u_.sid)
            small_ = [u_ for u_ in under_ if u_.sid < 1000]
            if small_:
                u0_ = min(small_, key=lambda u_: u_.size)
                self.click_under[key] = kind(u0_)
                self.click_under_form[key] = (u0_.bbox[2] - u0_.bbox[0] + 1, u0_.bbox[3] - u0_.bbox[1] + 1, u0_.size)
        for s in after:
            p = b_by.get(s.sid)
            if p is None or "terrain" in s.tags:
                continue
            if dict(p.cells) != dict(s.cells) and s.sid not in moved:
                changed_now.add(s.sid)          # changed in place (a move is not a change)
                if s.sid not in pressed_sids:
                    self.change_presses[s.sid] += 1
                    self.change_keys[s.sid].add(key)
        for sid in list(changed_now):
            if len(self.change_keys[sid]) >= 3 and self.change_presses[sid] >= 0.5 * self.presses_total and self.presses_total >= 6:
                if sid not in self.counters:
                    p = b_by[sid]; s_ = a_by[sid]
                    self.counters[sid] = (kind(p), s_.size - p.size)     # cells gained per press (negative: spent)
                for k_ in list(self.fixed.eff):
                    self.fixed.eff[k_].pop(sid, None)
        moved = {sid: v for sid, v in moved.items() if sid not in self.counters}
        # A CLOCKWORK: a thing that has moved under different keys and always
        # by the one vector moves on its own (a wall that rises every other
        # press, whatever was pressed): its motion is no key's law, and it
        # is not one of the bodies the keys move
        for sid, v in list(moved.items()):
            self.move_keys[(kind(b_by[sid]), v)].add(key) if not isinstance(key, tuple) else None
        for sid, v in list(moved.items()):
            kd = kind(b_by[sid])
            vecs = {v_ for (k_, v_) in self.move_keys if k_ == kd}
            if len(vecs) == 1 and len(self.move_keys[(kd, v)] - self.undo_keys) >= 2 and v != (0, 0):
                self.selfmovers.add(kd)
        # A JOURNEY: a thing that moves press after press by vectors of its
        # own — not with the body this press, not against the body (pushed)
        # — travels on its own, choosing its way: an AGENT of its kind (the
        # helper that carries crates to the pen; a foe).  Its motion is no
        # key's law, and what it holds is in its hands, not ours.
        try:
            act_ = set(self.active or ())
            body_vs = {v for sid, v in moved.items() if sid in act_ or (sid >= 1000 and sid // 1000 in act_)}
            body_boxes = [b_by[sid].bbox for sid in act_ if sid in b_by]
            bound_ = {o_ for (_, o_) in getattr(self, "bound", ())}
            bodies_ = [g for g in self.bodies() if len(g) >= 2]
            # (a thing that went as the KEY sends things — by the key's law
            # for anything — answered the key: a second block the keys move,
            # held back where the first was not, is no agent)
            key_vs = {tuple(e_[1]) for s_, e_ in self.fixed.eff.get(key, {}).items() if e_[0] == "translate" and s_ in self.ever_active} if not isinstance(key, tuple) else set()
            for sid, v in moved.items():
                if sid >= 1000 or sid in act_ or sid in bound_ or v == (0, 0) or sid not in b_by:
                    continue
                if v in body_vs or (tuple(v) in key_vs and _claims.on("key_sent_is_no_agent")) or sid in self.ever_active:
                    continue                  # (it went as the body went, or as the key sends things: not its own way)
                # (WHAT THE PRESS ITSELF SENT IS NOT ON ITS OWN WAY: a thing whose
                # law under THIS key — a click's as much as a key's — is the very
                # motion seen went as it was sent.  vc33 L3: the columns' markers,
                # raised by one foot's click and lowered by another's, were read
                # as travellers and their laws thrown away after three presses)
                e_k_ = self.fixed.eff.get(key, {}).get(sid)
                if e_k_ is not None and e_k_[0] == "translate" and tuple(e_k_[1]) == tuple(v) and not _ENV.get("AGENT_IGNORE_LAW"):
                    continue
                if any(sid in grp for grp in bodies_):
                    continue                  # (a member of a body the keys move)
                bb = b_by[sid].bbox
                if any(max(bb[0] - x[2], x[0] - bb[2], bb[1] - x[3], x[1] - bb[3]) <= 1 for x in body_boxes):
                    continue                  # (against the body: a push, not a journey)
                # (a thing that went as a neighbour went was taken along by
                # it — the crate in the helper's hands: only a thing that
                # went ALONE is on its own way)
                with_ = [o_ for o_, v_ in moved.items() if o_ != sid and o_ < 1000 and v_ == v and o_ in b_by
                         and max(bb[0] - b_by[o_].bbox[2], b_by[o_].bbox[0] - bb[2], bb[1] - b_by[o_].bbox[3], b_by[o_].bbox[1] - bb[3]) <= 1]
                if with_:
                    continue
                self.own_ways[kind(b_by[sid])][sid].append(tuple(v))
            # WE ARE IN ITS WAY: an agent that stood still this press while
            # our body stands where its last own step would have taken it
            # is stalled by us (Tushar: "we blocked it") — the continuity
            # monitor's reading, acted on by stepping aside
            for a_, va_ in list(self.last_way.items()):
                if a_ in moved or a_ not in a_by or a_ not in b_by or kind(a_by[a_]) not in self.agents:
                    continue
                ab = a_by[a_].bbox
                ahead_ = (ab[0] + va_[0], ab[1] + va_[1], ab[2] + va_[0], ab[3] + va_[1])
                ours_ = [b_by[m_].bbox for m_ in act_ if m_ in b_by]
                if any(not (x[0] > ahead_[2] or ahead_[0] > x[2] or x[1] > ahead_[3] or ahead_[1] > x[3]) for x in ours_):
                    self.stalled[a_] = self.stalled.get(a_, 0) + 1
                    if _ENV.get("MOVE_DEBUG") or getattr(self, "verbose", False):
                        print(f"   stalled: the agent S{a_} (kind {kind(a_by[a_])}) stands still with our body in its way ({self.stalled[a_]}x): we are blocking it")
                else:
                    self.stalled.pop(a_, None)
            for sid, v in moved.items():
                if sid in b_by and kind(b_by[sid]) in self.agents and sid not in act_:
                    self.last_way[sid] = tuple(v); self.stalled.pop(sid, None)
            for kd, by_sid in self.own_ways.items():
                for sid_, vs in by_sid.items():
                    if len(vs) >= 3 and len(set(vs)) >= 2 and kd not in self.agents and not _ENV.get("NO_AGENTS"):   # (NO_AGENTS: the ablation switch)
                        self.agents.add(kd)
                        if _ENV.get("MOVE_DEBUG") or getattr(self, "verbose", False):
                            print(f"   agent: things of kind {kd} travel on their own (S{sid_} went {vs[-4:]}): their motion is no key's law")
            if self.agents:
                # (the parallel record: fed every press, agents' own ticks as inputs)
                try:
                    self.agent_record.observe(self.agents, before, after, moved, kind, our_sids=act_)
                except Exception as ex_:
                    if _ENV.get("AGENT_DEBUG"):
                        import traceback; traceback.print_exc()
                self.fixed.forget({sid_ for sid_ in self.kind_seen if self.kind_seen.get(sid_) in self.agents and sid_ not in act_})
                # (what went with an agent — against it, by its vector — is in
                # its hands: that motion is the agent's, not the key's law)
                ag_now = {sid: v for sid, v in moved.items() if sid in b_by and kind(b_by[sid]) in self.agents and sid not in act_}
                taken_ = {}
                for sid, v in moved.items():
                    if sid in ag_now or sid in act_ or sid not in b_by:
                        continue
                    bb = b_by[sid].bbox
                    for a_, va_ in ag_now.items():
                        ab = b_by[a_].bbox
                        if va_ == v and max(bb[0] - ab[2], ab[0] - bb[2], bb[1] - ab[3], ab[1] - bb[3]) <= 1:
                            taken_[sid] = a_; break
                self.fixed.forget(set(taken_))
                # WHERE THE HELPER PUTS THEM: a thing that went with an agent
                # and now stays while the agent goes on without it came to
                # REST — the place things of its kind are brought to (the
                # pen): a law of the kind, read off the helper's work
                for sid, a_ in taken_.items():
                    if sid in a_by:
                        self.in_hands[sid] = (a_, a_by[sid].bbox)
                for sid, (a_, bb_) in list(self.in_hands.items()):
                    if sid in taken_ or sid not in a_by:
                        if sid not in a_by:
                            self.in_hands.pop(sid, None)
                        continue
                    if sid in moved:
                        continue                  # (still going; with us, maybe)
                    ab_ = a_by[a_].bbox if a_ in a_by else None
                    bb = a_by[sid].bbox
                    apart_ = ab_ is None or max(bb[0] - ab_[2], ab_[0] - bb[2], bb[1] - ab_[3], ab_[1] - bb[3]) > 1
                    if apart_ and bb == bb_:
                        kd_ = kind(a_by[sid])
                        if bb not in self.rests[kd_]:
                            self.rests[kd_].append(bb)
                            if _ENV.get("MOVE_DEBUG") or getattr(self, "verbose", False):
                                print(f"   rest: a thing of kind {kd_} the agent carried was left at {bb}: where things of the kind are brought")
                        self.in_hands.pop(sid, None)
                moved = {sid: v for sid, v in moved.items() if (kind(b_by[sid]) not in self.agents or sid in act_) and sid not in taken_}
        except Exception as ex_:
            if _ENV.get("MOVE_DEBUG"):
                print(f"   own ways: {ex_!r}")
        if _ENV.get("MOVE_DEBUG"):
            print(f"   moved under {key}: {[(sid, kind(b_by[sid]), b_by[sid].size, b_by[sid].bbox, v) for sid, v in moved.items()]} selfmovers {self.selfmovers}")
        if self.selfmovers:
            self.fixed.forget({sid_ for sid_ in self.kind_seen if self.kind_seen.get(sid_) in self.selfmovers})
            moved = {sid: v for sid, v in moved.items() if kind(b_by[sid]) not in self.selfmovers}
        # A PORTAL: the key moved a controlled thing, but not by its law —
        # from THAT place the press lands it elsewhere (a teleporter, a
        # wrap at the edge).  Remembered by place and key; the plan uses it
        if not isinstance(key, tuple) and self.active:
            law_ = self.fixed.moving(key)
            for sid, v in moved.items():
                if sid not in self.active or sid not in b_by or sid not in a_by:
                    continue
                lv_ = law_.get(sid)
                # A SLIDE: the thing went where the key sends it and ON along
                # one axis until something stopped it — after every move it
                # slides that way (a gravity; up is a direction like any).
                # One sight is the theory; a move after which it could have
                # slid and did not is a sight against it
                if lv_ is not None and lv_[0] == "translate" and lv_[1] != v and frame_after is not None:
                    ex_ = (v[0] - lv_[1][0], v[1] - lv_[1][1])
                    if (ex_[0] == 0) != (ex_[1] == 0):
                        d_ = ((ex_[0] > 0) - (ex_[0] < 0), (ex_[1] > 0) - (ex_[1] < 0))
                        body_ = set()
                        for m_ in self.active:
                            if m_ in a_by:
                                body_ |= set(a_by[m_].cells)
                        beyond_ = {(r + d_[0], c + d_[1]) for (r, c) in body_} - body_
                        H_, W_ = frame_after.shape
                        stopped_ = any(not (0 <= r < H_ and 0 <= c < W_) or int(frame_after[r, c]) not in self.ground_ok for (r, c) in beyond_)
                        # ...and it slid THROUGH ground: every cell between where
                        # the key put it and where it came to rest wore a ground
                        # colour (a jump across a wall is a portal, not a slide)
                        body_b0 = set()
                        for m_ in self.active:
                            if m_ in b_by:
                                body_b0 |= set(b_by[m_].cells)
                        landed_ = _shift(body_b0, lv_[1])
                        crossed_ = _sweep(landed_, ex_) if frame is not None else set()
                        through_ground = bool(self.ground_ok) and all(0 <= r < H_ and 0 <= c < W_ and int(frame[r, c]) in self.ground_ok for (r, c) in crossed_)
                        if stopped_ and through_ground and sid == min(m_ for m_ in self.active if m_ in b_by and m_ in moved):
                            old_d = self.slide()
                            if old_d is not None and old_d != d_:
                                # THE BELIEF RESET: the body slid the OTHER way — one
                                # such sight overturns the law, whatever its votes
                                # (Tushar: "see if that information needs to reset
                                # the entire belief"); what the body touched on the
                                # press before is remembered as the likely cause
                                self.slide_flips.append((old_d, d_, sorted(self._touching_kinds(before))))
                                self.slide_votes = Counter(); self.slide_against = Counter()
                                self.stops = set()
                                self.slide_recent = []
                            self.slide_votes[d_] += 1
                            self.slide_recent = (self.slide_recent + [("for", d_)])[-4:]
                            if _ENV.get("NOTE_DEBUG"):
                                print(f"   slide? S{sid} moved {v} under {key} whose law is {lv_[1]}: on by {ex_} until stopped — slides {d_} (votes {dict(self.slide_votes)})")
                            continue
                if lv_ is not None and lv_[0] == "translate" and lv_[1] == v and self.slide_votes and frame_after is not None \
                        and sid == min(m_ for m_ in self.active if m_ in b_by and m_ in moved):
                    # AGAINST A SLIDE: it moved by the key's law alone though the
                    # ground lay open beyond it in the slide's direction
                    body_ = set()
                    for m_ in self.active:
                        if m_ in a_by:
                            body_ |= set(a_by[m_].cells)
                    H_, W_ = frame_after.shape
                    for d_ in list(self.slide_votes):
                        beyond_ = {(r + d_[0], c + d_[1]) for (r, c) in body_} - body_
                        if beyond_ and all(0 <= r < H_ and 0 <= c < W_ and int(frame_after[r, c]) in self.ground_ok for (r, c) in beyond_):
                            self.slide_against[d_] += 1
                            self.slide_recent = (self.slide_recent + [("against", d_)])[-4:]
                            if _ENV.get("NOTE_DEBUG"):
                                print(f"   slide against {d_}: S{sid} moved by the law alone with ground beyond (against {dict(self.slide_against)})")
                if lv_ is not None and lv_[0] == "translate" and lv_[1] != v and abs(v[0]) + abs(v[1]) > 2 * (abs(lv_[1][0]) + abs(lv_[1][1])):
                    p = b_by[sid]
                    if _ENV.get("NOTE_DEBUG"):
                        cb_, ca_ = getattr(self, "_clock_pair", (None, None))
                        print(f"   portal? S{sid} {p.bbox[:2]} -> {a_by[sid].bbox[:2]} by {key}; clock {self.clock} colour {self.clock_colour} rate {self.clock_rate} before {(cb_.sid, self.clock_value(cb_)) if cb_ else None} after {(ca_.sid, self.clock_value(ca_)) if ca_ else None} active {self.active}")
                    self.portals[(kind(p), p.bbox[0], p.bbox[1], key)] = (a_by[sid].bbox[0], a_by[sid].bbox[1])
        # A PAINT: a key (not a click) under which a thing took new colours
        # on the same cells while nothing moved — the key paints it; which
        # cells and what colour is remembered with the press (the where may
        # depend on where the controlled thing stood: the protocol asks)
        if not isinstance(key, tuple) and not moved:
            for sid in changed_now:
                p = b_by[sid]; s_ = a_by[sid]
                if sid >= 1000 or sid in self.counters or sid == self.clock or "hud" in s_.tags or set(p.cells) != set(s_.cells):
                    continue                # (the clock recolours itself at every press: not a paint)
                diff = {q: s_.cells[q] for q in s_.cells if p.cells[q] != s_.cells[q]}
                cols = Counter(diff.values())
                if diff and len(cols) == 1:
                    self.paints[key].append((sid, frozenset(diff), next(iter(cols))))
        # A TRIGGER: the thing the keys move ARRIVED on another thing this
        # press (it overlaps it now and did not before), and in the same
        # press a third thing changed without moving (a display turned, a
        # door opened): the arrival is what changed it.  Recorded by the
        # kind of the thing arrived on; the changed thing is a DISPLAY
        mover_sids = set(moved) | set(self.active or ())
        # (what moves with the mover is the mover: its pieces are not
        # things it arrives on)
        bodies_now = self.bodies() if moved else []
        for b in bodies_now:
            if b & mover_sids:
                mover_sids |= set(b)
        arrived = []
        for msid in mover_sids:
            m_after = a_by.get(msid); m_before = b_by.get(msid)
            if m_after is None or m_before is None or msid in self.counters:
                continue
            for t in before:
                if t.sid == msid or t.sid in mover_sids or "terrain" in t.tags or "hud" in t.tags or t.sid in self.counters:
                    continue
                if kind(t) in {kind(b_by[m]) for m in mover_sids if m in b_by}:
                    continue            # a thing of the mover's own colour is not a tile it stepped on
                now_over = bool(set(m_after.cells) & set(t.cells))
                was_over = bool(set(m_before.cells) & set(t.cells))
                if now_over and not was_over:
                    arrived.append(t)
        if arrived:
            for sid in changed_now:
                if sid in mover_sids or sid in self.counters or sid in moved:
                    continue
                if sid in self.ever_active or sid in (self.active or ()):
                    continue            # (a body the keys move is no panel: its turn in place is its own)
                d = a_by[sid]
                if "hud" in d.tags and d.size < 2:
                    continue
                for t in arrived:
                    self.triggers[kind(t)].append((sid, key))
                    self.displays[sid] += 1
        # THE CLOCK: the counter every press spends is the time left; a
        # thing whose arrival refilled it is a REFILL, used up by the visit
        # (its value is the count of the colour it spends — a bar that turns
        # colour from one end — or its size when it shrinks)
        # (what it spends is what most presses spend: a refill or a reset
        # is one press, and not the rule)
        for sid in self.counters:
            if sid not in a_by or sid not in b_by:
                continue
            cb = Counter(b_by[sid].cells.values()); ca = Counter(a_by[sid].cells.values())
            dec = {c: cb[c] - ca.get(c, 0) for c in cb if cb[c] > ca.get(c, 0)}
            if dec:
                col = max(dec, key=dec.get)
                self.clock_votes[(sid, col, dec[col])] += 1
                if sid == self.clock and col == self.clock_colour:
                    # (a press may spend more than the rate — a refused click's
                    # penalty; the largest spend seen bounds what one press can take)
                    if dec[col] > getattr(self, "clock_max_drop", 0):
                        self.clock_max_drop = dec[col]; self.clock_max_drop_n = 1
                    elif dec[col] == getattr(self, "clock_max_drop", 0):
                        self.clock_max_drop_n = getattr(self, "clock_max_drop_n", 0) + 1
        if self.clock_votes:
            (csid, col, rate), n_ = self.clock_votes.most_common(1)[0]
            need_ = 1 if getattr(self, "clock_rate_carried", False) else 2       # (a carried rate is a prior: the first press here says the level's)
            if n_ >= need_ and (self.clock is None or self.clock == csid or self.clock not in a_by):
                self.clock_rate_carried = False
                self.clock, self.clock_colour, self.clock_rate = csid, col, rate
        cb_, ca_ = getattr(self, "_clock_pair", (None, None))
        if self.clock in a_by and (cb_ is None or ca_ is None):
            cb_ = b_by.get(self.clock); ca_ = a_by[self.clock]
        if _ENV.get("CLOCK_DEBUG"):
            print(f"   [clock] key {key} clock {self.clock} rate {self.clock_rate} colour {self.clock_colour} pair {(cb_.sid if cb_ is not None else None, ca_.sid if ca_ is not None else None)} values {(self.clock_value(cb_) if cb_ is not None else None, self.clock_value(ca_) if ca_ is not None else None)} arrived {[(t.sid, kind(t)) for t in arrived]} votes {dict(self.clock_votes)}")
        if cb_ is not None and ca_ is not None:
            self.clock_box = ca_.bbox
            self.clock_full = max(self.clock_full, self.clock_value(ca_), self.clock_value(cb_))
            # (the floor under the arrival is not the refill — when something
            # else was arrived on too; a floor colour that alone was stepped
            # onto as the clock jumped is what the jump is attributed to)
            def floor_(t):
                return "terrain" in t.tags or kind(t) in self.ground_ok or kind(t) in self.flat
            credited = [t for t in arrived if not floor_(t)] or [t for t in arrived if "terrain" not in t.tags]
            if self.clock_value(ca_) > self.clock_value(cb_) and arrived:
                for t in credited:
                    self.refills[kind(t)] += 1; self.refills_here[kind(t)] += 1
            # A DRAIN: an arrival on which the clock lost more than a press
            # spends is the refill's twin — the thing spent it (twist suite
            # tw02: the pad that refilled for two levels drains on the third;
            # the body died one step later on empty floor, and the death
            # ledger, reading only the last step, blamed nothing)
            elif arrived and self.clock_rate and self.clock_value(cb_) - self.clock_value(ca_) > 2 * self.clock_rate:
                for t in credited:
                    self.drains[kind(t)] += 1
        # WHERE A THING IS decides what a button does to it (a track: every
        # press moves every piece one slot on; a corner piece turns).  The
        # effect is recorded by SLOT (kind, top-left), and a key that moved
        # several things by different vectors in one press is POSITIONAL:
        # planned by slot, not by identity
        for sid, v in moved.items():
            p = b_by[sid]
            if "hud" in p.tags or "terrain" in p.tags:
                continue
            slot = (p.bbox[0], p.bbox[1])       # the slot, whatever sits on it
            self.pos_kinds[key].add(kind(p))
            old_v = self.pos_eff[key].get(slot)
            if old_v is not None and old_v != v:
                self.pos_bad[key].add(slot)
            self.pos_eff[key][slot] = v
        if len(moved) >= 3 and len(set(moved.values())) >= 2:
            # a track is worked by a BUTTON: the thing under the click stays
            # put; a click on a thing that then moves is a pull, not a track
            under_click = None
            if click is not None:
                under = [s for s in before if s.bbox[0] <= click[0] <= s.bbox[2] and s.bbox[1] <= click[1] <= s.bbox[3]
                         and "terrain" not in s.tags and "hud" not in s.tags]
                under_click = min(under, key=lambda s: s.size) if under else None
            # ...and the pieces take each other's places (a ring turning):
            # things that fly to fresh ground are pulled, not tracked
            origins = {(b_by[sid_].bbox[0], b_by[sid_].bbox[1]) for sid_ in moved}
            landed = sum(1 for sid_ in moved if (a_by[sid_].bbox[0], a_by[sid_].bbox[1]) in origins)
            cyclic = landed >= 0.6 * len(moved)
            if cyclic and (click is None or (under_click is not None and under_click.sid not in moved)):
                self.pos_evidence[key].append({sid_: (kind(b_by[sid_]), v_) for sid_, v_ in moved.items()})
        # (a track is read from presses whose movers are the key's own: a
        # clockwork found later takes its pieces out of the evidence)
        for k_, evs in self.pos_evidence.items():
            ok_ = False
            for ev in evs:
                ev_ = {sid_: v_ for sid_, (kd_, v_) in ev.items() if kd_ not in self.selfmovers}
                if len(ev_) >= 3 and len(set(ev_.values())) >= 2:
                    ok_ = True; break
            if ok_:
                self.positional.add(k_)
            else:
                self.positional.discard(k_)
            # (a click on empty ground that moves things is a pull toward
            # the click — the relative law below — not a track)
        # THE MARK THAT CHANGES HANDS is the selection mark (two blocks
        # may each wear a centre dot; only one dot's host ever changes)
        # (once the selection's colour is known, only a mark of that colour
        # changing hands is a hand-over: a dot's ring covered by a passing
        # arm is not a selection moving)
        swapped = [(h, col) for h, col in marks_a.items() if marks_b.get(h) != col and col in marks_b.values()
                   and (self.select_color is None or col == self.select_color)]
        for h, col in swapped:
            if click is None and not isinstance(key, tuple):
                src = next((hb for hb, cb in marks_b.items() if cb == col), None)
                self.cycle_keys.setdefault(key, []).append((src, h))
        # (a mark is a SELECTION mark only once it has been seen to change
        # hands — by a click or a key: a body's own outline, its dark front
        # row, moves with it every press and never changes hands; a livery)
        if swapped:
            self.mark_swapped = True
        if click is None:
            # SELECTION: which of the moved things wore the mark?  The mark
            # colour worn by what the keys move IS the selection colour
            # (when two marks swap, the keys tell them apart)
            for sid, v in moved.items():
                if sid in marks_b:
                    self.mark_votes[marks_b[sid]] += 1
            if self.mark_votes and (getattr(self, "mark_swapped", False) or not _claims.on("mark_must_change_hands")):
                self.select_color = self.mark_votes.most_common(1)[0][0]
            elif swapped and self.select_color is None:
                self.select_color = swapped[0][1]
            for sid, v in moved.items():
                if sid in marks_b and (self.select_color is None or marks_b[sid] == self.select_color):
                    self.sel_moves[key].append((sid, v))
            hosts = {h for h, _ in self.sel_moves[key]}
            vs = Counter(v for _, v in self.sel_moves[key])
            if len(hosts) >= 2 and len(vs) == 1:
                self.key_selected[key] = next(iter(vs))
            elif len(hosts) == 1 and len(vs) == 1 and marks_b and (self.select_kinds or self.cycle_keys or swapped):
                # one host so far but the mark is known to change hands (a
                # click drew it, a key cycled it): provisional.  A lone thing
                # wearing a dot is not a selection — its fixed record is the
                # simpler reading and stands
                self.key_selected.setdefault(key, next(iter(vs)))
            # BODIES AND THE ACTIVE ONE (no mark needed): things that
            # always move together are one body; the body the keys move
            # is the active one; a key that moves nothing but after which
            # the keys move another body is a selector
            # (a press under which something TURNED says nothing about
            # bodies: the pieces of a turning thing shift by different
            # vectors and are still one thing)
            turned_press = any(_eff(b_by[s.sid], s)[0] == "rotate" for s in after if s.sid in b_by and "hud" not in s.tags)
            if not turned_press:
                for sid in moved:
                    self.moved_count[sid] += 1
                ms = sorted(moved)
                for i in range(len(ms)):
                    for j in range(i + 1, len(ms)):
                        if moved[ms[i]] == moved[ms[j]]:
                            self.comove[(ms[i], ms[j])] += 1
            if moved:
                bodies = self.bodies()
                owners = {b for b in bodies for sid in moved if sid in b}
                # (things that moved together by one vector, touching, are one
                # body on this press — a piece and the marks on its sides that
                # had never moved before)
                if len(owners) > 1 and len(set(moved.values())) == 1:
                    ms_ = [a_by[m] for m in moved if m in a_by]
                    if all(any(abs(r - r2) + abs(c - c2) <= 2 for (r, c) in x.cells for y in ms_ if y is not x for (r2, c2) in y.cells) for x in ms_) if len(ms_) > 1 else True:
                        owners = {frozenset(moved)}
                # (a PART that moved alone — a mark read apart from its piece —
                # is not a body the keys move: the body keeps its identity)
                if len(owners) == 1 and (any(m_ < 1000 for m_ in next(iter(owners))) or not _claims.on("lone_part_is_no_body")):
                    active = next(iter(owners))
                    vset = set(moved.values())
                    if len(vset) == 1:
                        v = next(iter(vset))
                        # (a body carrying a thing is the same body: what is
                        # bound to it is not a second body handed the key)
                        self.body_moves[key].append(((frozenset(a_ for a_ in active if a_ not in {o_ for (_, o_) in self.bound}) or active) if _claims.on("carried_is_no_second_body") else frozenset(active), v))
                        bset = {b for b, _ in self.body_moves[key]}
                        vs2 = Counter(v_ for _, v_ in self.body_moves[key])
                        # (two bodies handed the key in turn — DISJOINT bodies:
                        # a pair that moved together and then one of them alone,
                        # the other held by a wall, is one law on two things,
                        # not a selection)
                        if len(bset) >= 2 and len(vs2) == 1:
                            self.key_selected.setdefault(key, v)     # (meaningful only once a way to select is known: selection_in_play)
                        # (once clicking is known to choose the thing the keys
                        # move, a key that moved the chosen thing is a key of
                        # the chosen thing — one sight, as the others' were)
                        elif self.select_kinds and self.key_selected and len(vs2) == 1 and key not in self.key_selected \
                                and sel_b is not None and sel_b in active:
                            self.key_selected[key] = v
                    if self.pending_cycle is not None:
                        k0, before_active = self.pending_cycle
                        if before_active is not None and before_active != active:
                            self.cycle_bodies.setdefault(k0, []).append((before_active, active))
                            self.key_selected.setdefault(key, v)
                        self.pending_cycle = None
                    # A CLICK THAT RECOLOURED A THING which the keys then
                    # move: the click selected it, and its colour shows it
                    # (a mark is not the only way selection is drawn)
                    if self.pending_select is not None:
                        psid, pkind, pclick = self.pending_select
                        if _ENV.get("SELECT_DEBUG"):
                            print(f"   [select] pending {self.pending_select} key {key} moved {sorted(moved)} active {sorted(active)}")
                        if psid in moved and psid in active:
                            self.select_kinds.add(pkind)
                            self.select_click[pkind] += 1
                            self.select_by_colour[psid] = pkind
                            self.selected_colour = kind(b_by[psid]) if psid in b_by else None
                            self.rel_keys.add(pclick)
                        self.pending_select = None
                    self.active = active
                    self.ever_active |= set(active)
                    # (a bound pair seen to move apart is bound no more)
                    for a_, b_ in list(getattr(self, "bound", ())):
                        if a_ in a_by and b_ in a_by and ((a_ in moved) != (b_ in moved)):
                            self.bound.discard((a_, b_))
                    # PASSING THROUGH: a moved thing now standing on cells another
                    # thing stood on (the other still there, or hidden under it)
                    # is not stopped by that kind — the two pass through each
                    # other (a piece slides over another to dock); solidity is
                    # learned, not assumed
                    ck_ = self.carried_kinds()
                    for m_ in active:
                        s_m = a_by.get(m_)
                        if s_m is None or m_ >= 1000:
                            continue
                        # (a thing of the carried kinds moving with the body is
                        # carried: it passes for itself, not for the body)
                        if ck_ and kind(s_m) in ck_ and len(active) > 1:
                            continue
                        mc_ = set(s_m.cells)
                        for o_ in before:
                            if o_.sid == m_ or o_.sid >= 1000 or o_.sid in active or "hud" in o_.tags or "terrain" in o_.tags or o_.size <= 4:
                                continue
                            # (a thing that MOVED AWAY this press — pushed ahead — was not
                            # passed through: the mover stands where it stood because it
                            # shoved it; nor is a kind the body pushes ever passable —
                            # tw01, "the player passes through blocks" carried to L2, and
                            # no push was ever planned there)
                            if (o_.sid in moved or kind(o_) in self.pushable) and not _ENV.get("PASS_PUSHABLE"):
                                continue
                            if len(mc_ & set(o_.cells)) >= 3:
                                if _ENV.get("FLAT_DEBUG"):
                                    print(f"   passable? S{m_} kind {kind(s_m)} {s_m.bbox} over S{o_.sid} kind {kind(o_)} {o_.bbox} key {key} active {sorted(active)} bound {sorted(getattr(self, 'bound', ()))}")
                                self.passable.add((kind(s_m), kind(o_)))
                                self.passable.add((kind(o_), kind(s_m)))
            elif not isinstance(key, tuple) and not self.fixed.moving(key):
                self.pending_cycle = (key, self.active)
            # CONTACT: a thing that moved by v into another that moved too
            # (by v, or further: a shove that slides) — contact transmits
            # motion; the other's displacement is what a push does to it
            # A push is told from a chain that moves as one body by the
            # press where the other thing did NOT move (no contact): a
            # pushed thing moves only when hit.
            # displacement of everything, translate or not (a shoved thing
            # may pass through a transition frame)
            disp = {}
            for s in after:
                p = b_by.get(s.sid)
                if p is None or "hud" in s.tags or "terrain" in s.tags:
                    continue
                cb, ca = _centre(p.cells), _centre(s.cells)
                dv = (ca[0] - cb[0], ca[1] - cb[1])
                if abs(dv[0]) + abs(dv[1]) >= 1:
                    disp[s.sid] = (int(round(dv[0])), int(round(dv[1])))
            # the intended moves: what the key does to the marked thing
            # (even when the press was refused), and the fixed translates
            intents = dict(moved)
            if getattr(self, "_put_back_now", False) and self.active and _ENV.get("PUT_BACK_SKIP"):
                # (the controlled things were set back to the start: their
                # "intents" this press are no refusal of any ground, no wall)
                intents = {s_: v_ for s_, v_ in intents.items() if s_ not in self.active and (s_ // 1000 if s_ >= 1000 else s_) not in self.active}
            key_intents = set()             # what the KEY itself meant to move
            v_sel = self.key_selected.get(key) if self.selection_in_play() else None
            sel_now = [h for h, col in marks_b.items() if self.select_color is None or col == self.select_color]
            if sel_now:
                self.last_sel = sel_now[0]
            if v_sel is not None and self.last_sel in b_by:
                intents.setdefault(self.last_sel, v_sel)
                key_intents.add(self.last_sel)
            elif v_sel is not None and self.active:
                # no mark in play: the key meant to move the active body
                for sid_ in self.active:
                    if sid_ in b_by and sid_ < 1000:
                        intents.setdefault(sid_, v_sel)
                        key_intents.add(sid_)
            for sid_, e_ in self.fixed.moving(key).items():
                if getattr(self, "_put_back_now", False) and self.active and _ENV.get("PUT_BACK_SKIP") and (sid_ in self.active or (sid_ // 1000 if sid_ >= 1000 else sid_) in self.active):
                    continue
                # with a selection in play, only the selected thing's move is
                # the key's own; without one, the key's fixed translates are
                # (of the active body, when one is known: a HUD icon that
                # shifts under a key teaches nothing of the ground)
                if e_[0] == "translate" and sid_ in b_by and not self.selection_in_play() \
                        and (not self.active or sid_ in self.active or "hud" not in b_by[sid_].tags and b_by[sid_].size > 3 and not self.active):
                    key_intents.add(sid_)
                # a selectable thing moves only when it is the selected
                # one: its fixed record is no intent when it is not
                if e_[0] == "translate" and sid_ in b_by and sid_ not in intents \
                        and kind(b_by[sid_]) not in self.select_kinds and not self.selection_in_play():
                    intents[sid_] = e_[1]
            # A WALL BETWEEN HERE AND THERE: a move the key makes elsewhere
            # was refused from this place (nothing moved, nothing pushed):
            # remembered by place and direction
            if not moved and len(intents) == 1:
                # (a refusal of one walker is about its place; a refusal of
                # a chain of things is about the chain's whole configuration
                # and is not remembered by any one piece's place)
                occ = set()
                for o in before:
                    if "terrain" not in o.tags and "hud" not in o.tags and kind(o) not in self.flat and "hidden" not in o.tags:
                        occ |= set(o.cells)     # flat things are walked over: not occupants
                for sid_, v_ in intents.items():
                    p_ = b_by[sid_]
                    a_ = a_by.get(sid_)
                    if a_ is None or set(a_.cells) != set(p_.cells):
                        continue        # it changed (a flash, a transition frame): no verdict
                    dest = _shift(p_.cells, v_) - set(p_.cells)
                    if dest & occ:
                        # something stood there: it refused the walker — a wall
                        # of its kind (a pushable thing is told apart by the
                        # push record)
                        for o in before:
                            if "terrain" not in o.tags and "hud" not in o.tags and set(o.cells) & dest and kind(o) not in self.pushable:
                                self.wall_kinds[kind(o)] += 1
                        continue
                    # A COLOUR IS FLAT WHERE IT WAS WALKED, NOT WHEREVER IT SHOWS:
                    # refused into cells of a walkable colour with nothing else
                    # there, the PATCH of that colour those cells belong to (its
                    # connected run in the frame, when it is no floor the body
                    # stands on — a quarter of the frame at most) is a wall for
                    # this level.  (wa30: a hatched fence drawn in the pens'
                    # floor colour, learned one refused cell at a time — L4's
                    # yard had the body bump it at every spot before carrying)
                    try:
                        if frame is not None and not _ENV.get("NO_WALL_PATCH"):
                            fr_ = np.asarray(frame)
                            H_, W_ = fr_.shape
                            bgc_ = int(np.bincount(fr_.ravel()).argmax())
                            walk_ = set(self.flat) | set(self.ground_ok)
                            # (the ground's cells only: a THING standing there — a block
                            # that would not budge — is the push record's, not a wall patch)
                            thing_cells_ = set()
                            for o in before:
                                if "terrain" not in o.tags and "hud" not in o.tags:
                                    thing_cells_ |= set(o.cells)
                            seeds_ = [(r, c) for (r, c) in dest if 0 <= r < H_ and 0 <= c < W_ and int(fr_[r, c]) != bgc_ and int(fr_[r, c]) in walk_ and (r, c) not in thing_cells_]
                            if seeds_ and len(seeds_) >= max(1, len(dest) // 2):
                                col_ = int(fr_[seeds_[0]])
                                patch_ = set(); todo_ = [q for q in seeds_ if int(fr_[q]) == col_]
                                while todo_ and len(patch_) <= (H_ * W_) // 4:
                                    q = todo_.pop()
                                    if q in patch_:
                                        continue
                                    patch_.add(q)
                                    for dr in (-1, 0, 1):
                                        for dc in (-1, 0, 1):
                                            r2, c2 = q[0] + dr, q[1] + dc
                                            if 0 <= r2 < H_ and 0 <= c2 < W_ and (r2, c2) not in patch_ and int(fr_[r2, c2]) == col_:
                                                todo_.append((r2, c2))
                                if len(patch_) <= (H_ * W_) // 4:
                                    self.wall_patches |= patch_
                                    if _ENV.get("NOTE_DEBUG") or getattr(self, "verbose", False):
                                        print(f"   wall: the patch of colour {col_} at {min(patch_)}..{max(patch_)} ({len(patch_)} cells) refused the body: a wall here, whatever the colour is elsewhere")
                    except Exception:
                        pass
                    self.refuse((kind(p_), p_.bbox[0], p_.bbox[1], v_), frame=frame, size=(p_.bbox[2] - p_.bbox[0] + 1, p_.bbox[3] - p_.bbox[1] + 1))
            for sid, v in intents.items():
                if sid >= 1000:
                    continue                # (a part rides its thing: it pushes nothing)
                p = b_by[sid]
                ahead = _shift(p.cells, v)
                for o in before:
                    if o.sid == sid or "terrain" in o.tags or "hud" in o.tags:
                        continue
                    if o.sid >= 1000 and o.sid // 1000 == sid:
                        continue                # (its own part — the centre it wears — is not a thing it pushes)
                    w = disp.get(o.sid)
                    if ahead & set(o.cells) and w is not None and (w[0] * v[0] > 0 or w[1] * v[1] > 0):
                        if _ENV.get("CONTACT_DEBUG"):
                            print(f"   [contact] key {key} mover S{sid} v {v} moved {sid in moved} -> S{o.sid} kind {kind(o)} {o.bbox} w {w}")
                        self.contact_seen[(key, o.sid)] = (kind(o), (v[0] > 0) - (v[0] < 0), (v[1] > 0) - (v[1] < 0), (w, sid in moved), max(abs(v[0]), abs(v[1])))
                        self.pushed_count[(key, o.sid)] += 1
            # WHAT THE GROUND ALLOWS: the colours under an accepted move's
            # destination against those under a refused one — a colour
            # only ever refused is a wall (the letterbox, a pit)
            if frame is not None:
                # (what stands in the way and might have been the refusal's
                # cause: the things that MOVED this press (pushed, shoved)
                # and the mover's own body — a tile that stayed put under an
                # accepted move is ground; under a refused one, a wall)
                occupied = set()
                mover_body = set(self.active or ()) | set(moved)
                for o in before:
                    if "terrain" not in o.tags and "hud" not in o.tags:
                        if o.sid in mover_body or (o.sid // 1000 if o.sid >= 1000 else o.sid) in mover_body:
                            occupied |= set(o.cells)
                        # (a thing of a kind that pushes is a thing in the way,
                        # never the ground: a push it refused — blocked behind —
                        # says nothing of its colour)
                        elif kind(o) in self.pushable or kind(o) in self.select_kinds:
                            occupied |= set(o.cells)
                        # (A THING THAT LEFT ITS PLACE AS THE MOVER ARRIVED IS NOT
                        # THE GROUND THERE: whatever stood on the cells crossed and
                        # is not there after — moved, split off, vanished — was in
                        # the way and gave way; only what STAYS under the mover,
                        # or lies hidden beneath it, is ground.  tw01: a block
                        # pushed out of its pair was born anew, in no 'moved'
                        # record, and its colour became ground the walker may
                        # cross — the plan then walked through the blocks)
                        elif o.sid < 1000 and o.size > 1 and not _ENV.get("NO_GAVE_WAY"):
                            o_a_ = a_by.get(o.sid)
                            if o_a_ is None or ("hidden" not in o_a_.tags and set(o_a_.cells) != set(o.cells) and not (set(o_a_.cells) & set(o.cells))):
                                occupied |= set(o.cells)
                for sid, v in intents.items():
                    if sid >= 1000 or sid not in key_intents:
                        continue            # a part rides; a shoved thing teaches nothing of the ground
                    p = b_by[sid]
                    law_ = self.fixed.moving(key).get(sid)
                    if law_ is not None and law_[0] == "translate" and law_[1] != v and sid in moved:
                        # A SLIDE TEACHES AT BOTH ENDS: what it crossed is
                        # ground; what it stopped against — the cells one step
                        # on from where it came to rest — refused it (a thing
                        # there: a wall of its kind; else the ground's colour)
                        if (key, sid) in getattr(self.fixed, "slides", ()) and sid in a_by and frame_after is not None \
                                and ((law_[1][0] == 0) == (v[0] == 0)) and ((law_[1][1] == 0) == (v[1] == 0)) \
                                and (law_[1][0] * v[0] >= 0 and law_[1][1] * v[1] >= 0):
                            H_, W_ = frame.shape
                            crossed_ = _sweep(p.cells, v) - set(p.cells)
                            own_s = set()
                            for o in before:
                                if o.sid != sid and o.sid >= 1000 and o.sid // 1000 == sid:
                                    own_s |= set(o.cells)
                            crossed_ -= own_s
                            if crossed_ and not (crossed_ & (occupied - own_s)) and all(0 <= r < H_ and 0 <= c < W_ for r, c in crossed_):
                                if _ENV.get("GROUND_DEBUG"):
                                    print(f"   [ground] key {key} S{sid} slide crossed: {sorted({int(frame[r, c]) for r, c in crossed_})}")
                                self.ground_ok |= {int(frame[r, c]) for r, c in crossed_}
                            beyond_ = _shift(a_by[sid].cells, law_[1]) - set(a_by[sid].cells)
                            if beyond_ and all(0 <= r < H_ and 0 <= c < W_ for r, c in beyond_):
                                hit_ = [o for o in after if o.sid != sid and o.sid < 1000 and "hud" not in o.tags and set(o.cells) & beyond_
                                        and (o.sid // 1000 if o.sid >= 1000 else o.sid) not in (self.active or ())]
                                if hit_:
                                    for o in hit_:
                                        if kind(o) not in self.pushable and "terrain" not in o.tags:
                                            self.wall_kinds[kind(o)] += 1
                                        elif "terrain" in o.tags:
                                            for r, c in beyond_ & set(o.cells):
                                                self.ground_refused[int(frame_after[r, c])] += 1
                                else:
                                    for r, c in beyond_:
                                        self.ground_refused[int(frame_after[r, c])] += 1
                        continue            # not the key's step (a portal, a shove): the ground crossed teaches nothing
                    dest = _sweep(p.cells, v)                # the cells crossed, not the landing alone
                    if any(r < 0 or c < 0 or r >= frame.shape[0] or c >= frame.shape[1] for r, c in dest):
                        continue
                    # (the mover's own parts — the centre it wears — go with it:
                    # what its outline crosses of its own inside is not ground)
                    own_ = set()
                    for o in before:
                        if o.sid != sid and o.sid >= 1000 and o.sid // 1000 == sid:
                            own_ |= set(o.cells)
                    dest = dest - own_
                    if dest & (occupied - own_):
                        continue            # something was there: not the ground's doing
                    cols = {int(frame[r, c]) for r, c in dest}
                    if moved.get(sid) == v:
                        # THE GROUND IS WHAT STAYS: of the cells crossed, those the
                        # press left as they were; and what shows where the
                        # mover stood (a cell that changed with the move was
                        # the mover's own — an outline in the wall's colour)
                        if frame_after is not None:
                            stay = {p_ for p_ in dest if int(frame_after[p_[0], p_[1]]) == int(frame[p_[0], p_[1]])}
                            left = set(p.cells) - set(a_by[sid].cells) if sid in a_by else set()
                            # (what shows where the mover stood is ground only where
                            # NO THING now stands: a thing that moved into the
                            # mover's old place — a block the helper shoved there,
                            # tw01 — is a thing, not the floor)
                            taken_ = set()
                            own_body_ = {sid} | {m_ for b_ in self.bodies() if sid in b_ for m_ in b_} | set(self.active or ())
                            for o_ in after:
                                if o_.sid not in own_body_ and o_.sid < 1000 and o_.size > 1 and "terrain" not in o_.tags and "hud" not in o_.tags:
                                    # (only cells the thing did NOT hold before: a tile
                                    # that lay hidden under the walker all along is the
                                    # ground it walked on — tu93; what ARRIVED is a thing)
                                    was_ = set(b_by[o_.sid].cells) if o_.sid in b_by else set()
                                    taken_ |= set(o_.cells) - was_
                            if not _ENV.get("NO_LEFT_TAKEN"):
                                left -= taken_
                            cols = {int(frame[r, c]) for r, c in stay} | {int(frame_after[r, c]) for r, c in left
                                                                           if 0 <= r < frame.shape[0] and 0 <= c < frame.shape[1] and int(frame_after[r, c]) >= 0}
                            # THE MOVER WALKS ON WHAT IT WAS SEEN ON: what shows where the
                            # mover stood is the ground under it.  When that ground is a
                            # colour other than the background, the mover rides a track of
                            # that colour, and the background is a bank to it until a press
                            # shows otherwise (g50t: a ring on a 5-track, planned over the
                            # 0 background — "stayed, nothing in the way" — and the law
                            # dropped).  When the ground under it IS the background, the
                            # floor is the floor and nothing changes.
                            try:
                                if sid in (self.active or ()) and left:
                                    for r, c in left:
                                        if 0 <= r < frame.shape[0] and 0 <= c < frame.shape[1] and int(frame_after[r, c]) >= 0:
                                            self.uncovered[int(frame_after[r, c])] += 1
                                    if _ENV.get("GROUND_DEBUG"):
                                        print(f"   [ground] uncovered under S{sid} by {key}: {dict(self.uncovered)} ground_ok {sorted(self.ground_ok)}")
                            except Exception:
                                pass
                            # A LIVERY IN THE WALL'S COLOUR: cells beside the body
                            # that changed with the move and wore a colour the
                            # ground has refused are the body's own outline —
                            # they go where it goes and are no wall to it
                            H_, W_ = frame.shape
                            body_b = set()
                            for m_ in (self.active or ()):
                                if m_ in b_by:
                                    body_b |= set(b_by[m_].cells)
                            ring_ = {(r + dr_, c + dc_) for (r, c) in body_b for dr_ in (-1, 0, 1) for dc_ in (-1, 0, 1)} - body_b
                            outline_ = {q_ for q_ in ring_ if 0 <= q_[0] < H_ and 0 <= q_[1] < W_
                                        and int(frame_after[q_[0], q_[1]]) in cols and int(frame[q_[0], q_[1]]) not in cols
                                        and int(frame[q_[0], q_[1]]) not in self.ground_ok}
                            if _ENV.get("LIVERY_DEBUG"):
                                print(f"   ground: stay5 {[q_ for q_ in stay if int(frame[q_[0], q_[1]]) == 5][:6]} left5 {[q_ for q_ in left if 0 <= q_[0] < H_ and 0 <= q_[1] < W_ and int(frame_after[q_[0], q_[1]]) == 5][:6]}")
                                print(f"   livery? sid {sid} active {self.active} cols {cols} ring changed {[(q_, int(frame[q_[0], q_[1]]), int(frame_after[q_[0], q_[1]])) for q_ in ring_ if 0 <= q_[0] < H_ and 0 <= q_[1] < W_ and int(frame_after[q_[0], q_[1]]) != int(frame[q_[0], q_[1]])][:8]} outline {sorted(outline_)[:6]}")
                            if outline_ and sid == min(m_ for m_ in self.active if m_ in b_by):
                                b0_ = _bbox(body_b)
                                self.livery_votes[frozenset((q_[0] - b0_[0], q_[1] - b0_[1], int(frame[q_[0], q_[1]])) for q_ in outline_)] += 1
                        if _ENV.get("GROUND_DEBUG") and (cols - self.ground_ok):
                            print(f"   [ground] key {key} S{sid} v {v} dest cols {sorted(cols)} new {sorted(cols - self.ground_ok)}; stay {len(stay) if frame_after is not None else None} left {sorted({int(frame_after[r, c]) for r, c in left}) if frame_after is not None else None}; mover {p.bbox}->{a_by[sid].bbox if sid in a_by else None}")
                        self.ground_ok |= cols
                        for c_ in cols:
                            n_c = sum(1 for r, c in stay if int(frame[r, c]) == c_) + sum(1 for r, c in left if 0 <= r < frame.shape[0] and 0 <= c < frame.shape[1] and int(frame_after[r, c]) == c_)
                            self.ground_size[c_] = max(self.ground_size.get(c_, 0), n_c)
                    elif sid not in moved:
                        if _ENV.get("GROUND_DEBUG"):
                            print(f"   [ground] refused: key {key} S{sid} v {v} dest cols {sorted(cols)} mover {p.bbox} put_back {getattr(self, '_put_back_now', False)}")
                        for c in cols:
                            self.ground_refused[c] += 1
            # FLAT THINGS: a mover that came to lie over another's cells
            # (that other lost those cells) walks over it — a frame on the
            # floor, a painted patch: no wall
            # (the thing under a click this press is what the click did to it
            # — removed, turned — not a thing walked over)
            clicked_now_ = set()
            if click is not None:
                for s_ in before:
                    if s_.bbox[0] <= click[0] <= s_.bbox[2] and s_.bbox[1] <= click[1] <= s_.bbox[3] and "terrain" not in s_.tags:
                        clicked_now_.add(s_.sid); clicked_now_.add(s_.sid // 1000 if s_.sid >= 1000 else s_.sid)
            for sid, v in disp.items():
                s_now = a_by.get(sid)
                if s_now is None or s_now.size <= 3:
                    continue
                # (only the body the keys move walks over things; on a scroll press
                # a thing's twin is taken for it elsewhere — no reading)
                if (self.active and sid not in self.active and (sid // 1000 if sid >= 1000 else sid) not in self.active) or getattr(self, "_scroll_now", False):
                    continue
                # (a thing carried by the body — bound to it — lies over what
                # IT passes, not what the body walks over: a crate on the
                # hatching says nothing of the body's way)
                bound_ = {o_ for (_, o_) in getattr(self, "bound", ())}
                if sid in bound_ or (sid // 1000 if sid >= 1000 else sid) in bound_:
                    continue
                # (...and a thing of the carried kinds moving with others is
                # carried, whatever the bond record says: the tracker's cut
                # may have let the bond go)
                ck_ = self.carried_kinds()
                if ck_ and kind(s_now) in ck_ and len(disp) > 1:
                    continue
                now_cells = set(s_now.cells)
                for o in before:
                    if o.sid == sid or "terrain" in o.tags or "hud" in o.tags or o.size <= 3 or o.sid in clicked_now_ or (o.sid // 1000 if o.sid >= 1000 else o.sid) in clicked_now_:
                        continue
                    if o.sid in moved or (o.sid // 1000 if o.sid >= 1000 else o.sid) in moved:
                        continue            # (a thing that moved away this press was pushed, not walked over)
                    covered = now_cells & set(o.cells)
                    o_now = a_by.get(o.sid)
                    # the other lost those cells, or (under permanence) still
                    # holds them beneath the mover: either way it is under
                    # (a thing that is gone altogether was removed, not walked over:
                    # only a thing still there, under or beside the mover, is flat)
                    # (A PART PUSHED OFF IS NOT WALKED OVER: a thing that lost cells to the
                    # mover while a new thing of its colour and of that very size
                    # appeared this press was two things touching, one of them pushed
                    # ahead — tw01, two blocks read as one, "flat" after the first push)
                    # (a kind the body is seen to PUSH is solid by that very fact: never flat)
                    if kind(o) in self.pushable and not _ENV.get("FLAT_PUSHABLE"):
                        continue
                    lost_n_ = (o.size - o_now.size) if o_now is not None else 0
                    born_like_ = lost_n_ > 0 and any(x_.body == o.body and x_.sid < 1000 and x_.sid != o.sid
                                                      and ((x_.sid not in b_by and x_.size == lost_n_)                        # born, of that size
                                                           or (x_.sid in b_by and x_.size - b_by[x_.sid].size == lost_n_))   # or a like thing grown by it (joined another in the pen)
                                                      for x_ in after)
                    if born_like_:
                        continue
                    # (...and a thing now standing displaced by the mover's own step is
                    # a thing the mover pushed ahead, whatever the identity record says)
                    if o_now is not None and (o_now.bbox[0] - o.bbox[0], o_now.bbox[1] - o.bbox[1]) == tuple(v) and tuple(v) != (0, 0):
                        continue
                    # (...or the very cells the mover now covers show up in the thing
                    # displaced by the mover's step: that part of it was pushed ahead —
                    # two blocks touching at a corner, read as one)
                    if o_now is not None and covered and tuple(v) != (0, 0) and all((r_ + v[0], c_ + v[1]) in o_now.cells for (r_, c_) in covered):
                        continue
                    if covered and o_now is not None and (not (covered & set(o_now.cells))
                                                          or "hidden" in o_now.tags or (covered & set(o_now.cells))):
                        if _ENV.get("FLAT_DEBUG"):
                            print(f"   flat? S{sid} now {s_now.bbox} over S{o.sid} kind {kind(o)} {o.bbox} covered {len(covered)} o_now {o_now.bbox} tags {sorted(o_now.tags)} key {key} click {click}; after of its colour: {[(x_.sid, x_.size, x_.bbox, x_.sid in b_by and b_by[x_.sid].size) for x_ in after if x_.body == o.body and x_.sid < 1000]}")
                        self.flat.add(kind(o))
                        self.flat_size[kind(o)] = max(self.flat_size.get(kind(o), 0), o.size)
            if moved:
                for o in before:
                    if o.sid not in moved and o.sid in a_by:
                        self.stayed[(key, o.sid)] += 1
            for (k2, osid), (ko, dr, dc, w, step_) in list(self.contact_seen.items()):
                if k2 != key:
                    continue
                # (a thing bound to the body — carried — moved with it: not pushed)
                if getattr(self, "carry", None) and ko in self.carry.get("kinds", ()):
                    continue
                # some press of this key: something moved, it did not
                if self.stayed.get((key, osid), 0) > 0:
                    self.pushable.add(ko)
                    self.push_vec[(ko, dr, dc)] = w
                    # WHERE IT CAME TO REST: against something (the frame's edge,
                    # a thing, a colour the ground refuses) — then it SLID UNTIL
                    # STOPPED and the run seen is no law of its own; with open
                    # ground beyond, the run is the law (tw03: one push sent the
                    # block nine cells to the pen and the wall; the next level's
                    # pushes go other distances)
                    o_a_ = a_by.get(osid)
                    if o_a_ is not None and frame_after is not None and any(w):
                        H_, W_ = frame_after.shape
                        # (the cells one pixel on from where it rests, in the shove's direction)
                        beyond_ = _shift(set(o_a_.cells), (dr, dc)) - set(o_a_.cells)
                        self.push_step[(ko, dr, dc)] = step_
                        if any(r_ < 0 or c_ < 0 or r_ >= H_ or c_ >= W_ for r_, c_ in beyond_):
                            self.push_stopped[(ko, dr, dc)] = True
                        else:
                            occ_ = any(set(x_.cells) & beyond_ for x_ in after if x_.sid != osid and "terrain" not in x_.tags and "hud" not in x_.tags and x_.sid < 1000)
                            cols_ = {int(frame_after[r_, c_]) for r_, c_ in beyond_}
                            bg_ = int(np.bincount(frame_after.ravel()).argmax())
                            # (what is not the floor nor the empty ground is a thing it rests against)
                            self.push_stopped[(ko, dr, dc)] = bool(occ_) or any(c_ != bg_ and c_ not in self.ground_ok for c_ in cols_)
                    # WHAT THE SHOVED THING CROSSED: the colours under its way
                    # from where it was to where it stopped (a pit the pusher
                    # may not enter, which the shoved thing slides over)
                    o_b, o_a = b_by.get(osid), a_by.get(osid)
                    if o_b is not None and o_a is not None and frame is not None:
                        # (the way it ACTUALLY went — from where it was to where
                        # it stopped, the cells strictly between, read in the
                        # frame before the press; the floor it and the mover
                        # stand on anyway is not what it "slid over")
                        swept_ = Counter()
                        vec_ = (o_a.bbox[0] - o_b.bbox[0], o_a.bbox[1] - o_b.bbox[1])
                        n_ = max(abs(vec_[0]), abs(vec_[1]))
                        own_ = set(o_b.cells)
                        for t_ in range(1, int(n_) + 1):
                            for (r_, c_) in o_b.cells:
                                q_ = (r_ + (vec_[0] * t_) // n_ if n_ else r_, c_ + (vec_[1] * t_) // n_ if n_ else c_)
                                if q_ in own_:
                                    continue
                                if 0 <= q_[0] < frame.shape[0] and 0 <= q_[1] < frame.shape[1]:
                                    swept_[int(frame[q_[0], q_[1]])] += 1
                        bg_ = Counter(int(x_) for x_ in frame.ravel()).most_common(1)[0][0] if n_ else None
                        for col_, cnt_ in swept_.items():
                            if col_ == kind(o_b) or col_ in set(o_b.cells.values()) or col_ == bg_ or col_ in self.ground_ok:
                                continue
                            if cnt_ >= 3:
                                self.slid_over[col_] += 1
                    # it moved because it was hit, not because of the key —
                    # and so did whatever rides on it (its mark)
                    self.fixed.eff[key].pop(osid, None)
                    self.fixed.contradicted[key].discard(("refused", osid))
                    host = b_by.get(osid)
                    if host is not None:
                        hb = host.bbox
                        for t in before:
                            if t.size <= 3 and t.sid != osid and hb[0] <= t.bbox[0] and hb[1] <= t.bbox[1] and hb[2] >= t.bbox[2] and hb[3] >= t.bbox[3]:
                                self.fixed.eff[key].pop(t.sid, None)
                                self.fixed.contradicted[key].discard(("refused", t.sid))
            # a bump that moved nothing: the thing ahead refused
            if not moved and self.last_selected_pos is not None:
                pass
        else:
            # did the mark move onto the clicked thing?
            clicked = [s for s in before if s.bbox[0] <= click[0] <= s.bbox[2] and s.bbox[1] <= click[1] <= s.bbox[3]
                       and "terrain" not in s.tags and "hud" not in s.tags]
            clicked = min(clicked, key=lambda s: s.size) if clicked else None
            if clicked is not None and clicked.size <= 3:
                host = livery_host(clicked, before)
                if host is not None:
                    clicked = host          # a click on the mark is a click on its wearer
            if clicked is not None and clicked.sid in marks_a and marks_a[clicked.sid] != marks_b.get(clicked.sid):
                self.select_click[kind(clicked)] += 1
                self.select_kinds.add(kind(clicked))
                self.rel_keys.add(key)      # a selection click is not a button
            # a click that recoloured the thing under it in place: it may
            # have selected it — the next key press tells
            if clicked is not None:
                host_ = clicked.sid // 1000 if clicked.sid >= 1000 else clicked.sid
                for sid_ in (clicked.sid, host_):
                    p_, s2 = b_by.get(sid_), a_by.get(sid_)
                    if p_ is not None and s2 is not None and set(p_.cells) == set(s2.cells) and dict(p_.cells) != dict(s2.cells) and sid_ < 1000:
                        self.pending_select = (sid_, kind(p_), key)
                        break
                if _ENV.get("SELECT_DEBUG"):
                    print(f"   [select] click {click} on {clicked.sid if clicked is not None else None} -> pending {self.pending_select}; recoloured {[s2.sid for s2 in after if s2.sid in b_by and set(b_by[s2.sid].cells) == set(s2.cells) and dict(b_by[s2.sid].cells) != dict(s2.cells)]}")
            # A CLICK THAT REMOVES: the thing under the click is gone after
            # it (its cells wear the ground) — a click on a thing of its
            # kind takes it away (a breakable block); a click after which
            # the thing stays contradicts it
            if _ENV.get("MOVE_DEBUG"):
                print(f"   click {click}: clicked {(clicked.sid, clicked.body, clicked.size, sorted(clicked.tags)) if clicked is not None else None} in after {clicked is not None and clicked.sid in a_by}")
            whole_ = clicked
            if clicked is not None and clicked.sid >= 1000:
                whole_ = b_by.get(clicked.sid // 1000)      # (a part's click is its object's)
            if _ENV.get("RAISE_DEBUG") and click is not None:
                print(f"   click {click}: clicked {(clicked.sid, clicked.body, clicked.size, sorted(clicked.tags)) if clicked is not None else None} whole {(whole_.sid, whole_.size, sorted(whole_.tags)) if whole_ is not None else None} frames {frame is not None} {frame_after is not None}")
            if whole_ is not None and whole_.sid < 1000 and whole_.size <= 160 and "terrain" not in whole_.tags and frame_after is not None and frame is not None:
                # (gone = its cells now wear what surrounded it: the ground
                # closed over it; a thing that changed colour or shape is not gone)
                H_, W_ = frame.shape
                b_ = whole_.bbox
                ring_ = {(r_, c_) for r_ in range(b_[0] - 1, b_[2] + 2) for c_ in range(b_[1] - 1, b_[3] + 2)
                         if 0 <= r_ < H_ and 0 <= c_ < W_ and (r_, c_) not in whole_.cells}
                surround_ = ({int(frame[r_, c_]) for (r_, c_) in ring_} | set(self.ground_ok) | set(self.flat)) - set(whole_.colors)
                after_cols = [int(frame_after[r_, c_]) for (r_, c_) in whole_.cells]
                n_ground = sum(1 for c_ in after_cols if c_ in surround_)
                gone_ = bool(surround_) and n_ground == len(after_cols)
                if not gone_ and surround_ and n_ground >= 0.6 * len(after_cols):
                    # MOSTLY GONE, A MARKING LEFT: the ground closed over most of
                    # it and what remains of its colour is lone cells (a box
                    # that became a dotted mark of itself — passable ground
                    # with a sign on it, until a click makes it a box again)
                    left_ = {(r_, c_) for (r_, c_) in whole_.cells if int(frame_after[r_, c_]) not in surround_}
                    lone_ = all(not any((r_ + dr_, c_ + dc_) in left_ for dr_, dc_ in ((1, 0), (-1, 0), (0, 1), (0, -1))) for (r_, c_) in left_)
                    if lone_ and len(left_) <= 8:
                        gone_ = True
                        self.click_marks[kind(whole_)] += 1
                if gone_:
                    self.click_gone[kind(whole_)] += 1
                    self.click_gone_size[kind(whole_)] = min(self.click_gone_size.get(kind(whole_), 10 ** 6), whole_.size)
                    # WHAT THE REMOVAL CLEARS: the cells that took a ground
                    # colour with it beyond the thing's own — its box grown
                    # by so much (a tile's black edges go with the tile)
                    if frame_after is not None and frame is not None:
                        b_ = whole_.bbox
                        H_, W_ = frame.shape
                        cleared_ = {(r_, c_) for r_ in range(max(0, b_[0] - 3), min(H_, b_[2] + 4)) for c_ in range(max(0, b_[1] - 3), min(W_, b_[3] + 4))
                                    if int(frame[r_, c_]) != int(frame_after[r_, c_]) and (r_, c_) not in whole_.cells}
                        if cleared_:
                            grow_ = (b_[0] - min(r_ for r_, _ in cleared_), b_[1] - min(c_ for _, c_ in cleared_),
                                     max(r_ for r_, _ in cleared_) - b_[2], max(c_ for _, c_ in cleared_) - b_[3])
                            grow_ = tuple(max(0, g_) for g_ in grow_)
                            old_ = self.click_clears.get(kind(whole_), (0, 0, 0, 0))
                            self.click_clears[kind(whole_)] = tuple(max(a_, g_) for a_, g_ in zip(old_, grow_))
                else:
                    self.click_stays[kind(whole_)] += 1
                    # A CLICK THAT RAISES: a small thing (a mark) after which a
                    # thing of its colour several times its size stands over
                    # its place — the click raised it (the toggle's other way:
                    # the mark clicked is the block again)
                    # (the mark clicked is the connected piece under the click:
                    # three marks in a row may be read as one constellation)
                    cells_w = set(whole_.cells)
                    seed_ = min(cells_w, key=lambda q: abs(q[0] - click[0]) + abs(q[1] - click[1]))
                    piece_ = {seed_}; front_ = [seed_]
                    while front_:
                        q = front_.pop()
                        for dr_ in (-1, 0, 1):
                            for dc_ in (-1, 0, 1):
                                q2 = (q[0] + dr_, q[1] + dc_)
                                if q2 in cells_w and q2 not in piece_:
                                    piece_.add(q2); front_.append(q2)
                    if _ENV.get("RAISE_DEBUG"):
                        print(f"   raise? clicked S{whole_.sid} {whole_.body}/{whole_.size}@{whole_.bbox} piece {len(piece_)} after-near {[(s2.sid, s2.body, s2.size, s2.bbox) for s2 in after if abs(s2.centre[0] - whole_.centre[0]) + abs(s2.centre[1] - whole_.centre[1]) <= 6 and 'terrain' not in s2.tags]}")
                    if len(piece_) <= 12:
                        b_ = _bbox(piece_)
                        over_ = [s2 for s2 in after if s2.sid < 1000 and "terrain" not in s2.tags and whole_.body in s2.colors
                                 and s2.size >= 3 * len(piece_) and s2.bbox[0] <= b_[0] and s2.bbox[1] <= b_[1] and s2.bbox[2] >= b_[2] and s2.bbox[3] >= b_[3]
                                 and s2.bbox[2] - s2.bbox[0] < 3 * (b_[2] - b_[0] + 1) and s2.bbox[3] - s2.bbox[1] < 3 * (b_[3] - b_[1] + 1)]
                        if over_:
                            s2 = min(over_, key=lambda x_: x_.size)
                            dims_ = (s2.bbox[2] - s2.bbox[0] + 1, s2.bbox[3] - s2.bbox[1] + 1)
                            self.click_grows[kind(whole_)] = (len(piece_), dims_)     # (the mark's size; the block's height and width, centred on the mark)
            # A COLOUR CLICK: a click on a small thing after which ANOTHER
            # thing took one of the clicked thing's colours on the same
            # cells — the click hands that colour on (a palette)
            if clicked is not None and clicked.size <= 60:
                host_sid = clicked.sid // 1000 if clicked.sid >= 1000 else clicked.sid
                for sid in changed_now:
                    if sid in (clicked.sid, host_sid) or sid >= 1000 or sid in self.counters:
                        continue
                    p = b_by[sid]; s_ = a_by[sid]
                    if "hud" in s_.tags or set(p.cells) != set(s_.cells):
                        continue
                    diff = {q: s_.cells[q] for q in s_.cells if p.cells[q] != s_.cells[q]}
                    cols = Counter(diff.values())
                    if diff and len(cols) == 1 and next(iter(cols)) in clicked.colors:
                        self.colour_clicks.append((click, clicked.sid, host_sid, sid, next(iter(cols)), frozenset((q[0] - p.bbox[0], q[1] - p.bbox[1]) for q in diff)))
                        self.colour_targets[sid] += 1
            # RELATIVE: a click not on a thing (or on the thing itself)
            # that moved something: the click's offset is the cause, not
            # the place — such a click is not a button
            if clicked is None or clicked.size > 60 or clicked.sid in moved:
                for sid, v in moved.items():
                    p = b_by[sid]
                    if "terrain" in p.tags or p.size > 60:
                        continue
                    c0 = _centre(p.cells)
                    off = (click[0] - c0[0], click[1] - c0[1])
                    self.rel[kind(p)].append((off, v))
                    self.rel_keys.add(key)
                # HUNG BETWEEN ITS HANDLES: a thing that moved a FRACTION of
                # what the clicked-to thing moved, and stood at the mean of
                # the like things' places before and after, sits at their
                # centroid — it is moved by moving them (Tushar, r11l: "select
                # an end, click far — the purple goes there", half way)
                tele_ = [sid for sid, v in moved.items() if sid in b_by
                         and abs((click[0] - _centre(b_by[sid].cells)[0]) - v[0]) + abs((click[1] - _centre(b_by[sid].cells)[1]) - v[1]) <= 1.5]
                for hs in tele_:
                    hp = b_by[hs]
                    hd = (hp.bbox[2] - hp.bbox[0] + 1, hp.bbox[3] - hp.bbox[1] + 1)
                    like_ = [s_ for s_ in before if s_.sid < 1000 and "terrain" not in s_.tags and "hud" not in s_.tags
                             and abs(s_.size - hp.size) <= max(2, 0.25 * hp.size)
                             and (s_.bbox[2] - s_.bbox[0] + 1, s_.bbox[3] - s_.bbox[1] + 1) == hd]
                    if len(like_) < 2 or hs not in {s_.sid for s_ in like_}:
                        continue
                    k_ = len(like_)
                    vh = moved[hs]
                    for sid, v in moved.items():
                        if sid == hs or sid >= 1000 or sid in {s_.sid for s_ in like_} or sid not in a_by:
                            continue
                        if abs(v[0] - vh[0] / k_) > 1.0 or abs(v[1] - vh[1] / k_) > 1.0:
                            continue
                        p = b_by[sid]; s_after = a_by[sid]
                        if "terrain" in p.tags or p.size > 100:
                            continue
                        mb_ = (sum(_centre(s_.cells)[0] for s_ in like_) / k_, sum(_centre(s_.cells)[1] for s_ in like_) / k_)
                        ma_ = (sum(_centre((a_by[s_.sid] if s_.sid in a_by else s_).cells)[0] for s_ in like_) / k_,
                               sum(_centre((a_by[s_.sid] if s_.sid in a_by else s_).cells)[1] for s_ in like_) / k_)
                        cb_, ca_ = _centre(p.cells), _centre(s_after.cells)
                        if abs(cb_[0] - mb_[0]) + abs(cb_[1] - mb_[1]) <= 2.0 and abs(ca_[0] - ma_[0]) + abs(ca_[1] - ma_[1]) <= 2.0:
                            self.centroid_of[sid] = tuple(sorted(s_.sid for s_ in like_))
                            self.centroid_kinds[kind(p)] = (hp.size, hd)
                            for s_ in like_:
                                self.select_presumed.add(kind(s_))     # (a handle answers a click by becoming the one the clicks move)
                # WHERE A CLICK MAY SEND A THING: the ground under a landing
                # that took is walkable; a click that sent nothing, where a
                # thing is known to go to the click, was refused by what lies
                # there — a wall to that thing (learned like a refused step)
                if frame is not None:
                    H_, W_ = frame.shape
                    for sid, v in moved.items():
                        p = b_by[sid]
                        if p.size > 60 or "terrain" in p.tags:
                            continue
                        c0 = _centre(p.cells)
                        if abs((click[0] - c0[0]) - v[0]) + abs((click[1] - c0[1]) - v[1]) <= 1.5:
                            dest_ = {(r + v[0], c + v[1]) for (r, c) in p.cells}
                            if _ENV.get("GROUND_DEBUG"):
                                print(f"   [ground] key {key} third site: {sorted({int(frame[r, c]) for (r, c) in dest_ if 0 <= r < H_ and 0 <= c < W_})}")
                            self.ground_ok |= {int(frame[r, c]) for (r, c) in dest_ if 0 <= r < H_ and 0 <= c < W_}
                    if not moved:
                        tele_kinds = {kind(s_) for s_ in before if s_.size <= 60 and "terrain" not in s_.tags and "hud" not in s_.tags
                                      and self.relative_law(kind(s_)) is not None and self.relative_law(kind(s_))[0] == "teleport"}
                        goers_ = [s_ for s_ in before if kind(s_) in tele_kinds and (not self.active or s_.sid in self.active)]
                        for s_ in goers_[:1]:
                            c0 = _centre(s_.cells)
                            v_ = (int(round(click[0] - c0[0])), int(round(click[1] - c0[1])))
                            dest_ = {(r + v_[0], c + v_[1]) for (r, c) in s_.cells}
                            for col_ in {int(frame[r, c]) for (r, c) in dest_ if 0 <= r < H_ and 0 <= c < W_}:
                                if col_ not in self.ground_ok:
                                    self.ground_refused[col_] += 1
                if not moved:
                    for s in before:
                        if s.size <= 60 and "terrain" not in s.tags and "hud" not in s.tags:
                            c0 = _centre(s.cells)
                            self.rel_miss[kind(s)].append(abs(click[0] - c0[0]) + abs(click[1] - c0[1]))

    def kinds_worn_by_selected(self):
        """Every colour a thing selected by colour has worn (selected and
        not): its fixed records are conditional on selection."""
        out = set(self.select_by_colour.values()) | {self.selected_colour}
        for sid in self.select_by_colour:
            out |= set(self.kinds_worn.get(sid, ()))
        return out - {None}

    def presume_selectable(self, comps):
        """A LAW LEARNED ON KINDS IS PRESUMED OF LIKE-FORMED THINGS: a thing
        of a kind not yet clicked, wearing what every selectable thing wore
        last level (its marks), is presumed selectable — the plan may select
        it; the click that does not select refutes the kind."""
        if not self.select_form and self.select_kinds:
            # (the form is read on this level's selectable things when the
            # last level taught none: what they all wear)
            form_ = None
            for s in comps:
                if s.sid < 1000 and kind(s) in self.select_kinds and "hud" not in s.tags and "terrain" not in s.tags:
                    w_ = worn_colours(s, comps)
                    form_ = set(w_) if form_ is None else (form_ & w_)
            self.select_form = set(form_ or ())
        if not self.select_form or not self.select_kinds:
            return self.select_presumed
        for s in comps:
            if s.sid >= 1000 or "hud" in s.tags or "terrain" in s.tags or s.size <= 16:
                continue
            if kind(s) in self.select_kinds or kind(s) in self.select_refused or kind(s) in self.select_form:
                continue
            if worn_colours(s, comps) & self.select_form:
                self.select_presumed.add(kind(s))
        return self.select_presumed

    def complete_selected_keys(self, comps):
        """THE KEYS OF THE SELECTED THING: a key whose record on things of
        the selected colour is one translate is a key that moves whatever is
        selected (the other three arrows, once one has shown the pattern)."""
        worn = self.kinds_worn_by_selected()
        by_sid = {s.sid: s for s in comps}
        # (a selection handed on by a key rather than a click: the things
        # that have worn the mark — ever active under the keys — are the
        # selectable ones, whatever their kinds)
        hosts_ = set()
        if not self.select_kinds and self.select_color is not None and self.cycle_keys:
            hosts_ = {sid_ for sid_ in self.ever_active if sid_ < 1000}
        if (not worn or not self.select_kinds) and not hosts_:
            return
        for k in list(self.fixed.keys()):
            if isinstance(k, tuple) or k in self.key_selected or k in self.cycle_keys:
                continue
            vs = {e[1] for sid, e in self.fixed.moving(k).items()
                  if e[0] == "translate" and sid in by_sid and (kind(by_sid[sid]) in worn or sid in self.select_by_colour or sid in hosts_)}
            if len(vs) == 1:
                self.key_selected[k] = next(iter(vs))

    def selection_in_play(self):
        """Whether the keys are known to move a CHOSEN thing: a key law by
        selection means nothing until some way of choosing is known (a
        click that selects, a mark's colour, a key that hands the mark on)
        — two things moved by one law, one of them held by a wall, are not
        a selection."""
        return bool(self.key_selected) and bool(self.select_kinds or self.select_color is not None or self.cycle_keys or self.select_presumed)

    def selectable_kinds(self):
        return set(self.select_kinds) | (set(self.select_presumed) - set(self.select_refused))

    def refute_presumed(self, kd):
        """A click on a thing of a presumed kind that did not select it."""
        if kd in self.select_presumed:
            self.select_presumed.discard(kd)
            self.select_refused.add(kd)
            return True
        return False

    def sel_ctx(self, sid):
        """The selection as a context for laws: the body the selected thing
        belongs to (an axis read in two pieces, both marked, is one
        selection), named by its smallest member.  No context until the
        keys have shown a mark to be a selection (select_color)."""
        # (selection shown by a mark, by the thing's own colour, or by
        # nothing but which thing answers the keys: each is a context)
        if sid is None or (self.select_color is None and self.selected_colour is None and not self.select_kinds):
            return None
        for b in self.bodies():
            if sid in b:
                return min(b)
        return sid

    def selected_now(self, comps):
        """The thing that answers the keys now: the wearer of the selection
        mark, or — when selection is shown by colour — the thing of the
        selected colour."""
        marks = selected_hosts(comps)
        sel = [h for h, col in marks.items() if self.select_color is None or col == self.select_color]
        if sel and self.select_color is None and self.active and not any(h in self.active for h in sel):
            sel = []                    # (the keys move a body that wears no mark: a dotted wall elsewhere is no selection)
        if len(sel) >= 2 and self.active:
            # (TWO WEAR THE MARK: ours is the one the keys move or the one
            # against a body the keys move; a marked thing nobody of ours
            # touches is in another's hands — wa30 L4, the block the helper
            # carried off still "selected", and every law unknown under it)
            by_ = {s.sid: s for s in comps}
            act_bb = [by_[a].bbox for a in self.active if a in by_]
            near_ = [h for h in sel if h in self.active or (h in by_ and any(
                max(by_[h].bbox[0] - b[2], b[0] - by_[h].bbox[2], by_[h].bbox[1] - b[3], b[1] - by_[h].bbox[3]) <= 1 for b in act_bb))]
            if near_:
                sel = near_
        if sel:
            return sel[0]
        if self.selected_colour is not None:
            cands = [s for s in comps if s.sid < 1000 and kind(s) == self.selected_colour and "hud" not in s.tags and "terrain" not in s.tags]
            if cands:
                return max(cands, key=lambda s: s.size).sid
        # SELECTION WITHOUT A LIVERY: where clicks choose which thing the
        # keys move and nothing is drawn to show it, the selected thing is
        # the one the keys last moved (a click on it again lets it go)
        if self.select_kinds and self.key_selected and self.active:
            by_ = {s.sid: s for s in comps}
            act_ = [by_[a] for a in self.active if a in by_ and a < 1000 and "hud" not in by_[a].tags and "terrain" not in by_[a].tags]
            if act_:
                return max(act_, key=lambda s: s.size).sid
        return None

    def complete_directions(self):
        """THE FOURTH DIRECTION: when three keys translate the controlled
        body by unit steps in three of the four directions and a fourth
        key has done nothing yet (it was pressed where that way was
        barred), that key is read as the fourth direction — provisional,
        verified by use (the plan replans at the first surprise).  The
        arrow keys of these games are a set; a set with one silent member
        is still a set."""
        if not self.active:
            return
        seen = {}                               # key -> vector on the active body
        for k in list(self.fixed.eff):
            if isinstance(k, tuple):
                continue
            law = self.fixed.moving(k)
            vs = {e[1] for sid, e in law.items() if sid in self.active and sid < 1000 and e[0] == "translate"}   # (a part's own record — an eye that faces — is not the body's step)
            if len(vs) == 1:
                seen[k] = next(iter(vs))
        if len(seen) != 3:
            return
        vecs = set(seen.values())
        n = max(abs(x) for v in vecs for x in v)
        dirs = {(-n, 0), (n, 0), (0, -n), (0, n)}
        if not vecs <= dirs:
            return
        missing = dirs - vecs
        silent = [k for k in self.fixed.presses if isinstance(k, str) and k not in seen and not self.fixed.moving(k)
                  and k not in self.undo_keys and k in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")]
        if len(missing) == 1 and len(silent) == 1:
            v = next(iter(missing)); k = silent[0]
            for sid in self.active:
                if sid >= 1000:
                    continue
                if sid not in self.fixed.eff[k] or self.fixed.eff[k][sid][0] == "none":
                    self.fixed.eff[k][sid] = ("translate", v)
                    self.fixed.support[k][sid] = 0          # provisional
                    self.fixed.contradicted[k].discard(("refused", sid))
            return k, v

    def bodies(self):
        """Things that never moved apart, as frozensets of sids (a thing
        that moved alone is its own body)."""
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for sid in self.moved_count:
            find(sid)
        for (a, b), n in self.comove.items():
            if n == self.moved_count[a] == self.moved_count[b]:
                parent[find(a)] = find(b)
        # (things bound by a docking — a piece's marks on another's — move
        # as one until a press shows them apart)
        for a, b in getattr(self, "bound", ()):
            parent[find(a)] = find(b)
        groups = defaultdict(set)
        for sid in list(parent):
            groups[find(sid)].add(sid)
        return [frozenset(g) for g in groups.values()]

    def livery(self):
        """The outline (offsets from the body's corner, with colours) that
        moves with the controlled body, seen at least twice; else None."""
        if not self.livery_votes:
            return None
        off, n = self.livery_votes.most_common(1)[0]
        return off if n >= 2 else None

    def observe_path(self, key, frames, body_colour, body_size=None):
        """THE PATH WITHIN A PRESS (Tushar, 2026-10-24: "when I press up the
        game can send me 5 pixels … a Super Mario Bros-like game … the program
        needs to make sense of the feedback").  The engine shows a press as
        a run of frames; the body's place in each is its PATH.  What the
        path says: its first move is the key's own step; a trailing run of
        equal steps along one axis that differ from the first — one cell a
        frame until it rests — is a FALL, and the way of the fall, seen
        twice, is the GRAVITY the planner pays after every move (a slide
        down that every key shares)."""
        try:
            import numpy as _np
            fr = _np.asarray(frames)
            if fr.ndim != 3 or len(fr) < 2 or body_colour is None:
                return
            cents = []
            for g in fr:
                ys, xs = _np.nonzero(g == body_colour)
                if len(ys) == 0 or (body_size is not None and abs(len(ys) - body_size) > max(4, body_size // 2)):
                    cents.append(None); continue
                cents.append((float(ys.mean()), float(xs.mean())))
            steps_ = []
            for a_, b_ in zip(cents, cents[1:]):
                if a_ is None or b_ is None:
                    steps_.append(None); continue
                steps_.append((int(round(b_[0] - a_[0])), int(round(b_[1] - a_[1]))))
            self.paths[key] = steps_
            po_ = getattr(self, "path_own", None)
            if po_ is None:
                po_ = self.path_own = {}
            # the own step: the path's steps before the trailing fall (all of them when there is none)
            n_fall_ = 0
            if len(steps_) >= 2 and steps_[-1] is not None and steps_[-1] != (0, 0) and (steps_[-1][0] == 0) != (steps_[-1][1] == 0):
                for v_ in reversed(steps_):
                    if v_ == steps_[-1]:
                        n_fall_ += 1
                    else:
                        break
                if n_fall_ < 2 or n_fall_ >= len(steps_):
                    n_fall_ = 0
            pre_ = [v_ for v_ in steps_[:len(steps_) - n_fall_] if v_ is not None]
            po_[key] = (sum(v_[0] for v_ in pre_), sum(v_[1] for v_ in pre_))
            # AN ARC: the own steps are not one straight run (a step across
            # both axes, or steps that differ) — the body's way was read frame
            # by frame; no slide and no misreading explains its length
            pa_ = getattr(self, "path_arc", None)
            if pa_ is None:
                pa_ = self.path_arc = {}
            nz_ = [v_ for v_ in pre_ if v_ != (0, 0)]
            pa_[key] = bool(nz_) and (len(set(nz_)) > 1 or any(v_[0] and v_[1] for v_ in nz_))
            # WHAT THE BODY RESTS ON HOLDS IT: under gravity, the colours just
            # beyond the body the way of the fall, in the press's last frame,
            # are a support — the slide stops at them (the ground is terrain
            # to the tracker and refused no press; it has to be learned as
            # what the body comes to rest on)
            try:
                g_ = getattr(self, "gravity", None)
                if g_ is not None and cents[-1] is not None:
                    last_g = fr[-1]
                    ys, xs = _np.nonzero(last_g == body_colour)
                    H_, W_ = last_g.shape
                    beyond_ = set()
                    body_ = set(zip(ys.tolist(), xs.tolist()))
                    for (r, c) in body_:
                        q = (r + g_[0], c + g_[1])
                        if q not in body_ and 0 <= q[0] < H_ and 0 <= q[1] < W_:
                            beyond_.add(int(last_g[q]))
                    bg_ = int(_np.bincount(last_g.ravel()).argmax())
                    for col_ in beyond_:
                        if col_ != bg_:
                            self.support[col_] += 1
            except Exception:
                pass
            if len(steps_) < 2:
                return
            # the trailing run of equal axis-aligned steps
            last_ = steps_[-1]
            if last_ is None or last_ == (0, 0) or (last_[0] == 0) == (last_[1] == 0):
                return
            n_ = 0
            for v_ in reversed(steps_):
                if v_ == last_:
                    n_ += 1
                else:
                    break
            if n_ >= 2 and n_ < len(steps_) and steps_[len(steps_) - n_ - 1] != last_:
                d_ = ((last_[0] > 0) - (last_[0] < 0), (last_[1] > 0) - (last_[1] < 0))
                self.gravity_votes[d_] += 1 if n_ < 3 else 2      # (a fall of three frames or more is unmistakable)
                if self.gravity is None and self.gravity_votes[d_] >= 2:
                    self.gravity = d_
                if _ENV.get("NOTE_DEBUG") or getattr(self, "verbose", False):
                    print(f"   path under {key}: {steps_[:8]}{'…' if len(steps_) > 8 else ''} — a fall of {n_} frames {d_} (gravity votes {dict(self.gravity_votes)})")
        except Exception as ex_:
            if _ENV.get("AGENT_DEBUG"):
                import traceback; traceback.print_exc()

    def slide(self):
        """The direction the controlled thing slides after a move, or None.
        Votes decide — but the last three sights all against it drop the
        belief however many votes it carried (what changed lately outweighs
        what was learned long ago).  A GRAVITY read off the frames (a fall
        within a press, seen twice) is the slide of every key, and stands
        before the votes."""
        if getattr(self, "gravity", None) is not None and not _ENV.get("NO_GRAVITY"):
            return self.gravity
        if not self.slide_votes:
            return None
        d, n = self.slide_votes.most_common(1)[0]
        rec = getattr(self, "slide_recent", [])
        if len(rec) >= 3 and all(e == ("against", d) for e in rec[-3:]):
            return None
        return d if n > self.slide_against.get(d, 0) else None

    def removable(self):
        """Kinds a click takes away (seen more often gone than staying)."""
        return {kd for kd, n in self.click_gone.items() if n > self.click_stays.get(kd, 0)}

    def track_way(self, bg):
        """The colours the mover rides on, when the ground under it was never
        the background: the way is those colours, the background a bank.
        None when the mover has stood on the background (the floor is the
        floor), or nothing was uncovered yet."""
        if _ENV.get("NO_TRACK_WAY"):
            return None
        unc = {c_ for c_, n_ in self.uncovered.items() if n_ > 0}
        if not unc or bg in unc or bg in self.ground_ok:
            return None
        return unc | set(self.ground_ok)

    def blocking(self):
        """Colours of the ground that refused a move and never allowed one.
        (A PUSHABLE KIND IS NO WALL: it refuses the walker only when its own
        way is blocked — the plan shoves it and reads what stands behind;
        tw01, the blocks "blocking" after one push against the pen's side.)"""
        # (AN AGENT IS NO WALL OF ITS KIND: a thing that travels on its own
        # refused the walker where it stood, and will stand elsewhere; where
        # it is now is a solid for this step, its kind is not a wall — tw01,
        # the helper read as blocking after one refusal, every plan through
        # the room lost)
        return {c for c, n in self.ground_refused.items() if c not in self.ground_ok and (c not in self.pushable or _ENV.get("BLOCK_PUSHABLE"))
                and (c not in self.agents or _ENV.get("BLOCK_AGENTS"))}

    def push_run(self, k=None):
        """How far a shove sends a thing at most: the longest slide seen
        (in every direction the same law, until contradicted)."""
        mags = [abs(w[0]) + abs(w[1]) for (k2, _, _), (w, _) in self.push_vec.items() if k is None or k2 == k]
        # (a run that ended against something is no bound: the thing slides
        # until stopped — the frame's width is the only bound known.  Concluded
        # only when EVERY run seen of the kind came to rest against something
        # and one of them was longer than a step; a run that stopped with open
        # ground beyond is a run of its own, and the runs seen are the law)
        keys_ = [key_ for key_ in self.push_vec if k is None or key_[0] == k]
        if keys_ and all(self.push_stopped.get(key_) for key_ in keys_) and mags:
            # (a slide was seen at all: a run longer than the pusher's own step)
            if any(abs(w[0]) + abs(w[1]) > self.push_step.get(key_, 10**6) for key_, (w, _) in self.push_vec.items() if key_ in keys_):
                return 64
        return max(mags) if mags else 0

    def push_law(self, k, d, v):
        """What a push in direction d does to a thing of kind k: the
        displacement seen in that direction, else the magnitude seen in
        another direction turned to d (the same law in every direction
        until contradicted), else it moves with the pusher."""
        got = self.push_vec.get((k, d[0], d[1]))
        if got is not None:
            return got
        for (k2, dr, dc), (w, pm) in self.push_vec.items():
            if k2 == k:
                mag = abs(w[0]) + abs(w[1])
                return ((mag * d[0], mag * d[1]), pm)
        return (v, True)

    # ------------------------------------------------------------ summaries
    def relative_law(self, k):
        """('teleport', reach) | ('pull', cap, reach) | None for a kind."""
        obs = self.rel.get(k, [])
        if len(obs) < 2:
            return None         # once is a coincidence (a transition frame)
        tele = sum(1 for off, v in obs if abs(off[0] - v[0]) + abs(off[1] - v[1]) <= 1.5)
        toward = [(off, v) for off, v in obs if all((o == 0) or (o * d > 0) or d == 0 for o, d in zip(off, v))
                  and off[0] * v[0] + off[1] * v[1] > 0]
        reach = max(abs(off[0]) + abs(off[1]) for off, _ in obs)
        reach_max = reach      # the largest offset known to work; probes extend it
        if tele >= 2 and tele >= len(obs) - tele:
            return ("teleport", reach_max)
        if len(toward) >= 2:
            cap = max(max(abs(v[0]), abs(v[1])) for _, v in toward)
            return ("pull", cap, reach_max)
        return None

    # ---- WHAT CARRIES TO THE NEXT LEVEL: everything known by KIND (a
    # colour), nothing known by identity (a sid is this level's name)
    # -------------------------------------------------------------- clock
    def clock_value(self, s):
        if self.clock_colour is not None:
            return sum(1 for c in s.cells.values() if c == self.clock_colour)
        return s.size

    def clock_thing(self, comps):
        """The clock now: by identity, else by place and colour (a bar that
        empties is read anew when it refills)."""
        if self.clock is None:
            return None
        by = {s.sid: s for s in comps}
        if self.clock in by:
            return by[self.clock]
        if self.clock_box is None:
            return None
        box = self.clock_box
        for s in comps:
            if s.sid >= 1000 or (self.clock_colour is not None and self.clock_colour not in s.colors):
                continue
            b = s.bbox
            if b[0] <= box[2] and box[0] <= b[2] and b[1] <= box[3] and box[1] <= b[3]:
                self.clock = s.sid
                return s
        return None

    def clock_left(self, comps):
        """Presses left before the clock runs out (None: no clock read)."""
        s = self.clock_thing(comps)
        if s is None or not self.clock_rate:
            return None
        return self.clock_value(s) // self.clock_rate

    def export(self, comps, won_hyp=None):
        s_by = {s.sid: s for s in comps}
        c_ = self.clock_thing(comps) if self.clock_rate else None
        kind_of = dict(self.kind_seen)
        kind_of.update({s.sid: kind(s) for s in comps})
        eff_by_kind = defaultdict(dict)           # key -> kind -> effect (well supported, not contradicted)
        best_sup = defaultdict(dict)
        for k in self.fixed.keys():
            if isinstance(k, tuple):
                continue                          # a click is a place, not a law
            for sid, e in self.fixed.moving(k).items():
                if sid in self.counters or sid >= 1000 or sid not in kind_of:
                    continue
                if sid in self.displays:
                    continue                      # a display's change is a consequence, not the key's law
                sup = self.fixed.support[k][sid] + self.fixed.alt[k][sid][e]
                # (A PUSHED THING HAS NO LAW OF THE KEY: a thing that moved under
                # the key only on presses where the mover came against it was
                # shoved — the push record carries, the "law" does not; tw04 L3
                # planned the block walking to the mark by itself)
                if kind_of[sid] in self.pushable and self.pushed_count.get((k, sid), 0) >= self.fixed.support[k][sid]:
                    continue
                # (a law carried in and never contradicted here is carried
                # on: an attempt that did not press the key does not unlearn
                # it — st70, the keys forgotten between two looks)
                carried_ = sid in getattr(self, "seeded", {}).get(k, ()) and self.fixed.alt[k][sid].get(("none", (0, 0)), 0) == 0 and sup < 2
                if carried_:
                    sup = 2
                if sup >= 2 and sup > best_sup[k].get(kind_of[sid], 0):
                    eff_by_kind[k][kind_of[sid]] = e     # the best-supported law of the kind
                    best_sup[k][kind_of[sid]] = sup
        # (a kind whose things answered one key in DIFFERENT ways — two like
        # dots moved in mirror — has no one law to carry: the next level's
        # press will tell each thing's own; the step length is still carried
        # through the keys that agreed)
        for k in list(eff_by_kind):
            by_kind_vs = defaultdict(set)
            for sid, e in self.fixed.moving(k).items():
                if sid in kind_of and sid < 1000 and e[0] == "translate" and self.fixed.support[k][sid] + self.fixed.alt[k][sid][e] >= 2:
                    by_kind_vs[kind_of[sid]].add(tuple(e[1]))
            for kd, vs in by_kind_vs.items():
                if len(vs) >= 2 and kd in eff_by_kind[k]:
                    if _ENV.get("EXPORT_DEBUG"):
                        print(f"   [export] {k} kind {kd}: laws disagree {sorted(vs)} on {[(sid, e, self.fixed.support[k][sid], dict(self.fixed.alt[k][sid])) for sid, e in self.fixed.moving(k).items() if kind_of.get(sid) == kd]}: not carried")
                    del eff_by_kind[k][kd]
        # the selectable FORM: what every selectable thing on the board wore
        # (the colour of its marks) — a mark-wearing piece, whatever its colour
        form_ = None
        for s in comps:
            if s.sid < 1000 and kind(s) in self.select_kinds and "hud" not in s.tags and "terrain" not in s.tags:
                w_ = worn_colours(s, comps)
                if _ENV.get("FORM_DEBUG"):
                    print(f"   [form] S{s.sid} kind {kind(s)} size {s.size} bbox {s.bbox} wears {sorted(w_)}; small near: {[(o.sid, o.body, o.size, o.bbox) for o in comps if o.sid != s.sid and (o.sid >= 1000 or o.size <= 16)]}")
                form_ = set(w_) if form_ is None else (form_ & w_)
        if form_ is None:
            form_ = set(self.select_form)
        slides_ = {(k, kind_of[sid]) for (k, sid) in getattr(self.fixed, "slides", ()) if sid in kind_of and kind_of[sid] in eff_by_kind.get(k, {})}
        arcs_ = {(k, kind_of[sid]) for (k, sid) in getattr(self.fixed, "arcs", ()) if sid in kind_of and kind_of[sid] in eff_by_kind.get(k, {})}
        # (the conditioned readings, by kind — only of keys conditioned on a thing here)
        cond_ = {}
        try:
            for kc_, by_sid in getattr(self.fixed, "cond", {}).items():
                for sc_, recs in by_sid.items():
                    # (the records of a key read two ways on a thing carry whether or
                    # not the condition is read yet — a condition is read from
                    # sightings, and the sightings add up across levels: tw05's
                    # jump, 12 then 20 once each on every level, was contradicted
                    # on each and never carried)
                    if sc_ in kind_of and sc_ < 1000 and len({e_ for _, e_ in self.fixed._cond_recs_all(kc_, sc_)}) >= 2:
                        dst = cond_.setdefault(kc_, {}).setdefault(kind_of[sc_], {})
                        for cnd_, cnt_ in recs.items():
                            for e_, n_ in cnt_.items():
                                dst.setdefault(cnd_, Counter())[e_] += n_
        except Exception:
            cond_ = {}
        if _ENV.get("EXPORT_DEBUG"):
            print(f"   [export] eff {dict(eff_by_kind)}; keys {list(self.fixed.keys())}; per sid {{k: [(sid, e, self.fixed.support[k][sid], dict(self.fixed.alt[k][sid])) for sid, e in self.fixed.moving(k).items()] for k in self.fixed.keys()}}")
        return {"eff": {k: dict(v) for k, v in eff_by_kind.items()}, "slides": slides_, "arcs": arcs_, "drift": getattr(self.fixed, "drift", None), "cond": cond_,
                "key_selected": dict(self.key_selected) if self.selection_in_play() else {}, "key_rotate": dict(self.key_rotate), "select_kinds": set(self.select_kinds), "select_color": self.select_color,
                "select_form": set(form_), "select_refused": set(self.select_refused),
                "pushable": set(self.pushable), "push_vec": dict(self.push_vec), "push_stopped": dict(self.push_stopped), "push_step": dict(self.push_step), "flat": set(self.flat), "ground_ok": set(self.ground_ok),
                "passable": set(self.passable), "slid_over": Counter(self.slid_over), "dock_colour": self.dock_colour, "legend_map": dict(getattr(self, "legend_map", {}) or {}), "legend_effects": dict(getattr(self, "legend_effects", {}) or {}),
                "ground_refused": Counter(self.ground_refused), "wall_kinds": Counter(self.wall_kinds), "undo_keys": set(self.undo_keys), "silent_click_kinds": Counter(self.silent_click_kinds),
                "positional": {k for k in self.positional if not isinstance(k, tuple)},
                "counters": [(frozenset(self.kinds_worn.get(sid, {kind_of[sid]})), v, s_by[sid].bbox, s_by[sid].size) for sid, v in self.counters.items() if sid in kind_of and sid in s_by]
                            + ([(frozenset(self.kinds_worn.get(c_.sid, set())) | set(c_.colors) | {self.clock_colour}, (self.clock_colour, -self.clock_rate), c_.bbox, c_.size)]
                               if c_ is not None and c_.sid not in self.counters else []),
                "refills": dict(self.refills), "drains": dict(self.drains), "clock_full": self.clock_full, "clock_colour": self.clock_colour,
                "click_gone": dict(self.click_gone), "click_stays": dict(self.click_stays), "selfmovers": set(self.selfmovers), "agents": set(self.agents), "rests": {kd_: list(v_) for kd_, v_ in self.rests.items()}, "click_clears": dict(self.click_clears),
                "death_touch": dict(self.death_touch + self.prior_death_touch), "touched_alive": dict(self.touched_alive + self.prior_touched_alive), "death_enter": dict(self.death_enter + self.prior_death_enter), "entered_alive": dict(self.entered_alive + self.prior_entered_alive), "hazard_kinds": sorted(self.hazard_touch() | self.hazard_enter()), "click_marks": dict(self.click_marks), "click_gone_size": dict(self.click_gone_size), "flat_size": dict(self.flat_size), "ground_size": dict(self.ground_size), "click_grows": dict(self.click_grows), "slide_flips": list(self.slide_flips),
                "slide_votes": dict(self.slide_votes), "slide_against": dict(self.slide_against), "gravity": getattr(self, "gravity", None), "gravity_votes": dict(getattr(self, "gravity_votes", {}) or {}), "support": dict(getattr(self, "support", {}) or {}),
                "livery_votes": dict(self.livery_votes),
                "undo_votes": {k: set(v) for k, v in self.undo_votes.items()},
                "thing_laws": self.thing_laws.export(),
                "clock_rate": self.clock_rate, "clock_box": self.clock_box,
                "rel": {k: list(v) for k, v in self.rel.items()}, "centroid_kinds": dict(self.centroid_kinds),
                "click_laws": self.click_laws_by_kind(kind_of),
                "cast": self.cast(kind_of),
                "paints": {k: sorted({kind_of[sid] for sid, _, _ in v if sid in kind_of}) for k, v in self.paints.items()},
                "colour_target": kind_of.get(self.colour_targets.most_common(1)[0][0]) if self.colour_targets else None,
                "display_goal": getattr(self, "display_goal", None),
                # (a trigger kind is a thing arrived on, not the floor under
                # the arrival — terrain, or large — nor the mover's own
                # colours; a tile is walked over AND a trigger, so being
                # ground does not exclude it)
                "triggers": {tk: sorted({(kind_of[d], "hud" in s_by[d].tags, s_by[d].size, s_by[d].bbox) for d, _ in lst if d in kind_of and d in s_by})
                             for tk, lst in self.triggers.items()
                             if tk not in {kind_of.get(a) for a in self.ever_active}
                             and not any(kind(s) == tk and ("terrain" in s.tags or s.size > 64) for s in comps)},
                "won": (won_hyp.rule(), won_hyp.m_color, _won_colour(won_hyp), won_hyp.note, won_hyp.mover_bigger()) if won_hyp is not None else None,
                # THE METHOD CARRIES: a key that bound a kind to the body, the
                # form of what it binds, and the kind of the thing the bound
                # things were taken to — a law by kind, not a place (the next
                # level's pen is where its thing of that kind stands)
                "carry_law": self.carry_law(comps)}

    def read_materials(self, comps):
        """A KIND SEEN ON A MATERIAL PASSES IT: a thing lying within a
        textured thing's box (the pattern goes on under it) stands on the
        material — its kind, and every colour it has worn, pass that
        material (a crate on the hatching the body cannot enter)."""
        for t in comps:
            if "texture" not in t.tags or t.sid >= 1000:
                continue
            tb = t.bbox; tk = kind(t); tc = set(t.cells)
            for c in comps:
                if c.sid >= 1000 or c is t or "texture" in c.tags or "terrain" in c.tags or "hud" in c.tags:
                    continue
                cb = c.bbox
                if tb[0] <= cb[0] and tb[1] <= cb[1] and cb[2] <= tb[2] and cb[3] <= tb[3] and not (set(c.cells) & tc):
                    # (ON the material, not merely within its box: the pattern
                    # goes on beside the thing — texture cells against its box on
                    # two sides at least.  A ring of hatching has everything in
                    # its box; the body in the yard it fences stood "on" the
                    # fence, and every carry was planned through it — wa30 L4)
                    sides_ = 0
                    for (r0, r1, c0, c1) in ((cb[0] - 1, cb[0] - 1, cb[1], cb[3]), (cb[2] + 1, cb[2] + 1, cb[1], cb[3]),
                                             (cb[0], cb[2], cb[1] - 1, cb[1] - 1), (cb[0], cb[2], cb[3] + 1, cb[3] + 1)):
                        if any((r, c) in tc for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
                            sides_ += 1
                    if sides_ < 2 and not _ENV.get("OLD_MATERIAL"):
                        continue
                    for kd in set(self.kinds_worn.get(c.sid, ())) | {kind(c)}:
                        if (kd, tk) not in self.passable:
                            self.passable.add((kd, tk))
                            if _ENV.get("MOVE_DEBUG") or getattr(self, "verbose", False):
                                print(f"   material: a thing of kind {kd} stands on the {tk}-kind material at {tb}: the kind passes it")

    def carry_law(self, comps):
        cr = getattr(self, "carry", None)
        if not cr:
            return None
        law = {k_: cr[k_] for k_ in ("key", "kinds", "dims", "core", "me_kind", "held_colour", "toggle", "hatch_kinds") if k_ in cr}
        law["kinds"] = set(law.get("kinds", ()))
        tk_ = cr.get("target_kind")
        tc_ = cr.get("target_cells")
        if tk_ is None and tc_:
            tc_ = set(tc_)
            for s in comps:
                if s.sid < 1000 and "hud" not in s.tags and kind(s) not in law["kinds"] and tc_ & set(s.cells):
                    tk_ = kind(s)
                    break
        law["target_kind"] = tk_
        return law

    def cast(self, kind_of):
        """THE CAST OF THE ORGANS, BY KIND (Tushar: "reinterpreting the learning
        is the intelligent part"): what the level's mechanisms were made of,
        named so the next level can find its own — a launch button's kind and
        the kinds it sent travelling, the colours a toggle flips between.
        Nothing here is a place."""
        out = {}
        if self.journeys:
            launches = {}; forms = {}
            for k_, jj, _ in self.journeys:
                uk = self.click_under.get(k_) if isinstance(k_, tuple) else k_
                if uk is None:
                    continue
                launches.setdefault(uk, set()).update(kind_of[sid_] for sid_ in jj if sid_ in kind_of)
                fm = self.click_under_form.get(k_)
                if fm is not None:
                    forms[uk] = fm
            if launches:
                out["launch"] = {str(uk): sorted(v) for uk, v in launches.items()}
                out["launch_form"] = {str(uk): list(fm) for uk, fm in forms.items()}   # (box height, box width, cells): the button's shape, kind aside
        if self.flips:
            out["flips"] = {int(a): int(b) for a, b in self.flips.items()}
        if self.leap:
            out["leap"] = {k_: v_ for k_, v_ in self.leap.items() if isinstance(v_, (int, float, str, tuple))}
        return out

    def click_laws_by_kind(self, kind_of):
        """A BUTTON CARRIES BY ITS KIND (Tushar, 2026-10-14: "the levels must be
        won because the learnings are transferred"): a click on a thing of
        kind K that moved things of kind M by v is the law "K sends M by v",
        whatever the button's place on the next level.  Two buttons of one
        kind that disagree leave no law (lp85: the two buttons are of two
        colours; tn36: the strip's slots are of one kind and do one thing)."""
        laws = defaultdict(dict); contra = set()
        for k in self.fixed.keys():
            if not isinstance(k, tuple) or k not in self.click_under or k in self.rel_keys:
                continue
            uk = self.click_under[k]
            for sid, e in self.fixed.moving(k).items():
                if sid >= 1000 or sid in self.counters or sid not in kind_of or kind_of[sid] == uk:
                    continue
                if self.fixed.support[k].get(sid, 0) < 1:
                    continue                      # (seeded and never confirmed: not a lesson)
                mk = kind_of[sid]
                if mk in laws[uk] and laws[uk][mk] != e:
                    contra.add((uk, mk))
                laws[uk][mk] = e
        return {uk: {mk: e for mk, e in d.items() if (uk, mk) not in contra} for uk, d in laws.items()
                if any((uk, mk) not in contra for mk in d)}

    def seed(self, carry, comps):
        """Open a level with what the last one taught: laws by kind
        become provisional records on this level's things of that kind
        (a press confirms or contradicts them — the fiddle need not
        rediscover them)."""
        self.seeded = defaultdict(set)            # key -> sids seeded
        # (the drift carries: a standing push on the body is of the game, not
        # the level — the first press that moves without it contradicts it)
        if carry.get("drift") is not None and getattr(self.fixed, "drift", None) is None:
            self.fixed.drift = tuple(carry["drift"])
        by_kind = defaultdict(list)
        for s in comps:
            if s.sid < 1000 and "hud" not in s.tags and "terrain" not in s.tags:
                by_kind[kind(s)].append(s)
        for k, kinds in carry.get("eff", {}).items():
            for kd, e in kinds.items():
                for s in by_kind.get(kd, []):
                    if s.sid not in self.fixed.eff[k]:
                        self.fixed.eff[k][s.sid] = e
                        self.fixed.support[k][s.sid] = 0      # provisional: the first press that differs replaces it
                        self.seeded[k].add(s.sid)
                        if (k, kd) in carry.get("slides", set()):
                            self.fixed.slides.add((k, s.sid))
                        if (k, kd) in carry.get("arcs", set()):
                            self.fixed.arcs.add((k, s.sid))
        # (the CONDITIONED readings carry by kind: what a key did to a thing
        # of the kind after each press-before and heading — the running jump
        # learned on one level is known on the next; the records are the
        # readings themselves, and a new reading under a known condition
        # that disagrees undoes the conditioning as it would have here)
        try:
            for k, kinds in carry.get("cond", {}).items():
                for kd, recs in kinds.items():
                    for s in by_kind.get(kd, []):
                        if s.sid in self.fixed.cond.get(k, {}):
                            continue
                        for c_, cnt_ in recs.items():
                            for e_, n_ in cnt_.items():
                                self.fixed.cond[k][s.sid][c_][e_] += n_
                        # (a key conditioned on the kind by the records has a law
                        # here — its commonest outcome stands in until the plan asks
                        # the condition; the key would else never be pressed)
                        if s.sid not in self.fixed.eff[k] and self.fixed.conditioned(k, s.sid):
                            tot_ = Counter()
                            for cnt_ in recs.values():
                                for e_, n_ in cnt_.items():
                                    if e_[0] != "none":
                                        tot_[e_] += n_
                            if tot_:
                                self.fixed.eff[k][s.sid] = tot_.most_common(1)[0][0]
                                self.fixed.support[k][s.sid] = 0
                                self.seeded[k].add(s.sid)
        except Exception:
            pass
        # the buttons: every thing of a button's kind is presumed to do what
        # the button did, on this level's things of the kinds it moved — a
        # provisional record, confirmed or replaced by the first click
        for uk, d in carry.get("click_laws", {}).items():
            buttons = [s for s in by_kind.get(uk, []) if s.size <= 400]
            if not buttons or len(buttons) > 12:
                continue
            for b_ in buttons:
                k = ("click", click_point(b_))
                for mk, e in d.items():
                    for s in by_kind.get(mk, []):
                        if s.sid not in self.fixed.eff[k]:
                            self.fixed.eff[k][s.sid] = e
                            self.fixed.support[k][s.sid] = 0
                            self.seeded[k].add(s.sid)
                self.click_under[k] = uk
        self.key_selected.update(carry.get("key_selected", {}))
        # (a key that turned the selected body turns whatever is selected:
        # the belief is about the key and the selection, not a kind)
        for k_, t_ in carry.get("key_rotate", {}).items():
            self.key_rotate.setdefault(k_, t_)
        self.select_kinds |= carry.get("select_kinds", set())
        self.select_form = set(carry.get("select_form", set()))
        self.select_refused |= carry.get("select_refused", set())
        self.presume_selectable(comps)
        if self.select_color is None:
            self.select_color = carry.get("select_color")
        self.pushable |= carry.get("pushable", set()); self.push_vec.update(carry.get("push_vec", {})); self.push_stopped.update(carry.get("push_stopped", {})); self.push_step.update(carry.get("push_step", {}))
        self.passable |= carry.get("passable", set())
        self.slid_over.update(carry.get("slid_over", {}))
        if self.dock_colour is None:
            self.dock_colour = carry.get("dock_colour")
        self.flat |= carry.get("flat", set()); self.ground_ok |= carry.get("ground_ok", set())
        for kd, sz_ in carry.get("flat_size", {}).items():
            self.flat_size[kd] = max(self.flat_size.get(kd, 0), sz_)
        for kd, sz_ in carry.get("ground_size", {}).items():
            self.ground_size[kd] = max(self.ground_size.get(kd, 0), sz_)
        self.ground_refused.update(carry.get("ground_refused", {})); self.wall_kinds.update(carry.get("wall_kinds", {}))
        self.silent_click_kinds.update(carry.get("silent_click_kinds", {}) or {})
        self.undo_keys |= carry.get("undo_keys", set()); self.positional |= carry.get("positional", set())
        for k_, v_ in carry.get("undo_votes", {}).items():
            self.undo_votes[k_] |= set(v_)          # (a key seen once to take a press back: the next level may show it twice)
        self.thing_laws.seed(carry.get("thing_laws"))
        # the clock is where it was: a thing of the counter's kind whose box
        # overlaps the old one's and whose size is alike
        for worn, v, box, size in carry.get("counters", []):
            for s in comps:
                if s.sid < 1000 and kind(s) in worn and 0.5 * size <= s.size <= 2 * size:
                    b = s.bbox
                    if b[0] <= box[2] and box[0] <= b[2] and b[1] <= box[3] and box[1] <= b[3]:
                        self.counters[s.sid] = v
        if carry.get("clock_rate"):
            self.clock_colour = carry.get("clock_colour"); self.clock_rate = carry["clock_rate"]
            self.clock_rate_carried = True
            self.clock_box = tuple(carry["clock_box"]) if carry.get("clock_box") else None
            self.clock_full = carry.get("clock_full", 0)
            self.clock = -1                     # bound to a thing by place at the first look
            c_ = self.clock_thing(comps)
            if c_ is None:
                self.clock = None
        for kd, n_ in carry.get("refills", {}).items():
            self.refills[kd] += n_
        for kd, n_ in carry.get("drains", {}).items():
            self.drains[kd] += n_
        for kd, n_ in carry.get("click_gone", {}).items():
            self.click_gone[kd] += n_
        for kd, n_ in carry.get("click_stays", {}).items():
            self.click_stays[kd] += n_
        self.click_clears.update(carry.get("click_clears", {}))
        # WHAT THIS LEVEL SHOWS OUTRANKS WHAT WAS CARRIED (twist suite tw02:
        # a floor colour stood on safely for two levels drained the budget
        # on the third; the carried "entered alive" counts outvoted the
        # death and the core walked onto it again): carried counts are
        # priors, consulted only for kinds this level has not yet judged
        for kd, n_ in carry.get("death_enter", {}).items():
            self.prior_death_enter[kd] += n_
        for kd, n_ in carry.get("entered_alive", {}).items():
            self.prior_entered_alive[kd] += n_
        for kd, n_ in carry.get("death_touch", {}).items():
            self.prior_death_touch[kd] += n_
        for kd, n_ in carry.get("touched_alive", {}).items():
            self.prior_touched_alive[kd] += n_
        for kd in carry.get("hazard_kinds", ()):        # a verdict reached earlier stands as a prior
            self.prior_death_touch[kd] = max(self.prior_death_touch[kd], self.prior_touched_alive[kd] + 1) if kd in carry.get("death_touch", {}) else self.prior_death_touch[kd]
            self.prior_death_enter[kd] = max(self.prior_death_enter[kd], self.prior_entered_alive[kd] + 1) if kd in carry.get("death_enter", {}) else self.prior_death_enter[kd]
        for kd, sz_ in carry.get("click_gone_size", {}).items():
            self.click_gone_size[kd] = min(self.click_gone_size.get(kd, 10 ** 6), sz_)
        self.click_grows.update(carry.get("click_grows", {}))
        self.slide_flips = list(carry.get("slide_flips", []))
        for kd, n_ in carry.get("click_marks", {}).items():
            self.click_marks[kd] += n_
        self.selfmovers |= set(carry.get("selfmovers", set()))
        self.agents |= set(carry.get("agents", set()))
        for d_, n_ in carry.get("slide_votes", {}).items():
            self.slide_votes[d_] += n_
        # (the gravity carries: a fall is of the game, not the level)
        for d_, n_ in (carry.get("gravity_votes") or {}).items():
            self.gravity_votes[d_] += n_
        if carry.get("gravity") is not None:
            self.gravity = tuple(carry["gravity"])
        for c_, n_ in (carry.get("support") or {}).items():
            self.support[c_] += n_
        for d_, n_ in carry.get("slide_against", {}).items():
            self.slide_against[d_] += n_
        for o_, n_ in carry.get("livery_votes", {}).items():
            self.livery_votes[o_] += n_
        self.centroid_kinds.update(carry.get("centroid_kinds", {}) or {})
        for kd, obs_ in carry.get("rel", {}).items():
            self.rel[kd].extend(obs_)
        for k, kinds in carry.get("paints", {}).items():
            for kd in kinds:
                big = [s for s in by_kind.get(kd, []) if s.size >= 16]
                if big:
                    c = max(big, key=lambda s: s.size)
                    self.paints[k].append((c.sid, frozenset(), kd))
        # a display is seeded only on a thing like the one that was one: the
        # same kind, in the hud or not as it was, of a like size
        # (a display is a panel: it is where it was, of the kind and size
        # it had — the controlled thing of the panel's colour is not it)
        for tk, dsigs in carry.get("triggers", {}).items():
            for dk, was_hud, dsize, dbox in dsigs:
                if _ENV.get("SEED_DEBUG"):
                    print(f"   seed display {dk} hud={was_hud} size={dsize} box={dbox}: near {[(s.sid, kind(s), s.size, s.bbox, sorted(s.tags)) for s in comps if s.bbox[0] <= dbox[2] and dbox[0] <= s.bbox[2] and s.bbox[1] <= dbox[3] and dbox[1] <= s.bbox[3]]}")
                # (its colour may follow the level's — the gate glyph and
                # its picture are orange on the third level of ls20)
                moved_kinds_ = {kd_ for kinds_ in carry.get("eff", {}).values() for kd_ in kinds_}
                for s in comps:
                    b = s.bbox
                    if s.sid < 1000 and "terrain" not in s.tags and ("hud" in s.tags) == was_hud and 0.5 * dsize <= s.size <= 2 * dsize \
                            and b[0] <= dbox[2] and dbox[0] <= b[2] and b[1] <= dbox[3] and dbox[1] <= b[3] \
                            and (kind(s) not in moved_kinds_ or not _claims.on("moved_kind_is_no_panel")):        # (a thing of a kind the keys moved is no panel)
                        self.triggers[tk].append((s.sid, None))
                        self.displays[s.sid] += 1
        if carry.get("display_goal"):
            self.display_goal = carry["display_goal"]
        # THE METHOD CARRIES: things of the bound kind's form stand here too
        # — the carry law is held (nothing held yet); the protocol re-tests
        # it at the first bind (a thing that does not come along unlearns it)
        law_ = carry.get("carry_law")
        if law_ and getattr(self, "carry", None) is None:
            dims_ = tuple(law_.get("dims") or ())
            here_ = [s for s in comps if s.sid < 1000 and "hud" not in s.tags and "terrain" not in s.tags and kind(s) in law_.get("kinds", ())
                     and (not dims_ or (s.bbox[2] - s.bbox[0] + 1, s.bbox[3] - s.bbox[1] + 1) == dims_)]
            if here_:
                self.carry = {**law_, "kinds": set(law_.get("kinds", ())), "held": None, "offset": None, "seeded": True}
        self.carried = carry

    def describe(self):
        out = {"fixed_keys": len(self.fixed.keys()), "key_selected": dict(self.key_selected),
               "select_kinds": sorted(self.select_kinds), "select_color": self.select_color,
               "select_form": sorted(self.select_form), "select_presumed": sorted(self.select_presumed), "select_refused": sorted(self.select_refused),
               "passable": sorted(self.passable), "body_moves": {k: [(sorted(b), v) for b, v in bm] for k, bm in self.body_moves.items()},
               "cycle_keys": {k: v[:3] for k, v in self.cycle_keys.items()},
               "bodies": [sorted(b) for b in self.bodies()], "active": sorted(self.active) if self.active else None,
               "cycle_bodies": {k: [(sorted(a), sorted(b)) for a, b in v[:3]] for k, v in self.cycle_bodies.items()},
               "pushable": sorted(self.pushable), "blocking": sorted(self.blocking()), "removable": sorted(self.removable()), "click_clears": dict(self.click_clears), "click_grows": dict(self.click_grows), "slide_flips": list(self.slide_flips), "selfmovers": sorted(self.selfmovers), "hazards": sorted(self.hazards()), "slide": self.slide(), "livery": sorted(self.livery()) if self.livery() else None, "ground_ok": sorted(self.ground_ok), "ground_refused": dict(self.ground_refused), "flat": sorted(self.flat), "undo": sorted(self.undo_keys), "positional": sorted(map(str, self.positional)),
               "counters": dict(self.counters), "triggers": {k: v[:3] for k, v in self.triggers.items()},
               "relative": {k: self.relative_law(k) for k in self.rel}}
        return out
