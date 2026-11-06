"""THE NEW CORE PLAYS A LEVEL, WITH FIDDLING: the Fiddler's repertoire
teaches a GeneralModel (fixed / selected / relative / contact effects);
the goal comes from the first-frame readings; the plan is a best-first
search over the general successor function; predict-act-verify.

  .venv/bin/python tools/core_play2.py ka59:1 re86:1 su15:1 r11l:1 s5i5:1-2 vc33:1 [-v]
"""
import types, argparse, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import arc_agi
except Exception:                      # (the submission harness has no arcade)
    arc_agi = None
from arcengine import GameAction, GameState
from agent.perception import background_color, segment, detect_playfield
from agent.core.entities import read_frame, read_components, EntityTracker, Sprite
from agent.core.scene import SceneTracker
import os
_ENV = dict(os.environ)      # (the switches are read once: a lookup in os.environ is a Mapping call, and one stood in the planner's successor loop — 20 million calls, 60 s of a 300 s level on bp35 L2)
SCENE = _ENV.get("SCENE", "1") == "1"
LEVEL_SECONDS = float(_ENV.get("LEVEL_SECONDS", "600"))     # thinking time per level (a wall clock; hand-chosen)


def _level_seconds():
    """The thinking time of this turn: what the game loop allotted from the
    game's remaining clock (in the harness), else the constant."""
    return float(getattr(play_level, "seconds", None) or LEVEL_SECONDS)
from agent.core.planner import plan_general, apply as apply_eff, _shift, _bbox, _rot, _sweep
from agent.core.fiddle import Fiddler, GeneralModel, selected_hosts, livery_host, part_host, kind, click_point, beside_points
from agent.core.program import read_journeys, read_panel, read_lattice, correspond, compose, clicks_to_set, slot_form
from agent.core.leap import read_leap, lattice as leap_lattice, plan_leaps
from agent.core.perthread import PerThread
from agent.core import ledger as _ledger
from agent.core.progress import Progress as _Progress
from agent.core import journal as _journal
from agent.core import claims as _claims
from agent import reader as _reader
from agent.goals import goal_scan, GoalHypothesis
from agent.core.relations import read_frame_goals, RelationWeights
from collections import Counter


def with_derived(tr, model):
    """The tracker's things plus the ones the laws between things say
    are there (an image below the frame): appended to tr.prev in place."""
    try:
        laws = getattr(model, "thing_laws", None)
        if laws is None or not laws.laws():
            return tr.prev
        base = [s for s in tr.prev if "derived" not in s.tags]
        tr.prev = base + laws.derive(base)
    except Exception:
        pass
    return tr.prev


def note(text):
    """What the core is doing now, for the replay (one line per stage)."""
    play_level.phase = text
    if _ENV.get("NOTE_DEBUG"):
        print(f"   note: {text}")


def _seen(tr, model):
    """WHAT THE CORE SAW at this moment, for the replay: every thing the
    tracker holds (its name, colour, size, box, tags) with the ROLE the
    model gives it right now, the model's standing beliefs, and the
    remembered board with the camera on it."""
    try:
        comps = tr.prev
        active = set(model.active or ())
        haz = model.hazards() if hasattr(model, "hazards") else set()
        rem = model.removable() if hasattr(model, "removable") else set()
        grows = getattr(model, "click_grows", {}) or {}
        things = []
        for x in comps:
            if x.sid >= 1000:
                continue
            kd = x.body
            if x.sid in active:
                role = "body"
            elif "hud" in x.tags or "bar" in x.tags:
                role = "hud"
            elif "hidden" in x.tags:
                role = "hidden"
            elif "terrain" in x.tags:
                role = "terrain"
            elif kd in haz:
                role = "hazard"
            elif kd in getattr(model, "selfmovers", set()):
                role = "clockwork"
            elif kd in grows and x.size <= 2 * max(grows[kd][0], 5):
                role = "mark"
            elif kd in rem and (getattr(model, "click_gone_size", {}).get(kd) is None or x.size >= 0.5 * model.click_gone_size[kd]):
                role = "removable"
            elif kd in getattr(model, "pushable", set()):
                role = "pushable"
            elif hasattr(model, "is_flat") and model.is_flat(kd, x.size):
                role = "flat"
            elif hasattr(model, "is_ground") and model.is_ground(kd, x.size):
                role = "ground"
            elif kd in model.blocking():
                role = "wall"
            else:
                role = "thing"
            things.append({"sid": int(x.sid), "c": int(kd), "n": int(x.size), "bb": [int(v) for v in x.bbox], "role": role,
                           "tags": sorted(t for t in x.tags if t not in ("terrain", "hud"))})
        beliefs = {"camera": [int(v) for v in getattr(model, "camera", (0, 0))], "slide": model.slide() if hasattr(model, "slide") else None,
                   "hazards": sorted(int(k) for k in haz), "removable": sorted(int(k) for k in rem), "marks": sorted(int(k) for k in grows),
                   "ground": sorted(int(k) for k in model.ground_ok), "walls": sorted(int(k) for k in model.blocking()),
                   "flat": sorted(int(k) for k in model.flat), "stops": len(getattr(model, "stops", ())), "refused": len(model.refused_at),
                   "keys": {k: {str(sid): [e[0], [int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v) for v in e[1]] if isinstance(e[1], tuple) else str(e[1])] for sid, e in model.fixed.moving(k).items()}
                            for k in model.fixed.keys() if isinstance(k, str)}}
        world = None
        w = getattr(model, "world", None)
        if w:
            rs = [r for r, _ in w]; cs = [c for _, c in w]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if (r1 - r0 + 1) * (c1 - c0 + 1) <= 400 * 400:
                rows = ["".join("0123456789abcdef"[w[(r, c)] & 15] if (r, c) in w else "." for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1)]
                world = {"r0": int(r0), "c0": int(c0), "rows": rows}
        return {"things": things, "beliefs": beliefs, "world": world}
    except Exception as ex:
        return {"things": [], "beliefs": {}, "world": None, "error": str(ex)}


def _hex(frame):
    return "".join("0123456789abcdef"[int(v) & 15] for v in np.asarray(frame).ravel())


def _marks_of(cells):
    """A part read as one thing may be several marks (two stubs of a piece
    read as one part): its 4-connected pieces, each a mark."""
    left = set(cells); out = []
    while left:
        q0 = left.pop(); comp = {q0}; fr = [q0]
        while fr:
            r, c = fr.pop()
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in left:
                    left.discard(q); comp.add(q); fr.append(q)
        out.append(frozenset(comp))
    return out


def centre(cells):
    r0, c0, r1, c1 = _bbox(cells)
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _shift_hyp(h, u):
    """A reading made on an earlier screen, moved with the screen."""
    def sh(x):
        if isinstance(x, tuple) and len(x) == 2 and all(isinstance(t, (int, float)) for t in x):
            return (x[0] + u[0], x[1] + u[1])
        if isinstance(x, (frozenset, set)):
            return frozenset((r + u[0], c + u[1]) for (r, c) in x)
        return x
    for a in ("target", "t_cells", "m_start", "m_cells"):
        if hasattr(h, a):
            try:
                setattr(h, a, sh(getattr(h, a)))
            except Exception:
                pass


def off_centre(a, b, H=64, W=64):
    """The thing is not where predicted: its centre is off by more than
    a cell (a flash on an edge, a mark's colour, are not divergence).
    What the prediction put beyond the screen's edge is not seen: the
    prediction is compared as it would show."""
    b_ = [q for q in b if 0 <= q[0] < H and 0 <= q[1] < W]
    if not b_:
        return True
    ca, cb = centre(a), centre(b_)
    return abs(ca[0] - cb[0]) > 1.0 or abs(ca[1] - cb[1]) > 1.0


def make_successors(model, comps, solids, keys, mover_sid, target, H=64, W=64, frame=None, frozen=(), clock=None, keep=(), target_cells=None):
    """Actions the plan may take, from any state.  `frozen`: things the
    plan must not push (the subject of an experiment).  `clock`: plan
    within the time left (a state carries `_clock`, the clock's cells;
    every action spends the rate; arriving on a REFILL fills it) —
    "known": refills seen to work; "prior": also things of the clock's
    own colour (the clock's colour on the floor says refill)."""
    frozen = set(frozen)
    kinds = {s.sid: kind(s) for s in comps}
    # (the band's things — the clock's bar among them — are no part of the
    # planned world: a law on them spends nothing)
    hud_sids_ = {s.sid for s in comps if "hud" in s.tags or "bar" in s.tags}
    if getattr(model, "clock", None) is not None:
        hud_sids_.add(model.clock)              # (the clock's bar, by whatever name the tracker gives it)
    cb_ = getattr(model, "clock_box", None)
    if cb_ is not None:
        for s in comps:
            b_ = s.bbox
            if s.sid < 1000 and b_[0] <= cb_[2] and cb_[0] <= b_[2] and b_[1] <= cb_[3] and cb_[1] <= b_[3] and (getattr(model, "clock_colour", None) is None or model.clock_colour in getattr(s, "colors", (s.body,))):
                hud_sids_.add(s.sid)
    # DERIVED THINGS follow their sources (an image is where the shape and
    # the line put it): not moved by any law, not in anything's way, and
    # recomputed after every action
    derived_ = {s.sid: s.source for s in comps if "derived" in s.tags and hasattr(s, "source")}
    from agent.core.thinglaws import _mirror_cells as _mir, _line_of as _lof
    # HUNG BETWEEN ITS HANDLES: a thing at the centroid of others is moved
    # by moving them — it has no law of its own, is in nothing's way, and
    # is put at the centroid after every action
    cent_ = {d_: hs_ for d_, hs_ in (getattr(model, "centroid_of", {}) or {}).items()
             if d_ in kinds and all(h_ in kinds for h_ in hs_)}
    solids = set(solids) - set(derived_) - set(cent_)

    def _icentre(cells_):
        b_ = _bbox(cells_)
        return ((b_[0] + b_[2]) // 2, (b_[1] + b_[3]) // 2)

    def _hang(new_):
        for d_, hs_ in cent_.items():
            if d_ not in new_ or not all(h_ in new_ for h_ in hs_):
                continue
            cs_ = [_icentre(new_[h_]) for h_ in hs_]
            want_ = (sum(c_[0] for c_ in cs_) // len(cs_), sum(c_[1] for c_ in cs_) // len(cs_))
            have_ = _icentre(new_[d_])
            v_ = (want_[0] - have_[0], want_[1] - have_[1])
            if v_ != (0, 0):
                new_[d_] = frozenset(_shift(new_[d_], v_))
        return new_
    # (the line things are mirrored across is a reference, not a wall: what
    # crosses it has its image on the other side)
    for dsid, (a_sid, L_sid, line0) in derived_.items():
        solids.discard(L_sid)
    for L_sid in {L for (_, L, _) in derived_.values()}:
        for o in list(solids):
            pass
    clock_on = bool(clock) and bool(getattr(model, "clock_rate", 0))
    refill_sids = set()
    if clock_on:
        for s in comps:
            if s.sid >= 1000 or s.sid == model.clock or "hud" in s.tags or "terrain" in s.tags or "bar" in s.tags:
                continue
            if s.sid in keep:
                continue                        # a refill kept for later (the one by the door)
            if kinds[s.sid] in model.refills or (clock == "prior" and kinds[s.sid] == model.clock_colour and s.size <= 32):
                refill_sids.add(s.sid)
        solids = set(solids) - refill_sids
    active_ = set(model.active or ())
    # (before any press of this level has named the body, the plan's mover
    # is the body: the slide and the gravity act on it — tw05 L2, the hop
    # over the gap predicted with no fall because nothing was "active" yet)
    if not active_ and mover_sid is not None and not isinstance(mover_sid, str):
        active_ = {mover_sid}
    # the ground: the frame with the movable things lifted off it
    blocking = model.blocking()
    wall_cells = set()
    if frame is not None and blocking:
        lifted = set()
        # (what moves with the body — a thing bound to it, carried — is
        # not a wall in its way, whatever colour it wears)
        with_body_ = set()
        for b_ in model.bodies():
            if b_ & set(model.active or ()):
                with_body_ |= set(b_)
        wb_boxes_ = [s.bbox for s in comps if s.sid in with_body_]
        for s in comps:
            # (a patch of a wall colour is wall however small it is read)
            within_body_ = any(hb[0] < s.bbox[0] and hb[1] < s.bbox[1] and s.bbox[2] < hb[2] and s.bbox[3] < hb[3] for hb in wb_boxes_)
            # (a TEXTURE of a wall colour is wall too: a patterned wall read as
            # one material — kind colour+16 — is not lifted off the ground by
            # its kind's absence from the blocking colours; m0r0 L2 died with
            # its walls lifted, the day the materials were read first)
            if "terrain" not in s.tags and "hud" not in s.tags and ((kinds.get(s.sid) not in blocking and not ("texture" in s.tags and s.body in blocking))
                                                                      or s.sid in with_body_ or within_body_
                                                                      or (s.sid >= 1000 and s.sid // 1000 in with_body_)):
                lifted |= set(s.cells)
        wall_cells = {(r, c) for r in range(frame.shape[0]) for c in range(frame.shape[1])
                      if int(frame[r, c]) in blocking and (r, c) not in lifted}
    # THE MOVER WALKS ON WHAT IT WAS SEEN ON: riding a track, the background
    # is a bank to it (fiddle.track_way)
    try:
        if hasattr(model, "track_way"):
            fr_t_ = np.asarray(frame) if frame is not None else getattr(model, "_frame_now", None)
            if fr_t_ is not None:
                bg_t_ = background_color(np.asarray(fr_t_))
                tw_ = model.track_way(bg_t_)
                if _ENV.get("TRACK_DEBUG"):
                    print(f"      [track] way {tw_} bg {bg_t_} frame {'given' if frame is not None else 'the model\'s'} walls before {len(wall_cells)}")
                if tw_ is not None:
                    fr_t_ = np.asarray(fr_t_)
                    wall_cells = set(wall_cells) | {(int(r), int(c)) for r, c in zip(*np.nonzero(fr_t_ == bg_t_))}
    except Exception as ex_:
        if _ENV.get("TRACK_DEBUG"):
            print(f"      [track] failed: {ex_!r}")
    # (a patch of a walkable colour that refused the body is a wall here)
    wp_ = getattr(model, "wall_patches", None)
    if frame is not None and wp_:
        wall_cells = set(wall_cells) | {q for q in wp_ if 0 <= q[0] < frame.shape[0] and 0 <= q[1] < frame.shape[1] and int(frame[q]) != int(np.bincount(np.asarray(frame).ravel()).argmax())}
    # A SHOVED THING STOPS AT GROUND IT WAS NEVER SEEN TO CROSS: the body
    # learns a wall by bumping it (one press, nothing lost); a thing pushed
    # into unknown terrain may be lost there for the level (tw04 L3: the
    # block pushed to the border, the mark then out of reach).  Terrain of a
    # colour no shove has slid over and the body has not walked on refuses
    # a pushed thing in the plan until a press says otherwise.
    shove_walls = set()
    if frame is not None:
        known_ = set(model.ground_ok) | set(getattr(model, "slid_over", {}) or {}) | set(getattr(model, "flat", ())) | set(getattr(model, "_shove_ok", ()) or ())
        for s in comps:
            if "terrain" in s.tags and s.body not in known_ and s.size > 2:
                shove_walls |= set(s.cells)
    # THE CURSOR'S BOX: a mover standing inside a box — a thing of another
    # colour, small against the frame, whose outline surrounds it on every
    # side with room between — moves within that box; what lies outside its
    # outline is a wall to it until a press says otherwise.  (dc22: the
    # selector in its 3x3 box, planned sixteen steps out of it; the second
    # step was refused "with nothing in the way")
    box_walls = set()
    if frame is not None and mover_sid is not None and not _ENV.get("NO_BOX_BOUND"):
        mv_ = next((s for s in comps if s.sid == mover_sid), None)
        if mv_ is not None:
            mb_ = mv_.bbox
            H_, W_ = frame.shape
            boxes_ = [s for s in comps if s.sid != mover_sid and s.sid < 1000 and "hud" not in s.tags and "terrain" not in s.tags
                      and s.body != mv_.body and s.size >= 8
                      and s.bbox[0] < mb_[0] and s.bbox[1] < mb_[1] and s.bbox[2] > mb_[2] and s.bbox[3] > mb_[3]
                      and (s.bbox[2] - s.bbox[0] + 1) * (s.bbox[3] - s.bbox[1] + 1) <= H_ * W_ // 4
                      and len(s.cavity()) >= mv_.size
                      # (a hole the mover fills exactly is the mover's own shadow on
                      # what it lies on — su15's block on its disc — not a box drawn
                      # round it: a box has room beyond the mover)
                      and len(s.cavity() - set(mv_.cells)) >= 1]
            # (A PART THE KEYS MOVE MOVES WITHIN ITS THING: the mover read as a
            # part of a host — the selector drawn on its panel — has the host's
            # cells for its ground and nothing beyond them)
            host_ = next((s for s in comps if mover_sid >= 1000 and s.sid == mover_sid // 1000), None)
            # (...a host that itself moves under some key carries its part with
            # it — s5i5's marks ride their pieces; only a host no key moves is
            # the part's whole ground)
            host_moves_ = host_ is not None and any(model.fixed.moving(k_).get(host_.sid, ("none",))[0] != "none" for k_ in model.fixed.keys())
            if host_ is not None and host_.size >= 2 * mv_.size and not host_moves_:
                dom_ = set(host_.cells) | set(mv_.cells)
                box_walls = {(r, c) for r in range(H_) for c in range(W_) if (r, c) not in dom_}
                if _ENV.get("WALL_DEBUG"):
                    print(f"      [walls] the mover S{mover_sid} is a part of S{host_.sid} {host_.bbox}: its ground is the host")
            elif boxes_:
                bx_ = min(boxes_, key=lambda s: (s.bbox[2] - s.bbox[0] + 1) * (s.bbox[3] - s.bbox[1] + 1))
                bb_ = bx_.bbox
                box_walls = {(r, c) for r in range(H_) for c in range(W_) if not (bb_[0] < r < bb_[2] and bb_[1] < c < bb_[3])}
                if _ENV.get("WALL_DEBUG"):
                    print(f"      [walls] the mover S{mover_sid} stands in the box S{bx_.sid} {bb_}: outside it is wall")
    # (the box bounds the MOVER — a shove's walls are for the shoved)
    if wall_cells:
        if _ENV.get("WALL_DEBUG"):
            print(f"      [walls] blocking {blocking} raw {sum(1 for r in range(frame.shape[0]) for c in range(frame.shape[1]) if int(frame[r, c]) in blocking)} lifted {len(lifted)} -> {len(wall_cells)}; lifted things {[(s.sid, s.body, s.size, kinds.get(s.sid), sorted(s.tags)) for s in comps if set(s.cells) <= lifted and s.size > 20]}")
        # THE TARGET'S OWN SURROUND: what wears the wall's colour within a
        # cell of the thing aimed at is its outline, not a wall (the way
        # to it is what the plan is for)
        if target_cells:
            tb_ = _bbox(target_cells)
            wall_cells -= {(r, c) for r in range(tb_[0] - 1, tb_[2] + 2) for c in range(tb_[1] - 1, tb_[3] + 2)}
        # (the body's own outline in the wall's colour is not a wall)
        liv_cells_ = set()
        liv_ = model.livery() if hasattr(model, "livery") else None
        body_cells_ = set()
        if model.active:
            for s in comps:
                if s.sid in model.active:
                    body_cells_ |= set(s.cells)
        if liv_ and body_cells_:
            mb_ = _bbox(body_cells_)
            liv_cells_ = {(mb_[0] + dr_, mb_[1] + dc_) for (dr_, dc_, _c) in liv_}
            wall_cells -= liv_cells_
        # AN EDGE IS NOT A WALL: a lone line of the wall's colour, one cell
        # thin, drawn between two things or grounds (a tile's black edge)
        # is a drawing, not a wall — what stands on either side decides
        # (the body and its outline count as open: they stand on ground)
        edge_ = set()
        Hf, Wf = frame.shape
        wall0_ = frozenset(wall_cells)           # (a wall is a wall whatever put it there: the track's bank as much as a blocking colour)
        for (r, c) in wall_cells:
            up_ = (r - 1, c); dn_ = (r + 1, c); lf_ = (r, c - 1); rt_ = (r, c + 1)
            def open_(q):
                return 0 <= q[0] < Hf and 0 <= q[1] < Wf and ((int(frame[q[0], q[1]]) not in blocking and q not in wall0_) or q in liv_cells_ or q in body_cells_)
            if (open_(up_) and open_(dn_)) or (open_(lf_) and open_(rt_)):
                edge_.add((r, c))
        wall_cells -= edge_
        # (a line the body has rested against is a wall, drawing or not)
        cam_ = getattr(model, "camera", (0, 0))
        for (r, c) in getattr(model, "stops", ()):
            q_ = (r - cam_[0], c - cam_[1])
            if 0 <= q_[0] < Hf and 0 <= q_[1] < Wf and int(frame[q_[0], q_[1]]) in blocking and q_ not in liv_cells_:
                wall_cells.add(q_)
        if _ENV.get("EXPLORE_DEBUG") and body_cells_:
            mb_ = _bbox(body_cells_)
            print(f"      [succ] body bbox {mb_} members {sorted((s.sid, s.body, s.bbox) for s in comps if s.sid in model.active)} livery {sorted(liv_cells_)[:20]} walls near body {sorted((r, c) for (r, c) in wall_cells if mb_[0] - 2 <= r <= mb_[2] + 2 and mb_[1] - 8 <= c <= mb_[3] + 8)} blocking {blocking}")
            for q_ in [(mb_[0] - 2, mb_[1]), (mb_[0] - 2, mb_[1] + 6), (mb_[0] - 1, mb_[1]), (mb_[0] - 1, mb_[1] + 6)]:
                own_ = [(s.sid, s.body, s.size, sorted(s.tags), s.sid in solids, kinds.get(s.sid)) for s in comps if q_ in s.cells]
                print(f"      [succ] cell {q_} frame {int(frame[q_[0], q_[1]]) if 0 <= q_[0] < frame.shape[0] else None} owners {own_} wall {q_ in wall_cells} edge {q_ in edge_}")
    hosts_by_kind = defaultdict(list)
    for s in comps:
        hosts_by_kind[kinds[s.sid]].append(s.sid)
    fixed = model.fixed
    has_cond_ = bool(getattr(fixed, "has_cond", lambda: False)()) and not _ENV.get("NO_COND")
    # FLAT IS OF A FILL, NOT OF A TEXTURE: what the body lay over was a solid
    # fill of the colour; a TEXTURE of that colour (a pattern strewn over the
    # ground — a hatched fence drawn in the pens' floor colour, wa30) is
    # another material, walked only when a thing of the mover's kind has
    # been seen within one (passable).  (wa30 L4: every carry was planned
    # through the fence and refused at it, one place at a time.)
    texture_sids_ = {s.sid for s in comps if "texture" in s.tags}
    passable_k_ = getattr(model, "passable", set()) or set()
    mover_kind_ = kinds.get(mover_sid)
    tex_cells_ = [(s.sid, kinds.get(s.sid), frozenset(s.cells)) for s in comps if "texture" in s.tags and s.size > 2] if not _ENV.get("TEXTURE_FLAT") else []
    cr_ = getattr(model, "carry", None) or {}
    hatch_kinds_ = set(cr_.get("hatch_kinds", ()) or ())
    carry_kinds_ = set(cr_.get("kinds", ()) or ()) | ({cr_.get("held_colour")} if cr_.get("held_colour") is not None else set()) | ({cr_["core"][0]} if cr_.get("core") else set())

    def flat_here_(o, ocs):
        if not model.is_flat(kinds.get(o), len(ocs)):
            return False
        if o in texture_sids_ and not _ENV.get("TEXTURE_FLAT") and (mover_kind_, kinds.get(o)) not in passable_k_:
            return False
        return True
    if has_cond_ and _ENV.get("COND_DEBUG"):
        print(f"   [cond] the plan carries its condition: {[(k_, s_, fixed._cond_map(k_, s_)) for k_ in fixed.cond for s_ in fixed.cond[k_] if s_ in fixed.eff.get(k_, {}) and fixed.conditioned(k_, s_)]}"[:600])
    if hasattr(model, "presume_selectable"):
        model.presume_selectable(comps)          # (a piece shows its rest colour only once another is selected)
        model.complete_selected_keys(comps)
    sel_kinds = model.selectable_kinds() if hasattr(model, "selectable_kinds") else model.select_kinds
    rel_law = model.relative_law(kinds.get(mover_sid))
    bodies_ = model.bodies()

    refused_ = model.refused_now(frame) if hasattr(model, "refused_now") else set(model.refused_at)
    # (BEYOND A FLOOR IS FLOOR: a move refused from a place along one axis
    # is refused from every place further that way on the same line — the
    # arrow on a pumped column that cannot sink below its floor cannot sink
    # from lower still; the plan does not route round a limit by going
    # past it)
    # (a FLOOR: the refusal met nothing — only the ground or a terrain stood
    # where the move would have gone; a refusal by a thing in the way says
    # nothing of the places beyond it — tw02 L1, the block refused by its
    # neighbour was kept from its socket)
    refused_lines_ = defaultdict(list)
    floor_cols_ = {s_.body for s_ in comps if "terrain" in s_.tags}
    if frame is not None:
        try:
            floor_cols_.add(background_color(np.asarray(frame)))
        except Exception:
            pass
    seen_ = getattr(model, "refused_seen", {}) or {}
    # (WITHDRAWN: the walls are terrain too, and a refusal by a wall at the
    # top of a column kept m0r0 L2's mover from every place below it; the
    # statement stands only as a switch, FLOOR_EXTEND, until a floor can be
    # told from a wall)
    for (kd_, r0_, c0_, v_) in (refused_ if _ENV.get("FLOOR_EXTEND") else ()):
        sn_ = seen_.get((kd_, r0_, c0_, v_))
        if sn_ is None or any(col_ not in floor_cols_ for _, col_ in sn_):
            continue
        if v_[0] and not v_[1]:
            refused_lines_[(kd_, "col", c0_, 1 if v_[0] > 0 else -1)].append(r0_)
        elif v_[1] and not v_[0]:
            refused_lines_[(kd_, "row", r0_, 1 if v_[1] > 0 else -1)].append(c0_)

    # A REFUSAL WITH NOTHING IN THE WAY IS A BOUND: a move refused where only
    # the background lay ahead, at the same coordinate along the move's axis
    # from TWO different places across it, is the edge of the mover's range —
    # refused from every place on that line, and beyond it.  (sp80: the bar
    # goes no higher than one row; the plan went up from every column in
    # turn, twenty-three refusals, and the law was dropped.)  Not the
    # floor-extend of 37z: only the background, never a wall or a terrain.
    bound_lines_ = defaultdict(set)
    try:
        bg_b_ = background_color(np.asarray(frame)) if frame is not None else None
    except Exception:
        bg_b_ = None
    if bg_b_ is not None and not _ENV.get("NO_BOUNDS"):
        for (kd_, r0_, c0_, v_) in refused_:
            sn_ = seen_.get((kd_, r0_, c0_, v_))
            if not sn_ or any(col_ != bg_b_ for _, col_ in sn_):
                continue
            if v_[0] and not v_[1]:
                bound_lines_[(kd_, "v", 1 if v_[0] > 0 else -1, r0_)].add(c0_)
            elif v_[1] and not v_[0]:
                bound_lines_[(kd_, "h", 1 if v_[1] > 0 else -1, c0_)].add(r0_)
    bounds_ = {(kd_, ax_, sg_): sorted(co_ for (k2_, a2_, s2_, co_), places_ in bound_lines_.items() if (k2_, a2_, s2_) == (kd_, ax_, sg_) and len(places_) >= 2)
               for (kd_, ax_, sg_, _) in bound_lines_}
    bounds_ = {k_: v_ for k_, v_ in bounds_.items() if v_}

    def is_refused_(kd_, r0_, c0_, v_):
        if (kd_, r0_, c0_, v_) in refused_:
            return True
        if v_[0] and not v_[1]:
            sg_ = 1 if v_[0] > 0 else -1
            if any(r0_ * sg_ >= r_ * sg_ for r_ in bounds_.get((kd_, "v", sg_), ())):
                return True
            return any(r0_ * sg_ >= r_ * sg_ for r_ in refused_lines_.get((kd_, "col", c0_, sg_), ()))
        if v_[1] and not v_[0]:
            sg_ = 1 if v_[1] > 0 else -1
            if any(c0_ * sg_ >= c_ * sg_ for c_ in bounds_.get((kd_, "h", sg_), ())):
                return True
            return any(c0_ * sg_ >= c_ * sg_ for c_ in refused_lines_.get((kd_, "row", r0_, sg_), ()))
        return False

    def slide_shove(new, o, x, v):
        """A SHOVE: thing o, struck along v (by x, or by nothing named),
        slides on a step at a time until something stops it or it has
        gone the run a shove was seen to send a thing — and on, while it
        is over what a shoved thing slides over (a pit the pusher may not
        enter).  Returns (cells where it comes to rest, distance) or None
        when it cannot move at all."""
        unit_v = max(abs(v[0]), abs(v[1])) or 1
        dstep = ((v[0] > 0) - (v[0] < 0), (v[1] > 0) - (v[1] < 0))
        step_v = (dstep[0] * unit_v, dstep[1] * unit_v)
        dv0, _ = model.push_law(kinds.get(o), dstep, v)
        run_ = max(abs(dv0[0]) + abs(dv0[1]), getattr(model, "push_run", lambda k=None: 0)(kinds.get(o)))
        # (what it slides over: colours seen under a shoved thing's way that
        # are not the floor the mover walks on)
        over_ = set(getattr(model, "slid_over", {}) or {}) - set(model.ground_ok)
        cur_ = frozenset(new[o]); gone = 0; nstep = 0
        while True:
            nxt_ = _shift(cur_, step_v)
            if any(r < 0 or c < 0 or r >= H or c >= W for r, c in nxt_):
                break
            if wall_cells and ((nxt_ & wall_cells) - {q for q in nxt_ if frame is not None and int(frame[q[0], q[1]]) in over_}):
                break
            if shove_walls and (nxt_ & shove_walls):
                break
            if any(nxt_ & new[z] for z in new if not isinstance(z, str) and z not in (o, x) and z in solids
                   and kinds.get(z) not in model.pushable and frozenset((o, z)) not in fixed.overlap_ok):
                break
            cur_ = nxt_; gone += unit_v; nstep += 1
            on_over = frame is not None and over_ and any(int(frame[q[0], q[1]]) in over_ for q in cur_ if 0 <= q[0] < H and 0 <= q[1] < W)
            if gone >= run_ and not on_over:
                break
            if nstep > 40:
                break
        if nstep == 0:
            return None
        return cur_, (dstep[0] * gone, dstep[1] * gone)

    def push_move(state, sid, v, changed, body=frozenset(), own=frozenset()):
        """Translate sid by v; things in the way move too when their kind
        is pushable; a wall refuses.  Returns the new state or None."""
        new = dict(state)
        queue = [sid]
        moved = set()
        # (what is bound to the mover — carried — is one body with it: not
        # in its way, and moved along after it)
        bound_ = set()
        for b_ in bodies_:
            if sid in b_:
                bound_ |= {o_ for o_ in b_ if o_ != sid and o_ in new and o_ not in body and not isinstance(o_, str)
                           and o_ not in rider_host             # (a rider goes with its host, not twice)
                           and o_ not in changed}               # (a thing bound to a body of two members goes once, not once per member)
        while queue:
            x = queue.pop()
            if x in moved:
                continue
            if x not in new:
                make_successors.why = ("not in state", x); return None
            if x in derived_:
                moved.add(x); return new          # (it follows its source; recomputed after the press)
            xb = _bbox(new[x])
            if is_refused_(kinds.get(x), xb[0], xb[1], v):
                make_successors.why = ("refused here", x); return None
            land = model.portals.get((kinds.get(x), xb[0], xb[1], cur_key[0])) if getattr(model, "portals", None) else None
            if land is not None and x == sid:
                v = (land[0] - xb[0], land[1] - xb[1])      # from here this press lands elsewhere
            cs = _shift(new[x], v)
            if any(r < 0 or c < 0 or r >= H or c >= W for r, c in cs):
                if x not in active_ and x != mover_sid:
                    # (something else leaving the screen: gone from the state, no refusal)
                    new.pop(x, None)
                    for pid in parts_of.get(x, ()):
                        new.pop(pid, None)
                    moved.add(x)
                    continue
                # THE SCREEN IS A WINDOW: the mover may go partly beyond its
                # edge — its arms out of sight, its middle still in view —
                # until a press refuses it there (which is then remembered
                # by place); what leaves the screen is kept in the state
                # and comes back into view when it returns
                bb_ = _bbox(cs)
                mid_ = ((bb_[0] + bb_[2]) / 2.0, (bb_[1] + bb_[3]) / 2.0)
                in_ = sum(1 for r, c in cs if 0 <= r < H and 0 <= c < W)
                if not (0 <= mid_[0] < H and 0 <= mid_[1] < W and in_ >= 0.5 * len(cs)):
                    make_successors.why = ("bounds", x); return None
            swept = _sweep(new[x], v) if x not in derived_ else cs
            if wall_cells and ((swept & wall_cells) - new.get("_cleared", frozenset())):
                make_successors.why = ("wall", x, sorted(swept & wall_cells)[:3]); return None
            # A MATERIAL REFUSES WHAT WAS NEVER SEEN ON IT: a textured thing
            # (a hatched fence) is a wall to a mover whose kind no sight put
            # within one — and lets through what was (the crates lodged in
            # the hatching: passable).  wa30 L4: the fence, read whole at
            # last, was terrain to the plan and the carry went through it.
            if tex_cells_ and not isinstance(x, str):
                xh_ = x if x < 1000 else x // 1000
                xk_ = kinds.get(xh_)
                for tsid_, tk_, tc_ in tex_cells_:
                    if (xk_, tk_) in passable_k_ or model.is_flat(tk_) or not (swept & tc_):
                        continue
                    # (a material that KILLS is entered, not refused: the plan's
                    # deadly test must see the body on it — m0r0 L2's checker,
                    # a wall to the plan, let one body of two "stay" while the
                    # world walked it in and sent both home)
                    if tk_ in hazard_kinds or (isinstance(tk_, int) and tk_ - 16 in hazard_kinds):
                        continue
                    # (what was seen lodged in the material passes it: the carry's
                    # hatching record, and any overlap the world has shown)
                    if frozenset((xh_, tsid_)) in fixed.overlap_ok or frozenset((x, tsid_)) in fixed.overlap_ok:
                        continue
                    if tk_ in hatch_kinds_ and xk_ in carry_kinds_:
                        continue
                    make_successors.why = ("material", x, tk_); return None
            if shove_walls and x != sid and x not in active_ and x not in bound_ and (swept & shove_walls):
                make_successors.why = ("shoved into unknown ground", x, sorted(swept & shove_walls)[:3]); return None
            if box_walls and x == sid and (swept & box_walls):
                make_successors.why = ("the mover's box", x, sorted(swept & box_walls)[:3]); return None
            new[x] = frozenset(cs)
            move_parts(new, x, v, own)
            moved.add(x)
            if getattr(make_successors, "trace", False):
                print(f"      [move] S{x} by {v} -> {_bbox(cs)} (mover S{sid}; bound {sorted(bound_)}; body {sorted(b_ for b_ in body if not isinstance(b_, str))})")
            # (what is bound to it — a thing carried by the body — goes the
            # same way, walls refusing it as they refuse the body)
            if x == sid:
                for o_ in bound_:
                    if o_ not in moved:
                        queue.append(o_)
            for o, ocs in new.items():
                if isinstance(o, str) or o == x or o in moved or o in changed or o in body or o in bound_ or (o not in solids and not o < 0):
                    continue        # `changed`/`body`: moved by the same press — one body
                if cs & ocs:
                    if flat_here_(o, ocs) or (model.is_ground(kinds.get(o), len(ocs)) and not (o in texture_sids_ and (mover_kind_, kinds.get(o)) not in passable_k_ and not _ENV.get("TEXTURE_FLAT"))):
                        continue            # (ground walked over or through is no obstacle — a texture only when the mover's kind was seen within one)
                    if target_cells and (cs & ocs) <= frozenset(target_cells) and kinds.get(o) not in model.pushable:
                        continue            # (the cells aimed at — the marks to land on — are no obstacle; a pushable thing's cells are the thing, shoved below)
                    # (A MATERIAL IS NOT A THING: a texture is never shoved — wa30 L4's
                    # hatched fence, of a pushable colour, was "pushed" by the plan)
                    if kinds.get(o) in model.pushable and o not in frozen and not (o in texture_sids_ and not _ENV.get("TEXTURE_FLAT")):
                        dv, pusher_moves = model.push_law(kinds[o], ((v[0] > 0) - (v[0] < 0), (v[1] > 0) - (v[1] < 0)), v)
                        if _ENV.get("PUSH_DEBUG"):
                            print(f"      [push] S{x} into S{o} kind {kinds.get(o)} v {v}: law {dv} pusher_moves {pusher_moves} push_vec {dict(model.push_vec)}")
                        if dv == v and pusher_moves:
                            queue.append(o)             # (pushed along: it moves with the pusher)
                        else:
                            # A SHOVE: the other slides on in the push's
                            # direction; the pusher stays where it was when
                            # that is what the record showed
                            try:
                                sh_ = slide_shove(new, o, x, v)
                            except Exception as ex_:
                                if _ENV.get("PUSH_DEBUG"):
                                    import traceback; traceback.print_exc()
                                raise
                            if _ENV.get("PUSH_DEBUG"):
                                print(f"      [push] shove of S{o} from {_bbox(new[o])}: {None if sh_ is None else (_bbox(sh_[0]), sh_[1])} wall_cells {len(wall_cells)} solids {sorted(solids)} over {set(getattr(model, 'slid_over', {}) or {}) - set(model.ground_ok)} blocking {blocking} stopper-in-walls {(40, 12) in wall_cells} terrain {[(s_.sid, s_.body, s_.size) for s_ in comps if 'terrain' in s_.tags]}")
                            if sh_ is None:
                                make_successors.why = ("shove blocked", x, o); return None
                            ocs2, dv = sh_
                            new[o] = frozenset(ocs2); move_parts(new, o, dv); moved.add(o)
                            if not pusher_moves:
                                new[x] = frozenset(state[x]) if x in state else new[x]
                                moved.discard(x); moved.add(o)
                                changed.add(o)
                                return new
                    elif frozenset((x, o)) in fixed.overlap_ok:
                        continue
                    elif ((kinds.get(x), kinds.get(o)) in passable_ or (kinds.get(o), kinds.get(x)) in passable_) and _claims.on("passable_by_kind"):
                        continue            # (kinds seen to stand on each other — a crate on the hatching — do not block each other)
                    else:
                        if _ENV.get("PUSH_DEBUG"):
                            print(f"      [push] S{x} into S{o} kind {kinds.get(o)!r}: solid, not pushable (pushable {sorted(map(str, model.pushable))}) frozen {o in frozen}")
                        make_successors.why = ("solid", x, o); return None
        changed.update(moved)
        return new

    # A SLIDE: after every move the controlled body goes on in one direction
    # until something stops it (a gravity) — applied to every successor
    slide_d = model.slide() if hasattr(model, "slide") and not getattr(model, "_no_slide", False) else None
    # (a DRIFT read by the law-maker stands before the slide's votes: the
    # body's extra step was the drift's one step, not a slide to the wall)
    if getattr(fixed, "drift", None) is not None:
        slide_d = None

    # (the body moves on a lattice: the slide goes by the same step the keys
    # move it, and stops before a step that anything blocks)
    unit_ = 0
    for k_ in keys:
        for sid_, e_ in fixed.moving(k_).items():
            if e_[0] == "translate" and sid_ in active_:
                m_ = max(abs(e_[1][0]), abs(e_[1][1]))
                if m_ and (unit_ == 0 or m_ < unit_):
                    unit_ = m_
    unit_ = unit_ or 1

    # THE DRIFT (a standing push read by the law-maker): after every press
    # the body also goes the drift's one step, when the way is free — a
    # slide of one step
    drift_ = getattr(fixed, "drift", None)
    support_ = {c_ for c_, n_ in (getattr(model, "support", {}) or {}).items() if n_ >= 1} if getattr(model, "gravity", None) is not None else set()

    def slide_body(st):
        if slide_d is None and drift_ is None:
            if _ENV.get("GRAV_DEBUG") and getattr(model, "gravity", None) is not None:
                print(f"      [gravity] no slide in the plan: slide_d {slide_d} no_slide {getattr(model, '_no_slide', False)}")
            return st
        members = [m for m in active_ if m in st and not isinstance(m, str)]
        if not members:
            if _ENV.get("GRAV_DEBUG"):
                print(f"      [gravity] no members: active {sorted(active_)} state keys {[k for k in st if not isinstance(k, str)][:6]}")
            return st
        new = dict(st)
        sv_ = (slide_d[0] * unit_, slide_d[1] * unit_) if slide_d is not None else tuple(drift_)
        for _ in (range(64) if slide_d is not None else range(1)):
            union = set()
            for m in members:
                union |= set(new[m])
            nxt = _sweep(union, sv_)
            ahead = nxt - union
            if any(r < 0 or c < 0 or r >= H or c >= W for r, c in ahead):
                break
            if wall_cells and ((ahead & wall_cells) - new.get("_cleared", frozenset())):
                break
            # (what the body was seen to rest on holds it: the ground, learned
            # as a support colour under gravity, stops the fall)
            if support_ and frame is not None and any(int(frame[r, c]) in support_ for r, c in ahead):
                if _ENV.get("GRAV_DEBUG"):
                    print(f"      [gravity] stop at {sorted(ahead)[:2]} on {sorted({int(frame[r, c]) for r, c in ahead})} (support {sorted(support_)})")
                break
            hit = False
            for o, ocs in new.items():
                if isinstance(o, str) or o in members or (o not in solids and not o < 0) or flat_here_(o, ocs) or model.is_ground(kinds.get(o), len(ocs)):
                    continue
                if ahead & ocs:
                    # (the cells aimed at are no obstacle to the slide either:
                    # the goal the drift carries the body onto is reached)
                    if target_cells and (ahead & ocs) <= frozenset(target_cells) and kinds.get(o) not in model.pushable and not _ENV.get("SLIDE_AIM_OLD"):
                        continue
                    hit = True; break
            if hit:
                if _ENV.get("GRAV_DEBUG"):
                    print(f"      [gravity] hit S{o} kind {kinds.get(o)} at {sorted(ahead & ocs)[:2]} from body {_bbox(union)}")
                break
            for m in members:
                new[m] = frozenset(_shift(new[m], sv_))
                move_parts(new, m, sv_)
        return new

    # A HAZARD: a kind the body touched on a press that lost the level — a
    # state where the body touches one is not a state to plan through
    hazard_kinds = model.hazards() if hasattr(model, "hazards") else set()

    def _with_material_(ks_):
        # (A HAZARD'S COLOUR IS A HAZARD IN EVERY FORM: the checker's lone
        # cells taught the colour; the same colour woven as a material —
        # kind colour+16 — kills as surely.  m0r0 L2)
        ks_ = set(ks_)
        return ks_ | {k_ + 16 for k_ in ks_ if isinstance(k_, int) and 0 <= k_ < 16} | {k_ - 16 for k_ in ks_ if isinstance(k_, int) and k_ >= 16}
    hazard_kinds = _with_material_(hazard_kinds)
    # (A HAZARD ENTERED IS NOT A WALL: a thing of a kind the body dies on
    # entering does not stop the body in the plan — it kills it there, and
    # the deadly test says so.  m0r0 L2's checker, read whole as one thing
    # at last, was a solid: one body of two "stayed" at it in the plan while
    # the world walked it in and sent both home)
    try:
        enter_haz_ = _with_material_(model.hazard_enter()) if hasattr(model, "hazard_enter") else set()
        if enter_haz_ and not _ENV.get("HAZARD_SOLID"):
            solids = {o for o in solids if kinds.get(o) not in enter_haz_}
    except Exception:
        pass
    # (a hazard by TOUCH is kept a cell away; one by ENTERING — a strip the
    # way runs along — only not stood on)
    touch_kinds = _with_material_(model.hazard_touch()) if hasattr(model, "hazard_touch") else set(hazard_kinds)

    def deadly(st):
        if not hazard_kinds or not active_:
            return False
        body = set()
        for m in active_:
            if m in st:
                body |= set(st[m])
        ring = {(r + dr, c + dc) for (r, c) in body for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        haz_ = set()
        for o, ocs in st.items():
            if isinstance(o, str) or o in active_ or kinds.get(o) not in hazard_kinds:
                continue
            if (ring if kinds.get(o) in touch_kinds else body) & ocs:
                if _ENV.get("RAISE_DEBUG"):
                    print(f"      [deadly] ring touches S{o} kind {kinds.get(o)} at {sorted(ring & ocs)[:3]} body {_bbox(body)}")
                return True
            haz_ |= ocs
        # A HAZARD AHEAD ALONG THE SLIDE: what the body would go on into
        # with nothing solid between (a drawn line is not a stop) kills
        # it as surely as a touch — the body cannot stop short of it
        if haz_ and slide_d is not None:
            stop_ = set(wall_cells) - set(st.get("_cleared", frozenset()))
            for o, ocs in st.items():
                if isinstance(o, str) or o in active_ or (o not in solids and not o < 0) or flat_here_(o, ocs) or model.is_ground(kinds.get(o), len(ocs)):
                    continue
                if not (ocs & haz_):
                    stop_ |= ocs
            front_ = set(body)
            for k_ in range(1, 64):
                front_ = {(r + slide_d[0], c + slide_d[1]) for (r, c) in front_} - body
                front_ = {q for q in front_ if q not in stop_ and 0 <= q[0] < H and 0 <= q[1] < W}
                if not front_:
                    break
                if front_ & haz_:
                    if _ENV.get("RAISE_DEBUG"):
                        print(f"      [deadly] ray from {_bbox(body)} meets hazard at {sorted(front_ & haz_)[:3]} after {k_} steps; 114 in st {114 in st} cells {sorted(st.get(114, ()))[:3]} in stop {sorted(set(st.get(114, ())) & stop_)[:3]} 114 solid {114 in solids} tags {[sorted(c_.tags) for c_ in comps if c_.sid == 114]} rider {114 in riders} haz&114 {sorted(set(st.get(114, ())) & haz_)[:3]} kinds114 {kinds.get(114)}")
                    return True
        return False

    click_keys = [k for k in fixed.keys() if isinstance(k, tuple) and k not in model.rel_keys]
    # a mark rides its wearer: it has no motion of its own, and it is not
    # in anything's way; a PART (id anchored to its object: sid*1000+colour)
    # rides its object under every transform
    riders = {c.sid for c in comps if livery_host(c, comps) is not None}
    parts_of = defaultdict(list)
    for c in comps:
        if c.sid >= 1000 and (c.sid // 1000) in kinds:
            parts_of[c.sid // 1000].append(c.sid)
            riders.add(c.sid)
    # A MARK ON A PIECE'S SIDE IS OF THE PIECE: a small thing lying against
    # a larger thing of another colour (touching it, however the frame was
    # cut into things) rides that thing — it moves with it and has no law
    # of its own (a piece's docking marks, read apart from the piece)
    for c in comps:
        if c.sid >= 1000 or c.sid in riders or c.size > 16 or "hud" in c.tags or "terrain" in c.tags:
            continue
        host_c = None
        for h in comps:
            if h.sid == c.sid or h.sid >= 1000 or h.body == c.body or "terrain" in h.tags or "hud" in h.tags or ("texture" in h.tags and _claims.on("texture_carries_no_rider")):
                continue                        # (a material — ground, a texture — carries no marks)
            hb = h.bbox; cb = c.bbox
            # (a thing lying WITHIN another's box — a crate's centre in its
            # border — is of it whatever their sizes)
            within_ = hb[0] < cb[0] and hb[1] < cb[1] and cb[2] < hb[2] and cb[3] < hb[3]
            if h.size <= 4 * c.size and not within_:
                continue
            if max(hb[0] - cb[2], cb[0] - hb[2], hb[1] - cb[3], cb[1] - hb[3]) > 1:
                continue                    # (boxes of touching things are one apart)
            if any(abs(r - hr) + abs(c_ - hc) == 1 for r, c_ in c.cells for hr, hc in h.cells):
                if host_c is None or h.size < host_c.size:
                    host_c = h
        if host_c is not None and (c.sid not in (model.active or ()) or host_c.sid in (model.active or ())):
            # (a mark the keys moved rides the piece the keys moved with it —
            # not a stranger it happens to touch)
            parts_of[host_c.sid].append(c.sid)
            riders.add(c.sid)
    # (a thing read as HIDDEN under another is an inference, not a sight:
    # it is in nobody's way until it shows)
    solids = set(solids) - riders - {c.sid for c in comps if "hidden" in c.tags}

    parts_of_all = {pid for ps in parts_of.values() for pid in ps}
    rider_host = {pid: host for host, ps in parts_of.items() for pid in ps}
    # WHAT THE MOVER PASSES THROUGH IS NOT IN ITS WAY: kinds seen to stand on
    # each other's cells are not solid to one another (the mover's kind, its
    # piece's, and the colour a selected thing wears)
    passable_ = getattr(model, "passable", set())
    if passable_:
        mk_ = {kinds.get(mover_sid), kinds.get(rider_host.get(mover_sid, mover_sid // 1000 if (mover_sid or 0) >= 1000 else mover_sid))}
        # (the mover's own kind, and what the selected thing wears — not every
        # member of the body: a front row of the held thing's colour would
        # pass what the held thing passes)
        mk_ |= set(model.kinds_worn_by_selected()) | {kinds.get(h_) for h_ in (model.active or ()) if not any(h_ == o_ for (_, o_) in getattr(model, "bound", ()))}
        mk_.discard(None)
        # (a kind the carried thing wears — the body's nose may wear it too —
        # is the carried thing's leave, not the body's)
        if hasattr(model, "carried_kinds"):
            mk_ -= (model.carried_kinds() - {kinds.get(mover_sid)})
        solids = {o for o in solids if not any((k_, kinds.get(o)) in passable_ for k_ in mk_)}
        # (a law learned on kinds is presumed of like-formed things: a piece
        # seen to pass through a piece passes through the other pieces —
        # the selectable kinds — until a press refuses)
        piece_kinds_ = set(sel_kinds) | set(model.kinds_worn_by_selected())
        # (only a piece seen through a PIECE says so: a piece seen over a
        # frame on the floor says nothing about the other pieces)
        if any(k1_ in piece_kinds_ and k2_ in piece_kinds_ for k1_, k2_ in passable_):
            solids = {o for o in solids if kinds.get(o) not in piece_kinds_ or o in frozen}

    def move_parts(new, x, v, skip=frozenset()):
        for pid in parts_of.get(x, ()):
            if pid in new and pid not in skip:
                new[pid] = frozenset(_shift(new[pid], v))

    keys = [k for k in keys if k not in model.undo_keys]
    cur_key = [None]                            # the key whose successors are being made (portals are by key)
    # a DISPLAY shows a consequence: its "laws" under a key (the picture
    # changing) are not moves the plan makes — it stays where it is
    displays_ = set(getattr(model, "displays", {}) or {})

    def succ(state):
        out = []
        sel = state.get("_sel")
        # keys, and the clicks on things that did something (buttons)
        make_successors.whys = {}
        for k in list(keys) + click_keys:
            make_successors.why = None
            cur_key[0] = k
            eff = dict(fixed.moving(k, model.sel_ctx(sel)))     # the key's laws under this state's selection (by body)
            if has_cond_:
                # THE OUTCOME CONDITIONED ON WHAT CAME BEFORE: the key's law on a
                # thing may depend on the press before it and the thing's
                # heading (a running jump); the plan's state carries both, the
                # world's last press and headings stand for the start
                prev_k_ = state["_prev"] if "_prev" in state else fixed.last_key
                unknown_ = None
                for psid in list(eff):
                    if isinstance(psid, str):
                        continue
                    cl_ = fixed.cond_law(k, psid, prev_k_, state.get(f"_head{psid}", fixed.heading.get(psid)))
                    if cl_ is None:
                        continue
                    if cl_ == "unknown":
                        unknown_ = psid; break
                    eff[psid] = cl_
                if unknown_ is not None:
                    # (never pressed under this condition: not planned with — a press there will tell)
                    make_successors.whys[k] = ("unknown under this condition", prev_k_, unknown_)
                    continue
            for psid in list(eff):
                if psid in derived_ or psid in cent_:
                    eff.pop(psid); continue
                if psid in displays_ or (psid >= 1000 and psid // 1000 in displays_):
                    # (a change the key made on a minority of its presses is
                    # a consequence of where the mover was, not the key's law;
                    # one it makes at every press is a law, display or not)
                    if fixed.alt[k][psid][eff[psid]] < 0.5 * fixed.presses[k]:
                        eff.pop(psid)
            # a part with its own record under this key moves by it; a part
            # without one rides its object
            # (what moves with the mover is the mover: a part whose object
            # moves under the key rides it — its own record is how it is
            # drawn within the object, e.g. an eye turning to the side —
            # and is not a motion to check against walls)
            for sid_ in list(eff):
                if sid_ in parts_of_all and (sid_ // 1000 if sid_ >= 1000 else rider_host.get(sid_)) in eff:
                    eff.pop(sid_)
                elif sid_ in parts_of_all and sid_ < 1000:
                    # a side mark has no law of its own: it goes where its
                    # piece goes (its record under the key is the piece's)
                    eff.pop(sid_)
            own_parts = {sid_ for sid_ in eff if sid_ in parts_of_all}
            if getattr(make_successors, "trace", False):
                print(f"      [eff] {k}: laws {dict(fixed.moving(k, model.sel_ctx(sel)))} -> {eff}; displays {sorted(displays_)[:6]} parts_of_all∩ {sorted(set(fixed.moving(k)) & parts_of_all)} rider_host {{s_: rider_host.get(s_) for s_ in fixed.moving(k) if s_ in rider_host}} presses {fixed.presses.get(k)} alt {[(s_, dict(fixed.alt[k][s_])) for s_ in fixed.moving(k) if s_ in displays_]}")
            if k in model.positional:
                # BY SLOT: each thing moves as things at its slot did; a
                # thing at a slot never seen stays (the press will teach)
                pe = model.pos_eff.get(k, {})
                new = dict(state); changed = set(); ok = True
                unknown = 0
                for psid, cs in state.items():
                    if isinstance(psid, str) or psid not in kinds:
                        continue
                    r0, c0, _, _ = _bbox(cs)
                    v = pe.get((r0, c0))
                    if v is None or (r0, c0) in model.pos_bad.get(k, set()):
                        unknown += 1
                        continue
                    cs2 = _shift(cs, v)
                    if any(r < 0 or c < 0 or r >= H or c >= W for r, c in cs2):
                        ok = False; break
                    new[psid] = frozenset(cs2); changed.add(psid)
                if ok and changed:
                    out.append((k, new))
                continue
            v_sel = model.key_selected.get(k)
            new = dict(state)
            changed = set()
            ok = True
            sel_body = frozenset()
            if sel is not None:
                sel_body = frozenset(o for b in bodies_ if sel in b for o in b if o in new) | {sel}
            if v_sel is not None and sel is not None and sel not in frozen:
                # the key moves the selected thing AND what always moves with
                # it; fixed translates on things of the selectable kinds are
                # dropped (they were the selected one when recorded)
                for psid in list(eff):
                    host_k = kinds.get(psid // 1000) if psid >= 1000 else None     # a part's host (a mark on a block)
                    if kinds.get(psid) in sel_kinds or kinds.get(psid) == kinds.get(sel) or psid in sel_body or host_k in sel_kinds \
                            or (model.selected_colour is not None and kinds.get(psid) in model.kinds_worn_by_selected()):
                        eff.pop(psid)
                own_parts = {sid_ for sid_ in eff if sid_ in parts_of_all}      # (a part whose record was dropped rides its piece)
                res = new
                for member in sorted(sel_body, key=lambda o: -len(new.get(o, ()))):
                    if member not in new:
                        continue
                    if member in parts_of_all and rider_host.get(member, member // 1000 if member >= 1000 else None) in sel_body:
                        continue            # (a mark of a member rides it: moved once, with its piece)
                    res = push_move(new, member, v_sel, changed, frozenset(eff) | sel_body, own_parts)
                    if res is None:
                        break
                    new = res
                else:
                    res = new
                if res is None:
                    make_successors.whys[k] = make_successors.why
                    continue
            turn = model.key_rotate.get(k)
            if turn is not None and sel is not None and sel not in frozen:
                # the key turns the selected body about the corner of its box
                for psid in list(eff):
                    if psid in sel_body or kinds.get(psid) in sel_kinds or kinds.get(psid) == kinds.get(sel):
                        eff.pop(psid)
                members = [o for o in sel_body if o in new]
                whole = set()
                for o in members:
                    whole |= set(new[o])
                if not whole:
                    make_successors.whys[k] = ("nothing selected to turn", sel)
                    continue
                # (the marks on its sides turn with it, about the box of the
                # whole — piece and marks — as a body's members do)
                pids_ = [pid for o in members for pid in parts_of.get(o, ()) if pid in new and pid not in members]
                for pid in pids_:
                    whole |= set(new[pid])
                _, turned = _rot(whole, turn, [set(new[o]) for o in members + pids_])
                bad = False
                for o, tc in zip(members + pids_, turned):
                    if any(r < 0 or c < 0 or r >= H or c >= W for r, c in tc):
                        bad = True; break
                    new[o] = frozenset(tc); changed.add(o)
                if bad:
                    make_successors.whys[k] = ("turn out of frame", sel)
                    continue
            n_bodies_ = len({next((i_ for i_, b_ in enumerate(bodies_) if psid_ in b_), ("one", psid_))
                             for psid_, e_ in eff.items() if e_[0] == "translate"})
            apart_ = set()
            if n_bodies_ >= 2:
                tr_ = [psid_ for psid_, e_ in eff.items() if e_[0] == "translate" and psid_ in state and not isinstance(psid_, str)]
                for psid_ in tr_:
                    b1_ = _bbox(state[psid_])
                    if all(max(b1_[0] - _bbox(state[q_])[2], _bbox(state[q_])[0] - b1_[2], b1_[1] - _bbox(state[q_])[3], _bbox(state[q_])[1] - b1_[3]) > 1
                           for q_ in tr_ if q_ != psid_):
                        apart_.add(psid_)
            for psid, e in eff.items():
                if psid not in new:
                    if e[0] == "extend" and new.get(f"_last{psid}"):
                        new[psid] = frozenset(new[f"_last{psid}"]); changed.add(psid)     # (regrown from its place)
                    elif e[0] == "retract" and not isinstance(psid, str) and psid < 1000 and psid not in hud_sids_:
                        # (A PUMP WITH NOTHING TO GIVE GIVES NOTHING: the thing it
                        # draws from is gone — pumped down to nothing in the plan,
                        # or standing empty from the start (vc33 L2–L3's columns).
                        # A THING, not a part (the part the body's step covers,
                        # tw02 L1), and not the band's — the clock's bar the key
                        # also shortens (tw02 L3): those spend nothing)
                        ok = False; make_successors.why = ("spent", psid); break
                    continue
                # (a pump at its limit refuses for ITSELF: the key still moves
                # the body — each thing answers the key on its own)
                if e[0] == "retract" and len(new[psid]) <= fixed.min_size.get(psid, 0):
                    if psid == mover_sid or psid in active_ or not _claims.on("pump_limit_refuses_only_itself"):
                        ok = False; break
                    continue
                if e[0] == "extend" and len(new[psid]) >= fixed.max_size.get(psid, 10**9):
                    if psid == mover_sid or psid in active_ or not _claims.on("pump_limit_refuses_only_itself"):
                        ok = False; break
                    continue
                if e[0] == "translate":
                    res = push_move(new, psid, e[1], changed, frozenset(eff), own_parts)
                    if res is not None and (k, psid) in getattr(fixed, "slides", ()):
                        # A SLIDE goes on, step after step, until something
                        # stops it: the plan takes it where it stops
                        for _ in range(64):
                            more = push_move(res, psid, e[1], changed, frozenset(eff), own_parts)
                            if more is None:
                                break
                            res = more
                    if res is None:
                        # EACH BODY ANSWERS THE KEY FOR ITSELF: when a key moves
                        # several bodies and a press has shown one of them
                        # staying while another moved (a wall before one, not
                        # the other), a refused body stays and the rest go on
                        # (Tushar, m0r0: "if you press against a wall, one block
                        # does not move, another one does")
                        # (bodies that stand apart are presumed to answer for
                        # themselves; bodies that touch may be one chain —
                        # those only once a press has shown one staying)
                        if n_bodies_ >= 2 and psid not in sel_body and (model.stayed.get((k, psid), 0) > 0 or psid in apart_):
                            if getattr(make_successors, "trace", False) or _ENV.get("SKIP_DEBUG"):
                                print(f"      [skip] {k} S{psid}: {make_successors.why} (stayed {model.stayed.get((k, psid), 0)}, apart {psid in apart_}, bodies {n_bodies_})")
                            make_successors.why = None
                            continue
                        ok = False; break
                    new = res
                elif e[0] == "rotate":
                    # the thing turns about its box's corner; its parts turn
                    # with it (a stub on a piece)
                    b0 = _bbox(new[psid])
                    cs = apply_eff(new[psid], e)
                    if any(r < 0 or c < 0 or r >= H or c >= W for r, c in cs):
                        ok = False; break
                    # ...and so do the things that always move with it (its
                    # stubs) when the key gives them no law of their own
                    comovers = [o for b in bodies_ if psid in b for o in b if o != psid and o in new and o not in eff]
                    pids = [pid for pid in parts_of.get(psid, ()) if pid in new and pid not in own_parts] + comovers
                    if pids:
                        _, turned = _rot(set(new[psid]), e[1], [set(new[pid]) for pid in pids])
                        for pid, tc in zip(pids, turned):
                            new[pid] = frozenset(tc)
                    new[psid] = frozenset(cs); changed.add(psid)
                else:
                    cs = apply_eff(new[psid], e)
                    if e[0] == "retract" and not cs:
                        # (A THING PUMPED DOWN TO NOTHING is still a thing: it
                        # keeps its place and grows again from there — vc33's
                        # emptied piston refilled from its neighbour)
                        new[f"_last{psid}"] = frozenset(new[psid])
                        del new[psid]
                        for pid in parts_of.get(psid, ()):
                            new.pop(pid, None)
                        changed.add(psid)
                        continue
                    if any(r < 0 or c < 0 or r >= H or c >= W for r, c in cs):
                        ok = False; break
                    # a part at the growing end rides the end (a tip on a
                    # bar); a part inside the cut region of a retract too
                    side, kk = e[1]
                    dvec = {"top": (-kk, 0), "bottom": (kk, 0), "left": (0, -kk), "right": (0, kk)}[side]
                    if e[0] == "retract":
                        dvec = (-dvec[0], -dvec[1])
                    b0 = _bbox(new[psid])
                    for pid in parts_of.get(psid, ()):
                        if pid not in new or pid in own_parts:
                            continue
                        pb = _bbox(new[pid])
                        at_edge = ((side == "right" and pb[3] >= b0[3] - kk + 1) or (side == "left" and pb[1] <= b0[1] + kk - 1)
                                   or (side == "bottom" and pb[2] >= b0[2] - kk + 1) or (side == "top" and pb[0] <= b0[0] + kk - 1))
                        if at_edge:
                            new[pid] = frozenset(_shift(new[pid], dvec))
                    new[psid] = frozenset(cs); changed.add(psid)
            if not ok or not changed:
                if not changed:
                    make_successors.why = make_successors.why or ("nothing changed", sel, v_sel, list(eff)[:4])
                make_successors.whys[k] = make_successors.why
                if _ENV.get("PUSH_DEBUG") and any(isinstance(o_, int) and o_ in new and o_ != mover_sid and new.get(o_) != state.get(o_) for o_ in state):
                    print(f"      [push] successor dropped after a shove: {k} why {make_successors.why} ok {ok} changed {changed}")
                continue
            # collisions for non-translate changes (translates were
            # settled in push_move)
            for psid in changed:
                if psid not in solids or psid not in new or (eff.get(psid, ("translate",))[0] == "translate"):
                    continue
                for o, ocs in new.items():
                    if o == psid or o in changed or o not in solids or frozenset((psid, o)) in fixed.overlap_ok or kinds.get(o) in model.flat:
                        continue
                    if new[psid] & ocs:
                        ok = False; make_successors.why = ("grow-solid", psid, o); break
                if not ok:
                    break
            if ok:
                new = slide_body(new)
                if deadly(new):
                    make_successors.whys[k] = ("deadly", sel)
                    continue
                if has_cond_:
                    new["_prev"] = k
                    # (a thing faces the way it last moved or turned)
                    for psid, e in eff.items():
                        if not isinstance(psid, str) and psid in changed and (e[0] == "rotate" or (e[0] == "translate" and (e[1][0] or e[1][1]))):
                            new[f"_head{psid}"] = k
                out.append((k, new))
            else:
                make_successors.whys[k] = make_successors.why or "limit/none"
        # A CLICK THAT TAKES A THING AWAY: a thing of a kind a click removes
        # (a breakable block) is one click from gone — the ground it stood
        # on is walked on after; what stood against it slides on
        removable_ = model.removable() if hasattr(model, "removable") and not getattr(model, "_no_remove", False) else set()
        # (a thing is taken away — or raised — when it is in the way: one within
        # reach of the body — touching it, or in its line along the slide — else
        # the search would try every change everywhere)
        body_ = set()
        for m in active_:
            if m in state:
                body_ |= set(state[m])
        # (within two steps of the body: a thing beside the way, removed before
        # stepping under it)
        step_ = 1
        for k_ in keys:
            for e_ in fixed.moving(k_).values():
                if e_[0] == "translate":
                    step_ = max(step_, abs(e_[1][0]), abs(e_[1][1]))
        # (IN THE WAY: where the body would go in one or two steps of any
        # key — a thing elsewhere within reach is not the search's business:
        # every change everywhere is what made the search explode)
        near_ = set()
        if body_:
            for k_ in keys:
                for m in active_:
                    e_ = fixed.moving(k_).get(m)
                    if e_ and e_[0] == "translate" and (e_[1][0] or e_[1][1]):
                        v_ = e_[1]
                        near_ |= set(_sweep(body_, v_)) | set(_sweep(body_, (2 * v_[0], 2 * v_[1])))
            # (and what lies within one step of the body on any side: the
            # block it stands against, the block it would fall through)
            bb_ = _bbox(body_)
            for r in range(bb_[0] - step_, bb_[2] + step_ + 1):
                for c in range(bb_[1] - step_, bb_[3] + step_ + 1):
                    near_.add((r, c))
        if slide_d is not None and body_:
            for k_ in range(1, 64):
                for (r, c) in body_:
                    near_.add((r + slide_d[0] * k_, c + slide_d[1] * k_))
        if removable_:
            for o in list(state):
                if isinstance(o, str) or o not in solids or kinds.get(o) not in removable_ or o in derived_ or o in frozen:
                    if _ENV.get("REMOVE_DEBUG") and not isinstance(o, str) and kinds.get(o) in removable_:
                        print(f"      [remove] S{o} skipped: solids {o in solids} derived {o in derived_} frozen {o in frozen}")
                    continue
                cs_ = state[o]
                if not cs_ or (body_ and not (set(cs_) & near_) and not _ENV.get("REMOVE_ANY")):
                    if _ENV.get("REMOVE_DEBUG"):
                        print(f"      [remove] S{o} not near: body {_bbox(body_) if body_ else None} thing {_bbox(cs_) if cs_ else None}")
                    continue
                # (a mark of the kind — far smaller than what a click took
                # away — is not taken away by a click: it is raised, below)
                gs_ = getattr(model, "click_gone_size", {}).get(kinds.get(o))
                if gs_ is not None and len(cs_) < 0.5 * gs_:
                    continue
                r_, c_ = centre(cs_)
                pt = min(cs_, key=lambda q: abs(q[0] - r_) + abs(q[1] - c_))
                new = dict(state)
                del new[o]
                for pid in parts_of.get(o, ()):
                    new.pop(pid, None)
                # (what the removal clears beyond the thing — its edges — is
                # wall no more: recorded per state as cleared cells)
                g_ = model.click_clears.get(kinds.get(o)) if hasattr(model, "click_clears") else None
                if g_ and any(g_):
                    b_ = _bbox(cs_)
                    cleared = frozenset((r2, c2) for r2 in range(b_[0] - g_[0], b_[2] + g_[2] + 1) for c2 in range(b_[1] - g_[1], b_[3] + g_[3] + 1))
                    new["_cleared"] = frozenset(state.get("_cleared", frozenset())) | cleared
                new = slide_body(new)
                if deadly(new):
                    continue
                out.append((("click", pt), new))
        # A CLICK THAT RAISES A BLOCK FROM ITS MARK (the toggle's other way):
        # a mark of a kind seen to rise is offered as a click; the block it
        # raises stands in the state as a solid of its own (id negative:
        # nothing else's), within the body's reach or along its slide
        grows_ = getattr(model, "click_grows", {}) if not getattr(model, "_no_remove", False) else {}
        if grows_ and body_:
            # (a mark may be read as several small pieces, or several marks
            # as one constellation: the mark is the cluster of the kind's
            # small cells within a block's span; the block rises centred on it)
            for kd_, (msz_, dims_) in grows_.items():
                small_ = {}
                for o in state:
                    if isinstance(o, str) or o < 0 or o in derived_ or o in frozen or kinds.get(o) != kd_ or not state[o]:
                        continue
                    if len(state[o]) <= max(12, 2 * msz_) * 3:
                        small_[o] = set(state[o])
                if not small_:
                    continue
                cells_all = {q: o for o, cs_ in small_.items() for q in cs_}
                left_ = set(cells_all); clusters_ = []
                while left_:
                    q0 = min(left_); cl_ = {q0}; fr_ = [q0]; left_.discard(q0)
                    while fr_:
                        q = fr_.pop()
                        for dr_ in range(-(dims_[0] // 2), dims_[0] // 2 + 1):
                            for dc_ in range(-(dims_[1] // 2), dims_[1] // 2 + 1):
                                q2 = (q[0] + dr_, q[1] + dc_)
                                if q2 in left_:
                                    left_.discard(q2); cl_.add(q2); fr_.append(q2)
                    clusters_.append(frozenset(cl_))
                clusters_.sort(key=min)
                if _ENV.get("RAISE_DEBUG"):
                    print(f"      [raise] kind {kd_} small {sorted(small_)} clusters {[(len(c_), _bbox(c_)) for c_ in clusters_]} body {_bbox(body_)} near {len(near_)}")
                for ci_, cl_ in enumerate(clusters_):
                    if len(cl_) > 2 * max(msz_, 5):
                        continue
                    r_, c_ = centre(cl_)
                    r0_ = int(round(r_ - (dims_[0] - 1) / 2.0)); c0_ = int(round(c_ - (dims_[1] - 1) / 2.0))
                    box_ = frozenset((r2, c2) for r2 in range(r0_, r0_ + dims_[0]) for c2 in range(c0_, c0_ + dims_[1]) if 0 <= r2 < H and 0 <= c2 < W)
                    if not (box_ & near_) and not _ENV.get("REMOVE_ANY"):
                        if _ENV.get("RAISE_DEBUG"):
                            print(f"      [raise] cluster {_bbox(cl_)} not near")
                        continue
                    if box_ & body_:
                        continue                    # (it would rise into the body)
                    owners_ = {cells_all[q] for q in cl_}
                    if any((not isinstance(o2, str)) and o2 not in owners_ and (state[o2] & box_) for o2 in state if o2 in solids or (isinstance(o2, int) and o2 < 0)):
                        if _ENV.get("RAISE_DEBUG"):
                            print(f"      [raise] cluster {_bbox(cl_)}: something solid where it would rise {[o2 for o2 in state if not isinstance(o2, str) and o2 not in owners_ and (state[o2] & box_)]}")
                        continue                    # (something solid stands where it would rise)
                    pt = min(cl_, key=lambda q: abs(q[0] - r_) + abs(q[1] - c_))
                    new = dict(state)
                    for o2 in owners_:
                        rest_ = frozenset(small_[o2]) - cl_
                        if rest_:
                            new[o2] = rest_
                        else:
                            del new[o2]
                    new[-(min(owners_) * 100 + ci_ + 1)] = box_
                    new = slide_body(new)
                    if deadly(new):
                        if _ENV.get("RAISE_DEBUG"):
                            bb2 = _bbox(set().union(*[set(new[m]) for m in active_ if m in new]))
                            print(f"      [raise] hazards {hazard_kinds} flat {model.flat} flat_size {getattr(model, 'flat_size', {})} slide {slide_d} S114-flat {model.is_flat(12, 25)}")
                            print(f"      [raise] cluster {_bbox(cl_)}: deadly after (body {bb2}); solids near {[(o2, sorted(new[o2])[:2], kinds.get(o2), o2 in solids) for o2 in new if not isinstance(o2, str) and o2 not in active_ and any(bb2[0] - 8 <= r <= bb2[2] + 2 and bb2[1] - 2 <= c <= bb2[3] + 2 for (r, c) in new[o2])]}")
                        continue
                    out.append((("click", pt), new))
        # selection clicks
        for kd in sel_kinds:
            for h in hosts_by_kind.get(kd, []):
                if h != sel and h in state and h not in frozen:
                    new = dict(state); new["_sel"] = h; new["_sel_by_plan"] = True
                    out.append((("select", h), new))
        # a key that hands the mark to the next thing
        for k, pairs in model.cycle_keys.items():
            nxt = next((b for a, b in pairs if a == sel), None)
            if nxt is None:
                cands = [h for h, _ in pairs] + [b for _, b in pairs]
                cands = [h for h in cands if h != sel and h in state]
                nxt = cands[0] if cands else None
            if nxt is not None and nxt in state and nxt != sel:
                new = dict(state); new["_sel"] = nxt
                out.append((k, new))
        # A THING HUNG BETWEEN HANDLES IS AIMED BY ITS HANDLES: the selected
        # handle is sent (a click, where a click sends it) to the place that
        # puts the centroid on the target — or as far that way as the
        # screen allows, the other handles then doing the rest
        if mover_sid in cent_ and mover_sid in state:
            hs_ = [h_ for h_ in cent_[mover_sid] if h_ in state]
            sel_ = state.get("_sel")
            # (the click law is the handles': learned on the one that was
            # selected, worn by whichever is selected — a selected handle is
            # drawn otherwise, so its kind alone does not carry the law)
            law_ = next((model.relative_law(kinds.get(h_)) for h_ in hs_ if model.relative_law(kinds.get(h_)) is not None), None)
            if hs_ and (sel_ not in hs_ or (not state.get("_sel_by_plan") and model.relative_law(kinds.get(sel_)) is None)):
                # (until the plan itself selects a handle, the selected one is
                # the one the clicks were seen to send — its kind bears the law)
                sel_ = next((h_ for h_ in hs_ if model.relative_law(kinds.get(h_)) is not None), hs_[0])
            if hs_ and sel_ in hs_ and law_ is not None and law_[0] == "teleport":
                k_ = len(hs_)
                others_ = [_icentre(state[h_]) for h_ in hs_ if h_ != sel_]
                dc_ = _icentre(state[mover_sid])
                # (the target as the centroid must show it: the mover's centre there)
                want_ = (int(round(target[0])) * k_ - sum(c_[0] for c_ in others_), int(round(target[1])) * k_ - sum(c_[1] for c_ in others_))
                hb_ = _bbox(state[sel_]); hh_, hw_ = hb_[2] - hb_[0], hb_[3] - hb_[1]
                clip_ = (max(hh_ // 2, min(H - 1 - (hh_ - hh_ // 2), want_[0])), max(hw_ // 2, min(W - 1 - (hw_ - hw_ // 2), want_[1])))
                hc_ = _icentre(state[sel_])
                # (the place itself; else as far that way as the screen and
                # the walls allow — the other handles do the rest)
                cands_ = [want_] if want_ == clip_ else [want_, clip_]
                for t_ in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
                    cands_.append((int(round(hc_[0] + t_ * (want_[0] - hc_[0]))), int(round(hc_[1] + t_ * (want_[1] - hc_[1])))))
                offered_ = 0
                for pt_ in cands_:
                    v_ = (pt_[0] - hc_[0], pt_[1] - hc_[1])
                    if v_ == (0, 0):
                        continue
                    cs_ = frozenset(_shift(state[sel_], v_))
                    if any(r < 0 or c < 0 or r >= H or c >= W for r, c in cs_):
                        continue
                    if wall_cells and (cs_ & wall_cells):
                        continue
                    if (kinds.get(sel_), _bbox(state[sel_])[0], _bbox(state[sel_])[1], v_) in refused_:
                        continue
                    new = dict(state); new[sel_] = cs_; move_parts(new, sel_, v_)
                    new["_sel"] = sel_
                    out.append((("rclick", (int(pt_[0]), int(pt_[1]))), _hang(new)))
                    offered_ += 1
                    if offered_ >= 2:
                        break
        # relative clicks toward the target, for the mover
        if rel_law is not None and mover_sid in state and mover_sid not in cent_:
            c0 = centre(state[mover_sid])
            d = (target[0] - c0[0], target[1] - c0[1])
            if abs(d[0]) + abs(d[1]) >= 1:
                if rel_law[0] == "teleport":
                    reach = rel_law[1]
                    n = abs(d[0]) + abs(d[1])
                    f = min(1.0, reach / n) if n > 0 else 0
                    v = (int(round(d[0] * f)), int(round(d[1] * f)))
                else:
                    cap, reach = rel_law[1], rel_law[2]
                    v = (int(max(-cap, min(cap, d[0]))), int(max(-cap, min(cap, d[1]))))
                # (a step refused once is not the only step: half of it is
                # offered too — the call lands the thing short of what stood
                # in the way, and the next call goes on from there)
                vs_ = [v] if v != (0, 0) else []
                if rel_law[0] == "teleport" and max(abs(v[0]), abs(v[1])) >= 2:
                    vh_ = (int(round(v[0] / 2.0)), int(round(v[1] / 2.0)))
                    if vh_ != (0, 0) and vh_ != v:
                        vs_.append(vh_)
                for v in vs_:
                    new = dict(state); changed = set()
                    res = push_move(new, mover_sid, v, changed)
                    if _ENV.get("PLAN_DEBUG"):
                        print(f"      [rclick] mover S{mover_sid} c0 {c0} d {d} v {v} -> {'ok' if res is not None else 'blocked'}; solids {sorted(solids)[:6]} walls {len(wall_cells or ())} why {getattr(make_successors, 'why', None) if res is None else ''}")
                    if res is not None:
                        pt = (int(round(c0[0] + v[0])), int(round(c0[1] + v[1])))
                        out.append((("rclick", pt), res))
        if cent_:
            out = [(act, _hang(dict(new))) for act, new in out]
        if derived_:
            out3 = []
            for act, new in out:
                new = dict(new)
                for dsid, (a_sid, L_sid, line0) in derived_.items():
                    if a_sid not in new:
                        continue
                    line = line0
                    if L_sid in new:
                        Lc = new[L_sid]
                        r0_ = min(r for r, _ in Lc); r1_ = max(r for r, _ in Lc); c0_ = min(c for _, c in Lc); c1_ = max(c for _, c in Lc)
                        line = (line0[0], (c0_ + c1_) / 2.0 if line0[0] == "v" else (r0_ + r1_) / 2.0)
                    new[dsid] = frozenset(_mir(new[a_sid], line))
                out3.append((act, new))
            out = out3
        if clock_on and "_clock" in state:
            out2 = []
            for act, new in out:
                c_ = new.get("_clock", state["_clock"]) - model.clock_rate
                new = dict(new)
                if mover_sid in new and new[mover_sid] != state.get(mover_sid):
                    for o in refill_sids:
                        if o in new and (new[o] & new[mover_sid]):
                            c_ = max(c_, model.clock_full); new.pop(o); break
                if c_ < 0:
                    make_successors.whys[act] = ("clock",)
                    continue
                new["_clock"] = c_
                out2.append((act, new))
            out = out2
        return out

    def shove_of(state, o, v):
        """Where thing o would come to rest if struck along v in this
        state (the slide the successors use), with its parts moved."""
        new = dict(state)
        if o not in new or not new[o]:
            return None
        sh_ = slide_shove(new, o, None, v)
        if sh_ is None:
            return None
        cells_, dv_ = sh_
        new[o] = frozenset(cells_); move_parts(new, o, dv_)
        return new
    succ.shove = shove_of
    succ.wall_cells = wall_cells
    succ.solids = solids
    succ.kinds = kinds
    succ.sel_kinds = set(sel_kinds)
    succ.parts_of = parts_of
    return succ


make_successors = PerThread(make_successors)


def plan_by_shove(model, succ, start, sid, goal_t, dist, depth=2, docked=frozenset(), verbose=False, max_seconds=60.0):
    """A THING THAT CANNOT GO THERE MAY BE SENT THERE (Tushar, ka59:
    "hitting another object makes it go far ... it needed a plan").  When
    the thing's own moves reach no state the goal accepts, it is walked
    to a LAUNCH — a place from which a strike by another piece sends it
    sliding (the shove the successors know: its run, and on across what
    a shoved thing slides over) to a landing from which its own moves,
    or one more strike, reach the goal — and a pusher is routed to the
    place behind the launch: another selectable piece, by its own moves
    or, failing that, by a strike of its own.  Nothing here names a
    place: the launches are wherever the thing can stand, the strikes
    whatever the keys send, the pushers whatever the clicks select.
    Returns a flat list of actions (selects, keys) or None.
    `docked`: pieces already where a reading wants them — used last."""
    import time
    from collections import deque
    t_end = time.time() + max_seconds
    ksel = dict(model.key_selected or {})
    if not ksel:
        return None

    def keys_only(st):
        return [(a, n) for a, n in succ(st) if isinstance(a, str)]

    def select_to(st, who):
        if st.get("_sel") == who:
            return [], st
        nxt = next((n for a, n in succ(st) if a == ("select", who)), None)
        if nxt is None:
            n2 = dict(st); n2["_sel"] = who
            return [("select", who)], n2
        return [("select", who)], nxt

    def own_plan(st, who, goal, heur, max_states, max_depth=40):
        pre, st1 = select_to(st, who)
        if goal(st1):
            return pre, st1
        p = plan_general(st1, keys_only, goal, heuristic=heur, max_states=max_states, max_depth=max_depth, max_seconds=max(1.0, t_end - time.time()))
        if p is None:
            return None, None
        st2 = st1
        for a in p:
            st2 = next((n for aa, n in keys_only(st2) if aa == a), st2)
        return pre + p, st2

    def region(st, who, limit):
        """The states the thing's own moves reach (and the way to each) —
        the moves that carry it alone: a step that shoves another thing
        is not a place to stand but an event, planned as such."""
        pre, st1 = select_to(st, who)
        seen = {_sig(st1)}
        out = [(st1, list(pre))]
        q = deque([(st1, list(pre))])
        own_ = {who} | set(succ.parts_of.get(who, ()))
        while q and len(out) < limit and time.time() < t_end:
            s, path = q.popleft()
            for a, n in keys_only(s):
                if any((not isinstance(o, str)) and o not in own_ and n.get(o) != s.get(o) for o in set(s) | set(n)):
                    continue
                g = _sig(n)
                if g in seen:
                    continue
                seen.add(g)
                out.append((n, path + [a])); q.append((n, path + [a]))
        return out

    def _sig(st):
        return tuple(sorted((str(k), hash(v)) for k, v in st.items()))

    def pushers_in(st, who):
        return [o for o in st if not isinstance(o, str) and 0 < o < 1000 and o != who and succ.kinds.get(o) in succ.sel_kinds]

    memo = {}

    def route(st, who, goal, heur, depth_, pusher_depth, label):
        mk_ = (who, depth_, pusher_depth, _sig(st), id(goal))
        if mk_ in memo:
            return memo[mk_]
        res_ = _route(st, who, goal, heur, depth_, pusher_depth, label)
        memo[mk_] = res_
        return res_

    def _route(st, who, goal, heur, depth_, pusher_depth, label):
        if not st.get(who):
            return None, None           # (the thing is not in this state: nothing to route)
        p, st_end = own_plan(st, who, goal, heur, max_states=2500)
        if p is not None:
            return p, st_end
        if depth_ <= 0 or time.time() > t_end:
            return None, None
        reg = region(st, who, limit=1200)
        # the launches: every place reached, every key's strike from it —
        # ranked by how near the landing comes to the goal
        cands = []
        seen_land = set()
        for s, path in reg:
            if not s.get(who):
                continue
            for k, v in ksel.items():
                s2 = succ.shove(s, who, v)
                if s2 is None or s2[who] == s[who]:
                    continue
                key_ = (_bbox(s2[who]), _bbox(s[who]))
                if key_ in seen_land:
                    continue
                seen_land.add(key_)
                cands.append((heur(s2), len(path), s, path, k, v, s2))
        cands.sort(key=lambda c: (c[0], c[1]))
        if verbose:
            print(f"      [shove] {label}: S{who} region {len(reg)} launches {len(cands)}; nearest landings {[(round(c[0], 1), _bbox(c[2][who])[:2], c[4], _bbox(c[6][who])[:2]) for c in cands[:6]]}")
        best = None
        tried = 0
        for (h2, n_path, s, path, k, v, s2) in cands:
            if time.time() > t_end or tried >= 10:
                break
            tried += 1
            # from the landing: own moves, or one more strike
            rest, _ = route(s2, who, goal, heur, depth_ - 1, pusher_depth, label + ">")
            if rest is None:
                continue
            # a pusher behind the launch: it strikes when its next step by
            # this key would enter the thing's cells
            mover_cells = s[who]
            mb = _bbox(mover_cells)
            mc = centre(mover_cells)
            unit_ = max(abs(v[0]), abs(v[1])) or 1

            def goal_b(stb, _b=None, _m=mover_cells, _v=v):
                b = stb.get(_b)
                if not b or (b & _m):
                    return False
                return bool(_shift(b, _v) & _m)
            found = None
            for b in sorted(pushers_in(s, who), key=lambda o: (o in docked, abs(centre(s[o])[0] - mc[0]) + abs(centre(s[o])[1] - mc[1]))):
                if time.time() > t_end:
                    break
                bd = _bbox(s[b])
                want = (mc[0] - (v[0] / unit_) * ((mb[2] - mb[0] + 1) + (bd[2] - bd[0] + 1)) / 2.0,
                        mc[1] - (v[1] / unit_) * ((mb[3] - mb[1] + 1) + (bd[3] - bd[1] + 1)) / 2.0)

                def heur_b(stb, _b=b, _w=want):
                    c = stb.get(_b)
                    if not c:
                        return 999.0
                    cc = centre(c)
                    return (abs(cc[0] - _w[0]) + abs(cc[1] - _w[1])) / float(unit_)
                gb = lambda stb, _b=b: goal_b(stb, _b)
                pb, sb = route(s, b, gb, heur_b, pusher_depth, 0, label + f" pusher S{b}")
                if pb is None:
                    continue
                # the strike, as the successors predict it with the pusher there
                s_after = next((n for a, n in succ(sb) if a == k), None)
                if s_after is None or s_after.get(who) == sb.get(who):
                    if verbose:
                        print(f"      [shove] {label}: S{b} at {_bbox(sb[b])} strike {k} on S{who} at {mb}: no strike predicted ({make_successors.whys.get(k)})")
                    continue
                rest2, s_end = route(s_after, who, goal, heur, depth_ - 1, pusher_depth, label + ">>")
                if rest2 is None:
                    continue
                cost = len(path) + len(pb) + 1 + len(rest2)
                if found is None or cost < found[0]:
                    found = (cost, path + pb + [k] + rest2, s_end, b)
                if b not in docked:
                    break                   # (a free pusher found: enough)
            if found is not None and (best is None or found[0] < best[0]):
                best = found
                if verbose:
                    print(f"      [shove] {label}: S{who} launch {mb[:2]} strike {k} by S{found[3]} -> {_bbox(s2[who])[:2]}: {found[0]} steps")
        if best is None:
            return None, None
        return best[1], best[2]

    p, _ = route(dict(start), sid, goal_t, lambda st: dist(st) / 3.0, depth, 1, "route")
    return p


def approach_and_push(tr, model, keys, pusher, o, d, k, step, f, cur_sel, verbose, presses=2):
    """Stand `pusher` on the far side of `o` from direction d (touching)
    and press k `presses` times.  Returns the number of steps or None
    when no approach exists."""
    hr, wc = o.dims
    mh, mw = pusher.dims
    want = (o.centre[0] - d[0] * (hr + mh) / 2.0, o.centre[1] - d[1] * (wc + mw) / 2.0)
    comps = tr.prev
    start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
    start["_sel"] = pusher.sid
    solids0 = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2}
    succ0 = make_successors(model, comps, solids0, keys, pusher.sid, want, frame=f, frozen=(o.sid,))

    def at_want(st, _m=pusher.sid, _w=want):
        a_ = st.get(_m)
        if not a_:
            return 999
        ca = centre(a_)
        return abs(ca[0] - _w[0]) + abs(ca[1] - _w[1])
    p0 = plan_general(start, succ0, lambda st: at_want(st) <= 1.0, heuristic=lambda st: at_want(st) / 3.0, max_states=20000, max_depth=30)
    if verbose:
        print(f"   push: S{pusher.sid}@{pusher.bbox} -> S{o.sid}@{o.bbox} along {d} want {want}: approach {p0}")
    if p0 is None:
        if verbose:
            print(f"      no approach: {getattr(plan_general, 'last', {})}; start successors {[a for a, _ in succ0(start)][:8]} why {getattr(make_successors, 'whys', None)} {[(s_.sid, s_.body, s_.size, s_.bbox, s_.tags) for s_ in tr.prev if 'hud' not in s_.tags]} sel_kinds {model.select_kinds} ksel {model.key_selected}")
        return None
    n = 0
    if pusher.sid != cur_sel:
        step(("select", pusher.sid)); n += 1
        if verbose:
            pu = next((s for s in tr.prev if s.sid == pusher.sid), None)
            print(f"      selected S{pusher.sid}: {pu.bbox if pu else None} marks {selected_hosts(tr.prev)}")
    for k_ in p0[:14]:
        step(k_); n += 1
        if verbose:
            pu = next((s for s in tr.prev if s.sid == pusher.sid), None)
            print(f"      approach {k_}: pusher {pu.bbox if pu else None} marks {selected_hosts(tr.prev)}")
    for _ in range(presses):
        step(k); n += 1
        if verbose:
            pu = next((s for s in tr.prev if s.sid == pusher.sid), None); oo = next((s for s in tr.prev if s.sid == o.sid), None)
            print(f"      pressed {k}: pusher {pu.bbox if pu else None} o {oo.bbox if oo else None} marks {selected_hosts(tr.prev)} contact {list(model.contact_seen)} stayed {model.stayed.get((k, o.sid))}")
    return n


def open_run(f, model, occupied, bbox, d, limit=25):
    """Cells of open ground beyond `bbox` along d (bounds, other things
    and known wall colours stop it)."""
    blocking = model.blocking()
    r0, c0, r1, c1 = bbox
    run = 0
    for n_ in range(1, limit):
        if d[0]:
            line = [((r1 if d[0] > 0 else r0) + d[0] * n_, c) for c in range(c0, c1 + 1)]
        else:
            line = [(r, (c1 if d[1] > 0 else c0) + d[1] * n_) for r in range(r0, r1 + 1)]
        if any(r < 0 or c < 0 or r >= f.shape[0] or c >= f.shape[1] or (r, c) in occupied or int(f[r, c]) in blocking for r, c in line):
            break
        run = n_
    return run


def best_lane(tr, model, me, d, f, tried=frozenset()):
    """Among the places the thing could stand (its own footprint slid
    across the axis perpendicular to d, on its key lattice), the one with
    the longest open run along d.  Returns (run, (r0, c0)) or None."""
    occupied = set()
    for s_ in tr.prev:
        if "terrain" not in s_.tags and "hud" not in s_.tags and s_.sid != me.sid:
            occupied |= set(s_.cells)
    blocking = model.blocking()
    steps_ = sorted({abs(v[0]) + abs(v[1]) for v in model.key_selected.values()} or {1})
    pitch = steps_[0]
    r0, c0, r1, c1 = me.bbox
    h, w = r1 - r0 + 1, c1 - c0 + 1
    best = None
    for off in range(-30, 31):
        if off % pitch:
            continue
        nr0, nc0 = (r0 + off, c0) if d[1] else (r0, c0 + off)
        cells = [(r, c) for r in range(nr0, nr0 + h) for c in range(nc0, nc0 + w)]
        if any(r < 0 or c < 0 or r >= f.shape[0] or c >= f.shape[1] or (r, c) in occupied or int(f[r, c]) in blocking for r, c in cells):
            continue
        run = open_run(f, model, occupied, (nr0, nc0, nr0 + h - 1, nc0 + w - 1), d)
        if (me.sid, d, (nr0, nc0)) in tried:
            continue            # an experiment already made
        if best is None or run > best[0] or (run == best[0] and abs(off) < abs(best[2])):
            best = (run, (nr0, nc0), off)
    return (best[0], best[1]) if best else None


def nudge_toward(tr, model, keys, sids, tcs, step, verbose, f=None):
    """For each mover still away from a marker: make it the marked thing
    (a click, when clicks select) and press the key toward the marker
    a few times; what happens is recorded like any fiddling."""
    import itertools
    n = 0
    # (only the selected-key laws that are translations push along an axis)
    ksel_ = {k_: v_ for k_, v_ in model.key_selected.items()
             if isinstance(v_, tuple) and len(v_) == 2 and all(isinstance(x_, (int, float)) for x_ in v_)}
    if not ksel_:
        return 0
    movers = [next((s for s in tr.prev if s.sid == sid), None) for sid in sids]
    movers = [m for m in movers if m is not None]
    # movers to markers: the assignment with the least total distance
    best = None
    for perm in itertools.permutations(range(len(tcs)), min(len(movers), len(tcs))):
        tot = sum(abs(movers[i].centre[0] - tcs[j][0]) + abs(movers[i].centre[1] - tcs[j][1]) for i, j in enumerate(perm))
        if best is None or tot < best[0]:
            best = (tot, perm)
    if best is None:
        return 0
    for i, j in enumerate(best[1]):
        me = movers[i]; sid = me.sid; t = tcs[j]
        d = (t[0] - me.centre[0], t[1] - me.centre[1])
        if verbose:
            print(f"   nudge: S{sid}@{me.bbox} -> {t} d={d}")
        if abs(d[0]) + abs(d[1]) < 1.5:
            continue
        marks = selected_hosts(tr.prev)
        sel = [h_ for h_, col in marks.items() if model.select_color is None or col == model.select_color]
        cur_sel = sel[0] if sel else None
        # PUSH IT THERE: when the thing can be pushed, another thing that
        # the keys move stands behind it and presses toward the marker
        if me.body in model.pushable and f is not None:
            axis = 0 if abs(d[0]) >= abs(d[1]) else 1
            dv = ((d[0] > 0) - (d[0] < 0) if axis == 0 else 0, (d[1] > 0) - (d[1] < 0) if axis == 1 else 0)
            k = max(ksel_, key=lambda k_: ksel_[k_][axis] * dv[axis])
            # SET UP THE SHOT: a shove carries the thing along open ground;
            # where it stands now the run toward the marker may be nil (a
            # wall beside it) — first walk it (it takes the keys when
            # clicked) to the lane with the longest run toward the marker
            if ksel_[k][axis] * dv[axis] > 0 and me.body in model.select_kinds:
                lane = best_lane(tr, model, me, dv, f, tried=model.tried_nudges)
                if lane is not None and lane[0] >= 3:
                    model.tried_nudges.add((sid, dv, lane[1]))
                if lane is not None and lane[0] >= 3 and lane[1] != (me.bbox[0], me.bbox[1]):
                    run, (lr, lc) = lane
                    want = ((lr + me.bbox[2] - me.bbox[0]) / 2.0 + lr / 2.0, (lc + me.bbox[3] - me.bbox[1]) / 2.0 + lc / 2.0)
                    want = (lr + (me.bbox[2] - me.bbox[0]) / 2.0, lc + (me.bbox[3] - me.bbox[1]) / 2.0)
                    comps = tr.prev
                    start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
                    start["_sel"] = sid
                    solids0 = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2}
                    succ0 = make_successors(model, comps, solids0, keys, sid, want, frame=f)

                    def at_lane(st, _m=sid, _w=want):
                        a_ = st.get(_m)
                        if not a_:
                            return 999
                        ca = centre(a_)
                        return abs(ca[0] - _w[0]) + abs(ca[1] - _w[1])
                    p0 = plan_general(start, succ0, lambda st: at_lane(st) <= 1.0, heuristic=lambda st: at_lane(st) / 3.0, max_states=20000, max_depth=30)
                    if verbose:
                        print(f"   lane: S{sid} -> {want} (run {run}): {p0 if p0 is None else len(p0)} steps")
                    if p0:
                        note(f"nudge: no plan under what is known — setting up the shot: S{sid} to the lane with the longest open run, then pushing")
                        if cur_sel != sid:
                            step(("select", sid)); n += 1
                        for k_ in p0[:14]:
                            step(k_); n += 1
                        me = next((s for s in tr.prev if s.sid == sid), me)
                        marks = selected_hosts(tr.prev)
                        sel = [h_ for h_, col in marks.items() if model.select_color is None or col == model.select_color]
                        cur_sel = sel[0] if sel else None
            ks_ = ksel_[k]
            if not (isinstance(ks_, tuple) and len(ks_) == 2 and all(isinstance(x_, (int, float)) for x_ in ks_)):
                continue            # (a selected-key law that is no translation — a turn — pushes nothing along an axis)
            if ks_[axis] * dv[axis] > 0:
                pushers = [s for s in tr.prev if s.sid != sid and (s.sid == cur_sel or s.body in model.select_kinds) and "hud" not in s.tags]
                pushers.sort(key=lambda s: (0 if s.sid == cur_sel else 1, abs(s.centre[0] - me.centre[0]) + abs(s.centre[1] - me.centre[1])))
                done = None
                for pu in pushers[:2]:
                    done = approach_and_push(tr, model, keys, pu, me, dv, k, step, f, cur_sel, verbose)
                    if done is not None:
                        n += done
                        break
                if done is not None:
                    continue
        if (not sel or sel[0] != sid) and me.body in model.select_kinds:
            step(("select", sid)); n += 1
        for axis in sorted((0, 1), key=lambda a: -abs(d[a])):
            if abs(d[axis]) < 1.5:
                continue
            best_k = max(ksel_, key=lambda k: ksel_[k][axis] * d[axis])
            if ksel_[best_k][axis] * d[axis] <= 0:
                continue
            for _ in range(4):
                before = {s.sid: s.centre for s in tr.prev}
                step(best_k); n += 1
                after = {s.sid: s.centre for s in tr.prev}
                if before.get(sid) == after.get(sid):
                    break           # refused: this way is shut
            break
    return n


def _coupled_aligns(hyps, tr, model):
    """Align readings whose movers (or targets) one key moves together:
    [(h, mover sid, target sid, axes)] — None when fewer than two couple."""
    res = []
    for h in hyps:
        if h.kind != "align" or not getattr(h, "t_cells", None) or getattr(h, "members", None) or "marks onto" in (h.note or ""):
            continue
        # (the two things by their KINDS and where the reading saw them — a
        # mark that rides a piston is a part; the arrow's old place may now
        # lie under what grew over it)
        cands = [s_ for s_ in tr.prev if s_.body == h.m_color and "hud" not in s_.tags and "terrain" not in s_.tags
                 and 0.5 * h.m_size <= s_.size <= 2 * h.m_size]
        if not cands:
            continue
        mv = min(cands, key=lambda s_: abs(s_.centre[0] - h.m_start[0]) + abs(s_.centre[1] - h.m_start[1]))
        tc = set(h.t_cells); tcen_ = centre(tc)
        tcol_ = getattr(h, "t_color", None)
        if tcol_ is None:
            m_ = re.search(r"C(\d+)\[", h.note or "")
            tcol_ = int(m_.group(1)) if m_ else None
        tgs = [s_ for s_ in tr.prev if s_.sid != mv.sid and "hud" not in s_.tags and "terrain" not in s_.tags
               and (tcol_ is None or s_.body == tcol_) and 0.5 * len(tc) <= s_.size <= 2 * len(tc)]
        tg = min(tgs, key=lambda s_: abs(s_.centre[0] - tcen_[0]) + abs(s_.centre[1] - tcen_[1])) if tgs else None
        axes = [False, False]
        for k in model.fixed.keys():
            for sid_ in (mv.sid, tg.sid if tg is not None else None):
                if sid_ is None:
                    continue
                e = model.fixed.moving(k).get(sid_)
                if e and e[0] == "translate":
                    axes[0] |= bool(e[1][0]); axes[1] |= bool(e[1][1])
                elif e and e[0] in ("extend", "retract"):
                    axes[0] |= e[1][0] in ("top", "bottom"); axes[1] |= e[1][0] in ("left", "right")
        if not any(axes):
            continue            # (a pair no key moves is not a pair to couple)
        res.append((h, mv.sid, tg.sid if tg is not None else None, tuple(axes)))
    # one reading per pair (a reading and its turned-round twin are one)
    seen = set(); out = []
    for item in res:
        key_ = frozenset((item[1], item[2]))
        if item[1] in seen or key_ in seen:
            continue
        seen.add(item[1]); seen.add(key_); out.append(item)
    if len(out) < 2:
        return None
    # the coupling: a key that moves things of two readings
    sids_of = [{m_} | ({t_} if t_ is not None else set()) for _, m_, t_, _ in out]
    coupled = False
    for k in model.fixed.keys():
        mv_ = set(model.fixed.moving(k))
        if sum(1 for ss in sids_of if ss & mv_) >= 2:
            coupled = True; break
    return out if coupled else None


def play_joint_mixed(tr, model, keys, mix, step, get_obs, lv0, verbose):
    """Every coupled reading satisfied at once: mover level with its target
    on the axes the mover has, planned over the keys' laws jointly; the
    plan is followed with the world checked each step, replanned on a
    divergence (three times)."""
    tol = 1.0
    sids = [m_ for _, m_, _, _ in mix]

    def pair_dist(state):
        tot = 0.0
        for h, m_, t_, axes in mix:
            cs = state.get(m_)
            if not cs:
                return 999.0
            c = centre(cs)
            tcen = centre(state[t_]) if (t_ is not None and state.get(t_)) else centre(h.t_cells)
            tot += (abs(c[0] - tcen[0]) if axes[0] else 0) + (abs(c[1] - tcen[1]) if axes[1] else 0)
        return tot

    def goal(state):
        for h, m_, t_, axes in mix:
            cs = state.get(m_)
            if not cs:
                return False
            c = centre(cs)
            tcen = centre(state[t_]) if (t_ is not None and state.get(t_)) else centre(h.t_cells)
            if (axes[0] and abs(c[0] - tcen[0]) > tol) or (axes[1] and abs(c[1] - tcen[1]) > tol):
                return False
        return True
    for attempt in range(8):
        comps = tr.prev
        if goal({s_.sid: frozenset(s_.cells) for s_ in comps}):
            return False, 0
        solids = set()
        keep = set(sids) | {t_ for _, _, t_, _ in mix if t_ is not None}
        for s_ in comps:
            if s_.sid in keep or "hud" in s_.tags or "terrain" in s_.tags or s_.size <= 2:
                continue
            solids.add(s_.sid)
        start = {s_.sid: frozenset(s_.cells) for s_ in comps if "hud" not in s_.tags}
        sel0 = [x_ for x_ in (model.selected_now(comps),) if x_ is not None]
        start["_sel"] = sel0[0] if sel0 else None
        h0, m0, t0_, _ = mix[0]
        tc0 = centre(start[t0_]) if (t0_ is not None and start.get(t0_)) else centre(h0.t_cells)
        succ = make_successors(model, comps, solids, keys, m0, tc0, frame=np.array(get_obs().frame)[-1])
        t0 = time.time()
        p = plan_general(start, succ, goal, heuristic=lambda st: pair_dist(st) / 3.0, max_states=60000, max_depth=60)
        if verbose:
            print(f"   joint mixed plan over {len(mix)} readings: {p if p is None else len(p)} steps ({time.time()-t0:.1f}s) last={getattr(plan_general, 'last', {})} plan={p[:12] if p else None}")
            print(f"   joint mixed: refused {sorted(model.refused_at)} standing {sorted(model.refused_now(np.array(get_obs().frame)[-1]))}; laws {[(k_, dict(model.fixed.moving(k_))) for k_ in model.fixed.keys() if isinstance(k_, tuple)][:10]}; movers {[(h_.note[:60], m_, next((x_.bbox for x_ in tr.prev if x_.sid == m_), None), t_, next((x_.bbox for x_ in tr.prev if x_.sid == t_), None), ax_) for h_, m_, t_, ax_ in mix]}")
        if p is None:
            return False, 0
        for i, k in enumerate(p):
            if verbose:
                print(f"   joint mixed step {i+1}/{len(p)} {k}: dist {pair_dist({s_.sid: frozenset(s_.cells) for s_ in tr.prev}):.1f}")
            pred = None
            before_st = {**{s_.sid: frozenset(s_.cells) for s_ in tr.prev if "hud" not in s_.tags}, "_sel": start["_sel"]}
            for act, new in succ(before_st):
                if act == k:
                    pred = new; break
            if pred is None:
                # (the plan's next press is not one the laws allow from where
                # the world stands now: the world diverged unseen — replan)
                if verbose:
                    print(f"   joint mixed step {i+1} {k}: no longer allowed from here; replanning")
                break
            kinds_b = {s_.sid: s_.body for s_ in tr.prev}
            step(k)
            if get_obs().levels_completed > lv0:
                return True, i + 1
            if pred is not None:
                now = {s_.sid: frozenset(s_.cells) for s_ in tr.prev}
                if any(sid in pred and (now.get(sid) is None or off_centre(now[sid], pred[sid])) for sid in sids):
                    # (A REFUSED STEP IS REMEMBERED where it was refused: every
                    # thing the key was to move, from the place it stood — the
                    # pump at its limit; the next plan goes round it)
                    moved_any = any(not isinstance(q_, str) and q_ in now and now[q_] != before_st.get(q_) for q_ in pred)
                    if not moved_any:
                        for q_ in pred:
                            if isinstance(q_, str) or q_ not in before_st or pred[q_] == before_st[q_]:
                                continue
                            b0 = _bbox(before_st[q_]); b1 = _bbox(pred[q_])
                            v_ = (b1[0] - b0[0], b1[1] - b0[1])
                            try:
                                # (what stood where the thing would have gone: its own
                                # footprint there, not a six-by-six block — the refusal
                                # must not lapse because a neighbour pumped)
                                model.refuse((kinds_b.get(q_), b0[0], b0[1], v_), size=(b0[2] - b0[0] + 1, b0[3] - b0[1] + 1))
                            except Exception:
                                pass
                        if verbose:
                            print(f"   joint mixed step {i+1} {k}: refused — remembered for {[q_ for q_ in pred if not isinstance(q_, str) and q_ in before_st and pred[q_] != before_st[q_]]}")
                    if verbose:
                        print(f"   joint mixed step {i+1} {k}: a mover not where predicted; replanning")
                    break
        else:
            if goal({s_.sid: frozenset(s_.cells) for s_ in tr.prev}):
                return False, len(p)
    return False, 0


def play_joint(env, tr, model, keys, h, m_starts, tcs, step, get_obs, lv0, verbose, nudge=True):
    """All movers of the kind at markers of the kind, at once."""
    comps = tr.prev
    # the movers: one thing of the kind per reading, nearest to where the
    # reading saw it (each thing used once — the fiddling moved them)
    cands = [s for s in comps if s.body == h.m_color and 0.5 * h.m_size <= s.size <= 2 * h.m_size]
    pairs = sorted((abs(s.centre[0] - ms[0]) + abs(s.centre[1] - ms[1]), i, s.sid) for i, ms in enumerate(m_starts) for s in cands)
    used_i, sids = set(), []
    for d, i, sid in pairs:
        if i in used_i or sid in sids:
            continue
        used_i.add(i); sids.append(sid)
    if len(sids) < 2:
        return False, 0
    tcells = set(getattr(h, "t_cells", ()))

    def jdist(state):
        pos = []
        for sid in sids:
            cs = state.get(sid)
            if not cs:
                return 999.0
            pos.append(centre(cs))
        left = list(tcs); tot = 0.0
        for p_ in pos:
            if not left:
                break
            j = min(range(len(left)), key=lambda i: abs(p_[0] - left[i][0]) + abs(p_[1] - left[i][1]))
            tot += abs(p_[0] - left[j][0]) + abs(p_[1] - left[j][1]); left.pop(j)
        return tot
    for attempt in range(5):
        comps = tr.prev
        solids = set()
        for s_ in comps:
            near = any(abs(s_.centre[0] - t[0]) + abs(s_.centre[1] - t[1]) <= 4 for t in tcs)
            if (near and s_.size <= 20) or "hud" in s_.tags or "terrain" in s_.tags or s_.size <= 2:
                continue
            solids.add(s_.sid)
        marks = selected_hosts(comps)
        sel0 = [x_ for x_ in (model.selected_now(comps),) if x_ is not None]
        start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
        start["_sel"] = sel0[0] if sel0 else None
        # the successor function needs one mover for relative clicks; use the first
        succ = make_successors(model, comps, solids, keys, sids[0], tcs[0], frame=np.array(get_obs().frame)[-1])
        t0 = time.time()
        p = plan_general(start, succ, lambda st: jdist(st) < 1.0, heuristic=lambda st: jdist(st) / 3.0)
        note(f"joint plan for {len(sids)} movers: {'none under what is known' if p is None else str(len(p)) + ' steps'}")
        if verbose:
            print(f"   joint plan {len(sids)} x {h.rule()}: {p if p is None else len(p)} steps ({time.time()-t0:.1f}s) sel={start['_sel']} plan={p[:10] if p else None} last={getattr(plan_general, 'last', {})} refused_at={sorted(model.refused_at)[:12]}")
        if p is None:
            # NO WAY UNDER WHAT IS KNOWN: try the obvious anyway — each
            # mover, pressed toward its marker (a wall may not be one)
            if attempt >= 3 or not model.key_selected or not nudge:
                return False, 0
            n_before = len(model.push_vec) + len(model.fixed.overlap_ok)
            nudged = nudge_toward(tr, model, keys, sids, tcs, step, verbose, f=np.array(get_obs().frame)[-1])
            if verbose:
                print(f"   nudge: {nudged} presses; push_vec={model.push_vec}")
            if get_obs().levels_completed > lv0:
                return True, nudged
            if nudged == 0:
                return False, 0
            continue
        diverged = False
        for i, k in enumerate(p):
            pred = None
            note(f"joint plan step {i+1}/{len(p)} ({k}): {len(sids)} movers to their markers")
            marks_now = selected_hosts(tr.prev)
            sel_now = [h_ for h_, col in marks_now.items() if model.select_color is None or col == model.select_color]
            for act, new in succ({**{s.sid: frozenset(s.cells) for s in tr.prev if "hud" not in s.tags}, "_sel": sel_now[0] if sel_now else None}):
                if act == k:
                    pred = new; break
            step(k)
            if get_obs().levels_completed > lv0:
                return True, i + 1
            if pred is not None:
                now = {s.sid: frozenset(s.cells) for s in tr.prev}
                if any(sid in pred and (now.get(sid) is None or off_centre(now[sid], pred[sid])) for sid in sids):
                    diverged = True
                    if verbose:
                        print(f"   joint step {i+1} {k}: a mover not where predicted; replanning")
                        for sid in sids:
                            print(f"      S{sid}: predicted {_bbox(pred[sid]) if pred.get(sid) else None} actual {_bbox(now[sid]) if now.get(sid) else None}")
                        print("      events:", [(e.kind, e.sid, e.data) for e in tr.last_events if e.kind != 'moved' or e.sid in sids][:8])
                    break
        if not diverged:
            return False, len(p)
    return False, 0


class _Died(Exception):
    pass


def play_level(env, obs, budget, confirmed=(), verbose=False, carry=None):
    """A level, played; a death ends it at once (the game restarts at its
    first level) and is reported as such."""
    play_level.steps = 0
    try:
        return _play_level_inner(env, obs, budget, confirmed, verbose, carry)
    except _Died as d_:
        return False, play_level.steps, "died", d_.args[0]


play_level = PerThread(play_level)



def _objs_by_units(objs, unit_form, elastic=()):
    """THE UNIT OF A KIND, in the first look too: a component of one colour
    that is a whole row or column of the colour's units (the tracker's,
    carried across levels) is read as that many things — the readings then
    name a block, not the pair."""
    if not unit_form or _ENV.get("NO_UNITS") or _ENV.get("NO_UNITS_LOOK"):
        return objs
    from agent.perception import Obj as _Obj
    out = []
    nid = max((o.id for o in objs), default=-1) + 1
    for o in objs:
        u_ = unit_form.get(o.color) if o.color not in set(elastic or ()) else None
        b_ = o.bbox; h_, w_ = b_[2] - b_[0] + 1, b_[3] - b_[1] + 1
        if u_ is None or h_ * w_ != len(o.cells):
            out.append(o); continue
        if (h_, w_) == tuple(u_):
            # (one whole unit standing alone is left to the readings as before:
            # marking it kept two crates' cores apart in wa30's pen and the
            # reading order changed by one press at a zero-margin clock)
            out.append(o); continue
        if h_ == u_[0] and w_ % u_[1] == 0 and w_ // u_[1] >= 2:
            boxes_ = [(b_[0], b_[1] + i * u_[1], b_[2], b_[1] + (i + 1) * u_[1] - 1) for i in range(w_ // u_[1])]
        elif w_ == u_[1] and h_ % u_[0] == 0 and h_ // u_[0] >= 2:
            boxes_ = [(b_[0] + i * u_[0], b_[1], b_[0] + (i + 1) * u_[0] - 1, b_[3]) for i in range(h_ // u_[0])]
        else:
            out.append(o); continue
        for i, bx_ in enumerate(boxes_):
            cells_ = frozenset((r, c) for r in range(bx_[0], bx_[2] + 1) for c in range(bx_[1], bx_[3] + 1))
            rs_ = [r for r, _ in cells_]; cs_ = [c for _, c in cells_]
            no_ = _Obj(id=(o.id if i == 0 else nid), color=o.color, cells=cells_, bbox=bx_,
                       centroid=(sum(rs_) / len(rs_), sum(cs_) / len(cs_)))
            try:
                no_.unit = True
            except Exception:
                pass
            out.append(no_)
            if i > 0:
                nid += 1
    return out

def _play_level_inner(env, obs, budget, confirmed=(), verbose=False, carry=None):
    play_level.ledger = {"divs": [], "reading": None, "plan_len": 0, "hyps_tried": []}      # (the attempt ledger: agent/core/ledger.py)
    f = np.array(obs.frame)[-1]
    bg = background_color(f)
    tr = SceneTracker() if SCENE else EntityTracker(); model = GeneralModel(); model.verbose = verbose
    if (carry or {}).get("unit_form"):
        tr.unit_form = dict(carry["unit_form"])
    if (carry or {}).get("elastic"):
        tr.elastic = set(carry["elastic"])
    # (what counts: goals from what the presses make happen — the record of
    # this level carries across its attempts, as the laws do)
    _pc = (carry or {}).get("progress_obj")
    progress = _pc[1] if (_pc and _pc[0] == obs.levels_completed) else _Progress()
    play_level.progress = progress
    play_level._twin_tried = False
    progress._filled = False        # (the fill is tried once per attempt; the record stays)
    progress._merged = False
    if (carry or {}).get("progress_merges") and not progress.merges:
        progress.merges = dict(carry["progress_merges"][0]); progress.reach = int(carry["progress_merges"][1])
        progress.merges_carried = True
    model._frame_now = f; model._frame = f
    sprites = read_frame(f, bg)
    if SCENE:
        tr.update(f, bg)
    else:
        tr.update(read_components(f, bg))
    if getattr(play_level, "rec", None):
        play_level.rec[-1]["seen"] = _seen(tr, model)          # (the first look, before any press)
    model.read_materials(tr.prev)                              # (what stands on a material at the first look passes it)
    # WHAT THE LAST LEVEL TAUGHT opens this one: laws by kind seeded on
    # this level's things of those kinds, to be confirmed, not relearned
    model.coined = list((carry or {}).get("coined") or [])      # (what the floor's wins were read as: kind, mover colour, target colour, dims, relation)
    if carry:
        model.seed(carry, tr.prev)
        if carry.get("won"):
            confirmed = tuple(confirmed) + ((None, None, None, carry["won"][0], carry["won"][4]),)
    with_derived(tr, model)
    lv0 = obs.levels_completed
    size_first = {s_.sid: s_.size for s_ in tr.prev}
    first_cells_ = {s_.sid: frozenset(s_.cells) for s_ in tr.prev}       # (where everything stood at the level's start)
    clock = {"deaths": 0, "per_life": None, "life_start": 0}

    def clock_left():
        """Actions left on this life: from the clock's known length once a
        life was lost, else from the counter's rate."""
        if clock["per_life"]:
            return clock["per_life"] - (steps - clock["life_start"])
        for sid_ in set(model.counters) | {x.sid for x in tr.prev if "bar" in x.tags}:
            s_ = tr.get(sid_)
            if s_ is not None and sid_ in size_first and steps > 0:
                rate = (size_first[sid_] - s_.size) / float(steps)
                if rate > 0:
                    return s_.size / rate
        return None
    play_level.clock_left = clock_left
    play_level.phase = "first look: reading the frame"
    play_level.trail = []
    play_level.won_now = False
    play_level.t0 = time.time()
    play_level.journal = _journal.Journal(_frame_key(obs))
    play_level.first_frame = np.array(obs.frame)[-1].copy()
    _claims.reset((carry or {}).get("claims"))
    import agent.core.planner as _planner_mod
    _planner_mod.time_left = lambda: _level_seconds() - (time.time() - play_level.t0)
    play_level.scrolls = 0
    play_level.died = False
    play_level.cheap_probe = False
    play_level.start_world = None
    play_level.fiddle_clicked = set()
    if getattr(play_level, "rec", None) is not None:
        play_level.rec.append({"i": 0, "action": None, "click": None, "frame": _hex(f), "events": [], "phase": play_level.phase,
                               "lv": int(obs.levels_completed), "reset": False, "seen": None})
    acts = [GameAction.from_id(i) for i in obs.available_actions if i != 0]
    keys = [a.name for a in acts if a.name not in ("ACTION6", "RESET")]
    clickable = any(a.name == "ACTION6" for a in acts)
    objs = _objs_by_units(segment(f, bg, mask=None), getattr(tr, "unit_form", None), getattr(tr, "elastic", ()))
    try:
        pf = detect_playfield(f)
    except Exception:
        pf = None
    hud_cells = frozenset(q for s_ in tr.prev if "hud" in s_.tags for q in s_.cells)
    # (what the laws between things derive is read too: an image that is
    # not on the screen yet is still the thing a reading may name)
    for d_ in [s_ for s_ in tr.prev if "derived" in s_.tags]:
        from agent.perception import Obj as _Obj
        rs_ = [r for r, _ in d_.cells]; cs_ = [c for _, c in d_.cells]
        objs.append(_Obj(id=90000 + len(objs), color=int(d_.body), cells=frozenset(d_.cells), bbox=(min(rs_), min(cs_), max(rs_), max(cs_)),
                         centroid=(sum(rs_) / len(rs_), sum(cs_) / len(cs_))))
        objs[-1].derived = True
    # THE READINGS come from the relation language, weighted by what
    # earlier levels taught about the cues
    rweights = RelationWeights((carry or {}).get("relation_weights"))
    # A READING PURSUED TO ITS END WITH NOTHING HAPPENING IS NOT PURSUED FIRST
    # AGAIN — on this level's next attempt or the next level (tw01: "the player
    # fits the helper's shape", two like squares, walked to its end on every
    # attempt of every level, seventeen presses each time).  The record is by
    # rule, mover kind and target kind; it orders, it does not forbid.
    failed_readings = Counter((carry or {}).get("failed_readings") or {})
    play_level.failed_readings = failed_readings

    def _reading_id(h_):
        tcol_ = getattr(h_, "t_color", None)
        if tcol_ is None:
            m_ = re.search(r"C(\d+)\[", getattr(h_, "note", "") or "")
            tcol_ = int(m_.group(1)) if m_ else None
        return (h_.rule(), int(h_.m_color) if h_.m_color is not None else None, tcol_)
    play_level._reading_id = _reading_id
    hyps = read_frame_goals(objs, 64, 64, playfield=pf, hud_cells=hud_cells, weights=rweights.get())
    # THE READER SEAM (agent/reader.py): a reader set on play_level — a
    # language model, or anything that answers in the relation language —
    # is asked first, with the last attempt's failure report when there is
    # one; its readings go before the relations' and its laws are claims
    # the ledger holds it to
    if getattr(play_level, "reader", None) is not None:
        try:
            hr_, laws_ = play_level.reader.read(objs, f, tr.prev, model, report=(carry or {}).get("last_report"), playfield=pf, hud_cells=hud_cells)
            if hr_:
                note(f"the reader says: {[h_.note for h_ in hr_[:3]]}")
                if verbose:
                    print(f"   reader: {hr_}")
            hyps = list(hr_) + [h for h in hyps]
            _reader.apply_laws(laws_, model, tr.prev, note)
        except Exception as ex_:
            print(f"   reader failed: {ex_!r}", flush=True)
    play_level.rweights = rweights
    play_level.last = {"model": model, "tr": tr, "hyps": hyps, "keys": keys, "clickable": clickable}
    conf_rel = {(r, b) for _, _, _, r, b in confirmed}
    hyps.sort(key=lambda h: (0 if (h.rule(), h.mover_bigger()) in conf_rel else 1, -h.prio))
    if verbose:
        for h in hyps[:4]:
            print("   hyp", h)
    steps = 0

    def step(key):
        nonlocal obs, steps, f
        before = tr.prev
        # THE KEYS OF THE GAME ARE WHAT THE GAME OFFERS: a press the engine
        # does not list is not made (wa30 clicked ten times in a game with no
        # click; the clock ran on them in the ledger's eyes, not the game's)
        try:
            avail_ = {int(i_) for i_ in (getattr(obs, "available_actions", None) or [])}
            if avail_ and not _ENV.get("PRESS_ANY"):
                want_ = 6 if isinstance(key, tuple) else int(GameAction[key].value)
                if want_ not in avail_:
                    if verbose:
                        print(f"   not offered: {key} (the game offers {sorted(avail_)}); not pressed")
                    return
        except Exception:
            pass
        if _ENV.get("PRESS_DEBUG"):
            import traceback as _tb
            _st = [fr.name for fr in _tb.extract_stack()[-6:-1]]
            print(f"   [press {steps + 1}] {key} from {' < '.join(reversed(_st))}")
        play_level.last["comps"] = before       # the level as it was before the last action (a win loads the next level)
        f_before = f
        click = None
        if isinstance(key, tuple):
            if key[0] == "select":
                s_ = next((s for s in tr.prev if s.sid == key[1]), None)
                if s_ is None:
                    return tr.prev          # the thing to select is gone: nothing to press
                click = click_point(s_)
            else:
                click = key[1]
            click = (max(0, min(63, int(click[0]))), max(0, min(63, int(click[1]))))
            a = GameAction.ACTION6; data = {"x": int(click[1]), "y": int(click[0])}
            a.set_data(data); obs = env.step(a, data=data)
        else:
            obs = env.step(GameAction[key])
        steps += 1
        play_level.steps = steps
        f = np.array(obs.frame)[-1]
        try:
            tr.last_click = tuple(click) if click is not None else None     # (the pointer, for the tracker's exchange reading)
        except Exception:
            pass
        if _ENV.get("FRAME_DUMP"):
            os.makedirs(os.environ["FRAME_DUMP"], exist_ok=True)
            n_ = len(os.listdir(os.environ["FRAME_DUMP"]))
            np.save(os.path.join(os.environ["FRAME_DUMP"], f"{n_:04d}_{obs.levels_completed}_{key if not isinstance(key, tuple) else 'click_%d_%d' % tuple(click)}.npy"), f)
        # THE WINNING PRESS IS A PRESS LIKE ANY OTHER: the engine returns the
        # level's last frame before the next level's first; what the press
        # did there (a block pushed, slid into the pen) is observed as a law
        # before the next level is looked at (tw03 L1: the push that won was
        # the only push the level showed, and L2 opened not knowing a push)
        try:
            fr_all_ = np.array(obs.frame)
            if obs.levels_completed > lv0 and fr_all_.ndim == 3 and len(fr_all_) >= 2 and SCENE:
                f_win_ = fr_all_[0]
                if not np.array_equal(f_win_, f_before) and not np.array_equal(f_win_, f):
                    tr.update(f_win_, bg); cur_win_ = with_derived(tr, model)
                    if not _mover_gone(model, cur_win_, obs):
                        model.observe(key if not isinstance(key, tuple) else ("click", click), click, before, cur_win_, frame=f_before, frame_after=f_win_)
                        if verbose:
                            print(f"   the winning press observed on the win frame: {key} -> {dict(model.fixed.moving(key if not isinstance(key, tuple) else ('click', click)))} pushable {sorted(model.pushable)} push_vec {dict(model.push_vec)}")
        except Exception as ex_:
            if verbose:
                print(f"   (the winning press on the win frame: {ex_!r})")
        # THE PATH WITHIN A PRESS: the body's place in every frame of the press
        try:
            fr_all_ = np.array(obs.frame)
            if fr_all_.ndim == 3 and len(fr_all_) >= 2 and model.active and not isinstance(key, tuple):
                me_ = max((s_ for s_ in before if s_.sid in model.active and s_.sid < 1000), key=lambda s_: s_.size, default=None)
                if me_ is not None:
                    model.observe_path(key, np.concatenate([f_before[None], fr_all_], axis=0), me_.body, me_.size)
        except Exception as ex_:
            if verbose:
                print(f"   (the path within the press: {ex_!r})")
        if SCENE:
            tr.update(f, bg); cur = with_derived(tr, model)
            # A BODY STAYS TOGETHER: things that always move together are
            # adjacent; a member the tracker placed far from the rest was
            # misread — the thing of its colour and size beside the rest
            # is it (a put-back after a death fools the reading)
            if model.active and len(model.active) >= 2:
                mem = [x for x in cur if x.sid in model.active and x.sid < 1000]
                for m in mem:
                    others = [o for o in mem if o.sid != m.sid]
                    if not others:
                        continue
                    def gap(a, b):
                        return max(b.bbox[0] - a.bbox[2], a.bbox[0] - b.bbox[2], b.bbox[1] - a.bbox[3], a.bbox[1] - b.bbox[3])
                    if min(gap(m, o) for o in others) <= 2:
                        continue
                    if _ENV.get("NOTE_DEBUG"):
                        print(f"   body check: S{m.sid} {m.body}/{m.size}@{m.bbox} far from {[(o.sid, o.bbox) for o in others]}; like things {[(x.sid, x.body, x.size, x.bbox, sorted(x.tags)) for x in cur if x.body == m.body and x.sid < 1000]}")
                    worn_ = set(model.kinds_worn.get(m.sid, ())) | {m.body}      # (the misread may have recoloured it)
                    cand = [x for x in cur if x.sid not in model.active and x.sid < 1000 and x.body in worn_
                            and 0.5 * m.size <= x.size <= 2 * m.size and "hud" not in x.tags and "terrain" not in x.tags
                            and min(gap(x, o) for o in others) <= 2]
                    if len(cand) == 1 and tr.swap_ids(m.sid, cand[0].sid):
                        note(f"S{m.sid} was read far from its body: the like thing beside it is S{m.sid}")
                        cur = with_derived(tr, model)
                        break
        else:
            cur = read_components(f, bg)
            tr.update(cur)
        if obs.levels_completed <= lv0 and obs.state != GameState.GAME_OVER:
            cur = _merged_mover(tr, model, key, before, cur)
        # (a won level's frame after is the NEXT level: it teaches nothing
        # about this key — the mover "moved" to its new start, and the
        # ground under that is not ground this key allowed)
        play_level.won_now = obs.levels_completed > lv0
        if obs.levels_completed <= lv0 and not _mover_gone(model, cur, obs):
            # (a lost level's frame with the mover gone from it is not
            # physics — the key would read as doing nothing; the death is
            # noted below.  With the mover still shown where the press put
            # it, the press taught its law as any other)
            scroll_ = next((e.data["vec"] for e in tr.last_events if e.kind == "scrolled"), None)
            # (a press that split a thing: the fiddler reads the frame before with
            # the piece as its own thing — see GeneralModel.observe)
            model._splits = [(e.data["split_from"], e.sid, (int(round(e.data["vec"][0])), int(round(e.data["vec"][1]))))
                             for e in (tr.last_events or []) if e.kind == "born" and e.data.get("split_from") is not None and e.data.get("vec") is not None]
            # (A PUT-BACK IS NO READING OF THE KEY: the controlled things set
            # back to the level's start on this press — a jump no step makes —
            # teach the key nothing; the press is read for its hazard below.
            # m0r0 L2: the jump home read as a long translate contradicted the
            # mirrored body's law, the plan then moved one body of two, and
            # the one it did not move walked into the checker twelve times)
            model._put_back_now = False
            try:
                if isinstance(key, str) and model.active and obs.state != GameState.GAME_OVER:
                    act_now_ = {x.sid: frozenset(x.cells) for x in cur if x.sid in model.active and x.sid < 1000}
                    act_bef_ = {x.sid: frozenset(x.cells) for x in before if x.sid in model.active and x.sid < 1000}
                    if act_now_ and all(first_cells_.get(sid_) == cs_ for sid_, cs_ in act_now_.items()) \
                            and any(first_cells_.get(sid_) != act_bef_.get(sid_) for sid_ in act_now_):
                        for sid_, cs_ in act_now_.items():
                            b0_ = act_bef_.get(sid_)
                            if not b0_:
                                continue
                            d_ = (_bbox(cs_)[0] - _bbox(b0_)[0], _bbox(cs_)[1] - _bbox(b0_)[1])
                            law_ = model.fixed.moving(key).get(sid_)
                            m_ = max(abs(law_[1][0]), abs(law_[1][1])) if law_ and law_[0] == "translate" else 0
                            if max(abs(d_[0]), abs(d_[1])) > m_ + 1:
                                model._put_back_now = True
            except Exception:
                model._put_back_now = False
            model.observe(key if not isinstance(key, tuple) else ("click", click), click, before, cur, frame=f_before, frame_after=f, scroll=scroll_)
            model._splits = []
            model._put_back_now = False
            # THE LEVER AND THE TWIN (agent/core/lever): a thing that moves when
            # the body stands somewhere and back when it leaves; a key that
            # puts the body back at the start and a twin that re-walks its record
            if not _ENV.get("NO_LEVER") and scroll_ is None:
                # (a screen that scrolled moved everything: no lever, no rewind is
                # read off a press that moved the screen — bp35's arrows)
                try:
                    if not hasattr(model, "lever"):
                        from agent.core.lever import LeverModel as _LM, TwinModel as _TM
                        model.lever = _LM(); model.twin = _TM()
                    body_sids_ = {sid_ for sid_ in (model.active or ()) if sid_ < 1000}
                    n_lev_ = len(model.lever.levers)
                    model.lever.observe(before, cur, body_sids_, kind, frame_before=f_before, frame_after=f)
                    conf_ = model.twin.confirmed
                    law_v_ = None
                    try:
                        for sid_ in body_sids_:
                            e_ = model.fixed.moving(key).get(sid_) if isinstance(key, str) else None
                            if e_ and e_[0] == "translate":
                                law_v_ = tuple(e_[1]); break
                    except Exception:
                        law_v_ = None
                    start_ = next((first_cells_.get(sid_) for sid_ in body_sids_ if first_cells_.get(sid_)), None)
                    model.twin.observe(key if not isinstance(key, tuple) else ("click", click), before, cur, body_sids_, kind, law_vec=law_v_, start=start_)
                    if verbose and len(model.lever.levers) > n_lev_:
                        print(f"   lever: {model.lever.levers} (pads {list(model.lever.pads)[:4]})")
                    if verbose and model.twin.confirmed and not conf_:
                        print(f"   twin: {model.twin.describe()}")
                    # (a rewind key has no law on the things: what it did to the
                    # body is the jump home, not a step — the plan must not walk
                    # by it)
                    if model.twin.rewind_key is not None:
                        if model.twin.rewind_key in model.fixed.eff:
                            model.fixed.eff[model.twin.rewind_key].clear()
                        # (and no plan walks by it: a rewind is pressed on purpose,
                        # by the twin protocol, never as a step)
                        model.undo_keys.add(model.twin.rewind_key)
                    if _ENV.get("TWIN_DEBUG"):
                        print(f"   [twin] {key}: {model.twin.describe()}; lever cands {model.lever.cands} levers {model.lever.levers}")
                except Exception as ex_:
                    if verbose:
                        print(f"   (lever/twin failed: {ex_!r})")
            try:
                n_merge_ = getattr(progress, "merge_events", 0)
                pev_ = progress.observe(key if not isinstance(key, tuple) else ("click", click), click, before, cur, model)
                if pev_ and verbose:
                    print(f"   progress: {[(e_['score_sid'], e_['score_kind'], e_['how'], e_['cause'][:2]) for e_ in pev_][:4]}")
                # (A THING MADE BY A MERGE IS A NEW THING TO THE LAWS: the tracker
                # keeps a name through the morph — the dot's name on the block —
                # and what the keys were seen to do to the dot is not known of
                # the block; its laws by name are dropped, to be learned afresh)
                if getattr(progress, "merge_events", 0) > n_merge_ and getattr(progress, "last_merge_sid", None) is not None:
                    for k2_ in list(model.fixed.eff):
                        model.fixed.eff[k2_].pop(progress.last_merge_sid, None)
            except Exception as ex_:
                if verbose:
                    print(f"   (progress observe failed: {ex_!r})")
            if obs.state != GameState.GAME_OVER:
                model.note_alive(cur)
            if scroll_ is not None:
                # the readings' places are of the screen: shifted with it
                u_ = (int(scroll_[0]), int(scroll_[1]))
                for h_ in play_level.last.get("hyps", []) if hasattr(play_level, "last") and isinstance(play_level.last, dict) else []:
                    _shift_hyp(h_, u_)
                play_level.scrolls = getattr(play_level, "scrolls", 0) + 1
                note(f"the screen scrolled by {u_}: places remembered are shifted with it")
        model._frame_now = f
        # WHAT HAPPENS BETWEEN THE PRESS AND THE REST IS PART OF THE ANSWER:
        # a press that returns several frames and leaves a thing where it
        # stood may have sent it on a journey and put it back — a launch.
        # And a click that recolours a small mark in place (nothing else
        # changing) flips a toggle
        if obs.levels_completed <= lv0 and obs.state != GameState.GAME_OVER:
            try:
                fr_all = np.array(obs.frame)
                hud_rows_ = {r for s_ in before if "hud" in s_.tags or "bar" in s_.tags for r, _ in s_.cells}
                if fr_all.ndim == 3 and len(fr_all) >= 3:
                    # (a kind strewn many times over the board is the board:
                    # its tiles do not journey — a thing passing over one is
                    # not the tile found elsewhere)
                    n_kind_ = Counter((x.body, x.size, x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) for x in before if x.sid < 1000)
                    movers_ = [x for x in before if x.sid < 1000 and n_kind_[(x.body, x.size, x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1])] < 8]
                    j_ = read_journeys(fr_all, movers_, hud_rows=hud_rows_)
                    # (what the press painted and undid, and where it blinked
                    # at the end: a run's telling — see progress.read_telling)
                    try:
                        from agent.core.progress import read_telling as _read_telling
                        tell_ = _read_telling(fr_all, before, f_before, hud_rows=hud_rows_) if not _ENV.get("NO_TELLING") else None
                        if tell_:
                            progress.note_telling(key if not isinstance(key, tuple) else ("click", click), tell_)
                            if verbose:
                                print(f"   telling under {key if not isinstance(key, tuple) else 'click'}: painted {tell_['painted'][:8]} blink {[(min(c_)[0], min(c_)[1], len(c_)) for c_ in tell_['blink']][:4]} restored {tell_['restored']} frames {tell_['n']} hud rows {sorted(hud_rows_)[:12]}")
                    except Exception as ex_:
                        if verbose:
                            print(f"   (telling failed: {ex_!r})")
                    # (a TRACE drawn and taken back — ink from a source over the
                    # board — is a run's telling: a goal and a test, and a law
                    # the plan can run itself; agent/core/runkey)
                    try:
                        if not _ENV.get("NO_RUNKEY"):
                            from agent.core.runkey import read_run as _read_run
                            run_ = _read_run(fr_all, before, f_before, hud_rows=hud_rows_, bg=bg)
                            if run_ and len(run_["trace"]) >= 8 and run_["restored"]:
                                progress.note_run(key if not isinstance(key, tuple) else ("click", click), run_, f_before.copy(), before)
                                if verbose:
                                    print(f"   run under {key if not isinstance(key, tuple) else 'click'}: ink {run_['colour']} from S{run_['source']} {run_['direction']}, {len(run_['trace'])} cells; painted {run_['painted'][:6]} entered {run_['entered'][:6]} flash {run_['flash'][:6]} restored {run_['restored']}")
                    except Exception as ex_:
                        if verbose:
                            print(f"   (run reading failed: {ex_!r})")
                    if j_:
                        k_id = key if not isinstance(key, tuple) else ("click", click)
                        model.journeys.append((k_id, j_, f_before.copy()))
                        for sid_, st_ in j_.items():
                            note(f"a launch: S{sid_} travelled {st_} and was put back (read from the frames between)")
                        if verbose:
                            print(f"   journey under {k_id}: {j_}")
                if isinstance(key, tuple) and key[0] == "click" and fr_all.ndim == 3 and len(fr_all) == 1:
                    ch_ = [(int(r), int(c)) for r, c in np.argwhere(f != f_before) if int(r) not in hud_rows_]
                    if 1 <= len(ch_) <= 16:
                        from_ = {int(f_before[r, c]) for r, c in ch_}; to_ = {int(f[r, c]) for r, c in ch_}
                        near_ = any(abs(r - click[0]) <= 2 and abs(c - click[1]) <= 2 for r, c in ch_)
                        if len(from_) == 1 and len(to_) == 1 and near_:
                            a_, b_ = next(iter(from_)), next(iter(to_))
                            model.flips[a_] = b_
                            model.toggle_cells |= set(ch_)
                            if verbose:
                                print(f"   toggle: click {click} flipped {len(ch_)} cells {a_} -> {b_}")
                # A LEAP: the click before was on a thing; this click, two
                # pitches away over its like neighbour, moved it there and
                # took the neighbour away (select, then point)
                if isinstance(key, tuple) and key[0] == "click" and model.leap is None:
                    lp_ = read_leap(before, cur, model._last_click, click)
                    if lp_ is not None:
                        model.leap = lp_
                        note(f"a leap: the thing selected by the click before leapt over its like neighbour to the place clicked, and the neighbour is gone (pitch {lp_['pitch']})")
                        if verbose:
                            print(f"   leap read: {lp_}")
                if isinstance(key, tuple) and key[0] == "click":
                    model._last_click = click
            except Exception as ex_:
                if verbose:
                    print(f"   journey reading failed: {ex_!r}")
        # A PRESUMED SELECTION IS TESTED BY ITS CLICK: the key pressed after
        # selecting a thing of a presumed kind must move it; if it moved
        # something else (or nothing), the kind is not selectable
        if isinstance(key, tuple) and key[0] == "select" and obs.levels_completed <= lv0:
            s_ = next((x for x in before if x.sid == key[1]), None)
            play_level.presume_test = (key[1], s_.body) if s_ is not None and s_.body in getattr(model, "select_presumed", ()) and key[1] not in (model.active or ()) else None
        elif not isinstance(key, tuple) and getattr(play_level, "presume_test", None) is not None and obs.levels_completed <= lv0:
            psid_, pkd_ = play_level.presume_test
            play_level.presume_test = None
            if psid_ not in (model.active or ()) and pkd_ not in model.select_kinds and model.refute_presumed(pkd_):
                note(f"clicking the thing of colour {pkd_} did not select it: the presumption lapses for that kind")
                if verbose:
                    print(f"   presumed select refuted: kind {pkd_} (active {model.active})")
        if model.active and getattr(play_level, "start_world", None) is None:
            act_ = [x for x in tr.prev if x.sid in model.active]
            if act_:
                cells_ = set()
                for x in act_:
                    cells_ |= set(x.cells)
                c_ = centre(cells_)
                play_level.start_world = (c_[0] + model.camera[0], c_[1] + model.camera[1])
        # the trail: where the controlled things were after every press
        # (arrivals on a tile are counted from it)
        if getattr(play_level, "trail", None) is not None:
            play_level.trail.append({x.sid: frozenset(x.cells) for x in cur if x.sid in (model.active or ())})
        # THE CLOCK PER LIFE: a counter that jumps back to full while the
        # controlled thing jumps far — a death; the level's clock runs per
        # life, and the position (not the world) starts again
        for sid_ in set(model.counters) | {x.sid for x in tr.prev if "bar" in x.tags}:
            s_now = tr.get(sid_)
            s_bef = next((x for x in before if x.sid == sid_), None)
            if s_now is not None and s_bef is not None and sid_ in size_first and s_bef.size < 0.5 * size_first[sid_] \
                    and s_now.size >= 0.9 * size_first[sid_]:
                clock["deaths"] += 1
                clock["per_life"] = steps - clock["life_start"]
                clock["life_start"] = steps
                note(f"the clock ran out: a life lost, the position starts again ({clock['per_life']} actions a life)")
                break
        # PUT BACK: the controlled things stand where they stood at the
        # level's start again, having been elsewhere, on a press whose law
        # said they would step — a jump no key makes.  What that step
        # entered is a hazard without the loss (Tushar, m0r0 L2: "this is
        # where I made myself stuck" — the checkered cells send you back)
        if isinstance(key, str) and model.active and obs.levels_completed <= lv0 and obs.state != GameState.GAME_OVER:
            act_now = {x.sid: frozenset(x.cells) for x in cur if x.sid in model.active and x.sid < 1000}
            act_bef = {x.sid: frozenset(x.cells) for x in before if x.sid in model.active and x.sid < 1000}
            if act_now and all(first_cells_.get(sid_) == cs_ for sid_, cs_ in act_now.items()) \
                    and any(first_cells_.get(sid_) != act_bef.get(sid_) for sid_ in act_now):
                far_ = False
                for sid_, cs_ in act_now.items():
                    b0_ = act_bef.get(sid_)
                    if not b0_:
                        continue
                    d_ = (_bbox(cs_)[0] - _bbox(b0_)[0], _bbox(cs_)[1] - _bbox(b0_)[1])
                    law_ = model.fixed.moving(key).get(sid_)
                    m_ = max(abs(law_[1][0]), abs(law_[1][1])) if law_ and law_[0] == "translate" else 0
                    if max(abs(d_[0]), abs(d_[1])) > m_ + 1:
                        far_ = True             # (a jump no step makes; a step back onto the start is a step)
                if far_:
                    try:
                        found_ = model.note_death(cur, before=before, key=key, put_back=True)
                    except Exception as ex_:
                        found_ = []
                        if verbose:
                            print(f"   note_death (put back) failed: {ex_!r}")
                    note(f"put back to the start on that press: what the step entered ({sorted(found_) if found_ else 'unknown yet'}) is not to be stepped on")
                    if verbose:
                        print(f"   put back: key {key} entered hazards {sorted(found_)} death_enter {dict(model.death_enter)} entered_alive {dict(model.entered_alive)}")
        reset = False
        if obs.state == GameState.GAME_OVER:
            # THE LEVEL WAS LOST: what the body touched on this press, and had
            # never touched and lived, is a hazard — remembered by kind
            if _ENV.get("DEATH_DEBUG"):
                np.save("/tmp/death_before.npy", f_before); np.save("/tmp/death_after.npy", f)
                print(f"   [death] cam {model.camera} hazards {model.hazards()} click {click} key {key}")
            # A DEATH AS THE CLOCK RUNS OUT IS THE CLOCK'S, NOT THE PRESS'S: when
            # the level's counter — the bar every press spends — stood at its
            # last cells before this press, the press killed nothing; no
            # hazard is read off what the body touched, no press is marked
            # fatal, and the next attempt is not a search for a killer but the
            # same level within its budget.  (sp80: the bar spends two cells a
            # press and the level ends at zero; every attempt ended "died" with
            # nothing in the way, and the core looked again for the press)
            clock_out_ = False
            try:
                if model.clock is not None and model.clock_rate:
                    cb_sp = next((s_ for s_ in before if s_.sid == model.clock), None)
                    ca_sp = next((s_ for s_ in cur if s_.sid == model.clock), None)
                    # (the clock's value before the press at or under what one press
                    # can take — the rate, or the largest spend seen when seen twice;
                    # the frame after a death may blank the clock: it is no witness)
                    drop_ = getattr(model, "clock_max_drop", 0) if getattr(model, "clock_max_drop_n", 0) >= 2 else 0
                    if cb_sp is not None and model.clock_value(cb_sp) <= max(model.clock_rate, drop_):
                        clock_out_ = True
                    if verbose and cb_sp is not None:
                        print(f"   death with the clock at {model.clock_value(cb_sp)} (rate {model.clock_rate}, largest spend {getattr(model, 'clock_max_drop', 0)} x{getattr(model, 'clock_max_drop_n', 0)}): {'the clock' if clock_out_ else 'not the clock'}")
            except Exception:
                clock_out_ = False
            if clock_out_ and not _ENV.get("NO_CLOCK_DEATH"):
                note("the level was lost as the clock ran out: the press killed nothing — the level must be won within the budget")
                if verbose:
                    print(f"   died: the clock ran out (clock S{model.clock}, rate {model.clock_rate}) — no hazard, no fatal press")
                model.deaths = getattr(model, "deaths", 0) + 1
            else:
                try:
                    found_ = model.note_death(tr.prev, before=before, key=key if not isinstance(key, tuple) else None, frame=f_before)
                    if found_:
                        note(f"the level was lost on that press: touching colour {sorted(found_)} is death — never again")
                except Exception as ex_:
                    if verbose:
                        print(f"   note_death failed: {ex_!r}")
            play_level.died = True
            if getattr(play_level, "journal", None) is not None and not clock_out_:
                play_level.journal.fatal = (_frame_key_frame(f_before), _key_id(key))
            obs = env.step(GameAction.RESET)
            f = np.array(obs.frame)[-1]
            if SCENE:
                tr.update(f, bg)
            else:
                tr.update(read_components(f, bg))
            reset = True
            play_level.last = {"model": model, "tr": tr, "hyps": hyps, "keys": keys, "clickable": clickable, "comps": before}
            raise _Died(obs)
        ev = [f"{e.kind} S{e.sid}" + (f" {tuple(int(x) for x in e.data['vec'])}" if e.data.get("vec") is not None and e.kind == "moved" else "")
              for e in tr.last_events if e.kind != "recoloured" or e.sid < 1000][:8]
        if getattr(play_level, "journal", None) is not None:
            play_level.journal.press(steps, key, getattr(play_level, "phase", ""), ev, _frame_key(obs), reset=reset)
        if getattr(play_level, "rec", None) is not None:
            play_level.rec.append({"i": steps, "action": key if not isinstance(key, tuple) else key[0], "click": list(click) if click else None,
                                   "frame": _hex(f), "events": ev, "phase": getattr(play_level, "phase", ""),
                                   "lv": int(obs.levels_completed), "reset": reset, "seen": _seen(tr, model)})
        return cur

    def blockers_of(succ, start, mover_sid):
        """The things that refused the mover in the search's best state
        (the successor function says why each key failed there)."""
        best = (getattr(plan_general, "last", {}) or {}).get("best")
        st = dict(start)
        if best and best[1]:
            for a_ in best[1]:
                nxt = next((new for act, new in succ(st) if act == a_), None)
                if nxt is None:
                    break
                st = nxt
        succ(st)
        out = []
        for k_, why in (getattr(make_successors, "whys", {}) or {}).items():
            if why and why[0] == "solid" and len(why) >= 3 and why[2] != mover_sid and why[2] not in out:
                out.append(why[2])
        return out

    def unblock(blockers, mover_sid):
        """Try what might open the way; True when a blocker changed."""
        snap = {b: next((frozenset(x.cells) for x in tr.prev if x.sid == b), None) for b in blockers}

        def changed():
            for b, cells0 in snap.items():
                now = next((frozenset(x.cells) for x in tr.prev if x.sid == b), None)
                if now != cells0:
                    return True
            return False
        me = next((x for x in tr.prev if x.sid == mover_sid), None)
        if me is None:
            return False
        # (a) pads: small odd things the mover can stand on
        counts = Counter(x.body for x in tr.prev if "hud" not in x.tags and "terrain" not in x.tags and x.sid < 1000)
        pads = [x for x in tr.prev if x.sid < 1000 and x.sid != mover_sid and x.sid not in blockers and "hud" not in x.tags
                and "terrain" not in x.tags and 2 <= x.size <= 30 and counts[x.body] <= 2 and x.body != me.body]
        pads.sort(key=lambda x: abs(x.centre[0] - me.centre[0]) + abs(x.centre[1] - me.centre[1]))
        for pad in pads[:2]:
            note(f"unblock: standing on S{pad.sid} (colour {pad.body}) — does it open the way?")
            n_ = _walk_over(tr, model, keys, mover_sid, pad, step, verbose, max_steps=20)
            if obs.levels_completed > lv0:
                return True
            if changed():
                return True
        # (b) a click on the blocker
        if clickable:
            for b in blockers[:1]:
                bx = next((x for x in tr.prev if x.sid == b), None)
                if bx is not None:
                    note(f"unblock: clicking the blocker S{b}")
                    step(("click", click_point(bx)))
                    if obs.levels_completed > lv0 or changed():
                        return True
        # (c) each silent key once, from where the mover stands
        for k_ in keys:
            if k_ in model.undo_keys or model.fixed.moving(k_) or k_ in model.key_selected:
                continue
            note(f"unblock: pressing {k_}")
            step(k_)
            if obs.levels_completed > lv0 or changed():
                return True
        return False

    won_tpl_ = (carry or {}).get("won")
    tpl_kind_ = re.split(r" C\d+\[| \{|\(", str(won_tpl_[3] or ""))[0].strip() if won_tpl_ and won_tpl_[0] else ""

    def _template_reading(h):
        """A reading of the relation that won the last level: the same
        rule and kind of reading, aimed at the same colour of target."""
        if not won_tpl_ or not won_tpl_[0]:
            return False
        if h.rule() != won_tpl_[0] or (tpl_kind_ and not h.note.startswith(tpl_kind_)):
            return False
        if won_tpl_[2] is None:
            return True
        # (a reading names its target's colour in its note when it carries no
        # t_color: "aimed at the same colour" is judged on that, never waived —
        # tw03 L2 pursued the odd thing C9, a wall, as the template for C8)
        tcol_ = getattr(h, "t_color", None)
        if tcol_ is None:
            m_ = re.search(r"C(\d+)\[", h.note or "")
            tcol_ = int(m_.group(1)) if m_ else None
        return tcol_ in (None, won_tpl_[2])

    def _sits_in(h):
        """Whether something sits in the reading's frame now: a thing of
        the mover's kind within the frame's box, or the box's inside
        filled (the tracker may read thing and frame as one)."""
        tb_ = _bbox(h.t_cells)
        for s_ in tr.prev:
            if s_.sid < 1000 and s_.body == h.m_color and _same_form(s_, h) \
                    and tb_[0] <= s_.bbox[0] and s_.bbox[2] <= tb_[2] and tb_[1] <= s_.bbox[1] and s_.bbox[3] <= tb_[3]:
                return True
        fr_ = getattr(model, "_frame_now", None)
        if fr_ is None:
            return True
        inner_ = [(r_, c_) for r_ in range(tb_[0] + 1, tb_[2]) for c_ in range(tb_[1] + 1, tb_[3]) if (r_, c_) not in h.t_cells]
        if not inner_:
            return True
        filled_ = sum(1 for (r_, c_) in inner_ if 0 <= r_ < fr_.shape[0] and 0 <= c_ < fr_.shape[1] and int(fr_[r_, c_]) != bg)
        return filled_ >= 0.8 * len(inner_)

    def _docked_now(pool_):
        """The pieces that sit where a done reading wants them."""
        out_ = set()
        for h_ in pool_:
            if not getattr(h_, "_done", False) or not getattr(h_, "t_cells", None):
                continue
            tb_ = _bbox(h_.t_cells)
            for s_ in tr.prev:
                if s_.sid < 1000 and tb_[0] <= s_.bbox[0] and s_.bbox[2] <= tb_[2] and tb_[1] <= s_.bbox[1] and s_.bbox[3] <= tb_[3] \
                        and s_.body == h_.m_color and _same_form(s_, h_):
                    out_.add(s_.sid)
        return out_

    def _same_form(s_, h_):
        """Whether a thing is of the reading's mover's form: its size, and
        its proportions when the reading kept its mover's cells."""
        if not (0.5 * h_.m_size <= s_.size <= 2 * h_.m_size):
            return False
        mc_ = getattr(h_, "m_cells", None)
        if mc_:
            mb_ = _bbox(mc_)
            if (mb_[2] - mb_[0] + 1, mb_[3] - mb_[1] + 1) != (s_.bbox[2] - s_.bbox[0] + 1, s_.bbox[3] - s_.bbox[1] + 1):
                return False
        return True

    def pursue(hyps, max_single=8, early=False, depth=0):
        """Chase the readings: joint goals first, then each single reading
        — plan, act, verify, replan; returns the win tuple or None.  After a
        docking the world is read again and the chase starts over from the
        readings as they stand (a docking may have undone another)."""
        play_level.redo_pursuit = False
        kind_of_sid_pursue = {s_.sid: s_.body for s_ in tr.prev}
        # A THING DOES NOT AIM AT ITS OWN BODY: a reading whose target is a
        # member of the body its mover belongs to (the eye of the block
        # it rides) is no reading
        # A DOCKING UNDONE IS A READING AGAIN: a done docking whose piece no
        # longer wears a docked mark (its pad moved on to dock itself) is
        # pursued again — after the pad has settled
        for h in hyps:
            if getattr(h, "_done", False) and getattr(h, "_fit_host", None) is not None and "marks onto" in h.note:
                hf_ = h._fit_host
                marks_h = [s_ for s_ in tr.prev if (s_.sid >= 1000 and s_.sid // 1000 == hf_ and s_.size <= 40)
                           or (s_.sid < 1000 and s_.size <= 16 and s_.sid in {p_ for b_ in model.bodies() if hf_ in b_ for p_ in b_})]
                if marks_h and all(s_.body == h.m_color for s_ in marks_h) and any(s_.sid == hf_ for s_ in tr.prev):
                    h._done = False
                    note("a docking came undone when its pad moved: that reading is open again")
            # A FIT LEFT EMPTY IS A READING AGAIN: a done fit whose frame no
            # longer holds a thing of the mover's kind (the thing went to
            # push another) is pursued again
            if getattr(h, "_done", False) and h.kind in ("align", "cover", "inside") and getattr(h, "t_cells", None) and "marks onto" not in h.note:
                if not _sits_in(h):
                    h._done = False
                    note(f"the frame stands empty again ({h.note[:40]}): that reading is open again")
        hyps = [h for h in hyps if not _self_aimed(h) and not getattr(h, "_done", False)]
        # A READING IS ORIENTED BY THE EFFECTS (Tushar, 37r: hypotheses from
        # what changed, not only from the still frame): a symmetric reading —
        # a thing to its same-colour marker, a thing onto the same shape — is
        # read off the frame with the smaller thing as the mover.  When no key
        # moves that mover and the keys DO move the marker's kind, the reading
        # is the same relation the other way round: the marker is the mover,
        # the thing its mark.  (vc33 L3: the wall's mark read as the mover of
        # the column's base, which the clicks raise — no plan, seven readings,
        # twice.)  Added beside the original, not in its place.
        if verbose and _ENV.get("FLIP_DEBUG"):
            print(f"   [flip-entry] early {early} depth {depth} n {len(hyps)} aligns {sum(1 for h in hyps if h.kind == 'align')}")
        if not early and not _ENV.get("NO_FLIP") and depth == 0:
            flipped_ = []
            for h in hyps:
                if h.kind != "align" or getattr(h, "_flipped", False) or not ("same-colour marker" in h.note or "fits the same shape" in h.note):
                    continue
                m_ = re.search(r"C(\d+)\[", h.note)
                tcol_f = int(m_.group(1)) if m_ else None
                if tcol_f is None:
                    continue
                tcells_ = set(getattr(h, "t_cells", ()) or ())
                if verbose and _ENV.get("FLIP_DEBUG"):
                    print(f"   [flip?] {h.note[:40]} tcol {tcol_f} tcells {len(tcells_)} m_color {h.m_color}")
                if not tcells_:
                    continue
                # (the mover may be a PART — a mark drawn on a wall; parts ride
                # their host, and a wall's mark moves with nothing)
                mover_now_ = min([s_ for s_ in tr.prev if s_.body == h.m_color and "hud" not in s_.tags and 0.5 * h.m_size <= s_.size <= 2 * h.m_size
                                  and not (set(s_.cells) & tcells_)] or [None],
                                 key=lambda s_: (abs(s_.centre[0] - h.m_start[0]) + abs(s_.centre[1] - h.m_start[1])) if s_ is not None else 1e9)
                tthing_ = next((s_ for s_ in tr.prev if s_.sid < 1000 and set(s_.cells) == tcells_), None) or \
                          next((s_ for s_ in tr.prev if s_.sid < 1000 and s_.body == tcol_f and set(s_.cells) & tcells_), None)
                if mover_now_ is None or tthing_ is None or mover_now_.sid == tthing_.sid:
                    continue

                def _inst_moved(s_):
                    # (the THING, not its colour: the wall's mark and the column's
                    # base share a colour; only the base answers the clicks)
                    for k_ in model.fixed.keys():
                        e_ = model.fixed.moving(k_).get(s_.sid)
                        if e_ is not None and e_[0] != "none":
                            return True
                    return False
                if verbose and _ENV.get("FLIP_DEBUG"):
                    print(f"   [flip] {h.note[:40]} mover S{mover_now_.sid} moved {_inst_moved(mover_now_)} target S{tthing_.sid} kind {tthing_.body} moved {_inst_moved(tthing_)} laws {[(k_, dict(model.fixed.moving(k_))) for k_ in model.fixed.keys()][:3]}"[:400])
                if _inst_moved(mover_now_) or not _inst_moved(tthing_):
                    continue
                from types import SimpleNamespace as _NS
                tm_ = _NS(color=tthing_.body, size=tthing_.size, centre=tthing_.centre, cells=frozenset(tthing_.cells))
                g_ = GoalHypothesis("align", tm_, mover_now_.centre,
                                    f"same-colour marker C{h.m_color}[{mover_now_.size}@{mover_now_.bbox}] (turned round: the keys move this one)", h.prio, mover_now_.size)
                g_.t_cells = frozenset(mover_now_.cells); g_.t_color = h.m_color; g_._flipped = True
                g_.evidence = getattr(h, "evidence", ())
                if any(x_.m_color == g_.m_color and x_.m_start == g_.m_start and getattr(x_, "t_cells", None) == g_.t_cells for x_ in hyps):
                    continue
                flipped_.append(g_)
                if verbose:
                    print(f"   turned round: {h.note[:50]} -> {g_!r}"[:200])
            if flipped_:
                hyps = flipped_ + hyps
                if isinstance(play_level.last, dict):
                    play_level.last["hyps"] = list(play_level.last.get("hyps", [])) + flipped_
        # (a reading whose mover is the kind a carry law binds is the carry
        # protocol's to carry out, not the planner's: the planner cannot
        # move a thing the body must bind first)
        if getattr(model, "carry", None):
            hyps = [h for h in hyps if h.m_color not in model.carry.get("kinds", ()) or model.carry.get("target_kind") is None]
        for h in hyps:
            h._tried = True
        # THE WINNING READING FIRST: readings of the relation that won the
        # last level come before every joint or other reading, whenever
        # readings are chased on this level (not only at its opening)
        tpl_ = [h for h in hyps if _template_reading(h)]
        # THE HUB FIRST: among dockings, the piece most others dock onto is
        # the anchor — readings aimed at it go first (a piece docked onto
        # early may have to move again for its own docking later)
        # (a piece others dock onto docks first itself: a reading whose target
        # is a mover elsewhere waits for that mover's own docking — else the
        # early docking is undone when its pad moves)
        def _hosts(h_):
            mt = re.search(r" of C(\d+)", h_.note); mf = re.search(r" from C(\d+)", h_.note)
            return (mf.group(1) if mf else None, mt.group(1) if mt else None)
        movers_ = {_hosts(h)[0] for h in tpl_ if "marks onto" in h.note and _hosts(h)[0] is not None}
        tpl_.sort(key=lambda h: ((1 if ("marks onto" in h.note and _hosts(h)[1] in movers_ and _hosts(h)[1] != _hosts(h)[0]) else 0), -h.prio))
        ordered = [("single", h) for h in tpl_]
        # READINGS COUPLED BY THE KEYS ARE ONE GOAL: when one key moves the
        # things of two or more align readings (vc33 L3: a button pumps one
        # piston up and its neighbour down — each carries a mark to bring
        # level with its own arrow), pursuing one reading undoes another;
        # the coupled set is planned at once, before the singles.
        mix_ = None
        try:
            mix_ = _coupled_aligns(hyps, tr, model)
        except Exception as ex_:
            if verbose:
                print(f"   (coupled readings failed: {ex_!r})")
        groups = defaultdict(list)
        for h in hyps:
            groups[(h.rule(), h.m_color)].append(h)
        for (rule, col), hs in groups.items():
            movers = sorted({h.m_start for h in hs}); targets = sorted({h.target_centre() for h in hs})
            if len(movers) >= 2 and len(targets) >= 2 and not any("fits the same shape" in h.note or "marks onto" in h.note for h in hs):
                ordered.append(("joint", hs, movers, targets))     # (two fit readings are alternatives, not a joint goal)
        # (the coupled set comes after the singles of its readings: a single
        # that plans wins on its own — vc33 L2; the joint is for when the
        # singles undo one another — vc33 L3)
        if mix_:
            ordered.append(("jointmix_later", mix_))
            note(f"readings coupled by the keys ({len(mix_)}): one goal after the singles — {', '.join(h_.note[:30] for h_, _, _, _ in mix_)}")
        # THE SUBJECT MOVES AGAINST THE CROWD (goals from effects, statement 1
        # refined): when one key moves several things by one vector — the
        # field scrolling, a track sliding — the subject is what that key
        # moves OTHERWISE (sk48: the cart on its track, read as a crowd
        # member while a track segment was taken for the mover).  Readings
        # whose mover is such a thing come before the rest; the order is
        # kept otherwise.
        if not _ENV.get("NO_SUBJECT_ORDER"):
            try:
                subj_cols_ = set()
                for k_ in model.fixed.keys():
                    mv_k = {sid_: e_ for sid_, e_ in model.fixed.moving(k_).items() if sid_ < 1000}
                    vecs_ = Counter(tuple(e_[1]) for e_ in mv_k.values() if e_[0] == "translate")
                    if not vecs_:
                        continue
                    modal_, n_modal_ = vecs_.most_common(1)[0]
                    if n_modal_ < 3:
                        continue
                    for sid_, e_ in mv_k.items():
                        # (moved OTHERWISE: by another vector — a thing that
                        # grows or shrinks under the key is the mechanism that
                        # moves the crowd, not a subject: s5i5's pistons)
                        if e_[0] == "translate" and tuple(e_[1]) != modal_:
                            c_ = kind_of_sid_pursue.get(sid_)
                            if c_ is not None:
                                subj_cols_.add(c_)
                if subj_cols_:
                    hyps = sorted(hyps, key=lambda h_: 0 if h_.m_color in subj_cols_ else 1)
                    if verbose:
                        print(f"   subjects against the crowd: kinds {sorted(subj_cols_)}")
            except Exception as ex_:
                if verbose:
                    print(f"   (subject order failed: {ex_!r})")
        rest_ = [h for h in hyps if h not in tpl_][:max_single]
        # (the record ORDERS the readings pursued, it does not choose them:
        # the cut is by the reader's rank, the failed go last WITHIN it —
        # sorted before the cut, ls20 L2's "reach the odd thing" fell off the
        # end behind readings the reader ranked lower, and the walk that
        # shows the display's mechanic was never taken)
        if not _ENV.get("NO_FAILED_ORDER"):
            rest_.sort(key=lambda h_: failed_readings.get(_reading_id(h_), 0))      # (stable: the untried first, the failed last)
        for h in rest_:
            ordered.append(("single", h))
        # (the coupled set last: after every single has had its turn)
        ordered = [o_ for o_ in ordered if o_[0] != "jointmix_later"] + [("jointmix", o_[1]) for o_ in ordered if o_[0] == "jointmix_later"]
        if verbose:
            print(f"   pursue: {[(o[0], str(o[1])[:50] if o[0] == 'single' else (len(o[1]) if o[0] == 'jointmix' else (len(o[2]), len(o[3])))) for o in ordered]}")
        for item in ordered:
            if time.time() - play_level.t0 > _level_seconds():
                note("out of thinking time for this level")
                break                   # (a wall clock: the game's budget is actions, ours is time too)
            if item[0] == "jointmix":
                mix_ = item[1]
                note(f"joint goal over {len(mix_)} coupled readings: every mark level with its own")
                won_m, steps_m = play_joint_mixed(tr, model, keys, mix_, step, lambda: obs, lv0, verbose)
                if won_m:
                    play_level.last["won_hyp"] = mix_[0][0]
                    return True, steps, "joint", obs
                continue
            if item[0] == "joint":
                _, hs, m_starts, tcs = item
                h = hs[0]
                note(f"joint goal: {len(m_starts)} movers of colour {h.m_color} to {len(tcs)} markers at once — {h.note[:60]}")
                won_j, steps_j = play_joint(env, tr, model, keys, h, m_starts, tcs, step, lambda: obs, lv0, verbose, nudge=not early)
                if won_j:
                    play_level.last["won_hyp"] = h          # (the relation that won carries to the next level)
                    return True, steps, "joint", obs
                continue
            h = item[1]
            # (the parallel record judges an agent against the goal's place: the
            # cells the pursued reading aims at)
            try:
                tc0_ = getattr(h, "t_cells", None)
                if tc0_ and getattr(model, "agent_record", None) is not None:
                    model.agent_record.goal_place = _bbox(tc0_)
            except Exception:
                pass
            if h.kind == "collect":
                # ARRIVE ON EACH, nearest first; a member that neither vanished
                # nor changed on arrival refutes the reading
                # the collector is the thing the keys move
                act_ = [s_ for s_ in tr.prev if s_.sid in (model.active or ())]
                me_ = max(act_, key=lambda s_: s_.size) if act_ else next((s_ for s_ in tr.prev if s_.body == h.m_color and s_.sid < 1000), None)
                members = list(getattr(h, "members", ()))
                note(f"collect: {len(members)} like things — arriving on each, nearest first")
                collected = 0
                no_way_ = False
                while members and me_ is not None:
                    me_ = next((s_ for s_ in tr.prev if s_.sid == me_.sid), me_)
                    alive = [s_ for s_ in tr.prev if s_.sid in members]
                    if not alive:
                        break
                    # nearest first — but when the clock per life is known, a
                    # member whose leg fits what is left of this life comes
                    # before one that does not (a death restarts the position)
                    legs = {}
                    t_l_ = time.time()
                    if _leg_any(tr, model, keys, me_.sid, alive):
                        for s_ in alive:
                            legs[s_.sid] = _leg_length(tr, model, keys, me_.sid, s_)
                    else:
                        legs = {s_.sid: None for s_ in alive}
                    if verbose:
                        print(f"   collect legs {legs} ({time.time() - t_l_:.1f}s)")
                    left_ = clock_left()
                    fits = [s_ for s_ in alive if legs[s_.sid] is not None and (left_ is None or legs[s_.sid] <= left_)]
                    # (no way to ANY member under what is known: the reading is
                    # not walked — the walk would search the same space again,
                    # three times over, for the same answer.  bp35 L2 spent
                    # 250 s of its 300 on two such readings before the first
                    # reading with a plan was reached.  Left unplanned, not
                    # refuted: a way may open when the world changes)
                    if not any(legs[s_.sid] is not None for s_ in alive):
                        note(f"collect: no way to any of the {len(alive)} like things under what is known — not walked")
                        no_way_ = True
                        break
                    pool_ = fits or [s_ for s_ in alive if legs[s_.sid] is not None] or alive
                    # IN AN ORDER THAT LEAVES THE REST REACHABLE: where the way
                    # is not the same both ways (a drift, a slide, a one-way
                    # door), the nearest first may strand the others — among
                    # the members with a leg, the one from whose place the
                    # most of the rest still have a leg comes first, then the
                    # nearest.  (shw0, the study game: three goals under an
                    # eastward drift; the nearest-first order left one to
                    # the west, where nothing can go.)  Counted only when a
                    # one-way motion is known (the drift, a slide); else the
                    # nearest, as before.
                    reach_after_ = {}
                    if len(pool_) > 1 and (getattr(model.fixed, "drift", None) is not None or model.fixed.slides or (hasattr(model, "slide") and model.slide() is not None)):
                        for s_ in pool_:
                            if legs.get(s_.sid) is None:
                                continue
                            n_ = 0
                            for o_ in alive:
                                if o_.sid == s_.sid:
                                    continue
                                if _leg_length(tr, model, keys, me_.sid, o_, me_at=s_) is not None:
                                    n_ += 1
                            reach_after_[s_.sid] = n_
                        if verbose:
                            print(f"   collect: the rest reachable after each: {reach_after_}")
                    t_ = min(pool_, key=lambda s_: (-reach_after_.get(s_.sid, 0), legs.get(s_.sid) if legs.get(s_.sid) is not None else abs(s_.centre[0] - me_.centre[0]) + abs(s_.centre[1] - me_.centre[1])))
                    before_cells = dict(t_.cells)
                    _walk_over(tr, model, keys, me_.sid, t_, step, verbose)
                    if obs.levels_completed > lv0:
                        play_level.last["won_hyp"] = h
                        return True, steps, "collect", obs
                    now_ = next((s_ for s_ in tr.prev if s_.sid == t_.sid), None)
                    members.remove(t_.sid)
                    if now_ is not None and dict(now_.cells) == before_cells and "hidden" not in now_.tags:
                        break           # arrived and it is unchanged: not a collectible
                    collected += 1
                if no_way_:
                    pass                    # (unplanned, not refuted)
                elif collected == 0:
                    rweights.failed(h); failed_readings[_reading_id(h)] += 1
                elif model.displays and model.triggers and model.active:
                    # the world changed (things were collected): what was
                    # refused before is worth another try — the display
                    # protocol and its walk to the reference
                    note("collected: trying again what was refused before")
                    won_d, _ = display_protocol(env, tr, model, keys, step, lambda: obs, lv0, verbose)
                    if won_d:
                        play_level.last["won_hyp"] = h
                        return True, steps, "collect+display", obs
                continue
            m_col = h.m_color; m_start = h.m_start; tcen = h.target_centre()
            fit_sids = None

            def resolve_fit():
                """THE MARKS ARE THE MOVER: the marks of the reading's colour (or
                the colour they take when docked) on the host that answers the
                keys — the selected thing, else the host nearest where the
                reading saw them; the marks to land on are the OTHER host's,
                where they are NOW.  Returns (mover mark sids, target cells)."""
                marks_ = [s for s in tr.prev if s.sid < 1000 and s.size <= 16 and "hud" not in s.tags and "terrain" not in s.tags
                          and (s.body == m_col or s.body in model.kinds_worn.get(s.sid, ()) and m_col in model.kinds_worn.get(s.sid, ()))]

                bodies_r = model.bodies()

                def host_(mk):
                    # (a mark that has MOVED with a thing is that thing's — a
                    # mark touching two pieces is not read by which is smaller)
                    for b_ in bodies_r:
                        if mk.sid in b_:
                            hs_ = [x_ for x_ in tr.prev if x_.sid in b_ and x_.sid != mk.sid and x_.size > mk.size and x_.body != mk.body and x_.sid < 1000]
                            if hs_:
                                return max(hs_, key=lambda x_: x_.size)
                    best = None
                    for hh in tr.prev:
                        if hh.sid == mk.sid or hh.body == mk.body or hh.size <= mk.size or "terrain" in hh.tags or hh.sid >= 1000:
                            continue
                        if any(abs(r - hr) + abs(c - hc) <= 2 and (r != hr or c != hc) for r, c in mk.cells for hr, hc in hh.cells):
                            if best is None or hh.size < best.size:
                                best = hh
                    return best
                by_host_ = defaultdict(list)
                for mk in marks_:
                    hh = host_(mk)
                    if hh is not None:
                        by_host_[hh.sid].append(mk)
                # (marks read as PARTS of their piece are its marks too)
                for mk in tr.prev:
                    if mk.sid >= 1000 and (mk.body == m_col or m_col in model.kinds_worn.get(mk.sid, ())) and mk.size <= 40 \
                            and any(x_.sid == mk.sid // 1000 for x_ in tr.prev):
                        by_host_[mk.sid // 1000].append(mk)
                if len(by_host_) < 2:
                    if verbose:
                        print(f"      resolve_fit: no two hosts: marks {[(mk.sid, mk.body, mk.size, mk.bbox[:2]) for mk in marks_]} hosts {dict(by_host_)} things {[(x_.sid, x_.body, x_.size, x_.bbox, sorted(x_.tags)) for x_ in tr.prev if x_.size > 16 and 'hud' not in x_.tags]}")
                    return None, None
                # THE READING NAMES ITS MOVER BY PLACE: the host whose marks
                # are where the reading saw them (once found, that host —
                # things move); the selected thing only breaks a tie
                # (several pieces bear marks: the selection is not the reading)
                sel_ = model.selected_now(tr.prev)
                near_ = {hs: min(abs(mk.centre[0] - m_start[0]) + abs(mk.centre[1] - m_start[1]) for mk in by_host_[hs]) for hs in by_host_}
                # (a host with no free mark left is no mover: the reading is
                # resolved again by place — its marks may be another piece's)
                by_host_ = {hs: mks for hs, mks in by_host_.items() if any("hidden" not in mk.tags and mk.body == m_col for mk in mks)}
                if len(by_host_) < 2:
                    return None, None
                near_ = {hs: near_[hs] for hs in by_host_}
                if getattr(h, "_fit_host", None) in by_host_:
                    hsid_ = h._fit_host
                else:
                    best_ = min(near_.values())
                    hsid_ = sel_ if sel_ in by_host_ and near_[sel_] <= best_ + 3 else min(near_, key=near_.get)
                    h._fit_host = hsid_
                others_ = [hs for hs in by_host_ if hs != hsid_]
                if verbose:
                    print(f"      resolve_fit: m_start {m_start} near {near_} sel {sel_} -> host {hsid_} (kept {getattr(h, '_fit_host', None)}); hosts {{hs: [(mk.sid, mk.bbox[:2]) for mk in by_host_[hs]] for hs in by_host_}}")
                if getattr(h, "_fit_target", None) in others_:
                    osid_ = h._fit_target
                else:
                    tc0_ = getattr(h, "_t_start", None) or tcen
                    h._t_start = tc0_
                    osid_ = min(others_, key=lambda hs: min(abs(mk.centre[0] - tc0_[0]) + abs(mk.centre[1] - tc0_[1]) for mk in by_host_[hs]))
                    h._fit_target = osid_
                # (a mark already docked — covered, or wearing the docked
                # colour — is taken: neither a place to land nor a mark to land)
                def free_(mk):
                    return "hidden" not in mk.tags and mk.body == m_col
                live_t = set()
                for mk in by_host_[osid_]:
                    if free_(mk):
                        live_t |= set(mk.cells)
                movers_ = [mk.sid for mk in by_host_[hsid_] if free_(mk)]
                if not live_t or not movers_:
                    return None, None
                return movers_, frozenset(live_t)
            if "marks onto like marks" in h.note:
                fit_sids, live_t = resolve_fit()
                if not fit_sids:
                    continue
                h.t_cells = live_t; h.target = live_t
                tcen = h.target_centre()
                cands = [s for s in tr.prev if s.sid in fit_sids]
            else:
                cands = [s for s in tr.prev if s.body == m_col and 0.5 * h.m_size <= s.size <= 2 * h.m_size]
                # (THE READING NAMES ITS MOVER BY FORM before place: among
                # things of the kind, those of the reading's size — a small
                # block that wandered near where the tall one stood is not
                # the tall one)
                if cands:
                    off_ = min(abs(s.size - h.m_size) for s in cands)
                    cands = [s for s in cands if abs(s.size - h.m_size) <= off_ + max(2, 0.1 * h.m_size)]
                    # (...and of the reading's proportions: a wide block and
                    # a tall one are of one size and not of one form)
                    mc_ = getattr(h, "m_cells", None)
                    if mc_ and len(cands) > 1:
                        mb_ = _bbox(mc_)
                        md_ = (mb_[2] - mb_[0] + 1, mb_[3] - mb_[1] + 1)
                        same_ = [s for s in cands if (s.bbox[2] - s.bbox[0] + 1, s.bbox[3] - s.bbox[1] + 1) == md_]
                        if same_:
                            cands = same_
            if not cands:
                continue
            mover = min(cands, key=lambda s: abs(s.centre[0] - m_start[0]) + abs(s.centre[1] - m_start[1]))
            sid = mover.sid
            tcells = set(getattr(h, "t_cells", ()))
            if tcells and h.kind in ("cover", "align", "inside"):
                # LIKE WITH LIKE: the mover's centre is its cells' centroid;
                # the target's must be too (an L's box centre is not where
                # the L is — the fit was never "arrived" at by 2.4 cells)
                tcen = centre(tcells)
            solids = set()
            for s_ in tr.prev:
                near = abs(s_.centre[0] - tcen[0]) + abs(s_.centre[1] - tcen[1]) <= 4
                # (a thing the keys were seen to PUSH is solid even as the
                # target: it is not stood on but shoved — the plan must say
                # where it goes; tw03 L2 "reached" the block by displacing it)
                if "hud" in s_.tags or (s_.body not in model.pushable and ((tcells and len(set(s_.cells) & tcells) >= 0.8 * s_.size) or (near and s_.size <= 12))):
                    continue
                # (THE MARKS' OWN SURROUND: a small thing drawn round a mark
                # aimed at — the ring a dot sits in — is its setting, not a
                # wall on the way to it)
                if tcells and s_.size <= 16 and s_.body not in model.pushable and all(any(abs(r_ - tr_) <= 1 and abs(c_ - tc_) <= 1 for (tr_, tc_) in tcells) for (r_, c_) in s_.cells):
                    continue
                if "terrain" in s_.tags or s_.size <= 2:
                    continue        # ground is walked on; a dot is not a wall
                solids.add(s_.sid)
            m_kind = mover.body
            m_size = mover.size
            # THE AXES THE MOVER CAN MOVE ON: a slider aligns along its own
            # axis; the reading's other coordinate is not its concern
            axes = [False, False]
            for k in model.fixed.keys():
                e = model.fixed.moving(k).get(sid)
                if e and e[0] == "translate":
                    axes[0] |= bool(e[1][0]); axes[1] |= bool(e[1][1])
                elif e and e[0] in ("extend", "retract"):
                    axes[0] |= e[1][0] in ("top", "bottom"); axes[1] |= e[1][0] in ("left", "right")
            if model.key_selected and (m_kind in model.select_kinds or sid in selected_hosts(tr.prev) or model.cycle_keys):
                for v in model.key_selected.values():
                    axes[0] |= bool(v[0]); axes[1] |= bool(v[1])
            if model.relative_law(m_kind) is not None or m_kind in model.pushable:
                axes = [True, True]
            # (an axis the body travels by a slide or by gravity is an axis of
            # the reading too: the keys of tw05 move the body only sideways,
            # but it falls — a mark above or below is reached by falling or
            # jumping, not ignored)
            sl_ = model.slide() if hasattr(model, "slide") else None
            if sl_ is not None:
                axes[0] |= bool(sl_[0]); axes[1] |= bool(sl_[1])
            if not any(axes):
                axes = [True, True]
            axes = tuple(axes)

            # THE TARGET IS WHERE IT IS NOW: a reading aimed at a THING (not at a
            # place) is measured to that thing as it stands in each state — a
            # pushable thing shoved away by the arrival was not arrived at (tw03
            # L2: "reach the block" was satisfied by pushing the block into the
            # far corner, the player taking its place; the level was lost there)
            t_sid_ = None
            if tcells and h.kind in ("align", "inside") and not fit_sids:
                tcol_h_ = getattr(h, "t_color", None)
                if tcol_h_ is None:
                    m_ = re.search(r"C(\d+)\[", h.note or "")
                    tcol_h_ = int(m_.group(1)) if m_ else None
                for s_ in tr.prev:
                    # (the thing aimed at, by its colour when the reading names one —
                    # never the body that may be standing on it now)
                    if s_.sid != sid and s_.sid < 1000 and "hud" not in s_.tags and "terrain" not in s_.tags and s_.sid not in (model.active or ()) \
                            and (tcol_h_ is None or s_.body == tcol_h_) and len(set(s_.cells) & tcells) >= 0.9 * len(tcells):
                        t_sid_ = s_.sid
                        break
                # (A THING AIMED AT THAT HAS NO HOLE IS A PLACE TO LIE ON: it is
                # no solid for this plan, and reached means standing over it —
                # su15 L2's disc, the ring of L1 drawn solid)
                if t_sid_ is not None and h.kind == "align" and not getattr(h, "members", None) and not _ENV.get("NO_LIE_ON") \
                        and model.relative_law(m_kind) is not None:
                    # (a thing the clicks CALL lies where it is called; a walker
                    # stands against what it reaches — the rule is the caller's)
                    ts_h_ = next((s_ for s_ in tr.prev if s_.sid == t_sid_), None)
                    if ts_h_ is not None and not ts_h_.cavity() and ts_h_.size >= m_size:
                        solids.discard(t_sid_)
                        h.over = True
                # (A PUSHABLE THING IS NO PLACE TO ARRIVE AT — known before any
                # search: the body shoves it off as it comes; a reading that
                # would stand the body on it is set aside without the search
                # that cannot find a way — tw01, fifteen seconds a try)
                if t_sid_ is not None and tcol_h_ is not None and tcol_h_ in model.pushable and h.m_color != tcol_h_ \
                        and (not model.active or sid in model.active) and not getattr(h, "members", None) and not _ENV.get("NO_SET_ASIDE"):
                    note(f"reading set aside: {h.note[:50]} — a thing the body pushes is no place to arrive at")
                    if h.note not in play_level.ledger["hyps_tried"]:
                        play_level.ledger["hyps_tried"].append(h.note)
                    continue

            def dist(state, _kind=m_kind, _tc=tcen, _sid=sid, _ms=m_size, _axes=axes, _fit=fit_sids, _tcells=tcells, _tsid=t_sid_):
                # the mover is whatever wears its kind now: the thing itself,
                # or its pieces after a mark split it — or its marks together
                if _tsid is not None:
                    tcs_ = state.get(_tsid)
                    if tcs_:
                        _tc = centre(tcs_)
                if _fit:
                    # (each mark to the nearest target mark cell: a piece with
                    # two marks docks on two of the anchor's four, not on
                    # their centroid)
                    ds_ = []
                    for f_ in _fit:
                        cs_ = state.get(f_)
                        if not cs_:
                            continue
                        for mk_ in _marks_of(cs_):
                            c_ = centre(mk_)
                            ds_.append(min(abs(c_[0] - r) + abs(c_[1] - q) for (r, q) in _tcells) if _tcells else (abs(c_[0] - _tc[0]) + abs(c_[1] - _tc[1])))
                    if not ds_:
                        return 999.0
                    ds_.sort()
                    ds_ = ds_[:max(1, min(2, len(ds_)))]      # (the two marks that dock: the rest may stay free)
                    return sum(ds_) / len(ds_)
                cs = state.get(_sid)
                if not cs:
                    allc = set()
                    for k_, v_ in state.items():
                        if isinstance(k_, str):
                            continue
                        kk = kind_of_sid.get(k_)
                        if kk == _kind and len(v_) >= 3:
                            allc |= set(v_)
                    cs = allc
                if not cs:
                    return 999.0
                c = centre(cs)
                return (abs(c[0] - _tc[0]) if _axes[0] else 0) + (abs(c[1] - _tc[1]) if _axes[1] else 0)
            if h.kind == "meet":
                # THE TARGET MOVES TOO: a meeting is between the two movers as
                # they stand in each state — the other mover is the target,
                # neither is a wall to the other (they are to become one)
                other_ = next((s_ for s_ in tr.prev if s_.sid == getattr(h, "meet", (None, None))[1] and s_.sid != sid), None)
                if other_ is None:
                    others_ = [s_ for s_ in tr.prev if s_.sid != sid and s_.body == m_kind and s_.sid < 1000
                               and abs(s_.size - mover.size) <= max(2, 0.2 * mover.size)]
                    other_ = min(others_, key=lambda s_: abs(s_.centre[0] - tcen[0]) + abs(s_.centre[1] - tcen[1])) if others_ else None
                if other_ is None:
                    continue
                h.meet = (sid, other_.sid)
                h.t_cells = frozenset(other_.cells); tcells = set(h.t_cells); tcen = other_.centre
                solids.discard(sid); solids.discard(other_.sid)

                def dist(state, _a=sid, _b=other_.sid):
                    ca_, cb_ = state.get(_a), state.get(_b)
                    if not ca_ or not cb_:
                        return 999.0
                    a_, b_ = centre(ca_), centre(cb_)
                    return abs(a_[0] - b_[0]) + abs(a_[1] - b_[1])
            kind_of_sid = {s_.sid: s_.body for s_ in tr.prev}
            replans = 0
            tried_here = None
            fit_alts = 0
            unblocked_here = False
            split_out = False
            while replans < 8 and not split_out:
                comps = tr.prev
                # (the screen may have moved since the reading was placed: its
                # target is where the reading now says)
                if getattr(h, "_scrolled", 0) != getattr(play_level, "scrolls", 0):
                    h._scrolled = getattr(play_level, "scrolls", 0)
                    tcen = h.target_centre()
                    tcells = set(getattr(h, "t_cells", ()))
                    if tcells and h.kind in ("cover", "align", "inside"):
                        tcen = centre(tcells)
                if fit_sids:
                    fit_sids, live_t = resolve_fit()
                    if not fit_sids:
                        break
                    h.t_cells = live_t; h.target = live_t
                    tcen = h.target_centre(); tcells = set(live_t)
                    mover = next((s_ for s_ in comps if s_.sid in fit_sids), mover); sid = mover.sid
                marks = selected_hosts(comps)
                start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
                kind_of_sid.update({s_.sid: s_.body for s_ in comps})
                sel0 = [x_ for x_ in (model.selected_now(comps),) if x_ is not None]
                start["_sel"] = sel0[0] if sel0 else None
                succ = make_successors(model, comps, solids, keys, sid, tcen, frame=f, target_cells=tcells)
                t0 = time.time()
                tol = getattr(h, "tol", 1.0)
                if fit_sids:
                    # fitted = the marks lie on the marks: every one of the
                    # mover's marks on some target mark (two dock on two of four)
                    def goal_t(st, _fit=fit_sids, _tc=tcells):
                        # (a pair of marks on a pair of marks is a docking: the
                        # rest of the piece's marks may stay free for others)
                        on_ = 0; have_ = 0
                        for f_ in _fit:
                            cs = set(st.get(f_, ()))
                            if not cs:
                                continue
                            for mk_ in _marks_of(cs):
                                have_ += 1
                                if len(mk_ & _tc) >= 0.9 * len(mk_):
                                    on_ += 1
                        return bool(_tc) and have_ > 0 and on_ >= min(2, have_)
                elif getattr(h, "over", False):
                    # reached = the thing's centre lies within the mover's box
                    # (the thing's centre where it stands NOW: one shoved away
                    # by the arrival was not reached)
                    def goal_t(st, _sid=sid, _tc=tcen, _tsid=t_sid_):
                        cs = st.get(_sid)
                        if not cs:
                            return False
                        if _tsid is not None and st.get(_tsid):
                            _tc = centre(st[_tsid])
                        b_ = _bbox(cs)
                        return b_[0] - 0.5 <= _tc[0] <= b_[2] + 0.5 and b_[1] - 0.5 <= _tc[1] <= b_[3] + 0.5
                elif h.kind == "inside" and tcells:
                    # INSIDE = every cell of the mover within the target's box
                    # (a block in a pen twice its size: its centre need not be
                    # the pen's — anywhere within is in)
                    tb_in_ = _bbox(tcells)

                    def goal_t(st, _sid=sid, _tb=tb_in_):
                        cs = st.get(_sid)
                        if not cs:
                            return False
                        return all(_tb[0] <= r <= _tb[2] and _tb[1] <= c <= _tb[3] for (r, c) in cs)
                elif h.kind == "cover" and tcells:
                    # covered = the mover's cells lie on the target's (a fit
                    # of the same shape), or its centre does; marks docking
                    # onto like marks are covered when the MOVER'S cells all
                    # lie on target marks (two marks dock on two of four)
                    docking_ = "marks onto like marks" in h.note

                    def goal_t(st, _tol=tol, _sid=sid, _tc=tcells, _dock=docking_):
                        cs = st.get(_sid)
                        if cs and _dock and len(set(cs) & _tc) >= 0.9 * len(cs):
                            return True
                        if cs and not _dock and len(set(cs) & _tc) >= 0.9 * len(_tc):
                            return True
                        return dist(st) < _tol
                else:
                    def goal_t(st, _tol=tol):
                        return dist(st) < _tol
                # (a bounded search: a level with many movers and many readings
                # must not spend minutes on each — 80k states, and half that
                # once a search has already come up empty this level)
                _claims.begin()                    # (the claims this plan consults are credited or struck by its steps)
                p = plan_general(start, succ, goal_t, heuristic=lambda st: dist(st) / 3.0, max_states=80000 if not play_level.last.get("empty_search") else 4000,
                                 max_seconds=min(90.0, max(5.0, _level_seconds() / 4.0)))
                if p and verbose and _ENV.get("PLAN_DEBUG"):
                    # (walk the plan through the model: what each step is believed to do)
                    st_ = start
                    for a_ in p:
                        nx_ = next((n_ for aa_, n_ in succ(st_) if aa_ == a_), None)
                        if nx_ is None:
                            print(f"      [walk] {a_}: no such successor"); break
                        ch_ = [(o_, _bbox(nx_[o_])[:2]) for o_ in nx_ if not isinstance(o_, str) and nx_[o_] != st_.get(o_)]
                        print(f"      [walk] {a_}: sel {nx_.get('_sel')} changed {ch_[:6]} whys {dict(list(make_successors.whys.items())[:4])}")
                        st_ = nx_
                if p is None and model.slide() is not None and not getattr(model, "_no_slide", False) and getattr(model, "gravity", None) is None:
                    # (a GRAVITY read off the frames is no theory on one sight: a
                    # plan without it walks into the pit — tw05 L3)
                    model._no_slide = True
                    try:
                        succ = make_successors(model, comps, solids, keys, sid, tcen, frame=f, target_cells=tcells)
                        p = plan_general(start, succ, goal_t, heuristic=lambda st: dist(st) / 3.0, max_states=20000)
                        if p:
                            note(f"the slide {model.slide()} is a theory on one sight: no way with it — trying the way without it")
                    finally:
                        model._no_slide = False
                if p is None and (getattr(plan_general, "last", {}) or {}).get("states", 0) < 40000 and not getattr(plan_general, "last", {}).get("timed_out"):
                    # A WALL CARRIED FROM THE LAST LEVEL IS A THEORY OF THAT LEVEL:
                    # the search ran out of room, not time, and a colour the
                    # ground refused THERE may not refuse here (tw03 L3: the line
                    # the player could not cross on L1-2 is crossed on L3).  The
                    # way without each carried refusal is sought in turn; a plan
                    # found is followed, and its first step into that colour is
                    # the re-test — one press, which the observer reads as the
                    # ground allowing or refusing (this level's evidence outranks
                    # the carried).  Each colour is doubted once a level.
                    carried_ref_ = {int(c_) for c_ in ((carry or {}).get("ground_refused") or {})}
                    own_ref_ = {c_ for c_, n_ in model.ground_refused.items() if n_ > ((carry or {}).get("ground_refused") or {}).get(c_, 0)}
                    doubted_ = play_level.last.setdefault("walls_doubted", set())
                    # (and, for a thing the plan must SHOVE: terrain of a colour no
                    # shove was seen to cross — the plan stops a shoved thing at
                    # it; the way across it is tried the same way, one push the
                    # experiment)
                    known_t_ = set(model.ground_ok) | set(getattr(model, "slid_over", {}) or {}) | set(model.flat)
                    shove_cands_ = sorted({s_.body for s_ in tr.prev if "terrain" in s_.tags and s_.body not in known_t_ and s_.size > 2}) if model.pushable else []
                    cands_ = [("ground", c_) for c_ in sorted(carried_ref_ & model.blocking())] + [("shove", c_) for c_ in shove_cands_]
                    for how_, c_ in cands_:
                        if (how_, c_) in doubted_ or (how_ == "ground" and c_ in own_ref_):
                            continue        # (refused on THIS level too: no theory of another level)
                        doubted_.add((how_, c_))
                        if how_ == "ground":
                            model.ground_ok.add(c_)
                        else:
                            model._shove_ok = {c_}
                        try:
                            succ2_ = make_successors(model, comps, solids, keys, sid, tcen, frame=f, target_cells=tcells)
                            p2_ = plan_general(start, succ2_, goal_t, heuristic=lambda st: dist(st) / 3.0, max_states=20000)
                        finally:
                            if how_ == "ground":
                                model.ground_ok.discard(c_)
                            else:
                                model._shove_ok = set()
                        if p2_:
                            if how_ == "ground":
                                note(f"no way under the carried laws: colour {c_} refused the mover on the last level — the way through it is tried, the first step into it the re-test ({len(p2_)} steps)")
                            else:
                                note(f"no way with the shoved thing stopping at colour {c_}, never crossed: the way across it is tried, the first push into it the experiment ({len(p2_)} steps)")
                            p, succ = p2_, succ2_
                            break
                if p is None and (getattr(plan_general, "last", {}) or {}).get("states", 0) >= 40000:
                    play_level.last["empty_search"] = True
                if p is None and _ENV.get("PLAN_DEBUG"):
                    print("      [plan] frame now:")
                    for r_ in range(64):
                        print("      " + f"{r_:2d} " + "".join("0123456789abcdef"[int(x_)] for x_ in f[r_]))
                    # replay the best path and show what the search saw there
                    bp_ = (getattr(plan_general, "last", {}) or {}).get("best", (None, []))[1]
                    st_ = start
                    for a_ in bp_:
                        st_ = next((n_ for aa_, n_ in succ(st_) if aa_ == a_), st_)
                    print(f"      [plan] mover cells {sorted(st_[sid]) if sid in st_ else None} tcells {sorted(tcells)[:50]}")
                    print(f"      [plan] best path {bp_} mover {_bbox(st_[sid]) if sid in st_ else None} target {tcen} dist {dist(st_)} succ {[(a_, _bbox(n_[sid])[:2] if sid in n_ else None) for a_, n_ in succ(st_)]} whys {make_successors.whys} solids {sorted(solids)}")
                note(f"reading: {h.note[:70]} — plan {'none' if p is None else str(len(p)) + ' steps'}")
                if h.note not in play_level.ledger["hyps_tried"]:
                    play_level.ledger["hyps_tried"].append(h.note)
                if verbose:
                    print(f"   plan for {h}: {p if p is None else len(p)} steps ({time.time()-t0:.1f}s) {getattr(plan_general, 'last', {}).get('states')} states; best {getattr(plan_general, 'last', {}).get('best', (None,))[0]} sel={start['_sel']} marks={marks} tsid={t_sid_} blocking={sorted(model.blocking())} walls={len(getattr(succ, 'wall_cells', ()) or ())} plan={p[:8] if p else None}")
                    if p is None and not getattr(plan_general, 'last', {}).get('states'):
                        try:
                            s0_ = succ(start)
                            print(f"   (no successors from the start: {len(s0_)}; mover S{sid} in state {sid in start}; rel law {model.relative_law(m_kind)}; solids {len(getattr(succ, 'solids', ()) or ())}; mover kind {m_kind} in blocking {m_kind in model.blocking()})")
                        except Exception as ex_:
                            print(f"   (successor check failed: {ex_!r})")
                    if p is None:
                        print("      successors at start:", [a_ for a_, _ in succ(start)][:14], "whys", {str(k_): w_ for k_, w_ in getattr(make_successors, 'whys', {}).items()}, "solids", len(solids), "state", len(start), "max_size", dict(model.fixed.max_size), "min_size", dict(model.fixed.min_size))
                        for k_ in model.fixed.keys():
                            print("        ", k_, model.fixed.moving(k_))
                if p is None and not fit_sids and model.key_selected and model.push_vec and m_kind in model.pushable \
                        and any(s_.sid != sid and s_.sid < 1000 and s_.body in model.select_kinds for s_ in comps) \
                        and (sid, mover.bbox[:2]) not in play_level.last.setdefault("shove_tried", set()) and len(play_level.last["shove_tried"]) < 6:
                    # SENT THERE BY A STRIKE: no way by its own moves — a
                    # shove by another piece may carry it where its moves
                    # cannot go (across what it may not enter)
                    play_level.last["shove_tried"].add((sid, mover.bbox[:2]))
                    pool_ = list(play_level.last.get("hyps", hyps)) if isinstance(play_level.last, dict) else hyps
                    t0 = time.time()
                    p = plan_by_shove(model, succ, start, sid, goal_t, dist, depth=2, docked=_docked_now(pool_), verbose=verbose)
                    if verbose:
                        print(f"   shove plan for {h}: {p if p is None else len(p)} steps ({time.time()-t0:.1f}s) plan={p[:12] if p else None}")
                    if p:
                        note(f"no way by its own moves: S{sid} is walked to a launch and struck by another piece — {len(p)} steps")
                if p is None:
                    # WHAT THE PLAN RELIES ON IS CONFIRMED FIRST: a key whose law
                    # here is only carried from the last level (never seen to
                    # hold on this one) is pressed once before "no way" is
                    # concluded — a law by kind may not be each thing's law
                    # (two like things that move in mirrored ways)
                    probed_ = play_level.last.setdefault("probed_keys", set())
                    # (only a key with NO law here — never pressed on this level,
                    # nothing carried for it: a carried law is verified by use,
                    # the plan replanning at its first surprise; a key of which
                    # nothing is known is what "no way" may be missing)
                    prov_ = [k_ for k_ in keys if k_ not in model.undo_keys and k_ not in probed_
                             and not model.fixed.moving(k_) and model.fixed.presses.get(k_, 0) == 0]
                    if prov_:
                        for k_ in prov_:
                            probed_.add(k_)
                            note(f"no plan under the carried laws: what does {k_} do on this level?")
                            step(k_)
                            if obs.levels_completed > lv0:
                                return True, steps, "try-keys", obs
                        replans += 1
                        continue
                    if early:
                        break           # (looked first and no way under what is known: the repertoire's turn)
                    here_ = (next((s_.bbox[:2] for s_ in tr.prev if s_.sid == sid), None), model.selected_now(tr.prev))
                    if replans == 0 or tried_here != here_:
                        # NO PLAN FROM HERE: a key that did nothing where it
                        # was tried may do something here — each once (here =
                        # the mover's place AND what is selected)
                        tried_here = here_
                        n_try = 0
                        # (a key whose law under the present selection is not
                        # known yet — pressed under another — is pressed once
                        # here to learn it, before the silent keys)
                        sel_h = model.selected_now(tr.prev) if model.select_color is not None else None
                        for k_ in keys:
                            if k_ in model.undo_keys or sel_h is None or not model.fixed.moving(k_):
                                continue
                            if model.fixed.moving(k_, sel_h) or (k_, sel_h) in model.fixed.ctx:
                                continue
                            note(f"no plan from here: what does {k_} do with S{sel_h} selected?")
                            step(k_); n_try += 1
                            if obs.levels_completed > lv0:
                                return True, steps, "try-keys", obs
                        if n_try:
                            replans += 1
                            continue            # the laws under this selection are known now: plan again
                        for k_ in keys:
                            if k_ in model.undo_keys or model.fixed.moving(k_):
                                continue
                            note(f"no plan from here: trying silent key {k_} at this place")
                            step(k_); n_try += 1
                            if obs.levels_completed > lv0:
                                return True, steps, "try-keys", obs
                        if verbose:
                            print(f"   tried {n_try} silent keys here; keys {[(k_, dict(model.fixed.moving(k_))) for k_ in keys]} refused_at {sorted(model.refused_at)} mover {[s_.bbox for s_ in tr.prev if s_.sid == sid]}")
                        if n_try:
                            replans += 1
                            continue
                    # UNBLOCK: the way is barred by a THING (not a wall colour):
                    # what opens it?  The blocker is read off the search's
                    # best state; then, once each, the affordances a human
                    # would try — stand on each small odd thing within reach
                    # (a pad), click the blocker, press each key beside it —
                    # and after each the blocker is looked at again
                    if not unblocked_here and h.kind != "collect":
                        unblocked_here = True
                        blockers = blockers_of(succ, start, sid)
                        if blockers:
                            note(f"the way is barred by S{blockers[0]}: what opens it?")
                            if unblock(blockers, sid):
                                replans += 1
                                continue
                    break
                diverged = False
                arrived_gone = False
                play_level.ledger["reading"] = h.note
                play_level.ledger["plan_len"] = len(p)
                if h.note not in play_level.ledger["hyps_tried"]:
                    play_level.ledger["hyps_tried"].append(h.note)

                def _record_div(i_, k_, sid_, pred_, now_, comps_before_, reason_=""):
                    """The ledger's record of a step that did not go as predicted."""
                    # THE DIRECTED RE-TEST (the claim ledger): which claim, consulted
                    # for this very step, would have predicted what the world did?
                    # Each claim the step consulted is silenced in turn and the step
                    # re-predicted from the state before it; the one whose silence
                    # makes the prediction match is struck alone — the others are
                    # credited.  When none explains it, every claim the plan
                    # consulted is struck together (a record, not a diagnosis).
                    culprit_ = None
                    try:
                        led_c = _claims.current()
                        st_b = {**{s_.sid: frozenset(s_.cells) for s_ in comps_before_ if "hud" not in s_.tags},
                                "_sel": model.selected_now(comps_before_) if model.select_color is not None else None}

                        def predict_():
                            for a_, n_ in succ(st_b):
                                if a_ == k_:
                                    return n_.get(sid_)
                            return None
                        cands_ = led_c.consulted_by(predict_)
                        obs_cells_ = now_.get(sid_) if now_ else None
                        if cands_ and obs_cells_ is not None:
                            for name_ in sorted(cands_):
                                pr_ = led_c.with_off(name_, predict_)
                                if pr_ is not None and _bbox(pr_)[:2] == _bbox(obs_cells_)[:2]:
                                    culprit_ = name_; break
                    except Exception as ex_:
                        if verbose:
                            print(f"   (directed re-test failed: {ex_!r})")
                    if culprit_ is not None:
                        led_c.strike(culprit_, 2)
                        for n_ in led_c.touched:
                            if n_ != culprit_:
                                led_c.credit(n_)
                        note(f"the step went wrong where claim '{culprit_}' said otherwise: silenced for this level")
                        if verbose:
                            print(f"   claim '{culprit_}' explains the divergence (with it off the prediction matches): struck twice")
                    else:
                        # (something stood where the mover was sent: the lesson is
                        # that thing's — a wall, a kind not yet known to block — and
                        # no claim is struck for it; only a divergence with nothing
                        # in the way and no claim to explain it strikes them all)
                        in_way_ = False
                        try:
                            pc_ = set(pred_.get(sid_) or ()) if pred_ else set()
                            in_way_ = any(x_.sid != sid_ and "hud" not in x_.tags and (set(x_.cells) & pc_) for x_ in comps_before_) if pc_ else False
                        except Exception:
                            in_way_ = False
                        if not in_way_:
                            _claims.step(False)         # (the world left the plan for no visible reason: every claim the plan consulted is struck)
                    try:
                        sel_ = model.selected_now(tr.prev) if model.select_color is not None else None
                        law_ = None
                        if isinstance(k_, str):
                            mv_ = model.fixed.moving(k_, model.sel_ctx(sel_) if sel_ is not None else None)
                            law_ = (k_, tuple(mv_[sid_]) if sid_ in mv_ else None)
                        why_ = (play_level.ledger.get("whys_step") or {}).get(k_)
                        if why_ is not None:
                            law_ = (law_, "why", why_ if isinstance(why_, (str, tuple)) else str(why_)[:80])
                        d_ = _ledger.divergence(
                            i_ + 1, k_, sid_, kind, pred_.get(sid_) if pred_ else None, now_.get(sid_),
                            next((frozenset(x_.cells) for x_ in comps_before_ if x_.sid == sid_), None), tr.prev, law=law_, reason=reason_)
                        d_["i_action"] = int(getattr(play_level, "steps", 0) or 0)      # (the level's action count when it was seen: the replay's frame)
                        # WHAT THE PLAN KNEW (Tushar, 37o: the insight must come
                        # off the journal, not a re-run): the beliefs the step was
                        # predicted from — what blocks, what pushes and how far,
                        # which kinds on the frame have never been asked, which
                        # kinds travel on their own — so that "what did the law
                        # buy us" is answerable from the record alone
                        try:
                            present_k_ = {s_.body for s_ in comps_before_ if s_.sid < 1000 and "hud" not in s_.tags and "terrain" not in s_.tags}
                            asked_k_ = set()
                            for k2_ in model.fixed.eff:
                                asked_k_ |= {kind_of_sid.get(s2_) for s2_ in model.fixed.eff[k2_]}
                            asked_k_ |= set(model.ground_ok) | set(model.flat) | set(model.pushable) | set(model.wall_kinds) | set(model.blocking())
                            asked_k_ |= {a_ for pr_ in model.passable for a_ in pr_} | set(getattr(model, "agents", ()) or ())
                            d_["knew"] = {"blocking": sorted(model.blocking()), "pushable": sorted(model.pushable),
                                          "push_vec": {str(kk_): vv_[0] for kk_, vv_ in model.push_vec.items()},
                                          "push_stopped": {str(kk_): bool(vv_) for kk_, vv_ in model.push_stopped.items()},
                                          "agents": sorted(getattr(model, "agents", ()) or ()),
                                          "unasked": sorted(k3_ for k3_ in present_k_ if k3_ not in asked_k_ and k3_ is not None),
                                          "plan_len": len(p) if p else 0, "plan_states": (getattr(plan_general, "last", {}) or {}).get("states")}
                        except Exception:
                            pass
                        play_level.ledger["divs"].append(d_)
                    except Exception as ex_:
                        if verbose:
                            print(f"   (ledger: {ex_!r})")
                # (the docking livery, once known, is counted before the plan:
                # a docking shows as more cells of it afterwards)
                dock_col_ = getattr(model, "dock_colour", None)
                dock_n0_ = sum(1 for (r_, c_) in (tcells or ()) if 0 <= r_ < 64 and 0 <= c_ < 64 and int(f[r_, c_]) == dock_col_) if dock_col_ is not None else None
                for i, k in enumerate(p):
                    if i > 0:
                        _claims.step(True)          # (the step before held: the claims the plan consulted are credited)
                    # predicted mover position
                    pred_state = None
                    marks_now = selected_hosts(tr.prev)
                    sel_now = [x_ for x_ in (model.selected_now(tr.prev),) if x_ is not None]
                    succ_list_ = succ({**{s.sid: frozenset(s.cells) for s in tr.prev if "hud" not in s.tags}, "_sel": sel_now[0] if sel_now else None})
                    for act, new in succ_list_:
                        if act == k:
                            pred_state = new; break
                    play_level.ledger["whys_step"] = dict(getattr(make_successors, "whys", None) or {})      # (the planner's reason for each offered step, as of now)
                    if verbose and _ENV.get("PLAN_DEBUG"):
                        print(f"      [pred] {k} sel {sel_now}: {[(a_, _bbox(n_[sid])[:2] if sid in n_ else None, n_.get('_sel')) for a_, n_ in succ_list_]} whys {make_successors.whys} ctx {[(kk, ss, dict(cc.moving(kk))) for (kk, ss), cc in model.fixed.ctx.items() if kk == k]}")
                        if pred_state is not None:
                            now0_ = {s.sid: frozenset(s.cells) for s in tr.prev if "hud" not in s.tags}
                            print(f"      [pred] changed: {[(o_, _bbox(now0_[o_])[:2], _bbox(pred_state[o_])[:2]) for o_ in pred_state if not isinstance(o_, str) and o_ in now0_ and pred_state[o_] != now0_[o_]]}")
                    note(f"plan step {i+1}/{len(p)} ({k}) toward: {h.note[:60]}")
                    stays_ = pred_state is not None and sid in pred_state and pred_state[sid] == next((frozenset(s_.cells) for s_ in tr.prev if s_.sid == sid), None)
                    comps_step_ = list(tr.prev)             # (where things stood before THIS step: a refusal is judged against it)
                    fatal_here_ = (_frame_key(obs), _key_id(k)) in getattr(play_level, "fatal", set())
                    if fatal_here_:
                        note(f"not {k} from here: it was the death last time")
                        if verbose:
                            print(f"   refused: {k} from this very frame was the fatal press of the last attempt")
                    if not stays_ and (fatal_here_ or not _step_is_safe(model, tr, sid, k, allow=tcells)):
                        note(f"step {i+1} {k}: the ground ahead changed since the plan — replanning")
                        _record_div(i, k, sid, pred_state, {s_.sid: frozenset(s_.cells) for s_ in tr.prev}, comps_step_, "the ground ahead changed since the plan")
                        diverged = True; break
                    step(k)
                    if obs.levels_completed > lv0:
                        play_level.last["won_hyp"] = h
                        return True, steps, f"plan:{i+1}/{len(p)}", obs
                    if verbose:
                        _m = next((s_ for s_ in tr.prev if s_.sid == sid), None)
                        print(f"      step {i+1} {k}: mover S{sid} now {_m.bbox if _m else None}; predicted {(_bbox(pred_state[sid]) if pred_state and sid in pred_state else None)} sel {sel_now} marks {marks_now} law {dict(model.fixed.moving(k, sel_now[0] if sel_now else None)) if isinstance(k, str) else ''}")
                    if pred_state is None and isinstance(k, str):
                        # (no predicted state: the law under this selection says
                        # what should have moved — a thing that stayed is refused here)
                        law_ = model.fixed.moving(k, model.sel_ctx(sel_now[0]) if sel_now else None)
                        for sid_, e_ in law_.items():
                            if e_[0] != "translate":
                                continue
                            bw_ = next((x_.bbox for x_ in comps_step_ if x_.sid == sid_), None)
                            bn_ = next((x_.bbox for x_ in tr.prev if x_.sid == sid_), None)
                            if bw_ is not None and bn_ is not None and bn_[:2] == bw_[:2]:
                                model.refuse((kind_of_sid.get(sid_), bw_[0], bw_[1], tuple(e_[1])), size=(bw_[2] - bw_[0] + 1, bw_[3] - bw_[1] + 1))
                    if pred_state is not None:
                        now = {s.sid: frozenset(s.cells) for s in tr.prev}
                        # A REFUSED STEP IS REMEMBERED BY PLACE: the mover did not
                        # move where the plan said it would — the successor
                        # function must not offer that step here again
                        if sid in pred_state and now.get(sid) is not None:
                            b_now = _bbox(now[sid]); b_pred = _bbox(pred_state[sid])
                            b_was = next((x_.bbox for x_ in comps_step_ if x_.sid == sid), None)
                            # (A PUMP WITH NOTHING TO GIVE: the key's law shrinks a
                            # thing that is gone or at its least — the refusal is
                            # that thing's emptiness, not this place's)
                            pump_empty_ = False
                            for psid_, e_ in model.fixed.moving(k).items():
                                if e_[0] == "retract" and (now.get(psid_) is None or len(now[psid_]) <= model.fixed.min_size.get(psid_, 0)):
                                    pump_empty_ = True
                            if pump_empty_:
                                note(f"step {i+1} {k}: refused — what it draws on is spent, not this place")
                            elif b_was is not None and b_now[:2] == b_was[:2] and b_pred[:2] != b_was[:2]:
                                _record_div(i, k, sid, pred_state, now, comps_step_, "the mover stayed where the plan moved it")
                                # (the mover, what it rides on, and what moves
                                # with it — not everything the law named: a wrong
                                # law's other victims are not refusals)
                                own_ = {sid, sid // 1000 if sid >= 1000 else sid}
                                for b_ in model.bodies():
                                    if sid in b_:
                                        own_ |= set(b_)
                                for sid_ in pred_state:
                                    if isinstance(sid_, str) or sid_ not in now or sid_ not in own_:
                                        continue
                                    bw_ = next((x_.bbox for x_ in comps_step_ if x_.sid == sid_), None)
                                    bp_ = _bbox(pred_state[sid_]); bn_ = _bbox(now[sid_])
                                    if bw_ is not None and bn_[:2] == bw_[:2] and bp_[:2] != bw_[:2]:
                                        v_ = (bp_[0] - bw_[0], bp_[1] - bw_[1])
                                        model.refuse((kind_of_sid.get(sid_), bw_[0], bw_[1], v_), size=(bw_[2] - bw_[0] + 1, bw_[3] - bw_[1] + 1))
                                        if verbose and _ENV.get("PLAN_DEBUG"):
                                            print(f"      [refuse] S{sid_} kind {kind_of_sid.get(sid_)} at {bw_[:2]} v {v_}; refused_at now {len(model.refused_at)}")
                        if fit_sids:
                            # MARKS ARE VERIFIED AS A SET, NOT BY NAME: two like
                            # marks may trade names, one may take the colour of
                            # the mark it docked on — the predicted cells must be
                            # occupied by something that is not the ground
                            pred_cells = set()
                            for f_ in fit_sids:
                                pred_cells |= set(pred_state.get(f_, ()))
                            occ_ = sum(1 for (r_, c_) in pred_cells if 0 <= r_ < 64 and 0 <= c_ < 64 and int(f[r_, c_]) != bg)
                            if pred_cells and occ_ < 0.9 * len(pred_cells):
                                note(f"step {i+1} {k}: the marks are not where predicted — replanning")
                                _record_div(i, k, sid, pred_state, now, comps_step_, "the marks are not where predicted")
                                diverged = True; break
                            continue
                        if sid in pred_state and now.get(sid) is None and tcells and pred_state[sid] and \
                                all(min(r_ for r_, _ in tcells) - 1 <= r_ <= max(r_ for r_, _ in tcells) + 1 and
                                    min(c_ for _, c_ in tcells) - 1 <= c_ <= max(c_ for _, c_ in tcells) + 1 for (r_, c_) in pred_state[sid]):
                            # THE MOVER IS IN ITS FRAME AND READ AS ONE WITH IT: it
                            # went where the plan sent it and the tracker sees frame
                            # and thing as one — the reading is achieved
                            note(f"the thing sits in its frame (read as one with it now): {h.note[:50]} — achieved")
                            if verbose:
                                print(f"   arrived: mover S{sid} gone into the target's box — reading done")
                            h._done = True
                            arrived_gone = True
                            break
                        # (EXPERIMENT, off by default — SPLIT_WITHDRAW: a press that
                        # split the mover ends this reading and the frame is read
                        # again.  Tried as the explanation of what the wrong
                        # 'retract' law stood in for on tw01 (session 37o); alone
                        # it loses tw01 L2 — the reading about the KIND was fine,
                        # its mover re-resolves to a half.  Kept for the A/B.)
                        if _ENV.get("SPLIT_WITHDRAW") and sid in pred_state and any(e_.kind == "born" and e_.data.get("split_from") == sid for e_ in (tr.last_events or [])):
                            note(f"step {i+1} {k}: the mover split into two things — this reading is withdrawn; the frame is read again")
                            if verbose:
                                print(f"   step {i+1} {k}: mover S{sid} split — reading withdrawn, looking again")
                            h._done = True
                            look_again()
                            play_level.redo_pursuit = True
                            split_out = True
                            break
                        # WHAT STANDS WHERE THE PLAN PUT THE MOVER IS THE MOVER: two
                        # like things may trade names between the plan and the
                        # press (a pair of blocks, one in the pen — tw01 L3 looped
                        # on "not where predicted" with the other block's name);
                        # a thing of the mover's kind and size at the predicted
                        # place means the step held, under whichever name
                        if sid in pred_state and (now.get(sid) is None or off_centre(now[sid], pred_state[sid])) and not _ENV.get("NO_LIKE_NAME"):
                            like_ = None
                            for o_sid_, o_cells_ in now.items():
                                if isinstance(o_sid_, str) or o_sid_ == sid or o_sid_ >= 1000:
                                    continue
                                if kind_of_sid.get(o_sid_) == kind_of_sid.get(sid) and len(o_cells_) == len(pred_state[sid]) and not off_centre(o_cells_, pred_state[sid]):
                                    like_ = o_sid_; break
                            if like_ is not None:
                                if verbose:
                                    print(f"   step {i+1} {k}: the mover's name traded with S{like_} — the step held under that name")
                                note(f"step {i+1} {k}: a like thing stands where the plan put the mover — the step held, the name is now S{like_}")
                                for st_ in (pred_state, now):
                                    if sid in st_ and like_ in st_:
                                        st_[sid], st_[like_] = st_[like_], st_[sid]
                                continue
                        if sid in pred_state and (now.get(sid) is None or off_centre(now[sid], pred_state[sid])):
                            if verbose:
                                print(f"   step {i+1} {k}: mover not where predicted; replanning")
                            note(f"step {i+1} {k}: the mover is not where predicted — replanning")
                            _record_div(i, k, sid, pred_state, now, comps_step_, "the mover is not where predicted")
                            diverged = True; break
                if split_out:
                    break
                if arrived_gone:
                    if not early:
                        look_again()            # (the template's chase keeps the first frame's readings: the marks are resolved by place as they stand)
                    play_level.redo_pursuit = True
                    break
                if not diverged:
                    # THE PLAN ENDED: is the goal reached in the world?  A plan
                    # whose last steps were refused (an edge) ended short —
                    # replan from here (the refusal is remembered by place)
                    st_end = {s_.sid: frozenset(s_.cells) for s_ in tr.prev if "hud" not in s_.tags}
                    st_end["_sel"] = model.selected_now(tr.prev)
                    try:
                        short_ = not goal_t(st_end)
                    except Exception:
                        short_ = False
                    docked_now = False
                    if fit_sids:
                        # DOCKED: the mover's marks — by name, or the marks of its
                        # piece under whatever colour they now wear (a part is
                        # renamed when it changes colour) — lie on the target's
                        # marks and have taken another colour
                        host_ = getattr(h, "_fit_host", None)
                        dc_ = set()
                        for s_ in tr.prev:
                            own_ = s_.sid in fit_sids or (host_ is not None and s_.sid >= 1000 and s_.sid // 1000 == host_ and s_.size <= 40)
                            if own_ and s_.body != m_col:
                                dc_ |= set(s_.cells)
                        if dc_ and tcells and len(dc_ & set(tcells)) >= 0.9 * len(dc_) and len(dc_) >= 4:
                            docked_now = True; short_ = False
                    if verbose and fit_sids:
                        print(f"      plan end: short={short_} marks {[(f_, _bbox(st_end[f_]) if f_ in st_end else None) for f_ in fit_sids]} target {sorted(tcells)[:2]}.. n={len(tcells)} hidden {[s_.sid for s_ in tr.prev if 'hidden' in s_.tags]}")
                    if short_ and replans < 6:
                        note("the plan ended short of its goal — replanning")
                        diverged = True
                if not diverged and fit_sids:
                    # DOCKED: the marks lie on the marks and took their colour —
                    # this reading is achieved; the piece is now part of what it
                    # docked on, and the level goes on with the next reading
                    # (the frame is read again: new fits show on the joined thing)
                    now_m = [s_ for s_ in tr.prev if s_.sid in fit_sids]
                    # THE DOCKING LIVERY: the first docking teaches the colour a
                    # docked pair wears; from then on a docking is more cells of
                    # that colour on the board, and nothing else counts as one
                    if dock_col_ is not None:
                        # (docked = the marks aimed at now wear the docking colour —
                        # counted on the target's marks: a mover leaving another
                        # docking behind loses that colour elsewhere)
                        dock_n1_ = sum(1 for (r_, c_) in (tcells or ()) if 0 <= r_ < 64 and 0 <= c_ < 64 and int(f[r_, c_]) == dock_col_)
                        docked_now = dock_n1_ >= dock_n0_ + 4 and (not dc_ or len(dc_ & set(tcells)) >= 4)
                    elif docked_now or (now_m and all(s_.body != m_col for s_ in now_m)):
                        cols_ = Counter(int(f[r_, c_]) for (r_, c_) in (dc_ if dc_ else set().union(*[set(s_.cells) for s_ in now_m])) if 0 <= r_ < 64 and 0 <= c_ < 64)
                        if cols_:
                            model.dock_colour = cols_.most_common(1)[0][0]
                            note(f"a docked pair wears colour {model.dock_colour}: that is what a docking looks like")
                        docked_now = True
                    if docked_now:
                        h._done = True
                        # (the docked piece and what it docked on are one body now)
                        hf_, tf_ = getattr(h, "_fit_host", None), getattr(h, "_fit_target", None)
                        # (whether the docked pair now moves as one is for a press
                        # to show: co-motion is learned, never assumed)
                        note(f"docked: the marks took another colour — this reading is achieved; on to the next")
                        if verbose:
                            print(f"   docked: {h.note[:60]} — reading done")
                        if not early:
                            look_again()        # (as above: the early chase keeps its readings)
                        play_level.redo_pursuit = True
                        break
                if not diverged:
                    # THE PLAN ARRIVED AND NOTHING HAPPENED: some games judge on
                    # the action after the arrival — step off and back once
                    # (the arrival is made again) before the reading is dropped
                    if fit_sids and model.key_rotate and fit_alts < 1:
                        # A FIT THAT DID NOT TAKE: like marks may be alike only
                        # in the drawing — the other way round is the other fit
                        # (a half turn), tried once
                        tk = next(iter(model.key_rotate))
                        note("the marks are on the marks and nothing happened: the other way round (a half turn), then dock again")
                        step(tk); step(tk)
                        if obs.levels_completed > lv0:
                            play_level.last["won_hyp"] = h
                            return True, steps, "plan+", obs
                        fit_alts += 1
                        replans += 1
                        continue
                    # ONE OF SEVERAL LIKE THINGS IS IN; THE NEXT IS PURSUED: a reading
                    # that takes a thing INTO a place larger than it, achieved while
                    # the level goes on, with another thing of the mover's kind not
                    # yet inside — the same reading, that thing as the mover (tw01:
                    # two blocks into a pen; the level waits on the second)
                    if h.kind == "inside" and tcells and not fit_sids:
                        tb_in2_ = _bbox(tcells)
                        others_in_ = [s_ for s_ in tr.prev if s_.sid < 1000 and s_.body == h.m_color and "hud" not in s_.tags
                                      and 0.5 * h.m_size <= s_.size <= 2 * h.m_size
                                      and not all(tb_in2_[0] <= r_ <= tb_in2_[2] and tb_in2_[1] <= c_ <= tb_in2_[3] for (r_, c_) in s_.cells)]
                        mover_now_ = next((s_ for s_ in tr.prev if s_.sid == sid), None)
                        if others_in_ and mover_now_ is not None and all(tb_in2_[0] <= r_ <= tb_in2_[2] and tb_in2_[1] <= c_ <= tb_in2_[3] for (r_, c_) in mover_now_.cells) \
                                and getattr(h, "_members_done", 0) < 6:
                            h._members_done = getattr(h, "_members_done", 0) + 1
                            nxt_ = min(others_in_, key=lambda s_: abs(s_.centre[0] - mover_now_.centre[0]) + abs(s_.centre[1] - mover_now_.centre[1]))
                            h.m_start = nxt_.centre
                            note(f"one of the like things is in; the next is pursued: {h.note[:50]}")
                            play_level.redo_pursuit = True      # (the readings again, this one with its next thing as the mover)
                            break
                    rweights.failed(h); failed_readings[_reading_id(h)] += 1          # arrived, nothing happened: the cues that proposed it were misleading here
                    if p and isinstance(p[-1], str) and p[-1] in model.key_selected:
                        v_last = model.key_selected[p[-1]]
                        inv = next((k_ for k_, v_ in model.key_selected.items() if v_ == (-v_last[0], -v_last[1])), None)
                        if inv is not None:
                            note("arrived, nothing happened: stepping off and back (some games judge on the next action)")
                            step(inv)
                            if obs.levels_completed > lv0:
                                play_level.last["won_hyp"] = h
                                return True, steps, "plan+", obs
                            step(p[-1])
                            if obs.levels_completed > lv0:
                                play_level.last["won_hyp"] = h
                                return True, steps, "plan+", obs
                    # A FIT MADE WHILE THE LEVEL GOES ON: the thing sits in
                    # its frame — this reading is done (until the frame
                    # stands empty again); the level waits on the others
                    # (a reading achieved while the level goes on — the thing
                    # in its frame, the figure on its marks — is done; the
                    # level waits on the other readings, which are chased next)
                    try:
                        st_now_ = {s_.sid: frozenset(s_.cells) for s_ in tr.prev if "hud" not in s_.tags}
                        st_now_["_sel"] = model.selected_now(tr.prev)
                        holds_ = bool(goal_t(st_now_))
                    except Exception:
                        holds_ = False
                    # (a fit is judged by the thing sitting in its frame; a cover
                    # by its own test — the marks under the figure's cells —
                    # not by nearness, which a bigger thing passes too easily)
                    if h.kind in ("align", "cover", "inside") and tcells and "marks onto" not in h.note and ((_template_reading(h) and _sits_in(h)) or (holds_ and h.kind == "cover")):
                        h._done = True
                        note(f"achieved while the level goes on: {h.note[:50]} — done; the others remain")
                        if not early:
                            look_again()        # (as above: the early chase keeps its readings)
                        play_level.redo_pursuit = True      # (the readings as they stand now: a frame emptied for this one is open again)
                    break
                replans += 1
            if getattr(play_level, "redo_pursuit", False) and depth < 8 and time.time() - play_level.t0 <= _level_seconds():
                # (the readings as they stand now, the template ones first)
                pool_ = list(play_level.last.get("hyps", hyps)) if isinstance(play_level.last, dict) else hyps
                return pursue(pool_, max_single=max_single, early=early, depth=depth + 1)
        # EVERY DOCKING MADE AND THE LEVEL GOES ON: like marks may be alike only
        # in the drawing — a pair that took but did not count is the wrong way
        # round.  One docked piece at a time is turned half round and docked
        # again (a planned fiddle: made, judged by the level, the next tried)
        pool_ = list(play_level.last.get("hyps", hyps)) if isinstance(play_level.last, dict) else hyps
        done_ = [h for h in pool_ if getattr(h, "_done", False) and "marks onto" in h.note and not getattr(h, "_flipped", False)
                 and getattr(h, "_fit_host", None) is not None]
        open_ = [h for h in pool_ if _template_reading(h) and not getattr(h, "_done", False) and not getattr(h, "_tried", False)]
        # (only once no mark is left free: while a mark of the marks' colour
        # still shows, a docking remains to be made, not remade)
        mc_ = done_[0].m_color if done_ else None
        free_left = [s_ for s_ in tr.prev if mc_ is not None and s_.size <= 16 and "hidden" not in s_.tags and "hud" not in s_.tags
                     and (s_.body == mc_ or (s_.sid >= 1000 and s_.sid % 1000 == mc_))]
        if done_ and not open_ and not free_left and model.key_rotate and depth < 8 and time.time() - play_level.t0 <= _level_seconds():
            h = max(done_, key=lambda h_: h_.prio)
            hf_ = h._fit_host
            if any(s_.sid == hf_ for s_ in tr.prev):
                tk = next(iter(model.key_rotate))
                note(f"every docking is made and the level goes on: one pair must be the wrong way round — turning a docked piece half round and docking it again ({h.note[:50]})")
                if verbose:
                    print(f"   flip: half-turn S{hf_} for {h.note[:60]}")
                if model.selected_now(tr.prev) != hf_:
                    step(("select", hf_))
                step(tk); step(tk)
                if obs.levels_completed > lv0:
                    play_level.last["won_hyp"] = h
                    return True, steps, "flip", obs
                h._flipped = True; h._done = False
                for h2 in pool_:
                    if getattr(h2, "_done", False) and "marks onto" in h2.note and hf_ in (getattr(h2, "_fit_host", None), getattr(h2, "_fit_target", None)):
                        h2._done = False        # (what was docked on the turned piece came undone)
                return pursue(pool_, max_single=max_single, early=early, depth=depth + 1)
        return None

    # --- THE LEGEND WITHIN (free information, read before anything is
    # pressed): a miniature at the centre of a lattice of tiles names the
    # tiles around it; the panels that agree teach the mapping, the panel
    # that breaks it is the puzzle, and a click on a tile (which changes its
    # colour — the first click shows) sets each wrong tile right
    if clickable:
        try:
            from agent.core.legend import read_panels, learn_map, violations, satisfies
            hud_rows_ = {r_ for s_ in tr.prev if "hud" in s_.tags for (r_, _) in s_.cells}
            panels_, pitch_, dims_ = read_panels(f, bg, hud_rows=hud_rows_)
        except Exception as ex_:
            panels_ = []
            if verbose:
                print(f"   legend: reading failed {ex_!r}")
        if panels_:
            # (a panel whose tiles are all of one colour is an untouched
            # puzzle, not an example: it does not vote)
            examples_ = [p_ for p_ in panels_ if len({col_ for col_, _ in p_["ring"].values()}) >= 2]
            map_, votes_ = learn_map(examples_)
            carried_map = dict((carry or {}).get("legend_map") or {})
            # (the mapping carried from the last level is the working
            # hypothesis; this level's example panels add their votes)
            for g_, rel_ in carried_map.items():
                votes_[g_][rel_] += 10
            map_ = {g_: cnt_.most_common(1)[0][0] for g_, cnt_ in votes_.items()
                    if cnt_.most_common(1)[0][1] >= 2 or g_ in carried_map}
            model.legend_map = dict(map_)
            viol_ = violations(panels_, map_) if map_ else []
            note(f"a legend within: {len(panels_)} panels of tiles around a miniature; the miniature's colours name the tiles — {len(viol_)} tiles break it")
            if verbose:
                print(f"   legend: pitch {pitch_} tiles {dims_} panels {len(panels_)} map {map_} violations {[(pos_, col_) for pos_, col_, _ in viol_]}")
            from agent.core.legend import tile_look, solve_flips
            # THE TILES AND WHAT A CLICK ON EACH DOES: a tile's look (plain, or
            # a pattern on it) names its kind; the first click on a tile of a
            # kind shows which tiles it changes (itself alone, or its
            # neighbours too); tiles that look alike act alike
            tiles_all = {}
            for p_ in panels_:
                for _d, (col_, pos_) in p_["ring"].items():
                    tiles_all[pos_] = col_
            demands_by = {pos_: dm_ for pos_, _c, dm_ in viol_}
            all_demands = {}
            for p_ in panels_:
                cc_ = p_["glyph"].get((0, 0))
                for d_, (col_, pos_) in p_["ring"].items():
                    rel_ = map_.get(p_["glyph"].get(d_))
                    if rel_ is not None:
                        all_demands.setdefault(pos_, []).append((rel_, cc_))
            looks_ = {pos_: tile_look(f, pos_, dims_) for pos_ in tiles_all}
            effects_ = dict(getattr(model, "legend_effects", None) or (carry or {}).get("legend_effects") or {})   # look -> offsets toggled (carried)
            colours_now = lambda: {pos_: int(f[pos_[0], pos_[1]]) for pos_ in tiles_all}
            fixed_any = False
            wanted = {pos_ for pos_, col_, _dm in viol_}
            # 1. learn the effect of each look present among the wanted tiles —
            # on the instance with the most tiles around it (an effect on the
            # neighbours shows only where neighbours are)
            def _around(pos_):
                return sum(1 for q_ in tiles_all if q_ != pos_ and abs(q_[0] - pos_[0]) <= pitch_ + 0.5 and abs(q_[1] - pos_[1]) <= pitch_ + 0.5)
            by_look = {}
            for pos_ in sorted(wanted, key=lambda q_: -_around(q_)):
                by_look.setdefault(looks_[pos_][1], pos_)
            for lk_, pos_ in by_look.items():
                if obs.levels_completed > lv0:
                    break
                if lk_ in effects_:
                    continue
                before_ = colours_now()
                step(("click", pos_))
                fixed_any = True
                if obs.levels_completed > lv0:
                    break
                after_ = colours_now()
                changed_ = {q_ for q_ in tiles_all if before_[q_] != after_[q_]}
                if not changed_:
                    note("a click on the tile changed nothing: the tiles are not set by clicking")
                    break
                effects_[lk_] = frozenset((q_[0] - pos_[0], q_[1] - pos_[1]) for q_ in changed_)
                model.legend_effects = dict(effects_)
                if verbose:
                    print(f"   legend: a click on a tile of look {sorted(lk_)[:4]}... changes {sorted(effects_[lk_])}")
            if obs.levels_completed > lv0:
                play_level.last["won_hyp"] = None
                return True, steps, "legend", obs
            # 2. what must change now, and the clicks that do exactly that
            palette_ = sorted({c_ for c_ in tiles_all.values()} | {c_ for _p in panels_ for c_ in [_p["glyph"].get((0, 0))] if c_ is not None})
            now_ = colours_now()
            need_ = {pos_ for pos_, dm_ in all_demands.items() if not satisfies(now_[pos_], dm_)}
            plain_ = all(len(effects_.get(looks_[pos_][1], frozenset({(0, 0)}))) == 1 for pos_ in tiles_all)
            if need_ and not plain_ and len(palette_) <= 2:
                eff_by_pos = {pos_: {(pos_[0] + dy_, pos_[1] + dx_) for (dy_, dx_) in effects_.get(looks_[pos_][1], frozenset({(0, 0)}))} & set(tiles_all) for pos_ in tiles_all}
                clicks_ = solve_flips(need_, eff_by_pos, sorted(tiles_all))
                note(f"the tiles change their neighbours too: {len(clicks_) if clicks_ else 'no'} clicks make exactly the change the legend asks for")
                if verbose:
                    print(f"   legend: need {sorted(need_)} -> clicks {clicks_}")
                for _round in range(3):
                    if not clicks_:
                        break
                    for pos_ in clicks_:
                        before_ = colours_now()
                        step(("click", pos_)); fixed_any = True
                        if obs.levels_completed > lv0:
                            break
                        after_ = colours_now()
                        changed_ = frozenset((q_[0] - pos_[0], q_[1] - pos_[1]) for q_ in tiles_all if before_[q_] != after_[q_])
                        # (what the click did is the effect of its look — the
                        # fuller sight replaces the earlier one)
                        if changed_ and changed_ != effects_.get(looks_[pos_][1]):
                            effects_[looks_[pos_][1]] = changed_ | effects_.get(looks_[pos_][1], frozenset())
                            model.legend_effects = dict(effects_)
                    if obs.levels_completed > lv0:
                        break
                    now_ = colours_now()
                    need_ = {pos_ for pos_, dm_ in all_demands.items() if not satisfies(now_[pos_], dm_)}
                    if not need_:
                        break
                    eff_by_pos = {pos_: {(pos_[0] + dy_, pos_[1] + dx_) for (dy_, dx_) in effects_.get(looks_[pos_][1], frozenset({(0, 0)}))} & set(tiles_all) for pos_ in tiles_all}
                    clicks_ = solve_flips(need_, eff_by_pos, sorted(tiles_all))
                    if verbose:
                        print(f"   legend: still need {sorted(need_)} -> clicks {clicks_}")
            else:
                for pos_ in sorted(need_):
                    if obs.levels_completed > lv0:
                        break
                    for _try in range(len(palette_) + 1):
                        if satisfies(int(f[pos_[0], pos_[1]]), all_demands[pos_]):
                            break
                        before_c = int(f[pos_[0], pos_[1]])
                        step(("click", pos_)); fixed_any = True
                        if obs.levels_completed > lv0 or int(f[pos_[0], pos_[1]]) == before_c:
                            break
            if obs.levels_completed > lv0:
                play_level.last["won_hyp"] = None
                return True, steps, "legend", obs
            if viol_ and fixed_any:
                note("the tiles were set as the legend says and the level goes on")

    # --- A TABLE OF PAIRS (free information, read before fiddling): two
    # things joined by a connector are a pair; a column of pairs maps forms
    # to forms (a form being the same turned or mirrored); a row of forms
    # below is a question and the row under it the answer, position by
    # position, with a cursor at the position being answered.  The keys
    # are learned by one press each: one moves the cursor, one changes the
    # form under it; then each wrong position is set right.
    try:
        from agent.core.table import read_table, question_and_answer, answer_status
        hud_rows_t = {r_ for s_ in tr.prev if "hud" in s_.tags for (r_, _) in s_.cells}

        def _read_qa():
            t_ = read_table(f, bg, hud_rows=hud_rows_t)
            qa_ = question_and_answer(t_) if t_ else None
            if qa_ is None:
                return None
            qa_["status"] = answer_status(t_, qa_)
            return qa_
        qa0_ = _read_qa()
    except Exception as ex_:
        qa0_ = None
        if verbose:
            print(f"   table: reading failed {ex_!r}")
    if qa0_ and len(qa0_["map"]) >= 2 and qa0_["cursor"] is not None and keys and not clickable:
        n_ans = len(qa0_["answer"]["glyphs"])
        note(f"a table of pairs: {len(qa0_['map'])} forms mapped to forms; a question row of {len(qa0_['question']['glyphs'])} and an answer row of {n_ans} under it, the cursor at position {qa0_['cursor']}")
        if verbose:
            print(f"   table: map {len(qa0_['map'])} question {len(qa0_['question']['glyphs'])} answer {n_ans} cursor {qa0_['cursor']} wrong {[k_ for k_, (qf_, w_, h_) in enumerate(qa0_['status']) if w_ is not None and w_ != h_]}")
        roles_ = {}
        for k_ in keys:
            qa_a = _read_qa()
            if qa_a is None:
                break
            note(f"table: press {k_} — does it move the cursor, or change the form under it?")
            step(k_)
            if obs.levels_completed > lv0:
                return True, steps, "table", obs
            qa_b = _read_qa()
            if qa_b is None:
                break
            ca_, cb_ = qa_a["cursor"], qa_b["cursor"]
            if ca_ is not None and cb_ is not None and ca_ != cb_:
                d_ = (cb_ - ca_) % n_ans
                roles_[k_] = ("move", 1 if d_ == 1 else (-1 if d_ == n_ans - 1 else d_))
            elif ca_ is not None and ca_ < len(qa_a["status"]) and ca_ < len(qa_b["status"]) and qa_a["status"][ca_][2] != qa_b["status"][ca_][2]:
                roles_[k_] = ("cycle",)
        if verbose:
            print(f"   table: key roles {roles_}")
        mv_ = {d_: k_ for k_, r_ in roles_.items() if r_[0] == "move" for d_ in (r_[1],)}
        cyc_ = [k_ for k_, r_ in roles_.items() if r_[0] == "cycle"]
        if cyc_ and mv_:
            note(f"table: {cyc_[0]} changes the form under the cursor; {dict(mv_)} move it — setting each wrong position right")
            for _pass in range(n_ans + 2):
                qa_ = _read_qa()
                if verbose:
                    print(f"   table: pass {_pass}: cursor {qa_['cursor'] if qa_ else None} status {[(w_ is not None, w_ == h_) for (qf_, w_, h_) in qa_['status']] if qa_ else None}")
                if qa_ is None or qa_["cursor"] is None:
                    break
                wrong_ = [k_ for k_, (qf_, w_, h_) in enumerate(qa_["status"]) if w_ is not None and w_ != h_]
                if not wrong_:
                    break
                cur_ = qa_["cursor"]
                # (the nearest wrong position, the cursor moving the shorter way round)
                def _dist(k_):
                    return min((k_ - cur_) % n_ans, (cur_ - k_) % n_ans)
                k_ = min(wrong_, key=_dist)
                guard_ = 0
                while cur_ != k_ and guard_ < 2 * n_ans:
                    fwd_ = (k_ - cur_) % n_ans; back_ = (cur_ - k_) % n_ans
                    key_ = mv_.get(1) if (fwd_ <= back_ or -1 not in mv_) and 1 in mv_ else mv_.get(-1)
                    if key_ is None:
                        break
                    step(key_); guard_ += 1
                    if obs.levels_completed > lv0:
                        return True, steps, "table", obs
                    qa_ = _read_qa()
                    if qa_ is None or qa_["cursor"] is None:
                        break
                    cur_ = qa_["cursor"]
                if cur_ != k_:
                    break
                want_ = qa_["status"][k_][1]
                for _try in range(8):
                    step(cyc_[0])
                    if obs.levels_completed > lv0:
                        return True, steps, "table", obs
                    qa_ = _read_qa()
                    if qa_ is None or k_ >= len(qa_["status"]):
                        break
                    if qa_["status"][k_][2] == want_:
                        note(f"table: position {k_} now shows the form its question maps to")
                        break
            if obs.levels_completed > lv0:
                return True, steps, "table", obs
            note("table: every position answered as the table says, and the level goes on")

    # --- fiddle
    moves = Fiddler().moves(sprites, tr.prev, keys, clickable, budget,
                            first=tuple(c_[2] for c_ in (getattr(model, "coined", []) or []) if c_[0] == "click"))
    # (THE FIELD BEFORE THE BAND: a click on what stands outside the playfield
    # — the band's pictures, a palette — is asked after the field's things;
    # the band tells, it is not played.  su15: eight of the first clicks went
    # to the HUD while the clock drained by the press)
    try:
        if pf is not None:
            def _in_pf(m_):
                return not (isinstance(m_, tuple) and m_[0] == "click") or (pf[0] <= m_[1][0] < pf[1] and pf[2] <= m_[1][1] < pf[3])
            moves = [m_ for m_ in moves if _in_pf(m_)] + [m_ for m_ in moves if not _in_pf(m_)]
    except Exception:
        pass
    note("fiddling: every key twice, then a click on each thing and the keys again")
    # THE DIRECTED RE-TEST: a law the last attempt's first divergence
    # contradicted is tested first, on purpose — a key's law for a kind,
    # by that key's press before any other (the kind's thing is on the
    # level); a thing or ground held passable, by the planner no longer
    # crossing it except through the one press that tests it (unblock).
    # The verdict is written to the ledger: restored, or dropped.
    doubts_ = list((carry or {}).get("doubts") or [])
    if doubts_:
        first_keys_ = []
        for law_, key_ in doubts_:
            if law_ == "eff" and isinstance(key_, (tuple, list)) and len(key_) == 2 and key_[0] in keys and key_[0] not in first_keys_:
                if any(s_.body == key_[1] for s_ in tr.prev if s_.sid < 1000):
                    first_keys_.append(key_[0])
        if first_keys_:
            moves = list(first_keys_) + [m_ for m_ in moves if m_ not in first_keys_]
            note(f"re-test first: the last attempt's plan diverged on {first_keys_} — pressed before anything else, to see the law again")
        play_level.ledger["retests"] = [(law_, key_) for law_, key_ in doubts_]

    def collect_readings(ctrl, riders=frozenset()):
        """COLLECT: several like small things of one colour (not the mover's,
        not a HUD strip) — arrive on each; the first arrival tells whether
        they are collected (it vanishes or changes)."""
        class _M:
            color = ctrl.body; size = ctrl.size; centre = ctrl.centre; cells = frozenset(ctrl.cells)
        out = []
        by_kind_ = defaultdict(list)
        for x in tr.prev:
            if x.sid == ctrl.sid or x.sid in riders or "hud" in x.tags or "terrain" in x.tags or x.sid >= 1000:
                continue
            if getattr(model, "clock_colour", None) is not None and x.body == model.clock_colour:
                continue                # the clock's colour on the floor: a refill, spent when needed, not collected
            if 2 <= x.size <= 16 and x.body != ctrl.body:
                by_kind_[(x.body, x.cshape if hasattr(x, "cshape") else x.shape)].append(x)
        for (kb, _), xs in by_kind_.items():
            if len(xs) < 2 or len(xs) > 12:
                continue
            d = min(abs(x.centre[0] - ctrl.centre[0]) + abs(x.centre[1] - ctrl.centre[1]) for x in xs)
            h4 = GoalHypothesis("collect", _M, xs[0].centre, f"collect the {len(xs)} like things C{kb}[{xs[0].size}@{xs[0].bbox}]", 7.0 - d / 100.0, xs[0].size)
            h4.t_cells = frozenset(q for x in xs for q in x.cells)
            h4.members = [x.sid for x in xs]
            h4.t_color = kb
            out.append(h4)
        return out

    def _self_aimed(h):
        """The reading's mover and target are members of one body (the eye
        of the block it rides): read by kind and size, since the readings
        were made on the first frame and the body has moved since."""
        m_ = re.search(r"C(\d+)\[(\d+)@", h.note or "")
        if not m_:
            return False
        if getattr(h, "kind", None) == "meet":
            return False                # (a meeting is between two like movers — one body until a wall parts them)
        t_col, t_size = int(m_.group(1)), int(m_.group(2))
        for b_ in model.bodies():
            if len(b_) < 2:
                continue
            members_ = [(s_.body, s_.size) for s_ in tr.prev if s_.sid in b_]
            kinds_ = {k_ for k_, _ in members_}
            if h.m_color in kinds_ and (t_col, t_size) in members_:
                return True
        return False

    def actuated(h):
        """Is the reading's mover something the core can already move? A
        key that moved a thing of its colour; the keys move the selected
        thing and it is selectable (or is selected); a click law of its
        kind; a track that moved its kind."""
        col = h.m_color
        sids = {s_.sid for s_ in tr.prev if s_.body == col}
        if "marks onto" in h.note:
            return bool(model.key_selected) or bool(model.active)
        for k_ in model.fixed.keys():
            if any(sid_ in sids and e_[0] != "none" for sid_, e_ in model.fixed.moving(k_).items()):
                return True
        if model.key_selected and (col in model.select_kinds or model.selected_now(tr.prev) in sids or col in model.kinds_worn_by_selected()):
            return True
        if model.relative_law(col) is not None:
            return True
        # (A THING THE BODY PUSHES IS A THING THE CORE MOVES — the planner shoves
        # it where the reading wants it (tw01, the blocks to the pen): a push
        # takes it to a place (align, inside), not to a docking by marks nor a
        # turn.  And a thing, not a floor: a pushable kind of which the board
        # holds a crowd (tu93's tiles) is no mover to read — four, hand-chosen)
        if col in model.pushable and model.active and h.kind in ("align", "inside") and len(sids) <= 4:
            return True
        if any(col in model.pos_kinds.get(k_, ()) for k_ in model.positional):
            return True
        # (a thing hung between handles the clicks send: moved by them)
        if any(sid_ in sids for sid_ in (getattr(model, "centroid_of", {}) or {})):
            return True
        return False
    # LOOK FIRST, THEN TRY: the readings say what fits where; the keys are
    # pressed once each (who moves?), and a reading whose mover the core
    # can move is TRIED at once — the repertoire waits for the readings
    # that cannot be tried yet
    if hyps and not carry:
        for k in keys:
            note(f"who moves? press {k}")
            step(k)
            if obs.levels_completed > lv0:
                return True, steps, "keys", obs
        if not any(model.fixed.moving(k) for k in keys) and keys:
            for k in keys:                       # a first press may be swallowed: once more
                step(k)
                if obs.levels_completed > lv0:
                    return True, steps, "keys", obs
        tried_early = 0
        pursued_ = set()
        experimented_ = set()
        unact_ = set()                  # readings whose early chase found no way — retried once a law is learned
        retried_ = set()

        def try_actuated():
            """Every reading whose mover answers now is tried, best first —
            a click made for one reading may have actuated another's mover."""
            nonlocal tried_early
            for h in list(hyps[:4]):
                if verbose:
                    print(f"   try_actuated: {h.note[:40]} actuated {actuated(h)} pursued {id(h) in pursued_} unact {id(h) in unact_} retried {id(h) in retried_} centroid {dict(getattr(model, 'centroid_of', {}) or {})}")
                if not actuated(h):
                    continue
                # (a reading pursued before its mover could be moved is
                # pursued again once it can — the probe taught how)
                if id(h) in pursued_ and not (id(h) in unact_ and id(h) not in retried_ and getattr(model, "centroid_of", None)):
                    continue
                if id(h) in pursued_:
                    retried_.add(id(h))
                pursued_.add(id(h))
                note(f"looked first: {h.note[:60]} — the mover answers; trying it")
                res_ = pursue([x for x in hyps if x.rule() == h.rule() and x.m_color == h.m_color] or [h], max_single=1, early=True)
                if res_:
                    return res_
                unact_.add(id(h))
                tried_early += 1
                if tried_early >= 2:
                    break
            return None
        res_ = try_actuated()
        if res_:
            return res_
        for h in list(hyps[:4]):
            if tried_early >= 2:
                break
            if id(h) in pursued_ or id(h) in experimented_:
                continue
            if clickable:
                experimented_.add(id(h))
                # the mover does not answer the keys: click it (a selection?)
                # and press the keys, click beside it (a pull?)
                cands_ = [s_ for s_ in tr.prev if s_.body == h.m_color and s_.sid < 1000 and 0.5 * h.m_size <= s_.size <= 2 * h.m_size]
                if not cands_:
                    continue
                m_ = min(cands_, key=lambda s_: abs(s_.centre[0] - h.m_start[0]) + abs(s_.centre[1] - h.m_start[1]))
                # (a kind strewn many times over the board is the board, not
                # a mover to be probed: the distinct things are asked first)
                dims_ = (m_.bbox[2] - m_.bbox[0], m_.bbox[3] - m_.bbox[1])
                if sum(1 for s_ in tr.prev if s_.body == m_.body and s_.size == m_.size and (s_.bbox[2] - s_.bbox[0], s_.bbox[3] - s_.bbox[1]) == dims_) >= 8:
                    continue
                cp_ = click_point(m_)
                # (one probe per thing: readings that share a mover share what
                # its click and the keys after it showed — asked once)
                probed_pts_ = play_level.last.setdefault("probed_clicks", set()) if isinstance(play_level.last, dict) else set()
                if cp_ in probed_pts_:
                    continue
                probed_pts_.add(cp_)
                note(f"looked first: {h.note[:60]} — does the mover answer a click?")
                f0_ = f
                step(("click", cp_))
                if obs.levels_completed > lv0:
                    return True, steps, "keys", obs
                if np.any(f != f0_):
                    for k in keys[:4]:
                        step(k)
                        if obs.levels_completed > lv0:
                            return True, steps, "keys", obs
                for pt_ in ((cp_[0], min(63, cp_[1] + 5)), (max(0, cp_[0] - 5), cp_[1])):
                    step(("click", pt_))
                    if obs.levels_completed > lv0:
                        return True, steps, "keys", obs
                res_ = try_actuated()
                if res_:
                    return res_
        # the keys were pressed once: the repertoire's first round of them
        # is done (its later rounds stay — refusals map the walls)
        moves = moves[len(keys):]
    def look_again():
        """Readings from the frame as it is now join the first frame's."""
        # LOOK AGAIN: the fiddling may have brought things into being (a
        # block where dots were); readings from the frame as it is now join
        # the first frame's (first frame's first — they were confirmed there)
        try:
            objs2 = _objs_by_units(segment(f, bg, mask=None), getattr(tr, "unit_form", None), getattr(tr, "elastic", ()))
            hud_cells = frozenset(q for s_ in tr.prev if "hud" in s_.tags for q in s_.cells)
            hyps2 = read_frame_goals(objs2, 64, 64, playfield=pf, hud_cells=hud_cells, weights=rweights.get())
            # REACH THE ODD THING: the thing the keys move (known only now)
            # goes to each thing that is the only one of its colour — the
            # oldest reading there is, ranked below every relation
            ctrl = None
            marks = selected_hosts(tr.prev)
            sel0 = [x_ for x_ in (model.selected_now(tr.prev),) if x_ is not None]
            if sel0:
                ctrl = next((x for x in tr.prev if x.sid == sel0[0]), None)
            elif model.active:
                members = [x for x in tr.prev if x.sid in model.active]
                ctrl = max(members, key=lambda x: x.size) if members else None
            elif any(model.relative_law(x.body) is not None for x in tr.prev if x.sid < 1000 and "hud" not in x.tags and "terrain" not in x.tags):
                # (a click game: the mover is what the clicks call — the largest
                # thing of a kind with a relative law; su15 L2's block, made by
                # merging, is the thing to take to where the last level's went)
                members = [x for x in tr.prev if x.sid < 1000 and "hud" not in x.tags and "terrain" not in x.tags and model.relative_law(x.body) is not None
                           and (pf is None or (pf[0] <= x.bbox[0] and x.bbox[2] < pf[1] and pf[2] <= x.bbox[1] and x.bbox[3] < pf[3]))]
                ctrl = max(members, key=lambda x: x.size) if members else None
            elif carry and getattr(model, "seeded", None) and (carry.get("won") or (None,))[0]:
                # (before any press on a carried level, the mover is what the
                # carried laws say the keys move — of the colour the last
                # level's winning reading named as its mover; any other
                # presumption waits for a press)
                mcol_ = carry["won"][1]
                pres_ = {sid_ for k_ in model.seeded for sid_ in model.seeded[k_]
                         if any(e[0] == "translate" for e in [model.fixed.eff[k_].get(sid_, ("none", (0, 0)))])}
                members = [x for x in tr.prev if x.sid in pres_ and x.sid < 1000 and x.body == mcol_ and "hud" not in x.tags and "terrain" not in x.tags]
                ctrl = max(members, key=lambda x: x.size) if members else None
            if ctrl is not None:
                host = part_host(ctrl, tr.prev)
                if host is not None:
                    ctrl = host         # a part rides its whole: the whole is what moves
                riders = {x.sid for x in tr.prev if part_host(x, tr.prev) is not None and part_host(x, tr.prev).sid == ctrl.sid}
                counts = Counter(x.body for x in tr.prev if "hud" not in x.tags and "terrain" not in x.tags)

                class _M:      # what GoalHypothesis reads off a mover
                    color = ctrl.body; size = ctrl.size; centre = ctrl.centre; cells = frozenset(ctrl.cells)
                body_of_ctrl = set()
                for b_ in model.bodies():
                    if ctrl.sid in b_:
                        body_of_ctrl |= set(b_)
                def ground_around_(x_):
                    # (the colour the thing stands on: the modal colour of the ring of cells about its box)
                    r0_, c0_, r1_, c1_ = x_.bbox
                    ring_ = Counter()
                    for r_ in range(r0_ - 1, r1_ + 2):
                        for c_ in (c0_ - 1, c1_ + 1):
                            if 0 <= r_ < 64 and 0 <= c_ < 64:
                                ring_[int(f[r_, c_])] += 1
                    for c_ in range(c0_, c1_ + 1):
                        for r_ in (r0_ - 1, r1_ + 1):
                            if 0 <= r_ < 64 and 0 <= c_ < 64:
                                ring_[int(f[r_, c_])] += 1
                    return max(ring_, key=ring_.get) if ring_ else None
                ctrl_ground_ = ground_around_(ctrl)
                for x in tr.prev:
                    if x.sid == ctrl.sid or x.sid in riders or "hud" in x.tags or "terrain" in x.tags or x.size > 60 or counts[x.body] != 1 or x.body == ctrl.body:
                        continue
                    # (a thing standing outside the playfield — the picture in the
                    # band above the field — is not a thing to reach; the band tells)
                    if pf is not None and not _ENV.get("NO_PF_FILTER") and not (pf[0] <= x.bbox[0] and x.bbox[2] < pf[1] and pf[2] <= x.bbox[1] and x.bbox[3] < pf[3]) \
                            and (pf[0] <= ctrl.bbox[0] and ctrl.bbox[2] < pf[1]):
                        continue
                    # (a member of the mover's own body, or a thing beside it that
                    # moved when it moved — its eye — is not a thing to reach)
                    if x.sid in body_of_ctrl or (model.moved_count.get(x.sid, 0) > 0 and
                                                 any(abs(r_ - r2) <= 1 and abs(c_ - c2) <= 1 for (r_, c_) in x.cells for (r2, c2) in ctrl.cells)):
                        continue
                    d = abs(x.centre[0] - ctrl.centre[0]) + abs(x.centre[1] - ctrl.centre[1])
                    h3 = GoalHypothesis("align", _M, x.centre, f"reach the odd thing C{x.body}[{x.size}@{x.bbox}]", 6.0 - d / 100.0, x.size)
                    h3.t_cells = frozenset(x.cells)
                    h3.over = True                                 # reached = standing over it
                    h3.t_color = x.body                            # (the goal's colour, to be carried)
                    hyps2.append(h3)
                hyps2.extend([h_ for h_ in collect_readings(ctrl, riders) if getattr(h_, "t_color", None) not in model.hazards()])
                # WHAT THE FLOOR'S WINS WERE READ AS, on this level's things:
                # the class coined from a win (a colour and a form) names the
                # things to collect or to reach here, whatever their marks
                for cn_ in getattr(model, "coined", []) or []:
                    try:
                        kind_c, mcol_, tcol_, tdims_, rel_ = cn_
                        if kind_c == "click" or tcol_ in model.hazards():
                            continue
                        ts_ = [x for x in tr.prev if x.sid < 1000 and x.body == tcol_ and x.sid != ctrl.sid and x.sid not in riders
                               and x.sid not in body_of_ctrl and "hud" not in x.tags and "terrain" not in x.tags and x.size <= 400]
                        same_d = [x for x in ts_ if tdims_ and _dims(x) == tuple(tdims_)]
                        ts_ = same_d or ts_
                        if not ts_:
                            continue
                        if kind_c == "collect":
                            hc_ = GoalHypothesis("collect", _M, ts_[0].centre, f"won before: collect C{tcol_}[{ts_[0].size}@{ts_[0].bbox}]", 9.0, ts_[0].size)
                            hc_.t_cells = frozenset(q for x in ts_ for q in x.cells); hc_.members = [x.sid for x in ts_]; hc_.t_color = tcol_
                            hyps2.append(hc_)
                        else:
                            for x in ts_:
                                ha_ = GoalHypothesis("align", _M, x.centre, f"won before: reach C{tcol_}[{x.size}@{x.bbox}] ({rel_})", 8.5, x.size)
                                ha_.t_cells = frozenset(x.cells); ha_.over = rel_ != "against"; ha_.t_color = tcol_
                                hyps2.append(ha_)
                    except Exception:
                        continue
            # LIKE MOVERS MEET: when the keys move two like things (one kind,
            # one form) that answer the keys as separate bodies, the reading
            # is that they are to be made one (Tushar, m0r0: "2 objects which
            # need to be made 1") — the target is the other mover, wherever
            # it stands when the plan is made; nothing else on the board is
            # aimed at
            if model.active:
                act_ = [x for x in tr.prev if x.sid in model.active and x.sid < 1000 and "hud" not in x.tags and "terrain" not in x.tags]
                bodies_l = model.bodies()
                def _body_of(sid_):
                    return next((i_ for i_, b_ in enumerate(bodies_l) if sid_ in b_), ("one", sid_))
                for i_, a_ in enumerate(act_):
                    for b_ in act_[i_ + 1:]:
                        if a_.body != b_.body:
                            continue
                        # (two that stand apart: separate bodies, or one body
                        # whose members have not yet been parted by a wall)
                        gap_ = max(a_.bbox[0] - b_.bbox[2], b_.bbox[0] - a_.bbox[2], a_.bbox[1] - b_.bbox[3], b_.bbox[1] - a_.bbox[3])
                        if _body_of(a_.sid) == _body_of(b_.sid) and gap_ <= 1:
                            continue
                        da_ = (a_.bbox[2] - a_.bbox[0] + 1, a_.bbox[3] - a_.bbox[1] + 1)
                        db_ = (b_.bbox[2] - b_.bbox[0] + 1, b_.bbox[3] - b_.bbox[1] + 1)
                        if da_ != db_ or abs(a_.size - b_.size) > max(2, 0.2 * a_.size):
                            continue

                        class _MA:
                            color = a_.body; size = a_.size; centre = a_.centre; cells = frozenset(a_.cells)
                        h4 = GoalHypothesis("meet", _MA, b_.centre, f"like movers meet C{b_.body}[{b_.size}@{b_.bbox}]", 5.5, b_.size)
                        h4.t_cells = frozenset(b_.cells); h4.t_color = b_.body; h4.meet = (a_.sid, b_.sid)
                        hyps2.append(h4)
            def _key(h_):
                # (a docking reading is between two hosts: the same marks' colour
                # on another pair of pieces is another reading; a twin's reading
                # names its place — two discs for two twins are two readings)
                place_ = None
                if "HUD twin" in (h_.note or "") and getattr(h_, "t_cells", None):
                    place_ = _bbox(h_.t_cells)
                return (h_.rule(), h_.m_color, h_.t_color if hasattr(h_, "t_color") else None,
                        h_.note if "marks onto" in h_.note else None, place_)
            # (a docking reading not yet made is read afresh from the frame as
            # it is now — the marks have moved; the stale one is dropped)
            hyps[:] = [h for h in hyps if not ("marks onto" in h.note and not getattr(h, "_done", False))]
            known = {_key(h) for h in hyps}
            for h2 in hyps2:
                key2 = _key(h2)
                if key2 not in known:
                    hyps.append(h2); known.add(key2)
                    if verbose:
                        print("   hyp (after fiddling)", h2)
            # WHAT NEVER MOVED IS A WEAK MOVER: after the whole repertoire, a
            # reading whose mover's colour no action ever moved ranks below
            # the readings whose mover did
            moved_colours = {x.body for x in tr.prev if model.moved_count.get(x.sid, 0) > 0}
            moved_colours |= {kd for kd in model.rel} | set(model.select_kinds)
            pushed_colours = {kd for kd in model.pushable if sum(1 for x in tr.prev if x.body == kd and x.sid < 1000) <= 4} if model.active else set()
            if model.moved_count:
                for h in hyps:
                    if h.m_color not in moved_colours and not (h.m_color in pushed_colours and h.kind in ("align", "inside")) and not getattr(h, "_demoted", False):
                        h.prio -= 10.0; h._demoted = True       # (below every reading whose mover moved)
            # (A THING THAT TRAVELS ON ITS OWN IS NO MOVER TO READ: an agent's
            # readings are its own business — the helper "fitting the player's
            # shape" took forty seconds of search on tw01)
            for h in hyps:
                if h.m_color in (getattr(model, "agents", ()) or ()) and not getattr(h, "_demoted", False):
                    h.prio -= 10.0; h._demoted = True
            # a reading whose mover the core can move outranks one whose mover it cannot
            hyps.sort(key=lambda h: (0 if (h.rule(), h.mover_bigger()) in conf_rel else 1, 0 if actuated(h) else 1, -h.prio))
        except Exception as ex:
            if verbose:
                print("   look-again failed:", ex)

    # THE MICROMOUSE'S SECOND RUN: the same level again (its first frame is the
    # one the last attempt began on) after a death — the presses before the
    # world left the plan are a known way to where things were still right;
    # they are walked again, not rediscovered (the journal's safe prefix)
    hl_ = (carry or {}).get("highlight") or {}
    play_level.fatal = set()
    if hl_.get("fatal"):
        fk_ = hl_["fatal"]
        play_level.fatal.add((fk_[0], _key_id(fk_[1])))
    # (a death by the CLOCK leaves no safe prefix — the presses walked are
    # what spent it; the level is looked at anew with what they taught)
    if carry and carry.get("how") == "died" and hl_.get("safe_prefix") and hl_.get("start_hash") == play_level.journal.start_hash \
            and _ENV.get("NO_JOURNAL_REPLAY") is None and not carry.get("clock_death"):
        pre_ = [_key_id(k_) for k_ in hl_["safe_prefix"]][:40]
        note(f"the same level again: walking the {len(pre_)} presses that were right last time before looking anew")
        if verbose:
            print(f"   journal replay: {len(pre_)} presses of the last attempt's safe prefix")
        hashes_ = list(hl_.get("safe_hashes") or [])
        for j_, k_ in enumerate(pre_):
            if obs.levels_completed > lv0:
                return True, steps, "journal", obs
            step(k_)
            # (the world keeps its own clock: a press that did not lead where it
            # led last time ends the replay — from here the core looks anew)
            if j_ < len(hashes_) and hashes_[j_] is not None and _frame_key(obs) != hashes_[j_]:
                note(f"the world is not where it was after press {j_ + 1}: looking anew")
                if verbose:
                    print(f"   journal replay ended at press {j_ + 1}: the frame differs from last time")
                break
        if obs.levels_completed > lv0:
            return True, steps, "journal", obs
    if carry and (carry.get("eff") or carry.get("key_selected") or carry.get("cast") or carry.get("click_laws")):
        # (a level won by clicks carries no key law — its buttons, its cast
        # and its template still open the next level; before 2026-10-15 an
        # empty `eff` skipped this whole opening: "keys known []")
        # CONFIRM, DON'T RELEARN: each key once; a seeded law that held on
        # some thing of its kind is dropped for the things of that kind it
        # did not hold on; the repertoire is skipped when the keys confirm
        # (DIRECTED: one press that moves something is confirmation enough —
        # it says who moves; the other keys' carried laws are provisional
        # and are verified by use, the plan replanning at the first surprise.
        # A press that moved nothing — a wall — says nothing: the next key)
        confirmed_keys = set()
        moved_one_ = False
        # LOOK BEFORE THE FIRST PRESS: with the keys' laws carried in, the
        # level's readings can be planned from the start as it stands; the
        # plan's first press is the confirmation, and the start is not
        # spent on a blind press (st70: the goal one press left of the
        # mover; the confirming press slid it away for good)
        steps_look_ = steps
        try:
            look_again()
            top_ = [h for h in hyps if not getattr(h, "_demoted", False)][:3]
            if top_:
                note("the keys are known: a plan from the start before any press")
                res_ = pursue(top_, max_single=len(top_), early=True)
                if res_:
                    return res_
        except Exception as ex_:
            if verbose:
                print(f"   look before the first press failed: {ex_!r}")
        if steps > steps_look_ and model.active:
            moved_one_ = True           # (the plan's presses confirmed the keys by use: no blind press after them — cn04 L2, a docking undone by one)
            for k_ in keys:
                if any(sid_ in model.active for sid_ in model.fixed.moving(k_)):
                    confirmed_keys.add(k_)
        for k in keys:
            if k in model.undo_keys:
                continue
            # (a key nothing was ever learned about — never pressed before
            # the last attempt ended — is asked once whatever the others
            # showed: an unknown key is not a provisional law to verify by
            # use, it is a hole in the repertoire — st20, ACTION4 never
            # pressed, the way left never planned)
            unknown_ = k not in (carry.get("eff") or {}) and k not in model.seeded and model.fixed.presses.get(k, 0) == 0
            if moved_one_ and not unknown_:
                continue
            if _press_deadly(model, tr, k):
                note(f"{k} would step into a known hazard — not pressed")
                continue
            note(f"confirming what the last level taught: press {k}" if not unknown_ else f"a key never pressed before: what does {k} do?")
            b_ = {x.sid: x.bbox for x in tr.prev}
            step(k)
            if obs.levels_completed > lv0:
                return True, steps, "confirm", obs
            if any(x.sid in b_ and x.bbox != b_[x.sid] and "hud" not in x.tags and "terrain" not in x.tags for x in tr.prev):
                moved_one_ = True
        not_movers = set()
        for k, sids in list(model.seeded.items()):
            held = {sid for sid in sids if any(e[0] != "none" and n_ > 0 for e, n_ in model.fixed.alt[k][sid].items())}
            if held:
                confirmed_keys.add(k)
                for sid in sids - held:
                    if model.fixed.alt[k][sid].get(("none", (0, 0)), 0) > 0:
                        not_movers.add(sid)     # of the kind, but it stayed while another of the kind moved
        # a thing that stayed under one key while its kind moved is not the
        # mover under any key (the kind alone does not make a mover)
        for k, sids in list(model.seeded.items()):
            for sid in sids & not_movers:
                model.fixed.eff[k].pop(sid, None)
                model.fixed.contradicted[k].discard(("refused", sid))
        # THE STEP IS THE LEVEL'S: a key confirmed here with a step of another
        # length than it had last level (a finer board) says the lattice
        # changed — the other carried laws, not yet seen here, keep their
        # direction and take the new step (each is still verified by use)
        ratio_ = None
        for k in confirmed_keys:
            for sid, alts in model.fixed.alt[k].items():
                kd_ = next((x.body for x in tr.prev if x.sid == sid), None)
                old_ = (carry.get("eff", {}).get(k, {}) or {}).get(kd_)
                for e, n_ in alts.items():
                    if n_ > 0 and e[0] == "translate" and old_ and old_[0] == "translate":
                        m_new = max(abs(e[1][0]), abs(e[1][1])); m_old = max(abs(old_[1][0]), abs(old_[1][1]))
                        if m_new and m_old and m_new != m_old:
                            ratio_ = (m_new, m_old)
        if ratio_ is not None:
            m_new, m_old = ratio_
            n_scaled = 0
            for k2 in list(model.fixed.eff):
                for sid2, e2 in list(model.fixed.eff[k2].items()):
                    if e2[0] == "translate" and model.fixed.support[k2].get(sid2, 0) <= 0 and max(abs(e2[1][0]), abs(e2[1][1])) == m_old:
                        v2 = (int(round(e2[1][0] * m_new / m_old)), int(round(e2[1][1] * m_new / m_old)))
                        model.fixed.eff[k2][sid2] = ("translate", v2); n_scaled += 1
            if n_scaled:
                note(f"the keys step {m_new} here, not {m_old}: the carried laws take this level's step")
        if verbose:
            print(f"   carried: keys confirmed {sorted(confirmed_keys, key=str)} of seeded {sorted(model.seeded, key=str)}; key_selected {model.key_selected}; counters {model.counters} (carried {carry.get('counters')}); triggers {dict(model.triggers)}")
        if confirmed_keys and not carry.get("rel"):
            # THE KEYS ARE KNOWN; THE CLICKS ARE NOT: when the last level was
            # won without a click, what a click does is still to be learned
            # here — a click on each thing, and the keys after it (a piece
            # is chosen by clicking it)
            if not model.select_kinds and not model.rel_keys and not model.click_gone and not model.click_grows:
                # (the cheapest test: click a thing, press a known key and its
                # opposite — the thing answers or it does not, and stands where
                # it stood; the larger things first, a few of them)
                ks_ = dict(model.key_selected)
                if ks_:
                    k1_ = next(iter(ks_)); v1_ = ks_[k1_]
                    k2_ = next((k for k, v in ks_.items() if v == (-v1_[0], -v1_[1])), None)
                else:
                    k1_, k2_ = (keys[0] if keys else None), (keys[1] if len(keys) > 1 else None)
                probe_ = [k for k in (k1_, k2_) if k is not None]
                # (WHAT THE LAST LEVELS LEFT SILENT STAYS SILENT: a kind clicked
                # on an earlier level with nothing changed is not clicked again
                # to see — wa30 L4, four silent clicks of a 115-press clock)
                silent_k_ = {k_ for k_, n_ in (getattr(model, "silent_click_kinds", {}) or {}).items() if n_ >= 2} if not _ENV.get("NO_SILENT_KINDS") else set()
                big_ = sorted([s_ for s_ in tr.prev if s_.sid < 1000 and s_.size > 16 and "hud" not in s_.tags and "terrain" not in s_.tags
                               and s_.sid not in (model.active or ()) and kind(s_) not in silent_k_], key=lambda s_: -s_.size)[:6]
                moves = []
                for s_ in big_:
                    moves.append(("click", click_point(s_)))
                    moves += [("if", k) for k in probe_]
                play_level.cheap_probe = True
                note(f"the keys are known from the last level, a click is not: clicking each of {len(big_)} things, a key and its opposite after each")
            else:
                moves = []
        elif carry.get("rel"):
            moves = [m for m in moves if isinstance(m, tuple) and m[0] == "click"][:max(8, budget // 2)]
        # THE RUN PROTOCOL FIRST when it won the last level: one press of the
        # run key reads this level's telling (what flashes, where the ink
        # goes), and the law places the movers before any reading is
        # pursued (sp80 L3: the readings spent 36 presses of 63 before the
        # run had its turn)
        if carry.get("how") == "run" and carry.get("run_key") in keys and not _ENV.get("NO_RUNKEY"):
            note(f"the run ({carry['run_key']}) first: it won the last level — one press to read this level's telling")
            step(carry["run_key"])
            if obs.levels_completed > lv0:
                return True, steps, "run", obs
            if getattr(progress, "runs", None):
                if run_protocol(tr, model, keys, step, lambda: obs, lv0, verbose, progress, note, bg):
                    return True, steps, "run", obs
        # THE WORKING HYPOTHESIS: what won the last level is what wins this
        # one — the same goal, sought the same way, until the level says
        # otherwise.  The goal's kind in view: go to it.  Not in view: it
        # lies where it lay before (the direction from the start), so
        # explore that way by free paths, unknown things being banks, not
        # water; a removal only when no free way leads on.  Nothing else
        # is tried (no walking into things "to see") until this fails
        won_ = carry.get("won")
        # THE WINNING READING AS A TEMPLATE: what won the last level was a
        # relation between kinds (the piece's marks onto the anchor's like
        # marks, turned; the image covering the dots; the canvas equal to
        # the picture) — not that piece at that place.  The same relation,
        # read on THIS level's things, is pursued first, thing by thing,
        # before anything is fiddled with beyond confirming the keys
        same_ = []
        if won_ and won_[0] and hyps:
            same_ = [h for h in hyps if _template_reading(h)]
        if won_ and won_[0] and model.active and not same_:
            # (the winning relation may be one read only once the mover is
            # known — like movers meeting: the frame is looked at again with
            # the mover known before the template is given up)
            look_again()
            same_ = [h for h in hyps if _template_reading(h)]
        if won_ and won_[0] and hyps:
            kind_ = tpl_kind_
            if same_:
                # (the relation that won was between KINDS: among this level's
                # readings of it, those with the same mover come first — tw03 L2
                # read the pen as the stopper's, the player's and the block's)
                same_.sort(key=lambda h: (h.m_color != won_[1], -h.prio))
                note(f"the reading that won the last level, on this level's things: {kind_ or won_[0]} — {len(same_)} such reading(s), tried first")
                if verbose:
                    print(f"   template: {won_[0]} / {kind_!r} / target colour {won_[2]} -> {[h.note[:70] for h in same_[:6]]}")
                res_ = pursue(same_, max_single=len(same_), early=True)
                if res_:
                    return res_
        # (the winning relation read on this level IS the hypothesis: going
        # to things of the goal's colour is the fallback for when it cannot
        # be read here)
        if model.active and won_ and won_[2] is not None and not same_:
            model.explore_dir = carry.get("won_dir")
            act_ = [s_ for s_ in tr.prev if s_.sid in model.active]
            ctrl_ = max(act_, key=lambda s_: s_.size) if act_ else None
            for leg in range(40):
                if obs.levels_completed > lv0:
                    return True, steps, "goal", obs
                if time.time() - play_level.t0 > _level_seconds() * 0.6:
                    break
                goal_things = [x for x in tr.prev if x.body == won_[2] and "hud" not in x.tags and "terrain" not in x.tags and x.sid < 1000 and x.sid not in model.active]
                if goal_things and ctrl_ is not None:
                    x = min(goal_things, key=lambda x_: abs(x_.centre[0] - ctrl_.centre[0]) + abs(x_.centre[1] - ctrl_.centre[1]))

                    class _M:
                        color = ctrl_.body; size = ctrl_.size; centre = ctrl_.centre; cells = frozenset(ctrl_.cells)
                    h3 = GoalHypothesis("align", _M, x.centre, f"reach the odd thing C{x.body}[{x.size}@{x.bbox}]", 9.0, x.size)
                    h3.t_cells = frozenset(x.cells); h3.over = True; h3.t_color = x.body
                    note(f"the goal of the last level is in view (colour {x.body}): going to it")
                    res_ = pursue([h3], max_single=1, early=True)
                    if res_:
                        return res_
                    break
                note(f"the goal (colour {won_[2]}) is not in view: exploring the way it lay last time {carry.get('won_dir')}, by free paths first")
                went = explore_leg(tr, model, keys, step, note, verbose, lambda: obs, lv0, free_first=True)
                if obs.levels_completed > lv0:
                    return True, steps, "explore", obs
                if not went:
                    break
                act_ = [s_ for s_ in tr.prev if s_.sid in model.active]
                ctrl_ = max(act_, key=lambda s_: s_.size) if act_ else None
        # LIKE SMALL THINGS ON THE FLOOR ARE COLLECTED FIRST — only of a
        # kind the last level showed to be collected (arrived on, and gone)
        if model.active and carry.get("collected"):
            act_ = [s_ for s_ in tr.prev if s_.sid in model.active]
            ctrl_ = max(act_, key=lambda s_: s_.size) if act_ else None
            if ctrl_ is not None:
                for h_ in sorted([h for h in collect_readings(ctrl_) if h.t_color in carry["collected"]], key=lambda x_: -x_.prio)[:2]:
                    note(f"looked first: {h_.note[:60]} — collecting")
                    res_ = pursue([h_], max_single=1, early=True)
                    if res_:
                        return res_
        # THE PROTOCOL THAT WON THE LAST LEVEL IS TRIED FIRST — WITH ITS CAST
        # REINTERPRETED: the organ's parts are found again by KIND on this
        # level (a launch button of the launch's kind; marks of the colours
        # the toggles flip between), each confirmed by one press, and the
        # organ runs before any fiddling (tn36 L2: the strip the fiddle
        # never clicked)
        cast_ = carry.get("cast") or {}
        if carry.get("how") == "program" and cast_.get("launch") and cast_.get("flips") and clickable:
            note("the panel-and-launch protocol first, its cast found by kind: it won the last level")
            by_kind_ = defaultdict(list)
            for s_ in tr.prev:
                if s_.sid < 1000 and "hud" not in s_.tags and "terrain" not in s_.tags:
                    by_kind_[kind(s_)].append(s_)
            pressed_ = 0
            for uk_s, trav_ in cast_["launch"].items():
                try:
                    uk_ = int(uk_s)
                except ValueError:
                    continue
                fm_ = (cast_.get("launch_form") or {}).get(uk_s)
                cands_ = by_kind_.get(uk_, [])
                if fm_:
                    # (the button most like the one that launched: its box first, its size next)
                    cands_ = sorted(cands_, key=lambda s_: (abs((s_.bbox[2] - s_.bbox[0] + 1) - fm_[0]) + abs((s_.bbox[3] - s_.bbox[1] + 1) - fm_[1]), abs(s_.size - fm_[2])))
                for b_ in cands_[:3]:
                    if any(k_ == ("click", click_point(b_)) for k_, _, _ in model.journeys):
                        continue
                    if verbose:
                        print(f"   cast: launch candidate S{b_.sid} kind {uk_} {b_.size}@{b_.bbox} -> click {click_point(b_)}")
                    step(("click", click_point(b_))); pressed_ += 1
                    if obs.levels_completed > lv0:
                        return True, steps, "program", obs
                    if model.journeys:
                        break
            if model.journeys and not model.toggle_cells:
                # a mark of a toggle colour, small and among its like in a line: one click tells
                for a_, b_ in cast_["flips"].items():
                    marks_ = [s_ for s_ in tr.prev if s_.sid < 1000 and s_.body == int(a_) and s_.size <= 16 and "hud" not in s_.tags]
                    if len(marks_) < 2:
                        continue
                    m0_ = min(marks_, key=lambda s_: (s_.bbox[0], s_.bbox[1]))
                    step(("click", click_point(m0_))); pressed_ += 1
                    if obs.levels_completed > lv0:
                        return True, steps, "program", obs
                    if model.toggle_cells:
                        break
            if verbose:
                print(f"   cast: {pressed_} presses to find the launch and a toggle; journeys {len(model.journeys)} toggles {len(model.toggle_cells)}")
            if model.journeys and model.toggle_cells:
                if program_protocol(tr, model, step, lambda: obs, lv0, verbose, hyps, note, bg):
                    return True, steps, "program", obs
        if carry.get("how") == "display" and model.displays and model.triggers and model.active:
            note("display protocol first: it won the last level")
            won_d, _ = display_protocol(env, tr, model, keys, step, lambda: obs, lv0, verbose)
            if won_d:
                play_level.last["how"] = "display"
                return True, steps, "display", obs
    if verbose:
        print("   fiddle:", moves)
    got_ = model.complete_directions()
    if got_:
        note(f"{got_[0]} is read as the fourth direction {got_[1]} (three seen, one silent) — until it says otherwise")
    # THE FOURTH DIRECTION: three arrow keys seen, one silent — read as the
    # fourth (provisional; verified by use)
    got_ = model.complete_directions()
    if got_:
        note(f"{got_[0]} is read as the fourth direction {got_[1]} (three seen, one silent) — until it says otherwise")
    if verbose:
        print(f"   fourth-direction check: active {model.active} keys {[(k_, dict(model.fixed.moving(k_))) for k_ in model.fixed.eff if isinstance(k_, str)]} presses {dict(model.fixed.presses)}")
    # ADAPTIVE: a thing that appears or changes form under the fiddling
    # is fiddled with in turn (click it, press the keys, click beside it)
    seen_kinds = set()          # sids already fiddled with after appearing
    extra = 0
    i = 0
    # A PRESS THAT SPENDS THE CLOCK IS NOT FREE: when the level's clock drains
    # by the press (the rate known, carried or read), the fiddling gets a
    # quarter of what is left (hand-chosen) — su15 L4: 32 clicks a level,
    # the first look spent 18 of them before the ladder could be climbed
    fiddle_cap_ = None
    try:
        # (only a clock SEEN TO END A LEVEL rations the fiddle — s5i5's bar
        # drains by the press and ends nothing; the fiddle it needs is 32)
        left_c_ = model.clock_left(tr.prev) if getattr(model, "clock_rate", 0) and (carry or {}).get("clock_deadly") else None
        if left_c_ is not None and left_c_ > 0:
            fiddle_cap_ = max(4, int(left_c_) // 4)
            if verbose:
                print(f"   fiddle: the clock leaves {left_c_} presses; the fiddle takes at most {fiddle_cap_}")
    except Exception:
        fiddle_cap_ = None
    n_fiddle_ = 0
    while i < len(moves):
        k = moves[i]; i += 1
        if obs.levels_completed > lv0:
            return True, steps, "fiddle", obs
        # (a twin read: the fiddle has found the organ of the level; the rest
        # of the repertoire waits on the twin protocol)
        if getattr(model, "twin", None) is not None and model.twin.confirmed and not _ENV.get("NO_LEVER"):
            note("fiddling stopped: a rewind key and its twin are read")
            break
        if fiddle_cap_ is not None and n_fiddle_ >= fiddle_cap_:
            note(f"fiddling stopped at {n_fiddle_} presses: the clock is spent by the press and the readings need the rest")
            break
        n_fiddle_ += 1
        if isinstance(k, tuple) and k[0] == "if":
            k = k[1]            # a key pressed because the click before it did something
            note(f"fiddling: {k} after the click (the click did something)")
        elif isinstance(k, tuple) and k[0] == "click":
            note(f"fiddling: click at {k[1]} — what does this thing do?")
        elif isinstance(k, str):
            note(f"fiddling: press {k}")
        if k in model.undo_keys:
            continue            # an undo teaches nothing twice
        if getattr(model, "twin", None) is not None and isinstance(k, str) and k == model.twin.rewind_key:
            continue            # (a rewind teaches nothing twice, and each one spends a twin)
        if isinstance(k, str) and _press_deadly(model, tr, k):
            note(f"fiddling: {k} would step into a known hazard — not pressed")
            continue
        if isinstance(k, tuple) and k[0] == "click":
            # DIRECTED: a thing of a kind a click has already been seen to
            # take away answers as its kind did — not clicked again; a lone
            # cell of a colour strewn over the ground is texture, not a thing
            under0 = [x for x in tr.prev if x.bbox[0] <= k[1][0] <= x.bbox[2] and x.bbox[1] <= k[1][1] <= x.bbox[3] and "terrain" not in x.tags and "hud" not in x.tags]
            u0 = min(under0, key=lambda x: x.size) if under0 else None
            if u0 is not None and u0.sid >= 1000:
                u0 = next((x for x in tr.prev if x.sid == u0.sid // 1000), u0)     # (a part's click is its object's)
            skip_ = False
            if len(k) >= 3 and u0 is None:
                skip_ = True                    # the thing this click was owed to is gone (nothing under the point)
            k = ("click", k[1])
            if u0 is not None and u0.body in model.removable():
                skip_ = True
            if u0 is not None and u0.size == 1 and sum(1 for x in tr.prev if x.size == 1 and x.body == u0.body) >= 20:
                skip_ = True
            if skip_:
                j = i
                while j < len(moves) and isinstance(moves[j], tuple) and moves[j][0] == "if":
                    j += 1
                del moves[i:j]
                if verbose:
                    print(f"   fiddle: click {k[1]} skipped ({'S%d colour %d: its kind has answered' % (u0.sid, u0.body) if u0 is not None else 'its thing is gone'})")
                continue
        id0 = tr.next_id
        f0 = f
        before_click_ = list(tr.prev)
        step(k)
        # A SILENT CLICK handed the keys to nothing: the keys queued to be
        # pressed "after the click" are not pressed (the clock aside,
        # nothing in the frame changed)
        if isinstance(k, tuple) and k[0] == "click":
            hud_now = {q for s_ in tr.prev if "hud" in s_.tags or "bar" in s_.tags for q in s_.cells}
            hud_rows = {r for r, _ in hud_now}
            changed_cells = [(int(r), int(c)) for r, c in np.argwhere(f != f0) if int(r) not in hud_rows]
            # (a click that took the thing away handed the keys to nothing)
            gone_click = bool(u0 is not None and u0.body in model.removable() and not any(x.sid == u0.sid for x in tr.prev))
            if not changed_cells or gone_click:
                j = i
                while j < len(moves) and isinstance(moves[j], tuple) and moves[j][0] == "if":
                    j += 1
                if verbose and j > i:
                    print(f"   fiddle: click {k[1]} was silent; skipping {[m[1] for m in moves[i:j]]}")
                del moves[i:j]
                # (a click silent on a thing of a kind is remembered by the kind,
                # and carries: the next level does not click that kind again to see)
                try:
                    if u0 is not None and not changed_cells and hasattr(model, "silent_click_kinds") and not _ENV.get("NO_SILENT_KINDS"):
                        model.silent_click_kinds[kind(u0)] += 1
                        # (...and once TWO of the kind were silent, the others queued to
                        # be clicked here are not: two silent pens speak for the pens —
                        # one may have been silent for where it stood; s5i5's piston at
                        # its limit spoke for the pistons and the level was lost)
                        # (only within the cheap probe of the clicks — a fiddle's own
                        # sequence clicks a thing again on purpose)
                        j2 = i if model.silent_click_kinds[kind(u0)] >= 2 and getattr(play_level, "cheap_probe", False) else len(moves)
                        while j2 < len(moves):
                            m2 = moves[j2]
                            if isinstance(m2, tuple) and m2[0] == "click":
                                u2 = next((x for x in tr.prev if x.sid < 1000 and tuple(m2[1]) in set(x.cells)), None)
                                if u2 is not None and kind(u2) == kind(u0):
                                    j3 = j2 + 1
                                    while j3 < len(moves) and isinstance(moves[j3], tuple) and moves[j3][0] == "if":
                                        j3 += 1
                                    del moves[j2:j3]
                                    continue
                            j2 += 1
                except Exception:
                    pass
        if extra >= 12 or getattr(play_level, "cheap_probe", False):
            continue                # (a probe of the clicks is not the place for new things: they are read after)
        # (a part's id is anchored to its object: a part of an old thing is
        # not a new thing)
        # (a piece that appeared in the colour of a thing that moved or
        # changed form this press is that thing's fragment, not a new thing)
        churn_cols = {x.body for x in tr.prev if any(e.kind in ("morphed", "moved", "shrank", "grew") and e.sid == x.sid for e in tr.last_events)}
        new_things = [x for x in tr.prev if x.sid >= id0 and x.sid < 1000 and x.body not in churn_cols] + \
            [x for x in tr.prev if any(e.kind == "morphed" and e.sid == x.sid for e in tr.last_events)]
        # A MARK THAT APPEARED ON AN OLD THING is new too (Tushar, lf52: "a
        # new update came up in the board" — the ring where the selected
        # peg may go): the thing is old, the mark on it is not — the mark
        # is clicked, not the thing's centre
        marks_new = []
        if isinstance(k, tuple) and k[0] == "click":
            for e_ in tr.last_events:
                if e_.kind != "recoloured" or e_.sid >= 1000:
                    continue
                x_ = tr.get(e_.sid)
                if x_ is None or "hud" in x_.tags or "terrain" in x_.tags:
                    continue
                ch_ = [(r, c) for r, c in x_.cells if int(f[r, c]) != int(f0[r, c]) and int(f[r, c]) != x_.body]
                if not ch_ or len(ch_) > x_.size:
                    continue
                if any(x_.bbox[0] <= k[1][0] <= x_.bbox[2] and x_.bbox[1] <= k[1][1] <= x_.bbox[3] for _ in (0,)):
                    continue                # (the thing clicked itself changed: not a mark elsewhere)
                col_ = Counter(int(f[r, c]) for r, c in ch_).most_common(1)[0][0]
                if (x_.sid, col_) in seen_kinds:
                    continue
                seen_kinds.add((x_.sid, col_))
                mr_ = sum(r for r, _ in ch_) / len(ch_); mc_ = sum(c for _, c in ch_) / len(ch_)
                marks_new.append((x_, min(ch_, key=lambda q: abs(q[0] - mr_) + abs(q[1] - mc_)), col_))
        # A CLICK THAT RECOLOURED A THING IN PLACE IS FOLLOWED BY ONE KEY:
        # did the click hand the keys to it?  (sp80 L2: the bars are chosen
        # by a click and shown by their colour; a fiddle of clicks alone
        # never learned which bar the keys move, and the plan moved the
        # wrong one)
        # (only when the click passed a colour between two things — the
        # clicked one now wears what another wore, and that one wears what
        # the clicked one wore: a choice shown by colour; a click that merely
        # toggles a thing's colour is not followed by a key — ka59's marks)
        if isinstance(k, tuple) and k[0] == "click" and getattr(model, "pending_select", None) is not None \
                and not (i < len(moves) and isinstance(moves[i], tuple) and moves[i][0] == "if") and not _ENV.get("NO_SELECT_PROBE"):
            try:
                b0_by_ = {s_.sid: s_ for s_ in before_click_}
                exch_ = False
                psid_ = model.pending_select[0]
                pa_ = next((s_ for s_ in tr.prev if s_.sid == psid_), None); pb_ = b0_by_.get(psid_)
                if pa_ is not None and pb_ is not None and pa_.body != pb_.body:
                    for s_ in tr.prev:
                        q_ = b0_by_.get(s_.sid)
                        if q_ is not None and s_.sid != psid_ and q_.body == pa_.body and s_.body == pb_.body:
                            exch_ = True; break
            except Exception:
                exch_ = False
            k_sel_ = next((k2_ for k2_ in keys if k2_ in model.key_selected), None) or next((k2_ for k2_ in keys if any(e_[0] == "translate" for e_ in model.fixed.moving(k2_).values())), None)
            if k_sel_ is not None and exch_:
                moves.insert(i, ("if", k_sel_))
                if verbose:
                    print(f"   fiddle: click {k[1]} passed a colour to S{model.pending_select[0]}; {k_sel_} once to see whether it answers now")
        # (the thing just clicked, changed by the click, is not new; and the
        # keys queued after that click are pressed before the new thing is
        # fiddled with)
        just_clicked = None
        if isinstance(k, tuple) and k[0] == "click":
            under_ = [x for x in tr.prev if x.bbox[0] <= k[1][0] <= x.bbox[2] and x.bbox[1] <= k[1][1] <= x.bbox[3] and "terrain" not in x.tags]
            just_clicked = {x.sid for x in under_} | {x.sid // 1000 for x in under_ if x.sid >= 1000}
        j_ins = i
        while j_ins < len(moves) and isinstance(moves[j_ins], tuple) and moves[j_ins][0] == "if":
            j_ins += 1
        for x_, pt_, col_ in marks_new:
            # (a mark on a thing after its click: the thing is selected — the
            # mark is clicked, and the places beside the THING are probed,
            # toward its like neighbours: a leap lands beyond one)
            like_ = [t_ for t_ in tr.prev if t_.sid < 1000 and t_.sid != x_.sid and (t_.body, t_.size) == (x_.body, x_.size)]
            add = [("click", pt_)] + [("if", k_) for k_ in keys[:2]] + [("click", q_) for q_ in beside_points(x_, tr.prev, like_)]
            moves[j_ins:j_ins] = add
            j_ins += len(add)
            extra += len(add)
            if verbose:
                print(f"   fiddle+: a mark (colour {col_}) appeared on S{x_.sid} {x_.body}/{x_.size}@{x_.bbox} -> click {pt_}")
            note(f"fiddling: a mark of colour {col_} appeared on S{x_.sid} after the click — clicking the mark")
        # NEW THINGS THAT ENCLOSE A THING MARK IT: pieces that appeared
        # around one thing (a bracket, a frame) say that thing is selected —
        # the places beside THAT thing are probed, toward its like
        # neighbours (a leap: select, then point beside — lf52; read off
        # the scene, so a mirrored board is probed the same way)
        new_ids_ = {s_.sid for s_ in new_things}
        enclosed_ = Counter()
        for s_ in new_things:
            for x_ in tr.prev:
                if x_.sid >= 1000 or x_.sid in new_ids_ or "hud" in x_.tags or "terrain" in x_.tags or x_.size > 60:
                    continue
                if x_.bbox[0] - 2 <= s_.bbox[2] and s_.bbox[0] <= x_.bbox[2] + 2 and x_.bbox[1] - 2 <= s_.bbox[3] and s_.bbox[1] <= x_.bbox[3] + 2:
                    enclosed_[x_.sid] += 1
        subject_ = next((sid_ for sid_, n_ in enclosed_.most_common(1) if n_ >= 2), None)
        if subject_ is not None and clickable and subject_ not in seen_kinds:
            seen_kinds.add(subject_)
            x_ = tr.get(subject_)
            like_ = [t_ for t_ in tr.prev if t_.sid < 1000 and t_.sid != x_.sid and (t_.body, t_.size) == (x_.body, x_.size)]
            add = [("click", q_) for q_ in beside_points(x_, tr.prev, like_)]
            moves[j_ins:j_ins] = add
            j_ins += len(add)
            extra += len(add)
            if verbose:
                print(f"   fiddle+: S{x_.sid} {x_.body}/{x_.size}@{x_.bbox} is enclosed by {enclosed_[subject_]} new things (selected) -> {add}")
            note(f"fiddling: new pieces appeared around S{x_.sid} — it is selected; probing beside it, toward its like neighbours")
        for s_ in new_things:
            if s_.size < 2 or "hud" in s_.tags or "terrain" in s_.tags or s_.sid in seen_kinds or (just_clicked and s_.sid in just_clicked):
                continue
            seen_kinds.add(s_.sid)
            c = click_point(s_)
            like_ = [t_ for t_ in tr.prev if t_.sid < 1000 and t_.sid != s_.sid and (t_.body, t_.size) == (s_.body, s_.size)]
            add = [("click", c)] + [("if", k_) for k_ in keys[:2]] + ([("click", q_) for q_ in beside_points(s_, tr.prev, like_)] if subject_ is None else []) if clickable else list(keys[:4])
            moves[j_ins:j_ins] = add
            j_ins += len(add)
            extra += len(add)
            if verbose:
                print(f"   fiddle+: new S{s_.sid} {s_.body}/{s_.size}@{s_.bbox} -> {add}")
            note(f"fiddling: S{s_.sid} (colour {s_.body}, {s_.size} cells) is new or changed form — fiddling with it in turn")
    if verbose:
        _p = {k_: [(sid_, len(c_), col_) for sid_, c_, col_ in v_] for k_, v_ in model.paints.items()}
        print(f"   fiddled {steps} presses; paints {_p} colour clicks {[(r[0], r[3], r[4]) for r in model.colour_clicks]}")
    got_ = model.complete_directions()
    if got_:
        note(f"{got_[0]} is read as the fourth direction {got_[1]} (three seen, one silent) — until it says otherwise")
    # THE PANEL WRITES THE JOURNEY: a launch sent a thing on a journey and
    # put it back, and clicks flip marks in a strip — the marks are the
    # journey's steps in order; the journey to the reading's target is
    # written into them
    # (the cast carried in says what the toggles' colours are: a launch found
    # but no toggle yet — one click on a mark of that colour tells; the strip's
    # marks are strewn over the board and the fiddle leaves such kinds last)
    cast_ = (carry or {}).get("cast") or {}
    if model.journeys and not model.toggle_cells and cast_.get("flips") and clickable:
        for a_, b_ in cast_["flips"].items():
            marks_ = [s_ for s_ in tr.prev if s_.sid < 1000 and s_.body == int(a_) and s_.size <= 16 and "hud" not in s_.tags]
            if len(marks_) < 2:
                continue
            m0_ = min(marks_, key=lambda s_: (s_.bbox[0], s_.bbox[1]))
            note(f"the cast says marks of colour {a_} are toggles: one click to confirm")
            if verbose:
                print(f"   cast: toggle candidate S{m0_.sid} colour {a_} {m0_.size}@{m0_.bbox} -> click {click_point(m0_)}")
            step(("click", click_point(m0_)))
            if obs.levels_completed > lv0:
                return True, steps, "program", obs
            if model.toggle_cells:
                break
    if model.journeys and model.toggle_cells:
        if program_protocol(tr, model, step, lambda: obs, lv0, verbose, hyps, note, bg):
            return True, steps, "program", obs
    # REMOVE ALL OF THE KIND BY LEAPS: a leap was read (select, then point;
    # the neighbour leapt over is gone) — the fewest of the kind left is
    # the goal read from that first removal
    if model.leap is not None:
        if leap_protocol(tr, model, step, lambda: obs, lv0, verbose, note):
            return True, steps, "leap", obs
    # THE EQUAL READING: a key painted a big thing in place, and a static
    # picture of its dims stands elsewhere — the paint protocol comes
    # before every spatial reading (its signature is nowhere else)
    # (WHAT WON THE LAST LEVEL IS TRIED FIRST: a level won by the carry opens
    # with the carry, before the paint protocol's region test — wa30 L4's
    # ten clicks on the pens, of a 115-press clock, "setting a colour")
    # THE RECORD AS A SECOND BODY: a rewind key and its twin known, a lever
    # found or to be found — the twin holds the pad while the body passes
    # the gate (agent/core/lever; g50t)
    if getattr(model, "twin", None) is not None and not model.twin.confirmed and not _ENV.get("NO_LEVER"):
        if twin_probe(tr, model, keys, step, lambda: obs, lv0, verbose, note):
            return True, steps, "twin", obs
    if getattr(model, "twin", None) is not None and model.twin.confirmed and not _ENV.get("NO_LEVER"):
        play_level._twin_tried = True
        if twin_protocol(tr, model, keys, step, lambda: obs, lv0, verbose, note, hyps):
            return True, steps, "twin", obs
    # THE RUN'S TELLING IS A GOAL AND A TEST: a key drew a trace from a
    # source and took it back — before the paint protocol, which reads the
    # same press as a pour (sp80 L2: fifteen places walked in paint)
    # What flashed at the end is what the run is
    # about; the trace is a law the plan runs itself (agent/core/runkey)
    if getattr(progress, "runs", None) and not _ENV.get("NO_RUNKEY"):
        if run_protocol(tr, model, keys, step, lambda: obs, lv0, verbose, progress, note, bg):
            return True, steps, "run", obs
    if model.paints and carry and carry.get("how") == "carry" and getattr(model, "carry", None) and not _ENV.get("NO_CARRY_FIRST"):
        note("carry protocol first: it won the last level")
        if carry_protocol(tr, model, step, lambda: obs, lv0, verbose, hyps, note, keys):
            return True, steps, "carry", obs
    if model.paints:
        def budget_left():
            # the game's clock: a counter's cells per press, and what is left
            for sid_ in model.counters:
                s_ = tr.get(sid_)
                if s_ is not None and sid_ in size_first and steps > 0:
                    rate = (size_first[sid_] - s_.size) / float(steps)
                    if rate > 0:
                        return s_.size / rate
            return None
        won_p, n_p = paint_protocol(env, tr, model, keys, step, lambda: obs, lv0, verbose, 70, bg, budget_left)
        if won_p:
            return True, steps, "paint", obs
    # BUMP INTO THINGS: with the keys known, walk the marked thing into
    # each other small thing once (what does contact do?)
    if model.key_selected:
        marks = selected_hosts(tr.prev)
        sel0 = [x_ for x_ in (model.selected_now(tr.prev),) if x_ is not None]
        if sel0:
            me = next((s for s in tr.prev if s.sid == sel0[0]), None)
            others = [s for s in tr.prev if me is not None and s.sid != me.sid and "terrain" not in s.tags
                      and "hud" not in s.tags and 3 < s.size <= 60 and s.body != me.body or (me is not None and s.sid != me.sid and s.body == me.body and s.size > 3)]
            others.sort(key=lambda s: abs(s.centre[0] - me.centre[0]) + abs(s.centre[1] - me.centre[1]))
            for o in others[:2]:
                # a short plan to stand beside it, then the key toward it
                comps = tr.prev
                start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
                start["_sel"] = me.sid
                solids0 = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2}
                succ0 = make_successors(model, comps, solids0, keys, me.sid, o.centre, frame=f, frozen=(o.sid,))

                def near(st, _o=o.sid, _m=me.sid):
                    a_ = st.get(_m); b_ = st.get(_o)
                    if not a_ or not b_:
                        return 999
                    ca, cb = centre(a_), centre(b_)
                    return abs(ca[0] - cb[0]) + abs(ca[1] - cb[1])
                p0 = plan_general(start, succ0, lambda st: near(st) <= 6, heuristic=lambda st: near(st) / 3.0, max_states=20000, max_depth=30)
                note(f"bump: walking S{me.sid} into S{o.sid} (colour {o.body}) — what does contact do?")
                if verbose:
                    print(f"   bump: me S{me.sid} -> S{o.sid} {o.body}/{o.size}@{o.centre}: approach {p0}; marks {marks} parts {[(x.sid, x.body, sorted(x.cells)) for x in tr.prev if x.sid >= 1000 and x.sid < 20000]}")
                    if p0 is None:
                        print(f"      start succ {[a for a, _ in succ0(start)]} whys {getattr(make_successors, 'whys', None)} last {getattr(plan_general, 'last', None)}")
                if p0:
                    for k in p0[:12]:
                        step(k)
                    # now bump: the key whose vector points at it
                    me2 = next((s for s in tr.prev if s.sid == me.sid), None)
                    o2 = next((s for s in tr.prev if s.sid == o.sid), None)
                    if me2 is not None and o2 is not None:
                        d = (o2.centre[0] - me2.centre[0], o2.centre[1] - me2.centre[1])
                        best_k = max(model.key_selected, key=lambda k: model.key_selected[k][0] * d[0] + model.key_selected[k][1] * d[1])
                        for _ in range(2):
                            step(best_k)
                            if verbose:
                                print(f"      bump press {best_k}: events {[(e.kind, e.sid, e.data.get('vec')) for e in tr.last_events if e.kind != 'recoloured'][:6]} me {[x.bbox for x in tr.prev if x.sid == me.sid]} o {[x.bbox for x in tr.prev if x.sid == o.sid]} last_sel {model.last_sel}")
                        if verbose:
                            print(f"   bump: pressed {best_k} x2; pushable={sorted(model.pushable)} contact={list(model.contact_seen)[:4]}")
                if obs.levels_completed > lv0:
                    return True, steps, "bump", obs
    # A KEY THAT DID NOTHING IN THE OPEN MAY DO SOMETHING IN CONTACT
    # (Tushar, wa30: "push and fit" — a press toward a crate turns the body,
    # ACTION5 attaches the faced crate, the crate moves with the body):
    # the body is walked against each of the nearest things, the silent
    # keys pressed there, and a step away read — a thing that came along
    # is CARRIED by the key that bound it (and let go by it again)
    silent_ = [k for k in keys if k not in model.undo_keys and not any(e_[0] != "none" for e_ in model.fixed.moving(k).values())]
    if not model.key_selected and model.active and silent_ and getattr(model, "carry", None) is None and not model.pushable:
        act_ = [s for s in tr.prev if s.sid in model.active and s.sid < 1000]
        me = max(act_, key=lambda s: s.size) if act_ else None
        vec_of = {}
        for k in keys:
            e_ = model.fixed.moving(k).get(me.sid) if me is not None else None
            if e_ and e_[0] == "translate" and (e_[1][0] or e_[1][1]):
                vec_of[k] = e_[1]
        if me is not None and vec_of:
            others = [s for s in tr.prev if s.sid != me.sid and s.sid not in model.active and s.sid < 1000 and "terrain" not in s.tags
                      and "hud" not in s.tags and 3 < s.size <= 60 and s.body not in model.hazards()]
            others.sort(key=lambda s: abs(s.centre[0] - me.centre[0]) + abs(s.centre[1] - me.centre[1]))
            tried_kinds_ = set()
            for o in others[:3]:
                if kind(o) in tried_kinds_ or model.carry is not None:
                    continue
                tried_kinds_.add(kind(o))
                comps = tr.prev
                start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
                start["_sel"] = model.selected_now(comps)
                solids0 = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2}
                succ0 = make_successors(model, comps, solids0, keys, me.sid, o.centre, frame=f, frozen=(o.sid,))

                def near(st, _o=o.sid, _m=me.sid):
                    # (beside it along one axis — overlapping it on the other:
                    # a diagonal neighbour is not faced)
                    a_ = st.get(_m); b_ = st.get(_o)
                    if not a_ or not b_:
                        return 999
                    ba, bb = _bbox(a_), _bbox(b_)
                    ov_r = min(ba[2], bb[2]) - max(ba[0], bb[0]); ov_c = min(ba[3], bb[3]) - max(ba[1], bb[1])
                    gap_r = max(bb[0] - ba[2], ba[0] - bb[2], 0); gap_c = max(bb[1] - ba[3], ba[1] - bb[3], 0)
                    if ov_r >= 0 and ov_c < 0:
                        return gap_c
                    if ov_c >= 0 and ov_r < 0:
                        return gap_r
                    return gap_r + gap_c + 4
                p0 = plan_general(start, succ0, lambda st: near(st) <= 1, heuristic=lambda st: near(st) / 3.0, max_states=20000, max_depth=30)
                note(f"contact: walking the body against S{o.sid} (colour {o.body}) to press the silent keys there")
                if verbose:
                    print(f"   contact: me S{me.sid} -> S{o.sid} {o.body}/{o.size}@{o.bbox}: approach {p0}")
                if not p0:
                    continue
                for k in p0[:14]:
                    step(k)
                    if obs.levels_completed > lv0:
                        return True, steps, "contact", obs
                me2 = tr.get(me.sid); o2 = tr.get(o.sid)
                if me2 is None or o2 is None:
                    continue
                d = (o2.centre[0] - me2.centre[0], o2.centre[1] - me2.centre[1])
                toward = max(vec_of, key=lambda k: vec_of[k][0] * d[0] + vec_of[k][1] * d[1])
                tv_ = vec_of[toward]
                away = min(vec_of, key=lambda k: vec_of[k][0] * tv_[0] + vec_of[k][1] * tv_[1])
                if vec_of[away][0] * tv_[0] + vec_of[away][1] * tv_[1] >= 0:
                    away = None
                step(toward)                 # (facing it; a blocked press is a turn)
                if obs.levels_completed > lv0:
                    return True, steps, "contact", obs
                for K in silent_:
                    o_b = tr.get(o.sid)
                    if o_b is None:
                        break
                    cells_b = dict(o_b.cells) if isinstance(o_b.cells, dict) else {q: o_b.body for q in o_b.cells}
                    step(K)
                    if obs.levels_completed > lv0:
                        return True, steps, "contact", obs
                    o_a = tr.get(o.sid)
                    recol = o_a is not None and o_a.bbox == o_b.bbox and (dict(o_a.cells) if isinstance(o_a.cells, dict) else {}) != cells_b
                    if verbose:
                        print(f"   contact: {K} pressed against S{o.sid}: recoloured {recol}; events {[(e.kind, e.sid) for e in tr.last_events][:6]}")
                    if away is None:
                        continue
                    b_me = tr.get(me.sid).bbox; b_o = o_a.bbox if o_a is not None else None
                    step(away)
                    if obs.levels_completed > lv0:
                        return True, steps, "contact", obs
                    me3 = tr.get(me.sid); o3 = tr.get(o.sid)
                    if me3 is None or o3 is None or b_o is None:
                        continue
                    dm = (me3.bbox[0] - b_me[0], me3.bbox[1] - b_me[1]); do = (o3.bbox[0] - b_o[0], o3.bbox[1] - b_o[1])
                    if dm != (0, 0) and do == dm:
                        core_ = None
                        for c_ in tr.prev:
                            if c_.sid < 1000 and c_.sid != o3.sid and o3.bbox[0] <= c_.bbox[0] and o3.bbox[1] <= c_.bbox[1] and c_.bbox[2] <= o3.bbox[2] and c_.bbox[3] <= o3.bbox[3]:
                                core_ = (kind(c_), c_.bbox[0] - o3.bbox[0], c_.bbox[1] - o3.bbox[1], c_.bbox[2] - c_.bbox[0], c_.bbox[3] - c_.bbox[1])
                                break
                        model.carry = {"key": K, "kinds": {kind(o)}, "held": o.sid, "offset": (o3.bbox[0] - me3.bbox[0], o3.bbox[1] - me3.bbox[1]),
                                       "me_kind": kind(me3), "dims": (o.bbox[2] - o.bbox[0] + 1, o.bbox[3] - o.bbox[1] + 1), "core": core_,
                                       "held_colour": o3.body if o3.body != o.body else None}
                        model.bound.add((me.sid, o.sid))
                        # (what came along was carried: it has no law of its own
                        # under the keys, and is not the body the keys move)
                        for k_ in list(model.fixed.eff):
                            for sid_ in [s_ for s_ in model.fixed.eff[k_] if s_ == o.sid or (s_ >= 1000 and s_ // 1000 == o.sid)
                                         or any(s_ == c_.sid for c_ in tr.prev if c_.sid < 1000 and c_.sid != o.sid and o3.bbox[0] <= c_.bbox[0] and o3.bbox[1] <= c_.bbox[1] and c_.bbox[2] <= o3.bbox[2] and c_.bbox[3] <= o3.bbox[3])]:
                                model.fixed.eff[k_].pop(sid_, None)
                                model.fixed.alt[k_].pop(sid_, None)
                                model.fixed.support[k_].pop(sid_, None)
                                if model.active and sid_ in model.active:
                                    model.active = frozenset(model.active) - {sid_}
                        # (what came along was carried, not pushed: no push law of it)
                        model.pushable.discard(kind(o))
                        for pk_ in [pk_ for pk_ in model.push_vec if pk_[0] == kind(o)]:
                            model.push_vec.pop(pk_, None)
                        note(f"carried: after {K} against S{o.sid} it came along with the body — {K} binds the thing faced; the body carries it")
                        if verbose:
                            print(f"   carry law: key {K} binds kind {kind(o)}; offset {model.carry['offset']}")
                        # does the key let go again?
                        b_o = o3.bbox
                        step(K)
                        if obs.levels_completed > lv0:
                            return True, steps, "contact", obs
                        b_me = tr.get(me.sid).bbox
                        step(away)
                        if obs.levels_completed > lv0:
                            return True, steps, "contact", obs
                        me4 = tr.get(me.sid); o4 = tr.get(o.sid)
                        if me4 is not None and o4 is not None and (me4.bbox[0] - b_me[0], me4.bbox[1] - b_me[1]) != (0, 0) and o4.bbox == b_o:
                            model.carry["toggle"] = True
                            model.bound.discard((me.sid, o.sid)); model.carry["held"] = None
                            note(f"{K} again lets the carried thing go")
                        else:
                            model.carry["toggle"] = False
                        break
                    elif recol:
                        continue
    # CARRY EACH THING TO ITS PLACE: a key pressed against a thing bound it
    # to the body; the reading says where the things of its kind go
    if getattr(model, "carry", None):
        if carry_protocol(tr, model, step, lambda: obs, lv0, verbose, hyps, note, keys):
            return True, steps, "carry", obs
    # A PUSH FOUND: measure it in the open.  The first shove may have
    # been stopped by what stood behind; shove the thing again where the
    # ground beyond it is clear, to tell a fixed leap from a slide.
    if model.pushable and model.key_selected:
        marks = selected_hosts(tr.prev)
        sel0 = [x_ for x_ in (model.selected_now(tr.prev),) if x_ is not None]
        cur_sel = sel0[0] if sel0 else None
        # who can be the pusher: the marked thing, and anything a click
        # can hand the mark to
        pushers = [s for s in tr.prev if s.sid == cur_sel or (s.body in model.select_kinds and "hud" not in s.tags)]
        blocking = model.blocking()
        trials = []
        for me in pushers:
            for o in tr.prev:
                if o.sid == me.sid or o.body not in model.pushable or "hud" in o.tags:
                    continue
                occupied = set()
                for s_ in tr.prev:
                    if "terrain" not in s_.tags and "hud" not in s_.tags and s_.sid != me.sid:
                        occupied |= set(s_.cells)
                for k, v in model.key_selected.items():
                    d = ((v[0] > 0) - (v[0] < 0), (v[1] > 0) - (v[1] < 0))
                    r0, c0, r1, c1 = o.bbox
                    run = 0
                    for n_ in range(1, 25):
                        if d[0]:
                            line = [((r1 if d[0] > 0 else r0) + d[0] * n_, c) for c in range(c0, c1 + 1)]
                        else:
                            line = [(r, (c1 if d[1] > 0 else c0) + d[1] * n_) for r in range(r0, r1 + 1)]
                        if any(r < 0 or c < 0 or r >= 64 or c >= 64 or (r, c) in occupied or int(f[r, c]) in blocking for r, c in line):
                            break
                        run = n_
                    trials.append((run, 0 if me.sid == cur_sel else 1, me.sid, o.sid, k, d))
        trials.sort(key=lambda t: (-t[0], t[1]))
        if verbose:
            print(f"   shove test: trials {trials[:4]}")
        for run, _, me_sid, o_sid, k, d in trials:
            if run < 9:
                break
            me = next(s for s in tr.prev if s.sid == me_sid); o = next(s for s in tr.prev if s.sid == o_sid)
            hr, wc = o.dims
            mh, mw = me.dims
            want = (o.centre[0] - d[0] * (hr + mh) / 2.0, o.centre[1] - d[1] * (wc + mw) / 2.0)
            comps = tr.prev
            start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
            start["_sel"] = me.sid
            solids0 = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2}
            succ0 = make_successors(model, comps, solids0, keys, me.sid, want, frame=f, frozen=(o.sid,))

            def at_want(st, _m=me.sid, _w=want):
                a_ = st.get(_m)
                if not a_:
                    return 999
                ca = centre(a_)
                return abs(ca[0] - _w[0]) + abs(ca[1] - _w[1])
            p0 = plan_general(start, succ0, lambda st: at_want(st) <= 1.0, heuristic=lambda st: at_want(st) / 3.0, max_states=20000, max_depth=30)
            if verbose:
                print(f"   shove test: S{me.sid} shoves S{o.sid} along {d} run {run}: approach {p0 if p0 is None else len(p0)} steps")
            note(f"shove test: S{me.sid} shoves S{o.sid} along {d} where the ground beyond is clear — a fixed leap or a slide?")
            if p0 is None:
                continue
            if me.sid != cur_sel:
                step(("select", me.sid))
            for k_ in p0[:14]:
                step(k_)
            step(k); step(k)
            if verbose:
                print(f"   shove test: push_vec={model.push_vec}")
            if obs.levels_completed > lv0:
                return True, steps, "shove", obs
            break
    # A POSITIONAL KEY (a track): press it until it teaches nothing new —
    # every slot's answer is then known and the track can be planned on
    for k in list(model.positional):
        note(f"track: pressing {k} until every slot's answer is known")
        for _ in range(16):
            n0 = len(model.pos_eff.get(k, {}))
            step(k)
            if obs.levels_completed > lv0:
                return True, steps, "track", obs
            if len(model.pos_eff.get(k, {})) == n0:
                break
        if verbose:
            print(f"   track {k}: {len(model.pos_eff.get(k, {}))} slots known, {len(model.pos_bad.get(k, set()))} contested")
    # a relative law found: probe its reach with two longer offsets
    for kd in list(model.rel):
        law = model.relative_law(kd)
        if law is None:
            continue
        for off in (9, 14):
            host = next((s for s in tr.prev if s.body == kd and s.size <= 60), None)
            if host is None:
                break
            r, c = host.centre
            step(("click", (int(round(r)), int(round(min(62, c + off))))))
    play_level.last = {"model": model, "tr": tr, "hyps": hyps, "keys": keys, "clickable": clickable}
    look_again()
    if verbose:
        print("   model:", model.describe())
        print("   rel obs:", {k: v for k, v in model.rel.items()})
        for k in model.fixed.keys():
            print("   key", k, dict(model.fixed.moving(k)), "contra", model.fixed.contradicted.get(k))
    # --- goal and plan: several movers of one kind with several markers
    # of one kind is one goal (all at once), then the single readings

    # THE LADDER CARRIED: when the last level taught that like things brought
    # together become the next kind, and this level's readings want a kind
    # not on the field that the ladder makes, the merging comes first (su15
    # L4: the twin's kind made of eight dots; the junk readings would spend
    # the clock walking the pieces onto one another)
    if getattr(progress, "merges_carried", False) and clickable and not getattr(progress, "_merged", False):
        try:
            field_kinds_ = {s_.body for s_ in tr.prev if s_.sid < 1000 and "hud" not in s_.tags and "terrain" not in s_.tags
                            and (pf is None or (pf[0] <= s_.bbox[0] and s_.bbox[2] < pf[1] and pf[2] <= s_.bbox[1] and s_.bbox[3] < pf[3]))}
            tops_ = set(progress.merges.values()) - set(progress.merges)
            if tops_ and not (tops_ & field_kinds_) and (set(progress.merges) & field_kinds_):
                note("the ladder carried: the kind the readings want is not on the field and the merges make it — merging first")
                res_ = merge_up(tr, model, step, lambda: obs, lv0, verbose, progress, note, clickable, look_again, lambda: pursue(hyps))
                if res_:
                    return res_
        except Exception as ex_:
            if verbose:
                print(f"   (the ladder carried failed: {ex_!r})")
    res_ = pursue(hyps)
    if res_:
        return res_
    # A KIND WHOSE THINGS ANSWER A CLICK EACH IN ITS OWN WAY IS ASKED THING BY
    # THING: when two clicked things of one kind gave different laws (vc33 L3:
    # each foot raises one column and lowers another), the kind's answer is
    # per thing, and no reading is given up for want of a plan while a thing
    # of that kind stands unasked.  Each is clicked once; then the readings
    # again.  (At most eight things asked here — hand-chosen; NO_ASK_EACH)
    if clickable and not _ENV.get("NO_ASK_EACH") and obs.levels_completed <= lv0:
        try:
            clicked_pts_ = [k_[1] for k_ in model.fixed.eff if isinstance(k_, tuple) and k_[0] == "click"]
            by_kind_ = defaultdict(list)
            for s_ in tr.prev:
                if s_.sid >= 1000 or "hud" in s_.tags or "terrain" in s_.tags or s_.size > 16:
                    continue
                pts_ = [p_ for p_ in clicked_pts_ if p_ in s_.cells]
                if pts_:
                    law_ = model.fixed.moving(("click", pts_[0]))
                    by_kind_[s_.body].append((s_, frozenset((sid_, tuple(e_)) for sid_, e_ in law_.items())))
            asked_ = 0
            for kd_, lst_ in by_kind_.items():
                if len(lst_) < 2 or len({l_ for _, l_ in lst_}) < 2 or not any(l_ for _, l_ in lst_):
                    continue
                sz_ = {x_.size for x_, _ in lst_}
                for s_ in tr.prev:
                    if asked_ >= 8:
                        break
                    if s_.sid >= 1000 or s_.body != kd_ or "hud" in s_.tags or s_.size not in sz_ or any(p_ in s_.cells for p_ in clicked_pts_):
                        continue
                    cp_ = click_point(s_)
                    note(f"things of kind {kd_} answer a click each in its own way: asking S{s_.sid} at {cp_}")
                    if verbose:
                        print(f"   ask each: kind {kd_} S{s_.sid} at {cp_}")
                    step(("click", cp_))
                    asked_ += 1
                    if obs.levels_completed > lv0:
                        return True, steps, "keys", obs
            if asked_:
                look_again()
                res_ = pursue(hyps)
                if res_:
                    return res_
        except Exception as ex_:
            if verbose:
                print(f"   (ask each failed: {ex_!r})")
    # A KIND NEVER ASKED IS ASKED BEFORE THE LEVEL IS GIVEN UP (goals from
    # effects, statement 1: the subject is what the presses change — a thing
    # no press has ever touched may be the subject).  When no reading has a
    # plan and a kind stands on the frame that no law, no refusal, no push,
    # no click and no walk has ever reached (the failure report's "kinds
    # never asked" — wa30 L3: the block at the top, kind 12, while the carry
    # failed under the wall), the walker is sent onto a thing of it, or it
    # is clicked when nothing walks; then the readings again.  Kinds of few
    # things only (at most four — hand-chosen; a strewn kind is the board).
    if not _ENV.get("NO_ASK_UNASKED") and obs.levels_completed <= lv0:
        try:
            kind_by_sid_u = {s_.sid: s_.body for s_ in tr.prev}
            asked_u = set()
            for k2_ in model.fixed.eff:
                asked_u |= {kind_by_sid_u.get(s2_) for s2_ in model.fixed.eff[k2_]}
            asked_u |= set(model.ground_ok) | set(model.flat) | set(model.pushable) | set(model.wall_kinds) | set(model.blocking()) | set(model.ground_refused)
            asked_u |= {a_ for pr_ in model.passable for a_ in pr_} | set(getattr(model, "agents", ()) or ()) | set(getattr(model, "touched_alive", {}) or {})
            asked_u |= {kind_by_sid_u.get(s2_) for k2_ in model.fixed.eff if isinstance(k2_, tuple) for s2_ in model.fixed.eff[k2_]}
            clicked_u = [k_[1] for k_ in model.fixed.eff if isinstance(k_, tuple) and k_[0] == "click"]
            present_u = defaultdict(list)
            for s_ in tr.prev:
                if s_.sid < 1000 and "hud" not in s_.tags and "terrain" not in s_.tags and s_.size >= 2 and s_.body not in asked_u and not any(p_ in s_.cells for p_ in clicked_u):
                    present_u[s_.body].append(s_)
            asked_n = 0
            if verbose:
                print(f"   ask unasked: present {dict((k_, len(v_)) for k_, v_ in present_u.items())} asked {sorted(k_ for k_ in asked_u if k_ is not None)} active {sorted(model.active or ())}")
            act_u = [s_ for s_ in tr.prev if s_.sid in (model.active or ())]
            me_u = max(act_u, key=lambda s_: s_.size) if act_u else None
            for kd_, lst_ in sorted(present_u.items(), key=lambda kv: len(kv[1])):
                if len(lst_) > 4 or asked_n >= 4:
                    continue
                t_ = min(lst_, key=lambda s_: (abs(s_.centre[0] - me_u.centre[0]) + abs(s_.centre[1] - me_u.centre[1])) if me_u is not None else s_.size)
                note(f"kind {kd_} was never asked: asking S{t_.sid} at {t_.bbox}")
                if verbose:
                    print(f"   ask unasked: kind {kd_} S{t_.sid} at {t_.bbox} by {'walk' if me_u is not None else 'click'}")
                asked_n += 1
                if me_u is not None:
                    _walk_over(tr, model, keys, me_u.sid, t_, step, verbose, max_steps=40)
                elif clickable:
                    step(("click", click_point(t_)))
                if obs.levels_completed > lv0:
                    return True, steps, "keys", obs
            if asked_n:
                look_again()
                res_ = pursue(hyps)
                if res_:
                    return res_
        except Exception as ex_:
            if verbose:
                print(f"   (ask unasked failed: {ex_!r})")
    play_level.last = {"model": model, "tr": tr, "hyps": hyps, "keys": keys, "clickable": clickable}
    # THE DISPLAY PROTOCOL: a thing that changed when the mover arrived
    # somewhere is a DISPLAY; a static picture of the same colours is its
    # REFERENCE (at any scale); the readings say: change the display by
    # arriving again until the two pictures agree, then bring yourself to
    # the reference
    if model.displays and model.triggers:
        note("display protocol: a thing changed when the mover arrived — change it until it matches its picture")
        won, n_ = display_protocol(env, tr, model, keys, step, lambda: obs, lv0, verbose)
        if won:
            return True, steps, "display", obs
    # EXPLORE: nothing to aim at from here — the world may be larger than
    # the screen (a scene that scrolls), or the way may show only from
    # elsewhere.  The body walks to the farthest place it can reach; what
    # the walk shows (a slide, a scroll, a thing that comes into view) is
    # read, and the readings are tried again from there
    if model.active and keys:
        for leg in range(6):
            if time.time() - play_level.t0 > _level_seconds():
                break
            went = explore_leg(tr, model, keys, step, note, verbose, lambda: obs, lv0)
            if obs.levels_completed > lv0:
                return True, steps, "explore", obs
            # (a twin read on the way: the record is a second body — the twin
            # protocol before any further reading)
            if getattr(model, "twin", None) is not None and not model.twin.confirmed and not _ENV.get("NO_LEVER"):
                if twin_probe(tr, model, keys, step, lambda: obs, lv0, verbose, note):
                    return True, steps, "twin", obs
            if getattr(model, "twin", None) is not None and model.twin.confirmed and not _ENV.get("NO_LEVER") and not getattr(play_level, "_twin_tried", False):
                play_level._twin_tried = True
                if twin_protocol(tr, model, keys, step, lambda: obs, lv0, verbose, note, hyps):
                    return True, steps, "twin", obs
            if not went:
                break
            n_before = len(hyps)
            look_again()
            fresh = [h for h in hyps if not getattr(h, "_tried", False)]
            for h in hyps:
                h._tried = True
            if fresh:
                res_ = pursue(fresh)
                if res_:
                    return res_
    # THE IMPOSSIBLE SITUATION: no reading plans, nowhere to explore.
    # Tushar: "I fiddled and found the block that comes and goes on a
    # click" — a thing of a kind no click has ever touched is clicked
    # (one per kind, the nearest first); whatever it taught, the readings
    # and the exploration are tried again from there
    if model.active:
        for round_ in range(3):
            if time.time() - play_level.t0 > _level_seconds():
                break
            body_ = set()
            for x in tr.prev:
                if x.sid in model.active:
                    body_ |= set(x.cells)
            bc_ = centre(body_) if body_ else (32, 32)
            # (a kind is asked once per FORM: the block of a colour and the
            # mark of the same colour are two questions)
            def form_(sz_):
                return "mark" if sz_ <= 12 else "block"
            tried_ = set(getattr(play_level, "fiddle_clicked", set()))
            tried_ |= {(kd_, form_(sz_)) for kd_, sz_ in getattr(model, "click_gone_size", {}).items()}
            tried_ |= {(kd_, f_) for kd_ in model.click_stays for f_ in ("mark", "block") if kd_ not in getattr(model, "click_grows", {})}
            tried_ |= {(kd_, "mark") for kd_ in getattr(model, "click_grows", {})}
            cands_ = [x for x in tr.prev if x.sid < 1000 and "terrain" not in x.tags and "hud" not in x.tags and "bar" not in x.tags
                      and x.sid not in model.active and (x.body, form_(x.size)) not in tried_ and 2 <= x.size <= 160
                      and x.body not in model.hazards()]
            by_kind_ = {}
            for x in sorted(cands_, key=lambda x_: abs(x_.centre[0] - bc_[0]) + abs(x_.centre[1] - bc_[1])):
                by_kind_.setdefault((x.body, form_(x.size)), x)
            if not by_kind_:
                break
            learned_ = False
            for kd_, x in list(by_kind_.items())[:4]:
                pt_ = click_point(x)
                note(f"an impossible situation: nothing plans and nowhere to go — fiddling: a click on the {x.body}-coloured thing at {x.bbox[:2]} (its kind never clicked)")
                f0_ = f
                step(("click", pt_))
                play_level.fiddle_clicked = getattr(play_level, "fiddle_clicked", set()) | {kd_}      # (kd_ is (kind, form))
                if obs.levels_completed > lv0:
                    return True, steps, "fiddle", obs
                if (f != f0_).sum() > 2:
                    learned_ = True
                    note(f"the click changed the board ({int((f != f0_).sum())} cells): reading again")
                    break
            if not learned_:
                continue
            look_again()
            for h in hyps:
                h._tried = False
            res_ = pursue(hyps)
            if res_:
                return res_
            for leg in range(6):
                if time.time() - play_level.t0 > _level_seconds():
                    break
                went = explore_leg(tr, model, keys, step, note, verbose, lambda: obs, lv0)
                if obs.levels_completed > lv0:
                    return True, steps, "explore", obs
                if not went:
                    break
                look_again()
                fresh = [h for h in hyps if not getattr(h, "_tried", False)]
                for h in hyps:
                    h._tried = True
                if fresh:
                    res_ = pursue(fresh)
                    if res_:
                        return res_
    # WHAT COUNTS: THE PROGRESS LOOP (agent/core/progress.py).  Every change
    # of a thing the keys do not control, tied to what the controllable was
    # doing on that press, is a rule instance; its generalisation is the
    # goal — the cause done again to every thing of the cause's kind, until
    # the score completes.  The last resort, after every reading, organ,
    # exploration and fiddle (its presses are not free: ls20's display
    # tokens, bp35's dropped blocks); refuted when two targets in a row
    # move nothing.
    res_ = progress_protocol(tr, model, keys, step, lambda: obs, lv0, verbose, progress, note, clickable, look_again, lambda: pursue(hyps))
    if res_:
        return res_
    return obs.levels_completed > lv0, steps, "no_plan", obs


def _picture(cells):
    """A picture's shape, normalised and downscaled to unit pitch: the set
    of (r, c) of its cells with the smallest step between cells as 1."""
    cs = sorted(cells)
    if not cs:
        return frozenset()
    rs = sorted({r for r, _ in cs}); cc = sorted({c for _, c in cs})
    pitch = 1
    r0, c0 = rs[0], cc[0]
    cset = set(cs)
    for k in (4, 3, 2):
        if len(cs) < k * k:
            continue
        blocks = {((r - r0) // k, (c - c0) // k) for r, c in cs}
        if len(blocks) * k * k == len(cs) and all((r0 + br * k + i, c0 + bc * k + j) in cset
                                                  for br, bc in blocks for i in range(k) for j in range(k)):
            pitch = k; break
    return frozenset(((r - r0) // pitch, (c - c0) // pitch) for r, c in cs)


def _pnorm(cells):
    cs = list(cells)
    if not cs:
        return frozenset()
    r0 = min(r for r, _ in cs); c0 = min(c for _, c in cs)
    return frozenset((r - r0, c - c0) for r, c in cs)


def _dihedral_ops():
    """The eight rigid motions of a picture, as (name, fn) on normalised
    pictures."""
    def rot(p): return _pnorm([(c, -r) for r, c in p])
    def mir(p): return _pnorm([(r, -c) for r, c in p])
    ops = []
    for k in range(4):
        def f(p, k=k):
            for _ in range(k):
                p = rot(p)
            return p
        ops.append((f"rot{k}", f))
        def g(p, k=k):
            p = mir(p)
            for _ in range(k):
                p = rot(p)
            return p
        ops.append((f"mir+rot{k}", g))
    return ops


def _visit_law(history):
    """What one visit does to the picture: the rigid motion that maps every
    picture in the history to the next (None when none does, or when
    there is nothing to learn from yet)."""
    pairs = [(history[i], history[i + 1]) for i in range(len(history) - 1) if history[i] != history[i + 1]]
    if not pairs:
        return None
    for name, f in _dihedral_ops():
        if name == "rot0":
            continue
        if all(f(a) == b for a, b in pairs):
            return name, f
    return None


def _visits_to(pic, rpic, law, cap=8):
    """How many visits under `law` bring `pic` to `rpic` (None: never)."""
    cur = pic
    for n in range(1, cap + 1):
        cur = law[1](cur)
        if cur == rpic:
            return n
    return None


def display_protocol(env, tr, model, keys, step, get_obs, lv0, verbose):
    import itertools as _it
    n = 0
    disp_sids = [sid for sid, _ in model.displays.most_common(3)]
    for dsid in disp_sids:
        # the display now: the union of things of its colour near it (a
        # picture may be read as several pieces)
        d0 = next((s for s in tr.prev if s.sid == dsid), None)
        if d0 is None:
            continue

        # THE PICTURE IS THE DRAWING, NOT THE BOARD: a display read as one
        # object (a box with a glyph on it) shows its glyph in a colour
        # that is not its body; a display read as the glyph alone shows
        # its body.  The reference is read the same way.
        d_obj = next((s for s in tr.prev if s.sid == dsid), None)
        d_cols = Counter(d_obj.cells.values()) if d_obj is not None else Counter()
        pic_col = d_obj.body if d_obj is not None else None
        if d_obj is not None and len(d_cols) >= 2:
            # the INK is the sparser of the display's colours in the frame
            # around it (a glyph on a board; a board carries more paint than
            # its glyph).  Counted on the frame, not on the tracked cells: a
            # tracked piece keeps cells that were recoloured under it.
            fr_ = getattr(model, "_frame_now", None)
            near_ = [s for s in tr.prev if s.sid < 1000 and set(s.colors) & set(d_cols)
                     and abs(s.centre[0] - d_obj.centre[0]) + abs(s.centre[1] - d_obj.centre[1]) <= 12]
            if fr_ is not None and near_:
                r0_ = min(q[0] for s in near_ for q in s.cells); r1_ = max(q[0] for s in near_ for q in s.cells)
                c0_ = min(q[1] for s in near_ for q in s.cells); c1_ = max(q[1] for s in near_ for q in s.cells)
                cnt_ = Counter(int(fr_[r, c]) for r in range(r0_, r1_ + 1) for c in range(c0_, c1_ + 1))
                pic_col = min(d_cols, key=lambda c_: (cnt_.get(c_, 0), c_ == d_obj.body))
            else:
                pic_col = max((c_ for c_ in d_cols if c_ != d_obj.body), key=lambda c_: d_cols[c_])

        def picture_of(sid_ref, ink=None):
            ink = pic_col if ink is None else ink
            base = next((s for s in tr.prev if s.sid == sid_ref), None)
            if base is None:
                return None, None
            cells = set()
            near = []
            for s in tr.prev:
                if s.sid >= 1000 or abs(s.centre[0] - base.centre[0]) + abs(s.centre[1] - base.centre[1]) > 12:
                    continue
                if s.sid == base.sid or s.body == ink:
                    near.append(s)
                    cells |= {q for q, c_ in s.cells.items() if c_ == ink}
            # the picture is read from the FRAME inside the box of the near
            # things: tracker pieces may trade cells between frames (a glyph
            # that turns is two objects to the tracker) and their remembered
            # cells then lag the picture on screen
            fr = getattr(model, "_frame_now", None)
            if fr is not None and near:
                r0 = min(q[0] for s in near for q in s.cells)
                r1 = max(q[0] for s in near for q in s.cells)
                c0 = min(q[1] for s in near for q in s.cells)
                c1 = max(q[1] for s in near for q in s.cells)
                fcells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                          if int(fr[r, c]) == ink}
                if fcells:
                    cells = fcells
            return base, cells
        # THE DISPLAY'S STATE is what it shows: a shape in an ink.  Read
        # from the frame inside the display's box: the board is the
        # commonest colour there, the ink the commonest other colour (a
        # visit may change the ink as well as the shape — ls20's third
        # level cycles the carried glyph's colour)
        base0, dcells0 = picture_of(dsid)
        dbox = _bbox(dcells0) if dcells0 else base0.bbox
        # the BOARD is what surrounds the picture's box (the panel)
        fr0 = getattr(model, "_frame_now", None)
        board_col = None
        if fr0 is not None:
            r0_, c0_, r1_, c1_ = dbox
            ring = Counter(int(fr0[r, c]) for r in range(max(0, r0_ - 2), min(64, r1_ + 3)) for c in range(max(0, c0_ - 2), min(64, c1_ + 3))
                           if not (r0_ <= r <= r1_ and c0_ <= c <= c1_))
            if ring:
                board_col = ring.most_common(1)[0][0]

        def display_state():
            fr = getattr(model, "_frame_now", None)
            if fr is None:
                _, cs = picture_of(dsid)
                return _pnorm(_picture(cs or ())), pic_col
            r0, c0, r1, c1 = dbox
            cnt = Counter(int(fr[r, c]) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
            if not cnt:
                return frozenset(), pic_col
            board = board_col if board_col is not None else cnt.most_common(1)[0][0]
            others = [(n, c) for c, n in cnt.items() if c != board]
            if not others:
                # the box is the picture itself (a glyph read alone): ink = board
                cs = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if int(fr[r, c]) == board}
                return _pnorm(_picture(cs)), board
            ink = max(others)[1]
            cs = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if int(fr[r, c]) == ink}
            return _pnorm(_picture(cs)), ink
        # a reference: another picture far from the display, not a display
        # itself, whose normalised shape has about the display's number of
        # unit cells — a picture is a SHAPE: one of the display's colour
        # first, else of another (ls20's third level shows the gate's
        # glyph in blue and the display's in orange); not the controlled
        # thing (it is no picture of anything)
        refs = []
        base, dcells = picture_of(dsid)
        dpic = _picture(dcells)
        seen = set()
        active_ = set(model.active or ())
        # the display sits on a BOARD (the panel around it): a reference on
        # a board of the same colour ranks first (a glyph in a gate box, as
        # the glyph on the panel), then one in the display's ink, then the
        # nearest
        def on_board(s):
            if board_col is None:
                return False
            b = s.bbox
            return any(o.sid != s.sid and o.body == board_col and o.size > s.size
                       and o.bbox[0] <= b[0] and o.bbox[1] <= b[1] and o.bbox[2] >= b[2] and o.bbox[3] >= b[3] for o in tr.prev)
        for s in sorted(tr.prev, key=lambda s: (not on_board(s), pic_col not in s.colors, abs(s.centre[0] - base.centre[0]) + abs(s.centre[1] - base.centre[1]))):
            if s.sid in seen or s.sid >= 1000 or s.sid in model.displays or "hud" in s.tags or "terrain" in s.tags or s.sid in active_:
                continue
            if abs(s.centre[0] - base.centre[0]) + abs(s.centre[1] - base.centre[1]) <= 12:
                continue
            if s.size > 4 * len(dcells) or s.size < 2:
                continue
            ink_ = pic_col if pic_col in s.colors else s.body
            rb, rcells = picture_of(s.sid, ink_)
            for t in tr.prev:
                if set(t.cells) <= rcells:
                    seen.add(t.sid)
            rpic = _picture(rcells)
            if rpic and 0.5 * len(dpic) <= len(rpic) <= 2.0 * len(dpic):
                refs.append((rb, rcells, rpic, ink_))
        if verbose:
            print(f"   display S{dsid} picture {sorted(dpic)[:8]}... refs {[(rb.sid, len(rp), ik) for rb, _, rp, ik in refs]} triggers {dict(model.triggers)}")
        if not refs or not dpic:
            continue
        rb, rcells, rpic, rink = refs[0]
        model.display_candidate = (pic_col, rink)
        # the mover
        members = [x for x in tr.prev if x.sid in (model.active or ())]
        me = max(members, key=lambda x: x.size) if members else None
        if me is None:
            continue
        # the trigger things: those of the kinds that changed this display
        tkinds = [k for k, lst in model.triggers.items() if any(d == dsid for d, _ in lst)]
        # THE TILES: a trigger is a tile — the small things touching each
        # other at a place (the two colours of one glyph on the floor);
        # each tile has a LAW on the display's state, learned by visiting
        # it: what it does to the shape (a rigid motion, or nothing) and
        # what it does to the ink (ink -> ink).  Which tiles to visit, and
        # how often, is then a search over states; a tile whose law is not
        # known yet is visited once to learn it (nearest first)
        def tiles_now():
            act_ = set(model.active or ())
            cands = [t for t in tr.prev if t.body in tkinds and "hud" not in t.tags and "terrain" not in t.tags and t.sid < 1000
                     and t.size <= 32 and t.sid not in act_]
            tiles = []
            used = set()
            for t in sorted(cands, key=lambda t: -t.size):
                if t.sid in used:
                    continue
                group = [t]; used.add(t.sid)
                tb = t.bbox
                for u in tr.prev:
                    if u.sid in used or u.sid >= 1000 or u.size > 32 or "hud" in u.tags or "terrain" in u.tags or u.sid in act_:
                        continue
                    b = u.bbox
                    if b[0] <= tb[2] + 1 and tb[0] <= b[2] + 1 and b[1] <= tb[3] + 1 and tb[1] <= b[3] + 1:
                        group.append(u); used.add(u.sid)
                # (the tile's colours, less the mover's own: a piece of the
                # mover's colour is swallowed while it stands there)
                mcols_ = {x.body for x in tr.prev if x.sid in act_}
                key_ = frozenset(u.body for u in group if u.body not in mcols_) or frozenset(u.body for u in group)
                big = max(group, key=lambda u: u.size)
                cells_ = frozenset(q for u in group for q in u.cells)
                tiles.append((key_, big, cells_))
            return tiles
        laws = getattr(model, "tile_laws", None)
        if laws is None:
            laws = model.tile_laws = {}          # tile colours -> {"shape": op name | "same" | None, "ink": {ink: ink}}
        ops_ = dict(_dihedral_ops())

        def apply_law(law, st):
            shp, ink = st
            if law["shape"] is None:
                return None
            if law["shape"] != "same":
                shp = ops_[law["shape"]](shp)
            if law["ink"] and ink in law["ink"]:
                ink = law["ink"][ink]
            elif law["ink"]:
                return None                     # this ink's successor is not known yet
            return (shp, ink)

        def plan_visits(st, goal_st, tiles):
            """Shortest sequence of tile keys (known laws) from st to goal_st."""
            from collections import deque
            keys_ = [k_ for k_, _, _ in tiles if k_ in laws and laws[k_]["shape"] is not None]
            q = deque([(st, [])]); seen_ = {st}
            while q:
                cur, path = q.popleft()
                if cur == goal_st:
                    return path
                if len(path) >= 8:
                    continue
                for k_ in keys_:
                    nxt = apply_law(laws[k_], cur)
                    if nxt is None or nxt in seen_:
                        continue
                    seen_.add(nxt); q.append((nxt, path + [k_]))
            return None

        rpic_n = _pnorm(rpic)
        goal_st = (rpic_n, rink)
        deaths0 = getattr(model, "deaths", 0)
        st_before = None; tile_before = None
        bad_tiles = set()                       # tiles no walk reaches
        refilled_for = set()                    # tiles a refill was sought for
        for cycle in range(12):
            st_now = display_state()
            if getattr(model, "deaths", 0) != deaths0:
                deaths0 = model.deaths; st_before = None   # a death put the picture back: the last visit taught nothing
            # LEARN from the last visit — from as many ARRIVALS as the trail
            # shows (a walk may cross the tile twice: two applications)
            arrivals = 1
            if st_before is not None and tile_before is not None and getattr(play_level, "trail", None) is not None:
                tr_ = play_level.trail[trail0:]
                on_ = [bool(set().union(*[set(v) for v in fr_.values()]) & tcells_before) if fr_ else False for fr_ in tr_]
                arrivals = sum(1 for i_, o_ in enumerate(on_) if o_ and (i_ == 0 and not on_start or i_ > 0 and not on_[i_ - 1]))
            if st_before is not None and tile_before is not None and arrivals >= 1:
                law = laws.setdefault(tile_before, {"shape": None, "ink": {}})
                if st_now[0] == st_before[0]:
                    law["shape"] = "same" if arrivals == 1 or law["shape"] is None else law["shape"]
                else:
                    # (a turn before a mirror; the law already held before
                    # every other — a symmetric picture fits several)
                    def pow_(f, k, x):
                        for _ in range(k):
                            x = f(x)
                        return x
                    ops_l = [(nm, f) for nm, f in _dihedral_ops() if nm != "rot0"]
                    ops_l.sort(key=lambda nf: (nf[0] != law["shape"], nf[0].startswith("mir")))
                    op = next((nm for nm, f in ops_l if pow_(f, arrivals, st_before[0]) == st_now[0]), None)
                    law["shape"] = op if op is not None else law["shape"]
                    if op is not None:
                        model.visit_law = op
                if arrivals == 1:
                    law["ink"][st_before[1]] = st_now[1]
                elif st_before[1] != st_now[1] and st_before[1] not in law["ink"]:
                    pass                        # several arrivals: which ink followed which is not known
                note(f"display: the tile {sorted(tile_before)} {'turns the picture ' + law['shape'] if law['shape'] not in (None, 'same') else 'leaves the shape'}"
                     f"{', ink ' + str(st_before[1]) + ' -> ' + str(st_now[1]) if st_before[1] != st_now[1] else ''}")
            if st_now == goal_st:
                if verbose:
                    print(f"   display: pictures agree after {cycle} visits")
                model.display_goal = model.display_candidate
                # a refusal at the reference's door was the picture's: forgiven now
                rb_ = _bbox(rcells)
                model.refused_at = {e for e in model.refused_at
                                    if not (rb_[0] - 8 <= e[1] <= rb_[2] + 8 and rb_[1] - 8 <= e[2] <= rb_[3] + 8)}
                break
            tiles = [t_ for t_ in tiles_now() if t_[0] not in bad_tiles]
            if not tiles:
                break
            me_now = next((x for x in tr.prev if x.sid == me.sid), me)
            path = plan_visits(st_now, goal_st, tiles)
            if path:
                key_ = path[0]
                visits_left = len(path)
            else:
                # a tile never visited first (each tile once, nearest
                # first), then one whose ink law lacks the ink shown now;
                # none left: the visits cannot reach the reference
                never = [(k_, big, cs) for k_, big, cs in tiles if k_ not in laws or laws[k_]["shape"] is None]
                partial = [(k_, big, cs) for k_, big, cs in tiles if k_ in laws and laws[k_]["shape"] is not None
                           and laws[k_]["ink"] and st_now[1] not in laws[k_]["ink"]]
                unknown = never or partial
                if not unknown:
                    note("display: every tile is understood and no visits reach the reference")
                    break
                key_ = min(unknown, key=lambda x: abs(x[1].centre[0] - me_now.centre[0]) + abs(x[1].centre[1] - me_now.centre[1]))[0]
                visits_left = 1
            t = next(big for k_, big, cs in tiles if k_ == key_)
            tcells_ = next(cs for k_, big, cs in tiles if k_ == key_)
            # THE CLOCK: the visits to come cost two presses each; after
            # them the mover must still reach the reference (a death would
            # put the picture back).  Whether the trip from the tile fits
            # the clock left after the visits is asked of the planner (with
            # the refills it knows); when it does not, a refill comes first
            # if one is within reach — else the visits go on and the trip
            # takes the refills it can find
            left_ = model.clock_left(tr.prev) if getattr(model, "clock_rate", 0) else None
            need_ = 2 * (visits_left - 1) + (0 if path else 2)
            standing = bool(set(me_now.cells) & tcells_)
            cost_visits = 2 * visits_left if standing else 2 * (visits_left - 1)     # off-and-on per visit; a walk's last step arrives
            trip_fits = True
            if left_ is not None:
                trip_fits = False
                try:
                    step_len = max([abs(e_[1][0]) + abs(e_[1][1]) for k_ in keys for e_ in [model.fixed.moving(k_).get(me.sid)]
                                    if e_ and e_[0] == "translate"] or [1])
                    rc_ = centre(rcells)
                    dv_ = (int(round((t.centre[0] - me_now.centre[0]) / step_len)) * step_len,
                           int(round((t.centre[1] - me_now.centre[1]) / step_len)) * step_len)
                    body_ = {o for b in model.bodies() if me.sid in b for o in b} | {me.sid}
                    st_h = {x.sid: (frozenset(_shift(x.cells, dv_)) if x.sid in body_ else frozenset(x.cells)) for x in tr.prev if "hud" not in x.tags}
                    st_h["_sel"] = model.selected_now(tr.prev)
                    ct_ = model.clock_thing(tr.prev)
                    st_h["_clock"] = (model.clock_value(ct_) if ct_ is not None else 0) - cost_visits * model.clock_rate
                    solids_h = {x.sid for x in tr.prev if "terrain" not in x.tags and "hud" not in x.tags and x.size > 2
                                and x.body not in model.flat and not (set(x.cells) & set(rcells)) and x.sid not in body_}
                    succ_h = make_successors(model, tr.prev, solids_h, keys, me.sid, rc_, frame=(getattr(model, '_frame_now', None) if getattr(model, '_frame_now', None) is not None else model._frame), clock="known")

                    def goal_h(st_):
                        cs_ = set()
                        for o in body_:
                            if o in st_:
                                cs_ |= set(st_[o])
                        if not cs_:
                            return False
                        b_ = _bbox(cs_)
                        return b_[0] - 0.5 <= rc_[0] <= b_[2] + 0.5 and b_[1] - 0.5 <= rc_[1] <= b_[3] + 0.5
                    if st_h["_clock"] >= 0:
                        p_h = plan_general(st_h, succ_h, goal_h, heuristic=lambda st_: (abs(centre(st_[me.sid])[0] - rc_[0]) + abs(centre(st_[me.sid])[1] - rc_[1])) / 3.0 if me.sid in st_ else 999.0,
                                           max_states=6000, max_depth=40)
                        trip_fits = bool(p_h)
                except Exception:
                    trip_fits = True
            note(f"display: pictures differ — arrive on the tile {sorted(key_)} (S{t.sid}) to change the display (visit {cycle + 1}{', ' + str(visits_left) + ' planned' if path else ', to learn what it does'})")
            if verbose:
                print(f"   display now {sorted(st_now[0])} ink {st_now[1]} reference {sorted(rpic_n)} ink {rink} laws {laws} path {path}")
                print(f"   tiles {[(sorted(k_), big.sid, big.body, big.size, big.bbox, len(cs)) for k_, big, cs in tiles]} me {me_now.bbox} active {sorted(model.active or ())}")
                print(f"   things near me {[(x.sid, x.body, x.size, x.bbox, sorted(x.tags)) for x in tr.prev if abs(x.centre[0] - me_now.centre[0]) + abs(x.centre[1] - me_now.centre[1]) <= 8]}")
            st_before = st_now; tile_before = key_
            tcells_before = set(tcells_)
            trail0 = len(play_level.trail) if getattr(play_level, "trail", None) is not None else 0
            on_start = bool(set(me_now.cells) & tcells_before)
            # (with a clock read: the visits and the trip must fit the time
            # left — else a refill first, when one is within reach)
            if left_ is not None and (left_ < cost_visits or (path and not trip_fits)) and standing and key_ not in refilled_for:
                refills_ = [x for x in tr.prev if x.sid < 1000 and kind(x) in model.refills and "hud" not in x.tags]
                if not refills_:                # unproven: the clock's own colour on the floor
                    refills_ = [x for x in tr.prev if x.sid < 1000 and kind(x) == model.clock_colour and x.size <= 32
                                and "hud" not in x.tags and "terrain" not in x.tags and "bar" not in x.tags and x.sid != model.clock]
                if refills_:
                    rf = min(refills_, key=lambda x: abs(x.centre[0] - me_now.centre[0]) + abs(x.centre[1] - me_now.centre[1]))
                    note(f"display: {left_} presses left, {cost_visits} for the visits{'' if trip_fits else ' and the trip after does not fit'} — a refill first (S{rf.sid})")
                    refilled_for.add(key_)      # (once: a refill out of reach is not asked for again and again)
                    n0_ = n
                    n += _walk_over(tr, model, keys, me.sid, rf, step, verbose)
                    if n > n0_:
                        n += _walk_over(tr, model, keys, me.sid, t, step, verbose, need=need_)
                        if get_obs().levels_completed > lv0:
                            return True, n
                        st_before = None
                        continue
            # step off if standing on it, then arrive again
            if standing:
                # leave by the key opposite to the last that moved us
                last_v = None
                for k_ in keys:
                    v_ = model.fixed.moving(k_).get(me.sid)
                    if v_ and v_[0] == "translate":
                        last_v = (k_, v_[1])
                        break
                if last_v is None:
                    break
                step(last_v[0]); n += 1
                if get_obs().levels_completed > lv0:
                    return True, n
                # come back with the opposite key
                opp = next((k_ for k_ in keys if (model.fixed.moving(k_).get(me.sid) or ("", None))[0] == "translate"
                            and model.fixed.moving(k_)[me.sid][1] == (-last_v[1][0], -last_v[1][1])), None)
                if opp is None:
                    break
                step(opp); n += 1
                me_after = next((x for x in tr.prev if x.sid == me.sid), None)
                if me_after is None or not (set(me_after.cells) & tcells_):
                    st_before = None            # the step back did not arrive: nothing was learned
            else:
                n0_ = n
                # (the refill nearest the reference is kept for the way out)
                rc0_ = centre(rcells)
                rf_all = [x for x in tr.prev if x.sid < 1000 and "hud" not in x.tags and "terrain" not in x.tags
                          and (kind(x) in model.refills or (kind(x) == getattr(model, "clock_colour", None) and x.size <= 32 and x.sid != model.clock))]
                keep_ = ()
                if len(rf_all) >= 2:
                    keep_ = (min(rf_all, key=lambda x: abs(x.centre[0] - rc0_[0]) + abs(x.centre[1] - rc0_[1])).sid,)
                n += _walk_over(tr, model, keys, me.sid, t, step, verbose, need=need_, keep=keep_)
                me_after = next((x for x in tr.prev if x.sid == me.sid), None)
                if n == n0_:
                    bad_tiles.add(key_); st_before = None
                    note(f"display: no way to the tile {sorted(key_)} from here — another")
                    continue
                if me_after is None or not (set(me_after.cells) & tcells_):
                    st_before = None            # not there yet: nothing was learned
            if get_obs().levels_completed > lv0:
                return True, n
        # then bring yourself to the reference (the whole picture's centre)
        class _R:
            sid = rb.sid; body = rb.body; centre = centre(rcells); bbox = _bbox(rcells); cells = frozenset(rcells)
        n += _walk_over(tr, model, keys, me.sid, _R, step, verbose)
        if get_obs().levels_completed > lv0:
            model.display_goal = model.display_candidate
            model.display_won_by = "reach"      # the level ended when the mover reached the reference
            return True, n
    return False, n


def _press_deadly(model, tr, k):
    """A BLIND PRESS IS NOT PRESSED INTO A KNOWN HAZARD: when the key's law
    on the body is known and the step would put the body onto, or beside,
    a thing of a kind that has killed it, the press is skipped (the
    fiddle's repertoire and the confirming presses ask the keys — not
    the hazard twice; st20, the second look dying as the first)."""
    try:
        if not isinstance(k, str) or not model.active:
            return False
        hazard_kinds = model.hazards() if hasattr(model, "hazards") else set()
        if not hazard_kinds:
            return False
        touch_kinds = model.hazard_touch() if hasattr(model, "hazard_touch") else set(hazard_kinds)
        laws = model.fixed.moving(k)
        body = set(); dest = set()
        for x in tr.prev:
            if x.sid in model.active and x.sid < 1000:
                law = laws.get(x.sid)
                if law is None or law[0] != "translate":
                    continue
                body |= set(x.cells)
                v = law[1]
                if (k, x.sid) in getattr(model.fixed, "slides", ()):
                    dest |= _sweep(x.cells, (v[0] * 16, v[1] * 16))      # (a slide may go far: the whole way is asked)
                else:
                    dest |= _shift(x.cells, v)
        if not dest:
            return False
        ring = {(r + dr, c + dc) for (r, c) in dest for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        for o in tr.prev:
            if o.sid in model.active or o.sid >= 1000 or "hud" in o.tags:
                continue
            kd = kind(o)
            if kd not in hazard_kinds:
                continue
            if (ring if kd in touch_kinds else dest) & set(o.cells):
                return True
        return False
    except Exception:
        return False


def _step_is_safe(model, tr, me_sid, k, allow=None):
    if me_sid >= 90000:
        return True                     # a derived thing: its way is its source's business
    """LOOK BEFORE YOU STEP: the ground may have changed since the plan was
    made (a floor that falls away, a door that shut) — a key's translate
    of the mover onto cells that NOW wear a wall colour is not pressed."""
    v = model.key_selected.get(k) if isinstance(k, str) else None
    if v is None and isinstance(k, str):
        e = model.fixed.moving(k).get(me_sid)
        v = e[1] if e and e[0] == "translate" else None
    if v is None:
        return True
    f = getattr(model, "_frame_now", None)
    if f is None:
        return True
    # (the key moves the SELECTED thing: when another than the mover is
    # selected — a pusher on its way — that is what steps)
    if model.key_selected and hasattr(model, "selected_now"):
        s_ = model.selected_now(tr.prev)
        if s_ is not None and s_ != me_sid and not any(s_ in b and me_sid in b for b in model.bodies()):
            me_sid = s_
    me = next((x for x in tr.prev if x.sid == me_sid), None)
    if me is None:
        return True
    blocking = model.blocking()
    if not blocking:
        return True
    body = {me_sid}
    for b in model.bodies():
        if me_sid in b:
            body |= set(b)
    cells = set()
    for x in tr.prev:
        if x.sid in body:
            cells |= set(x.cells)
    dest = _sweep(cells, v)
    H, W = f.shape
    liv_ = model.livery() if hasattr(model, "livery") else None
    liv_cells_ = set()
    if liv_ and cells:
        mb_ = _bbox(cells)
        liv_cells_ = {(mb_[0] + dr_, mb_[1] + dc_) for (dr_, dc_, _c) in liv_}     # (the body's own outline is no wall)

    def edge_(q):
        # (a lone line of the wall's colour between two open cells is a drawing;
        # the body and its outline count as open)
        def open_(p):
            return 0 <= p[0] < H and 0 <= p[1] < W and (int(f[p[0], p[1]]) not in blocking or p in liv_cells_ or p in cells)
        return (open_((q[0] - 1, q[1])) and open_((q[0] + 1, q[1]))) or (open_((q[0], q[1] - 1)) and open_((q[0], q[1] + 1)))
    dest = {q for q in dest if not (0 <= q[0] < H and 0 <= q[1] < W and int(f[q[0], q[1]]) in blocking and edge_(q))}
    if allow:
        # (the target's own surround in the wall's colour is its outline: the way to it)
        tb_ = _bbox(allow)
        dest -= {(r, c) for r in range(tb_[0] - 1, tb_[2] + 2) for c in range(tb_[1] - 1, tb_[3] + 2)}
    dest -= liv_cells_
    # (AN EDGE IS NOT A WALL — as the plan had it: a one-cell line of the
    # wall's colour with open ground, or the body itself, on both sides is
    # a drawing or the body's own grey outline, not a wall)
    def open_(q):
        return 0 <= q[0] < H and 0 <= q[1] < W and (int(f[q[0], q[1]]) not in blocking or q in cells or q in liv_cells_)
    bad_ = [(r, c, int(f[r, c])) for r, c in dest if 0 <= r < H and 0 <= c < W and int(f[r, c]) in blocking
            and not ((open_((r - 1, c)) and open_((r + 1, c))) or (open_((r, c - 1)) and open_((r, c + 1))))]
    _step_is_safe.why = (v, sorted(bad_)[:6], "livery" if liv_ else None)
    return not bad_


def explore_leg(tr, model, keys, step, note, verbose, obs_fn, lv0, max_states=4000, free_first=False):
    """Walk to the farthest place the body can reach under what is known;
    stop at the first surprise (the body not where the plan put it — a
    slide, a scroll, a put-back).  Returns whether anything was pressed."""
    comps = tr.prev
    members = [x for x in comps if x.sid in model.active]
    if not members:
        return False
    me = max(members, key=lambda x: x.size)
    body0 = set()
    for m in members:
        body0 |= set(m.cells)
    c0 = centre(body0)
    solids = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2 and s.sid not in model.active}
    start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
    sel0 = [x_ for x_ in (model.selected_now(comps),) if x_ is not None]
    start["_sel"] = sel0[0] if sel0 else None
    f = getattr(model, "_frame_now", None)
    succ = make_successors(model, comps, solids, keys, me.sid, c0, frame=f)

    # (places are kept in the world's frame: the screen may have moved)
    cam = getattr(model, "camera", (0, 0))
    visited = getattr(model, "visited", None)
    if visited is None:
        visited = model.visited = set()
    visited.add((round(c0[0] + cam[0]), round(c0[1] + cam[1])))

    edir = getattr(model, "explore_dir", None)
    edir_keep = edir

    def far(st):
        """Farthest from everywhere the body has been — and, when the goal
        lay in a known direction, farthest that way."""
        cs = st.get(me.sid)
        if not cs:
            return 0.0
        c = centre(cs)
        w = (c[0] + cam[0], c[1] + cam[1])
        v = -min(abs(w[0] - r) + abs(w[1] - q) for (r, q) in visited)
        if edir:
            v -= 2.0 * ((w[0] - c0[0] - cam[0]) * edir[0] + (w[1] - c0[1] - cam[1]) * edir[1])
        return v
    best = (0, [])

    def progress_(path_):
        """How far the path's end lies along the explore direction."""
        if not edir or not path_:
            return 0
        st_ = start
        for k_ in path_:
            st_ = next((n_ for a_, n_ in succ(st_) if a_ == k_), None)
            if st_ is None or me.sid not in st_:
                return 0
        c_ = centre(st_[me.sid])
        return (c_[0] - c0[0]) * edir[0] + (c_[1] - c0[1]) * edir[1]
    if free_first:
        # (the way by free ground first: a removal is a change to the board,
        # made only when no free way leads on — on, when the goal's direction
        # is known, meaning that way)
        model._no_remove = True
        try:
            plan_general(start, succ, lambda st: False, heuristic=far, max_states=max_states, max_depth=40)
            best = (getattr(plan_general, "last", {}) or {}).get("best", (0, []))
            if verbose:
                print(f"   explore (free first): {(getattr(plan_general, 'last', {}) or {}).get('states')} states; best {best[0]:.1f} path {best[1]}; succ0 {[(a_, _bbox(n_[me.sid])[:2] if me.sid in n_ else None, round(far(n_), 1)) for a_, n_ in succ(start)]} whys {make_successors.whys}")
        finally:
            model._no_remove = False
        if edir and best[1] and progress_(best[1]) <= 0:
            best = (0, [])                      # (free, but not onward: a removal may lead on)
    if not best[1] or best[0] >= -2:
        plan_general(start, succ, lambda st: False, heuristic=far, max_states=max_states, max_depth=40)
        best2 = (getattr(plan_general, "last", {}) or {}).get("best", (0, []))
        if not edir or progress_(best2[1]) > progress_(best[1]) or not best[1]:
            best = best2
    if _ENV.get("EXPLORE_DEBUG") and (not best[1] or best[0] >= -2):
        for a_, n_ in succ(start):
            kids_ = [(b_, _bbox(m_[me.sid])[:2] if me.sid in m_ else None, round(far(m_), 1)) for b_, m_ in succ(n_)]
            print(f"      [explore] after {a_}: body {_bbox(n_[me.sid])[:2] if me.sid in n_ else None} far {round(far(n_), 1)} -> {kids_} whys {make_successors.whys}")
            if isinstance(a_, str):
                for b_, m_ in succ(n_):
                    if isinstance(b_, tuple):
                        gk_ = [(c_, _bbox(x_[me.sid])[:2] if me.sid in x_ else None, round(far(x_), 1)) for c_, x_ in succ(m_)]
                        print(f"         [explore] after {a_}, {b_}: body {_bbox(m_[me.sid])[:2] if me.sid in m_ else None} cleared {sorted(m_.get('_cleared', ()))[:6]} -> {gk_} whys {make_successors.whys}")
                        for c_, x_ in succ(m_):
                            if c_ == a_:
                                g4_ = [(d_, _bbox(y_[me.sid])[:2] if me.sid in y_ else None, round(far(y_), 1)) for d_, y_ in succ(x_)]
                                print(f"            [explore] after {a_}, {b_}, {c_}: body {_bbox(x_[me.sid])[:2] if me.sid in x_ else None} -> {g4_} whys {make_successors.whys}")
                                for w_ in make_successors.whys.values():
                                    if isinstance(w_, tuple) and w_[0] == "solid":
                                        o_ = next((z_ for z_ in comps if z_.sid == w_[2]), None)
                                        print(f"               solid S{w_[2]}: {(o_.body, o_.size, o_.bbox, sorted(o_.tags)) if o_ is not None else None} in state {w_[2] in x_} cells {sorted(x_.get(w_[2], ()))[:8]}")
    if edir and (not best[1] or best[0] >= -2 or progress_(best[1]) <= 0):
        # NOTHING LEADS ON: the way is not the straight one — a detour, the
        # farthest unvisited place whichever way (Tushar: an impossible
        # situation is where to look about)
        model.explore_dir = None
        try:
            edir = None
            plan_general(start, succ, lambda st: False, heuristic=far, max_states=max_states, max_depth=40)
            best3 = (getattr(plan_general, "last", {}) or {}).get("best", (0, []))
        finally:
            model.explore_dir = edir_keep
        if best3[1] and best3[0] < -2:
            best = best3
            note("explore: nothing leads on the way the goal lay — a detour to the farthest unvisited place")
    path = best[1]
    if verbose:
        print(f"   explore: {(getattr(plan_general, 'last', {}) or {}).get('states')} states; best {best[0]:.1f} path {path}; succ0 {[(a_, _bbox(n_[me.sid])[:2] if me.sid in n_ else None, round(far(n_), 1)) for a_, n_ in succ(start)]} whys {make_successors.whys} cam {cam} visited {sorted(visited)[:12]}")
    if not path or best[0] >= -2:
        note("explore: nowhere farther to go from here under what is known")
        return False
    note(f"explore: nothing to aim at — walking {len(path)} steps to the farthest place I can reach ({-best[0]:.0f} cells away)")
    st = start
    for i, k in enumerate(path):
        nxt = next((n_ for a_, n_ in succ(st) if a_ == k), None)
        if isinstance(k, str) and not _step_is_safe(model, tr, me.sid, k):
            note("explore: the ground ahead changed — stopping here")
            if verbose:
                print(f"   explore: unsafe {k}: {_step_is_safe.why}")
            return True
        note(f"explore: step {i + 1}/{len(path)} {k}")
        step(k)
        if obs_fn().levels_completed > lv0:
            return True
        now = {s_.sid: frozenset(s_.cells) for s_ in tr.prev}
        if me.sid in now:
            cam = getattr(model, "camera", (0, 0)); cn = centre(now[me.sid])
            visited.add((round(cn[0] + cam[0]), round(cn[1] + cam[1])))
        if nxt is None:
            return True
        if me.sid not in now or off_centre(now[me.sid], nxt[me.sid]):
            note(f"explore: step {i + 1} ({k}) put me elsewhere than expected — something to read (expected {_bbox(nxt[me.sid])[:2]}, am {(_bbox(now[me.sid])[:2] if me.sid in now else None)})")
            return True
        st = nxt
    return True


def _leg_length(tr, model, keys, me_sid, target, me_at=None):
    """How many actions the walk to `target` would take from here (a plan
    without acting), or None when there is no way under what is known.
    `me_at`: a thing whose place the body is imagined standing on instead
    (the leg from there — the order of a collection)."""
    comps = tr.prev
    me = next((x for x in comps if x.sid == me_sid), None)
    if me is None:
        return None
    tc = target.centre
    start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
    if me_at is not None:
        mb_, ab_ = me.bbox, me_at.bbox
        dv_ = (ab_[0] - mb_[0], ab_[1] - mb_[1])
        start[me_sid] = frozenset((r + dv_[0], c + dv_[1]) for r, c in me.cells)
        start.pop(me_at.sid, None)                  # (collected by then)
    start["_sel"] = model.selected_now(comps)
    solids = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2 and s.sid != target.sid
              and s.body not in model.flat}
    succ = make_successors(model, comps, solids, keys, me_sid, tc, frame=(getattr(model, '_frame_now', None) if getattr(model, '_frame_now', None) is not None else model._frame))

    def d_(st):
        cs = st.get(me_sid)
        if not cs:
            return 999.0
        c = centre(cs)
        return abs(c[0] - tc[0]) + abs(c[1] - tc[1])

    def goal(st):
        cs = st.get(me_sid)
        if not cs:
            return False
        b_ = _bbox(cs)
        return b_[0] - 0.5 <= tc[0] <= b_[2] + 0.5 and b_[1] - 0.5 <= tc[1] <= b_[3] + 0.5
    p = plan_general(start, succ, goal, heuristic=lambda st: d_(st) / 3.0, max_states=30000, max_depth=40)
    if p is None and _ENV.get("LEG_DEBUG"):
        s0_ = succ(start)
        print(f"   [leg] to S{target.sid}@{target.bbox} from {me.bbox}: none; first successors {[(a_, _bbox(n_[me_sid])) for a_, n_ in s0_]} whys {getattr(make_successors, 'whys', None)} drift {getattr(model.fixed, 'drift', None)} laws {[(k_, model.fixed.moving(k_).get(me_sid)) for k_ in keys]} last {getattr(plan_general, 'last', None)}")
    return None if p is None else len(p)


def _leg_any(tr, model, keys, me_sid, targets):
    """Is there a way from here to ANY of `targets` under what is known?
    One search with every target out of the solids (looser than any one
    leg: a 'no' here is a 'no' for each).  A search that finds no way is
    the expensive one — it exhausts the space — and three members asked
    one by one exhausted it three times (bp35 L2)."""
    comps = tr.prev
    me = next((x for x in comps if x.sid == me_sid), None)
    if me is None or not targets:
        return False
    tcs = [t.centre for t in targets]
    tsids = {t.sid for t in targets}
    start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
    start["_sel"] = model.selected_now(comps)
    solids = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2 and s.sid not in tsids
              and s.body not in model.flat}
    tc0 = min(tcs, key=lambda tc: abs(tc[0] - me.centre[0]) + abs(tc[1] - me.centre[1]))
    succ = make_successors(model, comps, solids, keys, me_sid, tc0, frame=(getattr(model, '_frame_now', None) if getattr(model, '_frame_now', None) is not None else model._frame))

    def d_(st):
        cs = st.get(me_sid)
        if not cs:
            return 999.0
        c = centre(cs)
        return min(abs(c[0] - tc[0]) + abs(c[1] - tc[1]) for tc in tcs)

    def goal(st):
        cs = st.get(me_sid)
        if not cs:
            return False
        b_ = _bbox(cs)
        return any(b_[0] - 0.5 <= tc[0] <= b_[2] + 0.5 and b_[1] - 0.5 <= tc[1] <= b_[3] + 0.5 for tc in tcs)
    p = plan_general(start, succ, goal, heuristic=lambda st: d_(st) / 3.0, max_states=30000, max_depth=40)
    return p is not None


def _plan_body(tr, model, keys, me_sid, goal_fn, dist_fn, body_cells=None, fixed_cells=None, max_states=30000, max_depth=60, solids=None, paint=None):
    """A plan for the body from a state: `body_cells` where the body is
    imagined standing (None: where it is); `fixed_cells`: {sid: cells}
    other things imagined elsewhere (a gate open).  Returns the key list
    or None."""
    comps = tr.prev
    start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
    if body_cells is not None:
        start[me_sid] = frozenset(body_cells)
    for sid_, cs_ in (fixed_cells or {}).items():
        if cs_ is None:
            start.pop(sid_, None)           # (a thing imagined gone)
        else:
            start[sid_] = frozenset(cs_)
    start["_sel"] = model.selected_now(comps)
    if solids is None:
        solids = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2 and s.body not in model.flat}
    me = next((x for x in comps if x.sid == me_sid), None)
    tc = centre(start[me_sid]) if me is not None else (32, 32)
    fr_ = getattr(model, '_frame_now', None) if getattr(model, '_frame_now', None) is not None else model._frame
    if paint:
        # (the ground as the plan imagines it: a gate drawn open, a pad the
        # body stood on drawn as ground)
        fr_ = np.array(fr_).copy()
        for q_, col_ in paint.items():
            if 0 <= q_[0] < fr_.shape[0] and 0 <= q_[1] < fr_.shape[1]:
                fr_[q_] = col_
    succ = make_successors(model, comps, solids, keys, me_sid, tc, frame=fr_)
    p_ = plan_general(start, succ, goal_fn, heuristic=lambda st: dist_fn(st) / 3.0, max_states=max_states, max_depth=max_depth)
    if p_ is None and _ENV.get("PLANB_DEBUG"):
        s0_ = succ(start)
        print(f"   [plan_body] none: start {_bbox(start[me_sid])} successors {[(a_, _bbox(n_[me_sid])) for a_, n_ in s0_ if isinstance(a_, str)]} whys {getattr(make_successors, 'whys', None)} last {getattr(plan_general, 'last', None)} solids {sorted(solids)[:8]}")
    return p_, start, succ


def twin_probe(tr, model, keys, step, get_obs, lv0, verbose, note):
    """A REWIND SUSPECTED, A TWIN NOT YET SEEN: one rewind on purpose, after
    the body has walked a record of two moves or more, then two moves that
    are not the record's first (the twin walks the record; the body must
    not stand on it) — does a twin appear?  Twice at most a level."""
    tw_ = getattr(model, "twin", None)
    if tw_ is None or tw_.confirmed or tw_.rewind_key is None or tw_.rewind_key not in keys or _ENV.get("NO_LEVER"):
        return False
    if getattr(tw_, "probes", 0) >= 2 or len(tw_.record) < 2:
        return False
    body_sids_ = [sid_ for sid_ in (model.active or ()) if sid_ < 1000]
    if not body_sids_:
        return False
    me_sid_ = max(body_sids_, key=lambda sid_: next((x_.size for x_ in tr.prev if x_.sid == sid_), 0))
    first_ = tw_.record[0]
    tw_.probes = getattr(tw_, "probes", 0) + 1
    note(f"{tw_.rewind_key} put the body back at the start before: pressing it once after a record of {len(tw_.record)} moves to see whether a twin walks it")
    step(tw_.rewind_key)
    if get_obs().levels_completed > lv0:
        return True
    # two moves that are not the record's first step
    laws_ = {}
    for k_ in keys:
        e_ = model.fixed.moving(k_).get(me_sid_)
        if e_ and e_[0] == "translate":
            laws_[k_] = tuple(e_[1])
    cands_ = [k_ for k_, v_ in laws_.items() if v_ != tuple(first_)] + [k_ for k_ in keys if k_ not in laws_ and k_ != tw_.rewind_key]
    n_ = 0
    for k_ in cands_:
        if n_ >= 2:
            break
        b0_ = next((x_.bbox for x_ in tr.prev if x_.sid == me_sid_), None)
        step(k_)
        if get_obs().levels_completed > lv0:
            return True
        b1_ = next((x_.bbox for x_ in tr.prev if x_.sid == me_sid_), None)
        if b1_ != b0_:
            n_ += 1
            k_again_ = k_
            while n_ < 2:
                b0_ = b1_
                step(k_again_)
                if get_obs().levels_completed > lv0:
                    return True
                b1_ = next((x_.bbox for x_ in tr.prev if x_.sid == me_sid_), None)
                if b1_ == b0_:
                    break
                n_ += 1
    if verbose:
        print(f"   twin probe: {tw_.describe()}")
    return False


def twin_protocol(tr, model, keys, step, get_obs, lv0, verbose, note, hyps):
    """THE RECORD AS A SECOND BODY (statements 7 and 8; g50t).  With a lever
    known (a pad whose standing-on opens a gate) and a twin known (a key
    that puts the body back at the start, after which a twin re-walks the
    record), the plan is: walk onto the pad (the record ends there), press
    the rewind, walk to the goal through the gate — waiting, where the
    gate is, until the twin has re-walked the record and holds the pad.
    Returns True when the level was won here."""
    tw_ = getattr(model, "twin", None); lv_ = getattr(model, "lever", None)
    if verbose:
        print(f"   twin protocol: twin {tw_.describe() if tw_ else None}; keys {keys}; active {model.active}; hyps {[h.note[:40] for h in (hyps or [])][:5]}")
    if tw_ is None or lv_ is None or not tw_.confirmed or tw_.rewind_key not in keys:
        return False
    body_sids_ = [sid_ for sid_ in (model.active or ()) if sid_ < 1000]
    if not body_sids_:
        return False
    me_sid_ = max(body_sids_, key=lambda sid_: next((x_.size for x_ in tr.prev if x_.sid == sid_), 0))
    me_ = next((x_ for x_ in tr.prev if x_.sid == me_sid_), None)
    if me_ is None:
        return False
    gates_ = lv_.gates()
    # the goal: the reading's target (the home), not the twin, the pad or the gate
    skip_cols_ = {tw_.twin_kind}
    body_cells_now_ = set(me_.cells) | set(tw_.start or ())
    twin_cells_ = set(next((x_.cells for x_ in tr.prev if x_.sid == tw_.twin), {}))
    tgts_ = []
    seen_t_ = set()
    for h_ in sorted(hyps or [], key=lambda h: (0 if getattr(h, "m_color", None) == me_.body else 1, -getattr(h, "prio", 0))):
        tc_ = getattr(h_, "t_cells", None)
        if not tc_ or h_.kind not in ("align", "cover", "inside") or getattr(h_, "t_color", None) in skip_cols_:
            continue
        if set(tc_) & (body_cells_now_ | twin_cells_):
            continue                    # (the body itself, its start, the twin: not a goal)
        if any(set(tc_) & (x_cells | y_cells) for _, x_cells, _, y_cells, _ in gates_):
            continue
        if frozenset(tc_) in seen_t_:
            continue
        seen_t_.add(frozenset(tc_))
        tgts_.append(h_)
    if not tgts_:
        return False
    tgt_ = tgts_[0]

    def dist_to_(cen_):
        def d_(st):
            cs = st.get(me_sid_)
            if not cs:
                return 999.0
            c = centre(cs)
            return abs(c[0] - cen_[0]) + abs(c[1] - cen_[1])
        return d_

    def at_(cen_):
        def g_(st):
            cs = st.get(me_sid_)
            if not cs:
                return False
            b_ = _bbox(cs)
            return b_[0] - 0.5 <= cen_[0] <= b_[2] + 0.5 and b_[1] - 0.5 <= cen_[1] <= b_[3] + 0.5
        return g_

    if not gates_:
        # THE LEVER IS NOT KNOWN YET: stand once on each thing the body can
        # stand on (a reachable place whose cells overlap a thing that is
        # not the twin and not the goal — a pad at the end of a wire, a
        # tile on the track) — what moves when the body stands there?
        # Three at most, nearest first.
        # (what the body may stand on is what the walk will tell: only the
        # kinds known to block are solid here — the pad is of no known kind)
        # (a kind refused once may be a compound whose other part is a pad:
        # a refusal costs one press, so nothing but the twin is solid here)
        soft_solids_ = {x_.sid for x_ in tr.prev if x_.sid == tw_.twin}
        p0_, st0_, succ0_ = _plan_body(tr, model, keys, me_sid_, lambda st: False, dist_to_(me_.centre), max_states=400, max_depth=20, solids=soft_solids_)
        seen_ = {st0_[me_sid_]: []}
        frontier_ = [(st0_, [])]
        n_ = 0
        while frontier_ and n_ < 400:
            st_, path_ = frontier_.pop(0)
            if len(path_) >= 20:
                continue
            try:
                nx_ = succ0_(st_)
            except Exception:
                nx_ = []
            for a_, st2_ in nx_:
                if not isinstance(a_, str) or a_ == tw_.rewind_key:
                    continue
                cs_ = st2_.get(me_sid_)
                if cs_ is None or cs_ in seen_:
                    continue
                seen_[cs_] = path_ + [a_]
                frontier_.append((st2_, path_ + [a_]))
                n_ += 1
        others_ = [x_ for x_ in tr.prev if x_.sid not in body_sids_ and x_.sid != tw_.twin and x_.sid < 1000 and "hud" not in x_.tags
                   and "terrain" not in x_.tags and kind(x_) != tw_.twin_kind and not (tgt_.t_cells and set(x_.cells) & set(tgt_.t_cells))]
        stands_ = {}
        for cs_, path_ in seen_.items():
            if not path_:
                continue
            for x_ in others_:
                ov_ = frozenset(cs_) & frozenset(x_.cells)
                if ov_ and (x_.sid, ov_) not in stands_:
                    stands_[(x_.sid, ov_)] = path_
        cands_ = sorted(stands_.items(), key=lambda kv: len(kv[1]))
        if verbose:
            print(f"   twin protocol: lever unknown; {len(seen_)} places in reach; stands {[(sid_, len(ov_), len(p_)) for (sid_, ov_), p_ in cands_][:8]}; target {tgt_.note[:50]}")
        tried_ = 0
        done_ = []
        for (sid_x, ov_), p_ in cands_:
            if tried_ >= 3 or any(ov_ & d_ for d_ in done_):
                continue
            done_.append(ov_)
            note(f"the twin is known ({tw_.rewind_key}); the lever is not: standing on S{sid_x} ({len(ov_)} cells, {len(p_)} presses) to see what moves")
            tried_ += 1
            for k_ in p_:
                step(k_)
                if get_obs().levels_completed > lv0:
                    return True
            if lv_.cands:
                # step off to confirm (one step back along the way)
                back_ = {"ACTION1": "ACTION2", "ACTION2": "ACTION1", "ACTION3": "ACTION4", "ACTION4": "ACTION3"}
                if p_ and p_[-1] in back_:
                    step(back_[p_[-1]])
                    if get_obs().levels_completed > lv0:
                        return True
            if verbose:
                print(f"   twin protocol: stood on S{sid_x}: lever cands {lv_.cands} levers {lv_.levers}")
            if lv_.levers:
                break
        gates_ = lv_.gates()
        if not gates_:
            return False
    X_, xc_, Y_, yc_closed_, yc_open_ = gates_[0]
    if yc_open_ is None:
        return False
    note(f"a lever: S{Y_} stands at {sorted(yc_open_)[0]} while something is on S{X_}; the twin ({tw_.rewind_key}, delay {tw_.delay}) will hold it")
    if verbose:
        print(f"   twin protocol: body S{me_sid_} pad S{X_} gate S{Y_} twin S{tw_.twin} rewind {tw_.rewind_key} delay {tw_.delay} record {len(tw_.record)} goal {tgt_.note[:60]}")
    bg_ = background_color(np.asarray(model._frame_now))
    # (the ground under the pad and under the gate is the mover's way — the
    # track it was seen on, when it walks a track; else the background)
    ground_ = bg_
    try:
        unc_ = getattr(model, "uncovered", None)
        if unc_:
            ground_ = max(unc_.items(), key=lambda kv: kv[1])[0]
    except Exception:
        ground_ = bg_
    gate_col_ = next((x_.body for x_ in tr.prev if x_.sid == Y_), None)
    paint_closed_ = {q_: ground_ for q_ in xc_}
    paint_closed_.update({q_: bg_ for q_ in yc_open_}); paint_closed_.update({q_: gate_col_ for q_ in yc_closed_})
    paint_open_ = {q_: ground_ for q_ in xc_}
    paint_open_.update({q_: ground_ for q_ in yc_closed_}); paint_open_.update({q_: gate_col_ for q_ in yc_open_})
    # 0. THE GOAL THAT HAS A WAY: of the readings' targets, the first one the
    # body can reach from the start with the gate open (a mark in the band
    # has no way; the home has)
    start_cells_ = frozenset(tw_.start)
    p2_ = None; st0_ = succ_ = None
    for h_ in tgts_:
        tcen_ = h_.target_centre() if hasattr(h_, "target_centre") else centre(h_.t_cells)
        # (the twin of the last rewind is gone at the next: one twin at a time)
        p2_, st0_, succ_ = _plan_body(tr, model, keys, me_sid_, at_(tcen_), dist_to_(tcen_), body_cells=start_cells_, fixed_cells={Y_: yc_open_, tw_.twin: None}, max_states=60000, max_depth=80, paint=paint_open_)
        if verbose:
            print(f"   twin protocol: goal {h_.note[:50]}: {'a way of %d with the gate open' % len(p2_) if p2_ else 'no way even with the gate open'}")
        if p2_:
            tgt_ = h_
            break
    if not p2_:
        note("no reading's target has a way with the gate open: the twin protocol stops")
        return False
    # the first step that enters the gate's closed cells
    j_ = None
    st_ = st0_
    for i_, k_ in enumerate(p2_):
        nx_ = next((n_ for a_, n_ in succ_(st_) if a_ == k_), None)
        if nx_ is None:
            break
        if nx_.get(me_sid_, frozenset()) & yc_closed_:
            j_ = i_; break
        st_ = nx_
    if j_ is None:
        note("the way to the goal does not pass the gate: walking it")
    # 1. onto the pad (the record ends there).  THE RECORD IS CLEANED FIRST:
    # the twin re-walks every step since the last rewind, and a record full
    # of the fiddle's wanderings walks the twin through the corridor where
    # the body must wait (and a messy record is a long wait); one rewind
    # now, then the shortest way to the pad, is the record the twin walks
    xcen_ = centre(xc_)
    me_now_ = next((x_ for x_ in tr.prev if x_.sid == me_sid_), None)
    if j_ is not None and tw_.record and not (set(me_now_.cells) & set(xc_)):
        note("the record is cleaned by one rewind before the walk to the pad: the twin will walk only that")
        step(tw_.rewind_key)
        if get_obs().levels_completed > lv0:
            return True
        me_now_ = next((x_ for x_ in tr.prev if x_.sid == me_sid_), None)
    if j_ is not None and not (set(me_now_.cells) & set(xc_)):
        p1_, _, _ = _plan_body(tr, model, keys, me_sid_, lambda st: bool(st.get(me_sid_, frozenset()) & xc_), dist_to_(xcen_), fixed_cells={Y_: yc_closed_, tw_.twin: None}, max_states=30000, max_depth=50, paint=paint_closed_)
        if verbose:
            print(f"   twin protocol: to the pad: {p1_}")
        if not p1_:
            note("no way onto the pad from here: the twin protocol stops")
            return False
        for k_ in p1_:
            step(k_)
            if get_obs().levels_completed > lv0:
                return True
    if j_ is not None:
        me_now_ = next((x_ for x_ in tr.prev if x_.sid == me_sid_), None)
        if me_now_ is None or not (set(me_now_.cells) & set(xc_)):
            note("the body did not reach the pad: the twin protocol stops")
            if verbose:
                print(f"   twin protocol: the body is at {me_now_.bbox if me_now_ else None}, not on the pad {sorted(xc_)[:2]}")
            return False
        L_ = len(tw_.record)
        # 2. the rewind: the twin will re-walk the record and hold the pad
        step(tw_.rewind_key)
        if get_obs().levels_completed > lv0:
            return True
        d_ = tw_.delay if tw_.delay is not None else 0
        ready_at_ = d_ + L_                            # the body's MOVES after the rewind before the pad is held
    else:
        ready_at_ = 0
    if verbose:
        print(f"   twin protocol: plan {p2_}; enters the gate at step {j_}; ready at {ready_at_} presses after the rewind")
    # 3. to the goal with the gate open, waiting where the gate is
    moves_ = 0
    back_ = {"ACTION1": "ACTION2", "ACTION2": "ACTION1", "ACTION3": "ACTION4", "ACTION4": "ACTION3"}

    def body_box_():
        b_ = next((x_ for x_ in tr.prev if x_.sid == me_sid_), None)
        return b_.bbox if b_ is not None else None

    def step_moved_(k_):
        b0_ = body_box_()
        step(k_)
        return body_box_() != b0_

    for i_, k_ in enumerate(p2_):
        if j_ is not None and i_ == j_:
            # wait until the twin holds the pad: a step and its undoing, as
            # many pairs as it takes — the twin walks on the body's moves,
            # and stays once its record ends
            n_wait_ = 0
            while moves_ < ready_at_ and n_wait_ < 12:
                w_ = back_.get(p2_[i_ - 1]) if i_ > 0 and p2_[i_ - 1] in back_ else None
                if w_ is None:
                    break
                note("waiting for the twin to reach the pad: a step back and forth")
                moves_ += 1 if step_moved_(w_) else 0
                moves_ += 1 if step_moved_(back_[w_]) else 0
                n_wait_ += 1
                if get_obs().levels_completed > lv0:
                    return True
        moves_ += 1 if step_moved_(k_) else 0
        if get_obs().levels_completed > lv0:
            return True
    return get_obs().levels_completed > lv0


def _walk_over(tr, model, keys, me_sid, target, step, verbose, max_steps=30, need=0, keep=()):
    """Plan and walk the mover until the target's centre lies in its box.
    `need`: presses the mover must still have on the clock when it gets
    there (the visit itself costs more).  With a clock read, the plan is
    made within the time left — through a refill when the direct way is
    too long — and only when no such way exists is the clock ignored (a
    death puts the mover back; the walk is then replanned from there)."""
    n = 0
    for attempt in range(4):
        t_w_ = time.time()
        comps = tr.prev
        me = next((x for x in comps if x.sid == me_sid), None)
        if me is None:
            return n
        tc = target.centre
        start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
        start["_sel"] = model.selected_now(comps)
        # (a target read as several pieces: none of them is in the way —
        # nor the small things touching it, the other colours of one tile)
        tcells = set(getattr(target, "cells", ()) or ())
        tb = target.bbox
        for s in comps:
            if s.sid != target.sid and s.sid < 1000 and s.size <= 8 and "hud" not in s.tags and "terrain" not in s.tags:
                b = s.bbox
                if b[0] <= tb[2] + 1 and tb[0] <= b[2] + 1 and b[1] <= tb[3] + 1 and tb[1] <= b[3] + 1:
                    tcells |= set(s.cells)
        solids = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2 and s.sid != target.sid
                  and s.body not in model.flat and not (tcells and set(s.cells) & tcells)}
        f = getattr(model, '_frame_now', None) if getattr(model, '_frame_now', None) is not None else model._frame      # (the frame as it is now, not before the last press)
        clock_left = model.clock_left(comps) if getattr(model, "clock_rate", 0) else None

        def d_(st):
            cs = st.get(me_sid)
            if not cs:
                return 999.0
            c = centre(cs)
            return abs(c[0] - tc[0]) + abs(c[1] - tc[1])

        # the mover is its whole BODY (the two halves of the player): the
        # target's centre may lie under either half
        body_ = {o for b in model.bodies() if me_sid in b for o in b} | {me_sid}

        def goal(st):
            cs = st.get(me_sid)
            if not cs:
                return False
            cs = set(cs)
            for o in body_:
                if o != me_sid and o in st:
                    cs |= set(st[o])
            b_ = _bbox(cs)
            if "_clock" in st and st["_clock"] < need * model.clock_rate:
                return False
            return b_[0] - 0.5 <= tc[0] <= b_[2] + 0.5 and b_[1] - 0.5 <= tc[1] <= b_[3] + 0.5
        modes = ["known", "prior", None] if clock_left is not None else [None]
        if keep and clock_left is not None:
            modes = [("known", keep), ("prior", keep)] + modes    # first without the refills kept for later
        p = None; succ = None; mode_used = None
        for mode in modes:
            keep_ = ()
            if isinstance(mode, tuple):
                mode, keep_ = mode
            succ = make_successors(model, comps, solids, keys, me_sid, tc, frame=f, clock=mode, keep=keep_)
            st0 = dict(start)
            if mode is not None:
                st0["_clock"] = model.clock_value(model.clock_thing(comps))
            if _ENV.get("WALK_DEBUG"):
                succ_inner = succ
                def succ(st, succ_inner=succ_inner):
                    res_ = succ_inner(st)
                    print(f"      [walk] me {_bbox(st[me_sid]) if me_sid in st else None} clock {st.get('_clock')} -> {[(a_, _bbox(n_[me_sid])[:2], n_.get('_clock')) for a_, n_ in res_]} whys {make_successors.whys}")
                    return res_
            p = plan_general(st0, succ, goal, heuristic=lambda st: d_(st) / 3.0, max_states=50000, max_depth=40)
            mode_used = mode
            if p:
                start = st0
                break
        if p is None and model.slide() is not None and not getattr(model, "_no_slide", False) and getattr(model, "gravity", None) is None:
            # A THEORY ON ONE SIGHT IS TESTED, NOT OBEYED: with the slide no
            # way is found — the way without it is walked, and the next
            # move says whether the body slides (the vote against, if not)
            model._no_slide = True
            try:
                succ = make_successors(model, comps, solids, keys, me_sid, tc, frame=f, clock=None)
                st0 = dict(start)
                p = plan_general(st0, succ, goal, heuristic=lambda st: d_(st) / 3.0, max_states=50000, max_depth=40)
                mode_used = None
                if p:
                    start = st0
                    note(f"the slide {model.slide()} is a theory on one sight: no way with it — walking the way without it, to see")
            finally:
                model._no_slide = False
        how_ = "" if clock_left is None else f" (clock {clock_left} presses{'' if mode_used is None else ', within it' if mode_used == 'known' else ', through an unproven refill'}{' — or not: none fits' if mode_used is None else ''})"
        note(f"walking S{me_sid} over S{target.sid} (colour {target.body}): plan {'none' if p is None else str(len(p)) + ' steps'}{how_}")
        if verbose:
            print(f"   walk: portals {getattr(model, 'portals', {})} spawn {getattr(model, 'spawn', {})} deaths {getattr(model, 'deaths', 0)} blocking {model.blocking()} ground_ok {sorted(model.ground_ok)} refused {dict(model.ground_refused)}")
            print(f"   walk to S{target.sid}@{target.bbox}: {p if p is None else len(p)} steps{how_} ({time.time() - t_w_:.1f}s); me S{me_sid}@{me.bbox} last={ {k_: (v_ if k_ != 'best' else v_[0]) for k_, v_ in (getattr(plan_general, 'last', {}) or {}).items()} } succ0={[a_ for a_, _ in succ(start)][:6]} whys={getattr(make_successors, 'whys', None)}")
            if p is None and _ENV.get("LEG_DEBUG"):
                s1_ = succ(start)
                print(f"      [walk] from {me.bbox}: {[(a_, _bbox(n_[me_sid])) for a_, n_ in s1_]}; then {[(a_, [(a2_, _bbox(n2_[me_sid])) for a2_, n2_ in succ(n_)]) for a_, n_ in s1_]}")
        if not p:
            return n
        st_ = start
        for k in p[:max_steps]:
            before_ = {x.sid: x.bbox for x in tr.prev}
            if not _step_is_safe(model, tr, me_sid, k):
                note("the ground ahead changed since the plan: replanning")
                if verbose:
                    print(f"   walk: step {k} unsafe: {getattr(_step_is_safe, 'why', None)}")
                break
            nxt_ = next((new_ for a_, new_ in succ(st_) if a_ == k), None)
            if nxt_ is None and _ENV.get("WALK_DEBUG"):
                print(f"      [walk] plan step {k} is not a successor of the tracked state: whys {make_successors.whys} clock {st_.get('_clock')} keys {[a_ for a_, _ in succ(st_)]}")
            step(k); n += 1
            if getattr(play_level, "won_now", False):
                return n                # the level is won: not a press more
            me2 = next((x for x in tr.prev if x.sid == me_sid), None)
            if me2 is not None and me2.bbox == before_.get(me_sid) and nxt_ is not None and me_sid in nxt_:
                # A REFUSED STEP IS REMEMBERED BY PLACE (as in a pursuit)
                bp_ = _bbox(nxt_[me_sid]); bw_ = before_[me_sid]
                if bp_[:2] != bw_[:2]:
                    model.refuse((kind(me2), bw_[0], bw_[1], (bp_[0] - bw_[0], bp_[1] - bw_[1])), size=(bw_[2] - bw_[0] + 1, bw_[3] - bw_[1] + 1))
            if _ENV.get("WALK_DEBUG"):
                print(f"      [walk step] {k}: me {me2.bbox if me2 else None} predicted {_bbox(nxt_[me_sid]) if nxt_ and me_sid in nxt_ else None} events {[str(e) for e in getattr(tr, 'last_events', [])][:6]}")
                if me2 is None:
                    fr_ = model._frame_now
                    for r_ in range(max(0, before_.get(me_sid, (0,0,0,0))[0] - 12), min(64, before_.get(me_sid, (0,0,0,0))[2] + 4)):
                        print("        ", "".join("%x" % int(fr_[r_, c_]) for c_ in range(0, 40)))
            if me2 is None or me2.bbox == before_.get(me_sid):
                break               # refused: replan (the refusal is remembered)
            if nxt_ is not None and me_sid in nxt_ and _bbox(nxt_[me_sid]) != me2.bbox:
                note(f"S{me_sid} is not where the plan put it (put back?): replanning")
                break
            if nxt_ is not None:
                st_ = nxt_
            else:
                # the tracked state drifted from the plan's: take the world as it is
                st_ = {s.sid: frozenset(s.cells) for s in tr.prev if "hud" not in s.tags}
                st_["_sel"] = model.selected_now(tr.prev)
                if "_clock" in start:
                    st_["_clock"] = model.clock_value(model.clock_thing(tr.prev)) if model.clock_thing(tr.prev) is not None else start["_clock"]
        me2 = next((x for x in tr.prev if x.sid == me_sid), None)
        if me2 is not None and goal({x.sid: frozenset(x.cells) for x in tr.prev if x.sid in body_}):
            return n
    return n


def _touching_clusters(things):
    """Things whose cells touch (a gap of at most one cell), as lists — a
    picture may be read as several pieces."""
    n = len(things); parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    halo = []
    for t in things:
        h = set()
        for r, c in t.cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    h.add((r + dr, c + dc))
        halo.append(h)
    for i in range(n):
        a = things[i].bbox
        for j in range(i + 1, n):
            b = things[j].bbox
            if a[0] <= b[2] + 1 and b[0] <= a[2] + 1 and a[1] <= b[3] + 1 and b[1] <= a[3] + 1:
                if halo[i] & set(things[j].cells):
                    parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(things[i])
    return list(groups.values())


def _grid(cells):
    """A filled rectangle's colour map as (h, w, {(r, c) from its corner:
    colour}), or None when the cells do not fill their box."""
    if not cells:
        return None
    r0, c0, r1, c1 = _bbox(cells)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if len(cells) != h * w:
        return None
    return h, w, {(r - r0, c - c0): col for (r, c), col in cells.items()}


def _rescale(grid, dims):
    """A colour map brought to dims by whole blocks, when the dims are
    integer multiples either way (a picture at another scale); else None."""
    h, w, m = grid; H, W = dims
    if (h, w) == (H, W):
        return m
    if h % H == 0 and w % W == 0:
        kr, kc = h // H, w // W
        out = {}
        for r in range(H):
            for c in range(W):
                block = {m[(r * kr + i, c * kc + j)] for i in range(kr) for j in range(kc)}
                if len(block) != 1:
                    return None
                out[(r, c)] = next(iter(block))
        return out
    if H % h == 0 and W % w == 0:
        kr, kc = H // h, W // w
        return {(r, c): m[(r // kr, c // kc)] for r in range(H) for c in range(W)}
    return None


def find_reference(tr, canvas, exclude=frozenset()):
    """A REFERENCE for a canvas: a filled rectangle of the canvas's dims (or
    an integer multiple of them) made of one or several touching things
    that never changed, away from the canvas — the one with most colours.
    Returns (map from the canvas's corner, its pieces) or None."""
    things = [s for s in tr.prev if s.sid < 1000 and s.sid != canvas.sid and s.sid not in exclude
              and "hud" not in s.tags and "bar" not in s.tags and "terrain" not in s.tags and s.size >= 4
              and not (set(s.cells) & set(canvas.cells))]
    best = None
    for grp in _touching_clusters(things) + [[t] for t in things]:
        cells = {}
        for t in grp:
            cells.update(t.cells)
        g = _grid(cells)
        if g is None:
            continue
        m = _rescale(g, canvas.dims)
        if m is None:
            continue
        key = (len(set(m.values())), len(cells))
        if best is None or key > best[0]:
            best = (key, m, grp)
    return (best[1], best[2]) if best else None


def _restore_body_laws(model, me_sid, keys):
    """THE BODY'S STEP IS WHAT IT DID MOST OFTEN: a key's translate of the
    body, seen twice or more, stands as its law though an odd reading
    (a turn read as a morph, two things confused) contradicted it."""
    for k in keys:
        alts = model.fixed.alt[k].get(me_sid)
        if not alts:
            continue
        best = max(((n_, e_) for e_, n_ in alts.items() if e_[0] == "translate" and (e_[1][0] or e_[1][1])), default=None)
        if best is None or best[0] < 2:
            continue
        cur = model.fixed.eff[k].get(me_sid)
        if cur != best[1] or me_sid in model.fixed.contradicted[k]:
            model.fixed.eff[k][me_sid] = best[1]
            model.fixed.support[k][me_sid] = max(model.fixed.support[k].get(me_sid, 0), best[0])
            model.fixed.contradicted[k].discard(me_sid)


def in_hands_wait(get_obs, lv0, step, keys, verbose, blocked_key=None):
    """The things left are in another carrier's hands: a few presses of a
    key that moves nothing here (time passes; the helper works).  Returns
    True if the level is still on."""
    k_ = blocked_key or next((k for k in keys if k not in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")), None)
    if k_ is None:
        return False
    if verbose:
        print(f"   carry: nothing of ours left to take: waiting on the helper ({k_} x3)")
    for _ in range(3):
        step(k_)
        if get_obs().levels_completed > lv0:
            return True
    return True


def _read_legend(things, buttons, button_sids, slot_sids):
    """THE LEGEND: the longest row (or column) of things of the buttons'
    kinds standing apart from the buttons and the slots — things, or parts
    of a backdrop (a terrain's other colours are things of their own; a
    part of several pieces is read piece by piece).  None when fewer than
    two stand in line."""
    from agent.core.progress import _components as _comps
    from agent.core.entities import Sprite as _Sp
    cand = []
    for s_ in things:
        if s_.body not in buttons or "hud" in s_.tags or "hidden" in s_.tags or s_.sid in button_sids or s_.sid in slot_sids:
            continue
        if s_.sid >= 1000:
            if s_.sid // 1000 in button_sids or s_.sid // 1000 in slot_sids:
                continue
            pieces = _comps(s_.cells.keys())
            for i_, pc in enumerate(sorted(pieces, key=lambda c_: min(c_))):
                cand.append(_Sp(sid=s_.sid + i_ * 1000000, cells={p: s_.cells[p] for p in pc}, parts=1, tags=set(s_.tags)))
        else:
            cand.append(s_)
    best = None
    for axis in (0, 1):
        groups = defaultdict(list)
        for s_ in cand:
            groups[s_.bbox[axis]].append(s_)
        for _, grp in groups.items():
            if len(grp) >= 2 and (best is None or len(grp) > len(best)):
                best = sorted(grp, key=lambda s_: s_.bbox[1 - axis])
    return best


def merge_up(tr, model, step, get_obs, lv0, verbose, progress, note, clickable, look_again, pursue_again):
    """The merge ladder as a plan: see the statements in the body."""
    did_any = False
    # TWO LIKE THINGS BROUGHT TOGETHER BECOME THE NEXT KIND (the record's
    # merges): the goal is to merge up — every kind that has merged, and
    # every kind made by a merge (a product merges too, until a click
    # between two of them does nothing: refuted for that kind).  Two things
    # of the kind, the nearest pair first: within twice the reach, the
    # click goes between them; else one is called a reach toward the other.
    # A click that called nothing where a call was expected shortens the
    # reach.  At most 24 clicks (hand-chosen); then the readings again.
    if progress.merges and clickable and not getattr(progress, "_merged", False):
        progress._merged = True
        try:
            pf_m_ = detect_playfield(np.array(get_obs().frame)[-1])
        except Exception:
            pf_m_ = None
        reach_ = max(2, int(progress.reach))
        reach_known_ = reach_ >= 4      # (a reach seen from a far call; else it is probed: doubled while calls answer)
        bound_ = getattr(progress, "reach_bound", None)      # (a call that went unanswered bounds the reach for good)
        if bound_ is not None:
            reach_ = min(reach_, bound_)
        kinds_merge_ = set(progress.merges) | set(progress.merges.values())
        refuted_m_ = set()
        cur_pair_ = None                # (the pair is kept until it merges or refuses)
        note(f"two like things brought together become the next kind ({dict(progress.merges)}): merging up, reach {reach_}")
        did_any = True
        for n_click_ in range(24):
            # the lowest kind (the one that merges into others first) with a pair standing
            kinds_merge_ = set(progress.merges) | set(progress.merges.values())      # (a product merges too)
            pairs_ = {}
            for kd_ in kinds_merge_ - refuted_m_:
                # (things of the kind on the field — not a picture of one in the
                # band above it: the merged things' own backdrop is the field's)
                th_ = [s_ for s_ in tr.prev if s_.sid < 1000 and s_.body == kd_ and "hud" not in s_.tags and "terrain" not in s_.tags and s_.size <= 64
                       and (pf_m_ is None or (pf_m_[0] <= s_.bbox[0] and s_.bbox[2] < pf_m_[1] and pf_m_[2] <= s_.bbox[1] and s_.bbox[3] < pf_m_[3]))]
                if len(th_) >= 2:
                    pairs_[kd_] = th_
            if not pairs_:
                if verbose:
                    print(f"   merge: no pair left among kinds {sorted(kinds_merge_ - refuted_m_)}: things {[(s_.sid, s_.body, s_.size, s_.bbox, sorted(s_.tags)) for s_ in tr.prev if s_.size <= 100]}")
                # THE LADDER'S TOP IS A THING TO TAKE SOMEWHERE: a kind made by a
                # merge is asked once how it answers a call — two calls of a
                # reach along the open way (the planner wants two to state the
                # law) — then the readings say where it goes
                top_ = [s_ for s_ in tr.prev if s_.sid < 1000 and s_.body in set(progress.merges.values()) - set(progress.merges)
                        and "hud" not in s_.tags and "terrain" not in s_.tags and s_.size <= 64
                        and (pf_m_ is None or (pf_m_[0] <= s_.bbox[0] and s_.bbox[2] < pf_m_[1] and pf_m_[2] <= s_.bbox[1] and s_.bbox[3] < pf_m_[3]))]
                if len(top_) == 1 and model.relative_law(top_[0].body) is None:
                    t_ = top_[0]
                    last_pt_ = None
                    for _ in range(2):
                        # (the thing by its kind and place now — the tracker may have renamed it)
                        t_ = next((x_ for x_ in tr.prev if x_.sid == t_.sid), None) or \
                            next((x_ for x_ in tr.prev if x_.sid < 1000 and x_.body == top_[0].body and x_.size == top_[0].size
                                  and (pf_m_ is None or pf_m_[0] <= x_.bbox[0] < pf_m_[1])), None)
                        if t_ is None:
                            break
                        tc_ = t_.centre
                        vr_, vc_ = 32 - tc_[0], 32 - tc_[1]
                        nn_ = max(abs(vr_), abs(vc_), 1)
                        # (a call well within the reach: the law wanted, not the reach's edge)
                        r_ = max(2, int(reach_) // 2 + 1)
                        pt_ = (int(round(tc_[0] + vr_ * r_ / nn_)), int(round(tc_[1] + vc_ * r_ / nn_)))
                        if pt_ == last_pt_:
                            break
                        last_pt_ = pt_
                        if verbose:
                            print(f"   merge: the ladder's top S{t_.sid} kind {t_.body} is called once to learn its law: click {pt_}")
                        step(("click", pt_))
                        if get_obs().levels_completed > lv0:
                            return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
                    if verbose:
                        print(f"   merge: the top's law now {model.relative_law(t_.body)} from {model.rel.get(t_.body)}")
                break
            # (a kind that merges into a present kind goes first: the ladder's bottom)
            def depth_(kd_):
                d_ = 0; k_ = kd_
                while k_ in progress.merges and d_ < 8:
                    k_ = progress.merges[k_]; d_ += 1
                return -d_
            kd_ = min(pairs_, key=depth_)
            th_ = pairs_[kd_]
            best_ = None
            if cur_pair_ is not None:
                A0_ = next((t_ for t_ in th_ if t_.sid == cur_pair_[0]), None); B0_ = next((t_ for t_ in th_ if t_.sid == cur_pair_[1]), None)
                if A0_ is not None and B0_ is not None:
                    a_, b_ = A0_.centre, B0_.centre
                    best_ = (max(abs(a_[0] - b_[0]), abs(a_[1] - b_[1])), A0_, B0_)
            if best_ is None:
                for i_ in range(len(th_)):
                    for j_ in range(i_ + 1, len(th_)):
                        a_, b_ = th_[i_].centre, th_[j_].centre
                        d_ = max(abs(a_[0] - b_[0]), abs(a_[1] - b_[1]))
                        if best_ is None or d_ < best_[0]:
                            best_ = (d_, th_[i_], th_[j_])
            d_, A_, B_ = best_
            cur_pair_ = (A_.sid, B_.sid)
            a_, b_ = A_.centre, B_.centre
            if d_ <= 2 * reach_:
                pt_ = (int(round((a_[0] + b_[0]) / 2)), int(round((a_[1] + b_[1]) / 2)))
                how_ = "between"
            else:
                # (the reach unknown: try twice the last answered call, up to the pair's gap)
                try_ = reach_ if reach_known_ else min(2 * reach_, d_ - 1)
                vr_, vc_ = b_[0] - a_[0], b_[1] - a_[1]
                sc_ = try_ / max(abs(vr_), abs(vc_))
                pt_ = (int(round(a_[0] + vr_ * sc_)), int(round(a_[1] + vc_ * sc_)))
                how_ = "toward"
            pt_ = (max(0, min(63, pt_[0])), max(0, min(63, pt_[1])))
            n_ev_ = len(progress.events); n_m_ = progress.merge_events
            if verbose:
                print(f"   merge: kind {kd_} pair S{A_.sid}@{A_.bbox[:2]} S{B_.sid}@{B_.bbox[:2]} d {d_} reach {reach_}: click {how_} at {pt_}")
            step(("click", pt_))
            if get_obs().levels_completed > lv0:
                return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
            moved_ = any(ev["score_kind"] in kinds_merge_ for ev in progress.events[n_ev_:])
            if verbose:
                print(f"   merge: after the click: moved {moved_} merges {progress.merge_events - n_m_} reach seen {progress.reach} events {[(ev['score_sid'], ev['score_kind'], ev['how']) for ev in progress.events[n_ev_:]]}")
            if progress.merge_events > n_m_:
                reach_ = max(reach_, int(progress.reach)) if bound_ is None else min(bound_, max(reach_, int(progress.reach)))
                cur_pair_ = None
                continue
            if not moved_:
                if how_ == "between" and d_ > 2:
                    reach_ = max(2, int(d_) // 2 - 1); reach_known_ = True
                    bound_ = reach_; progress.reach_bound = bound_
                    if verbose:
                        print(f"   merge: nothing came: reach is less than {d_ // 2}: now {reach_}")
                elif how_ == "between":
                    refuted_m_.add(kd_); cur_pair_ = None
                    note(f"a click between two things of kind {kd_} did nothing: they do not merge")
                elif not reach_known_:
                    reach_known_ = True        # (the doubled call went unanswered: the last answered reach stands)
                    bound_ = max(2, int(try_) - 1); progress.reach_bound = bound_; reach_ = min(reach_, bound_)
                    if verbose:
                        print(f"   merge: a call at {try_} went unanswered: reach stays {reach_}")
                else:
                    reach_ = max(2, reach_ - 2)
                    if reach_ <= 2:
                        refuted_m_.add(kd_); cur_pair_ = None
            else:
                reach_ = max(reach_, int(progress.reach)) if bound_ is None else min(bound_, max(reach_, int(progress.reach)))
                if how_ == "toward" and not reach_known_ and try_ >= d_ - 1:
                    reach_known_ = True        # (the pair is within one call now; what the reach is beyond it is not asked)
        look_again()
        res_ = pursue_again()
        if res_:
            return res_
    return None


def progress_protocol(tr, model, keys, step, get_obs, lv0, verbose, progress, note, clickable, look_again, pursue_again):
    """Do the cause again for every like thing, watching the score.  For
    the most-moved score's goal: each thing of the cause's kind not yet
    done, nearest first — walked onto (arrive, touch) or clicked (click);
    after each, the record says whether the score moved again.  Two
    targets that moved nothing refute the goal.  Then the frame is read
    again and the readings pursued (a goal completed may have opened one)."""
    if _ENV.get("NO_PROGRESS"):
        return None
    did_any = False
    # A SELECTION IS A MODE: a click that leaves a mark changes what every
    # other click does.  While a mark stands, the other kinds are asked once
    # (one thing each) — and a slot that takes the selected colour is a FILL.
    # THE LEGEND GIVES THE ORDER: when the selectable kinds stand elsewhere in
    # a row (or column) of as many things as there are slots, the slots are
    # filled in that order; then the keys are pressed once each (the submit).
    # (sb26: buttons below, four slots in a box, the order drawn above)
    buttons_ = progress.button_kinds() if clickable else []
    # (the buttons' line — the tray — holds buttons not yet clicked: their
    # kinds are buttons too)
    rows_b_ = getattr(progress, "button_rows", None) or Counter()
    if rows_b_ and buttons_:
        tray_b_ = max(rows_b_, key=rows_b_.get)
        for s_ in tr.prev:
            if s_.sid < 1000 and abs(s_.bbox[0] - tray_b_) <= 1 and 2 <= s_.size <= 64 and "hud" not in s_.tags and s_.body not in buttons_:
                buttons_.append(s_.body)
    if verbose:
        print(f"   progress entry: buttons {buttons_} fills {progress.fills()[:3]} mode_asked {getattr(progress, '_mode_asked', False)} filled {getattr(progress, '_filled', False)} events {len(progress.events)}")
    if buttons_ and not progress.fills() and not getattr(progress, "_mode_asked", False):
        progress._mode_asked = True
        # (one thing per kind and SIZE — a kind's dots and its bar are asked
        # each; the smaller first, a slot being small)
        others_ = {}
        for s_ in sorted(tr.prev, key=lambda s_: s_.size):
            if s_.sid < 1000 and "hud" not in s_.tags and "terrain" not in s_.tags and s_.body not in buttons_ and 2 <= s_.size <= 64 and (s_.body, s_.size) not in others_:
                others_[(s_.body, s_.size)] = s_
        b0_ = next((s_ for s_ in tr.prev if s_.sid < 1000 and s_.body == buttons_[0] and s_.size <= 64), None)
        if b0_ is not None and others_:
            note(f"a selection is a mode: asking {len(others_)} kinds with kind {buttons_[0]} selected")
            step(("click", click_point(b0_)))
            for kd_, t_ in list(others_.items())[:8]:
                step(("click", click_point(t_)))
                did_any = True
                if get_obs().levels_completed > lv0:
                    return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
    fills_ = progress.fills()
    if fills_ and clickable and not getattr(progress, "_filled", False):
        progress._filled = True
        slot_kind_ = fills_[0][0]
        slot_size_ = next((f_[3] for f_ in fills_ if f_[3]), None)
        # (the slots are the LIKE things: of the slot's kind and size — not the
        # bar of that colour, not the mark a used button keeps; a slot already
        # filled is hidden under what filled it: still a slot)
        slots_ = sorted([s_ for s_ in tr.prev if s_.sid < 1000 and s_.body == slot_kind_ and "hud" not in s_.tags
                         and (slot_size_ is None or s_.size == slot_size_)], key=lambda s_: (s_.bbox[0] // 4, s_.bbox[1]))
        slot_sids_ = {s_.sid for s_ in slots_}
        bsids_ = set(getattr(progress, "button_sids", ()))
        legend_ = _read_legend(tr.prev, buttons_, bsids_, slot_sids_)
        order_ = [s_.body for s_ in legend_] if legend_ else [fills_[0][1]] * len(slots_)
        if verbose:
            print(f"   fill: slots {[(s_.sid, s_.bbox, sorted(s_.tags)) for s_ in slots_]} buttons {buttons_} legend {[(s_.sid, s_.body, s_.bbox[:2]) for s_ in (legend_ or [])]}")
        note(f"fill: {len(slots_)} slots of kind {slot_kind_}; the legend reads {order_} ({'the legend' if legend_ else 'one colour'})")

        lsids_ = {l_.sid for l_ in legend_} if legend_ else set()
        # (the tray: the line where the buttons stood when they were clicked)
        rows_ = getattr(progress, "button_rows", None) or Counter()
        tray_ = max(rows_, key=rows_.get) if rows_ else None
        # THE BOXES: the big hollow things holding slots; a slot's box is the
        # smallest that holds it.  A CALL TILE: a hollow tile in the tray of a
        # box's colour (the box's run, done where the tile stands — learned
        # when the run passed such a tile and stopped next inside its box)
        boxes_ = []
        for s_ in tr.prev:
            if s_.sid >= 1000 or s_.size <= 24 or "hud" in s_.tags or s_.sid in lsids_ or s_.sid in slot_sids_:
                continue
            cav_s_ = s_.cavity()
            if cav_s_ and any(set(x_.cells) <= cav_s_ for x_ in slots_):
                boxes_.append(s_)
        box_colours_ = {b_.body for b_ in boxes_}

        def box_of_(sl_):
            bb_ = sl_.bbox
            hold_ = [b_ for b_ in boxes_ if b_.bbox[0] <= bb_[0] and b_.bbox[1] <= bb_[1] and b_.bbox[2] >= bb_[2] and b_.bbox[3] >= bb_[3]]
            return min(hold_, key=lambda b_: b_.size).sid if hold_ else None

        placed_ = []        # (entry index, slot, tile sid, is_call) — a call and the first slot of its box share the index

        def over_(t_):
            # 0: in the tray (free); 1: over a slot the run has not matched
            # (mine to move); 2: over a slot the run matched (the supply is spent)
            if tray_ is not None and abs(t_.bbox[0] - tray_) <= 1:
                return 0
            # (the slots' places are known from the placing: a tile over a
            # placed slot the run has matched is spent)
            for i_, sl_, _, _ in placed_:
                b_ = sl_.bbox
                if i_ < kk_now_[0] and t_.bbox[0] <= b_[0] and t_.bbox[1] <= b_[1] and t_.bbox[2] >= b_[2] and t_.bbox[3] >= b_[3]:
                    return 2
            return 1
        kk_now_ = [0]

        def tiles_of_(bk_, not_sids_=(), hollow=None):
            cands_b_ = [s_ for s_ in tr.prev if s_.sid < 1000 and s_.body == bk_ and s_.size <= 64 and s_.sid not in lsids_
                        and s_.sid not in slot_sids_ and s_.sid not in not_sids_ and "hidden" not in s_.tags
                        and (hollow is None or bool(s_.cavity()) == hollow)]
            cands_b_.sort(key=lambda t_: (over_(t_), 0 if t_.sid in bsids_ else 1))
            return cands_b_

        def tile_of_(bk_, not_sids_=()):
            # (a thing of the wanted kind to select — free or unmatched; a
            # tile the run already matched is not taken: that is a SHORTAGE)
            c_ = [t_ for t_ in tiles_of_(bk_, not_sids_) if over_(t_) < 2]
            return c_[0] if c_ else None

        def call_tile_(col_=None):
            # (a free hollow tile of a box's colour)
            for bc_ in ([col_] if col_ is not None else sorted(box_colours_)):
                c_ = [t_ for t_ in tiles_of_(bc_, hollow=True) if over_(t_) == 0]
                if c_:
                    return c_[0]
            return None

        def slot_at_(s_):
            # (the slot by its place: the thing of the slot's kind there, hidden
            # under a filling or not; a blinked hole is its own place)
            if s_.sid < 0:
                return s_
            return next((x_ for x_ in tr.prev if x_.sid == s_.sid), s_)

        # THE RUN TELLS HOW FAR: each key is pressed once; a key whose press
        # painted the legend's entries in order and blinked somewhere at the
        # end has run the slots and stopped — the slot it blinked at takes
        # the legend's next entry, and the key is pressed again.  A TILE IS
        # USED ONCE: when the entry's kind has no tile left to give, the
        # legend repeats a stretch, and a repeated stretch is a box's run —
        # the earlier stretch's first tile gives way to a call tile of a
        # box's colour (the box takes the stretch), or, when that stretch
        # already runs in a box, the call tile is placed where the run
        # stopped.  At most twice as many runs as slots, plus two (hand-chosen).
        run_key_ = None
        lsid_order_ = [l_.sid for l_ in legend_] if legend_ else []

        def reach_(tell_):
            # how many legend entries the run painted, in the legend's order
            if not tell_ or not lsid_order_:
                return 0, None
            pk_ = [x_ for x_ in tell_["painted"] if x_ in lsid_order_]
            k_ = 0
            while k_ < len(lsid_order_) and k_ < len(pk_) and pk_[k_] == lsid_order_[k_]:
                k_ += 1
            # the slot blinked at: the slot inside a blinking ring, else within two cells of one
            stop_ = None
            for comp_ in tell_["blink"]:
                r0_ = min(r for r, _ in comp_); c0_ = min(c for _, c in comp_); r1_ = max(r for r, _ in comp_); c1_ = max(c for _, c in comp_)
                for sl_ in slots_:
                    b_ = sl_.bbox
                    if b_[0] >= r0_ and b_[1] >= c0_ and b_[2] <= r1_ and b_[3] <= c1_:
                        stop_ = sl_; break
                if stop_ is not None:
                    break
            if stop_ is None:
                # (a blink that rings no slot I know, away from the legend, rings
                # a slot all the same — the run stops at slots: its hole is one)
                from agent.core.entities import Sprite as _Sp
                lcells_ = set()
                for l_ in (legend_ or []):
                    lcells_ |= set(l_.cells)
                for comp_ in tell_["blink"]:
                    if set(comp_) & lcells_:
                        continue
                    cav_ = _Sp(sid=-1, cells={p_: 0 for p_ in comp_}).cavity()
                    if 1 <= len(cav_) <= 64:
                        stop_ = _Sp(sid=-1, cells={p_: 0 for p_ in cav_}, tags={"blinked"}); break
            return k_, stop_

        for k_ in keys:
            n_tell_ = len(progress.tellings)
            step(k_)
            did_any = True
            if get_obs().levels_completed > lv0:
                return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
            if len(progress.tellings) > n_tell_:
                kk_, st_ = reach_(progress.tellings[-1][1])
                if verbose:
                    print(f"   run: {k_} reached {kk_} of {len(lsid_order_)}; stopped at {st_.sid if st_ is not None else None}")
                if (kk_ > 0 or st_ is not None) and run_key_ is None:
                    run_key_ = k_

        def put_(tile_, where_):
            step(("click", click_point(tile_)))
            step(("click", click_point(slot_at_(where_))))

        if run_key_ is not None and legend_:
            note(f"the run ({run_key_}) paints the legend as far as the slots agree and blinks where it stops: that slot takes the next entry")
            main_box_ = None
            for round_ in range(2 * len(slots_) + 2):
                kk_, st_ = reach_(progress.last_telling(run_key_))
                kk_now_[0] = kk_
                if st_ is None or kk_ >= len(lsid_order_):
                    break
                if main_box_ is None:
                    main_box_ = box_of_(st_)
                want_ = legend_[kk_].body
                b_ = tile_of_(want_, not_sids_={st_.sid})
                if b_ is not None:
                    if verbose:
                        print(f"   run: fixing slot S{st_.sid} at {st_.bbox[:2]} with kind {want_} (entry {kk_ + 1}) using S{b_.sid} at {b_.bbox[:2]}; candidates {[(t_.sid, t_.bbox[:2], over_(t_)) for t_ in tiles_of_(want_)]}; placed {[(i_, sl_.sid, t_, c_) for i_, sl_, t_, c_ in placed_]} kk {kk_now_[0]}")
                    put_(b_, st_)
                    placed_.append((kk_, st_, b_.sid, False))
                else:
                    # THE SHORTAGE: the kind's tiles are all matched already
                    pj_ = next((pl_ for pl_ in sorted(placed_, key=lambda x_: x_[0]) if pl_[0] < kk_ and legend_[pl_[0]].body == want_ and not pl_[3]), None)
                    if pj_ is None:
                        if verbose:
                            print(f"   run: no tile of kind {want_} left and none placed before: giving up; candidates {[(t_.sid, t_.bbox[:2], over_(t_)) for t_ in tiles_of_(want_)]}")
                        break
                    j_, sl_j_, _, _ = pj_
                    bj_ = box_of_(sl_j_)
                    if bj_ is not None and bj_ != main_box_:
                        # the earlier stretch runs in a box: call that box here
                        c_ = call_tile_(next((b2_.body for b2_ in boxes_ if b2_.sid == bj_), None))
                        if c_ is None:
                            if verbose:
                                print(f"   run: the stretch from entry {j_ + 1} runs in box S{bj_} but no call tile of its colour is free: giving up")
                            break
                        note(f"entry {kk_ + 1} ({want_}) has no tile left: the stretch from entry {j_ + 1} runs in a box — its call tile goes where the run stopped")
                        if verbose:
                            print(f"   run: calling box S{bj_} (tile S{c_.sid}) at slot S{st_.sid} for entry {kk_ + 1}")
                        put_(c_, st_)
                        placed_.append((kk_, st_, c_.sid, True))
                    else:
                        # the earlier stretch is the main run's: it was a box's — a
                        # call tile takes its first slot, the tile goes free
                        c_ = call_tile_()
                        if c_ is None:
                            if verbose:
                                print(f"   run: no call tile free to take entry {j_ + 1}'s place: giving up")
                            break
                        note(f"entry {kk_ + 1} ({want_}) has no tile left: the legend repeats from entry {j_ + 1} — that stretch is a box's run, called by a tile of the box's colour")
                        if verbose:
                            print(f"   run: a call tile S{c_.sid} (kind {c_.body}) takes slot S{sl_j_.sid} of entry {j_ + 1}; entries from there are placed again")
                        put_(c_, sl_j_)
                        placed_ = [pl_ for pl_ in placed_ if pl_[0] < j_]
                        placed_.append((j_, sl_j_, c_.sid, True))
                if get_obs().levels_completed > lv0:
                    return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
                n_tell_ = len(progress.tellings)
                step(run_key_)
                if get_obs().levels_completed > lv0:
                    return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
                if len(progress.tellings) <= n_tell_:
                    break
        # THE LEGEND GIVES THE ORDER when no run tells and it has one entry
        # per slot: the slots are filled in that order, line by line, and the
        # keys pressed once more
        if run_key_ is None and legend_ and len(legend_) == len(slots_):
            for i_, slot_ in enumerate(slots_):
                b_ = tile_of_(order_[i_])
                if b_ is None:
                    continue
                step(("click", click_point(b_)))
                step(("click", click_point(slot_at_(slot_))))
                did_any = True
                if get_obs().levels_completed > lv0:
                    return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
            for k_ in keys:
                step(k_)
                if get_obs().levels_completed > lv0:
                    return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
    res_m_ = merge_up(tr, model, step, get_obs, lv0, verbose, progress, note, clickable, look_again, pursue_again)
    if res_m_:
        return res_m_
    goals = progress.hypotheses(tr.prev, model)
    if verbose:
        print(f"   progress goals: {[g.note()[:70] for g in goals[:4]]}; record {progress.describe()[-4:]}")
    if not goals and not did_any:
        return None
    for g in goals[:3]:
        if g.cause == "press" and getattr(g, "with_selection", None) is not None:
            # A KEY ANSWERED WITH A SELECTION (sb26: click a button, confirm):
            # the buttons are the kinds whose click leaves a mark; each is
            # tried with the key and the score's sign read — a sign the score
            # has shown rarely is the answer (four wrong tries to one right:
            # the rare sign is the right one); the answer found, the next
            # round asks the buttons again.  At most four rounds of the
            # buttons (hand-chosen).
            k_ = g.keys[0]
            buttons_ = progress.button_kinds()
            if not buttons_ or not clickable:
                continue
            # (the score's sign when the key was pressed with NO selection is the
            # penalty — what a wrong answer or no answer costs; the good sign is
            # the other one.  With no such baseline, growth is taken as good)
            base_ = Counter(1 if ev["delta"] > 0 else (-1 if ev["delta"] < 0 else 0) for ev in progress.events
                            if ev["score_sid"] == g.score_sid and ev["cause"][0] == "press" and (len(ev["cause"]) < 4 or ev["cause"][3] is None))
            base_sign_ = max((sg for sg in base_ if sg != 0), key=lambda sg: base_[sg], default=None)
            good_ = -base_sign_ if base_sign_ is not None else 1
            note(f"{g.note()[:50]}: answering with the buttons {buttons_}; the score's good sign is {good_}")
            for round_ in range(4):
                found_ = False
                for bk_ in buttons_:
                    tb_ = [s_ for s_ in tr.prev if s_.sid < 1000 and s_.body == bk_ and "hud" not in s_.tags and s_.size <= 64]
                    if not tb_:
                        continue
                    n_all0 = len(progress.events)
                    step(("click", click_point(tb_[0])))
                    step(k_)
                    did_any = True
                    if get_obs().levels_completed > lv0:
                        return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
                    new_ = [ev for ev in progress.events[n_all0:] if ev["score_sid"] == g.score_sid]
                    sg_ = sum(1 if ev["delta"] > 0 else (-1 if ev["delta"] < 0 else 0) for ev in new_)
                    if verbose:
                        print(f"   progress: answered {bk_} with {k_}: score sign {sg_} (good {good_})")
                    if new_ and (sg_ > 0) == (good_ > 0) and sg_ != 0:
                        found_ = True
                        break
                if not found_:
                    break
            continue
        if g.cause == "press" and g.kind is None:
            # (a press that counted nowhere in particular: pressed again where it
            # counted, while the score keeps moving — at most eight, hand-chosen)
            k_ = g.keys[0] if g.keys else None
            if k_ is None or isinstance(k_, tuple):
                continue
            note(f"{g.note()[:60]}: pressing {k_} while the score moves")
            for _ in range(8):
                n0 = progress.scores.get(g.score_sid, 0)
                step(k_)
                did_any = True
                if get_obs().levels_completed > lv0:
                    return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
                if progress.scores.get(g.score_sid, 0) == n0:
                    break
            continue
        act_ = [s_ for s_ in tr.prev if s_.sid in (model.active or ())]
        me_ = max(act_, key=lambda s_: s_.size) if act_ else None
        if g.cause in ("arrive", "touch", "press") and me_ is None:
            continue
        if g.cause == "click" and not clickable:
            continue
        misses = 0
        for _round in range(12):
            targets = [s_ for s_ in tr.prev if s_.sid < 1000 and s_.body == g.kind and s_.sid not in g.done and "hud" not in s_.tags and "terrain" not in s_.tags]
            if not targets:
                break
            if me_ is not None:
                me_ = next((s_ for s_ in tr.prev if s_.sid == me_.sid), me_)
                t_ = min(targets, key=lambda s_: abs(s_.centre[0] - me_.centre[0]) + abs(s_.centre[1] - me_.centre[1]))
            else:
                t_ = targets[0]
            n0 = progress.scores.get(g.score_sid, 0); n_all0 = len(progress.events)
            note(f"{g.note()[:60]}: {g.cause} S{t_.sid} at {t_.bbox}")
            if verbose:
                print(f"   progress: {g.cause} S{t_.sid} kind {g.kind} at {t_.bbox} for score S{g.score_sid}")
            if g.cause == "click":
                step(("click", click_point(t_)))
            elif g.cause == "press":
                # (the press over a thing of the kind: go onto it, then press)
                _walk_over(tr, model, keys, me_.sid, t_, step, verbose, max_steps=40)
                if get_obs().levels_completed > lv0:
                    return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
                step(g.keys[0])
            else:
                _walk_over(tr, model, keys, me_.sid, t_, step, verbose, max_steps=40)
            did_any = True
            g.done.add(t_.sid)
            if get_obs().levels_completed > lv0:
                return True, int(getattr(play_level, "steps", 0) or 0), "progress", get_obs()
            if progress.scores.get(g.score_sid, 0) > n0 or len(progress.events) > n_all0:
                misses = 0
            else:
                misses += 1
                if misses >= 2:
                    note(f"{g.note()[:50]}: two targets moved nothing — refuted")
                    g.refuted = True
                    break
    if did_any:
        look_again()
        return pursue_again()
    return None


def run_protocol(tr, model, keys, step, get_obs, lv0, verbose, progress, note, bg):
    """THE RUN'S TELLING IS A GOAL AND A TEST (statement 6 of the no-reading
    class; sp80).  A key drew a trace and took it back: the things that
    flashed at its end are what the run is about — the goal is a run at
    which nothing flashes.  The trace is a law (runkey.flow); when the law
    redraws every trace seen, the plan places the movers where the law
    says nothing will flash and presses the run key once; when it does
    not, the run is a test: one press per place, nearest first.
    Returns True when the level was won here."""
    from agent.core import runkey as _rk
    runs_ = [x for x in progress.runs if x[1].get("source") is not None]
    if not runs_:
        return False
    run_key_ = runs_[-1][0]
    if isinstance(run_key_, tuple):
        return False                    # (a click's trace: not a key to press at places)
    obs0_ = get_obs()
    # the law, checked against every trace seen
    trusted_ = all(_rk.law_holds(fb_, r_, th_, bg, hud_rows=_hud_rows(th_)) for _, r_, fb_, th_ in runs_)
    flashers_ = []
    for _, r_, _, _ in runs_:
        for sid_ in r_["flash"]:
            if sid_ not in flashers_:
                flashers_.append(sid_)
    if not flashers_:
        # (nothing flashed: the run's judgement is unread — the things the
        # trace could fill are the goal)
        flashers_ = [s_.sid for s_ in tr.prev if s_.sid < 1000 and "hud" not in s_.tags and "terrain" not in s_.tags and _rk._rays_inside(set(s_.cells.keys()), s_.bbox)]
    # the movers: what the keys translate, and what a click can hand the
    # keys to (the selectable kinds, and the kind worn while selected)
    movers_ = []
    for k_ in keys:
        try:
            for sid_, e_ in model.fixed.moving(k_).items():
                if e_[0] == "translate" and sid_ not in movers_ and any(s_.sid == sid_ for s_ in tr.prev):
                    movers_.append(sid_)
        except Exception:
            pass
    sel_kinds_ = set(getattr(model, "select_kinds", ()) or ()) | ({model.selected_colour} if getattr(model, "selected_colour", None) is not None else set())
    for s_ in tr.prev:
        if s_.sid < 1000 and "hud" not in s_.tags and "terrain" not in s_.tags and kind(s_) in sel_kinds_ and s_.sid not in movers_:
            movers_.append(s_.sid)
    src_sid_ = runs_[-1][1]["source"]
    direction_ = runs_[-1][1]["direction"]
    ink_ = runs_[-1][1]["colour"]
    movers_ = [m_ for m_ in movers_ if m_ not in (runs_[-1][1].get("sources") or [src_sid_])]
    if not movers_:
        return False
    note(f"the run ({run_key_}) draws a trace from S{src_sid_} and takes it back; {len(flashers_)} things flash at its end — the goal is a run at which nothing flashes; the law {'redraws every trace seen' if trusted_ else 'does not redraw the traces: the run is a test'}")
    if verbose:
        print(f"   run protocol: key {run_key_} source S{src_sid_} dir {direction_} movers {movers_} flashers {flashers_} trusted {trusted_} runs {len(runs_)}")

    def frame_of_(state_):
        """The frame with the things where a state puts them."""
        f_ = np.array(model._frame_now).copy()
        moved_ = []
        for s_ in tr.prev:
            if s_.sid in state_ and s_.sid < 1000:
                cs_ = state_[s_.sid]
                if cs_ != frozenset(s_.cells.keys()):
                    for p_ in s_.cells:
                        f_[p_] = bg
                    moved_.append((s_, cs_))
        for s_, cs_ in moved_:
            b0_ = s_.bbox; b1_ = _bbox(cs_)
            v_ = (b1_[0] - b0_[0], b1_[1] - b0_[1])
            for p_, col_ in s_.cells.items():
                q_ = (p_[0] + v_[0], p_[1] + v_[1])
                if 0 <= q_[0] < f_.shape[0] and 0 <= q_[1] < f_.shape[1]:
                    f_[q_] = col_
        return f_

    def score_(f_):
        """How many of the flashers would flash under the law: a thing with
        an inside flashes when the ink does not reach it; a thing without
        one flashes when the ink touches it."""
        inked_, touched_ = _rk.flow_all(f_, runs_[-1][1], tr.prev, bg, _hud_rows(tr.prev))
        if not inked_:
            return 999, set()
        n_ = 0
        for sid_ in flashers_:
            s_ = next((x_ for x_ in tr.prev if x_.sid == sid_), None)
            if s_ is None:
                continue
            ins_ = _rk._rays_inside(set(s_.cells.keys()), s_.bbox)
            if ins_:
                if not (ins_ & inked_):
                    n_ += 1
            elif touched_ & set(s_.cells.keys()):
                n_ += 1
        return n_, inked_

    def placements_(m_sid_, cap_=400, depth_=14):
        """Where the keys can take a mover from here: {cells: path}."""
        comps_ = tr.prev
        start_ = {s_.sid: frozenset(s_.cells) for s_ in comps_ if "hud" not in s_.tags}
        m_ = next((s_ for s_ in comps_ if s_.sid == m_sid_), None)
        if m_ is None:
            return {}
        start_["_sel"] = m_sid_
        solids_ = {s_.sid for s_ in comps_ if "terrain" not in s_.tags and "hud" not in s_.tags and s_.size > 2}
        succ_ = make_successors(model, comps_, solids_, keys, m_sid_, m_.centre, frame=model._frame_now)
        out_ = {start_[m_sid_]: ([], start_)}
        frontier_ = [(start_, [])]
        n_ = 0
        while frontier_ and n_ < cap_:
            st_, path_ = frontier_.pop(0)
            if len(path_) >= depth_:
                continue
            try:
                nx_ = succ_(st_)
            except Exception:
                nx_ = []
            for a_, st2_ in nx_:
                if not isinstance(a_, str):
                    continue
                cs_ = st2_.get(m_sid_)
                if cs_ is None or cs_ in out_:
                    continue
                out_[cs_] = (path_ + [a_], st2_)
                frontier_.append((st2_, path_ + [a_]))
                n_ += 1
        return out_

    pressed_ = 0
    if trusted_:
        for round_ in range(2 * len(movers_) + 2):
            f_now_ = np.array(model._frame_now)
            n0_, _ = score_(f_now_)
            if verbose:
                print(f"   run protocol: round {round_}: {n0_} would flash here")
            if n0_ == 0:
                break
            best_ = None
            for m_sid_ in movers_:
                pl_ = placements_(m_sid_)
                for cs_, (path_, st_) in pl_.items():
                    if not path_:
                        continue
                    n_, _ = score_(frame_of_(st_))
                    cand_ = (n_, len(path_), m_sid_, path_)
                    if best_ is None or cand_[:2] < best_[:2]:
                        best_ = cand_
            if best_ is None or best_[0] >= n0_:
                if verbose:
                    print(f"   run protocol: no placement does better than {n0_} (best {best_[:3] if best_ else None}); the law is left")
                break
            n_, _, m_sid_, path_ = best_
            # (a part of the way for many presses is not taken: the movers
            # known now may not be all there are — the fiddle may yet find
            # the click that chooses another; six presses, hand-chosen)
            if n_ > 0 and len(path_) > 6:
                if verbose:
                    print(f"   run protocol: the best placement leaves {n_} flashing for {len(path_)} presses: not taken; the law is left")
                break
            note(f"the law says S{m_sid_} at {len(path_)} presses leaves {n_} flashing (now {n0_})")
            if verbose:
                print(f"   run protocol: moving S{m_sid_} by {path_} -> {n_} flashing")
            if model.selected_now(tr.prev) != m_sid_ and len(movers_) > 1:
                step(("select", m_sid_)); pressed_ += 1
                if verbose:
                    print(f"   run protocol: selected S{m_sid_}; selected now {model.selected_now(tr.prev)}")
                if model.selected_now(tr.prev) != m_sid_:
                    note(f"the click did not choose S{m_sid_} as read: the keys are not pressed for it")
                    continue
            for k_ in path_:
                step(k_); pressed_ += 1
                if get_obs().levels_completed > lv0:
                    return True
            # (the world's answer: when the mover is not where the law's plan
            # put it — a bound, a refusal — the plan is not made again)
            n_after_, _ = score_(np.array(model._frame_now))
            if n_after_ != n_:
                note(f"S{m_sid_} did not go where the plan put it ({n_after_} flashing, not {n_}): the law is left")
                if verbose:
                    print(f"   run protocol: after moving S{m_sid_} {n_after_} would flash, not {n_}: stopping")
                break
        f_now_ = np.array(model._frame_now)
        n0_, _ = score_(f_now_)
        if n0_ == 0:
            note(f"the law says nothing flashes here: pressing {run_key_}")
            n_runs_ = len(progress.runs)
            step(run_key_)
            if get_obs().levels_completed > lv0:
                return True
            if len(progress.runs) > n_runs_:
                r_ = progress.runs[-1][1]
                note(f"the run flashed {r_['flash']} where the law said nothing would: the law is doubted")
                if verbose:
                    print(f"   run protocol: the law failed: flash {r_['flash']} entered {r_['entered']}")
            trusted_ = False
    if not trusted_:
        # THE RUN AS A TEST: the first mover's places on the trace's line,
        # nearest first, one run each
        m_sid_ = movers_[0]
        tr_cells_ = set(runs_[-1][1]["trace"])
        if direction_[0]:
            lane_ = {c_ for _, c_ in tr_cells_}
            on_lane_ = lambda cs_: any(c_ in lane_ for _, c_ in cs_)
        else:
            lane_ = {r_ for r_, _ in tr_cells_}
            on_lane_ = lambda cs_: any(r_ in lane_ for r_, _ in cs_)
        pl_ = placements_(m_sid_, cap_=300, depth_=10)
        cands_ = sorted(((len(p_), cs_, p_) for cs_, (p_, _) in pl_.items() if p_ and on_lane_(cs_)), key=lambda x_: x_[0])
        tried_ = 0
        here_ = next((frozenset(s_.cells) for s_ in tr.prev if s_.sid == m_sid_), None)
        for n_, cs_, path_ in cands_[:6]:
            note(f"test run {tried_ + 1}: S{m_sid_} moved by {path_}, then {run_key_}")
            if model.selected_now(tr.prev) != m_sid_ and len(movers_) > 1:
                step(("select", m_sid_))
            for k_ in path_:
                step(k_)
            step(run_key_)
            tried_ += 1
            if get_obs().levels_completed > lv0:
                return True
            # (the path was from the first place: walk back before the next)
            back_ = {"ACTION1": "ACTION2", "ACTION2": "ACTION1", "ACTION3": "ACTION4", "ACTION4": "ACTION3"}
            for k_ in reversed(path_):
                if k_ in back_:
                    step(back_[k_])
    return get_obs().levels_completed > lv0


def _hud_rows(things):
    return {r for s_ in things if ("hud" in s_.tags or "bar" in s_.tags) for r, _ in s_.cells}


def carry_protocol(tr, model, step, get_obs, lv0, verbose, hyps, note, keys):
    """CARRY EACH THING TO ITS PLACE (wa30): a key pressed against a thing
    binds it to the body (the carry law); the reading names where the
    things of that kind go (the crates onto the pen: as many places in
    the pen as crates fit — the fit).  For each: walk against it, bind,
    plan it to its place with the body, let go.  Things are known by
    their kind and place, not their ids (a held thing wears another
    colour, and the tracker may swap two that touch).  Returns True
    when the level was won."""
    cr = model.carry
    if not cr:
        return False
    K = cr["key"]
    me_kind = cr.get("me_kind")
    core = cr.get("core")                   # (kind, dr, dc, h, w): what lies within the carried thing, and where

    def body_now():
        cs = [s for s in tr.prev if s.sid < 1000 and (me_kind is None or kind(s) == me_kind) and (s.sid in (model.active or ()) or me_kind is not None)]
        return max(cs, key=lambda s: s.size) if cs else None

    def within(cb, hb):
        return hb[0] <= cb[0] and hb[1] <= cb[1] and cb[2] <= hb[2] and cb[3] <= hb[3]

    def crates_now():
        # (identity survives the colour worn: a thing of the carried kind's
        # form — its dims, its core within — is one of them whatever its
        # border shows: idle, faced, held)
        out = []
        dims_ = tuple(cr["dims"])
        for s in tr.prev:
            if s.sid >= 1000 or "hud" in s.tags or "terrain" in s.tags:
                continue
            sd_ = (s.bbox[2] - s.bbox[0] + 1, s.bbox[3] - s.bbox[1] + 1)
            if sd_ == dims_:
                if kind(s) in cr["kinds"] or (core is not None and any(within(c_.bbox, s.bbox) for c_ in cores_now())):
                    out.append(s)
                continue
            # (things of the kind standing against each other are read as one
            # thing of their joined size: each core within it is one of them,
            # its box the core's, widened by the form)
            if core is not None and kind(s) in cr["kinds"] and sd_[0] >= dims_[0] and sd_[1] >= dims_[1]:
                cs_ = set(s.cells)
                for c_ in cores_now():
                    if not within(c_.bbox, s.bbox):
                        continue
                    box_ = (c_.bbox[0] - core[1], c_.bbox[1] - core[2], c_.bbox[0] - core[1] + dims_[0] - 1, c_.bbox[1] - core[2] + dims_[1] - 1)
                    cells_ = frozenset(q for q in cs_ if box_[0] <= q[0] <= box_[2] and box_[1] <= q[1] <= box_[3])
                    if not cells_:
                        continue
                    out.append(types.SimpleNamespace(sid=s.sid, bbox=box_, cells=cells_, body=s.body, size=len(cells_), tags=set(s.tags),
                                                     centre=((box_[0] + box_[2]) / 2.0, (box_[1] + box_[3]) / 2.0)))
        # (one thing per place: a core read twice — as a part and as a thing — is one)
        seen_ = set(); uniq_ = []
        for t in out:
            if t.bbox not in seen_:
                seen_.add(t.bbox); uniq_.append(t)
        # A THING IS KNOWN BY ITS FORM: a core with a border of one colour
        # around it, of the carried thing's dims, is one of them however the
        # tracker cut the picture (three set down side by side read as one)
        if core is not None:
            f_ = np.asarray(model._frame_now)
            bg_ = background_color(f_)
            for c_ in cores_now():
                box_ = (c_.bbox[0] - core[1], c_.bbox[1] - core[2], c_.bbox[0] - core[1] + dims_[0] - 1, c_.bbox[1] - core[2] + dims_[1] - 1)
                if box_ in seen_ or box_[0] < 0 or box_[1] < 0 or box_[2] >= f_.shape[0] or box_[3] >= f_.shape[1]:
                    continue
                core_cells = set(c_.cells)
                border_ = [(r, c) for r in range(box_[0], box_[2] + 1) for c in range(box_[1], box_[3] + 1) if (r, c) not in core_cells]
                cols_ = {int(f_[r, c]) for r, c in border_}
                if len(cols_) == 1:
                    col_ = next(iter(cols_))
                    if col_ != bg_:
                        cells_ = frozenset(border_) | frozenset(core_cells)
                        seen_.add(box_)
                        uniq_.append(types.SimpleNamespace(sid=900000 + len(uniq_), src=(c_.sid // 1000 if c_.sid >= 1000 else c_.sid), bbox=box_, cells=cells_, body=col_, size=len(cells_), tags=set(),
                                                           centre=((box_[0] + box_[2]) / 2.0, (box_[1] + box_[3]) / 2.0)))
        return uniq_

    def held_now():
        hc_ = cr.get("held_colour")
        return [s for s in crates_now() if hc_ is not None and s.body == hc_]

    def cores_now():
        if core is None:
            return []
        # (a core read as a part of its crate — id anchored to it — is a core too)
        return [s for s in tr.prev if kind(s) == core[0] and (s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1]) == (core[3], core[4])]

    def held_core(me):
        """The core of the thing held: within a crate wearing the held
        colour, or lying in no crate's box (its border merged with the
        body's) — the nearest to the body."""
        boxes = [c.bbox for c in crates_now() if c.body != cr.get("held_colour")]
        loose = [s for s in cores_now() if not any(within(s.bbox, b) for b in boxes)]
        if not loose or me is None:
            return None
        return min(loose, key=lambda s: abs(s.centre[0] - me.centre[0]) + abs(s.centre[1] - me.centre[1]))

    # (rounds are bounded by progress, not by count: a round that set a
    # thing down resets the allowance — another carrier may take the
    # thing walked to, and that costs a round, not the level)
    idle_rounds = 0
    done_before = -1
    def form_things(comps):
        """Things of the kind known by their form in the frame but cut
        otherwise by the tracker: they stand in the way all the same."""
        # (what the tracker holds as a thing of the kind covers; what it
        # holds as a recoloured patch of the target does not — that patch
        # IS the thing set down on it, read the other way round)
        all_ = crates_now()
        known_ = {t.sid for t in all_ if not (isinstance(t, types.SimpleNamespace) and t.sid >= 900000)}
        covered_ = set()
        for s in comps:
            if s.sid in known_ or (s.sid >= 1000 and s.sid // 1000 in known_):
                covered_ |= set(s.cells)
        out_ = []
        f_ = np.asarray(model._frame_now)
        for t in all_:
            if isinstance(t, types.SimpleNamespace) and t.sid >= 900000 and t.body != cr.get("held_colour") and len(set(t.cells) - covered_) >= 0.5 * len(t.cells):
                out_.append(Sprite(sid=t.sid, cells={q: int(f_[q[0], q[1]]) for q in t.cells}, parts=1, tags=set()))
        return out_

    # THE FIRST LOOK: a thing of the carried kind found set into a see-through
    # thing (a hatching: its box mostly background) that stands against it on
    # two opposite sides was LODGED in it — the hatching's kind is one the
    # carried things may pass (applied where the held thing is planned).
    # Hand-chosen: a hatching fills under 4/5 of its box.
    if cr.get("hatch_lv") != lv0:
        cr["hatch_lv"] = lv0
        cr["short"] = 0
        hk_ = set()
        for t_ in crates_now():
            tb_ = t_.bbox
            sides_ = {}
            for o_ in tr.prev:
                if o_.sid >= 1000 or "hud" in o_.tags or o_.size <= 4 or o_.bbox == tb_ or kind(o_) in cr["kinds"] or kind(o_) == me_kind:
                    continue
                ob_ = o_.bbox
                if o_.size > 0.8 * (ob_[2] - ob_[0] + 1) * (ob_[3] - ob_[1] + 1):
                    continue                    # (a filled thing is no hatching)
                oc_ = set(o_.cells)
                sd_ = sides_.setdefault(kind(o_), set())
                if any((tb_[0] - 1, c) in oc_ for c in range(tb_[1], tb_[3] + 1)): sd_.add("above")
                if any((tb_[2] + 1, c) in oc_ for c in range(tb_[1], tb_[3] + 1)): sd_.add("below")
                if any((r, tb_[1] - 1) in oc_ for r in range(tb_[0], tb_[2] + 1)): sd_.add("left")
                if any((r, tb_[3] + 1) in oc_ for r in range(tb_[0], tb_[2] + 1)): sd_.add("right")
            for kd_, sd_ in sides_.items():
                if {"above", "below"} <= sd_ or {"left", "right"} <= sd_:
                    hk_.add(kd_)
                    note(f"carry: a thing of the kind at {tb_} was found lodged in the {kd_}-coloured hatching")
        # (carried across levels by kind: what passed the hatching on one
        # level passes it on the next, lodged demonstration or none — wa30
        # L4's blocks start in a fenced yard with no block in the fence)
        cr["hatch_kinds"] = sorted(set(cr.get("hatch_kinds", ())) | hk_)
        if verbose and hk_:
            print(f"   carry: hatching kinds the carried things were found lodged in: {sorted(hk_)}")

    for round_ in range(60):
        if get_obs().levels_completed > lv0:
            return True
        # (a round that found a refusal and remembered it by place is a
        # probe along the wall — the gap is found by probing; six such
        # rounds without a thing set down, and the protocol yields)
        if idle_rounds >= 6:
            if verbose:
                print("   carry: six rounds without a thing set down: the protocol yields")
            return False
        comps = list(tr.prev)
        me = body_now()
        if me is None:
            return False
        comps = comps + form_things(comps)
        # WE ARE IN ITS WAY: an agent stalled against our body (the
        # continuity monitor's reading) — step aside, across its way,
        # where nothing stands; the helper goes on
        st_ = getattr(model, "stalled", {}) or {}
        # (THE RECORD SAYS WHO IT IS: a foe — what it moves goes away from
        # the goal's place — is not stepped aside for and not waited on;
        # tw01 L3, the helper that turned)
        rec_ = getattr(model, "agent_record", None)
        if st_ and rec_ is not None:
            st_ = {a_: n_ for a_, n_ in st_.items() if a_ not in rec_.tracks or rec_.tracks[a_].stance() != "foe"}
        if st_:
            a_sid = max(st_, key=st_.get)
            va_ = getattr(model, "last_way", {}).get(a_sid)
            occ_ = set()
            for s in comps:
                if s.sid != me.sid and s.sid < 1000 and "hud" not in s.tags and not any(s.sid == b_ for (_, b_) in model.bound):
                    occ_ |= set(s.cells)
            best_ = None
            for k in keys:
                e_ = model.fixed.moving(k).get(me.sid)
                if not (e_ and e_[0] == "translate" and (e_[1][0] or e_[1][1])):
                    continue
                if va_ is not None and (e_[1][0] * va_[0] + e_[1][1] * va_[1]) != 0:
                    continue                  # (along its way, toward or away: still in its lane)
                nxt_ = {(r + e_[1][0], c + e_[1][1]) for r, c in me.cells}
                if nxt_ & occ_ or any(r < 0 or c < 0 or r >= 64 or c >= 64 for r, c in nxt_):
                    continue
                best_ = k; break
            if best_ is not None:
                note(f"carry: the helper S{a_sid} stands still with us in its way — stepping aside ({best_})")
                if verbose:
                    print(f"   carry: stepping aside for the stalled agent S{a_sid} with {best_}")
                step(best_)
                if get_obs().levels_completed > lv0:
                    return True
                model.stalled.pop(a_sid, None)
                continue
        _restore_body_laws(model, me.sid, keys)
        things = crates_now()
        hc = held_core(me)
        if not things and hc is None:
            return False
        # (a thing of the kind not wearing the held colour is not held: a
        # bond left over from the test that found the law is let go of)
        unheld_ = {t.sid for t in things if t.body != cr.get("held_colour")}
        if any(o_ in unheld_ for (_, o_) in model.bound):
            if verbose:
                print(f"   carry: bonds to unheld things dropped: {[(m_, o_) for (m_, o_) in model.bound if o_ in unheld_]}")
            model.bound = {b_ for b_ in model.bound if b_[1] not in unheld_}
        # (A BOND IS A TOUCH: a thing bound to the body that no longer stands
        # against it — handed over, carried off by the helper — is let go of;
        # wa30 L4 planned every walk with a block the helper had taken)
        try:
            by_all_ = {s.sid: s for s in comps}
            far_ = set()
            for (m_, o_) in model.bound:
                if m_ != me.sid:
                    continue
                ob_ = by_all_.get(o_)
                if ob_ is None:
                    continue
                gap_ = max(ob_.bbox[0] - me.bbox[2], me.bbox[0] - ob_.bbox[2], ob_.bbox[1] - me.bbox[3], me.bbox[1] - ob_.bbox[3])
                if gap_ > 1:
                    far_.add(o_)
            if far_:
                if verbose:
                    print(f"   carry: bonds to things no longer against the body dropped: {sorted(far_)}")
                model.bound = {b_ for b_ in model.bound if b_[1] not in far_}
        except Exception:
            pass
        # WHERE THEY GO: a reading whose mover lies in a thing of the kind;
        # its target's box holds as many of them as fit side by side
        target_cells = cr.get("target_cells")
        if target_cells is None:
            # (the method carries: the last level's things of the kind went
            # onto a thing of a kind — a law a win tested, read before any
            # fresh reading guesses: here, the thing of that kind, whole)
            if cr.get("target_kind") is not None:
                crate_ids_ = {t.sid for t in things}
                cands_ = [s for s in comps if s.sid < 1000 and "hud" not in s.tags and s.sid not in crate_ids_ and kind(s) == cr["target_kind"]
                          and (s.bbox[2] - s.bbox[0] + 1) >= cr["dims"][0] and (s.bbox[3] - s.bbox[1] + 1) >= cr["dims"][1]]
                if cands_:
                    # (EVERY thing of that kind is a place — a level may have
                    # several pens, each holding as many as fit; wa30 L4's four
                    # pens for seven blocks, where one pen's two slots were all
                    # the places the carry knew)
                    cr["target_boxes"] = [s.bbox for s in cands_]
                    target_cells = frozenset((r, c) for s_ in cands_ for r in range(s_.bbox[0], s_.bbox[2] + 1) for c in range(s_.bbox[1], s_.bbox[3] + 1))
                    note(f"carry: the things of the kind go onto the {cr['target_kind']}-coloured things at {sorted(cr['target_boxes'])}, as on the last level")
            # (what the helper does with them: a thing of the kind an agent
            # carried came to rest on a thing — that is where the kind goes)
            if target_cells is None:
                rests_ = getattr(model, "rests", {}) or {}
                crate_ids_ = {t.sid for t in things}
                for kd_ in cr["kinds"] | {cr.get("held_colour")} | ({core[0]} if core else set()):
                    for rb_ in rests_.get(kd_, []):
                        under_ = [s for s in comps if s.sid < 1000 and "hud" not in s.tags and s.sid not in crate_ids_ and kind(s) not in cr["kinds"]
                                  and s.bbox[0] <= rb_[0] and s.bbox[1] <= rb_[1] and rb_[2] <= s.bbox[2] and rb_[3] <= s.bbox[3]
                                  and (s.bbox[2] - s.bbox[0] + 1) >= cr["dims"][0] and (s.bbox[3] - s.bbox[1] + 1) >= cr["dims"][1]]
                        if under_:
                            s_ = max(under_, key=lambda s: s.size)
                            target_cells = frozenset((r, c) for r in range(s_.bbox[0], s_.bbox[2] + 1) for c in range(s_.bbox[1], s_.bbox[3] + 1))
                            cr["target_kind"] = kind(s_)
                            note(f"carry: the helper left a thing of the kind on the {kind(s_)}-coloured thing at {s_.bbox}: that is where they go")
                            break
                    if target_cells is not None:
                        break
            for h in (hyps if target_cells is None else ()):
                if h.refuted:
                    continue
                inside = [t for t in things if t.bbox[0] - 0.5 <= h.m_start[0] <= t.bbox[2] + 0.5 and t.bbox[1] - 0.5 <= h.m_start[1] <= t.bbox[3] + 0.5]
                if not inside:
                    continue
                tc = frozenset(h.target) if not isinstance(h.target, tuple) else frozenset(getattr(h, "t_cells", ()) or ())
                if tc and not (tc & {q for t in things for q in t.cells}):
                    target_cells = tc
                    break
            if target_cells is None:
                if verbose:
                    print("   carry: no reading names where the carried kind goes")
                return False
            cr["target_cells"] = target_cells
            if len(cr.get("target_boxes") or ()) <= 1 or any(not (target_cells & frozenset((r, c) for r in range(b_[0], b_[2] + 1) for c in range(b_[1], b_[3] + 1))) for b_ in cr["target_boxes"]):
                cr["target_boxes"] = [_bbox(target_cells)]
            if cr.get("target_kind") is None:
                tc_ = set(target_cells)
                for s in comps:
                    if s.sid < 1000 and "hud" not in s.tags and kind(s) not in cr["kinds"] and (tc_ & set(s.cells)):
                        cr["target_kind"] = kind(s)
                        break
        tb = _bbox(target_cells)
        tboxes_ = cr.get("target_boxes") or [tb]
        if len(tboxes_) == 1:
            tboxes_ = [tb]

        def on_target(bb_):
            return any(within(bb_, b_) for b_ in tboxes_)
        th, tw = cr["dims"]
        places = []
        for tb_ in tboxes_:
            for r in range(tb_[0], tb_[2] - th + 2, th):
                for c in range(tb_[1], tb_[3] - tw + 2, tw):
                    places.append((r, c, r + th - 1, c + tw - 1))
        if not places:
            return False
        things = [t for t in things if t.body != cr.get("held_colour")]
        done = [t for t in things if any(t.bbox == p for p in places)]
        if len(done) > done_before:
            # (a thing set in place that we did not set down: the agent's
            # doing — its SIGN toward the goal, an ally's; a thing taken
            # out of place that we did not take is a foe's)
            if done_before >= 0 and not cr.get("_we_placed") and getattr(model, "agents", None):
                sg_ = getattr(model, "agent_sign", None)
                if sg_ is None:
                    sg_ = model.agent_sign = {}
                for kd_ in model.agents:
                    sg_[kd_] = sg_.get(kd_, 0) + (len(done) - done_before)
                    note(f"the agent of kind {kd_} set a thing in place: it advances the goal ({sg_[kd_]:+d})")
            idle_rounds = 0 if done_before >= 0 else idle_rounds
            done_before = len(done)
        else:
            if done_before >= 0 and len(done) < done_before and getattr(model, "agents", None):
                sg_ = getattr(model, "agent_sign", None)
                if sg_ is None:
                    sg_ = model.agent_sign = {}
                for kd_ in model.agents:
                    sg_[kd_] = sg_.get(kd_, 0) - (done_before - len(done))
                    note(f"a thing left its place that we did not take: the agent of kind {kd_} works against the goal ({sg_[kd_]:+d})")
                done_before = len(done)
            idle_rounds += 1
        cr["_we_placed"] = False
        todo = [t for t in things if t not in done]
        # (a thing brought as far as it goes and let go waits for the helper:
        # not taken up again while others remain)
        handed_ = cr.get("handed", [])
        if handed_:
            todo = [t for t in todo if t.bbox not in handed_]
        # in its hands: the helper is carrying it; we take another)
        agents_ = getattr(model, "agents", set())
        if agents_:
            ag_boxes = [s.bbox for s in comps if s.sid < 1000 and kind(s) in agents_ and s.sid != me.sid]
            # (...and what the record says an agent took and has not set down:
            # a thing it moved last press is in its hands wherever it stands
            # now — wa30 L2, the body chasing a block the helper carried off,
            # a press behind it each time)
            held_rec_ = set(getattr(model, "in_hands", {}) or {})
            in_hands = [t for t in todo if t.sid in held_rec_ or any(max(t.bbox[0] - b[2], b[0] - t.bbox[2], t.bbox[1] - b[3], b[1] - t.bbox[3]) <= 1 for b in ag_boxes)]
            if in_hands:
                if verbose:
                    print(f"   carry: {[t.bbox for t in in_hands]} against a thing of kind {sorted(agents_)} that travels on its own: in its hands, left to it")
                todo = [t for t in todo if t not in in_hands]
        # (A THING THAT MOVED WITHOUT US IS ON ITS WAY: one that stands elsewhere
        # than at our last look, though we never held it, travels — in another's
        # hands, or its own — and is not ours to take this round; wa30 L2, the
        # body chasing a block the helper carried along, a press behind it)
        try:
            last_seen_ = cr.get("_last_seen", {})
            held_now_ = {o_ for (_, o_) in model.bound} | {hc.sid} if hc is not None else {o_ for (_, o_) in model.bound}
            moving_ = [t for t in todo if t.sid in last_seen_ and last_seen_[t.sid] != t.bbox and t.sid not in held_now_
                       and not any(t.bbox == p for p in places)]
            if moving_:
                if verbose:
                    print(f"   carry: {[t.bbox for t in moving_]} moved since our last look without us: on its way, not taken this round")
                todo = [t for t in todo if t not in moving_]
            cr["_last_seen"] = {t.sid: t.bbox for t in things}
        except Exception:
            pass
        free = [p for p in places if not any(t.bbox == p for t in things)]
        if verbose:
            print(f"   carry: {len(things)} of the kind, {len(done)} in place, places free {free}, held core {hc.bbox if hc else None}")
            if _ENV.get("CARRY_DEBUG"):
                f_ = np.asarray(model._frame_now)
                print(f"      things {[(t.sid, t.body, t.bbox) for t in crates_now()]} me {me.bbox}")
                for r_ in range(max(0, tb[0] - 2), min(f_.shape[0], tb[2] + 3)):
                    print("      %2d " % r_ + "".join("%x" % f_[r_, c_] for c_ in range(max(0, tb[1] - 6), min(f_.shape[1], tb[3] + 7))))
        if not free:
            if hc is not None or held_now():
                step(K)                 # (every place filled, one still held: let go)
            return get_obs().levels_completed > lv0
        if hc is None:
            if not todo:
                # (a press the wall refuses passes the time without taking
                # or dropping anything: the move key whose step from here
                # lands on something solid)
                blocked_ = None
                occ_ = set()
                for s in comps:
                    if s.sid != me.sid and s.sid < 1000 and "hud" not in s.tags and not on_target(s.bbox):
                        occ_ |= set(s.cells)
                for t in things:
                    occ_ |= set(t.cells)
                for k in keys:
                    e_ = model.fixed.moving(k).get(me.sid)
                    if e_ and e_[0] == "translate" and (e_[1][0] or e_[1][1]):
                        nxt_ = {(r + e_[1][0], c + e_[1][1]) for r, c in me.cells}
                        if nxt_ & occ_ or any(r < 0 or c < 0 or r >= 64 or c >= 64 for r, c in nxt_):
                            blocked_ = k; break
                foes_ = {t_.kind for t_ in (getattr(model, "agent_record", None).tracks.values() if getattr(model, "agent_record", None) else ()) if t_.stance() == "foe"}
                if agents_ and len(done) < len(places) and not (set(agents_) <= foes_ and foes_) and in_hands_wait(get_obs, lv0, step, keys, verbose, blocked_key=blocked_):
                    continue            # (the rest are in the helper's hands: wait a little, then look again)
                return get_obs().levels_completed > lv0
            # the nearest pair (thing, free place)
            pairs = sorted(((abs(t.centre[0] - (p[0] + p[2]) / 2) + abs(t.centre[1] - (p[1] + p[3]) / 2), i, t, p) for t in todo for i, p in enumerate(free)), key=lambda x: (x[0], x[1]))
            _, _, t, p = pairs[0]
            # walk against it — on the side from which it can be set down
            # at its place (the body's own place then must be free) — and bind
            start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
            start["_sel"] = model.selected_now(comps)
            solids0 = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2}
            succ0 = make_successors(model, comps, solids0, keys, me.sid, t.centre, frame=model._frame_now, frozen=(t.sid,))
            # A WALK WITH NOTHING MOVED: when the way to the thing, shoves and
            # all, is beyond the search (a yard of six blocks — wa30 L4: every
            # shove of every block searched, 17k states, timed out three
            # cells from the side it wanted), the way with every other
            # pushable thing standing frozen is sought — a far smaller world.
            # (Sought second, not first: an equal-length walk found another
            # way shifted wa30 L2's zero-margin clock by the helper's timing.)
            frozen_walk_ = tuple({s.sid for s in comps if s.sid != me.sid and "hud" not in s.tags and kind(s) in model.pushable})
            succ0_walk = make_successors(model, comps, solids0, keys, me.sid, t.centre, frame=model._frame_now, frozen=frozen_walk_) if frozen_walk_ else None

            def approach_plan(goal_fn, heur_fn):
                p_ = plan_general(start, succ0, goal_fn, heuristic=heur_fn, max_states=30000, max_depth=40)
                if p_ is None and succ0_walk is not None:
                    p_ = plan_general(start, succ0_walk, goal_fn, heuristic=heur_fn, max_states=6000, max_depth=40)
                    if verbose and p_ is not None:
                        print(f"   carry: the way with shoves was beyond the search; a walk with nothing moved: {p_}")
                return p_
            group_ = set()
            for b_ in model.bodies():
                if me.sid in b_:
                    group_ |= set(b_)
            group_ = {g_ for g_ in group_ if g_ in start and g_ != t.sid} | {me.sid}
            occupied_ = set()
            crate_ids_ = {c_.sid for c_ in things}
            for s in comps:
                if s.sid in solids0 and s.sid not in group_ and s.sid != t.sid and not (s.sid >= 1000 and s.sid // 1000 == t.sid):
                    # (the target's own parts are no obstacle to standing on it:
                    # the body set the last level's things down from on it too)
                    if on_target(s.bbox) and s.sid not in crate_ids_ and (s.sid // 1000 if s.sid >= 1000 else s.sid) not in crate_ids_:
                        if _ENV.get("CARRY_DEBUG"):
                            print(f"   carry: S{s.sid} {s.body}/{s.size}@{s.bbox} lies on the target: not in the way")
                        continue
                    occupied_ |= set(s.cells)
            sides_ = []
            # (a strait in this level — a hatching the things were found lodged
            # in, a carry that ended short: the thing will be handed over, not
            # set down by us, so the side need not be free at the place)
            strait_ = bool(cr.get("hatch_kinds")) or cr.get("short", 0) > 0
            for dv_ in ((-th, 0), (th, 0), (0, -tw), (0, tw)):
                box_p = (p[0] + dv_[0], p[1] + dv_[1], p[2] + dv_[0], p[3] + dv_[1])
                cells_p = {(r, c) for r in range(box_p[0], box_p[2] + 1) for c in range(box_p[1], box_p[3] + 1)}
                if not strait_ and (box_p[0] < 0 or box_p[1] < 0 or box_p[2] >= 64 or box_p[3] >= 64 or (cells_p & occupied_)):
                    continue
                box_t = (t.bbox[0] + dv_[0], t.bbox[1] + dv_[1], t.bbox[2] + dv_[0], t.bbox[3] + dv_[1])
                cells_t = {(r, c) for r in range(box_t[0], box_t[2] + 1) for c in range(box_t[1], box_t[3] + 1)}
                if box_t[0] < 0 or box_t[1] < 0 or box_t[2] >= 64 or box_t[3] >= 64 or (cells_t & occupied_):
                    continue
                # (BEHIND IT, TOWARD ITS PLACE: the side from which the thing
                # lies between the body and its place is tried first — what
                # must pass a strait the body cannot is pushed through it,
                # not pulled; then the nearest side)
                to_ = ((p[0] + p[2]) / 2.0 - t.centre[0], (p[1] + p[3]) / 2.0 - t.centre[1])
                dot_ = dv_[0] * to_[0] + dv_[1] * to_[1]
                behind_ = 0 if dot_ < 0 else 1
                sides_.append((behind_, abs(box_t[0] - me.bbox[0]) + abs(box_t[1] - me.bbox[1]), box_t, dot_))
            sides_.sort(key=lambda x: x[:3])
            behind_boxes_ = [b_ for bh_, _, b_, _ in sides_ if bh_ == 0]
            # WHAT MUST BE HANDED OVER IS CARRIED IN FRONT: where the level has
            # a strait the body cannot pass (a hatching the things were found
            # lodged in; a carry that ended short of its place), the thing is
            # taken from the side squarely behind it on the way to its place —
            # the side whose way lies most along that way — and from no other
            # while that side can be reached (wa30 L3: a block taken from the
            # right is carried behind the body, and the body, not the block,
            # meets the fence)
            if strait_ and behind_boxes_:
                behind_boxes_ = [b_ for _, _, b_, _ in sorted((s_ for s_ in sides_ if s_[0] == 0), key=lambda x: (x[3], x[1]))][:1]
            sides_ = [(d_, b_) for _, d_, b_, _ in sides_]
            if verbose:
                print(f"   carry: sides to take {t.bbox} from: {[b_ for _, b_ in sides_]}")
            if not sides_:
                cr.setdefault("handed", []).append(t.bbox)
                note(f"carry: no side to take the thing at {t.bbox} from: left")
                continue
            boxes_ok = [b_ for _, b_ in sides_]

            def at_side(st):
                # (the body's own box lies within the side's box; its front
                # row — read apart, on whichever side it faces — is turned
                # toward the thing by the facing press after the walk)
                cs = st.get(me.sid)
                if not cs:
                    return False
                ba = _bbox(cs)
                return any(within(ba, b_) and ba[2] - ba[0] >= 1 and ba[3] - ba[1] >= 1 for b_ in boxes_ok)

            def near(st, _o=t.sid):
                cs = st.get(me.sid)
                if not cs:
                    return 999
                ba = _bbox(cs)
                return min(abs(ba[0] - b_[0]) + abs(ba[1] - b_[1]) for b_ in boxes_ok)
            # (the sides behind it, toward its place, are listed first; the
            # plan still ends at whichever side is nearest by its own count —
            # choosing the behind side alone cost L2 the walk round and the
            # level: a strait that must be PUSHED through needs a plan that
            # looks past the bind, which this one does not yet)
            p0 = None
            if strait_ and behind_boxes_ and len(behind_boxes_) < len(sides_):
                boxes_all_ = boxes_ok
                boxes_ok = list(behind_boxes_)
                p0 = approach_plan(at_side, lambda st: near(st) / 3.0)
                if verbose:
                    print(f"   carry: a strait in this level: the thing is taken from behind it, {boxes_ok[0]}: {'reached' if p0 is not None else 'not reachable — any side then'}")
                if p0 is None:
                    boxes_ok = boxes_all_
            if p0 is None:
                p0 = approach_plan(at_side, lambda st: near(st) / 3.0)
            note(f"carry: walking against the thing at {t.bbox} to bind it with {K}")
            if verbose:
                print(f"   carry: approach {t.bbox} from {me.bbox}: {p0}")
                if p0:
                    st_ = start; trail_ = []
                    for k in p0:
                        nx_ = next((n_ for a_, n_ in succ0(st_) if a_ == k), None)
                        if nx_ is None:
                            break
                        trail_.append(_bbox(nx_[me.sid])); st_ = nx_
                    print(f"   carry: predicted body places {trail_}; solids {sorted(solids0)[:12]} flat {sorted(model.flat)} pushable {sorted(model.pushable)} passable {sorted(model.passable)}")
            if p0 is None and verbose:
                print(f"   carry: approach failed: last {getattr(plan_general, 'last', None)} whys {getattr(make_successors, 'whys', None)} start succ {[(a_, _bbox(n_[me.sid])) for a_, n_ in succ0(start)]}")
                make_successors.trace = True
                try:
                    _ = succ0(start)
                finally:
                    make_successors.trace = False
                print(f"      body S{me.sid} laws: {[(k_, model.fixed.eff[k_].get(me.sid), model.fixed.support[k_].get(me.sid), dict(model.fixed.alt[k_][me.sid]), [c_ for c_ in model.fixed.contradicted.get(k_, ()) if c_ == me.sid or (isinstance(c_, tuple) and me.sid in c_)]) for k_ in keys]} active {model.active} bodies {model.bodies()} sel {start.get('_sel')} ctx {[(k_, s_) for (k_, s_) in model.fixed.ctx]}")
            if p0 is None:
                # (no way to stand beside it: it is beyond our reach — in the
                # helper's reach, maybe; the others are ours)
                cr.setdefault("handed", []).append(t.bbox)
                note(f"carry: no way to stand beside the thing at {t.bbox}: beyond our reach, left to the helper")
                if verbose:
                    print(f"   carry: the thing at {t.bbox} is beyond our reach: left")
                continue
            for k in p0:
                step(k)
                if get_obs().levels_completed > lv0:
                    return True
            me2 = body_now(); t2 = next((s for s in crates_now() if s.bbox == t.bbox), None)
            if me2 is None or t2 is None:
                continue
            if not any(within(me2.bbox, b_) for b_ in boxes_ok):
                if verbose:
                    print(f"   carry: the body stands at {me2.bbox}, not beside the thing: replanning")
                continue
            vec_of = {}
            for k in keys:
                e_ = model.fixed.moving(k).get(me2.sid)
                if e_ and e_[0] == "translate" and (e_[1][0] or e_[1][1]):
                    vec_of[k] = e_[1]
                else:
                    alts_ = [(n_, ee_[1]) for ee_, n_ in model.fixed.alt[k][me2.sid].items() if ee_[0] == "translate" and n_ > 0 and (ee_[1][0] or ee_[1][1])]
                    if alts_:
                        vec_of[k] = max(alts_)[1]
            if not vec_of:
                return False
            d = (t2.centre[0] - me2.centre[0], t2.centre[1] - me2.centre[1])
            toward = max(vec_of, key=lambda k: vec_of[k][0] * d[0] + vec_of[k][1] * d[1])
            if verbose:
                print(f"   carry: facing {t2.bbox}: d {d} keys {vec_of} -> {toward}")
            step(toward)
            if get_obs().levels_completed > lv0:
                return True
            if verbose:
                print(f"   carry: after facing: events {[(e.kind, e.sid) for e in tr.last_events][:8]}; body {[(s.sid, s.bbox) for s in tr.prev if kind(s) == kind(me2)]}")
                if _ENV.get("CARRY_DEBUG"):
                    f_ = np.asarray(model._frame_now)
                    for r_ in range(max(0, me2.bbox[0] - 6), min(f_.shape[0], me2.bbox[2] + 7)):
                        print("      %2d " % r_ + "".join("%x" % f_[r_, c_] for c_ in range(max(0, me2.bbox[1] - 12), min(f_.shape[1], me2.bbox[3] + 24))))
            step(K)
            if get_obs().levels_completed > lv0:
                return True
            if verbose:
                print(f"   carry: after {K}: events {[(e.kind, e.sid) for e in tr.last_events][:8]}; body {[(s.sid, s.bbox) for s in tr.prev if kind(s) == kind(me2)]}")
            me = body_now()
            # (a carry law claimed without its held colour — a reader's —
            # learns it at the first bind: the thing faced wears another
            # colour now; that colour is "held")
            if cr.get("held_colour") is None and me is not None:
                for s in tr.prev:
                    if s.sid < 1000 and s.bbox == t2.bbox and s.body not in cr["kinds"] and s.body != kind(me):
                        cr["held_colour"] = s.body
                        note(f"carry: the thing bound wears colour {s.body} while held")
                        break
            hc = held_core(me)
            if hc is None:
                if verbose:
                    print("   carry: nothing held after the bind")
                cr["misses"] = cr.get("misses", 0) + 1
                # (a carried-in law re-tested twice and refuted: not this level's)
                if cr.get("seeded") and cr["misses"] >= 2:
                    note(f"carry: {K} against the thing bound nothing twice — the carry law from the last level is dropped here")
                    model.carry = None
                    return False
                continue
            place = p
        else:
            place = min(free, key=lambda p: abs(hc.centre[0] - (p[0] + p[2]) / 2) + abs(hc.centre[1] - (p[1] + p[3]) / 2))
        # plan the held thing to its place, the body carrying it (its core
        # is what is tracked: the border wears the held colour)
        comps = list(tr.prev)
        me = body_now(); hc = held_core(me)
        if me is None or hc is None:
            continue
        extra_ = form_things(comps)
        if extra_:
            if verbose:
                print(f"   carry: things of the kind read by their form, in the way: {[e_.bbox for e_ in extra_]}")
            comps = comps + extra_
        _restore_body_laws(model, me.sid, keys)
        model.bound.add((me.sid, hc.sid))
        # WHAT WAS FOUND LODGED IN THE HATCHING PASSES THE HATCHING: a kind the
        # carried things were found set into at the level's first look (a
        # see-through thing against them on two opposite sides) does not stop
        # the carried thing — the thing, not the body (wa30 L3: the blocks set
        # in the fence are the helper's to take from its side; ours are
        # brought into the fence to be taken).  Read at the look, never
        # assumed of a solid wall: a filled thing is no hatching.
        # (by the THING, not its kind: the body's nose may wear the held
        # thing's colour, and a kind's leave would carry the body through)
        hatch_ = [s for s in comps if s.sid < 1000 and kind(s) in cr.get("hatch_kinds", ())]
        if hatch_:
            border_ = [s for s in comps if s.sid < 1000 and within(hc.bbox, s.bbox) and s.sid != hc.sid
                       and (s.bbox[2] - s.bbox[0] + 1) <= cr["dims"][0] and (s.bbox[3] - s.bbox[1] + 1) <= cr["dims"][1]]
            for h_ in hatch_:
                for s_ in [hc] + border_:
                    pr_ = frozenset((s_.sid, h_.sid))
                    if pr_ not in model.fixed.overlap_ok:
                        model.fixed.overlap_ok.add(pr_)
                        note(f"carry: the held thing S{s_.sid} passes the {kind(h_)}-coloured hatching S{h_.sid} it was found lodged in")
                        if verbose:
                            print(f"   carry: the held thing S{s_.sid} passes the hatching S{h_.sid} (kind {kind(h_)})")
        p_core = (place[0] + core[1], place[1] + core[2], place[0] + core[1] + core[3], place[1] + core[2] + core[4])
        start = {s.sid: frozenset(s.cells) for s in comps if "hud" not in s.tags}
        start["_sel"] = model.selected_now(comps)
        # (the held thing's own border, black now, is not in the way; nor is the pen)
        # (a border is of the thing's own form — no larger than it: a hatching
        # the core stands in holds its box too, and is no part of the thing)
        border_ = [s for s in comps if s.sid < 1000 and within(hc.bbox, s.bbox) and s.sid != hc.sid
                   and (s.bbox[2] - s.bbox[0] + 1) <= cr["dims"][0] and (s.bbox[3] - s.bbox[1] + 1) <= cr["dims"][1]]
        # (the target's own parts — the pen's border and fill — are no
        # obstacle; the things already placed on it, of the carried form, are)
        crate_ids_ = {c_.sid for c_ in crates_now()} | {e_.sid for e_ in extra_}
        solids1 = {s.sid for s in comps if "terrain" not in s.tags and "hud" not in s.tags and s.size > 2 and s.sid != hc.sid
                   and s not in border_ and not (on_target(s.bbox) and s.sid not in crate_ids_ and (s.sid // 1000 if s.sid >= 1000 else s.sid) not in crate_ids_)}
        pc = ((p_core[0] + p_core[2]) / 2.0, (p_core[1] + p_core[3]) / 2.0)
        for b_ in border_:
            model.bound.add((me.sid, b_.sid))
        # (the cells aimed at — no obstacle to the plan — are the PLACE, not
        # the whole target: what already stands on the target is in the way)
        place_cells_ = frozenset((r, c) for r in range(place[0], place[2] + 1) for c in range(place[1], place[3] + 1))
        succ1 = make_successors(model, comps, solids1, keys, me.sid, pc, frame=model._frame_now, target_cells=place_cells_)

        def at_place(st, _t=hc.sid, _p=p_core):
            cs = st.get(_t)
            return bool(cs) and _bbox(cs) == _p

        def dist1(st, _t=hc.sid, _pc=pc):
            cs = st.get(_t)
            if not cs:
                return 999.0
            c = centre(cs)
            return abs(c[0] - _pc[0]) + abs(c[1] - _pc[1])
        p1 = plan_general(start, succ1, at_place, heuristic=lambda st: dist1(st) / 3.0, max_states=40000, max_depth=40)
        note(f"carry: the held thing to its place {place}: plan {'none' if p1 is None else len(p1)} steps")
        if verbose:
            print(f"   carry: held core {hc.bbox} -> {p_core}: plan {p1}")
            if _ENV.get("CARRY_DEBUG"):
                print(f"   carry: solids1 {sorted(solids1)} border {[b_.sid for b_ in border_]} overlap_ok(me) {[sorted(pr_) for pr_ in model.fixed.overlap_ok if me.sid in pr_]} overlap_ok(hc) {[sorted(pr_) for pr_ in model.fixed.overlap_ok if hc.sid in pr_]} passable {sorted(model.passable)} flat {sorted(model.flat)} ground {sorted(model.ground_ok)} bound {model.bound}")
                if p1:
                    st_ = start; make_successors.trace = True
                    try:
                        for k_ in p1:
                            nx_ = next((n_ for a_, n_ in succ1(st_) if a_ == k_), None)
                            if nx_ is None:
                                break
                            st_ = nx_
                    finally:
                        make_successors.trace = False
            if p1 is None:
                print(f"   carry: bodies {model.bodies()} bound {model.bound} active {model.active}; start succ {[(a_, _bbox(n_[hc.sid]) if hc.sid in n_ else None, _bbox(n_[me.sid]) if me.sid in n_ else None) for a_, n_ in succ1(start)]} whys {getattr(make_successors, 'whys', None)} last {getattr(plan_general, 'last', None)}")
                make_successors.trace = True
                try:
                    _ = succ1(start)
                finally:
                    make_successors.trace = False
                for a1_, n1_ in succ1(start):
                    s2_ = succ1(n1_)
                    print(f"      after {a1_}: body {_bbox(n1_[me.sid])} core {_bbox(n1_[hc.sid]) if hc.sid in n1_ else None} -> {[(a2_, _bbox(n2_[me.sid])) for a2_, n2_ in s2_]} whys {getattr(make_successors, 'whys', None)}")
                # (every reachable state: where the body stood and what each key answered)
                seen_ = {}; fr_ = [start]
                while fr_ and len(seen_) < 400:
                    st_ = fr_.pop()
                    bb_ = _bbox(st_[me.sid])
                    if bb_ in seen_:
                        continue
                    sc_ = succ1(st_)
                    seen_[bb_] = (sorted(a_ for a_, _ in sc_), dict(getattr(make_successors, 'whys', {}) or {}))
                    fr_.extend(n_ for _, n_ in sc_)
                for bb_, (acts_, whys_) in sorted(seen_.items()):
                    print(f"      at {bb_}: {acts_} {whys_}")
                f_ = np.asarray(model._frame_now)
                for r_ in range(0, 24):
                    print("      %2d " % r_ + "".join("%x" % f_[r_, c_] for c_ in range(16, 48)))
                print(f"   carry: selection then: sel {start.get('_sel')} select_color {model.select_color} selected_colour {model.selected_colour} select_kinds {model.select_kinds} ctx {[(k_, s_, c_.presses.get(k_, 0), dict(c_.moving(k_))) for (k_, s_), c_ in model.fixed.ctx.items()]} moving4 {model.fixed.moving('ACTION4', start.get('_sel'))}")
                print(f"   carry: laws on the group: {[(k_, {sid_: e_ for sid_, e_ in model.fixed.moving(k_).items() if sid_ in (me.sid, hc.sid) or sid_ in {b_.sid for b_ in border_} or (sid_ >= 1000 and sid_ // 1000 in (me.sid, hc.sid))}) for k_ in keys]} hc S{hc.sid} me S{me.sid} border {[b_.sid for b_ in border_]} parts_of_me {[s.sid for s in comps if s.sid >= 1000 and s.sid // 1000 == me.sid]}")
        # (the bonds stay while the thing is carried: its motion is the
        # body's, not a law of its own, and what it lies over it passes —
        # not the body; they are let go of with the thing)
        if p1 is None:
            # AS FAR AS I CAN: no way to its place with the thing in hand —
            # the search's best state is the nearest the thing can be
            # brought; it is brought there and let go.  Beyond that is
            # where a helper reaches (an agent that advances the goal takes
            # what is set down nearest the target); with no helper, the
            # thing waits there and nothing is lost.
            last_ = getattr(plan_general, "last", None) or {}
            best_ = last_.get("best")
            d_now = dist1(start)
            cr["short"] = cr.get("short", 0) + 1       # (a strait in this level: the next thing is taken from behind)
            if best_ is not None:
                path_ = best_[1] if best_[0] * 3.0 < d_now - 3.0 else []
                note(f"carry: no way to the place with the thing in hand; it is brought as far as it goes ({len(path_)} steps, {d_now:.0f} -> {best_[0] * 3:.0f} cells from its place) and let go there")
                if verbose:
                    print(f"   carry: as far as it goes: {path_ or 'here'}")
                for k in path_[:40]:
                    step(k)
                    if get_obs().levels_completed > lv0:
                        return True
                step(K)
                if get_obs().levels_completed > lv0:
                    return True
                for b_ in border_:
                    model.bound.discard((me.sid, b_.sid))
                model.bound.discard((me.sid, hc.sid))
                me = body_now()
                handed_ = cr.setdefault("handed", [])
                near_ = [t_ for t_ in crates_now() if me is not None and abs(t_.centre[0] - me.centre[0]) + abs(t_.centre[1] - me.centre[1]) <= 8]
                if near_:
                    t_ = min(near_, key=lambda t: abs(t.centre[0] - me.centre[0]) + abs(t.centre[1] - me.centre[1]))
                    if t_.bbox not in handed_:
                        handed_.append(t_.bbox)
                cr["_we_placed"] = False
                continue
            step(K)                     # (let go; the place is given up for now)
            for b_ in border_:
                model.bound.discard((me.sid, b_.sid))
            model.bound.discard((me.sid, hc.sid))
            return False
        for k in p1[:40]:
            me_b = body_now()
            b_was = me_b.bbox if me_b is not None else None
            step(k)
            if get_obs().levels_completed > lv0:
                return True
            # (a press the world refused where the law said the body moves
            # is remembered BY PLACE — the plan routes round it, the law
            # stands: the executor's rule, kept here too)
            me_a = body_now()
            e_ = model.fixed.moving(k).get(me_b.sid) if me_b is not None else None
            if me_a is not None and b_was is not None and me_a.bbox[:2] == b_was[:2] and e_ and e_[0] == "translate" and (e_[1][0] or e_[1][1]):
                model.refuse((kind(me_b), b_was[0], b_was[1], tuple(e_[1])), size=(b_was[2] - b_was[0] + 1, b_was[3] - b_was[1] + 1))
                note(f"carry: {k} refused at {b_was[:2]} — remembered by place; replanning")
                if verbose:
                    print(f"   carry: {k} refused at {b_was[:2]}: remembered by place ({len(model.refused_at)} places)")
                break
        me = body_now(); hc2 = held_core(me)
        arrived_ = (hc2 is not None and hc2.bbox == p_core) or any(c_.bbox == p_core for c_ in cores_now())
        if not arrived_:
            if verbose:
                print(f"   carry: the held core stands at {hc2.bbox if hc2 else None}, not {p_core}: replanning")
                if _ENV.get("CARRY_DEBUG"):
                    f_ = np.asarray(model._frame_now); mb_ = me.bbox if me is not None else hc.bbox
                    for r_ in range(max(0, mb_[0] - 8), min(f_.shape[0], mb_[2] + 9)):
                        print("      %2d " % r_ + "".join("%x" % f_[r_, c_] for c_ in range(max(0, mb_[1] - 12), min(f_.shape[1], mb_[3] + 16))))
                    print(f"      events then: {[(e.kind, e.sid) for e in tr.last_events][:10]}")
                    print(f"      body laws then: {[(k_, model.fixed.eff[k_].get(me.sid), model.fixed.support[k_].get(me.sid), dict(model.fixed.alt[k_][me.sid]), [c_ for c_ in model.fixed.contradicted.get(k_, ()) if c_ == me.sid or (isinstance(c_, tuple) and me.sid in c_)]) for k_ in keys]} refused_at {[r_ for r_ in getattr(model, 'refused_at', ()) if r_[0] == kind(me)][:6]}")
                    print(f"      on the target then: {[(s.sid, s.body, s.size, s.bbox, s.sid in solids1) for s in comps if within(s.bbox, tb) or (s.bbox[0] <= tb[2] and tb[0] <= s.bbox[2] and s.bbox[1] <= tb[3] and tb[1] <= s.bbox[3])]} crate ids {sorted(crate_ids_)}")
            continue
        # let go: the key that bound it (a bind key is presumed to unbind;
        # what the next round sees says whether it did)
        step(K)
        cr["_we_placed"] = True
        for b_ in border_:
            model.bound.discard((me.sid, b_.sid))
        model.bound.discard((me.sid, hc.sid))
        if get_obs().levels_completed > lv0:
            return True
    return get_obs().levels_completed > lv0


def leap_protocol(tr, model, step, get_obs, lv0, verbose, note):
    """REMOVE ALL OF THE KIND BY LEAPS (lf52): the leap law read, the
    board is the lattice of the kind's things and the tiles they stand
    among; the plan leaves the fewest of the kind; each leap is two
    clicks (select, then point) and is checked by the count."""
    lp = model.leap
    kind, pitch = lp["kind"], lp["pitch"]
    for attempt in range(4):
        things = list(tr.prev)
        pegs_now = [s for s in things if s.sid < 1000 and (s.body, s.size) == kind and "hud" not in s.tags]
        if len(pegs_now) < 2:
            return False
        ref = pegs_now[0].centre
        cells, pegs = leap_lattice(things, kind, pitch, ref)
        if len(pegs) < 2:
            return False
        path = plan_leaps(cells, pegs)
        if verbose:
            print(f"   leap plan: {len(pegs)} of the kind on {len(cells)} cells -> {len(path)} leaps {path[:8]}")
        if not path:
            return False
        note(f"leaps: {len(pegs)} of the kind, a plan of {len(path)} leaps leaves {len(pegs) - len(path)}")
        ok = True
        for src_, dst_ in path:
            n0 = sum(1 for s in tr.prev if s.sid < 1000 and (s.body, s.size) == kind and "hud" not in s.tags)
            p_src = (int(round(ref[0] + src_[0] * pitch)), int(round(ref[1] + src_[1] * pitch)))
            p_dst = (int(round(ref[0] + dst_[0] * pitch)), int(round(ref[1] + dst_[1] * pitch)))
            step(("click", p_src))
            if get_obs().levels_completed > lv0:
                return True
            step(("click", p_dst))
            if get_obs().levels_completed > lv0:
                return True
            n1 = sum(1 for s in tr.prev if s.sid < 1000 and (s.body, s.size) == kind and "hud" not in s.tags)
            if n1 != n0 - 1:
                if verbose:
                    print(f"   leap: {src_} -> {dst_} did not remove one ({n0} -> {n1}); replanning")
                ok = False
                break
        if ok:
            return get_obs().levels_completed > lv0
    return False


def program_protocol(tr, model, step, get_obs, lv0, verbose, hyps, note, bg):
    """THE PANEL WRITES THE JOURNEY (tn36).  A launch sent a thing on a
    journey and put it back; clicks flip marks in a strip of slots.  The
    slots are the journey's steps in order (a row is a sequence; a
    legend learned by experiment): the reading that names the journeyed
    thing says where it must go; the journey wanted is composed of the
    steps known, the slots are set to write it, and the launch pressed.
    A launch that shows another journey refines the legend; a few tries.
    Returns True when the level was won."""
    if not model.journeys or not model.toggle_cells:
        return False
    launch_key = model.journeys[-1][0]
    movers = set()
    for k_, jj, _ in model.journeys:
        if k_ == launch_key:
            movers |= set(jj)
    if len(movers) != 1:
        return False
    sid = next(iter(movers))
    for attempt in range(5):
        f = model._frame_now
        s = tr.get(sid)
        if s is None:
            return False
        try:
            # (the panel as a lattice first — a row of slots is the lattice with one
            # row; the strip reader stays as the fallback)
            cols_ = set(model.flips) | set(model.flips.values())
            panel = read_lattice(f, sorted(model.toggle_cells), bg, colours=cols_)
            if panel is None:
                panel = read_panel(f, sorted(model.toggle_cells), bg)
            elif verbose:
                print(f"   program: panel read as a lattice {panel['grid']} at pitch {panel['pitch']}: {len(panel['slots'])} slots")
        except Exception as ex_:
            if verbose:
                print(f"   program: panel reading failed {ex_!r}")
            return False
        if panel is None:
            if verbose:
                print("   program: no panel read from the flipped marks")
            return False
        fcol = panel["frame_colour"]
        # the legend, from every launch seen: the slots' forms then, the journey then
        known = {}
        for k_, jj, f_b in model.journeys:
            if k_ != launch_key or sid not in jj:
                continue
            forms_b = [slot_form(f_b, sb, fcol) for sb in panel["slots"]]
            assign = correspond(forms_b, jj[sid], known)
            if assign is None and len(panel["slots"]) > len(jj[sid]):
                assign = correspond(forms_b, jj[sid], known, prefix=True)
                if assign is not None and verbose:
                    print(f"   program: the journey read as a prefix of the slots' steps ({len(jj[sid])} of {len(panel['slots'])} slots)")
            if assign is None:
                if verbose:
                    print(f"   program: the launches do not agree on a legend ({[sorted(x)[:1] for x in forms_b]} -> {jj[sid]})")
                return False
            known.update(assign)
        steps_known = {fm: v for fm, v in known.items() if v is not None}
        none_forms = [fm for fm, v in known.items() if v is None]
        if not steps_known or not none_forms:
            if verbose:
                print(f"   program: legend incomplete {[(sorted(k)[:1], v) for k, v in known.items()]}")
            # THE LEGEND LEARNED BY EXPERIMENT: a legend with one form knows one
            # step; to learn another, change one slot's form (flip a mark in it)
            # and launch — the journey then says what the new form means.  The
            # first slot is changed; its first mark flipped; a few tries.
            if attempt < 4 and model.flips:
                sb0 = panel["slots"][0]
                marks0 = [(r_, c_) for r_ in range(sb0[0], sb0[2] + 1) for c_ in range(sb0[1], sb0[3] + 1)
                          if 0 <= r_ < 64 and 0 <= c_ < 64 and int(f[r_, c_]) in model.flips]
                if not marks0:
                    return False
                pt_ = marks0[0]
                note(f"the legend knows one form: changing slot 1 (a mark flipped at {pt_}) and launching to learn another")
                if verbose:
                    print(f"   program: experiment — flip a mark at {pt_} in slot 1, then launch {launch_key}")
                step(("click", (int(pt_[0]), int(pt_[1]))))
                if get_obs().levels_completed > lv0:
                    return True
                step(launch_key if isinstance(launch_key, str) else ("click", tuple(launch_key[1])))
                if get_obs().levels_completed > lv0:
                    return True
                continue
            return False
        none_form = min(none_forms, key=len)
        # WHERE IT MUST GO: the reading that names the journeyed thing as mover
        body_cells = frozenset(q for q, col in s.cells.items() if col == s.body) if isinstance(s.cells, dict) else frozenset(s.cells)
        cands = []
        for h in hyps:
            if h.m_color != s.body or h.refuted:
                continue
            hc = h.m_start
            if abs(hc[0] - s.centre[0]) + abs(hc[1] - s.centre[1]) > 3:
                continue
            tc = h.target_centre()
            t_cells = frozenset(h.target) if not isinstance(h.target, tuple) else frozenset(getattr(h, "t_cells", ()) or ())
            cands.append((h, tc, t_cells))
        if not cands:
            if verbose:
                print("   program: no reading names the journeyed thing")
            return False
        # the journeys achievable: sums of the known steps over the slots
        n_slots = len(panel["slots"])
        wants = []
        for h, tc, t_cells in cands:
            d0 = (tc[0] - s.centre[0], tc[1] - s.centre[1])
            pr_ = panel["pitch"] if isinstance(panel["pitch"], int) else max(panel["pitch"])
            for dr in range(-pr_, pr_ + 1):
                for dc in range(-pr_, pr_ + 1):
                    want = (int(round(d0[0])) + dr, int(round(d0[1])) + dc)
                    prog = compose(steps_known, want, n_slots, none_form)
                    if prog is None:
                        continue
                    moved = frozenset((r + want[0], c + want[1]) for r, c in body_cells)
                    if t_cells:
                        bb_t = _bbox(t_cells); bb_m = _bbox(moved)
                        inside = bb_t[0] <= bb_m[0] and bb_t[1] <= bb_m[1] and bb_m[2] <= bb_t[2] and bb_m[3] <= bb_t[3]
                        overlap = len(moved & t_cells)
                        fit = 0 if (inside and overlap == 0) else (1 if overlap >= 0.8 * len(moved) else 2)
                    else:
                        fit = 1
                    off = abs(want[0] - d0[0]) + abs(want[1] - d0[1])
                    wants.append((fit, off, -h.prio, len([p for p in prog if p != none_form]), want, prog, h))
        if not wants:
            if verbose:
                print(f"   program: no journey of the known steps {list(steps_known.values())} reaches a reading's target")
            return False
        wants.sort(key=lambda w: w[:4])
        fit, off, _, _, want, prog, h = wants[0]
        note(f"the panel writes the journey: {len(steps_known)} step form(s) known; setting the slots for a journey of {want} to {h.note[:50]}")
        if verbose:
            print(f"   program: want {want} fit {fit} off {off:.1f} program {[steps_known.get(p) for p in prog]} for {h.note[:60]}")
        clicks = []
        for i, fm in enumerate(prog):
            cl = clicks_to_set(f, panel, i, fm, model.flips)
            if cl is None:
                if verbose:
                    print(f"   program: slot {i} cannot be set to that form by the flips known {model.flips}")
                return False
            clicks += cl
        for r, c in clicks:
            step(("click", (r, c)))
            if get_obs().levels_completed > lv0:
                return True
        n_j = len(model.journeys)
        step(launch_key)
        if get_obs().levels_completed > lv0:
            return True
        if len(model.journeys) == n_j:
            # (the launch showed no journey: the thing went and stayed, or nothing went)
            if verbose:
                print("   program: the launch showed no journey this time")
            return False
    return False


def paint_protocol(env, tr, model, keys, step, get_obs, lv0, verbose, budget, bg, budget_left=lambda: None):
    """THE PAINT PROTOCOL — the EQUAL reading.  A key that recoloured a
    big thing in place is a PAINT; a static picture of the same dims is
    the REFERENCE the painted thing (the CANVAS) must be made equal to.
    WHERE the paint lands may depend on where the controlled thing
    stands, and WHAT colour it lays may be handed to it by a click on a
    palette thing — both are learned by asking: the movement keys walk
    the controlled thing over its places (a graph of positions told apart
    by the cells the movement keys ever changed), the paint key is tried
    once at each new place, colour clicks are tried when a colour is
    needed.  Then the greedy plan: while canvas != reference, the place
    and colour that fix the most cells for the least damage."""
    n = 0
    votes = Counter()
    for k, lst in model.paints.items():
        for sid, _, _ in lst:
            votes[(k, sid)] += 1
    if not votes:
        return False, 0
    (pkey, csid), _ = votes.most_common(1)[0]
    canvas = tr.get(csid)
    if canvas is None or canvas.size < 16:
        return False, 0             # a painted cell is a toggle, not a canvas (16: a 4x4 picture at least)
    cbox = canvas.bbox
    changed = set(model.change_presses) | set(model.moved_count) | {csid}
    ref = find_reference(tr, canvas, exclude=changed)
    if ref is None:
        if verbose:
            print(f"   paint: canvas S{csid} {canvas.dims} but no reference")
        return False, 0
    rmap, rgrp = ref
    cm0 = {(r - cbox[0], cc - cbox[1]): col for (r, cc), col in canvas.cells.items()}
    if all(cm0.get(q) == rmap[q] for q in rmap) or len(set(rmap.values())) < 2 and set(rmap.values()) <= set(cm0.values()):
        if verbose:
            print(f"   paint: the picture S{[t.sid for t in rgrp]} shows nothing the canvas lacks: no equal reading")
        return False, 0
    pkeys = set(model.paints)
    mkeys = [k for k in keys if k not in pkeys and k not in model.undo_keys]
    ccells = set(canvas.cells)
    hud_cells = {q for s in tr.prev if "hud" in s.tags or "bar" in s.tags for q in s.cells}
    ref_cells = {q for t in rgrp for q in t.cells}
    if verbose:
        print(f"   paint: key {pkey} canvas S{csid}@{cbox} reference {[(t.sid, t.body) for t in rgrp]} colours {sorted(set(rmap.values()))} movement keys {mkeys}")
    model.paint_model = {"key": pkey, "canvas": sorted({int(c) for c in canvas.colors}), "reference": sorted({int(c) for c in rmap.values()}),
                         "move_keys": [], "ctrl": set(), "palette": [], "target": None, "confirmed": False}

    def cur_frame():
        return np.array(get_obs().frame)[-1]

    def won():
        return get_obs().levels_completed > lv0

    def canvas_now():
        c = tr.get(csid)
        if c is None:
            c = next((s for s in tr.prev if s.sid < 1000 and s.bbox == cbox), None)
        return c

    def cmap():
        c = canvas_now()
        if c is None:
            fr = cur_frame()
            return {(r - cbox[0], cc - cbox[1]): int(fr[r, cc]) for (r, cc) in ccells}
        return {(r - cbox[0], cc - cbox[1]): col for (r, cc), col in c.cells.items()}
    # --- the places: a graph over signatures
    mov = set()
    frames = []
    edges = {}
    regions = {}

    def sig(fr):
        return frozenset(q for q in mov if fr[q] != bg)

    def node_of(fr):
        s = sig(fr)
        for i, f0 in enumerate(frames):
            if sig(f0) == s:
                return i
        frames.append(fr.copy())
        return len(frames) - 1

    def press(k):
        nonlocal n
        fb = cur_frame()
        step(k); n += 1
        fa = cur_frame()
        return fb, fa

    def move_press(k):
        note(f"paint: press {k} — walking the controlled thing to another place (place {node} so far)")
        fb, fa = press(k)
        d = {(int(r), int(c)) for r, c in np.argwhere(fa != fb)} - hud_cells - ccells
        mov.update(d)
        if verbose:
            print(f"   paint: press {k} at place {node}: {len(d)} cells changed -> place {node_of(fa)}")
        if d:
            model.paint_model["ctrl"] |= {int(fb[q]) for q in d if fb[q] != bg} | {int(fa[q]) for q in d if fa[q] != bg}
            if k not in model.paint_model["move_keys"]:
                model.paint_model["move_keys"].append(k)
        return fa
    # --- the colour in hand
    sources = {}
    tsid = model.colour_targets.most_common(1)[0][0] if model.colour_targets else None
    tcells = [None]                 # the target's cells a colour click recolours, from its corner
    for rec in model.colour_clicks:
        click_, _, host, sid_, col, cells_ = rec
        sources[col] = click_
        if sid_ == tsid:
            tcells[0] = cells_
        if host not in model.paint_model["palette"]:
            model.paint_model["palette"].append(host)
    model.paint_model["target"] = tsid
    # THE COLOUR IN HAND is a state: set by a colour click, shown by a
    # paint; between the two nothing changes it.  The thing that holds it
    # (whose identity a turning basket may lose) only seeds it
    hand = [None]

    def colour_now():
        if hand[0] is not None:
            return hand[0]
        t = tr.get(tsid) if tsid is not None else None
        if t is not None:
            if tcells[0]:
                r0, c0 = t.bbox[:2]
                cs = Counter(t.cells[(r0 + r, c0 + c)] for r, c in tcells[0] if (r0 + r, c0 + c) in t.cells)
                if cs:
                    return cs.most_common(1)[0][0]
            return t.body
        # (the holder's identity lost — a turning basket: the colour in
        # hand is what the moving thing shows of the picture's colours)
        if mov:
            fr = cur_frame()
            cs = Counter(int(fr[q]) for q in mov if 0 <= q[0] < 64 and 0 <= q[1] < 64 and int(fr[q]) != bg and int(fr[q]) in set(rmap.values()))
            if cs:
                return cs.most_common(1)[0][0]
        return None

    def palette_candidates(col):
        out = []
        for s in tr.prev:
            if s.sid >= 1000 or s.size > 60 or "hud" in s.tags or "terrain" in s.tags or s.sid in changed or s.sid == csid:
                continue
            if set(s.cells) & ref_cells or set(s.cells) & mov:
                continue
            pts = [q for q, c_ in s.cells.items() if c_ == col]
            if pts:
                cen = s.centre
                out.append(min(pts, key=lambda q: abs(q[0] - cen[0]) + abs(q[1] - cen[1])))
        return out

    def set_colour(col):
        nonlocal tsid
        if colour_now() == col:
            return True
        tries = ([sources[col]] if col in sources else []) + [p for p in palette_candidates(col) if p != sources.get(col)]
        for pt in tries[:3]:
            n_rec = len(model.colour_clicks)
            note(f"paint: click at {pt} to take colour {col} in hand")
            press(("click", pt))
            hand[0] = None                  # (what the click did is read from the holder, not remembered)
            if len(model.colour_clicks) > n_rec:
                rec = model.colour_clicks[-1]
                if tsid is None:
                    tsid = rec[3]; model.paint_model["target"] = tsid
                if rec[3] == tsid:
                    tcells[0] = rec[5]
                sources[rec[4]] = rec[0]
                hand[0] = rec[4]
                if rec[2] not in model.paint_model["palette"]:
                    model.paint_model["palette"].append(rec[2])
            elif pt == sources.get(col):
                hand[0] = col           # a known source that showed no change: it already held the colour
            if colour_now() == col:
                return True
        return False

    def colours():
        out = set(sources) | {c for c in (colour_now(),) if c is not None}
        for col in set(rmap.values()):
            if col not in out and palette_candidates(col):
                out.add(col)
        return out

    def learn(node, before, after, col):
        """What a paint in colour col at this place taught: cells that
        changed are IN its region, cells of another colour that did not
        change are OUT, cells that already wore the colour stay UNKNOWN
        until a paint in another colour tells."""
        inn, out = regions.setdefault(node, [set(), set()])
        d = {q for q in rmap if before.get(q) != after.get(q)}
        inn |= d
        out |= {q for q in rmap if q not in d and before.get(q) != col}
        out -= inn
        return d

    def unknown(node):
        inn, out = regions[node]
        return {q for q in rmap if q not in inn and q not in out}

    def wanted_colour(before, node=None):
        """The colour to test a place in: AN EXPERIMENT MUST BE ABLE TO
        SHOW SOMETHING — first the colour a pour from here would change
        the most cells with (where the prior expects the pour to land;
        the whole canvas when it expects nothing), then, among those, the
        colour most of the mismatched cells want (the test also moves the
        canvas toward the picture).  (The invariance gauge found the old
        choice — the picture's want alone, ties by colour number — pouring
        a colour the near half already wore under a recolouring: a pour
        that changed nothing was then read as the prior failing.)"""
        need = Counter(rmap[q] for q in rmap if before.get(q) != rmap[q])
        cols_ = [c_ for c_ in colours() if c_ is not None]
        if not cols_:
            return None
        where = predicted(node) if node is not None else None
        if not where:
            where = set(rmap)
        order = list(sources)                          # (the order the palette was found in: not a colour's number)

        def gain(c_):
            return sum(1 for q in where if before.get(q) != c_)
        best = max(cols_, key=lambda c_: (gain(c_), need.get(c_, 0), -(order.index(c_) if c_ in order else len(order))))
        return best if (gain(best) > 0 or need) else None

    def region_test(node):
        """Press the paint key here, in the colour the mismatched cells want
        most, and learn the place's region."""
        before = cmap()
        col = colour_now()
        want = wanted_colour(before, node)
        if want is not None and col != want:
            if set_colour(want):
                col = want
                before = cmap()
        note(f"paint: testing place {node} — press {pkey} in colour {col}: which canvas cells does it paint from here?")
        fb, fa = press(pkey)
        after = cmap()
        d0 = [q for q in rmap if before.get(q) != after.get(q)]
        if not d0 and col is not None:
            # NOTHING CHANGED: the pour landed on cells already of its colour
            # (or nowhere) — that says nothing about where it lands; one
            # retest in another colour before anything is concluded
            others = [c_ for c_ in colours() if c_ is not None and c_ != col]
            if others:
                c2 = max(others, key=lambda c_: sum(1 for q in (predicted(node) or set(rmap)) if before.get(q) != c_))
                if set_colour(c2):
                    col = c2
                    before = cmap()
                    fb, fa = press(pkey)
                    after = cmap()
                    d0 = [q for q in rmap if before.get(q) != after.get(q)]
        if d0 and after.get(d0[0]) is not None:
            col = after[d0[0]]          # the paint shows the colour in hand
        if col is None:
            # nothing showed and no colour is known: take a colour from the
            # palette (a click that hands one on) and paint again
            for c_ in sorted(set(rmap.values())):
                if set_colour(c_):
                    before = cmap()
                    fb, fa = press(pkey)
                    after = cmap()
                    d0 = [q for q in rmap if before.get(q) != after.get(q)]
                    col = after[d0[0]] if d0 else c_
                    break
        if col is None:
            return False
        hand[0] = col
        d = learn(node, before, after, col)
        if verbose:
            t_ = tr.get(tsid) if tsid is not None else None
            print(f"   paint: place {node} in colour {col}: {len(d)} cells in, {len(regions[node][1])} out, {len(unknown(node))} unknown; canvas before {Counter(before.values())} after {Counter(after.values())}; target S{tsid} {t_.bbox if t_ else None} {dict(t_.colors) if t_ else None} tags {t_.tags if t_ else None}")
        return True

    def plan_greedy(cm, cols):
        """Places and colours that fix the most cells for the least damage,
        on what is known of each region (an unknown cell may be in it:
        the paint that follows tells, and the record is corrected) —
        among equal gains, the place with the fewest unknown cells."""
        cm = dict(cm); plan = []
        for _ in range(16):
            if all(cm.get(q) == rmap[q] for q in rmap):
                return plan
            best = None
            for node, (inn, out) in regions.items():
                n_unk = sum(1 for q in rmap if q not in inn and q not in out)
                for col in cols:
                    fix = sum(1 for q in inn if rmap[q] == col and cm.get(q) != col)
                    dmg = sum(1 for q in inn if rmap[q] != col and cm.get(q) == rmap[q])
                    gain = fix - dmg
                    if gain > 0 and (best is None or (gain, -n_unk) > (best[0], -best[3])):
                        best = (gain, node, col, n_unk)
            if best is None:
                return None
            _, node, col, _ = best
            for q in regions[node][0]:
                cm[q] = col
            plan.append((node, col))
        return None

    def inverses():
        """Key pairs seen to undo each other somewhere ((a,k)->b and
        (b,k')->a): the prior that keys come in inverse pairs lets a way
        back be predicted where only the way there was walked."""
        inv = Counter()
        for (a, k), b in edges.items():
            if a == b:
                continue
            for (a2, k2), b2 in edges.items():
                if a2 == b and b2 == a and k2 != k:
                    inv[(k, k2)] += 1
        return inv

    def path_to(src, dst):
        from collections import deque
        inv = inverses()
        guessed = {}
        for (a, k), b in edges.items():
            if a == b:
                continue
            for (k1, k2), n_ in inv.items():
                if k1 == k and (b, k2) not in edges and (b, k2) not in guessed:
                    guessed[(b, k2)] = a
        prev = {src: None}; dq = deque([src])
        while dq:
            u = dq.popleft()
            if u == dst:
                break
            for (a, k), b in list(edges.items()) + list(guessed.items()):
                if a == u and b not in prev:
                    prev[b] = (u, k); dq.append(b)
        if dst not in prev:
            return None
        path = []
        while prev[dst] is not None:
            u, k = prev[dst]; path.append(k); dst = u
        return path[::-1]

    seen_left = [None]
    retested = set()

    def room():
        left = budget_left()
        if left is not None and seen_left[0] is not None and left > seen_left[0] + 5:
            return False            # the clock was wound back: the level restarted
        seen_left[0] = left
        return n < budget and (left is None or left > 2.5)
    ccen = ((cbox[0] + cbox[2]) / 2.0, (cbox[1] + cbox[3]) / 2.0)
    distrust = set()

    def pourer_centre(i):
        s_ = sig(frames[i])
        if not s_:
            return None
        return (sum(r for r, _ in s_) / len(s_), sum(c for _, c in s_) / len(s_))

    def predicted(i):
        b = pourer_centre(i)
        if b is None:
            return None
        m = (2 * ccen[0] - b[0], 2 * ccen[1] - b[1])
        out = set()
        for q in rmap:
            a_ = (q[0] + cbox[0], q[1] + cbox[1])
            if (a_[0] - b[0]) ** 2 + (a_[1] - b[1]) ** 2 <= (a_[0] - m[0]) ** 2 + (a_[1] - m[1]) ** 2:
                out.add(q)
        return out

    node = node_of(cur_frame())
    if not region_test(node):
        return False, n
    # THE POUR REACHES THE PART OF THE CANVAS NEAREST THE POURER (Tushar,
    # cd82: "a person reads off the picture what to paint where"): a
    # geometric prior — the cells nearer the pourer than its mirror image
    # through the canvas's centre (a half from a side, a corner's triangle
    # from a corner) — checked against the first pour, then trusted for
    # every place walked to, each pour verified by use.  The plan is over
    # LAYERED pours (a search: what is painted first is painted over),
    # not the greedy fix
    def region_of(i):
        """What a pour at place i paints: the region learned there when
        the prior failed there, else the prior's prediction."""
        if i in distrust and i in regions:
            return set(regions[i][0])
        p_ = predicted(i)
        if p_ is None and i in regions:
            return set(regions[i][0])
        return p_

    def plan_layered(cm, cols, max_pours=4):
        """The fewest pours (places and colours) whose layering makes the
        canvas the picture; among equal counts, the nearest places."""
        regs = {i: region_of(i) for i in range(len(frames))}
        regs = {i: r_ for i, r_ in regs.items() if r_}
        targets_ = {q: rmap[q] for q in rmap}
        best = [None]
        t0_ = time.time()

        def dist_(a_, b_):
            p_ = path_to(a_, b_)
            return len(p_) if p_ is not None else 99

        def rec(cm_, at, seq, budget_):
            if time.time() - t0_ > 8.0:
                return
            if all(cm_.get(q) == targets_[q] for q in targets_):
                cost = sum(dist_(node if k == 0 else seq[k - 1][0], seq[k][0]) for k in range(len(seq)))
                if best[0] is None or (len(seq), cost) < (len(best[0]), best[1]):
                    best[0] = list(seq); best[1] = cost
                return
            if budget_ == 0 or (best[0] is not None and len(seq) + 1 >= len(best[0])):
                return
            for i, r_ in regs.items():
                for col in cols:
                    if col is None:
                        continue
                    # (a pour must fix something the picture wants, or make way for one)
                    if not any(cm_.get(q) != col and targets_[q] == col for q in r_) and budget_ == 1:
                        continue
                    cm2 = dict(cm_)
                    for q in r_:
                        cm2[q] = col
                    if cm2 == cm_:
                        continue
                    rec(cm2, i, seq + [(i, col)], budget_ - 1)
        best.append(0)
        for depth_ in range(1, max_pours + 1):
            rec(dict(cm), node, [], depth_)
            if best[0] is not None:
                break
        return best[0]

    # the places are walked first (every key at every place, a few presses),
    # so the plan may name any of them
    arrived_by = None
    for _ in range(24):
        if not room() or won():
            break
        # (the key that undoes the one just pressed leads back: known
        # without pressing, once keys have been seen to come in pairs)
        inv_ = inverses()
        untried = [k for k in mkeys if (node, k) not in edges
                   and not (arrived_by is not None and inv_.get((arrived_by, k), 0) > 0)]
        if untried:
            k = untried[0]
            n0 = node
            fa = move_press(k)
            if won():
                return True, n
            nn = node_of(fa)
            edges[(node, k)] = nn
            arrived_by = k if nn != n0 else None
            node = nn
            continue
        targets = [i for i in range(len(frames)) if any((i, k) not in edges for k in mkeys)]
        best_ = None
        for t in targets:
            p = path_to(node, t)
            if p is not None and (best_ is None or len(p) < len(best_[1])):
                best_ = (t, p)
        if best_ is None:
            break
        for k in best_[1]:
            n0 = node
            fa = move_press(k)
            if won():
                return True, n
            nn = node_of(fa)
            arrived_by = k if nn != n0 else None
            if edges.get((node, k)) != nn:
                edges[(node, k)] = nn
                node = nn
                break
            node = nn
    # the prior against the first pour seen
    prior_ok = False
    for i, (inn, out) in regions.items():
        p_ = predicted(i)
        if p_ is not None and inn and inn <= p_ and not (out & p_):
            prior_ok = True
        elif p_ is not None and inn:
            distrust.add(i)
    if verbose:
        print(f"   paint: {len(frames)} places walked; prior {'holds' if prior_ok else 'fails'} on the first pour; pourer centres {[pourer_centre(i) for i in range(len(frames))]}")
    if prior_ok:
        note(f"paint: the pour reaches the part of the canvas nearest the pourer — the first pour agrees; {len(frames)} places walked; planning the layered pours from the picture")
        while room() and not won():
            cm = cmap()
            if all(cm.get(q) == rmap[q] for q in rmap):
                break
            cols_ = [c_ for c_ in colours() if c_ is not None]
            plan = plan_layered(cm, cols_)
            if verbose:
                print(f"   paint: layered plan {plan} (mismatch {sum(1 for q in rmap if cm.get(q) != rmap[q])}, colours {cols_})")
            if not plan:
                break
            pnode, pcol = plan[0]
            path = path_to(node, pnode)
            if path is None:
                break
            ok = True
            for k in path:
                fa = move_press(k)
                if won():
                    return True, n
                nn = node_of(fa)
                if edges.get((node, k)) != nn:
                    edges[(node, k)] = nn; ok = False
                node = nn
                if not ok:
                    break
            if not ok:
                continue
            if not set_colour(pcol):
                if verbose:
                    print(f"   paint: no way to colour {pcol}: sources {sources} candidates {palette_candidates(pcol)} in hand {colour_now()} tsid {tsid}")
                break
            before = cmap()
            expect = {q for q in region_of(node) if before.get(q) != pcol}
            note(f"paint: planned — at place {node} press {pkey} in colour {pcol} ({len(plan)} pour(s) planned from the picture)")
            press(pkey)
            if won():
                return True, n
            after = cmap()
            d0 = [q for q in rmap if before.get(q) != after.get(q)]
            if d0:
                pcol = after[d0[0]]
            hand[0] = pcol
            d = learn(node, before, after, pcol)
            if set(d) != expect:
                distrust.add(node)
                if verbose:
                    print(f"   paint: place {node} painted {len(d)} cells, {len(expect)} predicted: the prior fails there — its region is what was seen")
            mism_after = sum(1 for q in rmap if after.get(q) != rmap[q])
            if mism_after < sum(1 for q in rmap if before.get(q) != rmap[q]):
                model.paint_model["confirmed"] = True
        if won():
            return True, n
    while room() and not won():
        cm = cmap()
        if all(cm.get(q) == rmap[q] for q in rmap):
            break
        plan = plan_greedy(cm, colours())
        if verbose:
            print(f"   paint: mismatch {sum(1 for q in rmap if cm.get(q) != rmap[q])}, colour in hand {colour_now()}, colours {sorted(colours())}, plan {plan}, left {budget_left()}")
        if plan:
            pnode, pcol = plan[0]
            path = path_to(node, pnode)
            if path is None:
                plan = None
            else:
                ok = True
                for k in path:
                    fa = move_press(k)
                    if won():
                        return True, n
                    nn = node_of(fa)
                    if edges.get((node, k)) != nn:
                        edges[(node, k)] = nn
                        ok = False
                    node = nn
                    if not ok:
                        break
                if not ok:
                    continue
                if not set_colour(pcol):
                    if verbose:
                        print(f"   paint: no way to colour {pcol}")
                    break
                before = cmap()
                note(f"paint: planned — at place {node} press {pkey} in colour {pcol} (fixes the most cells for the least damage)")
                press(pkey)
                if won():
                    return True, n
                after = cmap()
                d0 = [q for q in rmap if before.get(q) != after.get(q)]
                if d0:
                    pcol = after[d0[0]]
                hand[0] = pcol
                d = learn(node, before, after, pcol)
                if not d:
                    # nothing changed where a fix was predicted: the record
                    # was wrong — what did not change is out
                    inn, out = regions[node]
                    wrong = {q for q in inn if before.get(q) != pcol}
                    inn -= wrong; out |= wrong
                mism_after = sum(1 for q in rmap if after.get(q) != rmap[q])
                if mism_after < sum(1 for q in rmap if before.get(q) != rmap[q]):
                    model.paint_model["confirmed"] = True      # a planned paint brought the canvas nearer the picture
                if verbose:
                    print(f"   paint: at place {node} colour {pcol}: {len(d)} cells changed; mismatch now {mism_after}")
                continue
        # no plan from what is known: a known place whose UNKNOWN cells hold
        # most of what is wanted is tested again before a new place is
        # walked to (its unknowns wore the colour the test was in)
        mism = {q for q in rmap if cm.get(q) != rmap[q]}
        retest = None
        for i in regions:
            u = unknown(i) & mism
            if len(u) >= 0.5 * len(mism) and i not in retested:
                p_ = path_to(node, i)
                if p_ is not None and (retest is None or (len(u), -len(p_)) > retest[0]):
                    retest = ((len(u), -len(p_)), i, p_)
        if retest is not None:
            _, i, p_ = retest
            retested.add(i)
            ok = True
            for k in p_:
                fa = move_press(k)
                if won():
                    return True, n
                nn = node_of(fa)
                if edges.get((node, k)) != nn:
                    edges[(node, k)] = nn; node = nn; ok = False
                    break
                node = nn
            if ok:
                region_test(node)
            continue
        untried = [k for k in mkeys if (node, k) not in edges]
        if untried:
            # the keys that moved the thing elsewhere first (a key that
            # did nothing at two places is likelier to do nothing here)
            moved_n = Counter(k_ for (a_, k_), b_ in edges.items() if a_ != b_)
            still_n = Counter(k_ for (a_, k_), b_ in edges.items() if a_ == b_)
            untried.sort(key=lambda k_: (still_n[k_] - moved_n[k_], mkeys.index(k_)))
            k = untried[0]
            fa = move_press(k)
            if won():
                return True, n
            nn = node_of(fa)
            edges[(node, k)] = nn
            node = nn
            if node not in regions:
                region_test(node)
            continue
        # walk to the nearest place with an untried key
        targets = [i for i in range(len(frames)) if any((i, k) not in edges for k in mkeys)]
        best = None
        for t in targets:
            p = path_to(node, t)
            if p is not None and (best is None or len(p) < len(best[1])):
                best = (t, p)
        if best is None:
            # every place walked and still no plan: a region's UNKNOWN cells
            # may be the ones wanted — test that place again in a colour
            # those cells do not wear
            mism = {q for q in rmap if cm.get(q) != rmap[q]}
            cands = []
            for i in regions:
                u = unknown(i) & mism
                if u:
                    p_ = path_to(node, i)
                    if p_ is not None:
                        cands.append((-len(u), len(p_), i, p_))
            if not cands:
                if verbose:
                    print(f"   paint: every place known ({len(frames)}), no plan; regions {[(i, len(r[0])) for i, r in regions.items()]}")
                break
            cands.sort()
            _, _, i, p_ = cands[0]
            ok = True
            for k in p_:
                fa = move_press(k)
                if won():
                    return True, n
                nn = node_of(fa)
                if edges.get((node, k)) != nn:
                    edges[(node, k)] = nn; node = nn; ok = False
                    break
                node = nn
            if not ok:
                continue
            u = unknown(node)
            worn = Counter(cm.get(q) for q in u)
            cols_ = [c_ for c_ in colours() if c_ is not None]
            if not cols_:
                break
            c_ = min(cols_, key=lambda c: worn.get(c, 0))
            if not set_colour(c_):
                break
            before = cmap()
            note(f"paint: retesting place {node} in colour {c_} — its unknown cells may be the ones wanted")
            press(pkey)
            if won():
                return True, n
            after = cmap()
            d0 = [q for q in rmap if before.get(q) != after.get(q)]
            if d0:
                c_ = after[d0[0]]
            hand[0] = c_
            learn(node, before, after, c_)
            if verbose:
                print(f"   paint: place {node} retested in colour {c_}: {len(regions[node][0])} in, {len(unknown(node))} unknown")
            continue
        for k in best[1]:
            fa = move_press(k)
            if won():
                return True, n
            nn = node_of(fa)
            if edges.get((node, k)) != nn:
                edges[(node, k)] = nn
                node = nn
                break
            node = nn
    if verbose:
        print(f"   paint: graph {dict(edges)}")
        H_, W_ = canvas.dims
        for i, (inn, out) in regions.items():
            rows = ["".join("#" if (r, c) in inn else "." if (r, c) in out else "?" for c in range(W_)) for r in range(H_)]
            print(f"   paint: region {i}: " + " ".join(rows))
    model.paint_model["places"] = len(frames)
    model.paint_model["regions"] = {i: len(r[0]) for i, r in regions.items()}
    if won():
        model.paint_model["confirmed"] = True
    return won(), n


def core_report(model, tr, hyps, keys, won_hyp=None, comps=None):
    """The new core's beliefs in the gate's vocabulary (study/TRUTH_SCHEMA.md)."""
    import re
    # the things as they stood before the last action: a won level's
    # last frame, not the next level's first
    comps = comps if comps is not None else (getattr(play_level, "last", {}) or {}).get("comps") or tr.prev
    kind_of = dict(model.kind_seen)
    kind_of.update({s.sid: s.body for s in comps})
    ents, rules = [], []

    def ent(role, col):
        if col is not None:
            ents.append({"role": role, "colors": [int(col)]})
    # controlled: what the keys move (the active body, the marked thing's kind)
    ctrl_kinds = set()
    for sid in (model.active or ()):
        if sid in kind_of and sid < 1000:
            ctrl_kinds.add(kind_of[sid])
    if model.select_color is not None:
        for sid, col in selected_hosts(comps).items():
            if col == model.select_color and sid in kind_of:
                ctrl_kinds.add(kind_of[sid])
    for k, v in model.key_selected.items():
        for c in ctrl_kinds:
            rules.append({"action": k, "effect": "translate", "of": [int(c)], "when": "always"})
    for c in ctrl_kinds:
        ent("controlled", c)
    for c in model.select_kinds:
        ent("selector", c)
        rules.append({"action": "ACTION6", "effect": "select", "of": [int(c)], "when": f"click_on({int(c)})"})
    # a key that turns the selected body turns every kind that was selected
    for k in model.key_rotate:
        for c in sorted(ctrl_kinds | set(model.select_kinds) | model.kinds_worn_by_selected()):
            rules.append({"action": k, "effect": "rotate", "of": [int(c)], "when": "always"})
    for c in model.pushable:
        ent("mover", c)
        for k in model.key_selected:
            rules.append({"action": k, "effect": "push", "of": [int(c)], "when": f"blocked_by({int(c)})"})
    for c in model.blocking():
        ent("wall", c)
    for c, n in model.wall_kinds.items():
        ent("wall", c)
    for c in model.ground_ok:
        ent("floor", c)
    for c in model.flat:
        ent("floor", c)
    for sid, (c, rate) in model.counters.items():
        ent("budget", c)
    # displays outside the play: bars, strips and boxes tagged hud
    hud_like = Counter((s_.body, s_.shape) for s_ in comps if "hud" in s_.tags and s_.sid < 1000)
    for s_ in comps:
        if "hud" in s_.tags or "bar" in s_.tags:
            ents.append({"role": "hud", "colors": sorted({int(c_) for c_ in set(s_.cells.values())})})
            if "bar" in s_.tags:
                ents.append({"role": "budget", "colors": sorted({int(c_) for c_ in set(s_.cells.values())})})
            elif s_.sid < 1000 and hud_like[(s_.body, s_.shape)] >= 2 and s_.size <= 16:
                ents.append({"role": "budget", "colors": [int(s_.body)]})      # several like marks in the band: lives
    # a display (a panel the arrivals change) is shown outside the play;
    # the thing whose arrival changes it is a tool
    for dsid in getattr(model, "displays", {}) or {}:
        d_ = next((x_ for x_ in comps if x_.sid == dsid), None)
        if d_ is not None and d_.sid < 1000:
            ents.append({"role": "hud", "colors": sorted({int(c_) for c_ in set(d_.cells.values())})})
    for tk in getattr(model, "triggers", {}) or {}:
        if model.displays:
            ent("tool", tk)
    for kd in model.rel:
        law = model.relative_law(kd)
        if law:
            ent("controlled", kd)
            rules.append({"action": "ACTION6", "effect": "teleport" if law[0] == "teleport" else "translate", "of": [int(kd)], "when": "click_near"})
    # fixed effects: keys and buttons
    for k in model.fixed.keys():
        eff = model.fixed.moving(k)
        if isinstance(k, tuple):
            if k in model.rel_keys:
                continue
            pt = k[1]
            # the button is the thing clicked and the panel it sits in: every
            # non-terrain thing whose box holds the click point
            under = [s for s in comps if s.bbox[0] <= pt[0] <= s.bbox[2] and s.bbox[1] <= pt[1] <= s.bbox[3] and "terrain" not in s.tags and "hud" not in s.tags and s.size <= 200]
            bcols = sorted({int(c_) for s in under for c_ in set(s.cells.values())})
            bcol = min(under, key=lambda s: s.size).body if under else None
            if bcols:
                ents.append({"role": "button", "colors": bcols})
            act = "ACTION6"; when = f"click_on({bcol})" if bcol is not None else "click_on"
        else:
            act = k; when = "always"
        for sid, e in eff.items():
            col = kind_of.get(sid)
            if col is None or sid in model.counters:
                continue
            effect = {"translate": "translate", "extend": "grow", "retract": "shrink", "rotate": "rotate"}.get(e[0])
            if effect:
                # a big thing seen to grow or shrink once is a thing something
                # passed over, not a mover: a mover's record repeats
                if effect != "translate" and model.fixed.support[k][sid] < 2 and model.fixed.alt[k][sid][e] < 2:
                    continue
                worn = sorted({int(col)} | {int(c_) for c_ in model.kinds_worn.get(sid, ())})
                rules.append({"action": act, "effect": effect, "of": worn, "when": when})
                if not isinstance(k, tuple) and col not in ctrl_kinds:
                    ent("mover", col)
                elif isinstance(k, tuple):
                    ent("mover", col)
    # a track's button: the thing under a positional click; it moves
    # every kind it moved by slot
    for k in model.positional:
        kinds_ = sorted(int(c_) for c_ in model.pos_kinds.get(k, ()))
        if isinstance(k, tuple):
            pt = k[1]
            under = [s for s in comps if s.bbox[0] <= pt[0] <= s.bbox[2] and s.bbox[1] <= pt[1] <= s.bbox[3] and "terrain" not in s.tags and "hud" not in s.tags and s.size <= 200]
            bcols = sorted({int(c_) for s in under for c_ in set(s.cells.values())})
            bcol = min(under, key=lambda s: s.size).body if under else None
            if bcols:
                ents.append({"role": "button", "colors": bcols})
            act = "ACTION6"; when = f"click_on({bcol})" if bcol is not None else "click_on"
        else:
            act = k; when = "always"
        for c_ in kinds_:
            rules.append({"action": act, "effect": "translate", "of": [c_], "when": when})
            ent("mover", c_)
    # keys and clicks that did nothing: the schema's "nothing" rules
    # (a wall refuses every move: a key that translates the controlled
    # thing is refused at a wall colour whether or not that key was the
    # one seen refused)
    walls_ = bool(model.blocking())
    for k in keys:
        refused = any(isinstance(x, tuple) and x and x[0] == "refused" for x in model.fixed.contradicted.get(k, ()))
        moves_ctrl = any(sid_ in (model.active or ()) and e_[0] == "translate" for sid_, e_ in model.fixed.moving(k).items())
        if k not in model.undo_keys and (refused or (walls_ and moves_ctrl) or (not model.fixed.moving(k) and k not in model.key_selected)):
            rules.append({"action": k, "effect": "nothing", "of": [], "when": "always" if not model.fixed.moving(k) else "blocked_by"})
    if any(isinstance(k, tuple) and not model.fixed.moving(k) for k in model.fixed.presses):
        rules.append({"action": "ACTION6", "effect": "nothing", "of": [], "when": "always"})
    vlaw = getattr(model, "visit_law", None)          # what one arrival does to the display (rot1, mir+rot0, ...)
    for tk, lst in model.triggers.items():
        keys_ = [k_ for _, k_ in lst if k_ is not None and not isinstance(k_, tuple)]
        for dsid, key in lst[:1]:
            dcol = kind_of.get(dsid)
            if dcol is not None:
                ent("canvas", dcol)
                if vlaw is not None and keys_:
                    eff_ = "rotate" if vlaw.startswith("rot") else "recolour"
                    for k_ in sorted(set(keys_)):
                        rules.append({"action": k_, "effect": eff_, "of": [int(dcol)], "when": f"adjacent({int(tk)})"})
                else:
                    rules.append({"action": key if not isinstance(key, tuple) else "ACTION6", "effect": "recolour", "of": [int(dcol)], "when": f"adjacent({int(tk)})"})
                ent("button", tk)
    # the goal: the reading that won, else the first standing
    goal = None
    h = won_hyp or (hyps[0] if hyps else None)
    if h is not None:
        kind = "align" if h.kind == "align" else "all_collected" if h.kind == "collect" else "cover"
        if "socket" in h.note or "fits frame" in h.note or "HUD twin" in h.note:
            kind = "inside"
        if "constellation" in h.note and h.kind == "cover":
            # marks the mover covers (it is larger than their spread) are
            # covered; marks that frame a spot the mover fits are a socket
            tc = getattr(h, "t_cells", None)
            if tc:
                rs = [r for r, _ in tc]; cs_ = [c for _, c in tc]
                spread = (max(rs) - min(rs) + 1) * (max(cs_) - min(cs_) + 1)
                if h.m_size < spread * 0.9 and len(tc) <= h.m_size:
                    kind = "inside"
        if "reach the odd" in h.note or (h.m_color in ctrl_kinds and kind == "align"):
            kind = "reach"
        if "marks onto like marks" in h.note:
            kind = "align"          # one thing brought onto another by its marks
        mm = re.search(r" C(\d+)\[", h.note)
        tcol = int(mm.group(1)) if mm else None
        goal = {"kind": kind, "mover": [int(h.m_color)], "target": [tcol] if tcol is not None else []}
        ent("mover", h.m_color)
        if tcol is not None:
            ent("marker", tcol)
            if kind == "inside":
                ent("socket", tcol)
    if model.displays and model.triggers and getattr(model, "display_goal", None):
        d, r = model.display_goal
        if getattr(model, "display_won_by", None) == "reach" and ctrl_kinds:
            # the display had to show the reference AND the mover had to
            # reach it: the goal is the reaching (the picture is the key)
            goal = {"kind": "reach", "mover": sorted(int(c) for c in ctrl_kinds), "target": [int(r)]}
            ent("marker", r); ent("canvas", d)
        else:
            goal = {"kind": "equal", "mover": [int(d)], "target": [int(r)]}
            ent("reference", r); ent("canvas", d)
    # the paint protocol's reading: canvas, reference, the thing the
    # movement keys walk, the palette; paint = recolour of the canvas,
    # a palette click = recolour of the thing that holds the colour
    pm = getattr(model, "paint_model", None)
    if pm:
        if pm.get("confirmed"):
            goal = {"kind": "equal", "mover": pm["canvas"], "target": pm["reference"]}
        ents.append({"role": "canvas", "colors": pm["canvas"]})
        ents.append({"role": "reference", "colors": pm["reference"]})
        rules.append({"action": pm["key"], "effect": "recolour", "of": pm["canvas"], "when": "always"})
        ctrl = sorted(int(c) for c in pm["ctrl"])
        if ctrl:
            ents.append({"role": "controlled", "colors": ctrl})
            for k in pm["move_keys"]:
                rules.append({"action": k, "effect": "translate", "of": ctrl, "when": "always"})
        for host in pm["palette"]:
            hs = next((s_ for s_ in comps if s_.sid == host), None)
            hcols = sorted({int(c_) for c_ in hs.cells.values()}) if hs is not None else [int(kind_of[host])] if host in kind_of else []
            if hcols:
                ents.append({"role": "button", "colors": hcols})
                tcol = kind_of.get(pm["target"])
                rules.append({"action": "ACTION6", "effect": "recolour", "of": [int(tcol)] if tcol is not None else [], "when": f"click_on({hcols[0]})"})
    for click_, _, host, tsid_, col_, _ in model.colour_clicks:
        # (a colour handed on by a click, whether or not a paint followed)
        hs = next((s_ for s_ in comps if s_.sid == host), None)
        hcols = sorted({int(c_) for c_ in hs.cells.values()}) if hs is not None else []
        tcol = kind_of.get(tsid_)
        if hcols and tcol is not None:
            ents.append({"role": "button", "colors": hcols})
            rules.append({"action": "ACTION6", "effect": "recolour", "of": [int(tcol)], "when": f"click_on({hcols[0]})"})
    seen, out_e = set(), []
    for e in ents:
        kk = (e["role"], tuple(e["colors"]))
        if kk not in seen:
            seen.add(kk); out_e.append(e)
    seen, out_r = set(), []
    for r in rules:
        kk = (r["action"], r["effect"], tuple(r["of"]), r.get("when"))
        if kk not in seen:
            seen.add(kk); out_r.append(r)
    return {"entities": out_e, "rules": out_r, "goal": goal}


FLOOR_TURN = 150            # the floor's first turn on a level, in actions (hand-chosen); it doubles with every turn on the same level
FLOOR_TURN_MAX = 600        # the longest single turn (hand-chosen; a level won after 600 floor actions scores nothing under the quadratic rule — the turn is for reaching the next level, not for the score)
FLOOR_SHARE = 0.5           # of the actions left when a level began, the most the floor may spend on it while the core still has untried states (hand-chosen)
CORE_TRIES = 3              # the core's attempts on one level from different states (hand-chosen)
TURN_FRACTION = float(_ENV.get("TURN_FRACTION", "0.25"))        # a core turn may think for this fraction of the game's remaining clock, at most LEVEL_SECONDS (hand-chosen)


class Floor:
    """THE FLOOR: the crude loop as the standing policy of every level.
    Probe, the unique thing, the frontier, class-scored clicks, the death
    ledger, random known motion — it never stops and never thinks long,
    and on the hidden set that earns the score the core has no floor
    under (the post-mortem of 2026-09-22).  One instance per game: what
    it learns (known motions, click classes, confirmed goal kinds) is
    kept across levels, as it always was."""

    def __init__(self, seed=0):
        self.ag = None
        self.seed = seed

    def _ready(self, obs):
        from agent.mapper_agent import MapperAgent
        if self.ag is None:
            self.ag = MapperAgent(seed=self.seed)
            self.ag.on_episode_start(obs)
        elif obs.levels_completed != self.ag._levels:
            self.ag.on_level_up(obs)
        else:
            # (the same level, but the core has been at it since: re-sight
            # the level as after a restart — the map and the ledgers stay)
            self.ag.observe(GameAction.RESET, obs, True)

    def play(self, env, obs, n_max, deadline, since_reset, verbose=False, fresh=False):
        """Play from where the level stands until it is won, `n_max`
        actions are spent or the clock runs out.  `since_reset` = actions
        since the last RESET (two RESETs in a row throw the game back to
        its first level: the second is never sent).  `fresh`: a clean
        mind — the level read anew, nothing believed from before (what a
        level looked like when first seen mid-way may have misled it).
        Returns (won, actions, obs, since_reset)."""
        if fresh:
            self.ag = None
        self._ready(obs)
        ag = self.ag
        lv0 = obs.levels_completed
        n = 0
        self.trans = []                 # (key or click, click, frame before, frames after, state, levels): what the core watches
        while n < n_max and obs.levels_completed == lv0 and obs.state != GameState.WIN and time.time() < deadline:
            try:
                a = ag.act(obs)
            except Exception as ex_:
                if verbose:
                    print(f"   floor act failed: {ex_!r}")
                a = None
            if a is None or (a == GameAction.RESET and since_reset == 0):
                keys_ = [GameAction.from_id(i) for i in obs.available_actions if i not in (0, 6, 7)]
                a = keys_[n % len(keys_)] if keys_ else GameAction.ACTION6
                if a.is_complex():
                    a.set_data({"x": 32, "y": 32})
            data = a.action_data.model_dump() if a.is_complex() else None
            prev = np.array(obs.frame)
            obs2 = env.step(a, data=data)
            if obs2 is None:
                break
            obs = obs2; n += 1
            since_reset = 0 if a == GameAction.RESET else since_reset + 1
            try:
                click_ = (int(data["y"]), int(data["x"])) if data else None
                key_ = "RESET" if a == GameAction.RESET else (("click", click_) if click_ else a.name)
                self.trans.append((key_, click_, prev[-1] if prev.ndim == 3 else prev, np.array(obs.frame), obs.state, obs.levels_completed))
            except Exception:
                pass
            try:
                ag.observe(a, obs, not np.array_equal(prev, np.array(obs.frame)))
            except Exception as ex_:
                if verbose:
                    print(f"   floor observe failed: {ex_!r}")
            if obs.levels_completed > lv0:
                ag.on_level_up(obs)
                break
            if obs.state == GameState.GAME_OVER:
                obs = env.step(GameAction.RESET); n += 1; since_reset = 0
                try:
                    self.trans.append(("RESET", None, None, np.array(obs.frame), obs.state, obs.levels_completed))
                except Exception:
                    pass
                try:
                    ag.on_game_over(obs)
                except Exception as ex_:
                    if verbose:
                        print(f"   floor on_game_over failed: {ex_!r}")
        return obs.levels_completed > lv0, n, obs, since_reset


def watch_floor(st, trans, bg, lv0, verbose=False):
    """THE CORE WATCHES THE FLOOR.  Every press the floor made is an
    experiment the core did not choose, observed the same way its own
    are (the tracker follows the frames; what moved under which key,
    what a click did, what killed the body become laws) — so what the
    floor's play taught is the core's when it gets the level back, and
    carries to the next level.  Returns the things as they stood before
    the winning press (None if the floor did not win)."""
    model, tr = st["model"], st["tr"]
    comps_won = None
    st["journal"] = []              # (key, click, events, the things before, after): what the floor's presses did
    st["won_key"] = None
    model._journal = st["journal"]
    for key, click, f0, fr_all, state, lv in trans:
        try:
            before = tr.prev
            f = fr_all[-1] if fr_all.ndim == 3 else fr_all
            if SCENE:
                tr.update(f, bg); cur = with_derived(tr, model)
            else:
                cur = read_components(f, bg); tr.update(cur)
            if key == "RESET":
                continue
            if lv > lv0:
                comps_won = before          # (the level as it was before the winning press)
                st["won_key"] = (key, click)
                break
            if state != GameState.GAME_OVER:
                cur = _merged_mover(tr, model, key, before, cur)
            st["journal"].append((key, click, [(e.kind, e.sid) for e in tr.last_events], before, cur))
            gone_ = state == GameState.GAME_OVER and model.active and any(sid_ not in {x.sid for x in cur} for sid_ in model.active)
            if not gone_:
                scroll_ = next((e.data["vec"] for e in tr.last_events if e.kind == "scrolled"), None)
                model.observe(key, click, before, cur, frame=f0, frame_after=f, scroll=scroll_)
            if state == GameState.GAME_OVER:
                model.note_death(tr.prev, before=before, key=key if not isinstance(key, tuple) else None, frame=f0)
            else:
                model.note_alive(cur)
            model._frame_now = f
        except Exception as ex_:
            if verbose:
                print(f"   watching the floor failed on {key}: {ex_!r}")
    return comps_won


def _reading_won(hyps, things, model=None, extra_cols=()):
    """THE WIN, READ: which of the level's readings the winning state
    satisfied — a thing of the reading's mover kind, one that moved in the
    level, at, in, over or against its target.  The first (best) such
    reading, or None: a win no reading explains is a mechanic the
    vocabulary lacks (an M1 case)."""
    def bb(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return (min(rs), min(cs), max(rs), max(cs))
    active = set(getattr(model, "active", None) or ())
    moved = getattr(model, "moved_count", {}) or {}
    mcols = _mover_colours(model) | set(extra_cols)
    cell_col = {}
    for x in things:
        if x.sid < 1000 and "hud" not in x.tags:
            cell_col.update(x.cells)
    for h in hyps or []:
        try:
            # (A TARGET THAT WENT AWAY WAS NOT REACHED: the reading's target cells must
            # still show the target — or the mover covering it; a reading whose target
            # was itself pushed off is not what the win satisfied — tw03 L1)
            tcells0 = getattr(h, "t_cells", None) or frozenset()
            if tcells0:
                tcol = getattr(h, "t_color", None)
                if tcol is None:
                    m_ = re.search(r"C(\d+)\[", getattr(h, "note", "") or "")
                    tcol = int(m_.group(1)) if m_ else None
                there = [cell_col.get(q) for q in tcells0]
                if tcol is not None and sum(1 for v in there if v == tcol) < 0.5 * len(there):
                    if sum(1 for v in there if v == h.m_color) < 0.5 * len(there):
                        continue
                    # (the mover stands where the target was — and the target stands
                    # elsewhere, whole: it was pushed off, not reached)
                    if any(x.body == tcol and abs(x.size - (h.t_size or x.size)) <= 2 and not (set(x.cells) & set(tcells0)) for x in things if x.sid < 1000):
                        continue
            movers = [x for x in things if x.sid < 1000 and "hud" not in x.tags and x.body == h.m_color
                      and 0.5 * h.m_size <= x.size <= 2.5 * h.m_size
                      and (x.sid in active or moved.get(x.sid, 0) > 0 or x.body in mcols)]
            if not movers:
                continue
            tc = h.target_centre()
            tcells = getattr(h, "t_cells", None) or frozenset()
            tb = bb(tcells) if tcells else (tc[0], tc[1], tc[0], tc[1])
            tol = max(2.5, 0.5 * max(tb[2] - tb[0], tb[3] - tb[1]))
            # (a thing that never moved and stands on the target's own cells
            # is the target, not a mover that reached it)
            movers = [x for x in movers if not (tcells and x.sid not in active and moved.get(x.sid, 0) == 0 and x.body not in extra_cols
                                                and (x.bbox == tb or len(set(x.cells) & set(tcells)) >= 0.7 * x.size))]
            for m in movers:
                mc = m.centre
                if abs(mc[0] - tc[0]) <= tol and abs(mc[1] - tc[1]) <= tol:
                    return h                                        # at / in it
                if tcells and len(set(m.cells) & set(tcells)) >= 0.6 * len(tcells):
                    return h                                        # over it
                gap = max(tb[0] - m.bbox[2], m.bbox[0] - tb[2], tb[1] - m.bbox[3], m.bbox[1] - tb[3])
                if gap <= 1:
                    return h                                        # against it
        except Exception:
            continue
    return None


def _merged_mover(tr, model, key, before, cur):
    """IDENTITY BY LAW.  A mover that vanished on a press while a thing
    standing exactly where its law said it would go was recoloured to
    the mover's colour is that mover, arrived on the thing (the tracker
    read the merge as the thing changing colour and the mover gone).  The
    ids are put right so the laws — the move, the passing-through — are
    learned from it."""
    if isinstance(key, tuple) or not model.active:
        return cur
    try:
        laws = model.fixed.moving(key) or {}
    except Exception:
        laws = {}
    cur_by = {x.sid: x for x in cur}
    bef_by = {x.sid: x for x in before}
    recol = {e.sid for e in tr.last_events if e.kind == "recoloured"}
    for sid_ in list(model.active):
        if sid_ in cur_by or sid_ not in bef_by:
            continue
        law = laws.get(sid_)
        if _ENV.get("COIN_DEBUG"):
            print(f"   [merge] {key}: active S{sid_} {bef_by[sid_].body}@{bef_by[sid_].bbox} gone; law {law}; recoloured {sorted(recol)}; cur {[(x.sid, x.body, x.bbox) for x in cur if x.sid < 1000][:8]}")
        m0 = bef_by[sid_]
        d0 = (m0.bbox[2] - m0.bbox[0], m0.bbox[3] - m0.bbox[1])
        # the thing that took the mover's colour and form this press: at the
        # law's place (a step onto it), or anywhere (a step into a portal
        # that put the mover out elsewhere — a teleport, learned as one)
        hit = [x for x in cur if x.sid in recol and x.sid < 1000 and x.body == m0.body
               and (x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) == d0 and x.sid not in model.active]
        if law and law[0] == "translate":
            v = law[1]
            pb = (m0.bbox[0] + v[0], m0.bbox[1] + v[1], m0.bbox[2] + v[0], m0.bbox[3] + v[1])
            at_ = [x for x in hit if x.bbox == pb]
            hit = at_ or hit
        if len(hit) == 1 and hasattr(tr, "rename") and tr.rename(hit[0].sid, sid_, (hit[0].bbox[0] - m0.bbox[0], hit[0].bbox[1] - m0.bbox[1])):
            if _ENV.get("COIN_DEBUG"):
                print(f"   [merge] S{hit[0].sid} at the law's place wearing S{sid_}'s colour IS S{sid_}: renamed")
            return with_derived(tr, model) if SCENE else tr.prev
    return cur


def _mover_colours(model):
    """The colours the things the keys move have worn (the tracker may
    lose a mover that merges into what it reaches: its colour finds it)."""
    out = set()
    for sid_ in (getattr(model, "active", None) or ()):
        out |= set((getattr(model, "kinds_worn", {}) or {}).get(sid_, ()))
    for k_, c_, ev_, bef_, aft_ in (getattr(model, "_journal", None) or []):
        for x in bef_:
            if x.sid in (model.active or ()):
                out.add(x.body)
    return out


class _Mover:
    """What GoalHypothesis reads off a mover."""
    def __init__(self, x):
        self.color = x.body; self.size = x.size; self.centre = x.centre; self.cells = frozenset(x.cells)


def _dims(x):
    return (x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1])


def _rel(pb, t):
    """How a box stands to a thing: inside it, over it, against it, or not."""
    tb = t.bbox
    if tb[0] <= pb[0] and pb[2] <= tb[2] and tb[1] <= pb[1] and pb[3] <= tb[3] and (tb[2] - tb[0] > pb[2] - pb[0] or tb[3] - tb[1] > pb[3] - pb[1]):
        return "inside"
    if not (pb[2] < tb[0] or tb[2] < pb[0] or pb[3] < tb[1] or tb[3] < pb[1]):
        return "over"
    gap = max(tb[0] - pb[2], pb[0] - tb[2], tb[1] - pb[3], pb[1] - tb[3])
    return "against" if gap <= 1 else None


def coin_from_win(st, comps_won):
    """THE WIN, COINED.  No reading of the level explains the floor's win:
    the win itself is read.  What the keys move is the mover; on the
    winning press its law says where it went — into, over or against
    which thing (an arrival); if along the way things of one colour
    changed or vanished as the mover reached them and the winning press
    reached the last, the win is a collect; a winning click names the
    thing clicked.  Nothing here names a colour or a place: the classes
    are the level's own (a colour and a form), read from what happened.
    Returns (coined, hypothesis) or (None, None); coined =
    (kind, mover colour, target colour, target dims, relation)."""
    model = st["model"]; journal = st.get("journal") or []; won_key = st.get("won_key")
    if won_key is None or comps_won is None:
        return None, None
    key, click = won_key
    bg_ = st.get("bg")
    things = [x for x in comps_won if x.sid < 1000 and "hud" not in x.tags and "terrain" not in x.tags
              and x.size >= 2 and x.body != bg_ and "derived" not in x.tags]
    active = set(model.active or ())
    movers = [x for x in things if x.sid in active]
    if not movers:
        # (the mover merged into what it reached: the tracker kept the
        # reached thing's id and gave it the mover's colour — the colour finds it)
        mcols = _mover_colours(model)
        movers = [x for x in things if x.body in mcols and x.size <= 200]
    if not movers:
        moved_ = Counter()
        for k_, c_, ev_, bef_, aft_ in journal:
            for kind_, sid_ in ev_:
                if kind_ == "moved":
                    moved_[sid_] += 1
        if moved_:
            sid_m = moved_.most_common(1)[0][0]
            movers = [x for x in things if x.sid == sid_m]
    m = max(movers, key=lambda x: x.size) if movers else None
    if _ENV.get("COIN_DEBUG"):
        print(f"   [coin] key {key} click {click} active {sorted(active)} movers {[(x.sid, x.body, x.bbox) for x in movers]} journal {len(journal)} things {[(x.sid, x.body, x.size, x.bbox) for x in things][:12]}")
        for k_, c_, ev_, bef_, aft_ in journal:
            print(f"   [coin]   {k_} {ev_}")
    # a winning click: the thing clicked
    if isinstance(key, tuple) and click is not None:
        under = [x for x in things if x.bbox[0] <= click[0] <= x.bbox[2] and x.bbox[1] <= click[1] <= x.bbox[3] and x.size <= 200]
        if under:
            t = min(under, key=lambda x: x.size)
            h = GoalHypothesis("click", _Mover(t), t.centre, f"won before: click C{t.body}[{t.size}@{t.bbox}]", 9.0, t.size)
            h.t_cells = frozenset(t.cells); h.t_color = t.body
            return ("click", None, t.body, _dims(t), "click"), h
    if m is None:
        return None, None
    # where the winning press took the mover
    law = None
    try:
        law = model.fixed.moving(key).get(m.sid) if not isinstance(key, tuple) else None
    except Exception:
        law = None
    v = law[1] if law and law[0] == "translate" else (0, 0)
    pb = (m.bbox[0] + v[0], m.bbox[1] + v[1], m.bbox[2] + v[0], m.bbox[3] + v[1])
    # a collect: things of one colour and form that changed or vanished as
    # the mover reached them, the winning press reaching the last
    reached = Counter()
    for k_, c_, ev_, bef_, aft_ in journal:
        mb = next((x for x in aft_ if x.sid == m.sid), None) or next((x for x in bef_ if x.sid == m.sid), None)
        if mb is None:
            continue
        for kind_, sid_ in ev_:
            if kind_ not in ("recoloured", "vanished") or sid_ == m.sid or sid_ in active:
                continue
            t = next((x for x in bef_ if x.sid == sid_), None)
            if t is None or "hud" in t.tags or "terrain" in t.tags or t.size > 200:
                continue
            if _rel(mb.bbox, t) is not None:
                reached[(t.body, _dims(t))] += 1
    rank = {"inside": 3, "over": 2, "against": 1}
    for (tcol, tdims), n_ in reached.most_common():
        left = [x for x in things if x.body == tcol and _dims(x) == tdims and x.sid != m.sid and x.sid not in active]
        last = [x for x in left if _rel(pb, x) is not None]
        if n_ >= 1 and (not left or last) and n_ + len(last) >= 2:
            t = last[0] if last else left[0] if left else None
            h = GoalHypothesis("collect", _Mover(m), t.centre if t else m.centre, f"won before: collect C{tcol}[{t.size if t else 0}@{t.bbox if t else m.bbox}]", 9.0, t.size if t else 0)
            h.t_cells = frozenset(q for x in (last or left) for q in x.cells); h.members = [x.sid for x in (last or left)]; h.t_color = tcol
            return ("collect", m.body, tcol, tdims, "over"), h
    # an arrival: the thing the winning press took the mover into, over or against
    cands = [(rank[_rel(pb, x)], -x.size, x) for x in things if x.sid != m.sid and x.sid not in active and x.size <= 400 and _rel(pb, x) is not None]
    if cands:
        cands.sort(key=lambda c: (-c[0], -c[1]))
        r_, _, t = cands[0]
        rel = {3: "inside", 2: "over", 1: "against"}[r_]
        h = GoalHypothesis("align", _Mover(m), t.centre, f"won before: reach C{t.body}[{t.size}@{t.bbox}] ({rel})", 8.5, t.size)
        h.t_cells = frozenset(t.cells); h.over = rel != "against"; h.t_color = t.body
        return ("arrive", m.body, t.body, _dims(t), rel), h
    return None, None


def _mover_gone(model, cur, obs):
    """A lost level's frame from which the mover has gone."""
    if obs.state != GameState.GAME_OVER or not model.active:
        return False
    now_ = {x.sid for x in cur}
    return any(sid_ not in now_ for sid_ in model.active)


def _hazards_of(c):
    """The hazard kinds an export holds (touched or entered on a press
    that lost the level, more often than survived)."""
    if c.get("hazard_kinds") is not None:
        return set(c["hazard_kinds"])
    return {k for k, n in (c.get("death_touch") or {}).items() if n > (c.get("touched_alive") or {}).get(k, 0)} \
        | {k for k, n in (c.get("death_enter") or {}).items() if n > (c.get("entered_alive") or {}).get(k, 0)}


def _frame_key(obs):
    try:
        return hash(np.array(obs.frame)[-1].tobytes())
    except Exception:
        return None


def _key_id(k):
    """A hashable name for a press: a key's name, or (kind, place) for a click, (kind, sid) for a select."""
    if isinstance(k, str):
        return k
    if isinstance(k, (tuple, list)) and len(k) > 1:
        return (k[0], tuple(k[1]) if isinstance(k[1], (list, tuple)) else k[1])
    return k


def _frame_key_frame(f):
    try:
        return hash(np.asarray(f).tobytes())
    except Exception:
        return None


CORE_WAIT_S = 20            # how long a game waits for the core when another game has it, before its floor plays instead (hand-chosen)


def attempt_report(g, lv, n_attempt, how, st, learned=None):
    """The failure report of a lost attempt (agent/core/ledger.py), from
    the level's ledger and model: (text, doubts, divergences)."""
    if learned is None:
        learned = st["model"].export(st.get("comps") or st["tr"].prev, None)
    led_ = getattr(play_level, "ledger", None) or {}
    divs_ = led_.get("divs", [])
    doubts_ = _ledger.doubts_from(divs_[0] if divs_ else None, st["model"])
    with_laws_ = set()
    kind_by_sid_ = {s_.sid: s_.body for s_ in st["tr"].prev}
    keys_live_ = []
    for k_ in st.get("keys", []):
        try:
            mv_ = st["model"].fixed.moving(k_)
        except Exception:
            mv_ = {}
        if mv_:
            keys_live_.append(k_)
            with_laws_ |= {kind_by_sid_.get(sid_) for sid_ in mv_ if sid_ in kind_by_sid_}
    for k_, kinds_ in learned.get("eff", {}).items():
        with_laws_ |= set(kinds_)
    for key_ in ("ground_ok", "flat", "wall_kinds", "death_touch", "click_gone", "click_stays"):
        v_ = learned.get(key_) or {}
        with_laws_ |= set(v_.keys() if hasattr(v_, "keys") else v_)
    with_laws_ |= {a_ for pr_ in learned.get("passable", ()) for a_ in pr_}
    present_ = {s_.body for s_ in st["tr"].prev if s_.sid < 1000 and "hud" not in s_.tags and "terrain" not in s_.tags}
    retested_ = _ledger.verdicts(led_.get("retests"), st["model"], st["tr"].prev, kind)
    rep_ = _ledger.report(g, lv, n_attempt, how, led_.get("reading"), led_.get("plan_len", 0), divs_, doubts_,
                          hyps_tried=led_.get("hyps_tried", []), keys_known=keys_live_, kinds_unasked=present_ - with_laws_, retested=retested_,
                          agents=sorted(getattr(st["model"], "agents", ()) or ()), carry_law=learned.get("carry_law"),
                          agent_lines=(st["model"].agent_record.describe() if getattr(st["model"], "agent_record", None) else ()))
    return rep_, doubts_, divs_


def play_game(env, obs, budget=40, verbose=False, lo=1, hi=None, conf=(), g="game", persist=False,
              floor=True, max_actions=4000, seconds=None, lock=None):
    # (recorded reader answers for this game, replayed level by level)
    if getattr(play_level, "reader_answers", None) is not None:
        play_level.reader = _reader.ScriptedReader(play_level.reader_answers, game=g[:4])
        play_level.reader.level = lo - 1
    """Play a game level by level from `lo` to `hi` (or until the game is
    won): THE CORE reads and plans; when it leaves a level (no reading
    that plans, or a death) THE FLOOR plays the level from where it
    stands, and the core is given the level again only from a state it
    has not tried — a level is never given up while the game runs
    (actions on an unfinished level cost nothing banked).  Every won
    level exports what it taught to the next.  A death does not restart
    the game: the engine resets the level on the one RESET sent, and a
    second RESET is never sent (it would throw the game to its first
    level).  The same loop drives the command line and the submission."""
    if hi is None:
        hi = getattr(obs, "win_levels", None) or 10 ** 6
    deadline = time.time() + (seconds if seconds else 10 ** 9)
    fl = Floor() if floor else None
    carry = None
    st = None; rw = None; sw_ = None
    play_game.lessons = []          # (game, level, won, the reading the floor's win satisfied, hazards, actions)
    play_game.reports = []          # the failure reports (agent/core/ledger.py), one per lost core attempt
    play_game.carries = {}          # level -> what its play exported (the lesson of the level)
    play_game.clock_deadly = False
    play_game.highlights = {}       # level -> the journal's highlight (what we actually did)
    lv = lo
    spent = 0                       # actions in this call (the harness's cap is on the game)
    since_reset = 1
    while lv <= hi and (obs is None or obs.state != GameState.WIN) and spent < max_actions and time.time() < deadline:
        tried_from = set()          # the states this level was handed to the core from
        st = None
        turn = FLOOR_TURN
        retried = False
        restarted = False
        floor_turns = 0
        floor_spent = 0
        n_attempts = 0
        spent_at_level = spent
        start_key = _frame_key(obs)     # (the level as it began: a RESET brings it back)
        won = False
        while not won and spent < max_actions and time.time() < deadline:
            fk = _frame_key(obs)
            core_turn = fk not in tried_from and len(tried_from) < CORE_TRIES
            if core_turn and lock is not None:
                # (the core is busy with other games: the floor plays
                # meanwhile — unless the floor has had its share of this
                # level's actions, when the game waits its turn at the core)
                floor_capped = fl is not None and floor_spent >= FLOOR_SHARE * max(1, max_actions - spent_at_level)
                wait_ = max(CORE_WAIT_S, deadline - time.time()) if (floor_capped or fl is None) else CORE_WAIT_S
                if not lock.acquire(timeout=max(1.0, wait_)):
                    core_turn = False
                    if fl is None or time.time() >= deadline:
                        continue
            if core_turn:
                tried_from.add(fk)
                n_attempts += 1
                try:
                    try:
                        play_level.seconds = min(LEVEL_SECONDS, max(60.0, TURN_FRACTION * (deadline - time.time())))
                        won, steps, how, obs = play_level(env, obs, budget, conf, verbose, carry=carry)
                    except Exception as ex_:
                        # (a fault in the core is not the level's end: the floor
                        # plays on from where the core left it)
                        won, steps, how = False, play_level.steps, "crashed"
                        obs = getattr(env, "last", None) or getattr(env, "_last_response", None) or obs
                        print(f"{g}:{lv} the core failed: {ex_!r}", flush=True)
                        import traceback; traceback.print_exc()
                    # (what the attempt left on the function is read under the
                    # lock: another game's core may take it next)
                    st = play_level.last if isinstance(getattr(play_level, "last", None), dict) else None
                    rw = getattr(play_level, "rweights", None)
                    sw_ = getattr(play_level, "start_world", None)
                finally:
                    if lock is not None:
                        lock.release()
                spent += steps + (1 if how == "died" else 0)
                since_reset = 0 if how == "died" else since_reset + steps
                print(f"{g}:{lv} {'WON' if won else 'lost'} in {steps} actions ({how})", flush=True)
                if won or st.get("model") is None or obs is None:
                    break
                learned = st["model"].export(st.get("comps") or st["tr"].prev, None)
                learned["claims"] = _claims.export()
                learned["failed_readings"] = dict(getattr(play_level, "failed_readings", {}) or {})
                rep_c_ = _claims.report()
                if rep_c_:
                    print("  " + rep_c_, flush=True)
                taught = _hazards_of(learned) - _hazards_of(carry or {})
                # THE HIGHLIGHT: what we actually did, compressed from the journal
                try:
                    hl_ = play_level.journal.highlight(how.split(":")[0], getattr(play_level, "ledger", None), (getattr(play_level, "ledger", None) or {}).get("reading"))
                    learned["highlight"] = hl_
                    print(_journal.describe(hl_), flush=True)
                except Exception as ex_:
                    if verbose:
                        print(f"   (journal: {ex_!r})")
                learned["coined"] = list(getattr(st["model"], "coined", []) or [])
                learned["unit_form"] = dict(getattr(st["tr"], "unit_form", {}) or {})    # (the unit of a kind carries)
                learned["elastic"] = set(getattr(st["tr"], "elastic", ()) or ())
                learned["progress_obj"] = (int(obs.levels_completed), getattr(play_level, "progress", None))
                # (the run key that told the level — carried: the next level's
                # first press is that key, before any reading)
                try:
                    runs_c_ = [k_ for k_, r_, _, _ in (getattr(getattr(play_level, "progress", None), "runs", None) or []) if isinstance(k_, str) and r_.get("source") is not None]
                    learned["run_key"] = runs_c_[-1] if runs_c_ else None
                except Exception:
                    learned["run_key"] = None
                carry = dict(learned); carry["how"] = how.split(":")[0]
                # THE FAILURE REPORT: why, and what not to believe again — the
                # first divergence between the plan's prediction and the world
                # names the law; it is left out of the next attempt's laws
                try:
                    rep_, doubts_, _ = attempt_report(g, lv, n_attempts, how, st, learned)
                    print(rep_, flush=True)
                    play_game.reports.append(rep_)
                    carry = _ledger.apply_doubts(carry, doubts_)
                    carry["last_report"] = rep_                  # (the reader is shown why the last attempt was lost)
                except Exception as ex_:
                    print(f"{g}:{lv} (the ledger failed: {ex_!r})", flush=True)
                if how == "died" and not retried and (taught or (learned.get("highlight") or {}).get("fatal")):
                    # (one more attempt from the level's start: the state is
                    # the same, the knowledge is not — the death taught a
                    # hazard; a death that taught nothing, the clock's or a
                    # wall's, would be the same trial again)
                    retried = True
                    tried_from.discard(_frame_key(obs))
                    print(f"{g}:{lv} the death taught {sorted(taught) if taught else 'which press killed'}: the core looks again", flush=True)
                    continue
                # (a death that was the CLOCK's — the budget spent, no hazard
                # learned — is a level to try once more from the start with the
                # keys known, like any first look that spent the level)
                clock_died_ = how == "died" and getattr(st["model"], "deaths", 0) > 0 and not taught
                if clock_died_ and not retried and obs.state not in (GameState.WIN, GameState.GAME_OVER):
                    retried = True
                    carry["clock_death"] = True      # (no safe prefix: what was walked spent the clock)
                    carry["clock_deadly"] = True     # (this game's clock ends a level: the fiddle is rationed by it)
                    play_game.clock_deadly = True
                    tried_from.discard(_frame_key(obs))
                    print(f"{g}:{lv} the clock ran out: once more from the start, within the budget", flush=True)
                    continue
                if how != "died" and not restarted and since_reset > 0 and start_key is not None and _frame_key(obs) != start_key \
                        and obs.state not in (GameState.WIN, GameState.GAME_OVER):
                    # ONCE MORE FROM THE START, KNOWING: the first look fiddled
                    # blind and may have spent the level's start (st70: the goal
                    # one press left of where the mover began; the fiddle's
                    # first press slid it away for good).  One RESET is a level
                    # reset — nothing banked is lost — and the core looks at the
                    # start again with the laws, walls and hazards it now has
                    restarted = True
                    obs = env.step(GameAction.RESET); spent += 1; since_reset = 0
                    tried_from.discard(_frame_key(obs))
                    print(f"{g}:{lv} once more from the start, knowing the keys", flush=True)
                    continue
                if fl is None:
                    break
            elif fl is None:
                break
            n_turn = min(turn, max_actions - spent)
            fresh_ = False
            if floor_turns >= 1:
                # THE SECOND TURN STARTS CLEAN: the level is put back to its
                # start (one RESET, a level reset) and the floor reads it
                # anew — a level first seen mid-way, after the core's presses,
                # can be misread, and the misreading persists through its
                # own resets (shh0: 9 actions from the start, 400 lost after
                # the core's 15).  Nothing banked is lost on an unfinished level
                if since_reset > 0 and obs.state not in (GameState.WIN,):
                    obs = env.step(GameAction.RESET); spent += 1; since_reset = 0
                fresh_ = True
            floor_turns += 1
            print(f"{g}:{lv} the floor plays ({n_turn} actions{', from the start, fresh' if fresh_ else ''})", flush=True)
            won, n, obs, since_reset = fl.play(env, obs, n_turn, deadline, since_reset, verbose, fresh=fresh_)
            spent += n; floor_spent += n
            print(f"{g}:{lv} {'WON' if won else 'lost'} in {n} actions (floor)", flush=True)
            how = "floor"
            turn = min(turn * 2, FLOOR_TURN_MAX)
            # THE CORE WATCHES THE FLOOR: the floor's presses are observed
            # as the core's own would be, and a win is read against the
            # level's readings — the relation the win satisfied carries as
            # a won reading does, laws and hazards with it
            if st is not None and st.get("model") is not None and getattr(fl, "trans", None):
                try:
                    bg_ = background_color(fl.trans[0][2] if fl.trans[0][2] is not None else fl.trans[0][3][-1])
                    st["bg"] = bg_
                    comps_won = watch_floor(st, fl.trans, bg_, lv - 1, verbose)      # (levels completed before this one)
                    won_h = _reading_won(st.get("hyps"), comps_won, st["model"]) if won and comps_won is not None else None
                    coined_ = None
                    if won and won_h is None and comps_won is not None:
                        coined_, won_h = coin_from_win(st, comps_won)
                    carry = st["model"].export(comps_won or st["tr"].prev, won_h)
                    carry["how"] = "floor"
                    carry["coined"] = list(getattr(st["model"], "coined", []) or [])
                    if coined_ is not None and coined_ not in carry["coined"]:
                        carry["coined"].append(coined_)
                    if won:
                        if won_h is not None and coined_ is None:
                            print(f"{g}:{lv} the floor's win reads as {won_h}", flush=True)
                            if rw is not None:
                                rw.won(won_h); carry["relation_weights"] = rw.get()
                        elif coined_ is not None:
                            print(f"{g}:{lv} the floor's win matches no reading of the level; coined from the win: {coined_} — {won_h}", flush=True)
                        else:
                            print(f"{g}:{lv} the floor's win matches no reading of the level and could not be coined (a mechanic the vocabulary lacks)", flush=True)
                    play_game.lessons.append((g, lv, won, won_h, sorted(_hazards_of(carry)), n))
                except Exception as ex_:
                    if verbose:
                        print(f"   reading the floor's play failed: {ex_!r}")
        if not won:
            break
        if how == "floor":
            play_game.carries[lv] = carry
            lv += 1
            continue
        # THE WIN, READ HONESTLY (Tushar, lp85: "a win the agent cannot explain is
        # a win it cannot repeat"): the reading pursued is the story only if the
        # winning state satisfies it; else the reading the state does satisfy is
        # (tw03 L1: "reach the odd thing" pursued, the block slid into the pen)
        try:
            # (the winning press returns the winning frame first and the next level's
            # after it: the state at the win is read from that first frame)
            fr_all_ = np.array(obs.frame) if obs is not None else None
            if fr_all_ is not None and fr_all_.ndim == 3 and len(fr_all_) >= 2:
                f_win_ = fr_all_[0]
                after_ = read_components(f_win_, st.get("bg", background_color(f_win_)))
            else:
                after_ = list(st.get("comps") or [])
            # (what moved in the level: a thing standing elsewhere at the win than at the start, by colour and size)
            moved_cols_ = set()
            try:
                ff_ = getattr(play_level, "first_frame", None)
                if ff_ is not None:
                    first_ = read_components(ff_, st.get("bg", background_color(ff_)))
                    for x_ in after_:
                        if x_.size < 200 and "hud" not in x_.tags and not any(y_.body == x_.body and y_.size == x_.size and y_.bbox == x_.bbox for y_ in first_):
                            moved_cols_.add(x_.body)
            except Exception:
                pass
            h0_ = st.get("won_hyp")
            ok0_ = h0_ is not None and _reading_won([h0_], after_, st["model"], extra_cols=moved_cols_) is h0_
            # (a level won by an ORGAN with no reading pursued — carry, program,
            # legend, table — has the organ for its story; a reading the end
            # state happens to satisfy is not what won it, and carried as the
            # template it sends the next level after a coincidence — wa30 L2,
            # lost pursuing "the odd thing" read off L1's win)
            if h0_ is None:
                raise StopIteration
            if verbose:
                print(f"   the win's reading: pursued {h0_.note[:50] if h0_ is not None else None!r} satisfied {ok0_}; things at the win {[(x_.body, x_.bbox) for x_ in after_ if x_.size < 100][:6]}; hyps {len(st.get('hyps') or [])}")
            if not ok0_:
                # (every reading the first frame affords, not only those the level pursued:
                # the relation the win satisfied may be one the core set aside)
                all_h_ = list(st.get("hyps") or [])
                try:
                    ff_ = getattr(play_level, "first_frame", None)
                    if ff_ is not None:
                        bgf_ = st.get("bg", background_color(ff_))
                        all_h_ += read_frame_goals(segment(ff_, bgf_, mask=None), 64, 64)
                except Exception:
                    pass
                h1_ = _reading_won([h_ for h_ in all_h_ if h_ is not h0_], after_, st["model"], extra_cols=moved_cols_)
                if h1_ is not None:
                    print(f"{g}:{lv} the win read honestly: pursued {getattr(h0_, 'note', None) and h0_.note[:50]!r}, the state satisfies {h1_.note[:60]!r} — carried as the story", flush=True)
                    st["won_hyp"] = h1_
        except StopIteration:
            pass
        except Exception as ex_:
            if verbose:
                print(f"   (the win's reading: {ex_!r})")
        carry = st["model"].export(st.get("comps") or st["tr"].prev, st.get("won_hyp"))
        carry["how"] = how.split(":")[0]
        carry["coined"] = list(getattr(st["model"], "coined", []) or [])
        carry["claims"] = _claims.export()
        fr_ = dict(getattr(play_level, "failed_readings", {}) or {})
        # (a reading the win vindicated is not a misleading one: its earlier
        # "arrived, nothing happened" was a wrong INSTANCE of the rule, and
        # the win outranks it — ls20 L1 arrived at one odd thing to no effect,
        # won at another under the same rule, and the record then sorted the
        # rule last on L2, off the end of the pursuits)
        try:
            wh_ = st.get("won_hyp")
            if wh_ is not None and hasattr(play_level, "_reading_id"):
                fr_.pop(play_level._reading_id(wh_), None)
        except Exception:
            pass
        carry["failed_readings"] = fr_
        rep_c_ = _claims.report()
        if rep_c_:
            print("  " + rep_c_, flush=True)
        try:
            hl_ = play_level.journal.highlight(how.split(":")[0], getattr(play_level, "ledger", None), str(st.get("won_hyp")) if st.get("won_hyp") is not None else None)
            carry["highlight"] = hl_
            play_game.highlights[lv] = hl_
            print(_journal.describe(hl_), flush=True)
        except Exception as ex_:
            if verbose:
                print(f"   (journal: {ex_!r})")
        # WHERE THE GOAL WAS, from where the body began (in the world's
        # frame): the direction to explore when the goal is not in view
        try:
            h_ = st.get("won_hyp")
            if h_ is not None and sw_ is not None:
                tc_ = h_.target_centre(); cam_ = st["model"].camera
                dr_ = tc_[0] + cam_[0] - sw_[0]; dc_ = tc_[1] + cam_[1] - sw_[1]
                carry["won_dir"] = ((dr_ > 6) - (dr_ < -6), (dc_ > 6) - (dc_ < -6))
        except Exception:
            pass
        if rw is not None:
            if st.get("won_hyp") is not None:
                rw.won(st["won_hyp"])
            carry["relation_weights"] = rw.get()
            carry["unit_form"] = dict(getattr(st["tr"], "unit_form", {}) or {})    # (the unit of a kind carries)
            carry["elastic"] = set(getattr(st["tr"], "elastic", ()) or ())
        # (what the presses made happen carries BY KIND, as the laws do: the
        # merges seen and the reach of a call — the next level's twin may have
        # to be made before it can be taken anywhere)
        pr_ = getattr(play_level, "progress", None)
        if pr_ is not None and getattr(pr_, "merges", None):
            carry["progress_merges"] = (dict(pr_.merges), int(getattr(pr_, "reach", 0) or 0))
        # (the run key that told the level carries: the next level's first
        # press is that key, before any reading)
        try:
            runs_c_ = [k_ for k_, r_, _, _ in (getattr(pr_, "runs", None) or []) if isinstance(k_, str) and r_.get("source") is not None]
            carry["run_key"] = runs_c_[-1] if runs_c_ else None
        except Exception:
            carry["run_key"] = None
        if any((c_ or {}).get("clock_deadly") for c_ in play_game.carries.values()) or getattr(play_game, "clock_deadly", False):
            carry["clock_deadly"] = True
        if verbose:
            print(f"   carrying to L{lv+1}: {carry}")
        play_game.carries[lv] = carry
        lv += 1

    return obs


play_game = PerThread(play_game)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--no-floor", action="store_true", help="the core alone (diagnosis)")
    ap.add_argument("--max-actions", type=int, default=4000)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--transform", default=None, help="the invariance gauge: colour:SEED | mirror | flip | rot90 | rot270, joined by + (tools/invariance.py)")
    ap.add_argument("--reader", default=None, help="a JSON file of recorded reader answers keyed game:level (the Kaggle trial's reader_answers.json), replayed into the level openings")
    args = ap.parse_args()
    if getattr(args, "reader", None):
        import json as _json
        play_level.reader_answers = _json.load(open(args.reader))
    arc = arc_agi.Arcade(operation_mode="offline")
    from tools.core_play import reach_level
    for t in args.targets:
        g, lv = t.split(":")
        lo, hi = (int(lv.split("-")[0]), int(lv.split("-")[-1]))
        gid = [i.game_id for i in arc.get_environments() if i.game_id.startswith(g)][0]
        env = arc.make(gid)
        if args.transform:
            from tools.invariance import Transformed
            env = Transformed(env, args.transform)
        obs = env.reset()
        conf = ()
        if lo > 1:
            obs, ag, n0 = reach_level(env, obs, lo, args.v)
            conf = getattr(ag, "_goal_confirmed_kinds", ())
            if obs.levels_completed < lo - 1:
                print(f"{g}:{lo} could not reach the level"); continue
        play_game(env, obs, args.budget, args.v, lo, hi, conf, g, floor=not args.no_floor, max_actions=args.max_actions, seconds=args.seconds)


if __name__ == "__main__":
    main()
