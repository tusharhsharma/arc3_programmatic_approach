"""Object tracker: identity by spatiotemporal continuity, not appearance.

Solves three hazards real games throw at appearance-keyed perception:
  - avatars that animate / power-up (shape or color changes every frame),
  - autonomous movers (projectiles, patrolling enemies) that must not be
    mistaken for the avatar nor attributed to our actions,
  - drift/flow: the avatar's autonomous velocity component.

Tracks live in world coordinates. Matching: nearest predicted position
(pos + velocity), with appearance similarity as a tiebreaker only.
"""
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass
class Track:
    id: int
    pos: tuple                 # world centroid (float r, c)
    vel: tuple = (0.0, 0.0)    # EMA velocity per step
    size: int = 1
    color: int = 0
    shapes: deque = field(default_factory=lambda: deque(maxlen=8))
    action_moves: dict = field(default_factory=dict)  # action -> Counter of unit dirs
    steps_seen: int = 0
    steps_moved: int = 0
    missed: int = 0
    last_disp: tuple = (0.0, 0.0)
    disps: deque = field(default_factory=lambda: deque(maxlen=8))
    last_obj = None
    stale: int = 0             # consecutive commands it ignored while another track answered
    forced: bool = False       # identity assigned by direct evidence (a teleport landing)

    def predicted(self):
        return (self.pos[0] + self.vel[0], self.pos[1] + self.vel[1])

    @property
    def is_oscillating(self):
        """Wobbling panels / hover animations: moves but goes nowhere, AND
        the direction is independent of the action — the SAME action yields
        opposing displacements. An avatar bounced around by probing also has
        near-zero mean displacement, but each action gives it a consistent
        direction, so it never trips this."""
        moves = [d for d in self.disps if abs(d[0]) + abs(d[1]) >= 0.5]
        if len(moves) < 4:
            return False
        mr = sum(d[0] for d in moves) / len(moves)
        mc = sum(d[1] for d in moves) / len(moves)
        if abs(mr) >= 0.5 or abs(mc) >= 0.5:
            return False
        for ctr in self.action_moves.values():
            dirs = [d for d in ctr if d != (0, 0)]
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    if (dirs[i][0] + dirs[j][0], dirs[i][1] + dirs[j][1]) == (0, 0):
                        return True
        return False

    @property
    def is_autonomous(self):
        """Moves consistently regardless of which action was taken."""
        if self.steps_seen < 3 or self.steps_moved < 3:
            return False
        all_dirs = Counter()
        for ctr in self.action_moves.values():
            all_dirs.update(ctr)
        if not all_dirs:
            return False
        (top_dir, top_n) = all_dirs.most_common(1)[0]
        if top_dir == (0, 0):
            return False
        # same direction under >= 2 different actions and dominant overall
        actions_with_top = sum(1 for ctr in self.action_moves.values()
                               if ctr.get(top_dir, 0) > 0)
        return top_n >= 0.7 * sum(all_dirs.values()) and actions_with_top >= 2

    @property
    def control_score(self):
        """High when displacement direction depends on the action taken."""
        per_action_dirs = {}
        moving_actions = 0
        for a, ctr in self.action_moves.items():
            if not ctr:
                continue
            (d, n) = ctr.most_common(1)[0]
            if sum(ctr.values()) >= 1 and d != (0, 0):
                per_action_dirs[a] = d
                moving_actions += 1
        distinct = len(set(per_action_dirs.values()))
        if moving_actions == 0:
            return 0.0
        if self.is_autonomous or self.is_oscillating:
            return 0.0
        return distinct + 0.1 * moving_actions

    def drift(self):
        """Rounded autonomous velocity (flow/current estimate)."""
        return (round(self.vel[0]), round(self.vel[1]))


def unit_frac(vec):
    dr, dc = vec
    return ((0 if abs(dr) < 0.5 else (1 if dr > 0 else -1)),
            (0 if abs(dc) < 0.5 else (1 if dc > 0 else -1)))


class Tracker:
    MATCH_DIST = 10.0
    CLICK_PENALTY = 1.5

    def __init__(self):
        self.tracks = {}
        self._next = 0
        self._avatar_id = None      # sticky avatar choice

    def reset(self):
        self.__init__()

    def update(self, objs_world, action_name):
        """objs_world: [(shape_key, color, size, world_bbox, centroid_world)].
        Returns {obj_index: track_id}."""
        assignments = {}
        for t in self.tracks.values():
            t.last_disp = (0.0, 0.0)
        unmatched = set(range(len(objs_world)))
        # greedy nearest-predicted matching
        from collections import Counter as _Cc
        colcnt = _Cc(o[1] for o in objs_world)
        pairs = []
        for tid, t in self.tracks.items():
            pr, pc = t.predicted()
            for i in unmatched:
                _, color, size, _, (cr, cc) = objs_world[i]
                d = abs(pr - cr) + abs(pc - cc)
                if d > self.MATCH_DIST:
                    continue
                # appearance is a tiebreaker, never a veto — except
                # under a CLICK, which is not physics: a body that keeps
                # its appearance and jumps beats a body that stays put
                # and changes color (sx06 swap: position-first matching
                # bound identity to the anchor for four steps)
                bonus = 0.0
                if color != t.color:
                    # ...and only when the appearance is DISTINCTIVE
                    # (<=2 instances); among many same-colored pieces
                    # (r11l's rope) appearance says nothing and position
                    # keeps the say (r11l 46 -> 438 under a blanket veto)
                    distinctive = (colcnt.get(t.color, 0) <= 2
                                   and colcnt.get(color, 0) <= 2)
                    bonus += (self.CLICK_PENALTY
                              if action_name == "ACTION6" and distinctive
                              else 1.5)
                if size != t.size and max(size, t.size) > 2 * min(size, t.size):
                    bonus += 1.5
                pairs.append((d + bonus, tid, i))
        pairs.sort()
        used_tracks = set()
        for cost, tid, i in pairs:
            if tid in used_tracks or i not in unmatched:
                continue
            used_tracks.add(tid)
            unmatched.discard(i)
            t = self.tracks[tid]
            key, color, size, bbox, cen = objs_world[i]
            disp = (cen[0] - t.pos[0], cen[1] - t.pos[1])
            t.last_disp = disp
            t.disps.append(disp)
            t.vel = (0.5 * t.vel[0] + 0.5 * disp[0],
                     0.5 * t.vel[1] + 0.5 * disp[1])
            t.pos = cen
            t.color = color
            t.size = size
            t.shapes.append(key)
            t.steps_seen += 1
            u = unit_frac(disp)
            if u != (0, 0):
                t.steps_moved += 1
            t.action_moves.setdefault(action_name, Counter())[u] += 1
            t.missed = 0
            t.last_obj = objs_world[i]
            assignments[i] = tid
        # birth new tracks
        for i in unmatched:
            key, color, size, bbox, cen = objs_world[i]
            t = Track(id=self._next, pos=cen, color=color, size=size)
            t.shapes.append(key)
            self.tracks[self._next] = t
            assignments[i] = self._next
            self._next += 1
        # IDENTITY IS REVISABLE BY NON-RESPONSE: the sticky avatar track
        # that ignores command after command while some other track
        # answers has lost the body (observed sx06: a click-swap left the
        # avatar's track on the anchor now standing in its old cell, and
        # the real body walked away unowned for a whole level)
        if action_name not in ("ACTION6", "RESET") and self._avatar_id in self.tracks:
            av = self.tracks[self._avatar_id]
            if self._avatar_id in used_tracks:
                if unit_frac(av.last_disp) != (0, 0):
                    av.stale = 0
                elif any(unit_frac(t.last_disp) != (0, 0)
                         and not t.is_autonomous and not t.is_oscillating
                         for tid, t in self.tracks.items()
                         if tid in used_tracks and tid != self._avatar_id):
                    av.stale += 1
        # age out unmatched tracks
        for tid, t in list(self.tracks.items()):
            if tid not in used_tracks and tid in self.tracks \
                    and assignments.get(max(assignments, default=-1)) != tid:
                if t.steps_seen > 0 and tid not in used_tracks \
                        and all(v != tid for v in assignments.values()):
                    t.missed += 1
                    if t.missed > 3:
                        del self.tracks[tid]
        return assignments

    # ---------- queries ----------

    def avatar_track(self):
        # sticky: once chosen, keep the same track while it lives and stays
        # controlled (mirrored-twin games have TWO controlled objects, and
        # alternating identity corrupts the motion model)
        prev = self.tracks.get(self._avatar_id)
        if prev is not None and (prev.control_score > 0.9 or prev.forced) \
                and prev.stale < 4:
            return prev
        if prev is not None:
            prev.forced = False
        best, best_score = None, 0.9
        for t in self.tracks.values():
            if prev is not None and t.id == prev.id:
                continue        # the stale holder competes only by default
            s = t.control_score
            if s > best_score:
                best, best_score = t, s
        if best is not None:
            self._avatar_id = best.id
            best.stale = 0
            return best
        return prev if (prev is not None and prev.control_score > 0.9) else None

    def bind_to(self, world_tl, color):
        """Point the sticky identity at the track whose current object is
        the (color) body at world_tl — used when a click teleported the
        avatar and the evidence of where it landed is unambiguous."""
        for tid, t in self.tracks.items():
            lo = t.last_obj
            if lo is not None and lo[1] == color and (lo[3][0], lo[3][1]) == tuple(world_tl):
                self._avatar_id = tid
                t.stale = 0
                t.forced = True
                return True
        return False

    def autonomous_tracks(self):
        return [t for t in self.tracks.values() if t.is_autonomous]

    def explains(self, color, size, vec):
        """Does some autonomous or oscillating track account for this move?"""
        u = unit_frac(vec)
        for t in self.tracks.values():
            if t.color != color:
                continue
            if t.is_autonomous and unit_frac(t.vel) == u:
                return True
            if t.is_oscillating:
                return True     # wobble: any direction, it isn't our doing
        return False
