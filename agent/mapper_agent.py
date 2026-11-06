"""MapperAgent v2: micromouse strategy on real ARC-AGI-3 games.

Modes per level:
  PROBE      - resolve each simple action's semantics from varied positions.
  SEEK       - flood-fill plan to a salient goal candidate (world coords).
  EXPLORE    - plan to nearest frontier.
  CLICK      - for click-driven games: probe object centroids with ACTION6,
               prefer objects whose clicks changed the world.

Uses Diff.world_changed (playfield minus HUD indicators, camera-compensated),
never raw frame equality.
"""
import random
from collections import Counter, defaultdict

from arcengine import GameAction, GameState

from .agents import available
from .perception import Perceiver, last_grid, segment
from .tracker import Tracker
from .scene_analysis import find_block_match_tasks
from .transform_solver import TransformSolver
from .world_model import MicromouseModel, wc, CANVAS, unit
from .goals import goal_scan, goal_clusters, GoalActuator, GoalEffectMap
FALLBACK_OLD = False
from .docking import cnorm, crot_k, orientation_of, group_cells, best_poses, pose_target, port_colors  # noqa: E501
from .multibody import detect_bodies, plan_merge
from .permuter import op_from_moves, perm_solve, goal_assignments, augment_inverses, close_ring  # noqa: E501
from .aligner import deflector_placements
from .pegs import solve_pegs, jump_moves
from .transform_solver import OPS

PROBE_BUDGET = 6


class MapperAgent:
    REBIND_STALE = False
    name = "mapper"

    def __init__(self, seed=0, ablate=None):
        # ABLATION SWITCHES (T1): names of capabilities disabled for this
        # run, so every skill/prior must prove its contribution. Known:
        # docking, carrier, aligner, multibody, transform, permuter,
        # blockmatch, touch. Empty in competition builds.
        self.ablate = frozenset(ablate or ())
        self.rng = random.Random(seed)
        self.perc = Perceiver()
        self.tracker = Tracker()
        self.model = None
        self.prev_frame = None
        self.plan = []
        self.goal_color_prior = None
        self._seek_color = None
        self._seek_target = None
        self._gate_fails = {}
        self._gov_state = {}
        # HYPOTHESIS LEDGER: every adopted belief, with the alternatives
        # that were ALSO consistent at adoption time — so a post-run
        # reader can see where multiple genuine answers existed and
        # which one the program chose (and why it was defensible)
        self.hlog = getattr(self, "hlog", [])
        self._amb_frames = {}
        self._ambient = set()
        self._obs_i = 0
        self.unresolved = {}
        self.failed_at = {}
        self._last_fail = {}
        self.effect_counts = {}
        self.probe_failed_keys = set()
        self.click_stats = defaultdict(lambda: [0, 0])  # shape_key -> [changes, tries]
        self.transform = TransformSolver()
        self._prev_objs = []
        self._commit_i = 0
        self._traj = []             # avatar positions since last reset
        self._timer_runs = {}
        self._timer_confirmed = set()       # (band,row,color) -> list of run lengths
        self._shape_op_votes = {}   # action -> Counter of dihedral op names
        self._dock = None           # active docking pose {k, tl, base, wait}
        self._dock_cmds = 0         # commands issued since last observed motion
        self._dock_prev_cen = None
        self._sel_belief = None     # is the piece currently selected?
        self._just_clicked = False
        self._dock_inbounds = True  # last dock command stayed in bounds
        self._port_colors_seen = set()  # connector colors ever confirmed
        self._switch_cool = 0       # cooldown between piece switches
        self._comove = {}           # {frozenset(track_ids): co-move count}
        self._tid_by_key = {}       # (shape_key, world centroid) -> track id
        self._piece_memory = {}     # seed track id -> accepted part shape_keys
        self._sel_color = None      # learned selection livery (body recolor)
        self._last_click_frame_pos = None
        self._mb_plan = []          # queued joint-plan actions
        self._mb_pred = None        # predicted body positions after last cmd
        self._mb_last_positions = None
        self._mb_refuted = False    # merge archetype refuted this level
        self._mb_memory = {}        # (color,size,quadrant) -> action map
        self._perm_ops = {}         # button key -> {'op': {from:to}, 'click': (fr,fc)}
        self._perm_markers = {}     # marker color -> set(world cells)
        self._perm_pieces = set()   # piece class colors
        self._perm_sol = None       # queued op keys to execute
        self._perm_bad = set()      # refuted goal-assignment indexes
        self._perm_fail_sig = None
        self._perm_btn_sizes = set()   # pixel sizes of learned buttons
        self._perm_btn_tried = {}      # candidate button -> click attempts
        self._perm_lattice = set()     # every cell a piece was ever seen on
        self._perm_track = {}          # button -> cells its clicks change
        self._peg = {"mode": None, "queue": [], "await": None, "fails": 0}
        # edge-maze archetype: passability lives on EDGE MIDPOINTS
        self._em = {"open": {}, "closed": {}, "ec": None, "path": [],
                    "blocked": set(), "fails": 0, "dead": False}
        self._em_prev = None
        # template-overlay archetype (re86 family)
        self._ov = {"ctl": None, "fails": 0, "dead": False, "cycles": 0,
                    "last": None}
        # reference-assembly archetype (sb26 family)
        self._rf = {"pend": None, "dead": False, "submits": 0}
        # remote-actuator archetype (s5i5 family)
        self._ac = {"tried": {}, "last": None, "dead": False, "probes": 0}
        # constraint-glyph lattice archetype (ft09 family)
        self._gl = {"last": None, "dead": False, "clicks": {}}
        # layout-switch supervisor (dc22 family)
        self._ls = {"cool": 0, "ri": 0, "clicks": 0, "await": None,
                    "dropped": set(), "confirmed": set()}
        # SET-QUANTIFIED GOAL CLASSES (cardinality lesson in the generic
        # layer): collection games have MANY goals; the "unique odd
        # object" rule hard-coded cardinality 1. Persist across levels.
        self._collect_colors = set()   # touch -> instance vanished
        self._lethal_colors = set()    # touch -> death
        self._align_refuted = set() # deflector x positions that failed
        self._bigfx = {}            # action -> count of large world changes
        self._tile_fx = {}          # tile color -> 'rot'|'color'|'form'
        self._carried_box = None    # bbox of the carried-shape display
        self._carrier_gate = None
        self._display_votes = {}
        self._touched_cells = set()
        self._touch_tries = {}
        self._touch_gaveup = set()
        self._goal_refuted = set()
        self._goal_deferred = set()
        self._goal_arrivals = {}
        self._t2_clicks = 4
        self._anomalies = []
        self._anom_budget = 12
        self._anom_expect = None
        self._edge_probed = set()
        self._goal_hyps = None
        self._goal_act = None
        self._goal_emap = None
        self._goal_used = 0
        self._goal_confirmed_kinds = getattr(self, "_goal_confirmed_kinds", [])
        self._align_target = None   # x we are moving the deflector to
        self._align_committing = False
        self._dock_refuted = set()  # (k, topleft) poses that changed nothing
        self._timer_bars = set()    # {(frame_row, color)} of ACTIVE countdown bars
        self._timer_active = False  # a shrinking HUD bar predicts timeout deaths
        self._death_tail = {}       # final-cell -> how many cells to poison
        self._seq = []              # click ids this attempt (sequence games)
        self._banned_next = {}      # click-seq prefix -> set of banned next ids
        self._click_queue = []
        self._last_click_key = None
        self._levels = 0
        self._since_reset = 0

    # ---------- helpers ----------

    def _objs_world(self, obs):
        """[(shape_key, color, size, bbox_world_topleft, centroid_world)]."""
        objs = self.perc.objects(obs.frame)
        r0, c0 = self.perc.playfield[0], self.perc.playfield[2]
        out = []
        for o in objs:
            fl = (o.bbox[0] - r0, o.bbox[1] - c0)   # field-local
            world = (fl[0] + self.perc.cam[0], fl[1] + self.perc.cam[1])
            cen = (o.centroid[0] - r0 + self.perc.cam[0],
                   o.centroid[1] - c0 + self.perc.cam[1])
            out.append((o.shape_key, o.color, o.size, world, cen))
        return out

    def _nonlocal_change(self, d):
        """Did anything appear/vanish beyond the avatar's immediate cell?
        Terrain-colored blobs are ignored: the free-space region deforms
        whenever the avatar moves and must never count as an effect."""
        if not (d.appeared or d.vanished):
            return False
        terrain = self._terrain_colors()
        events = [o for o in d.appeared + d.vanished
                  if o.size <= 8 and o.color not in terrain
                  and o.color not in self._ambient]
        if not events:
            return False
        if self.model.avatar_pos is None:
            return True
        ar, ac = self.model.avatar_pos
        r0, c0 = self.perc.playfield[0], self.perc.playfield[2]
        for o in events:
            wr = o.centroid[0] - r0 + self.perc.cam[0]
            wcx = o.centroid[1] - c0 + self.perc.cam[1]
            if abs(wr - ar) > 1.5 or abs(wcx - ac) > 1.5:
                return True
        return False

    def _find_palette_block(self, obs, color, canvas_bbox):
        """A small object of the wanted color OUTSIDE the canvas panel:
        the palette/selector affordance."""
        r0, c0, r1, c1 = canvas_bbox
        best = None
        for o in self.perc.objects(obs.frame):
            if o.color != color or o.size > 80:
                continue
            orow, ocol = o.centroid
            if r0 - 1 <= orow <= r1 + 1 and c0 - 1 <= ocol <= c1 + 1:
                continue                # inside the canvas: not a palette
            if best is None or o.size < best.size:
                best = o
        if best is None:
            return None
        return (int(round(best.centroid[0])), int(round(best.centroid[1])))

    def _world_to_click(self, world_pos):
        r = world_pos[0] - self.perc.cam[0] + self.perc.playfield[0]
        c = world_pos[1] - self.perc.cam[1] + self.perc.playfield[2]
        self._last_click_frame_pos = (int(r), int(c))
        return self.perc.click_xy((r, c))

    # ---------- lifecycle ----------

    def on_episode_start(self, obs):
        self._new_level(obs)

    def on_level_up(self, obs):
        g = getattr(self, "_goal_act", None)
        if g is not None and not g.h.refuted:
            g.h.confirmed = True
            self._goal_confirmed_kinds.append((g.h.kind, g.h.m_color, g.h.note,
                                               g.h.rule(), g.h.mover_bigger()))
            self._hyp("goal_confirmed", gkind=g.h.kind, mover=g.h.m_color,
                      note=g.h.note, spent=g.h.spent)
        if self._seek_color is not None:
            self.goal_color_prior = self._seek_color
        if self._last_click_key is not None:
            # clicking this object class completed a level: strong prior
            self.click_stats[self._last_click_key][0] += 3
        self._new_level(obs)

    def on_game_over(self, obs):
        if self._align_target is not None and self._align_committing:
            # the arrangement was tested and failed: refute this placement
            self._align_refuted.add(self._align_target)
        self._align_target = None
        self._align_committing = False
        self._dock = None
        self._sel_belief = None
        self._dock_prev_cen = None
        self._mb_plan = []
        self._mb_pred = None
        if self._mb_last_positions and self.model:
            # multi-body death: ANY body may have hit the hazard — mark all
            # final cells (distinct-cell color voting keeps this honest)
            for pos in self._mb_last_positions:
                self.model.mark_wall_from_failure(
                    (int(pos[0]), int(pos[1])), (0, 0))
        self._mb_last_positions = None
        """Called after the runner RESETs a death. Micromouse contract:
        every death must make the next attempt strictly smarter."""
        # Escalating trajectory poisoning: dying again on the same final
        # cell marks a LONGER tail of the fatal path as impassable, so an
        # identical plan can never repeat indefinitely (converges).
        # LETHAL CLASS: dying where a small object stood condemns the
        # CLASS (same-appearance objects share semantics) — cell-level
        # poisoning alone re-dies once per instance
        if self._traj and self._prev_objs and not self._timer_active:
            # (timer games: TIME killed us — blaming a bystander object
            # near the final cell teaches the same lie as position
            # poisoning did; observed cn04 regression)
            dr_, dc_ = self._traj[-1]
            r0f3, c0f3 = self.perc.playfield[0], self.perc.playfield[2]
            cr3, cc3 = self.perc.cam
            cands_l = []
            wcs_l = self.model.wall_colors() if self.model else set()
            for o in self._prev_objs:
                if o.size > 60:
                    continue
                if o.color in wcs_l or o.color == self.perc.bg:
                    continue    # walls BLOCK, they don't kill — a wall
                                # fragment near the death site must not
                                # steal the blame from the hazard
                if (self.model is not None
                        and o.shape_key == self.model.avatar_shape_key):
                    continue
                ow = (o.centroid[0] - r0f3 + cr3, o.centroid[1] - c0f3 + cc3)
                dd_ = abs(ow[0] - dr_) + abs(ow[1] - dc_)
                # the recorded trajectory ends one step BEFORE the fatal
                # cell (observe skips GAME_OVER frames): the killer sits
                # one movement-step away, not underfoot
                if dd_ <= 4:
                    cands_l.append((dd_, o.color))
            if cands_l:
                # blame ONLY the nearest candidate: a generous radius
                # once condemned a GOAL that happened to stand near the
                # hazard (observed stb1 — level became unwinnable)
                dmin_ = min(d0 for d0, _ in cands_l)
                for d0, col in cands_l:
                    if d0 == dmin_:
                        # TWO WITNESSES TO CONDEMN (0.21 drift review):
                        # a single death near an innocent object must
                        # not strike its color from the goal set on a
                        # hidden game — one mis-blame can cost the level
                        self._lethal_votes = getattr(
                            self, "_lethal_votes", {})
                        self._lethal_votes[col] = \
                            self._lethal_votes.get(col, 0) + 1
                        if self._lethal_votes[col] >= 2:
                            self._lethal_colors.add(col)
                            self._collect_colors.discard(col)
                            self._hyp("lethal_condemned", color=col,
                                      alternatives=sorted(
                                          {c for _, c in cands_l
                                           if c != col}))
        if self._timer_active:
            # TIMEOUT death: time killed us, not the cell we stood on —
            # poisoning positions here teaches lies (observed: dc22, cn04)
            pass
        elif self.model and self._traj:
            final = self._traj[-1]
            self._death_tail[final] = self._death_tail.get(final, 0) + 1
            tail = self._death_tail[final]
            for pos in self._traj[-tail:]:
                self.model.mark_wall_from_failure(pos, (0, 0))
        elif self.model and self.model.avatar_pos is not None:
            self.model.mark_wall_from_failure(self.model.avatar_pos, (0, 0))
        self._traj = []
        # Click-sequence pruning: THIS sequence ending in THIS click is
        # fatal — ban that continuation after that exact prefix. Only the
        # standalone-fatal click (dies as sequence opener) is banned
        # everywhere. Systematic search over orders converges.
        if self._seq:
            prefix, last = tuple(self._seq[:-1]), self._seq[-1]
            self._banned_next.setdefault(prefix, set()).add(last)
            # position-level ban too (hazard-click games have intrinsically
            # fatal spots); a sequence game can still reach the position in
            # a different order once the opener-ban heuristic below expires
            self._death_clicks.add(last)
            if len(self._death_clicks) > 6:
                # too many "fatal positions" means order matters, not
                # position: it's a sequence game — trust the prefix bans
                self._death_clicks.clear()
        if self._prev_click_class is not None:
            self.click_stats[self._prev_click_class][1] += 1
        self._seq = []
        self._clicked = {}
        self.plan = []
        self.tracker.reset()
        self.prev_frame = obs.frame
        self.model.ctx = {"parity": 0, "prev_axis": None, "prev_action": None}
        self.model.locate_avatar(self._objs_world(obs))
        self._prev_av_pos = None

    def _new_level(self, obs):
        old = self.model
        self.perc.reset()
        self.tracker.reset()
        self.perc.logical(obs.frame)
        self.model = MicromouseModel()
        if old is not None:
            for name, bel in old.motion.items():
                for vec, n in bel.votes.items():
                    self.model.motion[name].votes[vec] += n
            for color, (f, w) in old.color_stats.items():
                self.model.color_stats[color][0] += f
                self.model.color_stats[color][1] += w
            self.model.avatar_shape_key = old.avatar_shape_key
            self.model.shape_op_actions = dict(old.shape_op_actions)
            # a coined law is a fact about the game, not the level
            self.model.laws = dict(old.laws)
            self.model.click_laws = dict(old.click_laws)
            self.model.click_trials = list(old.click_trials)
            self.model.avatar_colors = set(old.avatar_colors)
        self._sync_view(obs)
        self.model.locate_avatar(self._objs_world(obs))
        self.prev_frame = obs.frame
        self.plan = []
        self._seek_color = None
        self._seek_target = None
        self._prev_av_pos = None    # a new level is not a move
        self._gate_fails = {}
        self._gov_state = {}
        # HYPOTHESIS LEDGER: every adopted belief, with the alternatives
        # that were ALSO consistent at adoption time — so a post-run
        # reader can see where multiple genuine answers existed and
        # which one the program chose (and why it was defensible)
        self.hlog = getattr(self, "hlog", [])
        self._amb_frames = {}
        self._ambient = set()
        self._obs_i = 0
        self._dock = None
        self._dock_refuted = set()
        self._mb_plan = []
        self._mb_pred = None
        self._mb_refuted = False
        self._perm_ops = {}
        self._perm_markers = {}
        self._perm_pieces = set()
        self._perm_sol = None
        self._perm_bad = set()
        self._perm_fail_sig = None
        self._perm_btn_tried = {}   # sizes persist: same game, same buttons
        self._perm_hq = None
        self._perm_active_akey = None
        self._perm_need_info = False
        self._perm_lattice = set()
        self._perm_track = {}
        self._peg = {"mode": None, "queue": [], "await": None, "fails": 0}
        self._em.update(path=[], blocked=set(), fails=0, dead=False)
        self._em_prev = None
        self._ov = {"ctl": None, "fails": 0, "dead": False, "cycles": 0,
                    "last": None}
        self._rf = {"pend": None, "dead": False, "submits": 0}
        self._ac = {"tried": {}, "last": None, "dead": False, "probes": 0}
        self._gl = {"last": None, "dead": False, "clicks": {}}
        self._ls = {"cool": 0, "ri": 0, "clicks": 0, "await": None,
                    "dropped": set(), "confirmed": set()}
        self._align_refuted = set()
        self._align_target = None
        self._align_committing = False
        self._tile_fx = {}
        self._carried_box = None
        self._carrier_gate = None
        self._display_votes = {}
        self._touched_cells = set()
        self._touch_tries = {}
        self._touch_gaveup = set()
        self._goal_refuted = set()
        self._goal_deferred = set()
        self._goal_arrivals = {}
        self._t2_clicks = 4
        self._anomalies = []
        self._anom_budget = 12
        self._anom_expect = None
        self._edge_probed = set()
        self._goal_hyps = None
        self._goal_act = None
        self._goal_emap = None
        self._goal_used = 0
        self._goal_confirmed_kinds = getattr(self, "_goal_confirmed_kinds", [])
        self._levels = obs.levels_completed
        self._since_reset = 0  # RESET on a fresh level FULL-resets the game
        # PERSISTENCE UNTIL CONTRADICTED: knowledge carried from earlier
        # levels (motions, rotate/flip semantics, known-inert actions) needs
        # NO re-probing — its first planned use is its test, and the
        # blocked-verdict machinery catches any change for free
        known = self.model.known_motions
        acts = available(obs)
        settled = set(known) | set(self.model.shape_op_actions) \
            | set(self.model.inert_actions)
        self.unresolved = {a.name: PROBE_BUDGET for a in acts
                          if a != GameAction.ACTION6
                          and a.name not in settled}
        self.failed_at = {n: set() for n in self.unresolved}
        self._last_fail = {}
        self.effect_counts = {}
        self.probe_failed_keys = set()
        self._click_queue = []
        self._last_click_key = None
        self._clicked = {}          # world_pos -> times clicked this level
        self._death_clicks = set()  # world positions whose click killed us
        self.transform = TransformSolver()
        self._prev_objs = self.perc.objects(obs.frame)
        self._commit_i = 0
        self._traj = []
        self._death_tail = {}
        self._seq = []
        self._banned_next = {}
        self._refuted_canvases = set()
        self._match_click = None
        self._paint_color = None    # selected brush color (select-then-paint)
        self._palette_click = None
        self._active_hyp = None     # committed (canvas, template) hypothesis
        self._refuted_hyps = set()
        self._hyp_wait = 0
        self._prev_click_pos = None
        self._prev_click_class = None

    def _sync_view(self, obs):
        logical = self.perc.logical(obs.frame)
        self.model.update_view(self.perc.field_view(logical), self.perc.cam)

    # ---------- timers ----------

    def _update_timer(self, obs):
        """Detect a countdown bar: a horizontal uniform run in the frame's
        top/bottom bands whose length only shrinks. Timeout deaths must not
        poison the avatar's position (the cell did not kill us — time did)."""
        try:
            import numpy as _np
            g = _np.asarray(obs.frame[-1], dtype=int) if isinstance(obs.frame, list) \
                else _np.asarray(obs.frame, dtype=int)
            if g.ndim == 3:
                g = g[-1]
            H = g.shape[0]
            for band, rows in (("top", range(0, min(6, H))),
                               ("bot", range(max(0, H - 6), H))):
                for r in rows:
                    row = g[r]
                    # longest uniform run per color in this row
                    best = {}
                    start = 0
                    for i in range(1, len(row) + 1):
                        if i == len(row) or row[i] != row[start]:
                            c = int(row[start])
                            if (i - start) > best.get(c, 0):
                                best[c] = i - start
                            start = i
                    for c, ln in best.items():
                        if ln < 3 or ln == len(row):
                            continue
                        key = (band, r, c)
                        hist = self._timer_runs.setdefault(key, [])
                        if not hist or hist[-1] != ln:
                            hist.append(ln)
                            if len(hist) > 8:
                                del hist[0]
            # a timer: some tracked run has a strictly-monotone SEGMENT
            # of >=4 lengths. Segment, not whole history — the bar
            # REFILLS on death/level reset (sawtooth), and demanding
            # global monotonicity made confirmed timers flicker OUT of
            # the set, opening the avatar-adoption gate to a countdown
            # bar mid-run (observed cn04). Once confirmed, ALWAYS a
            # timer: objects do not stop being timers.
            for (band, r, c), h in self._timer_runs.items():
                run = 1
                for a, b in zip(h, h[1:]):
                    run = run + 1 if b < a else 1
                    if run >= 4:
                        self._timer_confirmed.add((r, c))
                        break
                run = 1
                for a, b in zip(h, h[1:]):
                    run = run + 1 if b > a else 1
                    if run >= 4:
                        self._timer_confirmed.add((r, c))
                        break
            self._timer_bars = set(self._timer_confirmed)
            self._timer_active = bool(self._timer_bars)
        except Exception:
            pass

    # ---------- observation ----------

    def observe(self, action, obs, changed):
        if obs.levels_completed != self._levels:
            self.prev_frame = obs.frame
            return  # on_level_up rebuilds; never learn across a boundary
        if obs.state == GameState.GAME_OVER:
            # death animations / screen wipes are not physics; on_game_over
            # will resync after the runner resets
            self.prev_frame = obs.frame
            return
        if action == GameAction.RESET:
            # deliberate restart: a teleport, not physics — never learn it
            self.prev_frame = obs.frame
            self.plan = []
            self._traj = []
            self._seq = []
            self._clicked = {}
            self.tracker.reset()
            self.model.ctx = {"parity": 0, "prev_axis": None, "prev_action": None}
            self.model.locate_avatar(self._objs_world(obs))
            self._prev_av_pos = None
            return
        name = action.name
        protect = None
        if self.model.avatar_pos is not None:
            protect = (self.model.avatar_pos[0] - self.perc.cam[0]
                       + self.perc.playfield[0],
                       self.model.avatar_pos[1] - self.perc.cam[1]
                       + self.perc.playfield[2])
        self._update_timer(obs)
        # RETROACTIVE IDENTITY REVISION (evidence-order sensitivity): a
        # timer bar can be adopted as the avatar BEFORE enough shrink
        # observations prove it is a timer, and sticky identity then
        # preserves the lie forever (observed cn04: avatar = the
        # countdown bar, rotation never learnable). A belief must remain
        # revisable when later evidence reclassifies its object.
        if (self._timer_bars and self.model is not None
                and self.model.avatar_shape_key is not None
                and self.model.avatar_pos is not None):
            av_col_t = self.model.avatar_shape_key[0]
            fr_t = (self.model.avatar_pos[0] - self.perc.cam[0]
                    + self.perc.playfield[0])
            if any(abs(fr_t - r) <= 2 and c == av_col_t
                   for r, c in self._timer_bars):
                self.model.avatar_shape_key = None
                self.model.avatar_pos = None
        if self._switch_cool > 0:
            self._switch_cool -= 1
        d = self.perc.diff(self.prev_frame, obs.frame, protect=protect)
        self._sync_view(obs)
        if (name not in ("ACTION6", "RESET")
                and name not in self.model.known_motions
                and name not in self.model.shape_op_actions
                and d.raw_changed >= 15):
            self._bigfx[name] = self._bigfx.get(name, 0) + 1
        cur_objs = self.perc.objects(obs.frame)
        # _prev_objs is overwritten mid-observe by the op-learning
        # section; consumers below need the TRUE pre-action objects
        prev_objs_snapshot = list(self._prev_objs or [])
        if name != "ACTION6":
            self.transform.update(name, self._prev_objs, cur_objs)
        # PERMUTATION-OPERATOR OBSERVATION: a click that rearranges SEVERAL
        # objects at once is a button applying a fixed operator over cells.
        # Loop shifts can jump farther than the tracker's match radius, so
        # the correspondence is built here: per color, a greedy nearest
        # bijection between the previous and current object positions.
        # Observations MERGE across states (Law 1: uncontested evidence is
        # trusted; a contradiction wipes the operator for relearning).
        if (name == "ACTION6" and self._prev_objs
                and getattr(action, "action_data", None) is not None):
            r0f_, c0f_ = self.perc.playfield[0], self.perc.playfield[2]
            cr_, cc_ = self.perc.cam

            def _wcen(o):
                return (round(o.centroid[0] - r0f_ + cr_),
                        round(o.centroid[1] - c0f_ + cc_))
            moves = []
            by_color_prev = {}
            for o in self._prev_objs:
                by_color_prev.setdefault((o.color, o.size), []).append(o)
            by_color_cur = {}
            for o in cur_objs:
                by_color_cur.setdefault((o.color, o.size), []).append(o)
                if (o.color, o.size) in self._perm_pieces:
                    # pieces only ever sit on tracks; the lattice must
                    # accumulate HERE, unconditionally — inside the
                    # solver it starved whenever the mode gate closed
                    self._perm_lattice.add(_wcen(o))
            for key2, prevs in by_color_prev.items():
                curs = by_color_cur.get(key2, [])
                if len(prevs) != len(curs) or not prevs:
                    continue
                pen = [(_wcen(o), o) for o in prevs]
                cen = [(_wcen(o), o) for o in curs]
                if {p2 for p2, _ in pen} == {c2 for c2, _ in cen}:
                    continue        # nothing of this class moved
                remaining = list(cen)
                for pp, _ in pen:
                    best_i = min(range(len(remaining)),
                                 key=lambda i2: abs(remaining[i2][0][0] - pp[0])
                                 + abs(remaining[i2][0][1] - pp[1]))
                    tgt = remaining.pop(best_i)[0]
                    if tgt != pp:
                        moves.append((pp, tgt, key2))
            # ADJACENT MOVES ONLY: when same-class pieces sit on
            # CONSECUTIVE loop slots and all shift one slot, the greedy
            # match self-anchors the interior at distance 0 and invents a
            # phantom tail->head jump spanning the whole run. A real
            # single-click move is one slot; anything longer is an
            # artifact. Threshold = the smallest step seen (this click or
            # already learned) — jumps are dropped, not learned wrong.
            if moves:
                dists = [abs(a2[0] - b2[0]) + abs(a2[1] - b2[1])
                         for a2, b2, _ in moves]
                thr = self._perm_thr(min(dists))
                moves = [m for m, dd in zip(moves, dists) if dd <= thr]
            # identify the clicked button FIRST (independent of what the
            # object segmentation saw move)
            bkey, bsz = None, 1 << 30
            try:
                ax = int(action.action_data.x)
                ay = int(action.action_data.y)
                frq = ay // max(self.perc.cell, 1)
                fcq = ax // max(self.perc.cell, 1)
                # the button = SMALLEST non-background object under
                # the click (a huge bg-colored frame contains every
                # point and would merge all buttons into one key)
                for o in cur_objs:
                    if (o.bbox[0] <= frq <= o.bbox[2]
                            and o.bbox[1] <= fcq <= o.bbox[3]
                            and o.color != self.perc.bg
                            and o.size < bsz):
                        bsz = o.size
                        bkey = (o.color, (round(o.centroid[0]),
                                          round(o.centroid[1])))
            except Exception:
                bkey = None
            # CELL-OCCUPANCY EVIDENCE: on crowded loops adjacent
            # same-color pieces MERGE and the object bijection goes
            # blind (class counts differ -> classes skipped). The frame
            # itself does not lie: read each known loop cell's occupant
            # color before/after the click; for each color an
            # unambiguous single departure->arrival at one-slot
            # distance is a trusted mapping.
            # TRACK ACCUMULATION + SHIFT CERTIFICATION: the cells this
            # button's clicks change, read from pixels over the piece
            # lattice, converge to its full track; the track's geometry
            # plus one clean before/after comparison certifies the whole
            # operator (see _perm_shift_cert).
            if bkey is not None and self._perm_lattice:
                try:
                    pg = last_grid(self.prev_frame)
                    cg = last_grid(obs.frame)
                    chg = {w for w in self._perm_lattice
                           if self._perm_grid_col(pg, w)
                           != self._perm_grid_col(cg, w)}
                    if len(chg) >= 3:
                        if bkey in self._perm_track:
                            # a REPEAT click on the same button moving
                            # its track is verification-by-use too —
                            # fragment noise alone starved lp85 L3's
                            # mode activation for a thousand actions
                            ent2 = self._perm_ops.get(bkey)
                            if ent2 is not None:
                                ent2["conf"] = ent2.get("conf", 0) + 1
                        tr = self._perm_track.setdefault(bkey, set())
                        tr |= chg
                        self._perm_shift_cert(bkey, obs, (frq, fcq))
                except Exception:
                    pass
            occ = []
            if (bkey is not None and self._perm_ops
                    and not self._perm_ops.get(bkey, {}).get("cert")):
                try:
                    occ = self._perm_occ_moves(obs, self._perm_thr(4))
                except Exception:
                    occ = []
            combined = {}
            conflicted = set()
            for a2, b2 in ([(m[0], m[1]) for m in moves] + occ):
                if combined.get(a2, b2) != b2:
                    conflicted.add(a2)
                combined[a2] = b2
            for a2 in conflicted:
                combined.pop(a2, None)
            op = op_from_moves(list(combined.items()))
            if getattr(self, "_perm_debug", False) and (moves or occ):
                print(f"    LEARN bkey-> obj={[(m[0], m[1]) for m in moves]} "
                      f"occ={occ}", flush=True)
            if op:
                try:
                    if bkey is not None:
                        if (bkey not in self._perm_ops
                                and len(op) < 2):
                            raise ValueError("one move: not yet a button")
                        if self._perm_ops.get(bkey, {}).get("cert"):
                            # a shift-certified operator outranks noisy
                            # per-piece correspondence fragments
                            self._perm_pieces |= {c2 for _, _, c2 in moves}
                            raise ValueError("certified: keep clean op")
                        ent = self._perm_ops.setdefault(
                            bkey, {"op": {}, "click": (frq, fcq)})
                        contradicted = False
                        for a2, b2 in list(op.items()):
                            old_b = ent["op"].get(a2)
                            if old_b is not None and old_b != b2:
                                # both beliefs are suspect for THIS cell:
                                # drop them and let a clean observation
                                # re-teach it. Wiping the whole operator
                                # here erased 20 good mappings per bad one.
                                del ent["op"][a2]
                                del op[a2]
                                contradicted = True
                                ent["bad"] = ent.get("bad", 0) + 1
                        grew = any(a2 not in ent["op"] for a2 in op)
                        if not contradicted:
                            # VERIFIED BY USE: a second click on the SAME
                            # button that confirms/extends its operator.
                            # Real buttons sit still and get re-clicked;
                            # click-noise "buttons" (a moving rope) never
                            # produce a second consistent observation.
                            ent["conf"] = ent.get("conf", 0) + 1
                        ent["op"].update(op)
                        self._perm_pieces |= {c2 for _, _, c2 in moves}
                        self._perm_btn_sizes.add(bsz)
                        if grew:
                            # still teaching: refill its completion budget
                            self._perm_btn_tried.pop(bkey, None)
                        # PERSISTENCE UNTIL CONTRADICTED: nearly every
                        # click GROWS an operator (mappings are only seen
                        # for occupied cells), so growth must not abandon
                        # the executing plan — that livelocked L2 at one
                        # click per solve. Growth does invalidate the
                        # failure cache; only contradiction kills the plan.
                        if contradicted:
                            self._perm_contra = getattr(
                                self, "_perm_contra", 0) + 1
                            self._perm_sol = None
                        if contradicted or grew:
                            self._perm_fail_sig = None
                except Exception:
                    pass
        # TILE-EFFECT LEARNING (carrier games): stepping onto a special
        # tile changes a FAR display object (the carried shape). One
        # observation teaches the tile class's attribute: rotation, color,
        # or form (Law 1).
        if (name in self.model.known_motions and self._prev_objs
                and self.model.avatar_pos is not None):
            try:
                av_r = self.model.avatar_pos[0]
                av_c = self.model.avatar_pos[1]
                far_changes = []
                for po in self._prev_objs:
                    if abs(po.centroid[0] - av_r) + abs(po.centroid[1] - av_c) < 15:
                        continue
                    match = [co for co in cur_objs
                             if abs(co.centroid[0] - po.centroid[0]) < 3
                             and abs(co.centroid[1] - po.centroid[1]) < 3
                             and co.size >= 3]
                    if not match or po.size < 3:
                        continue
                    co = match[0]
                    if co.shape_key != po.shape_key and co.size == po.size:
                        # displays TRANSFORM (size-preserving); counters
                        # and timers grow/shrink and are not displays
                        far_changes.append((po, co))
                # ambiguity kills learning: exactly ONE display-like change
                if len(far_changes) == 1:
                    po, co = far_changes[0]
                    # a DISPLAY is compact: reject region-sized objects
                    bh = po.bbox[2] - po.bbox[0] + 1
                    bw = po.bbox[3] - po.bbox[1] + 1
                    fh2 = self.perc.playfield[1] - self.perc.playfield[0]
                    fw2 = self.perc.playfield[3] - self.perc.playfield[2]
                    if bh * bw <= 0.12 * fh2 * fw2:
                        # the CARRIED display = box that changes most often,
                        # believed only after >=2 votes with a unique top
                        bk = (round(po.bbox[0] / 4), round(po.bbox[1] / 4))
                        n2, _ = self._display_votes.get(bk, (0, None))
                        self._display_votes[bk] = (n2 + 1, co.bbox)
                        top = max(v[0] for v in self._display_votes.values())
                        tops = [v for v in self._display_votes.values()
                                if v[0] == top]
                        if top >= 2 and len(tops) == 1 \
                                and self._display_votes[bk][0] == top:
                            self._carried_box = self._display_votes[bk][1]
                    else:
                        po = None
                    tile = None
                    # the avatar now COVERS the tile: find it in the
                    # PREVIOUS frame's objects at the current position
                    for o in self._prev_objs:
                        if (o.bbox[0] - 2 <= av_r <= o.bbox[2] + 2
                                and o.bbox[1] - 2 <= av_c <= o.bbox[3] + 2
                                and o.color != self.perc.bg
                                and o.size <= 60
                                and abs(o.centroid[0] - av_r)
                                + abs(o.centroid[1] - av_c) < 8):
                            tile = o
                            break
                    if tile is not None and po is not None:
                        pcells = po.shape_key[1]
                        ccells = co.shape_key[1]
                        if po.color != co.color and pcells == ccells:
                            self._tile_fx.setdefault(tile.color, "color")
                        elif po.color == co.color:
                            fx = "form"
                            for opf in OPS.values():
                                if opf(pcells) == ccells:
                                    fx = "rot"
                                    break
                            self._tile_fx.setdefault(tile.color, fx)
                        # color AND cells both changed: ambiguous, skip
            except Exception:
                pass
        self._prev_objs = cur_objs
        objs_w = self._objs_world(obs)
        # countdown bars shrink in response to EVERY action — the most
        # control-correlated object on screen. They are clocks, not avatars:
        # exclude them from identity, attribution, and motion learning.
        # (observed: cn04's model adopted the timer bar as the avatar)
        if self._timer_bars:
            timer_colors = {c for _, c in self._timer_bars}
            r0f = self.perc.playfield[0]

            def _is_timer(o):
                if o[1] not in timer_colors:
                    return False
                frow = round(o[4][0] - self.perc.cam[0] + r0f)
                return any(abs(frow - r) <= 2 for r, c in self._timer_bars
                           if c == o[1])
            objs_w = [o for o in objs_w if not _is_timer(o)]
            d.moved = [m for m in d.moved if m[0] not in timer_colors]
            d.appeared = [a for a in d.appeared
                          if getattr(a, "color", None) not in timer_colors]
            d.vanished = [v for v in d.vanished
                          if getattr(v, "color", None) not in timer_colors]
        if (name == "ACTION6" and self._last_click_frame_pos is not None
                and self.prev_frame is not None):
            fr, fc = self._last_click_frame_pos
            try:
                prev_l = self.perc.logical(self.prev_frame)
                cur_l = self.perc.logical(obs.frame)
                if 0 <= fr < cur_l.shape[0] and 0 <= fc < cur_l.shape[1]:
                    before, after = int(prev_l[fr, fc]), int(cur_l[fr, fc])
                    if before != after and after != self.perc.bg:
                        # clicking a piece recolors it: that recolor is the
                        # selection livery — the controlled piece wears it
                        self._sel_color = after
            except Exception:
                pass
            self._last_click_frame_pos = None
        assignments = self.tracker.update(objs_w, name)

        # CO-MOVEMENT LEDGER: tracks that displace by the same nonzero
        # vector in the same step belong to the same rigid body (a piece
        # and its ports). Geometric adjacency lies when pieces touch;
        # moving together never does.
        self._tid_by_key = {
            (objs_w[i][0], (round(objs_w[i][4][0]), round(objs_w[i][4][1]))): tid
            for i, tid in assignments.items() if i < len(objs_w)}
        movers = [(tid, (round(tr.last_disp[0]), round(tr.last_disp[1])))
                  for tid, tr in self.tracker.tracks.items()
                  if abs(tr.last_disp[0]) + abs(tr.last_disp[1]) >= 1]
        for i in range(len(movers)):
            for j in range(i + 1, len(movers)):
                if movers[i][1] == movers[j][1]:
                    key = frozenset((movers[i][0], movers[j][0]))
                    self._comove[key] = self._comove.get(key, 0) + 1

        # SHAPE-OP DETECTION: an action that maps some object's own cells to
        # a dihedral transform of themselves, IN PLACE, rotates/flips it.
        # Any stable track qualifies (the controlled piece may be tracked by
        # a rotation-invariant part, while its body is a separate track).
        # Guard against shape-shifting animations: the op must be DOMINANT
        # for this action (>= double any other action's votes for it).
        if name not in ("ACTION6", "RESET"):
            for tr in self.tracker.tracks.values():
                if (len(tr.shapes) < 2 or tr.is_autonomous
                        or tr.is_oscillating or tr.size > 200):
                    continue
                if abs(tr.last_disp[0]) + abs(tr.last_disp[1]) > 3:
                    continue        # translated: not an in-place transform
                prev_k, cur_k = tr.shapes[-2], tr.shapes[-1]
                if prev_k == cur_k or prev_k[0] != cur_k[0]:
                    continue
                for opn, opf in OPS.items():
                    if opf(prev_k[1]) == cur_k[1]:
                        v = self._shape_op_votes.setdefault(name, Counter())
                        v[opn] += 1
                        rival = max((c.get(opn, 0) for a, c in
                                     self._shape_op_votes.items()
                                     if a != name), default=0)
                        if v[opn] >= 2 and v[opn] >= 2 * rival:
                            self.model.shape_op_actions[name] = opn
                            self.model.motion.pop(name, None)
                            self.unresolved.pop(name, None)
                        break
                break               # one voting object per step is enough

        # coherent global motion (a hovering/wobbling panel shifts all its
        # pieces by one vector) is scenery, not the effect of our action --
        # and must never elect an avatar
        if len(d.moved) >= 3:
            vec_counts = Counter(m[2] for m in d.moved)
            top_vec, n_top = vec_counts.most_common(1)[0]
            if n_top >= 3 and n_top >= 0.8 * len(d.moved):
                d.moved = [m for m in d.moved if m[2] != top_vec]

        # autonomous movers (projectiles, patrols) are not our doing:
        # remove their motion before attributing effects to the action
        if d.moved:
            d.moved = [m for m in d.moved
                       if not self.tracker.explains(m[0], m[1], m[2])]

        # attribution consistency: when several things move (mirrored twin
        # avatars), prefer the mover matching the established avatar
        if len(d.moved) > 1 and self.model.avatar_shape_key is not None:
            av_color = self.model.avatar_shape_key[0]
            av_size = len(self.model.avatar_shape_key[1])
            mine = [m for m in d.moved if m[0] == av_color and m[1] == av_size]
            if mine:
                d.moved = mine

        # shape-changing movers (animated avatars) never appear in the
        # exact-shape diff — synthesize their motion from track continuity
        have = {(m[0], m[1]) for m in d.moved}
        for t in self.tracker.tracks.values():
            if t.missed or t.is_autonomous or t.is_oscillating:
                continue
            vec = (round(t.last_disp[0]), round(t.last_disp[1]))
            if vec != (0, 0) and (t.color, t.size) not in have:
                d.moved.append((t.color, t.size, vec))

        # RECURRING AMBIENT EVENT CLASSES: a color that appears/vanishes
        # in frame after frame regardless of what we do (fog redraw,
        # shimmer, drips) is ambience, not an effect — and if it keeps
        # counting as "non-local change", the blocked verdict never fires
        # and walls are never learned in its presence (observed ls20:
        # fog fragments starved wall learning; the agent pressed into
        # the same wall forever)
        self._obs_i += 1
        for o_amb in list(d.appeared) + list(d.vanished):
            if o_amb.size <= 8:
                fr_ = self._amb_frames.setdefault(o_amb.color, set())
                fr_.add(self._obs_i)
                if len(fr_) >= 5:
                    self._ambient.add(o_amb.color)

        # Blocked-move verdict: nothing moved AND every appear/vanish is
        # avatar-local (glitter/animation on a bump is not a world effect)
        blocked = not d.moved and d.camera_shift == (0, 0) \
            and not self._nonlocal_change(d)

        # LAYOUT-SWITCH confirmation: our supervised click either
        # rewrote the play area (button confirmed -> all terrain
        # knowledge is stale, forget it) or did nothing (drop candidate)
        ls_aw = self._ls.get("await")
        if ls_aw is not None and name == "ACTION6":
            self._ls["await"] = None
            if d.field_changed >= 8:
                self._ls["confirmed"].add(ls_aw)
                self.model.forget_terrain()
                self.plan = []
                # give the walker time on the NEW layout before the
                # supervisor may click it away
                self._ls["cool"] = 16
            elif ls_aw not in self._ls["confirmed"]:
                # a once-confirmed button stays a button — one quiet
                # re-click (wrapped cycle, animation timing) must not
                # orphan it
                self._ls["dropped"].add(ls_aw)
        self.model.observe_transition(name, d, objs_w, blocked=blocked)
        self.model.locate_avatar(objs_w)
        if (self.model.avatar_shape_key is not None
                and not any(o[0] == self.model.avatar_shape_key for o in objs_w)):
            if self._refute_playfield(obs):
                objs_w = self._objs_world(obs)
                self.model.locate_avatar(objs_w)
        self.model.finish_transition()
        cctx = getattr(self, "_click_ctx", None)
        if name == "ACTION6" and cctx is not None:
            self._click_ctx = None
            if self.model.record_click(cctx[0], cctx[1], cctx[2],
                                       self.model.avatar_pos, cctx[3]):
                if self.model.avatar_shape_key is not None:
                    self.tracker.bind_to(self.model.avatar_pos,
                                         self.model.avatar_shape_key[0])
                self.plan = []
        for ev, law, n in self.model.law_events:
            self._hyp("law_" + ev, group=list(law.group), feature=law.feature,
                      value=law.value, blocks_when=law.blocks_when, witnesses=n,
                      outcome=({str(k): list(v) for k, v in law.outcome.items()}
                               if law.outcome else None))
            if ev == "unknown":
                continue
            # NEW INFORMATION: RETRY — targets abandoned under the old
            # model (plans killed by refusals the law now predicts) are
            # live again
            self._touch_tries = {}
            self._touch_gaveup = set()
            self._goal_refuted = set()
            self._goal_deferred = set()
        self.model.law_events.clear()
        # COLLECTIBLE CLASS (Law 1, one observation): moving onto a small
        # object that VANISHES — no level-up, no death — marks its color
        # as a collectible class: visit EVERY instance, not the odd one.
        # NOTE: computed from _prev_objs directly, NOT the diff — the
        # anti-glitter protection suppresses diff events in the avatar's
        # own 3x3 zone, which is precisely where collection happens.
        if (name not in ("ACTION6", "RESET")
                and self.model.avatar_pos is not None and prev_objs_snapshot):
            r0f2, c0f2 = self.perc.playfield[0], self.perc.playfield[2]
            cr2, cc2 = self.perc.cam
            avr, avc = self.model.avatar_pos
            from collections import Counter as _C9
            prev_n = _C9((o.color, o.size) for o in prev_objs_snapshot)
            cur_n = _C9((o.color, o.size) for o in cur_objs)
            # COLLECTIBLES ARE SPARSE: a color covering a terrain-scale
            # share of the playfield is ground/water, whose fragments
            # "vanish" as the avatar passes (observed mz02: water adopted
            # as a collectible class at step 5 — a false belief the
            # hypothesis ledger surfaced)
            _tot_px = max(1, sum(o.size for o in prev_objs_snapshot))
            _col_px = {}
            for o in prev_objs_snapshot:
                _col_px[o.color] = _col_px.get(o.color, 0) + o.size
            for o in prev_objs_snapshot:
                if o.size > 60 or o.color in self._lethal_colors:
                    continue
                if _col_px.get(o.color, 0) > 0.15 * _tot_px:
                    continue
                if (self.model.avatar_shape_key is not None
                        and o.color == self.model.avatar_shape_key[0]):
                    continue
                ow = (o.centroid[0] - r0f2 + cr2, o.centroid[1] - c0f2 + cc2)
                if (abs(ow[0] - avr) + abs(ow[1] - avc) <= 2
                        and cur_n[(o.color, o.size)]
                        < prev_n[(o.color, o.size)]):
                    if o.color not in self._collect_colors:
                        self._hyp("collectible_adopted", color=o.color)
                    self._collect_colors.add(o.color)

        # identity by continuity: the control-correlated track backs the
        # shape-keyed avatar; the tracker keeps that choice STICKY so
        # mirrored-twin games don't flip identity every step
        av_t = self.tracker.avatar_track()
        # (a track that was NOT matched this frame carries a stale
        # last_obj: rebinding to it drags the avatar back to where it no
        # longer is — sx02: the body wrapped to the far edge, the model
        # was pulled back to the near edge, and the coined wrap law was
        # refuted by its own ghost)
        if av_t is not None and av_t.last_obj is not None and (av_t.missed == 0 or self.REBIND_STALE):
            # ROTATION PRESERVES THE SQUARES: a shape-op action changes
            # the body's shape key (killing its track) while small
            # rotation-symmetric fragments keep theirs, so the surviving
            # control-correlated track MIGRATES identity onto a fragment
            # (observed cn04 L2: one rotate press rebound the 90-cell
            # piece to a 3x3 port and the dock never re-engaged). A
            # rebind that shrinks the avatar >3x on the very step a
            # shape-op ran is that migration — identity stays with the
            # body: same color, similar size, nearest the old position.
            new_key = av_t.last_obj[0]
            old_key = self.model.avatar_shape_key
            aname = getattr(action, "name", None)
            if (aname in self.model.shape_op_actions
                    and old_key is not None
                    and len(old_key[1]) > 3 * max(len(new_key[1]), 1)):
                oldc, oldn = old_key[0], len(old_key[1])
                op = self.model.avatar_pos
                cands = []
                r0f3, c0f3 = self.perc.playfield[0], self.perc.playfield[2]
                cr3, cc3 = self.perc.cam
                for o in self.perc.objects(obs.frame):
                    if o.color != oldc:
                        continue
                    if not (0.4 * oldn <= o.size <= 2.5 * oldn):
                        continue
                    ow = (o.centroid[0] - r0f3 + cr3,
                          o.centroid[1] - c0f3 + cc3)
                    d = (abs(ow[0] - op[0]) + abs(ow[1] - op[1])
                         if op is not None else 0)
                    cands.append((d, o))
                if cands:
                    d, o = min(cands, key=lambda t: t[0])
                    if d <= 20:
                        self.model.avatar_shape_key = o.shape_key
                        self.model.avatar_pos = (
                            int(round(o.centroid[0] - r0f3 + cr3)),
                            int(round(o.centroid[1] - c0f3 + cc3)))
                        self.tracker._avatar_id = None
                    else:
                        self.model.avatar_shape_key = new_key
                        self.model.avatar_pos = av_t.last_obj[3]
                else:
                    self.model.avatar_shape_key = new_key
                    self.model.avatar_pos = av_t.last_obj[3]
            else:
                self.model.avatar_shape_key = new_key
                self.model.avatar_pos = av_t.last_obj[3]

        if self.model.avatar_pos is not None and (
                not self._traj or self._traj[-1] != self.model.avatar_pos):
            self._traj.append(self.model.avatar_pos)
        # A GATED GOAL YIELDS TO FREE INFORMATION: an entry refused at
        # the sought goal (the avatar aimed at it and did not arrive) is
        # not a wall — it is a CONDITION we do not meet yet (state-match
        # pads, locked doors). Two refusals gate the goal color; touching
        # a new interactable clears all gates for a retry — the delivery
        # loop emerges from existing priors. The refusal animation may
        # light distant pixels, so the honest signal is simply the
        # avatar not arriving.
        pav = getattr(self, "_prev_av_pos", None)
        if (pav is not None and pav == self.model.avatar_pos
                and self._seek_target is not None):
            vec_g = self.model.known_motions.get(name)
            if vec_g is not None:
                dest_g = (pav[0] + vec_g[0], pav[1] + vec_g[1])
                if (abs(dest_g[0] - self._seek_target[0])
                        + abs(dest_g[1] - self._seek_target[1])) <= 2:
                    self._gate_fails[self._seek_color] = \
                        self._gate_fails.get(self._seek_color, 0) + 1
                    self._gate_step = self._obs_i
        # ARRIVAL WITHOUT EFFECT REFUTES A GOAL: standing on the sought
        # object (it did not vanish, no level-up, it is still there or
        # elsewhere) twice means it is not the goal — a teleport anchor
        # adopted as "the unique small object" ping-ponged the avatar
        # forever (sx06) while the true goal sat in the alternatives
        if (self._seek_target is not None and self.model.avatar_pos is not None
                and tuple(self.model.avatar_pos) == tuple(self._seek_target)
                and pav != self.model.avatar_pos
                and self._seek_color not in self._collect_colors):
            arr = self._goal_arrivals = getattr(self, "_goal_arrivals", {})
            arr[self._seek_color] = arr.get(self._seek_color, 0) + 1
            if arr[self._seek_color] >= 2 and self._seek_color not in self._goal_refuted:
                self._goal_refuted.add(self._seek_color)
                self._hyp("goal_refuted", color=self._seek_color,
                          reason="reached_twice_no_effect")
                self._hyp_last_goal = None
                self._seek_target = None
                self.plan = []
        # THE ANOMALY LEDGER (Tushar, 2026-09-04): an observation that fits
        # no relation the program can name is neither rounded to the
        # nearest known concept nor dismissed as noise — it is recorded as
        # "unknown mechanic here" and re-tested before anything is built
        # on it (see _retest_anomaly). Two kinds are detected here:
        # an UNEXPLAINED MOVE (a confirmed step action produced a
        # different vector: wrap-arounds, teleports, knock-backs) and
        # CO-MOTION (a non-avatar object moved with the avatar's vector
        # from the cell in front of it: pushing, carrying, mirrored twins).
        self._detect_anomalies(name, d, pav, prev_objs_snapshot, cur_objs)
        self._prev_av_pos = self.model.avatar_pos

        # EDGE-MAZE accounting (self-anchored): the archetype tracks the
        # DISTINCTIVE BODY itself — the unique-color block whose position
        # changes under actions — because the generic mover attribution
        # is systematically poisoned here (wake phantoms, facing-dot edge
        # flips). It learns its own action->direction map and votes edge
        # midpoint colors from its own positions.
        em = self._em
        if not em["dead"] and name in ("ACTION1", "ACTION2",
                                       "ACTION3", "ACTION4"):
            objs_em = self.perc.objects(obs.frame)
            from collections import Counter as _CE
            colcnt = _CE(o.color for o in objs_em)
            r0e, c0e = self.perc.playfield[0], self.perc.playfield[2]
            cre, cce = self.perc.cam
            cur_map = {}
            for o in objs_em:
                if colcnt[o.color] == 1 and o.size >= 4:
                    cur_map[o.color] = (
                        int(round(o.centroid[0] - r0e + cre)),
                        int(round(o.centroid[1] - c0e + cce)))
            prev_map = em.get("umap") or {}
            # anchor color: a unique-color block that moves under actions
            if em.get("acolor") is None:
                for col, pos in cur_map.items():
                    pp = prev_map.get(col)
                    if pp is not None and pp != pos and                             4 <= abs(pos[0] - pp[0]) + abs(pos[1] - pp[1]) <= 16:
                        em.setdefault("avotes", {})
                        em["avotes"][col] = em["avotes"].get(col, 0) + 1
                        if em["avotes"][col] >= 2:
                            em["acolor"] = col
            ac = em.get("acolor")
            if ac is not None and ac in cur_map and ac in prev_map:
                pp, pos = prev_map[ac], cur_map[ac]
                dvec = (pos[0] - pp[0], pos[1] - pp[1])
                mag = abs(dvec[0]) + abs(dvec[1])
                lg = self.perc.logical(obs.frame)

                def _mcol(mr, mc):
                    fr, fc = mr - cre + r0e, mc - cce + c0e
                    if 0 <= fr < lg.shape[0] and 0 <= fc < lg.shape[1]:
                        return int(lg[fr, fc])
                    return None
                if mag >= 4 and mag % 2 == 0 and 0 in dvec:
                    # clean axis move: learn direction + open vote
                    em.setdefault("dirs", {}).setdefault(name, {})
                    em["dirs"][name][dvec] =                         em["dirs"][name].get(dvec, 0) + 1
                    mc_ = _mcol(pp[0] + dvec[0] // 2, pp[1] + dvec[1] // 2)
                    if mc_ is not None:
                        em["open"][mc_] = em["open"].get(mc_, 0) + 1
                elif mag == 0:
                    # blocked: needs the learned direction for this action
                    dd = em.get("dirs", {}).get(name)
                    if dd:
                        dv, nv = max(dd.items(), key=lambda kv: kv[1])
                        if nv >= 2:
                            mc_ = _mcol(pp[0] + dv[0] // 2,
                                        pp[1] + dv[1] // 2)
                            if mc_ is not None:
                                em["closed"][mc_] =                                     em["closed"].get(mc_, 0) + 1
            em["umap"] = cur_map

        # scientist loop: a match-task click that changed nothing refutes
        # the hypothesis that this panel is the editable canvas (raw pixel
        # change is the honest signal — canvas cells can be any size)
        if self._match_click is not None and name == "ACTION6":
            canvas_bbox, (mr, mc), want = self._match_click
            if d.raw_changed == 0:
                self._refuted_canvases.add(canvas_bbox)
            else:
                # the painted cell's resulting color reveals the brush
                self._paint_color = int(self.perc.logical(obs.frame)[mr, mc])
            self._match_click = None
        elif self._palette_click is not None and name == "ACTION6":
            self._paint_color = self._palette_click
            self._palette_click = None

        if name == "ACTION6" and self._last_click_key is not None:
            st = self.click_stats[self._last_click_key]
            st[1] += 1
            # click credit is object-level: a toggle flips a cell between
            # RECURRING colors (pixel-novelty calls that routine), but the
            # cell's object class changes -> appear/vanish events
            st[0] += int(bool(d.world_changed or d.moved
                              or d.appeared or d.vanished))

        if not blocked:
            if name in self.unresolved:
                vec = self.model.known_motions.get(name)
                votes = self.model.motion[name].votes.get(vec, 0) if vec else 0
                # a single AXIS-ALIGNED vote resolves provisionally
                # (verification-by-use); a diagonal/large first vote is
                # suspicious (rotations masquerade as translations) and
                # keeps its probe until a second sample settles it
                if vec and (votes >= 2 or vec[0] == 0 or vec[1] == 0):
                    if self.model.is_slide(name):
                        # a blocked SLIDE says the ADJACENT cell is the
                        # wall, not the cell a full-length slide away
                        vec = (max(-1, min(1, vec[0])),
                               max(-1, min(1, vec[1])))
                    for pos in self.failed_at.get(name, ()):
                        if pos is not None:
                            self.model.mark_wall_from_failure(pos, vec)
                    del self.unresolved[name]
                else:
                    # the action clearly does SOMETHING: refund the attempt
                    self.unresolved[name] = min(self.unresolved[name] + 1,
                                                PROBE_BUDGET)
                    if not d.moved:
                        self.effect_counts[name] = \
                            self.effect_counts.get(name, 0) + 1
                        if self.effect_counts[name] >= 3:
                            del self.unresolved[name]  # repeatable effect action
        else:
            # a failed attempt: remember WHERE it failed, so probing
            # diversifies away from this wall line
            fail_pos = self.model.avatar_pos
            self.failed_at.setdefault(name, set()).add(fail_pos)
            self._last_fail[name] = fail_pos
            self.probe_failed_keys.add((name, fail_pos))
            # a refusal the adopted law PREDICTED is part of the plan
            # (a deliberate bump to flip parity, or an axis reset)
            if self.plan and not self.model.last_stay_explained:
                self.plan = []
        self.prev_frame = obs.frame

    # ---------- goal induction from free information ----------

    def _goals_step(self, obs):
        """THE FIRST-FRAME READER: correspondences -> goal hypotheses ->
        effect learning -> greedy descent (agent/goals.py). Runs before
        every archetype: the core proposes, the level ending disposes."""
        logical = self.perc.logical(obs.frame)
        objs = segment(logical, self.perc.bg, mask=None)
        cl = goal_clusters(objs)
        if self._goal_hyps is None:
            hyps = goal_scan(objs, logical.shape[0], logical.shape[1],
                                playfield=self.perc.playfield,
                                terrain=self._terrain_colors())
            # SAME GAME, NEW LEVEL: a reading the level ending confirmed
            # (the RULE and which side moves, not the colour) goes first
            conf = {(k, m) for k, m, _, _, _ in self._goal_confirmed_kinds}
            conf_rel = {(r, b) for _, _, _, r, b in self._goal_confirmed_kinds}
            hyps.sort(key=lambda h: (0 if (h.kind, h.m_color) in conf else 1,
                                     0 if (h.rule(), h.mover_bigger()) in conf_rel else 1,
                                     -h.prio))
            self._goal_hyps = hyps
            if hyps:
                self._hyp("goal_hypotheses", candidates=[repr(h) for h in hyps])
        if not self._goal_hyps:
            return None
        g = self._goal_act
        if g is not None:
            pos = g.observe(cl, objs)
            if g.arrived or (pos is not None and g.moved and g.dist(pos) < 1.0):
                # reached and the level stands: this pair alone is not
                # the goal — SATISFIED, the next candidate (a second tip,
                # a second cross) takes its turn
                g.h.satisfied = True
                self._hyp("goal_satisfied", gkind=g.h.kind, mover=g.h.m_color,
                          note=g.h.note, spent=g.h.spent, merged=bool(g.arrived))
                self._goal_act = None
            elif pos is None:
                g.h.refuted = True
                self._hyp("goal_refuted", gkind=g.h.kind, mover=g.h.m_color,
                          reason="mover_lost", spent=g.h.spent)
                self._goal_act = None
            elif False:
                # reached and the level stands: this pair alone is not
                # the goal — SATISFIED, the next candidate (a second tip,
                # a second cross) takes its turn
                g.h.satisfied = True
                self._hyp("goal_satisfied", gkind=g.h.kind, mover=g.h.m_color,
                          note=g.h.note, spent=g.h.spent)
                self._goal_act = None
            elif g.budget <= 0 or g.stalled():
                g.h.refuted = True
                self._hyp("goal_refuted", gkind=g.h.kind, mover=g.h.m_color,
                          reason="budget" if g.budget <= 0 else "stalled", spent=g.h.spent)
                self._goal_act = None
        if self._goal_act is None:
            if self._goal_used >= 5:
                return None
            nxt = next((h for h in self._goal_hyps
                        if not h.refuted and not h.satisfied and not h.confirmed), None)
            if nxt is None:
                return None
            names = [a.name for a in available(obs)]
            if getattr(self, "_goal_emap", None) is None:
                self._goal_emap = GoalEffectMap()
            # zero-cost refutation: the level's pass never moved this colour
            if (self._goal_emap.tested and len(self._goal_emap.tested) >= 6
                    and not self._goal_emap.ever_moved(nxt.m_color)):
                nxt.refuted = True
                self._hyp("goal_refuted", gkind=nxt.kind, mover=nxt.m_color,
                          reason="colour_never_moved", spent=0)
                return self._goals_step(obs)
            self._goal_act = GoalActuator(nxt, names, self._goal_emap, budget=40,
                                              frame=logical.shape,
                                              playfield=self.perc.playfield)
            self._goal_used += 1
            self._hyp("goal_adopted", gkind=nxt.kind, mover=nxt.m_color, note=nxt.note)
            g = self._goal_act
            g.observe(cl, objs)
            if g.m_pos is None:
                g.h.refuted = True
                self._goal_act = None
                return None
        # THE ORGAN NAMES THE GOAL, THE PLANNER REACHES IT: when a key
        # that moves the mover toward the marker is refused here (ka59:
        # the pit between the block and its socket), the greedy descent
        # is blind; the terrain map knows the way round
        if g.handoff or g.blocked:
            route = g.handoff
            if not route and g.blocked:
                route = self._goal_route(g) or []
                g.blocked = False
            if route:
                key = route.pop(0)
                g.handoff = route
                g.take(key)
                self.plan = []
                return GameAction[key]
        key = g.choose()
        if key is None:
            g.h.refuted = True
            self._hyp("goal_refuted", gkind=g.h.kind, mover=g.h.m_color,
                      reason="no_effect_found", spent=g.h.spent)
            self._goal_act = None
            return None
        self.plan = []
        if isinstance(key, tuple):
            a6 = GameAction.ACTION6
            a6.set_data(self.perc.click_xy(g.resolve(key)))
            return a6
        return GameAction[key]

    def _goal_route(self, g):
        """A planner route for the organ's mover, when the world model's
        body IS the mover (its anchor rides within the mover's extent)."""
        m = self.model
        if not m.known_motions or m.avatar_pos is None or g.m_pos is None:
            return None
        r0, c0 = self.perc.playfield[0], self.perc.playfield[2]
        cam = self.perc.cam
        mw = (g.m_pos[0] - r0 + cam[0], g.m_pos[1] - c0 + cam[1])   # mover centre, world
        av = m.avatar_pos
        if abs(av[0] - mw[0]) > 3 or abs(av[1] - mw[1]) > 3:
            return None
        tc = g.h.target_centre()
        tw = (tc[0] - r0 + cam[0], tc[1] - c0 + cam[1])
        target = (int(round(tw[0] + (av[0] - mw[0]))), int(round(tw[1] + (av[1] - mw[1]))))
        plan = m.plan_to(target)
        if not plan:
            self._hyp("goal_route", found=False, target=list(target))
            return None
        names = [self._to_action(p).name for p in plan]
        self._hyp("goal_route", found=True, target=list(target), length=len(names))
        return names

    # ---------- the playfield is a hypothesis ----------

    def _refute_playfield(self, obs):
        """The band-split that reads a full-length wall line as a HUD
        divider is a PRIOR, not a fact: when the body vanishes from the
        playfield and stands, same key, in the excluded region, the
        world is bigger than the crop (sx02: the goal's half of the
        board was cut off and the wrap-around landed the avatar in it).
        The crop is widened to the edge-stripped frame; world coordinates
        survive when the origin is unchanged, otherwise the terrain map
        is rebuilt (motions, laws, colors stay)."""
        logical = self.perc.logical(obs.frame)
        key = self.model.avatar_shape_key
        full = segment(logical, self.perc.bg, mask=None)
        r0, r1, c0, c1 = self.perc.playfield
        hit = None
        for o in full:
            if o.shape_key == key:
                r, c = o.bbox[0], o.bbox[1]
                if not (r0 <= r < r1 and c0 <= c < c1):
                    hit = o
                    break
        if hit is None:
            return False
        h, w = logical.shape
        self._hyp("playfield_refuted", old=[int(r0), int(r1), int(c0), int(c1)],
                  body_at=[int(hit.bbox[0]), int(hit.bbox[1])])
        new = (0, h, 0, w)
        if (new[0], new[2]) != (r0, c0):
            self.model.forget_terrain()
            self.model.colors[:] = -1
        self.perc.playfield = new
        self._sync_view(obs)
        self.plan = []
        # new information: every refutation made under the old crop is void
        self._goal_refuted = set()
        self._goal_deferred = set()
        self._touch_gaveup = set()
        self._touch_tries = {}
        return True

    # ---------- the anomaly ledger ----------

    def _detect_anomalies(self, name, d, pav, prev_objs, cur_objs):
        m = self.model
        if (name in ("ACTION6", "RESET") or m is None or pav is None
                or m.avatar_pos is None or name in m.shape_op_actions):
            self._anom_expect = None
            return
        vec = (m.avatar_pos[0] - pav[0], m.avatar_pos[1] - pav[1])
        exp = self._anom_expect
        self._anom_expect = None
        known = m.motion[name].best()
        confirmed = known is not None and m.motion[name].votes.get(known, 0) >= 2
        # --- unexplained move
        # (a magnitude-only difference along the known direction is slide
        # or current territory, owned by the motion belief — not an
        # anomaly; st82/st71 lost levels to re-tests of ordinary slides)
        if (vec != (0, 0) and confirmed and vec != known
                and unit(vec) != unit(known)
                and not m.is_slide(name) and m.drift_estimate() == (0, 0)):
            if exp is not None and exp["kind"] == "move" and exp["action"] == name:
                if vec == exp["vec"]:
                    exp["reproduced"] += 1
                    self._hyp("unknown_mechanic_confirmed", akind="move",
                              action=name, vec=list(vec), at=list(pav))
                    exp["closed"] = True
                else:
                    exp["closed"] = True
                    self._hyp("anomaly_dismissed", akind="move", action=name)
            elif any(a["kind"] == "move" and a["action"] == name
                     and a["vec"] == vec and not a["closed"] for a in self._anomalies):
                # a SPONTANEOUS repeat is a confirmation too
                for a in self._anomalies:
                    if a["kind"] == "move" and a["action"] == name and a["vec"] == vec:
                        a["reproduced"] += 1
                        if not a["closed"]:
                            a["closed"] = True
                            self._hyp("unknown_mechanic_confirmed", akind="move",
                                      action=name, vec=list(vec), at=list(pav),
                                      witnesses=a["reproduced"] + 1)
            elif not any(a["kind"] == "move" and a["action"] == name
                         and a["vec"] == vec for a in self._anomalies):
                self._anomalies.append({"kind": "move", "action": name,
                                        "pos_before": tuple(pav), "vec": vec,
                                        "expected": known, "retests": 0,
                                        "reproduced": 0, "closed": False,
                                        "n": self._obs_i})
                self._hyp("anomaly", akind="move", action=name, vec=list(vec),
                          expected=list(known), at=list(pav))
        elif exp is not None and exp["kind"] == "move":
            exp["closed"] = True
            self._hyp("anomaly_dismissed", akind="move", action=exp["action"])
        # --- co-motion: something else moved with our vector from in front
        if vec != (0, 0) and d.moved:
            av_col = m.avatar_shape_key[0] if m.avatar_shape_key else None
            r0f, c0f = self.perc.playfield[0], self.perc.playfield[2]
            cr, cc = self.perc.cam
            u = unit(vec)
            mag = max(abs(vec[0]), abs(vec[1]))
            for color, size, mv in d.moved:
                if color == av_col or mv != vec or color in self._ambient:
                    continue
                if self.tracker.explains(color, size, mv):
                    continue
                # where was it? adjacent-in-front of the avatar's old cell
                front = None
                for o in prev_objs:
                    if o.color != color or o.size != size:
                        continue
                    ow = (round(o.centroid[0] - r0f + cr),
                          round(o.centroid[1] - c0f + cc))
                    dv = (ow[0] - pav[0], ow[1] - pav[1])
                    if unit(dv) == u and max(abs(dv[0]), abs(dv[1])) <= 2 * mag + 2:
                        front = ow
                        break
                if front is None:
                    continue
                now = (front[0] + vec[0], front[1] + vec[1])
                if (exp is not None and exp["kind"] == "comotion"
                        and exp["color"] == color):
                    exp["reproduced"] += 1
                    exp["closed"] = True
                    m.pushable.add(color)
                    self._hyp("relation_coined", relation="pushable", color=color,
                              witnesses=exp["reproduced"] + 1)
                    continue
                if not any(a["kind"] == "comotion" and a["color"] == color
                           for a in self._anomalies):
                    self._anomalies.append({"kind": "comotion", "color": color,
                                            "size": size, "entity": now,
                                            "vec": vec, "action": name,
                                            "retests": 0, "reproduced": 0,
                                            "closed": False, "n": self._obs_i})
                    self._hyp("anomaly", akind="comotion", color=color,
                              vec=list(vec), action=name, at=list(front))
                else:
                    for a in self._anomalies:
                        if a["kind"] == "comotion" and a["color"] == color:
                            a["entity"] = now
                            a["reproduced"] += 1
                            if not a["closed"]:
                                a["closed"] = True
                                m.pushable.add(color)
                                self._hyp("relation_coined", relation="pushable",
                                          color=color, witnesses=a["reproduced"] + 1)
        if exp is not None and exp["kind"] == "comotion" and not exp["closed"]:
            if vec == (0, 0):
                # neither moved: the object may be blocked — inconclusive,
                # the hypothesis stays live (its re-test budget was spent)
                pass
            else:
                exp["closed"] = True
                self._hyp("anomaly_dismissed", akind="comotion", color=exp["color"])

    def _retest_anomaly(self, obs):
        """ACT TO DISAMBIGUATE, NOT TO EXPLOIT: the cheapest action that
        re-produces a live anomaly — return to where it happened and
        repeat the action (move), or step into the object again from
        behind (co-motion). Bounded: two re-tests per anomaly, twelve
        actions per level."""
        m = self.model
        if self._anom_budget <= 0 or m is None or m.avatar_pos is None:
            return None
        for a in self._anomalies:
            if a["closed"] or a["retests"] >= 2:
                continue
            if a["kind"] == "move":
                target, action = a["pos_before"], a["action"]
            else:
                u = unit(a["vec"])
                target = (a["entity"][0] - u[0], a["entity"][1] - u[1])
                action = next((n for n, v in m.known_motions.items()
                               if unit(v) == u), None)
                if action is None:
                    a["closed"] = True
                    continue
            if tuple(m.avatar_pos) == tuple(target):
                a["retests"] += 1
                self._anom_budget -= 1
                self._anom_expect = a
                return GameAction[action]
            plan = m.plan_to(target)
            if not plan or len(plan) > 12:
                a["closed"] = True
                continue
            self._anom_budget -= 1
            self.plan = plan[1:]
            return self._to_action(plan[0])
        # THE EDGE IS A HYPOTHESIS TOO: the planner never steps toward
        # unseen cells, so "the boundary is a wall" is never tested. Once
        # per direction per level, standing at the boundary, press the
        # action that points outward (sx02: the wrap-around was found at
        # step 189 by accident; one deliberate press finds it at once).
        if self._anom_budget > 0 and not m.shape_op_actions:
            for act, v in m.known_motions.items():
                if act in self._edge_probed or m.is_slide(act):
                    continue
                if m.at_edge(m.avatar_pos, v):
                    self._edge_probed.add(act)
                    self._anom_budget -= 1
                    return GameAction[act]
        return None

    # ---------- action selection ----------

    def _motion_group(self, seed, parts, wcells):
        """Rigid group of `seed`, kept honest by co-movement: an adjacent
        object joins only if its track has co-moved with the seed's track
        (>=2 times), or if neither has any movement history yet."""
        colored, group = group_cells(seed, parts, wcells)
        # exact object->track mapping recorded by observe() this frame
        r0f, c0f = self.perc.playfield[0], self.perc.playfield[2]
        cr, cc = self.perc.cam

        def okey(o):
            return (o.shape_key,
                    (round(o.centroid[0] - r0f + cr),
                     round(o.centroid[1] - c0f + cc)))
        obj_tid = {id(o): self._tid_by_key.get(okey(o)) for o in group}
        seed_tid = obj_tid.get(id(seed))
        if seed_tid is None:
            return colored, group
        seed_tr = self.tracker.tracks.get(seed_tid)
        seed_moves = seed_tr.steps_moved if seed_tr else 0
        acc = self._piece_memory.setdefault(seed_tid, set())
        keep = []
        for o in group:
            tid = obj_tid.get(id(o))
            if o is seed or tid is None:
                keep.append(o)
                continue
            if o.shape_key in acc:
                keep.append(o)      # HYSTERESIS: accepted parts stay
                continue
            key = frozenset((seed_tid, tid))
            comoves = self._comove.get(key, 0)
            tr = self.tracker.tracks.get(tid)
            part_moves = tr.steps_moved if tr else 0
            # NEGATIVE evidence only: exclude a part that has its own
            # movement history yet has never once moved with the seed —
            # that is a touching NEIGHBOR, not an attached part
            if comoves == 0 and part_moves >= 2 and seed_moves >= 2:
                continue
            acc.add(o.shape_key)
            keep.append(o)
        if len(keep) < len(group):
            colored = frozenset().union(*[wcells(o) for o in keep])
        return colored, keep

    def _switch_piece(self, objs, wcells, group_ids):
        """Multi-piece assembly: our piece's ports are all connected (or it
        has none) — click another piece that still carries port cells and
        rebind identity to it. Clicking selects the sprite under the click."""
        if self._switch_cool > 0:
            return None
        # ONLY AFTER A VERIFIED SUCCESS: rebinding identity to a small
        # port-colored fragment is destructive (seed lookup dies with it).
        # Transient no-socket / no-port states during discovery (e.g. the
        # frame after a rotate press) must never trigger a switch —
        # observed cn04 L2: a mid-dock switch scattered identity across
        # remote port fragments and the dock never re-engaged.
        if not getattr(self, "_switch_ok", False):
            return None
        for o in objs:
            if id(o) in group_ids:
                continue
            if o.color in self._port_colors_seen and o.size <= 30:
                w = wcells(o)
                r, c, _ = next(iter(w))
                self.tracker._avatar_id = None
                self.model.avatar_shape_key = o.shape_key
                cen = (sum(rr for rr, _, _ in w) / len(w),
                       sum(cc for _, cc, _ in w) / len(w))
                self.model.avatar_pos = (int(round(cen[0])), int(round(cen[1])))
                self._dock = None
                self._switch_cool = 4
                self._switch_ok = False
                a6 = GameAction.ACTION6
                a6.set_data(self._world_to_click((int(r), int(c))))
                return a6
        return None

    # ---------- constraint-glyph lattice archetype ----------

    def _glyphs_step(self, obs):
        """CLICK-TOGGLE LATTICE WITH CONSTRAINT GLYPHS (ft09 family):
        uniform tiles on a regular lattice cycle color when clicked; a
        mixed-color glyph tile embedded in the lattice constrains its 8
        neighbors — a black glyph pixel demands the neighbor EQUAL the
        glyph's center color, any other marking demands it differ. Win =
        all glyph constraints satisfied. Boards ship with decorative
        LEGEND copies, so lattices are clustered and each cluster is
        verified-by-use independently: a click that fails to recolor
        kills only its own cluster (legend clicks are free no-ops)."""
        gl = self._gl
        if gl["dead"]:
            return None
        avail = getattr(obs, "available_actions", None)
        if avail != [6]:
            return None
        objs = self.perc.objects(obs.frame)
        from collections import Counter as _C7
        sq = [o for o in objs
              if 4 <= o.size <= 64
              and (o.bbox[2] - o.bbox[0]) == (o.bbox[3] - o.bbox[1])
              and o.size == (o.bbox[2] - o.bbox[0] + 1) ** 2]
        if len(sq) < 4:
            return None
        side_counts = _C7(o.bbox[2] - o.bbox[0] + 1 for o in sq)
        side = side_counts.most_common(1)[0][0]
        tiles = [o for o in sq if o.bbox[2] - o.bbox[0] + 1 == side]
        if len(tiles) < 4:
            return None
        # pitch: modal nearest-neighbor axis distance
        nn = []
        for a in tiles:
            best = None
            for b in tiles:
                if a is b:
                    continue
                d = abs(a.bbox[0] - b.bbox[0]) + abs(a.bbox[1] - b.bbox[1])
                if best is None or d < best:
                    best = d
            if best:
                nn.append(best)
        if not nn:
            return None
        pitch = _C7(nn).most_common(1)[0][0]
        if pitch <= side - 1:
            return None
        # cluster tiles by pitch-adjacency
        idx = {id(o): o for o in tiles}
        adj = {id(o): set() for o in tiles}
        for a in tiles:
            for b in tiles:
                if a is b:
                    continue
                dr = abs(a.bbox[0] - b.bbox[0])
                dc = abs(a.bbox[1] - b.bbox[1])
                if dr + dc <= 2 * pitch and dr % pitch <= 1 \
                        and dc % pitch <= 1:
                    adj[id(a)].add(id(b))
        seenc, clusters = set(), []
        from collections import deque as _dq4
        for o in tiles:
            if id(o) in seenc:
                continue
            q = _dq4([id(o)])
            comp = {id(o)}
            seenc.add(id(o))
            while q:
                cur = q.popleft()
                for nb in adj[cur]:
                    if nb not in seenc:
                        seenc.add(nb)
                        comp.add(nb)
                        q.append(nb)
            clusters.append([idx[i] for i in comp])
        lg = self.perc.logical(obs.frame)
        last = gl["last"]
        gl["last"] = None
        dead_cl = gl.setdefault("dead_cl", set())
        for cl in sorted(clusters, key=len, reverse=True):
            if len(cl) < 4:
                continue
            base = (min(o.bbox[0] for o in cl), min(o.bbox[1] for o in cl))
            ckey = base
            if ckey in dead_cl:
                continue
            tile_slots = {}
            for o in cl:
                gr = round((o.bbox[0] - base[0]) / pitch)
                gc = round((o.bbox[1] - base[1]) / pitch)
                tile_slots[(gr, gc)] = o
            if last is not None and last[0] == ckey:
                slot, prev_col = last[1], last[2]
                o = tile_slots.get(slot)
                if o is not None and o.color == prev_col:
                    dead_cl.add(ckey)   # click was a no-op: a legend
                    continue

            def slot_block(gr, gc):
                r0 = base[0] + gr * pitch
                c0 = base[1] + gc * pitch
                if (r0 < 0 or c0 < 0 or r0 + side > lg.shape[0]
                        or c0 + side > lg.shape[1]):
                    return None
                return lg[r0:r0 + side, c0:c0 + side]
            grs = [g for g, _ in tile_slots]
            gcs = [g for _, g in tile_slots]
            glyphs = []
            for gr in range(min(grs) - 1, max(grs) + 2):
                for gc in range(min(gcs) - 1, max(gcs) + 2):
                    if (gr, gc) in tile_slots:
                        continue
                    blk = slot_block(gr, gc)
                    if blk is None:
                        continue
                    vals = set(int(v) for v in blk.flatten())
                    if len(vals) < 2 or (self.perc.bg in vals
                                         and len(vals) < 3):
                        continue
                    if not any((gr + dr, gc + dc) in tile_slots
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                        continue
                    cen = int(blk[side // 2, side // 2])
                    cons = {}
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            if (gr + dr, gc + dc) not in tile_slots:
                                continue
                            pr = (0 if dr < 0 else
                                  side - 1 if dr > 0 else side // 2)
                            pc = (0 if dc < 0 else
                                  side - 1 if dc > 0 else side // 2)
                            pix = int(blk[pr, pc])
                            cons[(dr, dc)] = "eq" if pix == 0 else "ne"
                    if cons:
                        glyphs.append(((gr, gc), cen, cons))
            if not glyphs:
                continue
            for (gr, gc), cen, cons in glyphs:
                for (dr, dc), kind in cons.items():
                    slot = (gr + dr, gc + dc)
                    o = tile_slots.get(slot)
                    if o is None:
                        continue
                    bad = ((o.color != cen) if kind == "eq"
                           else (o.color == cen))
                    if not bad:
                        continue
                    n = gl["clicks"].get((ckey, slot), 0)
                    if n >= 5:
                        dead_cl.add(ckey)
                        break
                    gl["clicks"][(ckey, slot)] = n + 1
                    gl["last"] = (ckey, slot, o.color)
                    a6 = GameAction.ACTION6
                    a6.set_data(self.perc.click_xy(
                        (int(round(o.centroid[0])),
                         int(round(o.centroid[1])))))
                    self.plan = []
                    return a6
                else:
                    continue
                break
        return None

    # ---------- remote-actuator archetype ----------

    def _actuator_step(self, obs):
        """REMOTE DISCRETE ACTUATORS (s5i5 family): a click-only game
        where the salient objects are inert and small WIDGETS elsewhere
        apply fixed deltas to them (grow/shrink/rotate an arm carrying a
        marker toward a target ring). No archetype prior applies — the
        agent must LEARN operators by experiment and hill-climb: click a
        spot, measure how the marker-to-ring distance moved, prefer
        improving spots, stop probing when the budget-aware experiment
        table is full. Elongated widgets get a probe spot at each end
        (the two halves often mean opposite deltas)."""
        ac = self._ac
        if ac["dead"]:
            return None
        avail = getattr(obs, "available_actions", None)
        if avail != [6]:
            return None
        objs = self.perc.objects(obs.frame)
        from collections import Counter as _C6
        colcnt = _C6(o.color for o in objs)
        # rider/target color: a color rendered ONLY as tiny dots, forming
        # both singletons (riders on arm tips) and multi-dot outlines
        # (targets) — cluster its cells 8-connected to split the roles
        mcol, marks, targets = None, [], []
        for col in colcnt:
            group = [o for o in objs if o.color == col]
            if len(group) < 3 or any(o.size > 2 for o in group):
                continue
            cells = {cell for o in group for cell in o.cells}
            from collections import deque as _dq3
            seen3, clus = set(), []
            for c0 in cells:
                if c0 in seen3:
                    continue
                q = _dq3([c0])
                comp = {c0}
                seen3.add(c0)
                while q:
                    r_, c_ = q.popleft()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            n3 = (r_ + dr, c_ + dc)
                            if n3 in cells and n3 not in seen3:
                                seen3.add(n3)
                                comp.add(n3)
                                q.append(n3)
                clus.append(comp)
            singles = [c for c in clus if len(c) <= 2]
            multis = [c for c in clus if len(c) >= 3]
            if singles and multis:
                mcol = col
                marks = [(sum(a for a, _ in c) / len(c),
                          sum(b for _, b in c) / len(c)) for c in singles]
                targets = [(sum(a for a, _ in c) / len(c),
                            sum(b for _, b in c) / len(c)) for c in multis]
                break
        if mcol is None or not marks or not targets:
            return None
        widgets = [o for o in objs
                   if o.color != mcol and 3 <= o.size <= 25]

        def score():
            tot = 0.0
            for mp in marks:
                tot += min(abs(mp[0] - tp[0]) + abs(mp[1] - tp[1])
                           for tp in targets)
            return tot
        cur = score()
        last = ac["last"]
        ac["last"] = None
        if last is not None:
            spot, prev = last
            ac["tried"].setdefault(spot, []).append(cur - prev)
        if cur == 0:
            return None       # already aligned; the engine should score it
        # probe spots from widgets: ends of elongated ones, centroids of
        # compact ones (exclude rings/markers — inert by role)
        spots = []
        for o in widgets:
            hh = o.bbox[2] - o.bbox[0] + 1
            ww = o.bbox[3] - o.bbox[1] + 1
            if max(hh, ww) >= 2.5 * min(hh, ww):
                if hh >= ww:
                    spots.append((o.bbox[0] + hh // 5, (o.bbox[1] + o.bbox[3]) // 2))
                    spots.append((o.bbox[2] - hh // 5, (o.bbox[1] + o.bbox[3]) // 2))
                else:
                    spots.append(((o.bbox[0] + o.bbox[2]) // 2, o.bbox[1] + ww // 5))
                    spots.append(((o.bbox[0] + o.bbox[2]) // 2, o.bbox[3] - ww // 5))
            else:
                spots.append((round(o.centroid[0]), round(o.centroid[1])))
        spots = [(int(r), int(c)) for r, c in spots]
        if not spots:
            ac["dead"] = True
            return None
        # choose: best measured improver, else an untried spot
        best_spot, best_avg = None, -1e-6
        for sp in spots:
            hist = ac["tried"].get(sp)
            if hist:
                # RECENT evidence beats history: a spot that stopped
                # improving is saturated (arm at max), and its glorious
                # past must not earn it further budget
                recent = hist[-3:]
                avg = sum(recent) / len(recent)
                if avg < best_avg:
                    best_avg, best_spot = avg, sp
        if best_spot is None:
            untried = [sp for sp in spots if sp not in ac["tried"]]
            if untried and ac["probes"] < 2 * len(spots):
                ac["probes"] += 1
                best_spot = untried[0]
            else:
                ac["dead"] = True
                return None
        ac["last"] = (best_spot, cur)
        a6 = GameAction.ACTION6
        a6.set_data(self.perc.click_xy(best_spot))
        self.plan = []
        return a6

    # ---------- reference-assembly archetype ----------

    def _refassembly_step(self, obs):
        """REFERENCE-SEQUENCE ASSEMBLY WITH SUBMIT (sb26 family): a
        reference strip shows a color sequence, movable tokens sit in a
        tray, empty slots await them, and a separate RUN action validates
        the assembly. The protocol (select token -> place in slot -> ...
        -> submit) cannot be stumbled into by undirected clicking, and
        every operation drains a budget. Detection scans the FULL frame:
        trays live beyond separator lines that playfield detection reads
        as borders, so perc.objects never sees them."""
        rf = self._rf
        if rf["dead"]:
            return None
        avail = getattr(obs, "available_actions", None)
        if not avail or 6 not in avail or any(a in avail
                                              for a in (1, 2, 3, 4)):
            return None
        lg = self.perc.logical(obs.frame)
        bg = self.perc.bg
        h, w = lg.shape
        from collections import deque as _dq2
        blobs = []
        seen = set()
        for r in range(h):
            for c in range(w):
                col = int(lg[r, c])
                if col == bg or (r, c) in seen:
                    continue
                q = _dq2([(r, c)])
                comp = [(r, c)]
                seen.add((r, c))
                while q:
                    rr, cc2 = q.popleft()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            n = (rr + dr, cc2 + dc)
                            if (0 <= n[0] < h and 0 <= n[1] < w
                                    and n not in seen
                                    and int(lg[n[0], n[1]]) == col):
                                seen.add(n)
                                comp.append(n)
                                q.append(n)
                if 2 <= len(comp) <= 30:
                    cen = (sum(a for a, _ in comp) / len(comp),
                           sum(b for _, b in comp) / len(comp))
                    blobs.append((col, len(comp), cen))
        if len(blobs) < 6:
            return None
        rows = {}
        for col, sz, cen in blobs:
            rows.setdefault(int(round(cen[0] / 3)), []).append(
                (col, sz, cen))
        cand = sorted(((k, v) for k, v in rows.items() if len(v) >= 3),
                      key=lambda kv: kv[0])
        if len(cand) < 3:
            return None
        ref_row, tray_row = cand[0][1], cand[-1][1]
        # reference cells may be ring+interior stacks: cluster by column,
        # drop any color present in EVERY cluster (frame/fill), keep one
        # color per cluster
        clusters = {}
        for col, sz, cen in ref_row:
            clusters.setdefault(int(round(cen[1] / 4)), []).append(
                (col, sz, cen))
        allcols = [set(c for c, _, _ in v) for v in clusters.values()]
        common = set.intersection(*allcols) if len(allcols) > 1 else set()
        ref_cells = []
        for k in sorted(clusters):
            keep = [t for t in clusters[k] if t[0] not in common]
            if len(keep) != 1:
                return None
            ref_cells.append(keep[0])
        ref_cols = [t[0] for t in ref_cells]
        if len(set(ref_cols)) < 2 or len(ref_cols) < 2:
            return None
        tray = sorted(tray_row, key=lambda t: t[2][1])
        # slots: a middle row with count == len(ref)
        slot_row = None
        for k, v in cand[1:-1]:
            if len(v) == len(ref_cols):
                slot_row = sorted(v, key=lambda t: t[2][1])
                break
        if slot_row is None:
            return None
        i = 0
        while i < len(ref_cols) and slot_row[i][0] == ref_cols[i]:
            i += 1
        if i >= len(ref_cols):
            if rf["submits"] >= 3:
                rf["dead"] = True
                return None
            run = [a for a in avail if a not in (6, 7)]
            if not run:
                return None
            rf["submits"] += 1
            self.plan = []
            return GameAction[f"ACTION{min(run)}"]
        pend = rf["pend"]
        rf["pend"] = None
        if pend is not None and pend[0] == i:
            cen = slot_row[i][2]
            a6 = GameAction.ACTION6
            a6.set_data(self.perc.click_xy(
                (int(round(cen[0])), int(round(cen[1])))))
            self.plan = []
            return a6
        need = ref_cols[i]
        tok = next((t for t in tray if t[0] == need), None)
        if tok is None:
            rf["dead"] = True
            return None
        rf["pend"] = (i,)
        a6 = GameAction.ACTION6
        a6.set_data(self.perc.click_xy(
            (int(round(tok[2][0])), int(round(tok[2][1])))))
        self.plan = []
        return a6

    def _click_frame_obj(self, o):
        """Click data for an object's centroid (frame coords)."""
        return self._world_to_click(
            (int(round(o.centroid[0] - self.perc.playfield[0]
                       + self.perc.cam[0])),
             int(round(o.centroid[1] - self.perc.playfield[2]
                       + self.perc.cam[1]))))

    # ---------- template-overlay archetype ----------

    def _overlay_step(self, obs):
        """TEMPLATE OVERLAY WITH ENTITY CYCLING (re86 family): several
        movable outline shapes; the goal overlay leaves a few 1px dots of
        each shape's color; win = every dot exactly covered by its
        shape. Distance-to-goal heuristics score nothing here — only the
        exact superposition does — and the selector action's visible
        effect is a single pixel, so the generic layer never discovers
        multi-body switching. Engagement requires a UNIQUE exact template
        match (verified-by-use in spirit: no unique match, no takeover)."""
        ov = self._ov
        if ov["dead"] or not self.model.known_motions:
            return None
        # dominant step magnitude; a selector action mis-learned as a
        # long "move" (its 1px effect pairs across shapes) must not
        # poison the lattice estimate
        from collections import Counter as _CM
        magc = _CM(max(abs(v[0]), abs(v[1]))
                   for v in self.model.known_motions.values())
        stp = magc.most_common(1)[0][0]
        moves = {a: v for a, v in self.model.known_motions.items()
                 if max(abs(v[0]), abs(v[1])) == stp}
        if stp < 2 or len(moves) < 2:
            return None
        objs = self.perc.objects(obs.frame)
        r0f, c0f = self.perc.playfield[0], self.perc.playfield[2]
        cr, cc = self.perc.cam

        def wpos(o):
            return frozenset((r - r0f + cr, c - c0f + cc)
                             for r, c in o.cells)
        # movable templates: big objects whose color also appears as a
        # few SMALL dot objects elsewhere (the constellation)
        from collections import defaultdict
        by_color = defaultdict(list)
        for o in objs:
            by_color[o.color].append(o)
        pairs = {}          # color -> (template cells, dot cells)
        from collections import deque as _dq
        for col, os_ in by_color.items():
            if not any(o.size >= 8 for o in os_) or len(os_) < 2:
                continue
            # 8-CONNECTED re-clustering: the perceiver segments
            # 4-connected, which shatters diagonal outlines; a template
            # missing five pixels rejects its own true offset. Cluster
            # the color's cells ourselves.
            allc = set()
            for o in os_:
                allc |= wpos(o)
            if len(allc) > 400:
                continue
            clusters = []
            seen8 = set()
            for c0 in allc:
                if c0 in seen8:
                    continue
                q = _dq([c0])
                comp = {c0}
                seen8.add(c0)
                while q:
                    r_, c_ = q.popleft()
                    for dr8 in (-1, 0, 1):
                        for dc8 in (-1, 0, 1):
                            n8 = (r_ + dr8, c_ + dc8)
                            if n8 in allc and n8 not in seen8:
                                seen8.add(n8)
                                comp.add(n8)
                                q.append(n8)
                clusters.append(comp)
            clusters.sort(key=len, reverse=True)
            tmpl = clusters[0]
            if len(tmpl) >= 8:
                # movement tracking must survive dot-parsing failures
                # (clipped states): record the big cluster regardless
                ov.setdefault("tp2", {})[col] = frozenset(tmpl)
            rest_cl = clusters[1:]
            dots = [c for c in rest_cl if len(c) <= 4]
            if rest_cl and dots == rest_cl and 1 <= len(dots) <= 8:
                raw = frozenset().union(*dots)
                # STABILITY CERTIFICATION: a fragment of the shattered
                # outline travels with the template; a goal dot stays at
                # its world cell across DIFFERENT template positions.
                # Only twice-seen-elsewhere cells count as dots — the
                # wandering shape absorbing a nearby dot must not turn
                # the constellation ambiguous.
                dh = ov.setdefault("dh", {}).setdefault(col, {})
                tanchor = min(tmpl)
                for cell in raw:
                    dh.setdefault(cell, set()).add(tanchor)
                cert = frozenset(cell for cell, hs in dh.items()
                                 if len(hs) >= 2)
                if cert:
                    pairs[col] = (frozenset(tmpl), cert)
        if not pairs:
            return None
        # a color is DONE when its dots are covered (they vanish under
        # the template, leaving no small objects)
        undone = {}
        for col, (tmpl, dset) in pairs.items():
            rem = dset - tmpl
            if rem:
                undone[col] = (tmpl, rem, dset)
        if not undone:
            # colors with visible dots are all covered; if the level has
            # not advanced the engine wants other shapes too — nothing
            # more we can see to do
            return None
        # controlled shape: the one that moved after our last emitted
        # action — tracked over ALL big clusters (tp2), so a color whose
        # dot parsing failed this frame still testifies about motion
        # (losing it faked "nothing moved" and fired spurious cycles)
        prev_pos = ov.get("tpos") or {}
        cur_pos = dict(ov.get("tp2", {}))
        moved_col = None
        for col, cells in cur_pos.items():
            pp = prev_pos.get(col)
            if pp is not None and pp != cells:
                moved_col = col
        ov["tpos"] = cur_pos
        ov["tp2"] = {}
        last = ov.get("last")
        ov["last"] = None
        # OWN ACTION MAP: reconcile each emitted move with the observed
        # template delta (generic votes here get polluted by selector
        # presses; two actions claiming one direction stalls nav)
        if (last is not None and last[0] == "move" and len(last) >= 2
                and moved_col is not None and moved_col in cur_pos):
            aname_l = last[1]
            pp = prev_pos.get(moved_col)
            cur_t = cur_pos[moved_col]
            if pp is not None:
                da = (min(cur_t)[0] - min(pp)[0],
                      min(cur_t)[1] - min(pp)[1])
                # learn only from CLEAN samples: a template touching the
                # playfield border reveals/hides rows as it moves, so its
                # min-cell delta lies (observed: |da|=1 junk that poisoned
                # the map). Require a pure axis step of the lattice size
                # from a border-free template.
                fh_ = self.perc.playfield[1] - self.perc.playfield[0]
                fw_ = self.perc.playfield[3] - self.perc.playfield[2]

                def _clear(cells):
                    return all(0 < r_ - cr + r0f < fh_ - 1
                               and 0 < c_ - cc + c0f < fw_ - 1
                               for r_, c_ in cells)
                axis_ok = (sorted(map(abs, da)) ==
                           [0, max(abs(da[0]), abs(da[1]))])
                if (da != (0, 0) and axis_ok
                        and _clear(pp) and _clear(cur_t)):
                    am = ov.setdefault("amv", {}).setdefault(aname_l, {})
                    am[da] = am.get(da, 0) + 1
        if moved_col is not None:
            ov["ctl"] = moved_col
        elif last is not None and last[0] == "move":
            # our move produced no movement of any undone template: the
            # controlled shape is not among them (or blocked) — cycle
            ov["ctl"] = None
        if ov["ctl"] not in undone:
            ov["ctl"] = None
        if ov["ctl"] is None and self.model.avatar_shape_key is not None \
                and self.model.avatar_shape_key[0] in undone:
            # the generic layer's avatar IS the initially-selected shape
            ov["ctl"] = self.model.avatar_shape_key[0]
        if ov["ctl"] is None:
            # either probe (to see who moves) or cycle to change control
            if last is not None and last[0] == "probe":
                # probe moved nothing we track: press the cycler
                cyc = [a for a in GameAction
                       if a.name.startswith("ACTION")
                       and a.name not in moves
                       and a.name not in ("ACTION6", "ACTION7")
                       and int(a.name[-1]) <= 5]
                avail = getattr(obs, "available_actions", None)
                if avail:
                    cyc = [a for a in cyc if int(a.name[-1]) in avail]
                if cyc and ov["cycles"] < 12:
                    ov["cycles"] += 1
                    ov["last"] = ("cycle",)
                    self.plan = []
                    return cyc[0]
                ov["fails"] += 1
                if ov["fails"] >= 8:
                    ov["dead"] = True
                return None
            mlist = sorted(moves)
            aname = mlist[ov.get("pi", 0) % len(mlist)]
            ov["pi"] = ov.get("pi", 0) + 1
            ov["last"] = ("probe",)
            self.plan = []
            return GameAction[aname]
        # navigate the controlled template onto its dots: unique offset t
        # with dots ⊆ template+t, reachable on the step lattice
        tmpl, rem, dset = undone[ov["ctl"]]
        # match against the FULL certified constellation: dots currently
        # covered by the template still constrain the offset — matching
        # only the uncovered remainder goes ambiguous exactly when the
        # shape is nearly there (observed re86: 2 leftover collinear
        # dots -> 3 candidate offsets, permanent stall)
        # clipping tolerance: a shape parked partially off-grid has an
        # incomplete visible template; a dot whose preimage falls outside
        # the playfield cannot testify either way
        fh = self.perc.playfield[1] - self.perc.playfield[0]
        fw = self.perc.playfield[3] - self.perc.playfield[2]

        def _vis(cell):
            fr, fc = cell[0] - cr + r0f, cell[1] - cc + c0f
            return 0 <= fr < fh and 0 <= fc < fw

        cand_exc = {}
        for d0 in dset:
            for m in tmpl:
                t = (d0[0] - m[0], d0[1] - m[1])
                if t[0] % stp or t[1] % stp or t in cand_exc:
                    continue
                hits = misses = excused = 0
                for d in dset:
                    pre = (d[0] - t[0], d[1] - t[1])
                    if pre in tmpl:
                        hits += 1
                    elif _vis(pre):
                        misses += 1
                    else:
                        excused += 1
                if misses == 0 and hits >= 1:
                    cand_exc[t] = excused
        # the true offset maximizes REAL coverage: clipping excuses are
        # a last resort, and a uniquely least-excused candidate wins
        # (a wrong excused candidate once drove the shape off-grid)
        cand_ts = set()
        if cand_exc:
            emin = min(cand_exc.values())
            best_c = [t for t, e in cand_exc.items() if e == emin]
            if len(best_c) == 1:
                cand_ts = {best_c[0]}
        if len(cand_ts) != 1:
            self._dbg_ov = getattr(self, "_dbg_ov", [])
            if len(self._dbg_ov) < 6:
                self._dbg_ov.append((ov["ctl"], len(cand_ts),
                                     sorted(cand_ts)[:4], len(rem),
                                     sorted(rem)[:6], len(tmpl)))
            ov["fails"] += 1
            if ov["fails"] >= 8:
                ov["dead"] = True
            return None
        t = next(iter(cand_ts))
        if t == (0, 0):
            return None
        eff = dict(moves)
        for aname, am in ov.get("amv", {}).items():
            dv, nv = max(am.items(), key=lambda kv: kv[1])
            if nv >= 2:
                eff[aname] = dv
        best_a, best_gain = None, 0
        for aname, vec in eff.items():
            after = (abs(t[0] - vec[0]) + abs(t[1] - vec[1]))
            gain = (abs(t[0]) + abs(t[1])) - after
            if gain > best_gain:
                best_gain, best_a = gain, aname
        if best_a is None:
            # calibration fallback: an action's true vector may be
            # unverified (a corrupted generic vote can make a direction
            # unreachable forever — pressing it is the only cure)
            unver = [a for a in moves
                     if sum(ov.get("amv", {}).get(a, {}).values()) < 2]
            if unver and ov.get("calib_n", 0) < 8:
                ov["calib_n"] = ov.get("calib_n", 0) + 1
                a_ = unver[ov["calib_n"] % len(unver)]
                ov["last"] = ("move", a_)
                self.plan = []
                return GameAction[a_]
            ov["fails"] += 1
            if ov["fails"] >= 8:
                ov["dead"] = True
            return None
        ov["last"] = ("move", best_a)
        self.plan = []
        return GameAction[best_a]

    # ---------- edge-maze archetype ----------

    def _edgemaze_step(self, obs):
        """Lattice-graph navigation: nodes every p px; the midpoint cell
        between two nodes carries an OPEN color iff the edge is passable
        (tu93 family). Wall-model-by-collision is structurally wrong here
        and every keypress spends a metered budget, so exploration must
        become graph search the moment the edge color is certified
        (verified-by-use: >=2 open votes for one color, >=1 closed vote
        for a different one, zero contradictions). Fully self-anchored:
        see the accounting block in observe()."""
        em = self._em
        if em["dead"]:
            return None
        avail = getattr(obs, "available_actions", None)
        if avail and any(a >= 5 for a in avail):
            return None       # click/other actions exist: not this family
        ac = em.get("acolor")
        dirs = em.get("dirs", {})
        # settled directions: dominant vector with >=2 votes per action
        amap = {}
        for aname, dd in dirs.items():
            dv, nv = max(dd.items(), key=lambda kv: kv[1])
            if nv >= 2:
                amap[aname] = dv
        if ac is None or not amap:
            return None
        mags = {abs(v[0]) + abs(v[1]) for v in amap.values()}
        if len(mags) != 1:
            return None
        pmag = mags.pop()
        if pmag < 4 or pmag % 2:
            return None
        # ELIMINATION: four actions, four lattice directions — three
        # settled mappings determine the fourth for free
        allv = {(pmag, 0), (-pmag, 0), (0, pmag), (0, -pmag)}
        acts4 = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
        if len(amap) == 3:
            rem_d = allv - set(amap.values())
            rem_a = [a for a in acts4 if a not in amap]
            if len(rem_d) == 1 and len(rem_a) == 1:
                amap[rem_a[0]] = next(iter(rem_d))
        # CALIBRATION: a dithering explorer may never press some action
        # twice from an open node; spend a bounded number of deliberate
        # presses to settle the map (each also feeds edge votes)
        if len(amap) < 4:
            cal = em.setdefault("calib", {})
            for a in acts4:
                if a not in amap and cal.get(a, 0) < 3:
                    cal[a] = cal.get(a, 0) + 1
                    self.plan = []
                    return GameAction[a]
        # certify the open-edge color
        if em["ec"] is None:
            op, cl = em["open"], em["closed"]
            cands = [c for c, n in op.items()
                     if n >= 2 and cl.get(c, 0) == 0]
            if len(cands) == 1 and any(c != cands[0] and n >= 1
                                       for c, n in cl.items()):
                em["ec"] = cands[0]
        if em["ec"] is None:
            return None
        ec = em["ec"]
        pos = (em.get("umap") or {}).get(ac)
        if pos is None:
            return None
        logical = self.perc.logical(obs.frame)
        r0f, c0f = self.perc.playfield[0], self.perc.playfield[2]
        cr, cc = self.perc.cam

        def col_at(wr, wc):
            fr, fc = wr - cr + r0f, wc - cc + c0f
            if 0 <= fr < logical.shape[0] and 0 <= fc < logical.shape[1]:
                return int(logical[fr, fc])
            return None

        # progress check on an active path
        if em["path"]:
            if pos == em["path"][0]:
                em["path"].pop(0)
                em["fails"] = 0
            elif em.get("expect") == em["path"][0] and pos != em["path"][0]:
                em["blocked"].add(frozenset((pos, em["path"][0])))
                em["path"] = []
                em["fails"] += 1
                if em["fails"] >= 8:
                    em["dead"] = True
                    return None
        em["expect"] = None
        if not em["path"]:
            # target: a unique-color block of body scale on the lattice
            objs = self.perc.objects(obs.frame)
            from collections import Counter as _C
            colcnt = _C(o.color for o in objs)
            av_sz = next((o.size for o in objs if o.color == ac), 9)
            best_t, best_d = None, 1e9
            for o in objs:
                if o.color == ac or colcnt[o.color] != 1:
                    continue
                if not (0.4 * av_sz <= o.size <= 3 * av_sz):
                    continue
                ow = (int(round(o.centroid[0] - r0f + cr)),
                      int(round(o.centroid[1] - c0f + cc)))
                if ((ow[0] - pos[0]) % pmag) or ((ow[1] - pos[1]) % pmag):
                    continue
                d = abs(ow[0] - pos[0]) + abs(ow[1] - pos[1])
                if 0 < d < best_d:
                    best_d, best_t = d, ow
            if best_t is None:
                return None
            from collections import deque
            seen = {pos}
            par = {}
            q = deque([pos])
            found = False
            while q:
                cur = q.popleft()
                if cur == best_t:
                    found = True
                    break
                for dv in ((pmag, 0), (-pmag, 0), (0, pmag), (0, -pmag)):
                    nxt = (cur[0] + dv[0], cur[1] + dv[1])
                    if nxt in seen:
                        continue
                    if frozenset((cur, nxt)) in em["blocked"]:
                        continue
                    if col_at(cur[0] + dv[0] // 2,
                              cur[1] + dv[1] // 2) != ec:
                        continue
                    if col_at(*nxt) is None:
                        continue
                    seen.add(nxt)
                    par[nxt] = cur
                    q.append(nxt)
            if not found:
                em["fails"] += 1
                if em["fails"] >= 8:
                    em["dead"] = True
                return None
            path = [best_t]
            while path[-1] != pos:
                path.append(par[path[-1]])
            path.reverse()
            em["path"] = path[1:]
        if not em["path"]:
            return None
        nxt = em["path"][0]
        dv = (nxt[0] - pos[0], nxt[1] - pos[1])
        for aname, v in amap.items():
            if v == dv:
                em["expect"] = nxt
                self.plan = []
                return GameAction[aname]
        em["path"] = []
        return None

    # ---------- peg-solitaire archetype ----------

    def _peg_scan(self, obs):
        """Detect a tile lattice + pegs: the largest class of identical
        square tiles on a regular pitch; pegs = smaller objects whose
        centroids land on tile cells. Returns (board{cell:centroid},
        pegs{cell}) or None."""
        objs = self.perc.objects(obs.frame)
        cls = Counter((o.color, o.size) for o in objs)
        if not cls:
            return None
        tile_cls, n = max(cls.items(), key=lambda kv: kv[1])
        if n < 9:
            return None
        all_tiles = [o for o in objs if (o.color, o.size) == tile_cls]
        tsize = all_tiles[0].size

        def pitch(vals):
            vals = sorted(vals)
            ds = [b - a for a, b in zip(vals, vals[1:]) if b - a > 1.5]
            return sorted(ds)[len(ds) // 2] if ds else None

        pr0 = pitch([o.centroid[0] for o in all_tiles])
        pc0 = pitch([o.centroid[1] for o in all_tiles])
        if not pr0 or not pc0:
            return None
        # CARDINALITY LESSON: a level may hold SEVERAL boards (lf52 L2:
        # two lattices joined by a pipe). A board is a CONNECTED lattice
        # component — fitting one global grid fused them and invented
        # phantom cells. Cluster tiles by adjacency, fit per component,
        # play the component that holds the most pegs.
        thr_link = 1.7 * max(pr0, pc0)
        comp = list(range(len(all_tiles)))

        def find(i):
            while comp[i] != i:
                comp[i] = comp[comp[i]]
                i = comp[i]
            return i

        for i in range(len(all_tiles)):
            for j in range(i + 1, len(all_tiles)):
                a, b = all_tiles[i].centroid, all_tiles[j].centroid
                if (abs(a[0] - b[0]) + abs(a[1] - b[1])) <= thr_link:
                    comp[find(i)] = find(j)
        groups = {}
        for i, o in enumerate(all_tiles):
            groups.setdefault(find(i), []).append(o)
        tiles = max(groups.values(), key=len)
        if len(tiles) < 6:
            return None
        pr = pitch([o.centroid[0] for o in tiles]) or pr0
        pc = pitch([o.centroid[1] for o in tiles]) or pc0
        r0 = min(o.centroid[0] for o in tiles)
        c0 = min(o.centroid[1] for o in tiles)

        def cell_of(cy, cx):
            R = round((cy - r0) / pr)
            C = round((cx - c0) / pc)
            if (abs(r0 + R * pr - cy) > 0.3 * pr
                    or abs(c0 + C * pc - cx) > 0.3 * pc):
                return None
            return (R, C)

        board = {}
        for o in tiles:
            cell = cell_of(*o.centroid)
            if cell is None:
                return None
            board[cell] = o.centroid
        # an OCCUPIED tile splits into fragments and drops out of the
        # tile class — the board must include the cells the pegs sit on,
        # synthesized from lattice arithmetic
        rmax = max(o.centroid[0] for o in tiles) + pr
        cmax = max(o.centroid[1] for o in tiles) + pc
        pegs = set()
        for o in objs:
            if (o.color, o.size) == tile_cls or o.size >= tsize:
                continue
            if o.size < max(3, tsize // 4):
                continue        # corner nubs of split tiles, not pegs
            # only pegs INSIDE this component's extent: a satellite
            # board's pegs and connector decor are not on this board
            if not (r0 - pr < o.centroid[0] < rmax
                    and c0 - pc < o.centroid[1] < cmax):
                continue
            cell = cell_of(*o.centroid)
            if cell is None:
                continue
            pegs.add(cell)
            board.setdefault(cell, (r0 + cell[0] * pr, c0 + cell[1] * pc))
        if not (2 <= len(pegs) <= max(2, len(board) - 1)):
            return None
        return board, pegs

    def _peg_step(self, obs):
        """Select-and-jump piece removal (peg solitaire). VERIFIED BY USE
        before engaging: one two-click experiment must visibly remove a
        peg (total budget if this is NOT such a game: <= 6 clicks)."""
        st = self._peg
        if st["mode"] == "refuted":
            return None
        scan = self._peg_scan(obs)
        if scan is None:
            return None
        board, pegs = scan
        if st["queue"]:
            cell = st["queue"].pop(0)
            a6 = GameAction.ACTION6
            cy, cx = board.get(cell, (None, None))
            if cy is None:
                st["queue"] = []
                return None
            a6.set_data(self.perc.click_xy((round(cy), round(cx))))
            return a6
        if st["await"] is not None:
            prev_n, dst = st["await"]
            st["await"] = None
            if len(pegs) == prev_n - 1 and dst in pegs:
                st["mode"] = "on"       # the jump happened: mechanics real
            else:
                st["fails"] += 1
                if st["mode"] == "on":
                    st["mode"] = None   # world drifted: re-verify
                if st["fails"] >= 2:
                    st["mode"] = "refuted"
                    return None
        moves = jump_moves(set(board), pegs)
        if not moves:
            return None
        if st["mode"] == "on":
            sol = solve_pegs(set(board), pegs)
            if not sol:
                return None
            src_c, dst_c = sol[0]
        else:
            # the experiment: any legal jump
            src_c, _, dst_c = moves[0]
        st["queue"] = [dst_c]
        st["await"] = (len(pegs), dst_c)
        a6 = GameAction.ACTION6
        cy, cx = board[src_c]
        a6.set_data(self.perc.click_xy((round(cy), round(cx))))
        return a6

    def _perm_grid_col(self, g, w):
        """Occupant color of world cell w read straight from the frame —
        immune to the object-merge blindness that poisons segmentation."""
        r0f, c0f = self.perc.playfield[0], self.perc.playfield[2]
        cr, cc = self.perc.cam
        fr, fc = w[0] - cr + r0f, w[1] - cc + c0f
        if 0 <= fr < g.shape[0] and 0 <= fc < g.shape[1]:
            return int(g[fr, fc])
        return -1

    def _perm_shift_cert(self, bkey, obs, click):
        """WHOLE-TRACK SHIFT CERTIFICATION. Per-piece correspondence is
        unrecoverable when same-color pieces ride adjacent slots (the
        bijection invents swaps and every click contradicts the last).
        But the BUTTON'S TRACK — the set of cells its clicks ever change
        — is pixel-observable, its cyclic order is pure geometry (walk
        the one path/cycle through it), and the operator has ONE degree
        of freedom left: shift direction. Compare the track's occupancy
        vector before/after a click with its two rotations: an exact
        match certifies the ENTIRE operator in one observation."""
        track = self._perm_track.get(bkey)
        if not track or len(track) < 4:
            return
        cl = list(track)

        def dd(a, b):        # manhattan: for the strict triangle prune
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def dc(a, b):        # chebyshev: ring steps are UNIFORM under it
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

        # adjacency in CHEBYSHEV: an octagonal track's diagonal step has
        # twice the manhattan length of its straight step and manhattan
        # adjacency shattered the walk (lp85 L3); under chebyshev both
        # measure one pitch
        nns = [min(dc(c, d) for d in cl if d != c) for c in cl]
        thr = 2 * min(nns)
        nb = {}
        ends = []
        for c in cl:
            ns = [d for d in cl if d != c and dc(c, d) <= thr]
            # TRIANGLE PRUNE: at a track corner two cells one hop from
            # the corner sit within thr of EACH OTHER too — a shortcut
            # edge that breaks the 2-regular cycle test. An edge bridged
            # by a third cell with two strictly shorter legs is not a
            # track edge.
            ns = [d for d in ns
                  if not any(x != c and x != d
                             and dd(c, x) < dd(c, d) and dd(x, d) < dd(c, d)
                             for x in cl)]
            if len(ns) == 1:
                ends.append(c)
            elif len(ns) != 2:
                return
            nb[c] = ns
        if len(ends) not in (0, 2):
            return
        start = ends[0] if ends else cl[0]
        seq = [start, nb[start][0]]
        while True:
            nxt = [d for d in nb[seq[-1]] if d != seq[-2]]
            if len(seq) > len(cl):
                return
            if not nxt or nxt[0] == start:
                break
            seq.append(nxt[0])
        if len(seq) != len(cl):
            return
        pg = last_grid(self.prev_frame)
        cg = last_grid(obs.frame)
        pv = [self._perm_grid_col(pg, c) for c in seq]
        cv = [self._perm_grid_col(cg, c) for c in seq]
        if pv == cv:
            return
        n = len(seq)
        fwd = cv == pv[-1:] + pv[:-1]   # occupant of seq[i] -> seq[i+1]
        bwd = cv == pv[1:] + pv[:1]
        if fwd == bwd:
            return                      # no match, or ambiguous pattern
        if fwd:
            opm = {seq[i]: seq[(i + 1) % n] for i in range(n)}
        else:
            opm = {seq[i]: seq[(i - 1) % n] for i in range(n)}
        ent = self._perm_ops.setdefault(bkey, {"op": {}, "click": click})
        if ent["op"] != opm:
            ent["op"] = opm
            self._perm_fail_sig = None
        ent["cert"] = True
        self._perm_btn_tried.pop(bkey, None)

    def _perm_thr(self, fallback):
        """One-slot move threshold, calibrated from the LOOP LATTICE: a
        single click hops a piece to a nearest-neighbor slot, so the
        threshold is the largest nearest-neighbor gap among known loop
        cells (capped at twice the smallest, so one stray garbage cell
        cannot open the door to multi-slot chain jumps). Anchoring this
        on stored mapping lengths instead froze the threshold at the
        first observation's minimum and starved learning."""
        cells = set()
        for v in self._perm_ops.values():
            cells |= set(v["op"]) | set(v["op"].values())
        if len(cells) < 3:
            return fallback
        cl = list(cells)
        nns = [min(abs(c[0] - d[0]) + abs(c[1] - d[1])
                   for d in cl if d != c) for c in cl]
        return min(max(nns), 2 * min(nns))

    def _perm_occ_moves(self, obs, thr):
        """Cell-occupancy diff over known loop cells: for each piece
        color, an unambiguous single departure->arrival pair at one-slot
        distance is a trusted mapping. Immune to object-merge blindness."""
        lcells = set()
        for v in self._perm_ops.values():
            lcells |= set(v["op"]) | set(v["op"].values())
        if not lcells or not self._perm_pieces:
            return []
        pg = last_grid(self.prev_frame)
        cg = last_grid(obs.frame)
        r0f, c0f = self.perc.playfield[0], self.perc.playfield[2]
        cr, cc = self.perc.cam

        def col_at(g, w):
            fr, fc = w[0] - cr + r0f, w[1] - cc + c0f
            if 0 <= fr < g.shape[0] and 0 <= fc < g.shape[1]:
                return int(g[fr, fc])
            return None

        prev_occ = {w: col_at(pg, w) for w in lcells}
        cur_occ = {w: col_at(cg, w) for w in lcells}
        out = []
        for pc in {c for c, _ in self._perm_pieces}:
            vac = [w for w in lcells
                   if prev_occ[w] == pc and cur_occ[w] != pc]
            arr = [w for w in lcells
                   if cur_occ[w] == pc and prev_occ[w] != pc]
            if len(vac) == 1 and len(arr) == 1:
                d = (abs(vac[0][0] - arr[0][0])
                     + abs(vac[0][1] - arr[0][1]))
                if 0 < d <= thr:
                    out.append((vac[0], arr[0]))
        return out

    def _permuter_step(self, obs):
        """Operator-puzzle mode: with >=2 button operators learned, solve
        the configuration IN THE HEAD and execute the click sequence."""
        if len(self._perm_ops) < 2:
            return None
        # WHAT COUNTS AS A BUTTON: an operator that survives its own use.
        # Entries that keep contradicting themselves are click-noise
        # (rope/physics games shuffle objects too), and a game sprouting
        # MANY uncertified "buttons" is not a button game at all — without
        # this gate the permuter hijacked r11l's click budget entirely.
        ents = {k: v for k, v in self._perm_ops.items()
                if v["op"] and (v.get("cert")
                                or (not v.get("bad")
                                    and v.get("conf", 0) >= 2))}
        if len(ents) < 2:
            return None
        # CARDINALITY LESSON: no cap on how many buttons a level may
        # have (the ">8 = not a button game" rule was a band-aid that
        # blocked lp85 L4's 16 REAL buttons). What distinguishes a
        # button game is per-key VERIFICATION BY USE — already enforced
        # above — never the count of participants.
        ops = {k: v["op"] for k, v in ents.items()}
        ring_thr = self._perm_thr(4)
        ops = augment_inverses(ops)
        ops = {k: close_ring(v, ring_thr) for k, v in ops.items()}
        ops = augment_inverses(ops)
        loop_cells = set()
        for op in ops.values():
            loop_cells |= set(op) | set(op.values())

        def snap(cell):
            best, bd = None, 4      # goal fragments peek from slot EDGES
            for lc in loop_cells:
                d = abs(lc[0] - cell[0]) + abs(lc[1] - cell[1])
                if d < bd:
                    bd, best = d, lc
            return best

        objs = self.perc.objects(obs.frame)
        r0f, c0f = self.perc.playfield[0], self.perc.playfield[2]
        cr, cc = self.perc.cam
        pieces = {}
        frag_by_color = {}
        big_by_color = {}
        min_psz = min((s for _, s in self._perm_pieces), default=1 << 30)
        from collections import Counter as _C7
        _szc = _C7(s for _, s in self._perm_pieces)
        mode_psz = _szc.most_common(1)[0][0] if _szc else None
        for o in objs:
            w = (round(o.centroid[0] - r0f + cr),
                 round(o.centroid[1] - c0f + cc))
            if (o.color, o.size) in self._perm_pieces:
                self._perm_lattice.add(w)   # pieces only ever sit on tracks
                cell = snap(w)
                if cell is not None:
                    pieces[cell] = (o.color, o.size)
            elif o.size < min_psz:
                # goal markers are SUB-PIECE fragments (ring dots peeking
                # around a slot). Anything piece-sized or bigger that isn't
                # a piece is a MERGE — never marker material (it invented
                # goals wherever two pieces touched)
                frag_by_color.setdefault(o.color, []).append(w)
            else:
                big_by_color.setdefault(o.color, []).append(w)
        # the ring STEP (one-slot move distance) is known from the learned
        # operators; marker geometry scales with it
        step_r = min((abs(a3[0] - b3[0]) + abs(a3[1] - b3[1])
                      for v3 in self._perm_ops.values()
                      for a3, b3 in v3["op"].items()), default=8)
        # a goal MARKER is often a ring of fragments around a slot: cluster
        # same-color fragments and snap the cluster centroid to the slot
        for col, pts in frag_by_color.items():
            used = [False] * len(pts)
            for i in range(len(pts)):
                if used[i]:
                    continue
                cluster = [pts[i]]
                used[i] = True
                for j in range(i + 1, len(pts)):
                    # chain radius = one ring step plus rounding slack (the
                    # camera scales the grid non-integrally, so observed
                    # steps jitter +-1): same-slot dots connect, dots of
                    # NEIGHBORING slots (2 steps apart) stay apart — a
                    # wider radius fused two goals into one phantom marker
                    # floating between them (lp85 L2)
                    if not used[j] and any(
                            abs(pts[j][0] - q[0]) + abs(pts[j][1] - q[1])
                            <= step_r + 1 for q in cluster):
                        cluster.append(pts[j])
                        used[j] = True
                cen = (round(sum(q[0] for q in cluster) / len(cluster)),
                       round(sum(q[1] for q in cluster) / len(cluster)))
                # remember the RAW centroid, not an interpretation of it:
                # snapping at store time froze early mistakes (fragments of
                # a not-yet-learned loop snapped onto the wrong loop and
                # haunted the goal memory forever). Interpretation happens
                # at USE time, with current knowledge.
                ex = self._perm_markers.setdefault(col, set())
                if not any(abs(p[0] - cen[0]) + abs(p[1] - cen[1]) <= 2
                           for p in ex):
                    ex.add(cen)

        def snap_marker(cell):
            # tight: a marker cluster's centroid sits ON its slot; a
            # centroid far from every known loop cell belongs to a loop
            # we have not learned yet — refuse to guess
            best, bd = None, 3
            for lc in loop_cells:
                d = abs(lc[0] - cell[0]) + abs(lc[1] - cell[1])
                if d < bd:
                    bd, best = d, lc
            return best

        button_colors = {k[0] for k in ents}
        marker_cells = {}
        for c, pts2 in self._perm_markers.items():
            if c == self.perc.bg or c in button_colors:
                continue
            sc = {snap_marker(p) for p in pts2}
            sc.discard(None)
            if sc:
                marker_cells[c] = sc
        # MARKER-COLORED PIECE, STRUCTURALLY RECOGNIZED: a goal tile
        # WEARS its marker's color, and the solver is powerless unless
        # that class exists. Registration must not depend on catching one
        # clean move (lp85 L3's single goal tiles moved diagonally; the
        # adjacency filter dropped their pairs and the goal classes never
        # existed). A standard-piece-sized object of a MARKER color on a
        # track slot is a piece of that class.
        for o in objs:
            if (mode_psz is None or o.size != mode_psz
                    or o.color not in marker_cells):
                continue
            w = (round(o.centroid[0] - r0f + cr),
                 round(o.centroid[1] - c0f + cc))
            cell = snap(w)
            if cell is not None:
                pieces.setdefault(cell, (o.color, o.size))
                self._perm_pieces.add((o.color, o.size))
                self._perm_lattice.add(w)
        # PIECE FUSED WITH SAME-COLOR DECOR: a piece touching its own
        # color's marker dots (parked on the goal — or one slot away,
        # brushing them) merges into one over-sized object and VANISHES
        # from the class scan. Any over-sized object of a piece color
        # near the loop contains a piece: register it at its slot.
        for col, ws in big_by_color.items():
            psz = [s for c2, s in self._perm_pieces if c2 == col]
            if not psz:
                continue
            for w in ws:
                cell = snap(w)
                if cell is not None:
                    pieces.setdefault(cell, (col, psz[0]))
        if not pieces:
            return None

        if self._perm_sol is not None and not self._perm_sol:
            # solution fully executed yet we are still here (no level-up):
            # the active goal hypothesis was wrong — refute it
            ak = getattr(self, "_perm_active_akey", None)
            if ak is not None:
                self._perm_bad.add(ak)
                self._perm_active_akey = None
            self._perm_sol = None
        if self._perm_sol:
            key = self._perm_sol.pop(0)
            ent = self._perm_ops.get(key)
            if ent is None:
                self._perm_sol = None
                return None
            a6 = GameAction.ACTION6
            a6.set_data(self.perc.click_xy(ent["click"]))
            return a6

        # FREE INFORMATION FIRST (Law 3): a same-class button whose
        # operator is unknown is one click away from teaching it. An
        # incomplete operator set makes solvable levels look unsolvable
        # (observed: lp85 L2 has three loops = six buttons; reasoning
        # over two of them can never park pieces on the other loops).
        # Complete the set before searching.
        # ...but only PAY for completion once a solve has actually failed:
        # partial operators often already solve the level (lp85 L1 needs
        # 15 actions with half-learned ops; completing first cost 12 more)
        for o in (objs if getattr(self, "_perm_need_info", False) else ()):
            if (o.color not in button_colors
                    or o.size not in self._perm_btn_sizes):
                continue
            bk2 = (o.color, (round(o.centroid[0]), round(o.centroid[1])))
            known_op = ops.get(bk2) or {}
            if known_op and not (set(known_op.values())
                                 - set(known_op.keys())):
                continue    # operator CLOSED (a full permutation): done
            # an OPEN operator (a value cell we never saw mapped onward)
            # is still teaching: each click moves pieces onto unknown
            # cells and the next click maps them. Budget resets whenever
            # the operator grows (see the learning site), so clicking
            # stops only when it stops teaching.
            cap = 3 if not known_op else 12
            if self._perm_btn_tried.get(bk2, 0) >= cap:
                continue    # stopped teaching: not a button / exhausted
            self._perm_btn_tried[bk2] = self._perm_btn_tried.get(bk2, 0) + 1
            a6 = GameAction.ACTION6
            a6.set_data(self.perc.click_xy(
                (round(o.centroid[0]), round(o.centroid[1]))))
            return a6

        sig = (frozenset(pieces.items()), len(loop_cells),
               sum(len(v) for v in marker_cells.values()))
        if self._perm_fail_sig == sig:
            return None
        if getattr(self, "_perm_cool", 0) > 0:
            self._perm_cool -= 1
            return None
        for assign in goal_assignments(set(pieces.values()), marker_cells):
            # refutation is keyed by CONTENT: an assignment refuted with
            # yesterday's knowledge is retried once new goal cells appear
            akey = frozenset((cls, frozenset(cells))
                             for cls, cells in assign.items())
            if akey in self._perm_bad:
                continue
            if sum(1 for c2 in pieces.values() if c2 in assign) > 8:
                continue    # quotient space too big to be a real goal:
                            # the permissive catch-all tracks EVERY piece
                            # and only ever burned the node budget
            sol = perm_solve(pieces, ops, assign)
            if sol == []:
                # sim says solved but the level did not advance: this
                # goal assignment is wrong AS KNOWN TODAY
                self._perm_bad.add(akey)
                continue
            if sol:
                self._perm_sol = sol
                key = self._perm_sol.pop(0)
                a6 = GameAction.ACTION6
                a6.set_data(self.perc.click_xy(self._perm_ops[key]["click"]))
                return a6
        # ACTIVE HYPOTHESIS TESTING: when a class has MORE pieces than
        # known goal cells and the loop is always full (goals can never be
        # revealed by vacating), enumerate candidate hidden-goal cells and
        # TEST them — plan to cover known+candidate, execute; no level-up
        # refutes the candidate (content-keyed), and the space is finite.
        ident = {}
        for pc in set(pieces.values()):
            col = pc[0] if isinstance(pc, tuple) else pc
            if col in marker_cells and marker_cells[col]:
                ident[pc] = set(marker_cells[col])
        for cls, known in ident.items():
            n_pieces = sum(1 for c2 in pieces.values() if c2 == cls)
            missing = n_pieces - len(known)
            if missing != 1:
                continue        # v1: hypothesize one hidden goal at a time
            # candidate ordering: mirror of the known goal across the loop
            # centroid first (levels are commonly symmetric), then farthest
            lc = list(loop_cells)
            crm = sum(c[0] for c in lc) / len(lc)
            ccm = sum(c[1] for c in lc) / len(lc)
            kg = next(iter(known))
            mirror = (round(2 * crm - kg[0]), round(2 * ccm - kg[1]))
            def h_order(cell):
                dm = abs(cell[0] - mirror[0]) + abs(cell[1] - mirror[1])
                return (dm, -(abs(cell[0] - kg[0]) + abs(cell[1] - kg[1])))
            # ONE hypothesis solved per call (a full sweep per action
            # hangs the harness); the queue persists across calls
            if not getattr(self, "_perm_hq", None):
                self._perm_hq = [(cls, g) for g in
                                 sorted(loop_cells - known, key=h_order)]
            while self._perm_hq:
                hcls, g = self._perm_hq.pop(0)
                hyp = dict(ident)
                hyp[hcls] = ident.get(hcls, set()) | {g}
                akey = frozenset((c2, frozenset(v2))
                                 for c2, v2 in hyp.items())
                if akey in self._perm_bad:
                    continue
                sol = perm_solve(pieces, ops, hyp, max_nodes=40000)
                if sol == []:
                    self._perm_bad.add(akey)
                    continue
                if sol:
                    self._perm_sol = sol
                    self._perm_active_akey = akey
                    key = self._perm_sol.pop(0)
                    a6 = GameAction.ACTION6
                    a6.set_data(self.perc.click_xy(
                        self._perm_ops[key]["click"]))
                    return a6
                break           # solver budget spent this call
            break               # one class per call is enough

        self._perm_fail_sig = sig
        self._perm_cool = 12
        # solving failed with what we know: information now has positive
        # expected value — unlock operator-completion clicking
        self._perm_need_info = True
        return None

    def _multibody_step(self, obs):
        bodies = detect_bodies(self.tracker)
        # body knowledge survives tracker resets (deaths): remember each
        # body's action map by (color, size, spawn quadrant) and rebuild
        fh = self.perc.playfield[1] - self.perc.playfield[0]
        fw = self.perc.playfield[3] - self.perc.playfield[2]

        def quad(tr):
            return (0 if tr.pos[0] < fh / 2 else 1,
                    0 if tr.pos[1] < fw / 2 else 1)
        if bodies:
            for tid, amap in bodies:
                tr = self.tracker.tracks.get(tid)
                if tr is not None:
                    self._mb_memory[(tr.color, tr.size, quad(tr))] = dict(amap)
        elif len(self._mb_memory) >= 2:
            rebuilt = []
            for tid, tr in self.tracker.tracks.items():
                if tr.missed or tr.is_autonomous:
                    continue
                amap = self._mb_memory.get((tr.color, tr.size, quad(tr)))
                if amap:
                    rebuilt.append((tid, amap))
            if len(rebuilt) >= 2:
                bodies = rebuilt
        if not bodies:
            return None
        # step magnitude: modal slide length of any known motion
        mags = [max(abs(v[0]), abs(v[1]))
                for v in self.model.known_motions.values()]
        step = int(max(1, min(mags))) if mags else 1
        positions = []
        amaps = []
        for tid, amap in bodies:
            tr = self.tracker.tracks.get(tid)
            if tr is None:
                return None
            positions.append((int(round(tr.pos[0])), int(round(tr.pos[1]))))
            amaps.append(amap)
        # verification of the previous command's prediction: a body that
        # failed to move as predicted found an unknown wall — learn it
        if self._mb_pred is not None:
            for (pr, pc), (ar, ac) in zip(self._mb_pred, positions):
                if abs(pr - ar) + abs(pc - ac) > 2:
                    pass            # tracker re-assignment; ignore
                elif (pr, pc) != (ar, ac) and abs(pr - ar) + abs(pc - ac) <= 2 * step:
                    self.model.mark_wall_from_failure((pr, pc), (0, 0))
                    self._mb_plan = []
            self._mb_pred = None
        self._mb_last_positions = list(positions)

        passmask = self.model._passable()

        def passable(r, c):
            rr, cc = wc((r, c))
            if not (0 <= rr < CANVAS and 0 <= cc < CANVAS):
                return False
            return bool(passmask[rr, cc])

        if not self._mb_plan:
            # planner throttle: a failed search is expensive — do not retry
            # until the world has actually changed (positions or walls)
            if getattr(self, "_mb_fail_cool", 0) > 0:
                self._mb_fail_cool -= 1
                return None
            plan = plan_merge(positions, amaps, step, passable,
                              max_nodes=20000)
            if plan is None or not plan:
                if plan == []:
                    # already merged-adjacent and nothing happened =>
                    # the merge archetype is wrong for this game
                    self._mb_refuted = True
                self._mb_fail_cool = 12
                return None
            self._mb_fail_cool = 0
            self._mb_plan = plan
        a = self._mb_plan.pop(0)
        pred = []
        for pos, amap in zip(positions, amaps):
            d = amap.get(a)
            if d is None:
                pred.append(pos)
                continue
            nr, nc = pos[0] + d[0] * step, pos[1] + d[1] * step
            pred.append((nr, nc) if passable(nr, nc) else pos)
        self._mb_pred = pred
        return GameAction[a]

    def _align_step(self, obs):
        """Arrange-then-commit: align the deflector piece's ends over two
        targets beneath the source, then fire the commit action."""
        # the commit action is the ODD ONE OUT: available, but neither a
        # movement, nor a rotation, nor a click — the 'do it' button
        commits = [a.name for a in available(obs)
                   if a not in (GameAction.ACTION6, GameAction.RESET)
                   and a.name not in self.model.known_motions
                   and a.name not in self.model.shape_op_actions]
        if len(commits) != 1 or self.model.avatar_pos is None \
                or not self.model.known_motions:
            return None
        objs = self.perc.objects(obs.frame)
        r0f, c0f = self.perc.playfield[0], self.perc.playfield[2]
        # everything below works in ENGINE GRID units: the movement step
        # reveals the cell size (sizes from perception are raw pixels)
        mags = [max(abs(v[0]), abs(v[1]))
                for v in self.model.known_motions.values()]
        step = int(max(1, min(mags))) if mags else 1
        cell2 = step * step
        from collections import Counter as _C3
        classes = _C3((o.color, o.size) for o in objs)
        targets = None
        for (col, sz), n in classes.items():
            if n >= 2 and sz >= 4 * cell2:
                cand = [o for o in objs if o.color == col and o.size == sz]
                if all(abs(o.bbox[0] - cand[0].bbox[0]) <= step for o in cand):
                    targets = cand
                    break
        if not targets:
            return None
        smalls = [o for o in objs if o.size <= 2 * cell2]
        if not smalls:
            return None
        source = min(smalls, key=lambda o: o.bbox[0])
        source_gcol = round((source.centroid[1] - c0f
                             + self.perc.cam[1]) / step)
        av = None
        av_key = self.model.avatar_shape_key
        if av_key is None:
            return None
        for o in objs:
            if o.shape_key == av_key or (o.color == av_key[0]
                                         and o.size == len(av_key[1])):
                av = o
                break
        if av is None:
            return None
        piece_gw = (av.bbox[3] - av.bbox[1] + 1 + step - 1) // step
        av_gx = round((av.bbox[1] - c0f + self.perc.cam[1]) / step)
        spans = [(round((o.bbox[1] - c0f + self.perc.cam[1]) / step),
                  round((o.bbox[3] - c0f + self.perc.cam[1]) / step))
                 for o in targets]
        gw = (self.perc.playfield[3] - self.perc.playfield[2]) // step
        if self._align_target is None:
            cands = deflector_placements(piece_gw, source_gcol, spans, gw,
                                         self._align_refuted)
            if not cands:
                return None
            self._align_target = cands[0]
            self._align_committing = False
        dx = self._align_target - av_gx
        if dx != 0:
            step = 1
            for aname, vec in self.model.known_motions.items():
                if vec[1] != 0 and (vec[1] > 0) == (dx > 0):
                    self._align_committing = False
                    return GameAction[aname]
            return None
        # arranged: commit (and keep committing through the animation)
        self._align_committing = True
        return GameAction[commits[0]]

    def _carrier_step(self, obs):
        """Carrier-matching (ls20 engine): make the carried display equal
        the required display by stepping on attribute-cycling tiles, then
        walk to the gate (the required display's location)."""
        if self._carried_box is None or not self._tile_fx \
                or self.model.avatar_pos is None \
                or not self.model.known_motions:
            return None
        objs = self.perc.objects(obs.frame)
        r0f, c0f = self.perc.playfield[0], self.perc.playfield[2]
        cr, cc = self.perc.cam
        carried = None
        cb = self._carried_box
        for o in objs:
            if abs(o.bbox[0] - cb[0]) <= 3 and abs(o.bbox[1] - cb[1]) <= 3 \
                    and o.size >= 3:
                carried = o
                break
        if carried is None:
            return None
        # required display: most-similar-sized shape object elsewhere
        req, best = None, 1e9
        for o in objs:
            if o is carried or o.size < 3:
                continue
            if abs(o.bbox[0] - cb[0]) <= 4 and abs(o.bbox[1] - cb[1]) <= 4:
                continue        # part of the carried HUD
            if not (0.4 * carried.size <= o.size <= 2.5 * carried.size):
                continue
            d = abs(o.size - carried.size)
            if d < best:
                best, req = d, o
        if req is None:
            return None
        gate = (round(req.centroid[0] - r0f + cr),
                round(req.centroid[1] - c0f + cc))
        def interior_key(o):
            # displays are often perceived MERGED with their frames; the
            # meaningful shape is the INTERIOR (>=2 cells from the bbox
            # edge), which strips the frame ring
            r0b, c0b, r1b, c1b = o.bbox
            cells = [(r, c) for r, c in o.cells
                     if r - r0b >= 2 and r1b - r >= 2
                     and c - c0b >= 2 and c1b - c >= 2]
            if not cells:
                return o.shape_key
            rm = min(r for r, _ in cells)
            cm = min(c for _, c in cells)
            return (o.color, frozenset((r - rm, c - cm) for r, c in cells))

        ck, rk = interior_key(carried), interior_key(req)
        if ck[1] == rk[1] and ck[0] == rk[0]:
            need = None             # matched: go to the gate
        elif ck[0] != rk[0] and ck[1] == rk[1]:
            need = "color"
        else:
            need = "form"
            for opf in OPS.values():
                if opf(ck[1]) == rk[1]:
                    need = "rot"
                    break
        if need is None:
            plan = self.model.plan_to(gate)
            if plan:
                self.plan = plan[1:]
                return self._to_action(plan[0])
            return None
        # find a learned tile of the needed effect; else nearest unlearned
        # special tile (exploration teaches it)
        tile_color = next((tc for tc, fx in self._tile_fx.items()
                           if fx == need), None)
        if tile_color is None:
            return None
        cells = []
        for o in objs:
            if o.color == tile_color and o.size <= 60:
                cells.append((round(o.centroid[0] - r0f + cr),
                              round(o.centroid[1] - c0f + cc)))
        if not cells:
            return None
        av = self.model.avatar_pos
        cells.sort(key=lambda p2: abs(p2[0] - av[0]) + abs(p2[1] - av[1]))
        tgt = cells[0]
        here_d = abs(av[0] - tgt[0]) + abs(av[1] - tgt[1])
        if here_d <= 2:
            # standing on the tile but still mismatched: step OFF so the
            # next approach re-triggers the cycle
            for aname, vec in self.model.known_motions.items():
                nr, nc = av[0] + vec[0], av[1] + vec[1]
                if self.model.plan_to((nr, nc)):
                    return GameAction[aname]
            return None
        plan = self.model.plan_to(tgt)
        if plan:
            self.plan = plan[1:]
            return self._to_action(plan[0])
        return None

    def _dock_step(self, obs):
        rots = [a for a, op in self.model.shape_op_actions.items()
                if op.startswith("rot")]
        if (not rots or self.model.avatar_pos is None
                or self.model.avatar_shape_key is None
                or not self.model.known_motions):
            self._dock_why = "gates"
            return None
        objs = self.perc.objects(obs.frame)
        r0f, c0f = self.perc.playfield[0], self.perc.playfield[2]
        cr, cc = self.perc.cam

        def wcells(o):
            return frozenset((r - r0f + cr, c - c0f + cc, o.color)
                             for r, c in o.cells)

        # seed = the SELECTED piece when the livery is known (clicking a
        # piece recolors it — the piece wearing that color is controlled);
        # otherwise the tracked avatar object (may be one PART of a
        # compound piece); the rigid group completes it either way
        timer_colors = {tcl for _, tcl in self._timer_bars}
        parts = [o for o in objs if o.color not in timer_colors]
        seed = None
        # PIN THE SEED TO THE COMMITTED PIECE (2nd attempt, with the
        # frame-quality gate the first attempt taught us): in multi-piece
        # levels global avatar identity thrashes between pieces and port
        # fragments, and a dock that re-seeds from it navigates a
        # different piece every few frames. While a pose is committed the
        # piece is found by its OWN signature. The first attempt failed
        # because the identity path's d>20 abort was a load-bearing
        # transient filter — acting on every frame let arrival/refute
        # judgments fire on corrupted perception. The filter is restored
        # explicitly: arrival/refute below require a CLEAN frame
        # (orientation matched AND body cell-count exact).
        if self._dock is not None and self._dock.get("ssig"):
            sc_, sn_ = self._dock["ssig"]
            ref = self._dock.get("scen") or self.model.avatar_pos
            best_d = 1e9
            for o in objs:
                if o.color != sc_ or not (0.4 * sn_ <= o.size
                                          <= 2.5 * sn_):
                    continue
                w = wcells(o)
                cen = (sum(r for r, _, _ in w) / len(w),
                       sum(c for _, c, _ in w) / len(w))
                d = (abs(cen[0] - ref[0]) + abs(cen[1] - ref[1])
                     if ref is not None else 0)
                if d < best_d:
                    best_d, seed = d, o
        # NOTE: selection-livery seeding (learned body recolor) is parked —
        # it mislearns when control changes pieces; revisit with stricter
        # evidence (color that appears iff the clicked piece responds)
        if seed is None:
            av_color = self.model.avatar_shape_key[0]
            expect = max(len(self.model.avatar_shape_key[1]), 1)
            best_d = 1e9
            for o in objs:
                if o.color != av_color:
                    continue
                if not (0.4 * expect <= o.size <= 2.5 * expect):
                    continue
                w = wcells(o)
                cen = (sum(r for r, _, _ in w) / len(w),
                       sum(c for _, c, _ in w) / len(w))
                d = (abs(cen[0] - self.model.avatar_pos[0])
                     + abs(cen[1] - self.model.avatar_pos[1]))
                if d < best_d:
                    best_d, seed = d, o
            if seed is None or best_d > 20:
                self._dock_why = f"seed:{best_d if seed else 'none'}"
                return None
        # CONTACT IS THE GOAL, NOT A MERGER: once a pose is committed the
        # socket's cells are frozen in the pose (sock_pos). On the final
        # approach the piece comes within group_cells' adjacency gap of the
        # socket, and unfiltered rigid-group growth would swallow the
        # static socket into the avatar group — corrupting orientation,
        # centroid, and the socket-seed search at the exact moment of
        # docking (observed cn04: no_socket_seed abort loop one step from
        # the pose). Exclude committed-socket cells from group growth.
        if self._dock is not None and self._dock.get("sock_pos"):
            sp = self._dock["sock_pos"]
            parts = [o for o in parts
                     if not {(r, c) for r, c, _ in wcells(o)} <= sp]
        av_colored, av_group = self._motion_group(seed, parts, wcells)
        if self._dock is not None and self._dock.get("sock_pos"):
            # cell-wise subtraction, not just object filtering: ports share
            # a color across the two sides, so at contact the segmenter
            # FUSES piece port + socket port into one object (neither
            # subset of the frozen footprint) — observed cn04: orientation
            # failed exactly at contact distance, killing the pose one
            # step from the goal. The cavity is bg at commit, so the
            # piece's own cells are never in sock_pos.
            sp = self._dock["sock_pos"]
            # color-aware subtraction: while navigating INTO the cavity
            # the piece's body legitimately covers cells that were socket
            # at commit (the piece renders on top). A footprint cell
            # showing a piece BODY color is the piece; only cells showing
            # non-body colors are the socket. Position alone ate the body
            # mid-approach (observed cn04: piece_lost/orientation death
            # one step from the pose, forever re-committing the same pose).
            bcols = self._dock.get("body_cols", frozenset())
            av_colored = frozenset(
                t for t in av_colored
                if (t[0], t[1]) not in sp or t[2] in bcols)
            if not av_colored:
                self._dock_why = "piece_lost"
                return None
            # PORTS ARE PART OF THE PIECE: at contact the piece's port
            # fuses with the socket's port into one object whose track is
            # reborn mid-flicker — the co-movement veto then drops it
            # (comoves==0, part_moves>=2) and the piece appears to lose a
            # port at the exact moment of docking (observed cn04: the
            # group missed one 3x3 port block only at the pose).
            # Re-attach port-colored objects adjacent to the group,
            # trimmed of the frozen socket footprint.
            pcols_ = self._dock.get("port_cols", ())
            gpos = {(r, c) for r, c, _ in av_colored}
            self._dbg_ra = []
            for o in parts:
                if o.color not in pcols_:
                    continue
                w = {t for t in wcells(o) if (t[0], t[1]) not in sp}
                if not w or w <= av_colored:
                    self._dbg_ra.append((o.color, o.size, "sub_or_empty"))
                    continue
                if any(abs(r - r2) <= 1 and abs(c - c2) <= 1
                       for r, c, _ in w for (r2, c2) in gpos):
                    av_colored = frozenset(av_colored | w)
                    gpos |= {(r, c) for r, c, _ in w}
                    self._dbg_ra.append((o.color, o.size, "attached"))
                else:
                    self._dbg_ra.append((o.color, o.size, "not_adjacent",
                                         sorted(w)[:4]))
        group_ids = set(map(id, av_group))
        av_cen0 = (sum(r for r, _, _ in av_colored) / len(av_colored),
                   sum(c for _, c, _ in av_colored) / len(av_colored))

        # socket = the most hollow static group NOT part of the avatar
        logical = self.perc.logical(obs.frame)
        bg = self.perc.bg

        def is_bg_world(r, c):
            fr, fc = r - cr + r0f, c - cc + c0f
            if 0 <= fr < logical.shape[0] and 0 <= fc < logical.shape[1]:
                return int(logical[fr, fc]) == bg
            return False

        # the socket is wherever the avatar's PORT colors reappear outside
        # its own group (connectors share a color across the two sides) —
        # seed from the nearest such object and grow its rigid group.
        # DISCOVERY ONLY: with a pose committed, the socket is already
        # frozen in the pose; re-deriving it every step made the dock
        # abort (and _switch_piece even KILLED the pose) exactly when the
        # piece got close enough for groups to blur — and the port-color
        # emptiness test would fire on the success recolor itself,
        # switching pieces before success could be registered.
        sock_colored = None
        if self._dock is None:
            av_pos_cells = {(r, c) for r, c, _ in av_colored}
            from collections import Counter as _C
            col_counts = _C(col for _, _, col in av_colored)
            # connectors are small features: a PORT color is a minority
            # color of the piece (prongs/notches), never its body color
            av_port_colors = {col for col, n in col_counts.items()
                              if n <= 0.3 * len(av_colored)}
            if not av_port_colors:
                return self._switch_piece(objs, wcells, group_ids)
            best_seed, best_d2 = None, 1e18
            for o in parts:
                if id(o) in group_ids or o.color not in av_port_colors:
                    continue
                w = wcells(o)
                if {(r, c) for r, c, _ in w} & av_pos_cells:
                    continue
                cen = (sum(r for r, _, _ in w) / len(w),
                       sum(c for _, c, _ in w) / len(w))
                d2 = ((cen[0] - av_cen0[0]) ** 2 + (cen[1] - av_cen0[1]) ** 2
                      if av_cen0 else 0)
                if d2 < best_d2:
                    best_d2, best_seed = d2, o
            if best_seed is None:
                self._dock_why = "no_socket_seed"
                return self._switch_piece(objs, wcells, group_ids)
            sock_colored, _ = group_cells(
                best_seed, [o for o in parts if id(o) not in group_ids],
                wcells)
            if sock_colored & av_colored:
                self._dock_why = "groups_merged"
                return None

        # SELECTION BELIEF: in select-then-act games (cn04 engine) commands
        # only move the SELECTED piece, and clicking the piece TOGGLES its
        # selection. Never click blindly: click exactly once when a within-
        # bounds move was ignored, then re-test with a move.
        av_cen = (sum(r for r, _, _ in av_colored) / len(av_colored),
                  sum(c for _, c, _ in av_colored) / len(av_colored))
        # centroid comparison is only meaningful for the SAME piece: key it
        # to the seed's identity, else a seed switch reads as phantom motion
        sig = (seed.color, seed.size)
        prev = getattr(self, "_dock_prev_cen", None)
        if prev is not None and prev[0] == sig:
            moved_since = (abs(av_cen[0] - prev[1][0])
                           + abs(av_cen[1] - prev[1][1])) >= 0.5
        else:
            moved_since = True      # unknown/new piece: no basis to click
        if moved_since:
            self._sel_belief = True
            self._dock_cmds = 0
        elif self._dock_cmds >= 1 and self._sel_belief is not True \
                and not self._just_clicked and self._dock_inbounds:
            # a within-bounds command was ignored: piece not selected
            self._dock_cmds = 0
            self._just_clicked = True
            a6 = GameAction.ACTION6
            # CLICK A CELL OF THE PIECE, NOT ITS CENTROID: a hollow or
            # non-convex piece's centroid lands on bg and selects nothing
            # (observed cn04 L2: an eternal rotate/re-click loop on a
            # piece whose center is a hole)
            _cr, _cc = min(((r, c) for r, c, _ in av_colored),
                           key=lambda t: (abs(t[0] - av_cen[0])
                                          + abs(t[1] - av_cen[1])))
            a6.set_data(self._world_to_click((_cr, _cc)))
            self._dock_prev_cen = (sig, av_cen)
            return a6
        elif self._dock_cmds >= 3 and self._dock_inbounds:
            # even a believed-selected piece ignored 3 straight commands:
            # belief was stale (a probe click toggled it off) — re-click
            self._sel_belief = None
            self._dock_cmds = 0
            if not self._just_clicked:
                self._just_clicked = True
                a6 = GameAction.ACTION6
                _cr, _cc = min(((r, c) for r, c, _ in av_colored),
                               key=lambda t: (abs(t[0] - av_cen[0])
                                              + abs(t[1] - av_cen[1])))
                a6.set_data(self._world_to_click((_cr, _cc)))
                self._dock_prev_cen = (sig, av_cen)
                return a6
        self._just_clicked = False
        self._dock_prev_cen = (sig, av_cen)

        if self._dock is None:
            ar0 = min(r for r, _, _ in av_colored)
            ac0 = min(c for _, c, _ in av_colored)
            step = 1
            mags = [max(abs(v[0]), abs(v[1]))
                    for v in self.model.known_motions.values()]
            if mags:
                step = max(1, min(mags))
            fh = self.perc.playfield[1] - self.perc.playfield[0]
            fw = self.perc.playfield[3] - self.perc.playfield[2]
            poses = best_poses(av_colored, sock_colored, is_bg_world,
                               self._dock_refuted,
                               lattice=(ar0, ac0, int(step)),
                               bounds=(fh, fw))
            if not poses:
                self._dock_why = "no_poses"
                return None
            _, k, tl, fp = poses[0]
            pcols = port_colors(av_colored, sock_colored)
            self._port_colors_seen |= pcols
            # NAVIGATE ON THE BODY, NOT THE PORTS: connectors mate by
            # OVERLAP — at the pose the piece's port cells land exactly
            # on the socket's port cells (inside the frozen footprint)
            # and visually blend into the socket, so a full-piece
            # template can never match at or near the goal (observed
            # cn04: orientation missed a 3x3 port block at contact,
            # killing the pose one step from the win). The body lands in
            # the bg cavity and stays clean; port absorption is read
            # separately as the success signal.
            full_rot = crot_k(cnorm(av_colored), k)
            body_rel = ([(r, c) for r, c, col in full_rot
                         if col not in pcols]
                        or [(r, c) for r, c, _ in full_rot])
            tlb = (tl[0] + min(r for r, _ in body_rel),
                   tl[1] + min(c for _, c in body_rel))
            body0 = (frozenset(t for t in av_colored if t[2] not in pcols)
                     or av_colored)
            self._dock = {"k": k, "tl": tlb, "tlf": tl, "fp": fp,
                          "base_full": cnorm(av_colored),
                          "ssig": (seed.color, seed.size),
                          "scen": av_cen,
                          "base": cnorm(body0),
                          "body_cols": frozenset(
                              col for _, _, col in body0
                              if col not in pcols),
                          "pw": [(tl[0] + r, tl[1] + c)
                                 for r, c, col in full_rot if col in pcols],
                          # the level's palette at commit: a CONNECT
                          # recolor introduces a color outside it
                          "palette": frozenset(
                              int(v) for v in
                              __import__("numpy").unique(logical)),
                          "wait": 0, "port_cols": pcols,
                          "ports_n": sum(1 for _, _, col in av_colored
                                         if col in pcols),
                          # footprint for nav-time exclusion, NOT pose
                          # geometry: grown UNCAPPED so the big frame the
                          # port notches sit on is included — the capped
                          # sock_colored recorded only one small notch,
                          # and at contact the avatar group grew through
                          # the frame into cells outside the footprint
                          "sock_pos": frozenset(
                              (r, c) for r, c, _ in group_cells(
                                  best_seed,
                                  [o for o in parts
                                   if id(o) not in group_ids],
                                  wcells, max_ratio=1e9)[0])}

        dk = self._dock
        # body-only view for orientation and position (ports blend into
        # the socket at contact; see the commit-time note above)
        av_nav = (frozenset(t for t in av_colored
                            if t[2] not in dk.get("port_cols", ()))
                  or av_colored)
        # ORIENTATION NEEDS THE PORTS: a 4-fold-symmetric body matches
        # every rotation, so body-only j degenerates to 0 and a k=3 pose
        # demands rotations forever (observed cn04 L2: an eternal
        # ACTION5 loop on a symmetric piece). Away from the socket the
        # full piece (ports included) is visible — measure j there
        # authoritatively; near contact fall back to the body view, and
        # never press rotate on degenerate evidence.
        jf = orientation_of(dk.get("base_full", dk["base"]), av_colored)
        j = jf if jf is not None else orientation_of(dk["base"], av_nav)
        if j is None:
            # transient group flicker must not kill the pose: 2 strikes
            dk["omiss"] = dk.get("omiss", 0) + 1
            if dk["omiss"] >= 2:
                self._dock = None   # group truly changed outside our model
            return None
        dk["omiss"] = 0
        dk["scen"] = av_cen           # pin follows the piece
        # CLEAN FRAME = orientation matched AND the body is whole; all
        # arrival/refute judgments require it (the restored filter)
        clean = (len(av_nav) == len(dk["base"]))
        presses = (dk["k"] - j) % 4 if jf is not None else 0
        if presses:
            self._dock_cmds += 1
            self._dock_inbounds = True
            return GameAction[rots[0]]
        # navigate by TOP-LEFT: integer-exact, no centroid rounding
        cur_tl = (min(r for r, _, _ in av_nav),
                  min(c for _, c, _ in av_nav))
        # CONVERGENCE CONTRACT: a pose not reached within a bounded
        # number of navigation steps is unreachable-in-practice
        # (parity dithers and blocked corridors both end here) — refute
        # it and try the next, rather than bouncing forever
        dk["nav"] = dk.get("nav", 0) + 1
        if dk["nav"] > 40:
            self._dock_refuted.add(dk["fp"])
            self._dock = None
            return None
        drr = dk["tl"][0] - cur_tl[0]
        dcc = dk["tl"][1] - cur_tl[1]
        if drr == 0 and dcc == 0:
            # SUCCESS IS A NEW COLOR: coincident ports RECOLOR to a
            # connect color that exists nowhere else in the level (cn04
            # engine truth: 8/13 -> 3 at paired cells). Reading "any
            # non-port non-bg color" was confounded by every occluder —
            # the socket frame, our own body — because those are palette
            # colors. A color OUTSIDE the commit-time palette at an
            # expected port position is unfakeable.
            pal = dk.get("palette", frozenset())
            changed = still = 0
            for pr, pc in dk.get("pw", ()):
                fr, fc = pr - cr + r0f, pc - cc + c0f
                if (0 <= fr < logical.shape[0]
                        and 0 <= fc < logical.shape[1]):
                    col = int(logical[fr, fc])
                    if col in dk.get("port_cols", set()):
                        still += 1
                    elif pal and col not in pal:
                        changed += 1
            if changed and not still:
                # every port recolored: connected — multi-piece levels
                # advance only when every piece is done, so hand control
                # to the next port-bearing piece
                self._switch_ok = True
                self._dock = None
                return None
            if not clean:
                return None   # refute only on clean perception
            # the engine recolors ON the arrival frame: one clean
            # unconnected arrival is conclusive. Lingering just cedes
            # control to the generic layer, which drags the piece off
            # the pose and resets the wait forever (observed cn04 L2).
            self._dock_refuted.add(dk["fp"])
            self._dock = None
            return None
        # direct greedy step using known motions (piece overlaps are legal
        # in select-and-move games; terrain planning would refuse them)
        best_a, best_gain = None, 0
        for aname, vec in self.model.known_motions.items():
            # gain = distance actually REMOVED by the move (an overshoot
            # that lands farther than it started is negative). The old
            # min(|vec|, |remaining|) rewarded overshooting a 1-cell gap
            # with a 3-cell step from both sides — a permanent dither
            # (observed cn04: 92 ups vs 94 downs at a parity-offset pose)
            after_r = abs(drr - vec[0])
            after_c = abs(dcc - vec[1])
            gain = (abs(drr) + abs(dcc)) - (after_r + after_c)
            if gain > best_gain:
                best_gain, best_a = gain, aname
        if best_a is None and (drr or dcc):
            # nothing reduces the distance: the pose sits off the
            # movement parity — unreachable, refute and move on
            self._dock_refuted.add(dk["fp"])
            self._dock = None
            return None
        if best_a is not None:
            vec = self.model.known_motions[best_a]
            gr1 = max(r for r, _, _ in av_colored) + vec[0]
            gc1 = max(c for _, c, _ in av_colored) + vec[1]
            gr0 = cur_tl[0] + vec[0]
            gc0 = cur_tl[1] + vec[1]
            fh = self.perc.playfield[1] - self.perc.playfield[0]
            fw = self.perc.playfield[3] - self.perc.playfield[2]
            self._dock_inbounds = (gr0 >= 0 and gc0 >= 0
                                   and gr1 < fh and gc1 < fw)
            if not self._dock_inbounds:
                # the needed direction is edge-clamped: pose unreachable
                self._dock_refuted.add(dk["fp"])
                self._dock = None
                return None
            self._dock_cmds += 1
            return GameAction[best_a]
        self._dock_refuted.add(dk["fp"])
        self._dock = None
        return None

    def _hyp(self, kind, **kw):
        if len(self.hlog) < 600:
            kw["kind"] = kind
            kw["n"] = self._obs_i
            self.hlog.append(kw)

    def _gov(self, name, allowance, fn, obs):
        """FAIL-CLOSED GOVERNOR (the 0.21 lesson): the hidden set ranked
        our line v11=0.30 > v16=0.27 > v25=0.21 — every DEV gain since
        v11 subtracted hidden points, because an archetype that misfires
        on an unknown game burns actions RHAE punishes quadratically,
        and the battery can never catch it (its gates were tuned on the
        very games the battery tests). So: an archetype may spend at
        most `allowance` actions per level; overspend without a level-up
        kills it for the level. Any exception inside an archetype also
        kills it — a crash would otherwise trip the harness's
        random-action fallback, which is worse than generic play."""
        g = self._gov_state.setdefault(name, {"spent": 0, "dead": False})
        if g["dead"]:
            return None
        try:
            a = fn(obs)
        except Exception as e:
            g["dead"] = True
            self._hyp("archetype_crash", archetype=name, err=repr(e)[:80])
            return None
        if a is not None:
            g["spent"] += 1
            if g["spent"] > allowance:
                g["dead"] = True
                self._hyp("archetype_overspend", archetype=name,
                          allowance=allowance)
                return None
        return a

    def act(self, obs):
        self._last_click_key = None

        # THE CORE PROPOSES FIRST (failure atlas): a goal read off the
        # first frame, pursued by learned effects; archetypes are fallbacks
        if "goals" not in self.ablate:
            g_a = self._gov("goals", 130, self._goals_step, obs)
            if g_a is not None:
                return g_a

        # an ENGAGED peg-jump exchange takes priority over probing: the
        # two clicks of a jump must be consecutive (a stray probe —
        # ACTION7 undo especially — cancels the selection between them)
        if ("pegs" not in self.ablate
                and (self._peg["queue"] or self._peg["await"] is not None
                     or self._peg["mode"] == "on")):
            peg_a = self._gov("pegs", 50, self._peg_step, obs)
            if peg_a is not None:
                return peg_a

        # REFERENCE-ASSEMBLY runs before probing: its select->place
        # protocol dies to a stray probe (an undo press especially), and
        # every probe drains the energy budget the protocol needs
        if "refassembly" not in self.ablate:
            rf_a = self._gov("refassembly", 24, self._refassembly_step, obs)
            if rf_a is not None:
                return rf_a

        # REMOTE ACTUATORS likewise own their experiment budget
        if "actuators" not in self.ablate:
            ac_a = self._gov("actuators", 70, self._actuator_step, obs)
            if ac_a is not None:
                return ac_a

        # CONSTRAINT GLYPHS likewise
        if "glyphs" not in self.ablate:
            gl_a = self._gov("glyphs", 40, self._glyphs_step, obs)
            if gl_a is not None:
                return gl_a

        # PROBE FIRST: unresolved action semantics starve if plans always
        # win. Retry rules: a position where this action already FAILED is
        # never retried; after a failure, the next probe spot must differ
        # from it in both coordinates (walls run along lines); successful
        # attempts refund budget, so only genuine inertness exhausts it.
        pos = self.model.avatar_pos
        for name in list(self.unresolved):
            if self.unresolved[name] <= 0:
                del self.unresolved[name]
                # under drift, a truly inert action has delta == -drift;
                # the model can resurrect it from logic alone
                self.model.inert_actions.add(name)
                continue
        # pass 1 prefers spots diversified away from the last failure line;
        # pass 2 relaxes that (a hard rule deadlocks when fallback movement
        # keeps the avatar on the failure line — observed)
        for relaxed in (False, True):
            for name in self.unresolved:
                if (name, pos) in self.probe_failed_keys:
                    continue
                if not relaxed:
                    recent = self._last_fail.get(name)
                    if (pos is not None and recent is not None
                            and (pos[0] == recent[0] or pos[1] == recent[1])):
                        continue
                self.unresolved[name] -= 1  # charge the attempt
                self.plan = []              # position changes; stale plan dies
                return GameAction[name]

        # DOCKING mode: a rotate action exists => orientation matters; fit
        # the avatar into a static object's cavity at the right pose.
        # Runs BEFORE plan-following: docking re-evaluates every step, and
        # its own nav plans go through self.plan anyway.
        if self.model.shape_op_actions and "docking" not in self.ablate:
            dock_a = self._gov("docking", 170, self._dock_step, obs)
            if dock_a is not None:
                return dock_a

        # CARRIER-MATCHING: attribute-cycling tiles + a carried display
        # (not in rotate-ACTION games: there the docking skill owns play)
        if (self._carried_box is not None and not self.model.shape_op_actions
                and "carrier" not in self.ablate):
            car_a = self._gov("carrier", 60, self._carrier_step, obs)
            if car_a is not None:
                return car_a

        # ARRANGE-THEN-COMMIT: a repeatable-effect action plus a movable
        # deflector and multi-instance targets => geometric placement
        if (not self.model.shape_op_actions and self.model.known_motions
                and "aligner" not in self.ablate):
            al_a = self._gov("aligner", 40, self._align_step, obs)
            if al_a is not None:
                return al_a

        # MULTI-BODY mode: several bodies answer the same input with
        # different motions => plan over the joint state (merge archetype)
        if (not self.model.shape_op_actions and not self._mb_refuted
                and "multibody" not in self.ablate):
            mb_a = self._gov("multibody", 80, self._multibody_step, obs)
            if mb_a is not None:
                return mb_a

        # TEMPLATE-OVERLAY mode: movable outlines + same-color dot
        # constellations => exact-superposition goal with entity cycling
        if "overlay" not in self.ablate:
            ov_a = self._gov("overlay", 40, self._overlay_step, obs)
            if ov_a is not None:
                return ov_a

        # EDGE-MAZE mode: certified midpoint-color passability => moves
        # are graph edges and every keypress is metered; graph-search
        # replaces exploration outright
        if "edgemaze" not in self.ablate:
            em_a = self._gov("edgemaze", 110, self._edgemaze_step, obs)
            if em_a is not None:
                return em_a

        if self.plan:
            step = self.plan.pop(0)
            return self._to_action(step)

        # TRANSFORM mode: rotate/flip the piece until congruent with the
        # template, then press a commit candidate
        tf_plan = None if "transform" in self.ablate else \
            self.transform.solve(self.perc.objects(obs.frame))
        if tf_plan is not None:
            if tf_plan:
                self.plan = tf_plan[1:]
                return GameAction[tf_plan[0]]
            commits = self.transform.commit_candidates()
            if commits:
                a = commits[self._commit_i % len(commits)]
                self._commit_i += 1
                return GameAction[a]

        if (self._gate_fails
                and self._obs_i - getattr(self, "_gate_step", 0) > 40):
            self._gate_fails = {}    # gates DECAY: refusal may have been
            self._gate_step = self._obs_i    # timing, not a condition

        # LAYOUT-SWITCH supervisor (dc22 family): when the avatar is
        # visibly STUCK (barely moved for a dozen actions) and a click
        # action exists, the missing move is often not a move at all —
        # side-panel buttons rewrite the platform layout. Click static
        # candidates round-robin; a confirmed rewrite invalidates all
        # terrain knowledge and the generic layers replan on the new
        # world. Every action is metered in these games, so the stuck
        # detector is what stops the dithering that kills us.
        if ("layoutswitch" not in self.ablate
                and self.model.avatar_pos is not None
                and self.model.known_motions):
            avail_ls = getattr(obs, "available_actions", None)
            if (avail_ls is None or 6 in avail_ls):
                ls = self._ls
                if ls["cool"] > 0:
                    ls["cool"] -= 1
                else:
                    tr = self._traj[-16:]
                    stuck = (len(tr) >= 12
                             and len(set(tr)) <= 4)
                    if stuck and ls["clicks"] < 20:
                        # FULL-FRAME candidate scan: buttons live on side
                        # panels that playfield detection crops away (the
                        # sb26 tray lesson) — perc.objects never sees them
                        lg_ls = self.perc.logical(obs.frame)
                        bg_ls = self.perc.bg
                        h_ls, w_ls = lg_ls.shape
                        from collections import deque as _dq5
                        blobs_ls, seen_ls = [], set()
                        for r_ in range(h_ls):
                            for c_ in range(w_ls):
                                col = int(lg_ls[r_, c_])
                                if col == bg_ls or (r_, c_) in seen_ls:
                                    continue
                                q = _dq5([(r_, c_)])
                                comp = [(r_, c_)]
                                seen_ls.add((r_, c_))
                                while q:
                                    rr_, cc_ = q.popleft()
                                    for dr_ in (-1, 0, 1):
                                        for dc_ in (-1, 0, 1):
                                            n_ = (rr_ + dr_, cc_ + dc_)
                                            if (0 <= n_[0] < h_ls
                                                    and 0 <= n_[1] < w_ls
                                                    and n_ not in seen_ls
                                                    and int(lg_ls[n_[0],
                                                                  n_[1]])
                                                    == col):
                                                seen_ls.add(n_)
                                                comp.append(n_)
                                                q.append(n_)
                                if 3 <= len(comp) <= 100:
                                    cen = (sum(a for a, _ in comp)
                                           / len(comp),
                                           sum(b for _, b in comp)
                                           / len(comp))
                                    blobs_ls.append((col, len(comp), cen))
                        av_c = (self.model.avatar_shape_key[0]
                                if self.model.avatar_shape_key else None)
                        avp = self.model.avatar_pos
                        r0l = self.perc.playfield[0]
                        c0l = self.perc.playfield[2]
                        cands_ls = []
                        for col, sz, cen in blobs_ls:
                            if col == av_c:
                                continue
                            # POSITION-ONLY key: a layout button's glyph
                            # displays its state and changes on click —
                            # keying by color would orphan the confirmed
                            # set after every cycle
                            key = (round(cen[0] / 3), round(cen[1] / 3))
                            if (key in ls["dropped"]
                                    and key not in ls["confirmed"]):
                                continue
                            d_ls = (abs(cen[0] - r0l + self.perc.cam[0]
                                        - avp[0])
                                    + abs(cen[1] - c0l + self.perc.cam[1]
                                          - avp[1]))
                            cands_ls.append((d_ls, key, cen))
                        # prefer confirmed buttons (re-clicks cycle
                        # further layouts), then FAR objects (side panels)
                        cands_ls.sort(key=lambda t: (
                            t[1] not in ls["confirmed"], -t[0]))
                        # once buttons are KNOWN, stop paying for
                        # exploration — cycle the confirmed ones only
                        conf_ls = [t for t in cands_ls
                                   if t[1] in ls["confirmed"]]
                        if conf_ls and ls["clicks"] >= 6:
                            cands_ls = conf_ls
                        if cands_ls:
                            _, key, cen = cands_ls[ls["ri"] % len(cands_ls)]
                            ls["ri"] += 1
                            ls["clicks"] += 1
                            ls["cool"] = 3
                            ls["await"] = key
                            a6 = GameAction.ACTION6
                            a6.set_data(self.perc.click_xy(
                                (int(round(cen[0])), int(round(cen[1])))))
                            self.plan = []
                            return a6
        cand = self._goal_candidate(obs)
        if cand is not None and self.model.known_motions:
            target, color = cand
            plan = self.model.plan_to(target)
            if not plan and self.model.shape_op_actions:
                # tolerance is a DOCKING-CLASS concession only (born for
                # cn04's off-lattice pose targets); everywhere else an
                # unreachable goal keeps its honest meaning
                plan = self.model.plan_to(target, tol=True)
            if plan:
                self._seek_color = color
                self._seek_target = target
                self.plan = plan[1:]
                return self._to_action(plan[0])
            elif color not in self._collect_colors:
                # UNREACHABLE IS NOT "NOT THE GOAL" (failure atlas,
                # 2026-09-05): the map-exhausted refutation struck the TRUE
                # goal in dc22, g50t, ls20, su15, tu93 and sx02 — an
                # unreachable goal means a mechanic is missing, which is the
                # open hypothesis's business. Instead: HOLD the candidates,
                # pursue whichever is reachable now, defer the rest.
                deferred = self._goal_deferred
                if color not in deferred:
                    deferred.add(color)
                    self._hyp("goal_deferred", color=color, reason="unreachable_now")
                for alt in self._goal_alternatives(obs, exclude=deferred):
                    t2, c2 = alt
                    plan2 = self.model.plan_to(t2)
                    if plan2:
                        self._seek_color = c2
                        self._seek_target = t2
                        self._hyp_last_goal = c2
                        self.plan = plan2[1:]
                        return self._to_action(plan2[0])

        # LIVE HYPOTHESES FIRST when the learnings gave no plan (Tushar:
        # go with what we know; when that fails, disambiguate the unknown)
        a_an = self._retest_anomaly(obs)
        if a_an is not None:
            return a_an

        # TOUCH THE INTERACTABLES (free-information prior): a small object
        # class sitting ON walkable ground is worth one visit — special
        # tiles teach their semantics on first contact. Prioritize unvisited
        # ones over distant frontiers.
        if (self.model.known_motions and self.model.avatar_pos is not None
                and "touch" not in self.ablate):
            wcs2 = self.model.wall_colors()
            av2 = self.model.avatar_pos
            av_col = self.model.avatar_shape_key[0] \
                if self.model.avatar_shape_key else None
            cand_cells = []
            from collections import Counter as _C4
            counts4 = _C4(o.color for o in self.perc.objects(obs.frame))
            for o in self.perc.objects(obs.frame):
                if (o.color in wcs2 or o.color == self.perc.bg
                        or o.color == av_col or o.size > 80
                        or o.color in self._lethal_colors
                        or counts4[o.color] > 6):
                    continue
                w2 = (round(o.centroid[0] - self.perc.playfield[0]
                            + self.perc.cam[0]),
                      round(o.centroid[1] - self.perc.playfield[2]
                            + self.perc.cam[1]))
                if w2 in self._touched_cells or w2 in self._touch_gaveup:
                    continue
                if abs(w2[0] - av2[0]) + abs(w2[1] - av2[1]) <= 2:
                    self._touched_cells.add(w2)     # standing on it now
                    continue
                cand_cells.append(w2)
            if cand_cells:
                # stepping ON a tile means standing at its cell on the
                # avatar's own movement lattice — snap targets to it
                mags2 = [max(abs(v[0]), abs(v[1]))
                         for v in self.model.known_motions.values()]
                st2 = int(max(1, min(mags2))) if mags2 else 1
                snapped = []
                for g2 in cand_cells:
                    snapped.append((av2[0] + round((g2[0] - av2[0]) / st2) * st2,
                                    av2[1] + round((g2[1] - av2[1]) / st2) * st2))
                cand_cells = snapped
                cand_cells.sort(key=lambda p2: abs(p2[0] - av2[0])
                                + abs(p2[1] - av2[1]))
                for tgt2 in cand_cells[:3]:
                    plan = self.model.plan_to(tgt2)
                    if plan:
                        # a tile counts as touched when REACHED (standing
                        # on it, above), not when a plan toward it was
                        # made: a plan dropped by an unexplained refusal
                        # left the goal tile 'touched' and never retried
                        # (sx05: L1 170 actions with a correct plan in
                        # hand). Three abandoned plans = give up on it.
                        tt = getattr(self, "_touch_tries", None)
                        if tt is None:
                            tt = self._touch_tries = {}
                        tt[tgt2] = tt.get(tgt2, 0) + 1
                        if tt[tgt2] >= 3:
                            self._touch_gaveup.add(tgt2)
                        self._gate_fails = {}    # new information: retry
                        self.plan = plan[1:]
                        return self._to_action(plan[0])

        # EXPLORATION IN TIERS (free information first): (1) cells whose
        # passability the color statistics cannot predict; (2) the
        # exhaustive walk (tu93: walls live on edges between identical
        # floor cells — only visiting tells). Between them the standard
        # click policy gets its turn far earlier than before (sx06: 180
        # of 306 actions were spent visiting an open room cell by cell
        # while the one informative click waited). A dedicated "probe
        # every scarce class" stage was tried here and REFUTED: it
        # preempted the class-scored click policy and r11l's click-tax
        # returned (46 -> 437).
        # (a multi-cell step lattice keeps the exhaustive walk: there a
        # cell's passability is a lattice fact, not a color fact, and the
        # dock's pose search depends on the walked terrain — cn04 99 ->
        # 168 under the strict tier)
        frontier = self.model.nearest_frontier(
            strict=not self.unresolved and not self.model.shape_op_actions)
        if frontier is not None:
            plan = self.model.plan_to(frontier)
            if plan:
                self.plan = plan[1:]
                return self._to_action(plan[0])
        # a few class-scored clicks before the exhaustive walk (value of
        # information: one click vs a hundred steps), through the same
        # policy the click games are tuned on
        if (self._t2_clicks > 0 and self.model.avatar_pos is not None
                and any(a == GameAction.ACTION6 for a in available(obs))
                and not self.model.shape_op_actions):
            click = self._next_click(obs)
            if click is not None:
                self._t2_clicks -= 1
                return click
        frontier = self.model.nearest_frontier(strict=False)
        if frontier is not None:
            plan = self.model.plan_to(frontier)
            if plan:
                self.plan = plan[1:]
                return self._to_action(plan[0])

        # STUCK ESCAPE: a goal is visible but unreachable and nothing is
        # left to explore — we're in an absorbing state (e.g. pinned at a
        # wall by current). A human restarts the level; so do we. The world
        # model survives the reset, so the second attempt plans from the
        # start position with full knowledge. Guard: the engine FULL-resets
        # the whole game if RESET is the first action of a level.
        self._since_reset += 1
        movement_stuck = (cand is not None and self.model.known_motions
                          and self.model.avatar_pos is not None)
        # transform games have a move-budget gauge: when learned transform
        # actions stop having ANY effect, the budget is spent — restart
        transform_stuck = (self.transform.action_ops()
                           and all(self.transform.no_change.get(a, 0) >= 3
                                   for a in self.transform.action_ops()))
        if (movement_stuck or transform_stuck) and self._since_reset > 30:
            self._since_reset = 0
            return GameAction.RESET

        acts = available(obs)
        # PERMUTATION-OPERATOR mode first: learned button operators make
        # the puzzle solvable in the head — clicks then execute a solution
        perm_a = None if "permuter" in self.ablate else self._gov("permuter", 90, self._permuter_step, obs)
        if perm_a is not None:
            return perm_a

        # PEG-SOLITAIRE mode (select piece, jump over neighbor into a
        # hole two cells away); engages only after a verified experiment
        if (any(a == GameAction.ACTION6 for a in available(obs))
                and not self.model.shape_op_actions
                and "pegs" not in self.ablate):
            peg_a = self._gov("pegs", 50, self._peg_step, obs)
            if peg_a is not None:
                return peg_a

        # in select-then-act games (a rotate action exists) stray clicks
        # TOGGLE piece selection and sabotage docking — clicking belongs to
        # the docking/switch machinery alone there
        if self.model.shape_op_actions:
            acts = [a for a in acts if a != GameAction.ACTION6]
        if any(a == GameAction.ACTION6 for a in acts):
            # LOOK-derived goal hypotheses first: 'make canvas match
            # template' tasks are visible in the frame and cost nothing
            try:
                tasks = [] if "blockmatch" in self.ablate else \
                    find_block_match_tasks(self.perc.objects(obs.frame),
                                           self.perc.bg)
            except Exception:
                tasks = []
            # HYPOTHESIS COMMITMENT: pursue ONE (canvas, template) match
            # hypothesis to completion; competing templates disagree cell by
            # cell, and alternating between them repaints forever.
            act_t = None
            if (self._active_hyp is not None
                    and self._active_hyp[0] in self._refuted_canvases):
                self._refuted_hyps.add(self._active_hyp)
                self._active_hyp = None     # its canvas doesn't respond
            if self._active_hyp is not None:
                for t in tasks:
                    if (t.canvas_bbox, t.template_bbox) == self._active_hyp:
                        act_t = t
                        break
                if act_t is None:
                    # canvas now matches this template; if that were the
                    # goal the level would have advanced — refute, move on
                    self._hyp_wait += 1
                    if self._hyp_wait >= 1:
                        self._refuted_hyps.add(self._active_hyp)
                        self._active_hyp = None
            if act_t is None and self._active_hyp is None:
                for t in tasks:
                    hyp = (t.canvas_bbox, t.template_bbox)
                    if (hyp in self._refuted_hyps
                            or t.canvas_bbox in self._refuted_canvases
                            or not t.diff_cells):
                        continue
                    self._active_hyp = hyp
                    self._hyp_wait = 0
                    act_t = t
                    break
            if act_t is not None and act_t.diff_cells:
                r, c, want = act_t.diff_cells[0]
                # select-then-paint: switch brush via a palette block first
                if self._paint_color is not None and self._paint_color != want:
                    pal = self._find_palette_block(obs, want, act_t.canvas_bbox)
                    if pal is not None:
                        self._palette_click = want
                        a6 = GameAction.ACTION6
                        a6.set_data(self.perc.click_xy(pal))
                        return a6
                self._match_click = (act_t.canvas_bbox, (r, c), want)
                a6 = GameAction.ACTION6
                a6.set_data(self.perc.click_xy((r, c)))
                return a6

            click = self._next_click(obs)
            if click is not None:
                return click

        motions = list(self.model.known_motions.keys())
        # a single 'known motion' (su15: the undo key, whose only motion
        # was putting a blob back) must not become the whole repertoire:
        # half the time the fallback is a fair draw over every key
        if motions and (len(motions) >= 2 or self.rng.random() < 0.5 or FALLBACK_OLD):
            a = GameAction[self.rng.choice(motions)]
        else:
            a = self.rng.choice(acts)
        if a == GameAction.ACTION6:
            a.set_data({"x": self.rng.randrange(64), "y": self.rng.randrange(64)})
        return a

    def _to_action(self, name):
        if isinstance(name, tuple):
            # a planned teleport: click the object the law names
            _, tl = name
            color = next((o[1] for o in self.model.last_objs
                          if (o[3][0], o[3][1]) == tuple(tl)), None)
            self._click_ctx = (color, tuple(tl), self.model.avatar_pos,
                               self.model.avatar_shape_key[0]
                               if self.model.avatar_shape_key else None)
            a = GameAction.ACTION6
            a.set_data(self._world_to_click(tl))
            return a
        return GameAction[name]

    # ---------- click exploration ----------

    def _next_click(self, obs):
        """Pick the most promising POSITION to click. Class statistics say
        which kinds of object respond to clicks; per-position counts spread
        clicks across all instances instead of hammering one representative
        (a grid of identical cells is one class but many positions)."""
        objs_w = self._objs_world(obs)
        cands = [(k, color, size, world, cen) for k, color, size, world, cen in objs_w
                 if size <= 40 and k != self.model.avatar_shape_key]
        if not cands:
            return None

        def class_score(k):
            ch, tr = self.click_stats[k]
            return 2.0 if tr == 0 else ch / tr

        best, best_s = None, -1e9
        banned_here = self._banned_next.get(tuple(self._seq), set())
        for item in cands:
            if item[3] in self._death_clicks:
                continue            # this click killed us before, any context
            if item[3] in banned_here:
                continue            # this click after THIS sequence is fatal
            s = class_score(item[0]) - 0.7 * self._clicked.get(item[3], 0)
            if s > best_s:
                best, best_s = item, s
        if best is None:
            return None
        if best_s <= 0.05:
            return None  # everything clickable looks exhausted; fall back
        self._clicked[best[3]] = self._clicked.get(best[3], 0) + 1
        self._last_click_key = best[0]
        self._prev_click_pos = best[3]
        self._prev_click_class = best[0]
        self._seq.append(best[3])
        self._click_ctx = (best[1], (best[3][0], best[3][1]),
                           self.model.avatar_pos,
                           self.model.avatar_shape_key[0]
                           if self.model.avatar_shape_key else None)
        a = GameAction.ACTION6
        xy = self._world_to_click(best[3])
        a.set_data(xy)
        return a

    # ---------- goal heuristics ----------

    def _terrain_colors(self):
        """Colors the avatar has walked on (or bumped): terrain, not
        targets.  WALKED OVER IS NOT 'NOT THE GOAL' either: a colour that
        covers only a few cells of the known world is a thing, not ground
        (st71: a slide carried the body across the goal, and the goal
        became floor)."""
        out = set()
        colors = getattr(self.model, "colors", None)
        for color, (f, w) in self.model.color_stats.items():
            if f + w >= 1:
                if colors is not None:
                    try:
                        area = int((colors == color).sum())
                    except Exception:
                        area = 999
                    if area < 24 and w == 0:
                        continue
                out.add(color)
        return out

    def _goal_alternatives(self, obs, exclude=()):
        """The other defensible goal candidates (unique small non-terrain
        objects), nearest first — held, not refuted, while one is unreachable."""
        objs_w = self._objs_world(obs)
        av_key = self.model.avatar_shape_key
        terrain = self._terrain_colors()
        small = [t for t in objs_w if t[0] != av_key and t[2] <= 60
                 and t[1] not in terrain and t[1] not in self._lethal_colors
                 and t[1] not in exclude and t[1] not in self._goal_refuted]
        counts = Counter(t[1] for t in small)
        uniques = [t for t in small if counts[t[1]] == 1]
        if self.model.avatar_pos is not None:
            ar, ac = self.model.avatar_pos
            uniques.sort(key=lambda t: abs(t[3][0] - ar) + abs(t[3][1] - ac))
        return [(t[3], t[1]) for t in uniques]

    def _goal_candidate(self, obs):
        objs_w = self._objs_world(obs)
        av_key = self.model.avatar_shape_key
        terrain = self._terrain_colors()
        hazard = {t.color for t in self.tracker.autonomous_tracks()}
        small = [t for t in objs_w if t[0] != av_key and t[2] <= 60
                 and t[1] not in terrain and t[1] not in hazard
                 and t[1] not in self._lethal_colors
                 and t[1] not in self._goal_refuted
                 and (self._gate_fails.get(t[1], 0) < 2
                      or self._obs_i - getattr(self, "_gate_step", 0)
                      > 40)]
        if not small:
            # A CONDEMNATION THAT BLANKS EVERY GOAL IS WRONG (0.21
            # lesson class): a mis-blamed lethal color on an unknown
            # game must not blind goal-seek entirely — retry without
            # the lethal exclusion rather than wander forever
            small = [t for t in objs_w if t[0] != av_key and t[2] <= 60
                     and t[1] not in terrain and t[1] not in hazard]
        if not small:
            return None
        if self._collect_colors:
            cands = [t for t in small if t[1] in self._collect_colors]
            if cands and self.model.avatar_pos is not None:
                ar, ac = self.model.avatar_pos
                t = min(cands, key=lambda t2: abs(t2[3][0] - ar)
                        + abs(t2[3][1] - ac))
                return t[3], t[1]
        if self.goal_color_prior is not None:
            for k, color, size, world, o in small:
                if color == self.goal_color_prior:
                    return world, color
        counts = Counter(t[1] for t in small)
        uniques = [t for t in small if counts[t[1]] == 1]
        if uniques:
            # COMMIT UNTIL REFUTED: with several equally defensible goals,
            # re-picking by list order every frame flapped the target
            # (hypothesis ledger, r11l: 0 <-> 3 every few steps) and the
            # avatar zig-zagged between them. Keep the current choice
            # while it still exists; switch only when it is gone or gated.
            prev_c = getattr(self, "_hyp_last_goal", None)
            keep = [u for u in uniques if u[1] == prev_c]
            t = keep[0] if keep else uniques[0]
            if len(uniques) > 1 and t[1] != getattr(
                    self, "_hyp_last_goal", None):
                # multiple genuinely defensible goals existed here
                self._hyp("goal_choice", chosen=t[1],
                          alternatives=[u[1] for u in uniques[1:]])
                self._hyp_last_goal = t[1]
            return t[3], t[1]
        return None
