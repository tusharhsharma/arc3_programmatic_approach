# Pilot: can the agent discover the variable its description is missing? (7 Nov 2026)

A separate prototype; the submitted agent is untouched. No language model, no learned weights.

**World.** A player on a grid, one pushable box, a pad, a gate on the only way to the target. The
gate's rule depends on history the frame does not show. Perception is given (object positions and
the gate's displayed state): the pilot tests rule acquisition and planning only.

**Learner.** (1) Keeps every observation of the gate with the history it occurred in. (2) Hypotheses
are small executable expressions over grounded primitives — on pad, number of entries, turns since
the last entry or exit, turn parity — combined by NOT/AND/OR (~1,300 expressions), simplest first.
(3) Active version: plans under the simplest surviving rule; if rules holding most of the prior
weight say the plan fails, it first takes the cheapest action sequence on which the simplest rule
and the heaviest dissenter predict different gate states. Every action is charged. Passive
version: plans under the simplest survivor, learning only from what its plans happen to show.
(4) The planner's state carries the memory the surviving rule uses. If every hypothesis is
refuted, the learner reports that its description is missing a variable.

**Frozen baseline.** The submitted agent's lever form only (open while the pad is occupied) plus
its fallback (go when open, else explore at random), 10 seeds.

**Protocol and its honesty.** v1 was run once on 3 development and 5 further games
(`results_v1` in run.py); its active policy experimented whenever rules disagreed and lost to the
passive one. It was revised once (prior-weighted policy, a fallback when stuck), frozen
(learner_v2_frozen.py, sha256 prefix 24cf019a2891cbc7), and only then were the four "held out"
games and the corridor map written. The same person wrote the learner and every game, so nothing
here is independently held out. Budget 120 actions. Efficiency = (oracle shortest ÷ actions)²,
capped at 1.15, where the oracle knows the rule.

## Results (v2)
| map | set | game | oracle | frozen: won/10, mean eff | passive: actions (eff) | active: actions (eff), experiments | active's rule | memory carried | missing variable |
|---|---|---|---|---|---|---|---|---|---|
| small | dev | occupancy | 14 | 7/10, 0.17 | won 14 (1.00) | won 17 (0.68), 4 | onpad | onpad | no |
| small | dev | toggle | 8 | 7/10, 0.10 | won 14 (0.33) | won 8 (1.00), 2 | entries odd | entries | no |
| small | dev | timer4 | 8 | 7/10, 0.10 | won 14 (0.33) | won 8 (1.00), 2 | since_entry<4 | since entry | no |
| small | seen after v1 (not held out) | second_entry | 10 | 5/10, 0.09 | won 16 (0.39) | won 10 (1.00), 3 | entries>=2 | entries | no |
| small | seen after v1 (not held out) | delay3 | 8 | 7/10, 0.10 | won 14 (0.33) | won 8 (1.00), 2 | since_entry<4 | since entry | no |
| small | seen after v1 (not held out) | latch | 8 | 7/10, 0.10 | won 14 (0.33) | won 8 (1.00), 2 | entries odd | entries | no |
| small | seen after v1 (not held out) | clock | 6 | 10/10, 0.17 | won 6 (1.00) | won 6 (1.00), 0 | t even | t mod 2 | no |
| small | seen after v1 (not held out) | last_move_L | 8 | 10/10, 0.08 | LOST 12 (0.00) | LOST 12 (0.00), 3 | None |  | yes |
| small | held out (after v2 freeze) | inverse | 6 | 10/10, 1.00 | won 6 (1.00) | won 6 (1.00), 0 | not onpad | onpad | no |
| small | held out (after v2 freeze) | brief_second | 10 | 5/10, 0.09 | won 16 (0.39) | won 10 (1.00), 3 | entries>=2 | entries | no |
| small | held out (after v2 freeze) | box_only | 14 | 0/10, 0.00 | won 14 (1.00) | LOST 120 (0.00), 3 | onpad and since_exit<1 | onpad, since exit | no |
| small | held out (after v2 freeze) | exits_odd | 8 | 7/10, 0.08 | won 14 (0.33) | won 8 (1.00), 2 | since_exit<4 | since exit | no |
| corridor | held out map | occupancy | 28 | 3/10, 0.07 | won 32 (0.77) | won 49 (0.33), 12 | onpad | onpad | no |
| corridor | held out map | toggle | 24 | 3/10, 0.07 | won 24 (1.00) | won 38 (0.40), 14 | entries odd | entries | no |
| corridor | held out map | timer14 | 24 | 3/10, 0.07 | won 24 (1.00) | won 42 (0.33), 18 | since_entry<14 | since entry | no |
| corridor | held out map | latch | 24 | 3/10, 0.07 | won 24 (1.00) | won 40 (0.36), 18 | entries>=1 | entries | no |
| corridor | held out map | inverse | 2 | 10/10, 1.00 | won 2 (1.00) | won 2 (1.00), 0 | not onpad | onpad | no |
| corridor | held out map | brief_second | unsolvable | | | | | | |
| corridor | held out map | box_only | 28 | 0/10, 0.00 | LOST 21 (0.00) | LOST 120 (0.00), 7 | onpad and since_exit<1 | onpad, since exit | no |

## What it shows
- **A missing variable can be found from paid actions and carried into planning.** On the small
  map, of the ten games other than the two with no usable primitive, the active learner matched
  the oracle's action count (efficiency 1.00) in nine (0.68 on the tenth), with 0–4 deliberate
  experiments, and its planner carried the variable the rule needed (turns since entry for the
  timer, the entry count for the toggle). On the same ten games the passive learner averaged 0.54
  and the frozen agent 0.20.
- **Whether an experiment pays depends on its cost, and v2 ignores cost.** On the corridor map,
  where reaching a disagreement point costs a long walk, the active learner spent 12–18 actions on
  experiments and fell to 0.33–0.40, while the passive learner reached 0.77–1.00.
- **Solving is not identifying.** In 4 of those ten games the active learner's final
  rule was not the true rule but agreed with it on the path it took (e.g. "since entry < 4" for a
  gate that stays open 3 turns after leaving). A reviewer must see a prediction before a
  deliberately disagreeing action to tell the two apart.
- **Detecting a missing variable works only when every candidate is refuted.** For "the gate opens
  on the last move being left" every candidate was refuted and the learner reported a missing
  variable. For "only the box on the pad counts" a spurious conjunction survived, the learner
  pursued it and lost both maps — the misleading-correlation case the proposal anticipates.

## v3: a cost-aware decision rule (rules, perception and planner unchanged)

Proposed by Tushar on 7 Nov. Before each probe the learner estimates, over its leading rules,
the remaining actions if it (A) continues toward the goal and replans where a prediction fails,
or (B) takes the probe and then the best plan for its outcome, counting travel from where each
leaves it; if the leading rules' plans agree on the next move it takes that move and learns along
the way. v3 was run once on the two existing maps and revised once (rules with no route from the
current position do not vote; a probe must save at least one expected action). v3.1 was frozen
(learner_v31_frozen.py, sha256 prefix 21e94da79ba61853) and only then were three new layouts
(middle, pocket, rooms) and two new timer lengths written.

### Existing maps (development for v3)
| map | game | oracle | frozen agent: won/10, mean eff | passive | v2 active (probes) | v3 cost-aware (probes) |
|---|---|---|---|---|---|---|
| small | occupancy | 14 | 7/10, 0.17 | won 14 (1.00), 0 | won 17 (0.68), 4 | won 14 (1.00), 2 |
| small | toggle | 8 | 7/10, 0.10 | won 14 (0.33), 0 | won 8 (1.00), 2 | won 8 (1.00), 2 |
| small | timer4 | 8 | 7/10, 0.10 | won 14 (0.33), 0 | won 8 (1.00), 2 | won 8 (1.00), 2 |
| small | second_entry | 10 | 5/10, 0.09 | won 16 (0.39), 0 | won 10 (1.00), 3 | won 10 (1.00), 2 |
| small | delay3 | 8 | 7/10, 0.10 | won 14 (0.33), 0 | won 8 (1.00), 2 | won 8 (1.00), 2 |
| small | latch | 8 | 7/10, 0.10 | won 14 (0.33), 0 | won 8 (1.00), 2 | won 8 (1.00), 2 |
| small | clock | 6 | 10/10, 0.17 | won 6 (1.00), 0 | won 6 (1.00), 0 | won 6 (1.00), 0 |
| small | inverse | 6 | 10/10, 1.00 | won 6 (1.00), 0 | won 6 (1.00), 0 | won 6 (1.00), 0 |
| small | brief_second | 10 | 5/10, 0.09 | won 16 (0.39), 0 | won 10 (1.00), 3 | won 10 (1.00), 2 |
| small | exits_odd | 8 | 7/10, 0.08 | won 14 (0.33), 0 | won 8 (1.00), 2 | won 8 (1.00), 2 |
| small | last_move_L | 8 | 10/10, 0.08 | LOST 12 (0.00), 0 | LOST 12 (0.00), 3 | LOST 12 (0.00), 2 |
| small | box_only | 14 | 0/10, 0.00 | won 14 (1.00), 0 | LOST 120 (0.00), 3 | LOST 120 (0.00), 2 |
| corridor | occupancy | 28 | 3/10, 0.07 | won 32 (0.77), 0 | won 49 (0.33), 12 | won 32 (0.77), 0 |
| corridor | toggle | 24 | 3/10, 0.07 | won 24 (1.00), 0 | won 38 (0.40), 14 | won 24 (1.00), 0 |
| corridor | timer14 | 24 | 3/10, 0.07 | won 24 (1.00), 0 | won 42 (0.33), 18 | won 24 (1.00), 0 |
| corridor | latch | 24 | 3/10, 0.07 | won 24 (1.00), 0 | won 40 (0.36), 18 | won 24 (1.00), 0 |
| corridor | inverse | 2 | 10/10, 1.00 | won 2 (1.00), 0 | won 2 (1.00), 0 | won 2 (1.00), 0 |
| corridor | box_only | 28 | 0/10, 0.00 | LOST 21 (0.00), 0 | LOST 120 (0.00), 7 | LOST 21 (0.00), 0 |

### New layouts, written after v3.1 was frozen (one run)
| map | game | oracle | frozen agent: won/10, mean eff | passive | v2 active (probes) | v3 cost-aware (probes) |
|---|---|---|---|---|---|---|
| middle | occupancy | 23 | 6/10, 0.18 | won 25 (0.85), 0 | won 28 (0.67), 4 | won 25 (0.85), 0 |
| middle | toggle | 17 | 6/10, 0.14 | won 17 (1.00), 0 | won 17 (1.00), 1 | won 17 (1.00), 0 |
| middle | latch | 17 | 6/10, 0.14 | won 17 (1.00), 0 | won 17 (1.00), 1 | won 17 (1.00), 0 |
| middle | second_entry | 19 | 4/10, 0.06 | won 19 (1.00), 0 | won 19 (1.00), 4 | won 19 (1.00), 0 |
| middle | delay3 | 23 | 3/10, 0.05 | won 23 (1.00), 0 | won 27 (0.73), 6 | won 23 (1.00), 0 |
| middle | timer6 | 17 | 1/10, 0.01 | won 17 (1.00), 0 | won 17 (1.00), 1 | won 17 (1.00), 0 |
| middle | box_only | 23 | 2/10, 0.01 | LOST 17 (0.00), 0 | LOST 120 (0.00), 4 | LOST 17 (0.00), 0 |
| pocket | occupancy | 26 | 7/10, 0.60 | won 26 (1.00), 0 | won 38 (0.47), 13 | won 26 (1.00), 0 |
| pocket | toggle | 26 | 7/10, 0.60 | won 26 (1.00), 0 | won 38 (0.47), 13 | won 26 (1.00), 0 |
| pocket | latch | 26 | 7/10, 0.60 | won 26 (1.00), 0 | won 38 (0.47), 13 | won 26 (1.00), 0 |
| pocket | delay3 | 26 | 7/10, 0.60 | won 26 (1.00), 0 | won 38 (0.47), 13 | won 26 (1.00), 0 |
| pocket | timer13 | 26 | 2/10, 0.03 | won 26 (1.00), 0 | LOST 120 (0.00), 13 | won 26 (1.00), 0 |
| pocket | box_only | 26 | 7/10, 0.60 | won 26 (1.00), 0 | won 38 (0.47), 13 | won 26 (1.00), 0 |
| rooms | occupancy | 12 | 6/10, 0.18 | won 12 (1.00), 0 | won 14 (0.73), 4 | won 14 (0.73), 1 |
| rooms | toggle | 10 | 10/10, 0.24 | won 12 (0.69), 0 | won 10 (1.00), 2 | won 10 (1.00), 1 |
| rooms | latch | 10 | 10/10, 0.25 | won 12 (0.69), 0 | won 10 (1.00), 2 | won 10 (1.00), 1 |
| rooms | second_entry | 12 | 10/10, 0.23 | won 16 (0.56), 0 | won 12 (1.00), 5 | won 12 (1.00), 1 |
| rooms | delay3 | 10 | 9/10, 0.06 | won 12 (0.69), 0 | won 10 (1.00), 2 | won 10 (1.00), 1 |
| rooms | timer4 | 10 | 9/10, 0.06 | won 12 (0.69), 0 | won 10 (1.00), 2 | won 10 (1.00), 1 |
| rooms | box_only | 12 | 1/10, 0.01 | won 12 (1.00), 0 | LOST 120 (0.00), 6 | LOST 13 (0.00), 1 |

### Judged against the stated criterion
- **Corridor recovered, small-map wins kept** (development maps): corridor 0.33–0.40 (v2) →
  0.77–1.00 (v3.1, no probes taken); small map unchanged, occupancy 0.68 → 1.00.
- **New layouts** (17 solvable games whose rule the primitives can express): v3.1 matched the
  rule-knowing agent in 15 and averaged 0.98; passive 0.89; v2 0.77 (and lost one game); the
  submitted agent's pressure-plate rule 0.24. v3.1 took probes only where they were cheap
  (rooms map: 0.69 → 1.00 on five games) and declined them where travel was long (pocket,
  middle). It was worse than passive on one game (rooms occupancy, one extra probe).
- **Unchanged limits:** "only the box counts" (no primitive) was still lost on two of three new
  layouts; same author for learner and games; perception given; the candidate space is the same
  fixed one, so this says nothing about scaling the rule space.
