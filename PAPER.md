---
title: "Learning Game Rules with a Programmatic Agent for ARC-AGI-3"
subtitle: "Planning with a rule was rarely the obstacle; acquiring the rule was"
author: "Tushar Sharma"
date: "ARC Prize 2026 Paper Track · November 2026"
---

**Kaggle submission:** `tusharhsharma/arc-prize-2026-arc-agi-3-starter`, Version 21 (bundle v38, sha256 `b48e6992245f33848baf72c28eb36a57f277073f46bfdd541e08c22d33397171`), hidden-set score 0.39. **Code:** https://github.com/tusharhsharma/arc3_programmatic_approach, commit [COMMIT], MIT-0. No language model, no learned weights, no GPU.

# Abstract

Can an agent solve unfamiliar interactive games by tracking objects, testing actions and planning with the rules it learns? I built a programmatic agent for ARC-AGI-3 to study this question. It combines persistent object tracking, candidate goals read from the first frame, experiments that induce rules over object classes, and a planner that composes those rules; a fallback explorer plays whatever our main system leaves, and the main system learns from every action the explorer takes. It contains no game identifiers, colours, coordinates or per-game thresholds. On the frozen final code it solves 22 of 25 public first levels and 43 of 183 public levels; on 26 withheld simulated games it solves 6 first levels on the first attempt, and on six withheld synthetic games 3 levels. Kaggle's hidden games scored 0.20–0.24 for five submissions and 0.39 for the final one. I test one hypothesis: *for this agent, planning with a rule is rarely the obstacle; acquiring the rule is.* Three pieces of evidence support it. Of 17 unsolved withheld games, 12 failed before any candidate interpretation existed. For five public games with the same label, human replays read action by action showed what was missing — two mechanisms and two relations the agent could not express till then, the fifth undetermined. The agent could not have learned them: it learns which rule of a given form holds, and with what parameters, but it never generated a new form of rule; every form it holds was written by hand. And when each missing mechanism was written as one general rule (flow from a transient trace; a pressure plate; a rewind with a replaying twin), the planner solved the level at or within the official human baseline, and disabling the rule lost that level. Against it: the added rules left every withheld count unchanged, and the hidden-score increase cannot be attributed to them. The contribution is an architecture, an auditable development method, and a sharply localised negative result: a planner over a hand-written vocabulary is bounded by the vocabulary, and the open problem is acquiring mechanisms from one's own observations.

# 1 Introduction

ARC-AGI-3 asks an agent to infer the goals and dynamics of an unfamiliar game by interacting with it, and scores it by action efficiency relative to human players [1]. Humans complete every game; frontier systems scored under one percent at release [1]. The difficulty is not only choosing actions: the agent must decide which objects matter, what an action changes, and what completion would look like. A planner can search only what its representation can express.

I built an agent around a small set of explicit assumptions — objects persist when moved, turned or recoloured; similar shapes may have related roles; contact can cause motion; rows and pairings may encode goals — and called the design an *equipped scientist*. Its equipment is a hand-written vocabulary of objects and relations; its experiments are actions whose effects confirm or refute candidate rules; it commits to a rule until the environment contradicts it, and it plans with the rules it holds.

The motivating hypothesis came from watching people play. A person meeting a tap, a pressure plate or an undo key on screen recognises it at once, from a lifetime of taps, levers and undo keys. Those priors are lived, not stated, and an agent can only be given the ones its builder thinks to state. If that is the main gap, the agent should fail mostly *before* it has an interpretation, not after; and supplying a missing mechanism as one general rule should turn a failure into an efficient solution without touching other tasks. Section 6 tests both predictions and reports the evidence against them.

Contributions: (i) an explicit architecture for learning and planning with game rules, with no per-game code (§4); (ii) an evaluation method that separates development data from withheld diagnostics, records predictions before scores and labels failures by where they occur (§3); (iii) three mechanisms written after studying human replays, each with its algorithm and an ablation (§5); (iv) a test of the acquisition hypothesis, with the evidence for and against (§6–7). The evidence concerns this implementation. It does not establish a ceiling for programmatic agents, and it does not show that lived experience is necessary.

# 2 Related work

**Theory-based model learning in games.** The closest ancestor is EMPA [3], which learns probabilistic generative models of games, written as programs for a game-engine simulator, explores to test them and plans by internal simulation, closely matching human learning efficiency on 90 Atari-style games. The equipped scientist shares the loop — model, experiment, plan — but has no program space to search: its rules are sentences over tracked objects, induced from pixel frames, and the space of mechanisms is whatever the hand-written vocabulary contains. My results locate the limit of that design exactly where EMPA's program space does its work. Object-oriented MDPs [4] and Schema Networks [5] gave earlier object-relational transition models; WorldCoder [6] has a language model write the world model as code from interaction, which is one way to *acquire* mechanisms rather than enumerate them.

**ARC.** The ARC Prize 2025 report describes refinement loops — candidate programs improved by a feedback signal — as the defining approach on the static ARC-AGI-2 [2]; ARC-AGI-3 moves the feedback signal into the environment. In the 2025 ARC-AGI-3 preview competition the winning agent completed 18 levels and the runner-up 13 [7]. On the 2026 leaderboard the leading entries (75.56, 70.24, 58.59 at the close of code submissions) are harnesses around language models; many entries, ours among them, sit below one point.

**What is not in prior work, to my knowledge,** is the use of the foundation's published human replays — every action and frame of a human playthrough — to diagnose an agent's representational failures action by action, and the reporting of an ARC-AGI-3 agent's development under withheld diagnostics and pre-written submission predictions.

# 3 Evaluation method

## 3.1 Development data and withheld diagnostics

The 25 public games were development data. Their frames, failures and selected human replays were inspected, and the agent was changed in response. Public success therefore measures competence on development tasks; a rule written without a game name can still be fitted to a development set.

Two suites served as withheld diagnostics. The first is 27 simulated games produced by my own generator from compositions of mechanics families. The second is six synthetic games, sx07–sx12, authored by a separate agent session, with their answer key sealed and never read. For both, only counts and failure labels were ever printed. They are not samples of the official hidden games, and repeated inspection of their counts makes them development diagnostics rather than an untouched test set. On 24 October one simulated game (shw0) was opened under a rule for studying a failed withheld task; it became development data and is reported separately, and later counts use 26.

## 3.2 Action accounting

The score is Relative Human Action Efficiency: per level, (human baseline ÷ agent actions)², level-weighted and averaged over games, on a 0–100 scale [1]. Two probes against the live API in August established that every action on a level counts, failed attempts and resets included, and that Kaggle's competition harness opens one session per game, so exploration cost is charged. A short successful attempt is therefore not evidence of human-level scoring. Every table below charges a level all actions of all attempts on it, and uses the official per-level baselines shipped with each game; the humans in the replays I read (§5) are individual players and are named as such.

## 3.3 Regression checks and failure labels

Every change ran a fixed battery: 18 public first levels, public climbs (two attempts per level until a level is lost), a 15-level synthetic twist suite built to test specific rules, and the withheld counts. A regression was bisected to the rule that caused it and the rule's condition was narrowed; no rule was given a per-game exception. Each attempt writes a report labelled by where it failed: *no interpretation* (no candidate goal or rule fits), *died* (hazard or budget), *interpretation without plan*, and *plan diverged* (a prediction contradicted by the next frame). These labels locate a failure in the pipeline; they do not by themselves identify its cause, which §6.2 checks against independent evidence.

## 3.4 Submission predictions

Each Kaggle submission was entered in a ledger, with a predicted score range, before its score was known. The ledger is a project document, not an independently timestamped preregistration, and seven ranged predictions are too few for a calibration study; I report them to expose mistaken expectations (Table 4).

# 4 Agent

## 4.1 Objects and candidate goals

Perception groups connected cells into objects and tracks them across movement, rotation, recolouring and partial occlusion, including compound and patterned objects. Two identity rules came late: two same-shape objects exchanging colours under a click on one of them is a selection, not two moves; and an object recoloured whole keeps its identity. Before the first action, an interpretation layer ranks candidate goals from first-frame relations: a shape and a socket, extremities and marks of their colour, a separated row as a sequence, paired objects as a legend, an enclosed region as a container. Candidates guide experiments; none is assumed true because it matches a pattern.

## 4.2 Experiments and rules

```
procedure PLAY_LEVEL(game)
  O ← track(frame);  G ← rank_candidate_goals(O)
  R ← rules carried from earlier levels
  while level not complete and budget remains:
      if a plan P to some g ∈ G exists under R:
          execute P one action at a time
          if a frame contradicts R's prediction: revise R; replan
      else:
          a ← next_experiment(O, R)      # each offered key 2–3×, one click per
                                          # distinct object, a path to its end
          Δ ← diff(track(frame_before), track(frame_after))
          R ← induce(R, a, Δ)             # moved / appeared / refused /
                                          # changed without being controlled
          G ← G ∪ goals_from_uncontrolled_changes(Δ)
  if no interpretation or no plan: hand the level to the fallback explorer
```

Rules describe object classes and action effects: carrying an adjacent object, merging two like objects, a jump whose length depends on the previous action. Conflicting observations are resolved by the coarsest condition that separates them. A refused move with nothing in the way is recorded as a boundary, not as evidence against the movement rule. A progress module records changes in objects the agent does not control, tied to what the controlled object was doing, and generalises them to selection, filling and sequence progress; it solved six of sb26's eight levels on 21 October (five on the final code, Table 1) with no sb26-specific rule and, unchanged, the first level of a withheld synthetic game (sx10).

## 4.3 Planning, fallback and harness

The planner searches joint states of the controllable objects under the induced rules, observed boundaries and hazards. A motion record tracks objects that move on their own (stride, target, sign towards the goal); the planner did not yet use it to reason about helpers and moving obstacles. When the main system has no interpretation or no plan, a fallback explorer — the agent's August version, embedded whole — continues from the current state, and the main system regains the level only from a state it has not tried. Every action the explorer takes is observed by the main system exactly as its own experiments are — what moved under which key, what a click did, what destroyed the controlled object — and when the explorer wins a level, the main system reads the win (an arrival, a collection, a click on an object) and carries it to the next level. The explorer therefore adds experience, but not new kinds of rule: its observations are filed under the forms the vocabulary already has. Harness scheduling mattered: a policy that left games idle for hours scored 0.00, and one that let one game's thinking stall all others produced three execution errors; the accepted schedule caps a game's hold on the shared processor at 45 s and gives each game about 245 s of thinking. The code is about 30k lines of Python (perception, rules and progress ~10k; play loop and protocols ~11k; fallback and harness adapter ~9k); the submission is one 630 KB file.

# 5 Mechanisms written after studying human replays

By 23 October the agent solved 41 public levels and none of five public first levels (g50t, sk48, dc22, sp80, sc25). For those five, and four second levels, I read the foundation's human replays action by action, asking what each action changed — including intermediate animation frames — and what the player evidently knew that the agent did not. A first reading from the opening frame and the key list was wrong for two of the games; only the per-action effects were reliable. Each finding was then written as one general rule, coded without the game, run against every other level by the regression battery, and narrowed until it was clean. The agent did not invent these mechanisms; I supplied them.

## 5.1 Flow from a transient trace (sp80)

A run key releases water from a tap; it falls, spreads along the bar it meets, falls off both ends, and fills the cups it enters; then the board is restored. Containers the water missed blink.

```
READ_RUN(frames):                       # a multi-frame action that draws and erases
  ink    ← modal colour of changed cells that belong to no object
  trace  ← those cells in order of first appearance
  source ← object of the ink's colour adjacent to the first cells
  entered← objects whose interior (cells enclosed on ≥3 sides) the trace reached
  flash  ← objects flipping in the last 8 frames         # the goal kind
  accept only if the trace cells are restored at the end

FLOW(frame, source, dir, breadth):      # the hand-written law
  advance from the source with its breadth over background;
  on meeting an object: spread along its near face both ways while the face is backed;
  where the face is open beyond: fall there with the same breadth; where shut: stop;
  on the board's ground: stop

TRUST(FLOW) iff Jaccard(FLOW, trace) ≥ 0.8 for every observed trace   # hand-chosen
PLAN: place the movable object where FLOW enters every flashing object;
      if not trusted, test placements nearest first, one run each
```

sp80 L1 was solved on the second attempt (10 actions, after a first attempt that ran out of the level's action clock at 30): 40 actions charged, official baseline 39. L2 took 24–25 actions on the first attempt (baseline 58). L3 failed because overlapping bars broke object identity.

## 5.2 A pressure plate and a replaying twin (g50t)

Standing on a pad moves a gate off the track; leaving moves it back. A fifth key returns the controlled ring to its start and, two actions later, a twin appears and replays the ring's recorded path one accepted move per action, so one player can hold the pad while being at the gate.

```
LEVER: entering cells X moved object Y by v, and leaving X moved Y by −v
       → Y is open while X is occupied (X = only the cells actually stood on)
REWIND(key): key returned the body to its start from ≥2 distinct origins,
             never stepped the body, never coincided with screen scrolling
TWIN: object of the body's shape at start + Σ record[:k]; advances once per
      accepted body move; exists for records of ≥2 moves; replaced at each rewind
PROTOCOL: stand on reachable overlaps until a lever appears; rewind once;
          walk the shortest path onto the pad; rewind; walk to the goal planning
          with the gate open, waiting at the gate until the twin reaches the pad
```

Three narrowings came from the regression battery: a key that put the body back after a hazard, an arrow key that returned it from one place only, and a key that scrolled the screen had each been read as a rewind once. g50t L1 was solved on the first attempt in 44–53 actions (official baseline 78; the replayed player used 51).

## 5.3 Two relations not coded

dc22 is a cursor that may walk only on coloured cells, where clicking a picture toggles the like-coloured piece between two places; sk48 is a crane whose arm extends and retracts and pulls what its end touches, with a legend giving the required order. Both were read and written down as rules but not coded before the deadline. sc25's replay was long, with resets; the player was evidently still learning, and I drew no rule from it.

# 6 Results

## 6.1 Public, withheld and hidden outcomes

Table 1. Public games on the frozen final code (3 November; the main system alone, 300 s of thinking and two attempts per level, a game played until a level is lost). Every action of every attempt on a level is charged; level efficiency is (official baseline ÷ charged actions)², capped at 115% as in the official scoring.

| Game | Levels won | Per won level: charged actions / official baseline (level efficiency) |
|----|----|----------------------------------------------|
| ft09 | 6/6 | L1 4/43 (115%); L2 7/12 (115%); L3 14/23 (115%); L4 16/28 (115%); L5 21/65 (115%); L6 13/37 (115%) |
| sb26 | 5/8 | L1 45/18 (16%); L2 52/28 (29%); L3 54/18 (11%); L4 55/19 (12%); L5 62/31 (25%) |
| tu93 | 2/9 | L1 28/19 (46%); L2 44/16 (13%) |
| cd82 | 2/6 | L1 65/55 (72%); L2 69/8 (1%) |
| m0r0 | 2/6 | L1 41/30 (54%); L2 44/111 (115%) |
| vc33 | 2/7 | L1 22/7 (10%); L2 22/18 (67%) |
| ka59 | 2/7 | L1 91/28 (9%); L2 57/109 (115%) |
| ls20 | 2/7 | L1 25/22 (77%); L2 45/123 (115%) |
| s5i5 | 2/8 | L1 50/20 (16%); L2 72/89 (115%) |
| re86 | 1/8 | L1 28/26 (86%) |
| wa30 | 3/9 | L1 54/71 (115%); L2 59/119 (115%); L3 78/183 (115%) |
| lf52 | 1/10 | L1 51/32 (39%) |
| sp80 | 2/6 | L1 40/39 (95%); L2 25/58 (115%) |
| g50t | 1/7 | L1 53/78 (115%) |
| su15 | 3/9 | L1 14/22 (115%); L2 42/42 (100%); L3 43/26 (37%) |
| tn36 | 1/7 | L1 39/32 (67%) |
| cn04 | 1/6 | L1 18/29 (115%) |
| bp35 | 1/9 | L1 28/21 (56%) |
| lp85 | 1/8 | L1 10/17 (115%) |
| r11l | 1/6 | L1 7/22 (115%) |
| ar25 | 1/8 | L1 20/32 (115%) |
| tr87 | 1/6 | L1 32/54 (115%) |
| sc25 | 0/6 | — |
| sk48 | 0/8 | — |
| dc22 | 0/6 | — |

Totals: 22 of 25 first levels and 43 of 183 levels. On the won levels, mean level efficiency is 78%; 22 of the 43 are at or under the official baseline. Development summaries had reported 44 levels: sb26 L6 was won on 21 October and is lost by the final code — a regression the battery did not catch, because sb26 was not in it.

Table 2. Withheld diagnostics. Counts only; the games were never inspected.

| Suite | Final code (3 Nov) | Earlier |
|---|---|---|
| Simulated holdout, 26 games, first level, first attempt | 6/26 | 6/26 (25 Oct, before §5); 3/27 (21 Sep) |
| Simulated holdout, first level within two attempts | not re-measured | 10/26 (24 Oct) |
| Synthetic blind suite, sx07–12, levels won | 3 levels | 3 levels (25 Oct: sx10 L1, sx11 L1–2) |
| Same suites, main system over fallback, first levels | — | 23/27 and 3–4/6 (25 Sep) |

The three §5 mechanisms changed no withheld count.

## 6.2 Testing the acquisition hypothesis

*Prediction 1: failures should occur before an interpretation exists.* In the 23 October audit, 17 of the 27 withheld simulated games were unsolved within two attempts: 12 were labelled no interpretation on both attempts, 3 died, 1 had an interpretation without a plan, 1 a plan that diverged. The four unsolved blind games were all labelled plan diverged. Twelve of seventeen supports the prediction for the simulated suite; the blind suite fails differently — its losses are in the model of objects and rules the agent did form.

*Validating the label.* The labels are produced by the system under study, so I checked them where independent evidence exists. The five public first levels labelled no interpretation were read against human replays action by action. Two (g50t, sp80) hid a mechanism the vocabulary could not express; two (dc22, sk48) hid a relation it could not express; one (sc25) was undetermined. In four of five the label agreed with an independent account of the cause, and in none was the cause search time: raising the per-level thinking allowance from about 75 s to 300 s solved none of the levels lost at 75 s.

*Prediction 2: supplying a missing mechanism as one general rule should solve its level efficiently and change nothing else.* Table 3 disables each rule.

Table 3. Ablations on the final code (single runs, two attempts per level).

| Disabled | sp80 L1 (baseline 39) | sp80 L2 (baseline 58) | g50t L1 (baseline 78) |
|---|---|---|---|
| nothing | won: 30 lost + 10 = 40 charged | won: 24–25 | won: 44–53 (53 in Table 1) |
| flow from trace | lost (30, 11) | not reached | — |
| lever and twin | — | — | lost, no plan (40, 22) |
| colour-exchange identity rule | won: 30 lost + 10 | lost (45, 45) | — |

"—": not run. With all three enabled, every other level of the regression battery is identical to the snapshot taken before they were added. Each rule is necessary for its own levels, and with it the agent is at (sp80 L1) or well within (sp80 L2, g50t L1) the official baseline. Prediction 2 holds on the public games; whether each rule is harmless elsewhere when *disabled* was not tested.

*Against the hypothesis.* The withheld counts did not move (Table 2), so these rules found nothing to apply to in 32 withheld games. The hidden score rose (Table 4), but the bundle carried five weeks of other changes, so the rise is not evidence for these rules. And the label check covers five games, all public.

## 6.3 Hidden-set submissions

Table 4. Kaggle submissions with a recorded prediction, plus repeats of identical code.

| Version | Recorded prediction | Score | Reading |
|---|---|---|---|
| V3 (Aug) | — | 0.30 | the same code later scored 0.21 |
| V11 | 0.22–0.28 | 0.23 | inside |
| V12 | 0.21–0.36 | 0.24 | inside |
| V14 | 0.20–1.5 | 0.24 | inside only at the low edge; the main system never ran on the hidden games |
| V15 | 0.5–3.5 | 0.00 | outside; scheduling failure |
| V19 | 0.3–3.5 (if it runs) | 0.20 | ran; range missed |
| V19 repeat (Nov) | — | 0.22 | identical code |
| V21 (final) | 0.18–0.40, point 0.24 | 0.39 | inside, at the upper end |

Five submissions from V7 to V19 scored 0.20–0.24. Identical code has been scored twice on two occasions, 0.20 then 0.22 and 0.30 then 0.21; these pairs show that swings of 0.02–0.09 occur but do not estimate a noise distribution. V21's 0.39 is 0.17 above the V19 repeat, about twice the largest swing seen, and was scored once: probably a real change, not an established one. V21 differs from V19 by about 5,750 added and 4,030 removed lines — the progress module, the motion record, conditioned outcomes, drift, hazards, the §5 mechanisms, the identity rules, and a repair to the bundle builder without which the newer core could not have loaded — so the score measures the combined submission and attributes nothing. In score terms 0.17 on a 0–100 scale averaged over about 110 games is roughly one more efficiently solved level.

# 7 Discussion

**What the evidence supports.** Within this agent, the planner was rarely what failed. When a mechanism could be expressed, the agent solved the level within the official baseline (Table 3; also ft09's six levels in 4, 7, 14, 16, 21 and 13 actions against baselines of 43, 12, 23, 28, 65 and 37). Where it failed on withheld games, it mostly failed before it had an interpretation, and where I could check that label independently it was right four times in five. A planner over a hand-written vocabulary is bounded by the vocabulary.

**The limitation, stated plainly.** I set out to build an agent that derives what it needs from universal inputs — the frames and the effects of its own actions — as a person seems to. It did so only in part, and the boundary is sharp.

| | What it covers |
|------------|----------------------------------------|
| **Given by hand** | The *forms* of rule the agent can hold: an object persists; a key moves a class of object by a vector; contact carries, pushes or merges; a click selects or fills; a refusal marks a boundary; an outcome may depend on the previous action; a row is a sequence, a pairing a legend; a press that draws and erases has told something. |
| **Derived by the agent** (no code change) | Which of those forms holds in a level, and its parameters: which key moves what and how far, under which condition, within which bounds, what destroys the controlled object, and what the goal is when an uncontrolled change shows it — enough for five levels of a public game with no rule written for it (sb26) and the first level of a withheld one (sx10). |
| **Never derived** | A new form of rule. |

A person meeting g50t's pressure plate or sp80's tap already holds the form — the state of one thing depends on where another stands; a liquid runs down, along and off what it meets and fills what it enters — from lived experience, and only has to bind it to this screen. The agent observed the same events, the gate sliding and the water running, and, lacking the form, filed them under forms it had: an object moving on its own; a press with no lasting effect. More observation did not help. The main system watched every action of the fallback explorer, which on the hidden games played up to thousands of actions per game, and generated no new form from them. Each of the three forms in §5 was added by me after reading a human replay. What humans know that this machine does not is therefore not a list of facts about particular games but a stock of rule forms; the agent can be taught a form and can then bind it on its own, but it has no mechanism for generating one. That, not search time or perception, is the limitation this work documents, and closing it requires a space of possible forms to search — the program space of EMPA [3], code written by a language model as in WorldCoder [6], or a grammar of effects learned from human replays.

**What it does not support.** That the vocabulary was the cause of the hidden-set score, which an aggregate cannot show; that rules supplied this way transfer, since none moved a withheld count; or that lived experience is necessary, which nothing here tests. Most of the development time went to implementation faults found by the regression battery — identity swaps between like objects, misread ground colours, the harness schedule — and those are not evidence either way.

**What I would do first, again.** The hidden score could never exceed the public one: 43 of 183 levels is a low ceiling on a board whose leaders score 75. Completing the public set should have preceded any submission strategy; a withheld gauge is still needed, because in September the agent solved 19 of 25 public first levels and 3 of 27 withheld ones.

**Next.** The hypothesis is falsifiable as follows. Generate new games whose solutions require a mechanism absent from the vocabulary (a conditioned object state, a transient trace, a replayed record), withheld from whoever writes the agent. An agent that *acquires* mechanisms should, without code changes, (a) predict the next frame after the mechanism's first appearance better than chance, and (b) solve the level within a fixed multiple of the baseline; an agent like this one should do neither. Three routes to acquisition are open: a reader of multi-frame effects general enough to describe a trace, a slide, a growth or a toggle without being told which (§5.1's reader is one narrow instance); a learned reader of action effects trained on the foundation's human replays, which label "what each action changed" for every public game; and a language-model hybrid that proposes mechanisms as code, as WorldCoder does [6], with this agent's tracker, rule language and planner as its tools.

# 8 Conclusion

A programmatic agent that tracks objects, experiments and plans with induced rules solved 43 of 183 public levels without per-game code and moved Kaggle's hidden score from a long-standing 0.20–0.24 to 0.39 in its final, unattributable submission. Its failures on withheld games were mostly failures to form an interpretation; when the missing mechanism was supplied as one general rule, the planner solved the level at or within the human baseline and the rest of the regression battery was unchanged. The obstacle for this design is acquiring rules, not planning with them, and the measurement method here — withheld counts, pre-written predictions, ablations and charged actions — is how a claim to have removed it should be tested.

# Acknowledgements

The agent and this paper were developed with extensive assistance from Anthropic's Claude models in an agentic coding environment. Design decisions, evaluation rules and the final text are the author's responsibility.

# References

[1] ARC Prize Foundation. ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence. arXiv:2603.24621, v2, 2026.

[2] F. Chollet, M. Knoop, G. Kamradt, B. Landers. ARC Prize 2025: Technical Report. arXiv:2601.10904, 2026.

[3] P. A. Tsividis, J. Loula, J. Burga, N. Foss, A. Campero, T. Pouncy, S. J. Gershman, J. B. Tenenbaum. Human-Level Reinforcement Learning through Theory-Based Modeling, Exploration, and Planning. arXiv:2107.12544, 2021.

[4] C. Diuk, A. Cohen, M. L. Littman. An Object-Oriented Representation for Efficient Reinforcement Learning. ICML 2008.

[5] K. Kansky et al. Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics. ICML 2017.

[6] H. Tang, D. Key, K. Ellis. WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment. NeurIPS 2024.

[7] ARC Prize Foundation. ARC-AGI-3 Preview Agent Competition results, 2025. arcprize.org/competitions/arc-agi-3-preview-agents.

# Appendix A. Reproduction

Repository https://github.com/tusharhsharma/arc3_programmatic_approach, commit [COMMIT]; `submission/my_agent_core_v38.py` is the submitted bundle (sha256 above); `tools/build_core_submission.py` rebuilds it; `tools/harness_run.py` runs it through the official ARC-AGI-3-Agents harness; `tools/core_play2.py <game>:<l1>-<l2> --no-floor --seconds 300` reproduces Table 1 (300 s of thinking per level, two attempts per level, the main system alone); `tools/referee_chain.sh` is the regression battery; the switches `NO_RUNKEY`, `NO_LEVER`, `NO_SWAP_RULE` reproduce Table 3. Dependencies: arc-agi 0.9.9, arcengine 0.9.3, numpy 2.5.4. Hand-chosen constants are listed in the repository record with the level that set each. `SUBMISSIONS.md` is the ledger. The withheld suites are not released.

Human replays used (arcprize.org session ids): g50t c71254fe-00b6-4ee2-b68a-941263eab37f; sp80 ee7752b0-a9ad-4c23-8de4-173307367f31; dc22 1b9a2041-bc99-4b7b-8d99-041f0503956f; sk48 b62df293-1d00-4689-83ef-75fa091b59d3; sc25 b0b236ff-bd06-4f16-a350-5f1ca9f2446d; tn36 90f13b17-9916-4d8a-81ad-dabf059bd943; lf52 d55d768c-0ead-4e34-9c5e-288449092557; bp35 092c5051-3cda-4a35-bf7f-1323253b797d; lp85 543d2fc4-84f4-497d-a87a-47bd578d8410.
