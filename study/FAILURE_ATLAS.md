# FAILURE ATLAS — what winning looks like, and where the loop broke (2026-09-05)

Method. Ground truth for all 24 real games was extracted from the engine
sources by four independent agents (study/truth/<game>.md: every claim
cited to file:line; every L1 winning sequence RUN and verified). The
current Branch-A agent (s23c) was then run on every game with full
belief telemetry (tools/failure_atlas.py -> study/runs/<game>.json:
final beliefs, hypothesis ledger, anomalies, archetype spend, action
tail). Each failure below is placed at the stage of the loop where the
evidence says it happened — not where generic reasoning would put it.
The 8 real-game HOLDOUT games (cd82 g50t sc25 su15 tn36 tr87 vc33 wa30)
were unfrozen for this study on Tushar's instruction; they are the
closest thing we have to the hidden set: games no part of the agent was
ever tuned on.

Stages: P = perception/identity, C = control semantics, R = relation
never coined (mechanic), G = goal inference, X = planning/execution,
B = budget/exploration burn, A = archetype hijack/gating.

## Headline numbers

| set | games | levels won | human L1 baselines |
|---|---|---|---|
| DEV (tuned on for 3 weeks) | 16 | 26 of 118 | — |
| HOLDOUT (never seen) | 8 | **0 of 57** | 7, 22, 32, 36, 54, 55, 71, 78 |

The holdout row is the honest proxy for Kaggle: on eight first-time
games the generic core plus every archetype wins nothing, while the
human first-time cost of those L1s ranges from 7 actions (vc33: click
one button three times) to 78. That is the whole leaderboard gap in one
row.

## Per game: winning, believing, failing

| game | what winning is (engine truth) | what the agent believed / did | stage | why — the critical reading |
|---|---|---|---|---|
| **ar25** L1 26/32, then 0 | The yellow REFLECTION of the grey shape across the blue axis must cover the blue dots; L2+ the axis itself is movable and ACTION5 cycles selection. | Chose the yellow image (colour 4) as the goal; 25 "move" anomalies because identity flipped between the shape and its mirror image (same size, opposite x-motion); multibody archetype overspent and was silenced. | P, G | The win is a relation between a DERIVED entity (the image) and static markers. The agent's goal vocabulary is "walk onto the unique object"; it walked the shape at the dots, and L1 fell only because the dots' handedness happened to line up. The mirrored co-mover is exactly what the co-motion ledger records — but with a mirrored vector, which it does not admit. |
| **bp35** 0/9, 45 deaths | Reach the gem (off-screen at start) in an inverted-gravity scroller: step sideways, fall UPWARD until blocked; click coloured blocks to delete them; a hunter bar creeps every other step. | Avatar = a 1296-pixel region (the wall/background), motions garbage, three "collectibles". | P, R | Identity failed before anything else could: the camera scrolls and the body falls after each step, so "the largest controlled mover" is the scenery. Gravity is a conditioned outcome ("after a step, fall until blocked") the outcome-law machinery could name if identity held. The goal is invisible on frame 1 — only exploration reveals it; the budget bar and hunter punish dithering. |
| **cd82** 0/6 (holdout, human 55) | Paint the 10x10 canvas to match the picture beside it: arrows jump the basket among 8 ring slots, ACTION5 pours the basket's colour onto the half/triangle nearest the basket, palette swatches recolour the basket. | ACTION5 pressed 635 times, RESET 24; ACTION1-4 marked unresolved after one probe each (their effect is a slot-jump, not a step); nothing coined. | C, G | The template/canvas prior EXISTS (scene_analysis: block-match tasks) but is welded to click-painting; here painting is action-driven and region-valued. A relation "ACTION5 paints region R(basket slot) with colour C(basket)" is one observation away and was never formed because a slot-jump is not a translation the motion model can hold. The goal (canvas == target) is the plainest free information on the screen. |
| **cn04** L1 122/29, then 0 | Every connector pixel of every piece paired with a same-KIND connector (kinds 8/13 render identically); click selects a piece; ACTION5 rotates it. | Docking archetype: L1 by compiled machinery, late and fragile (99-168 across trajectories); L2 multi-piece never solved. | A, R, X | Compiled, not induced. The core never coined "stubs pair"; the archetype did, and it cannot assign pairs across four pieces or undo a polarity mistake. |
| **dc22** 0/6 | Reach the goal pad; two panel buttons toggle colour-matched tile groups between solid and walkable; toggling under yourself costs 20 steps. | Motions correct; goal candidates 11, 8, 13 all REFUTED as "unreachable, map exhausted"; playfield crop [10,54,0,34] cut off the button panel; layout-switch supervisor never fired; dithered at the wall to the budget. | P, G, R | Two of this week's rules misfired together: the playfield prior removed the controls from the world, and the new refutation rule read "unreachable" as "not the goal" when it meant "a mechanic is missing". The colour correspondence button<->tile group is free information a human uses at once. |
| **ft09** 4/6 | Each card's black/red cells dictate equal/differ for its 8 neighbours; L5+ cross tiles toggle neighbours too. | Glyph archetype: 4 levels near-optimally; L5 fails at the neighbour-toggling tile. | A, R | Compiled. The rule was hand-read from the legend panels, not induced; the L5 extension (a tile whose click has a 5-cell effect) is one relation the archetype cannot grow. |
| **g50t** 0/7 (holdout, human 78) | Reach the goal; a door stays open only while a body holds the pad; ACTION5 rewinds and leaves a ghost that replays your moves — the ghost must hold the pad. | Motions correct; ACTION5 read as a move (-6 columns: the rewind teleport); goal (colour 1) refuted three times as unreachable; 12 RESETs. | R, G, X | The pad->door relation is observable in two steps (stand on pad: door slides; step off: closes) and was never recorded — no event-on-position relation exists in the ledger. Then the true goal was refuted for being unreachable. Even with both, the plan needs a time-shifted second body; hard for humans too (78). |
| **ka59** 0/7 | Each 3x3 block inside a 5x5 socket; moving into the other block SHOOTS it 15px (over the pit); click a block to control it. | Avatar = the 1-pixel black selection centre (size 1, colour 0); a click law "click(5) teleports the avatar" coined — true of the selection dot, useless for the body; carrier archetype dead; goal refuted. | P, R, G | The selection-livery lesson from cn04 recurs: "the body I control" is a concept the model lacks, so identity binds to the marker that moves when control moves. The push is a co-motion with a DIFFERENT magnitude (15 vs 3) — outside the ledger's "same vector" test. The goal (frames whose 3x3 interior fits the 3x3 blocks) is a size-fit relation, free on frame 1. |
| **lf52** L1 25/32, then 0 | Peg solitaire; L2+ a rail tile carries pegs by arrows; posts; colours; blue obstacles. | Pegs archetype clears L1; L2's carrier tile is outside it. | A, R | Compiled. |
| **lp85** 3/8 | Ring rotation buttons; goal tile into socket; L4 16 buttons at 57x57. | Permuter archetype, 3 levels; L4 efficiency. | A | Compiled; the strongest archetype, and the one whose hidden contribution was measured at zero. |
| **ls20** L1 27/22, then 0 | Reach the pad carrying (shape, colour, rotation) == pad's; tiles cycle one attribute per entry; L3+ pusher walls. | L1 by exploration; L2: 48 anomalies, goal colours 0 and 8 refuted as unreachable; carrier archetype gated. | G, R | The win is EQUALITY between the HUD glyph and the pad glyph (same-as(shape,rotation) — a grammar relation never instantiated as a goal). "Entering tile T rotates the HUD glyph" is an event-on-position relation the ledger cannot hold. Refutation-by-unreachability struck the true pad. |
| **m0r0** L1 50/30, then 0 | Two mirror-controlled dots must coincide; walls desynchronise them; L2 hazards reset; L3 click-a-block mode. | Multibody merges L1; in L2 the agent CLICKED 749 of 800 actions (clicks return to avatar mode = no-op). | A, B, G | No visible goal object (the goal is the dots' relation to each other), so goal-seek yields nothing and the fallback click policy runs to the budget with zero evidence that clicks do anything — the policy has no give-up. |
| **r11l** L1 48/22, then 0 | The disc sits at the centroid of its handles; click empty space to move the selected handle there; click a handle to select it; disc must overlap the same-colour diamond. | Two click laws coined ("click(3/15) teleports the avatar" — true: the selected handle moves to the click); lethal colours 0 and 3 CONDEMNED (the 60-click budget deaths blamed on nearby objects); L2's three handles/decoys unsolved. | G, X, B | The centroid relation (disc = mean of handles) is the mechanic; it is a binary relation on positions the grammar can express but no search ever proposes. Budget deaths again poisoned innocents. |
| **re86** 0/8 | Two crosses; ACTION5 switches the active one (black centre); each colour's dots must be covered by its colour's bars. | ACTION5 read as a (-33,-21) move; identity flips between the crosses; "pushable(11)" coined from a false co-motion; overlay archetype dead; ended pressing UP at the top edge for 100+ actions. | C, P, G | Control transfer (which body obeys me changes) is unmodelled, so every switch is an anomaly. The goal — plus-shaped dot constellations in the crosses' colours — is a covers(bar, dots) relation with colour matching: free information, no hypothesis. |
| **s5i5** L1 41/20, then 0 | Slider clicks grow/shrink same-colour arms; tips must sit on diamonds; L2+ chains. | Actuator archetype L1; chains unsolved. | A, R | Compiled experiment-planner; chain propagation is one relation beyond it. |
| **sb26** L1 9/18, then 0 | The frame is a PROGRAM: RUN reads slots left to right and must emit the target colours in order; rings are subroutine calls. | Reference-assembly archetype L1; rings unsolved. | A, R | Compiled; "a ring is a call into the frame of its colour" is a compositional rule nothing induces. |
| **sc25** 0/6 (holdout, human 36) | Click grid cells to draw the black-cell recipe (diamond) -> SIZE cast shrinks you; walk small through the 2px neck; touch the door. | ACTION1/2 (facing turns) read as ROTATIONS -> docking archetype hijacked 171 actions and died; door colours 9,10 condemned LETHAL (50-step budget deaths blamed on the door); grid never drawn. | C, A, B, G | Three failures compound: a facing change is not a dihedral op; a step budget is not a killer; the recipe (black cells == book glyph) is explicit free information with no reader. |
| **sk48** 0/8 | The cargo sequence on the track (head outward) must equal the reference row; extend/retract the track, rails move the head. | Identity on the head fragment; motions inconsistent; goal chosen 8 among five; 716 ACTION3 presses. | P, R, G | A body that GROWS is outside the translation model; capture-by-extension and sequence-equals-reference are two relations, both absent. |
| **sp80** L1 11/39, then 0 | Bar placement + pour; L2+ several bars (click selects), display rotated 180 with remapped arrows. | Aligner archetype L1; L2 unsolved. | A, R | Compiled geometry solver; selection and rotation break it. |
| **su15** 0/9 (holdout, human 22) | Click within 8px of the blob to PULL it toward the click (up to 16px); pull it into the disc along the dotted trail. | Goal (the disc) chosen then REFUTED at step 20 as unreachable; ACTION7 (undo) pressed 795 times; 5 clicks, all on object centroids (a click ON the blob moves nothing). | C, G, B | The only control is a parametric click->delta(avatar) ("toward the click point"); the click-law vocabulary holds one relation (teleport TO the clicked object). The trail and the HUD icon are explicit hints. The undo spam is the probe machinery refunding an action that visibly "does something". |
| **tn36** 0/7 (holdout, human 32) | Toggle bars encode a 5-step program; RUN executes it; the square must stop inside the ghost outline. | 788 clicks; glyph archetype spent 2; nothing learned. | R, G | Deferred effect: clicks change STATE, RUN produces a TRAJECTORY determined by that state. The agent has no "commit evaluates a configuration" relation (sb26's is hard-wired to its game). The ghost outline is the goal in the plainest form and nothing binds "end pose == outline". |
| **tr87** 0/6 (holdout, human 54) | Bottom row must be the rule-table translation of the top row; left/right move a cursor, up/down cycle the glyph. | No avatar, all four actions unresolved (5 probes each), random pressing to the budget. | C, G | Pure symbolic mapping (ARC-style): a table of pairs IS the rule; slots cycle through 7 values. The core has no symbol/sequence grammar at all — not a weak organ, a missing one. |
| **tu93** L1 33/19, then 0 | Reach the exit within the step budget; L2+ chargers lunge when you stand in front of their eye, patrollers shuttle, followers copy you. | Edge-maze archetype L1; 105 deaths on L2 from the charger. | A, R, X | Enemy behaviour relations (conditional on YOUR position) are never recorded; planning with moving hazards absent. |
| **vc33** 0/7 (holdout, human 7) | Click the button 3 times: each click shifts the piston and the marker by -2 until the marker's colour-11 base is level with the colour-11 wall mark. | Avatar = the 1064-pixel black piston (P); 785 clicks with no causal model of what a click does; 15 budget deaths. | P, R, G | The most damning row: seven human actions. A click law "click(button) -> marker moves -2" is a causal relation on a NON-avatar target, outside the one click law we have; the goal is alignment of two same-coloured marks — a same-as(coordinate) relation, free on frame 1. 785 clicks taught nothing because click credit is "the world changed", not "what changed, by how much, toward what". |
| **wa30** 0/9 (holdout, human 71) | Face a crate, ACTION5 grabs it (border turns black), drag it in any direction, ACTION5 drops; all three crates in the pen. | ACTION4 (a facing turn) read as ROTATION -> docking fired; ACTION5 pressed 356 times; goal = the pen's interior colour; oscillated beside the pen to the budget. | C, R, G | Facing != rotation (twice today); grab/attach is a state relation with a visible livery (border colour 3 -> black) the ledger could hold; "count(crates inside pen) == 3" is the goal, and the pen is exactly three crates wide — free information. |

## Where the loop fails — the cross-cutting reading

1. **There is no goal-inference organ.** In 18 of 24 games the win is a
   RELATION — cover, match, inside, aligned, equal-to-reference,
   sequence-equals, coincide — never "walk onto the unique small object".
   The agent's goal vocabulary is that one heuristic plus collectibles plus
   whatever an archetype hard-codes. Every holdout game is lost here, and
   in every one the relation is visible on frame 1 as a same-colour /
   same-shape / same-size correspondence between a movable thing and a
   static marker, a picture beside a canvas, a strip beside a track, or a
   table of pairs.
2. **"Unreachable" was misread as "not the goal".** The refutation rule
   added this week struck the TRUE goal in dc22, g50t, ls20, su15, tu93
   and sx02. Unreachable means a mechanic is missing; it belongs to the
   open-hypothesis state, not to goal refutation. WITHDRAWN today: an
   unreachable candidate is now DEFERRED (held, alternatives pursued
   while one is unreachable, re-opened on any new law or playfield
   change); only "reached twice without effect" refutes. Blind set 1
   unchanged; the games it had struck do not win yet — their missing
   mechanics are the real cause — but their true goal is no longer
   discarded.
3. **"Which body do I control" is not a concept.** cn04, ka59, re86,
   r11l, ar25, sp80, m0r0 transfer control by click or ACTION5, and the
   controlled body wears a livery (black centre/fill, black border).
   Identity binds to the livery pixel (ka59) or flips between bodies
   (re86, ar25). One concept — selection with a visible livery — is
   missing from perception and from the motion model.
4. **A facing change is not a rotation.** sc25 and wa30 turned the
   sprite's facing on arrow presses; the dihedral detector called it a
   shape-op and the docking archetype hijacked hundreds of actions.
5. **Budgets are everywhere and read as deaths.** All 24 games meter
   actions on a bar or counter (row 0 or 63, or a step limit); when the timer detector misses the
   bar (sc25, r11l, tu93 L2) budget deaths condemn innocent objects as
   lethal, including the door (sc25). Every dithering loop we have —
   probing, clicking, frontier walking — pays in the only currency the
   score counts.
6. **Clicks have one law; games have ten.** Our click vocabulary is
   "click(X) teleports the avatar to X". The games: select (ka59, r11l,
   cn04, sb26, sp80), toggle (ft09, tn36, sc25, dc22), pump/grow (vc33,
   s5i5), pull-toward-point (su15), rotate a ring (lp85), delete (bp35),
   move-to-point (r11l). And the click policy spends hundreds of clicks
   without building a causal model of WHAT changed (m0r0 749, vc33 785,
   tn36 788, su15's 795 undos): credit is "the world changed", never
   "object O moved by d / changed state s".
7. **Deferred effects / programs.** sb26, tn36, cd82, sp80, sc25: set up a
   configuration, then a commit action evaluates it. Only sb26's version
   exists, hard-wired.
8. **Perception priors that remove the game.** The playfield band-split
   cut the control panel out of dc22 and half the board out of sx02; the
   largest-mover identity picked scenery in bp35 and vc33.
9. **Compiled organs do not grow.** Every DEV game that scores does so
   through an archetype, and every one of them stops at the first level
   that adds one relation (ft09 L5, lf52 L2, sb26 L2, s5i5 L2, sp80 L2,
   lp85 L4, tu93 L2, cn04 L2). The archetypes' hidden contribution was
   measured at zero (0.21 with all of them, 0.30 with almost none).

## What a human does that we do not (the priors the study exposes)

Reading the 24 truth files, the same handful of first-frame inferences
win L1 after L1: two things share a colour -> they belong together
(disc/diamond, bar/dots, marker/mark, button/tile, block/socket); a
picture beside a blank of the same size -> make them equal; a strip that
repeats the play area's glyphs -> a reference to match; a frame whose
interior fits a block -> put the block in it; a table of pairs -> a
mapping to apply; a dotted line from A to B -> take A to B; a full bar
that shrinks when I act -> a budget. None of these is game-specific, and
none is in the core. They are exactly the "equipped scientist" priors
the doctrine named and the archetype campaign displaced.

## Consequences for the plan

The next organ is goal induction from free information, with the
hypothesis classes above (cover / inside / aligned / equal / sequence /
mapping / count) generated from same-colour, same-shape and same-size
correspondences on frame 1, pursued cheapest-first, and verified by the
level ending. Alongside it, three corrections the study makes
unavoidable: selection-with-livery as a perception concept; facing is
not rotation; and a universal budget reader so budget deaths never
condemn anything. The unreachable-goal refutation is withdrawn now.
