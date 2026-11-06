# Machine-readable ground truth (the answer key for the mechanics gate)

One file per game: `study/truth/<game>.json`, LEVEL 1 only, derived from
`study/truth/<game>.md` (the cited engine study). The agent NEVER reads
these; `tools/mechanics_gate.py` scores the agent's MODEL REPORT against
them. Everything is in the FRAME the agent sees (64x64 display pixels,
row/col), not engine units — the .md files usually give both.

Closed vocabularies (the report uses the same words):

ROLES      controlled | mover | marker | socket | button | selector |
           wall | floor | hazard | collectible | reference | canvas |
           tool | hud | budget | decor
  controlled : answers the arrow keys / is moved by clicks (the body)
  mover      : something the controlled thing or a button moves
  marker     : where a mover must end (mark, target, goal cell/outline)
  socket     : a marker whose cavity the mover must fill
  button     : a thing whose click does something elsewhere
  selector   : a thing whose click changes WHICH thing obeys the keys
  reference  : a picture the play area must be made equal to
  budget     : a bar/counter that shrinks per action or per time

EFFECTS    translate | slide | rotate | grow | shrink | recolour |
           select | appear | vanish | push | teleport | swap | toggle |
           merge | nothing
  translate: by a fixed vector per press;  slide: until blocked
  teleport : to the click point;           push: moves a neighbour along

CONDITIONS always | click_on(<entity>) | click_near(<entity>, r) |
           selected(<entity>) | blocked_by(<entity>) | at_edge |
           adjacent(<entity>) | budget_zero | parity | previous_action

GOAL KINDS align | cover | inside | equal | all_collected | count |
           sequence | mapping | merge | reach

Format:

{
  "game": "s5i5", "level": 1,
  "actions": ["ACTION6"],                         # available at L1
  "entities": [                                    # by COLOUR as displayed
    {"id": "arm11", "role": "mover", "colors": [11], "count": 1,
     "size": [24, 30], "note": "vertical bar with grey base"},
    ...
  ],
  "rules": [                                       # what every action does
    {"action": "ACTION6", "when": "click_on(slider11_far)",
     "effect": "grow", "of": "arm11", "by": 3, "note": "..."},
    {"action": "ACTION6", "when": "always", "effect": "nothing",
     "note": "any other click only spends a step"}
  ],
  "goal": {"kind": "inside", "mover": "tip13", "target": "diamond13",
           "all": true, "note": "every dot inside its diamond at once"},
  "budget": {"entity": "stepbar", "kind": "steps", "n": 50,
             "on_zero": "lose"},
  "hazards": []                                    # ids of lethal entities
}

Scoring (per game, the gate):
  ENTITIES  recall of (role, colour-set) pairs the agent reported.
  RULES     recall of (action, effect, of-colour) triples; conditions
            scored when the truth has one.
  GOAL      kind + mover colour + target colour.
  A game PASSES when entities >= 0.8, rules >= 0.8, goal exact.
