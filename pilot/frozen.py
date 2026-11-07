"""The frozen agent: the submitted agent's lever form only (a gate is open
while its pad is occupied), plus the fallback: go when the gate is open,
otherwise explore at random."""
import random
from collections import deque
import world as W

def bfs(p, b, passable_gate, goal):
    q = deque([(p, b, [])]); seen = {(p, b)}
    while q:
        pp, bb, path = q.popleft()
        if goal(pp, bb): return path
        for a in "UDLR":
            dr, dc = W.MOVES[a]; n = (pp[0] + dr, pp[1] + dc); b2 = bb
            if n in W.WALLS or (n == W.GATE and not passable_gate(pp, bb)): continue
            if n == bb:
                b2 = (bb[0] + dr, bb[1] + dc)
                if b2 in W.WALLS or b2 == W.GATE: continue
            if (n, b2) not in seen:
                seen.add((n, b2)); q.append((n, b2, path + [a]))
    return None

def play(rule, seed=0, budget=80):
    rnd = random.Random(seed); G = W.Game(rule); lever = None
    prev_on, prev_g = W.PAD in (G.pos, G.box), G.gate()
    while G.actions < budget and not G.won():
        if lever is not False and lever:
            path = bfs(G.pos, G.box, lambda p, b: W.PAD == b, lambda p, b: p == W.TARGET)
        elif G.gate():
            path = bfs(G.pos, G.box, lambda p, b: True, lambda p, b: p == W.TARGET)
        else:
            path = None
        a = path[0] if path else rnd.choice("UDLR")
        G.act(a)
        on, g = W.PAD in (G.pos, G.box), G.gate()
        if on != prev_on and g != prev_g and lever is None:
            lever = True
        elif lever and g != on:
            lever = False                       # the lever form is refuted; no other form
        prev_on, prev_g = on, g
    return {"won": G.won(), "actions": G.actions}
