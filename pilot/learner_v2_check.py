"""The missing-variable learner (frozen before the held-out games were written).
1. Keep every observation of the gate with the memory it occurred in.
2. Hypotheses: small executable expressions over grounded primitives
   (on pad; entry count; turns since entry / since exit; turn parity),
   combined by NOT / AND / OR, ordered by size.
3. Active: when the survivors disagree on whether a plan works, take the
   shortest action sequence to a state where they predict different gate
   states (every action charged).  Passive: plan under the simplest survivor.
4. The planner's state carries exactly the memory the surviving rules use.
If every hypothesis is refuted, the agent reports that its description
is missing a variable it has no primitive for."""
from collections import deque
from itertools import combinations
import world as W

def atoms():
    A = [("onpad", lambda m: m[4], {"onpad"}),
         ("entries odd", lambda m: m[0] % 2 == 1, {"entries"}),
         ("t even", lambda m: m[3] == 0, {"t mod 2"})]
    for n in (1, 2, 3):
        A.append((f"entries>={n}", (lambda n: lambda m: m[0] >= n)(n), {"entries"}))
    for k in range(1, 15):
        A.append((f"since_entry<{k}", (lambda k: lambda m: m[1] < k)(k), {"since entry"}))
        A.append((f"since_exit<{k}", (lambda k: lambda m: m[2] < k)(k), {"since exit"}))
    return A

def hypotheses():
    A = atoms(); H = []
    for n, f, u in A:
        H.append((1, n, f, u)); H.append((2, "not " + n, (lambda f: lambda m: not f(m))(f), u))
    for (n1, f1, u1), (n2, f2, u2) in combinations(A, 2):
        H.append((3, f"{n1} and {n2}", (lambda a, b: lambda m: a(m) and b(m))(f1, f2), u1 | u2))
        H.append((3, f"{n1} or {n2}", (lambda a, b: lambda m: a(m) or b(m))(f1, f2), u1 | u2))
    H.sort(key=lambda h: h[0])
    return H

import random
class Learner:
    def __init__(self, active=True, budget=120, depth=16, seed=0):
        self.active, self.budget, self.depth = active, budget, depth
        self.rnd = random.Random(seed)
        self.H = hypotheses(); self.data = []; self.log = []

    def survivors(self):
        return [h for h in self.H if all(bool(h[2](m)) == g for m, g in self.data)]

    def plan(self, s, hs, max_nodes=60000):
        """Shortest plan to the target that works under every rule in hs."""
        s0 = (s, tuple(bool(h[2](s[2])) for h in hs))
        q = deque([(s[0], s[1], s[2], [])]); seen = {(s[0], s[1], s[2])}
        while q and len(seen) < max_nodes:
            p, b, m, path = q.popleft()
            if p == W.TARGET: return path
            gs = {bool(h[2](m)) for h in hs}
            if len(gs) > 1 and self._adj_gate(p): continue   # rules disagree where it matters
            g = gs.pop()
            for a in "UDLR":
                n = W.step(p, b, m, g, a)
                if n not in seen:
                    seen.add(n); q.append((n[0], n[1], n[2], path + [a]))
        return None

    def _adj_gate(self, p):
        return abs(p[0] - W.GATE[0]) + abs(p[1] - W.GATE[1]) <= 1

    def experiment(self, s, groups):
        """Shortest action sequence to a state where the leading two groups
        predict different gate states (simulated under the first group)."""
        f = groups[0][2]
        q = deque([(s[0], s[1], s[2], [])]); seen = {(s[0], s[1], s[2])}
        while q and len(seen) < 60000:
            p, b, m, path = q.popleft()
            if path and len({bool(h[2](m)) for h in groups}) > 1: return path
            if len(path) >= self.depth: continue
            for a in "UDLR":
                n = W.step(p, b, m, bool(f(m)), a)
                if n not in seen:
                    seen.add(n); q.append((n[0], n[1], n[2], path + [a]))
        return None

    def runs(self, s, path, h):
        p, b, m = s
        for a in path:
            p, b, m = W.step(p, b, m, bool(h[2](m)), a)
            if p == W.TARGET: return True
        return p == W.TARGET

    def choose(self, s, S):
        """Active: plan under the simplest rule; if rules holding most of the
        prior weight (2^-size) say that plan fails, first take the cheapest
        experiment on which the simplest rule and the heaviest dissenter disagree."""
        top = S[:300]; wt = lambda h: 2.0 ** (-h[0])
        p = None; best = None
        for h in top:
            p = self.plan(s, [h])
            if p: best = h; break
        if not p: return None, "no plan"
        tot = sum(wt(h) for h in top); ok = [h for h in top if self.runs(s, p, h)]
        if sum(wt(h) for h in ok) >= 0.6 * tot: return p, "plan"
        dis = next((h for h in top if not self.runs(s, p, h)), None)
        e = self.experiment(s, [best, dis]) if dis else None
        if e:
            self.log.append(("experiment", best[1], dis[1], len(e)))
            return e, "experiment"
        return p, "plan"

    def play(self, rule):
        G = W.Game(rule); self.data.append((G.m, G.gate()))
        missing = False
        while G.actions < self.budget and not G.won():
            S = self.survivors()
            if not S:
                missing = True; break
            # group survivors by their behaviour on the states reachable soon
            groups = S[:40]
            s = (G.pos, G.box, G.m)
            if self.active:
                seq, why = self.choose(s, S)
            else:
                seq = None
                for h in S[:300]:
                    seq = self.plan(s, [h])
                    if seq: break
            if not seq:                                   # fallback: go if open, else explore
                seq = [self.rnd.choice("UDLR")]
            for a in seq:
                before = {h[1]: None for h in S[:2]}
                G.act(a); self.data.append((G.m, G.gate()))
                if len(self.survivors()) < len(S) or G.won() or G.actions >= self.budget:
                    break
        final = self.survivors()
        return {"won": G.won(), "actions": G.actions, "missing_variable": missing,
                "rule": final[0][1] if final else None,
                "memory": sorted(final[0][3]) if final else None, "n_left": len(final)}
