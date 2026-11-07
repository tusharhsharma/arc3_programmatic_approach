import world as W, learner as L, frozen as F
def eff(o, a, won): return 0.0 if not won else min(1.15, (o / a) ** 2)
rows = []
for rule in W.DEV + W.HELD:
    o = W.oracle_len(rule)
    fr = [F.play(rule, seed=s) for s in range(10)]
    fw = sum(r["won"] for r in fr); fe = sum(eff(o, r["actions"], r["won"]) for r in fr) / 10
    pa = L.Learner(active=False).play(rule); ac = L.Learner(active=True).play(rule)
    rows.append((rule, o, f"{fw}/10, eff {fe:.2f}",
                 f"{'won' if pa['won'] else 'lost'} {pa['actions']} (eff {eff(o, pa['actions'], pa['won']):.2f})",
                 f"{'won' if ac['won'] else 'lost'} {ac['actions']} (eff {eff(o, ac['actions'], ac['won']):.2f})",
                 ac["rule"], ac["memory"], ac["missing_variable"]))
print("| game | oracle | frozen (lever only, 10 seeds) | passive learner | active learner | active's final rule | memory carried | missing variable reported |")
print("|---|---|---|---|---|---|---|---|")
for r in rows: print("| " + " | ".join(str(x) for x in r) + " |")
