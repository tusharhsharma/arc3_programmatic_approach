import json,glob,re,sys
base={}
for m in glob.glob('/root/arc3/environment_files/*/*/metadata.json'):
    d=json.load(open(m)); g=d['game_id'][:4]
    if 'baseline_actions' in d: base[g]=d['baseline_actions']
pub="ft09 sb26 tu93 cd82 m0r0 vc33 ka59 ls20 s5i5 re86 wa30 lf52 sp80 g50t su15 tn36 cn04 bp35 lp85 r11l ar25 tr87 sc25 sk48 dc22".split()
rows={}
for f in ['/tmp/final/A.txt','/tmp/final/B.txt']:
    for line in open(f):
        m=re.match(r'(\w{4}):(\d+) (WON|lost) in (\d+) actions',line)
        if not m: continue
        g,l,r,a=m.group(1),int(m.group(2)),m.group(3),int(m.group(4))
        rows.setdefault(g,{}).setdefault(l,[]).append((r,a))
out=[]; tot=0; first=0; effs=[]
for g in pub:
    if g not in rows: out.append((g,'not run','')); continue
    won=[]
    for l in sorted(rows[g]):
        att=rows[g][l]
        if any(r=='WON' for r,_ in att):
            ch=sum(a for _,a in att); h=base[g][l-1]; e=min((h/ch)**2,1.15)
            won.append(f"L{l} {ch}/{h} ({e*100:.0f}%)"); effs.append(e); tot+=1
            if l==1: first+=1
    out.append((g,f"{len(won)}/{len(base[g])}","; ".join(won) if won else "—"))
print(f"FIRST {first} LEVELS {tot}")
print("| Game | Levels won | Per won level: charged actions / official baseline (level efficiency) |")
print("|---|---|---|")
for g,a,b in out: print(f"| {g} | {a} | {b} |")
print(f"\nmean level efficiency over won levels {sum(effs)/max(1,len(effs))*100:.0f}%")
