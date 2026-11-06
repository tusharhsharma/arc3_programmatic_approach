#!/bin/bash
cd /root/arc3
: > /tmp/inv/regC.txt
for g in tn36 lf52 cn04 bp35 wa30 lp85 r11l tu93 m0r0 vc33 ar25 ka59 cd82 ft09 ls20 s5i5 su15 tr87; do
  timeout 900 .venv/bin/python tools/core_play2.py $g:1 --no-floor --seconds 300 > /tmp/inv/regC_$g.log 2>&1
  grep -a "^$g:[0-9] \(WON\|lost\)" /tmp/inv/regC_$g.log | cut -c1-60 >> /tmp/inv/regC.txt
done
echo DONE >> /tmp/inv/regC.txt
: > /tmp/inv/climb.txt
for t in tu93:1-3 wa30:1-3 cd82:1-3 m0r0:1-3 vc33:1-3 ka59:1-3 ft09:1-6 cn04:1-2 ls20:1-2 re86:1-2 s5i5:1-2 tw01:1-3 tw02:1-3 tw03:1-3 tw04:1-3 tw05:1-3; do
  g=${t%%:*}
  timeout 2400 .venv/bin/python tools/core_play2.py $t --no-floor --seconds 300 > /tmp/inv/climb_$g.log 2>&1
  grep -a "^$g:[0-9] \(WON\|lost\|the core failed\)" /tmp/inv/climb_$g.log | cut -c1-70 >> /tmp/inv/climb.txt
done
echo DONE >> /tmp/inv/climb.txt
