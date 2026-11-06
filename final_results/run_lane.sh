#!/bin/bash
# frozen final public run: snapshot s90 code (= bundle v38 core), --no-floor, 300 s a level, two attempts a level
cd /root/arc3
out=$1; shift
: > $out
for spec in "$@"; do
  g=${spec%%:*}
  timeout 7200 .venv/bin/python tools/core_play2.py $spec --no-floor --seconds 300 > /tmp/final/$g.log 2>&1
  grep -a "^$g:[0-9]* \(WON\|lost\)" /tmp/final/$g.log >> $out
done
echo DONE >> $out
