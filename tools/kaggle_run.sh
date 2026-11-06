#!/bin/bash
# THE RERUN, AS THE NOTEBOOK DOES IT (read off the v15 notebook page):
# the framework copied, our bundle dropped in as agents/templates/my_agent.py,
# agents/__init__.py rewritten to register it, a .env pointing at the
# gateway (OPERATION_MODE=online, a trailing slash on the base URL, a fixed
# key), then `python main.py --agent myagent` — the Swarm over every game the
# gateway lists.  The gateway here is tools/gateway_server.py.
#
#   ARC_RUN_BUDGET_S=600 tools/kaggle_run.sh tn36,sp80          # (a game filter, as main.py -g takes it)
#   ARC_RUN_BUDGET_S=3000 tools/kaggle_run.sh                   # (every game the gateway lists)
set -u
ROOT=/root/arc3
PORT=${GATEWAY_PORT:-8791}
KH=/tmp/kaggle_harness
rm -rf $KH && cp -r /root/ARC-AGI-3-Agents $KH && rm -rf $KH/.git
cp $ROOT/submission/my_agent_core.py $KH/agents/templates/my_agent.py
cat > $KH/agents/__init__.py <<'EOF'
from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent
load_dotenv()
AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    'random': Random,
    'myagent': MyAgent,
}
EOF
cat > $KH/.env <<EOF
SCHEME=http
HOST=127.0.0.1
PORT=$PORT
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://127.0.0.1:$PORT/
OPERATION_MODE=online
ENVIRONMENTS_DIR=
RECORDINGS_DIR=/tmp/kaggle_harness/server_recording
EOF
$ROOT/.venv/bin/python $ROOT/tools/gateway_server.py $PORT > /dev/null 2> /tmp/kaggle_gateway.err &
GW=$!
for i in $(seq 1 60); do curl -sf http://127.0.0.1:$PORT/api/games > /dev/null && break; sleep 1; done
cd $KH
GAME_ARG=""
if [ $# -ge 1 ]; then GAME_ARG="-g $1"; fi
MPLBACKEND=agg $ROOT/.venv/bin/python main.py --agent myagent $GAME_ARG 2>&1 | grep -v "^INFO\|^2026-.*| INFO"
kill $GW 2>/dev/null
