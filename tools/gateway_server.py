"""A LOCAL GATEWAY: the arc_agi REST server in competition mode over the
local environment files — what the Kaggle rerun's gateway is to the
notebook.  Run by tools/gateway_run.py as a separate process (as on
Kaggle), so the client's CPU load and the server's are apart.

    .venv/bin/python tools/gateway_server.py 8765
"""
import os, sys, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
logging.basicConfig(level=logging.WARNING)
import arc_agi
from arc_agi.server import create_app

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
arc = arc_agi.Arcade(operation_mode="offline", environments_dir=str(ROOT / "environment_files"))
arc.logger.setLevel(logging.WARNING)
app, api = create_app(arc, competition_mode=True, save_all_recordings=False)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
app.run(host="127.0.0.1", port=port, threaded=True)
