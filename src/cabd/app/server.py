"""Local web server of the CABD demo app (standard library only).

usage: python -X utf8 -m cabd.app.server [--port 8765]   then open http://localhost:8765
API (JSON):  GET /api/meta   POST /api/geometry {x}   POST /api/forward {x}   POST /api/inverse {theta2, u2, dE}
The models (models/app_models_n1536.joblib, built by cabd.app.bundle) are loaded once at start-up.
"""
import argparse
import json
import math
import threading
import traceback
import warnings
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from . import engine as E  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
ENGINE = None                     # set in main() (or by a caller before serving)
INVERSE_LOCK = threading.Lock()


def to_json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def send_json(self, obj, code=200):
        body = json.dumps(obj, default=to_json).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/meta":
            return self.send_json(E.meta())
        return super().do_GET()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/geometry":
                return self.send_json(E.geometry_payload(np.asarray(req["x"], float)))
            if self.path == "/api/forward":
                return self.send_json(ENGINE.forward(np.asarray(req["x"], float)))
            if self.path == "/api/inverse":
                tg = {k: (float(req[k]) if req.get(k) not in (None, "") else None) for k in ("theta2", "u2", "dE")}
                if any(v is not None and not math.isfinite(v) for v in tg.values()):
                    return self.send_json({"ok": False, "message": "Targets must be finite numbers."})
                if tg["dE"] is not None and tg["dE"] <= 0:
                    return self.send_json({"ok": False, "message": "ΔE must be positive."})
                with INVERSE_LOCK:
                    return self.send_json(ENGINE.inverse(tg))
            self.send_json({"ok": False, "message": "unknown endpoint"}, 404)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"ok": False, "message": f"server error: {e}"}, 500)


def warm_up():
    """Start the worker processes once so the first inverse request is not slowed by process start-up."""
    try:
        E.Parallel(n_jobs=E.N_JOBS, backend="loky")(E.delayed(abs)(i) for i in range(E.N_JOBS))
    except Exception:
        pass


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-warm-up", action="store_true", help="do not pre-start the inverse worker processes")
    args = ap.parse_args()
    ENGINE = E.Engine()
    if not args.no_warm_up:
        threading.Thread(target=warm_up, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"CABD demo running at http://localhost:{args.port}  (models loaded in {ENGINE.load_s:.1f} s)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
