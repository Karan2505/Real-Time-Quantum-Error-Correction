"""
Operations console: REST API + live dashboard for a VirtualLab.

Standard library only (``http.server``).  Long operations (calibration,
experiment runs) are executed by a single background worker so the API stays
responsive; clients poll ``/api/jobs/<id>``.

Security: binds to 127.0.0.1 by default.  If ``token`` is set, every request
must carry ``X-Aura-Token: <token>`` (or ``?token=``).  There is no TLS -- put
a reverse proxy in front of it before exposing it beyond localhost.

Endpoints
---------
GET  /                      dashboard
GET  /api/status            device, calibration, link, recent runs, event log
GET  /api/device            device spec (topology)
GET  /api/calibration       current calibration snapshot
GET  /api/link              AURA-Link statistics (both endpoints)
GET  /api/runs              run list          GET /api/runs/<id>   full result
GET  /api/jobs              job list          GET /api/jobs/<id>   job status
POST /api/calibrate         {"mode": "full" | "maintain" | "oracle"}
POST /api/run               {"rounds": 3, "basis": "Z", "shots": 200, "decoder": "matching",
                             "mode": "stream", "branching": false}
POST /api/impairments       {"ber": 1e-6, "los_us": [t0, t1] | null, "direction": "down"|"up"}
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from ...core.errors import AuraError
from ...core.units import US
from ...link.phy import Impairments
from ..runner.results import _json_default

MAX_BODY = 64 * 1024


class _Worker:
    def __init__(self, lab):
        self.lab = lab
        self.q: "queue.Queue[tuple[str, str, dict]]" = queue.Queue()
        self.jobs: dict[str, dict] = {}
        self.t = threading.Thread(target=self._loop, name="aura-console-worker", daemon=True)
        self.t.start()

    def submit(self, kind: str, params: dict) -> str:
        jid = uuid.uuid4().hex[:8]
        self.jobs[jid] = {"id": jid, "kind": kind, "params": params, "state": "queued",
                          "submitted": time.time(), "result": None, "error": None}
        self.q.put((jid, kind, params))
        return jid

    def _loop(self) -> None:
        while True:
            jid, kind, p = self.q.get()
            job = self.jobs[jid]
            job["state"] = "running"
            job["started"] = time.time()
            try:
                if kind == "calibrate":
                    job["result"] = self.lab.calibrate(p.get("mode", "full"))
                elif kind == "run":
                    exp = self.lab.experiment(rounds=int(p.get("rounds", 3)), basis=str(p.get("basis", "Z")),
                                              branching=bool(p.get("branching", False)))
                    res = self.lab.run(exp, shots=int(p.get("shots", 100)), decoder=str(p.get("decoder", "matching")),
                                       mode=str(p.get("mode", "stream")))
                    job["result"] = {"run_id": res.run_id, "summary": res.summary()}
                else:
                    raise AuraError(f"unknown job kind {kind}")
                job["state"] = "done"
            except Exception as e:  # report every failure to the operator
                job["state"] = "failed"
                job["error"] = f"{type(e).__name__}: {e}"
                job["trace"] = traceback.format_exc(limit=5)
            job["finished"] = time.time()


def _validate_run(p: dict) -> dict:
    out = {}
    try:
        out["rounds"] = int(p.get("rounds", 3))
        out["shots"] = int(p.get("shots", 100))
    except (TypeError, ValueError):
        raise AuraError("rounds and shots must be integers")
    if not 1 <= out["rounds"] <= 1000:
        raise AuraError("rounds must be in 1..1000")
    if not 1 <= out["shots"] <= 10_000_000:
        raise AuraError("shots must be in 1..10^7")
    out["basis"] = str(p.get("basis", "Z")).upper()
    if out["basis"] not in ("Z", "X"):
        raise AuraError("basis must be Z or X")
    out["decoder"] = str(p.get("decoder", "matching"))
    if out["decoder"] not in ("matching", "oracle", "null"):
        raise AuraError("decoder must be matching, oracle or null")
    out["mode"] = str(p.get("mode", "stream"))
    if out["mode"] not in ("stream", "batch"):
        raise AuraError("mode must be stream or batch")
    out["branching"] = bool(p.get("branching", False))
    return out


def make_handler(lab, worker: _Worker, token: Optional[str]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AURA-Console/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            pass

        # ------------------------------------------------------------ helpers
        def _auth(self) -> bool:
            if not token:
                return True
            q = parse_qs(urlparse(self.path).query)
            if self.headers.get("X-Aura-Token") == token or q.get("token", [None])[0] == token:
                return True
            self._json({"error": "unauthorised"}, HTTPStatus.UNAUTHORIZED)
            return False

        def _json(self, obj: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(obj, default=_json_default).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise AuraError("request body too large")
            raw = self.rfile.read(n) if n else b"{}"
            try:
                d = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                raise AuraError("invalid JSON body")
            if not isinstance(d, dict):
                raise AuraError("JSON body must be an object")
            return d

        # ------------------------------------------------------------ GET
        def do_GET(self) -> None:  # noqa: N802
            if not self._auth():
                return
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if path == "/":
                    html = resources.files("aura.host.console").joinpath("dashboard.html").read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                elif path == "/api/status":
                    self._json(lab.status())
                elif path == "/api/device":
                    self._json(lab.spec.to_dict())
                elif path == "/api/calibration":
                    self._json(json.loads(lab.store.snapshot(lab.sim.now).to_json()))
                elif path == "/api/link":
                    self._json({"control": lab.link.control.stats(), "gateway": lab.link.gateway.stats()})
                elif path == "/api/runs":
                    self._json([{"run_id": r.run_id, "experiment": r.experiment, "mode": r.mode,
                                 "decoder": r.decoder, "shots": r.shots} for r in lab.runs])
                elif path.startswith("/api/runs/"):
                    rid = path.split("/")[-1]
                    for r in lab.runs:
                        if r.run_id == rid:
                            self._json(r.to_dict())
                            return
                    self._json({"error": "no such run"}, HTTPStatus.NOT_FOUND)
                elif path == "/api/jobs":
                    self._json(sorted(worker.jobs.values(), key=lambda j: j["submitted"], reverse=True)[:50])
                elif path.startswith("/api/jobs/"):
                    j = worker.jobs.get(path.split("/")[-1])
                    self._json(j if j else {"error": "no such job"}, HTTPStatus.OK if j else HTTPStatus.NOT_FOUND)
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        # ------------------------------------------------------------ POST
        def do_POST(self) -> None:  # noqa: N802
            if not self._auth():
                return
            path = urlparse(self.path).path.rstrip("/")
            try:
                body = self._body()
                if path == "/api/calibrate":
                    mode = str(body.get("mode", "full"))
                    if mode not in ("full", "maintain", "oracle"):
                        raise AuraError("mode must be full, maintain or oracle")
                    self._json({"job": worker.submit("calibrate", {"mode": mode})}, HTTPStatus.ACCEPTED)
                elif path == "/api/run":
                    self._json({"job": worker.submit("run", _validate_run(body))}, HTTPStatus.ACCEPTED)
                elif path == "/api/impairments":
                    ber = float(body.get("ber", 0.0))
                    los = body.get("los_us")
                    direction = str(body.get("direction", "down"))
                    if direction not in ("down", "up"):
                        raise AuraError("direction must be down or up")
                    intervals = []
                    if los:
                        t0 = lab.sim.now + int(float(los[0]) * US)
                        intervals = [(t0, t0 + int((float(los[1]) - float(los[0])) * US))]
                    imp = Impairments(ber=ber, los_intervals=intervals)
                    with lab.lock:
                        lab.link.set_impairments(**{direction: imp})
                    self._json({"ok": True, "direction": direction, "ber": ber, "los": intervals})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (AuraError, ValueError, TypeError) as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


class ConsoleServer:
    def __init__(self, lab, host: str = "127.0.0.1", port: int = 8765, token: Optional[str] = None):
        self.lab = lab
        self.worker = _Worker(lab)
        self.httpd = ThreadingHTTPServer((host, port), make_handler(lab, self.worker, token))
        self.httpd.daemon_threads = True
        self.address = self.httpd.server_address
        self._t: Optional[threading.Thread] = None

    def start(self) -> "ConsoleServer":
        self._t = threading.Thread(target=self.httpd.serve_forever, name="aura-console", daemon=True)
        self._t.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
