"""Measurement proxy between MobileGym and the model server (OpenAI-compatible, llama-server).

    python -m edge.meter_proxy --listen 127.0.0.1:9090 --upstream http://127.0.0.1:8080 \
        --log edge/results/<run>/proxy.jsonl --tag <run>

Forwards every request unchanged and appends one JSON line per chat call to the log:

  ts, end_ts          start / end of the request (epoch seconds, real clock)
  fingerprint         SHA-256 of the messages, images excluded (links the call to the saved prompt of a step)
  n_images, image_sizes   images sent and their sizes [[w, h], ...] in pixels
  stream              whether the client asked for streaming
  timings             llama-server timings: cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms
  wall_ms             total duration of the request seen by the proxy
  system_fingerprint  llama.cpp build reported by the server
  status, error       HTTP status of the upstream; error text if the upstream was unreachable

Streaming (SSE) and non-streaming requests are both supported. Standard library only.
"""
from __future__ import annotations

import argparse
import base64
import http.client
import json
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from edge.common import count_images, fingerprint

_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
               "transfer-encoding", "upgrade", "host", "content-length"}
_TIMING_KEYS = ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms")


def image_size(data: bytes) -> Optional[tuple[int, int]]:
    """(width, height) of a PNG or JPEG from its header; None if unknown."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            (seg_len,) = struct.unpack(">H", data[i + 2:i + 4])
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return int(w), int(h)
            i += 2 + seg_len
    return None


def image_sizes(messages: list) -> list[list[int]]:
    sizes: list[list[int]] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for it in content:
            if not isinstance(it, dict) or it.get("type") != "image_url":
                continue
            url = (it.get("image_url") or {}).get("url", "")
            if isinstance(url, str) and url.startswith("data:image") and ";base64," in url:
                try:
                    wh = image_size(base64.b64decode(url.split(";base64,", 1)[1]))
                except Exception:
                    wh = None
                sizes.append(list(wh) if wh else [0, 0])
    return sizes


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, upstream: str, log_path: Optional[Path], tag: str, timeout_s: float):
        super().__init__(addr, handler)
        u = urlsplit(upstream)
        self.up_host, self.up_port = u.hostname or "127.0.0.1", u.port or (443 if u.scheme == "https" else 80)
        self.up_https = u.scheme == "https"
        self.up_prefix = u.path.rstrip("/")
        self.log_path, self.tag, self.timeout_s = log_path, tag, timeout_s
        self.lock = threading.Lock()

    def write_log(self, rec: dict[str, Any]) -> None:
        if not self.log_path:
            return
        line = json.dumps(rec, ensure_ascii=False)
        with self.lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, fmt, *args):  # silence the default stderr access log
        pass

    # --- helpers -----------------------------------------------------------------------------
    def _chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def _connect(self):
        cls = http.client.HTTPSConnection if self.server.up_https else http.client.HTTPConnection
        return cls(self.server.up_host, self.server.up_port, timeout=self.server.timeout_s)

    # --- request -----------------------------------------------------------------------------
    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def _forward(self, method: str) -> None:
        if self.path == "/__health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        t0 = time.time()
        rec: dict[str, Any] = {"ts": t0, "tag": self.server.tag, "path": self.path, "method": method}

        req = None
        if method == "POST" and body:
            try:
                req = json.loads(body)
            except Exception:
                req = None
        is_chat = isinstance(req, dict) and isinstance(req.get("messages"), list) and "chat/completions" in self.path
        if is_chat:
            msgs = req["messages"]
            rec.update(fingerprint=fingerprint(msgs), n_images=count_images(msgs),
                       image_sizes=image_sizes(msgs), stream=bool(req.get("stream")))

        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
        headers["Content-Length"] = str(len(body))
        conn = self._connect()
        try:
            conn.request(method, self.server.up_prefix + self.path, body, headers)
            resp = conn.getresponse()
        except Exception as e:
            rec.update(end_ts=time.time(), wall_ms=round((time.time() - t0) * 1000, 1), status=502, error=repr(e))
            self.server.write_log(rec)
            msg = json.dumps({"error": {"message": f"upstream unreachable: {e!r}"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            conn.close()
            return

        timings: dict[str, Any] = {}
        sysfp: Optional[str] = None
        usage: Optional[dict] = None

        def absorb(obj: Any) -> None:
            nonlocal sysfp, usage
            if not isinstance(obj, dict):
                return
            tm = obj.get("timings")
            if isinstance(tm, dict):
                for k in _TIMING_KEYS:
                    if k in tm:
                        timings[k] = tm[k]
            if obj.get("system_fingerprint"):
                sysfp = obj["system_fingerprint"]
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]

        ctype = resp.getheader("Content-Type", "")
        try:
            if "text/event-stream" in ctype:
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    s = line.strip()
                    if s.startswith(b"data:") and s != b"data: [DONE]":
                        try:
                            absorb(json.loads(s[5:]))
                        except Exception:
                            pass
                    self._chunk(line)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                data = resp.read()
                try:
                    absorb(json.loads(data))
                except Exception:
                    pass
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in _HOP_BY_HOP:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            rec["status"] = resp.status
        except (BrokenPipeError, ConnectionResetError) as e:
            rec.update(status=resp.status, error=f"client disconnected: {e!r}")
        except Exception as e:
            rec.update(status=resp.status, error=repr(e))
        finally:
            conn.close()

        if is_chat:
            if usage and "prompt_n" not in timings:  # servers without "timings": fall back to usage
                timings.setdefault("prompt_n", usage.get("prompt_tokens"))
                timings.setdefault("predicted_n", usage.get("completion_tokens"))
            rec.update(timings=timings, system_fingerprint=sysfp)
            rec.update(end_ts=time.time(), wall_ms=round((time.time() - t0) * 1000, 1))
            self.server.write_log(rec)


def make_server(listen: str, upstream: str, log_path: Optional[str], tag: str = "",
                timeout_s: float = 3600.0) -> _Server:
    host, _, port = listen.rpartition(":")
    lp = Path(log_path) if log_path else None
    if lp:
        lp.parent.mkdir(parents=True, exist_ok=True)
    return _Server((host or "127.0.0.1", int(port)), _Handler, upstream, lp, tag, timeout_s)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="127.0.0.1:9090")
    ap.add_argument("--upstream", required=True, help="model server, e.g. http://127.0.0.1:8080")
    ap.add_argument("--log", required=True, help="proxy.jsonl to append to")
    ap.add_argument("--tag", default="", help="label stored in every line")
    ap.add_argument("--timeout", type=float, default=3600.0, help="upstream timeout in seconds")
    args = ap.parse_args(argv)
    srv = make_server(args.listen, args.upstream, args.log, args.tag, args.timeout)
    print(f"meter_proxy {args.listen} -> {args.upstream}  log={args.log}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
