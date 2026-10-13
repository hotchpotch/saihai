"""Serve a TypeSafe-style API over a separately managed llama-server."""

import argparse
import ipaddress
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .inference import LlamaDecisionClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args()
    address = ipaddress.ip_address(args.host)
    if address.version != 4 or not (address.is_loopback or address in ipaddress.ip_network("100.64.0.0/10")):
        raise ValueError("Bind to localhost or a Tailscale IPv4 address")
    client = LlamaDecisionClient(args.backend_url, args.contract)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            try:
                with urlopen(client.url + "/health", timeout=5):
                    self.respond(200, {"status": "ok", "model": client.contract["model_id"]})
            except (HTTPError, URLError, TimeoutError):
                self.respond(503, {"error": "Backend unavailable"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 4 * 1024 * 1024:
                    self.respond(413, {"error": "Invalid request size"})
                    return
                result = client.systemone(json.loads(self.rfile.read(size)))
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                self.respond(400, {"error": str(exc)})
            except (HTTPError, URLError, TimeoutError) as exc:
                self.respond(502, {"error": str(exc)})
            else:
                self.respond(200, result)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
