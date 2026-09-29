#!/usr/bin/env python3
# Serve out/comparar em 127.0.0.1:8765 e grava cada veredito do exame humano em
# out/comparar/vereditos.jsonl (uma linha por decisão; a última decisão de um par vale).
# Uso: scripts/servidor_revisao.py   →   http://127.0.0.1:8765/socofing.html
import json, os, http.server, functools
from datetime import datetime

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "comparar")
ARQ = os.path.join(OUT, "vereditos.jsonl")


class H(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/veredito":
            return self.send_error(404)
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        d["gravado_em"] = datetime.now().isoformat(timespec="seconds")
        with open(ARQ, "a") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        self.send_response(204)
        self.end_headers()


if __name__ == "__main__":
    print(f"http://127.0.0.1:8765/socofing.html  (vereditos → {os.path.abspath(ARQ)})")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8765), functools.partial(H, directory=OUT)).serve_forever()
