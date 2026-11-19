"""Kept-alive connections for the demo's cloud calls, and the earlier first clause of a reply. No network."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from convo import brain, http_pool


@pytest.fixture
def server():
    seen = {"conns": set(), "n": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"                 # keep-alive, like the real APIs

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            seen["conns"].add(self.client_address)
            seen["n"] += 1
            body = b"data: one\n\ndata: two\n\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    seen["url"] = f"http://127.0.0.1:{httpd.server_port}/v1/chat"
    yield seen
    httpd.shutdown()


def test_the_second_request_reuses_the_connection(server):
    for _ in range(3):
        lines = [l for l in http_pool.post_lines(server["url"], b"{}", {"Content-Type": "application/json"}, 5) if l.strip()]
        assert lines == [b"data: one\n", b"data: two\n"]
    assert server["n"] == 3 and len(server["conns"]) == 1


def test_a_warmed_connection_is_used_by_the_first_request(server):
    assert http_pool.warm(server["url"].replace("/v1/chat", "/other"))
    http_pool.post(server["url"], b"{}", {}, 5)
    assert len(server["conns"]) == 1


def test_warm_never_raises_when_the_host_is_down():
    assert http_pool.warm("http://127.0.0.1:9/") is False


def test_the_first_clause_is_spoken_before_the_sentence_ends(monkeypatch):
    pieces = ["أبشر، الدفعة ثلاث مية وخمسين ريال، ", "وتقدر تسددها من التطبيق", " أو بالرابط."]
    monkeypatch.setattr(brain, "stream", lambda msgs: iter(pieces))
    out = list(brain.sentences([]))
    assert out[0] == "أبشر، الدفعة ثلاث مية وخمسين ريال،"     # a short «أبشر،» alone is not cut off
    assert "".join(out).replace(" ", "") == "".join(pieces).replace(" ", "")
    assert len(out) == 2                                       # only the first piece may end at a comma
