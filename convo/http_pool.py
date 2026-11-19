"""Keep-alive HTTP(S) connections for the demo's cloud calls (brain and voice).

Every reply used to open a new TLS connection (urllib), which costs one or two network round trips
(about 100–300 ms) before the first byte of the reply. Here a finished connection goes back into a small pool
and the next request reuses it. `warm(url)` opens one in advance, so even the first reply of a call skips the
handshake. Demo path only: nothing in `turn/` uses this.
"""
from __future__ import annotations

import http.client
import threading
import urllib.error
from urllib.parse import urlsplit

_POOL: dict[tuple, list] = {}
_LOCK = threading.Lock()
MAX_IDLE = 4                  # connections kept per host


def _key(url: str) -> tuple:
    u = urlsplit(url)
    return u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80)


def _new(key: tuple, timeout: float) -> http.client.HTTPConnection:
    scheme, host, port = key
    cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    return cls(host, port, timeout=timeout)


def _take(key: tuple, timeout: float) -> tuple[http.client.HTTPConnection, bool]:
    with _LOCK:
        idle = _POOL.get(key)
        if idle:
            conn = idle.pop()
            conn.timeout = timeout
            if conn.sock is not None:
                conn.sock.settimeout(timeout)
            return conn, True
    return _new(key, timeout), False


def _give_back(key: tuple, conn: http.client.HTTPConnection) -> None:
    with _LOCK:
        idle = _POOL.setdefault(key, [])
        if len(idle) < MAX_IDLE:
            idle.append(conn)
            return
    conn.close()


def warm(url: str, timeout: float = 5.0) -> bool:
    """Open a connection (TCP + TLS) to the host of `url` now, so the next request starts at once. Never raises."""
    key = _key(url)
    with _LOCK:
        if _POOL.get(key):
            return True
    conn = _new(key, timeout)
    try:
        conn.connect()
    except OSError:
        conn.close()
        return False
    _give_back(key, conn)
    return True


def _path(url: str) -> str:
    u = urlsplit(url)
    return (u.path or "/") + (f"?{u.query}" if u.query else "")


def _send(url: str, body: bytes, headers: dict, timeout: float) -> tuple:
    """POST and return (connection, response). A reused connection that the server already closed is retried once."""
    key = _key(url)
    for attempt in range(2):
        conn, reused = _take(key, timeout)
        try:
            conn.request("POST", _path(url), body=body, headers=headers)
            resp = conn.getresponse()
        except (http.client.RemoteDisconnected, http.client.BadStatusLine, BrokenPipeError, ConnectionResetError):
            conn.close()
            if reused and attempt == 0:
                continue                      # the idle connection had timed out on the server: open a fresh one
            raise
        except BaseException:
            conn.close()
            raise
        if resp.status >= 400:
            data = resp.read()
            conn.close()
            raise urllib.error.HTTPError(url, resp.status, data[:200].decode("utf-8", "ignore"), resp.headers, None)
        return key, conn, resp
    raise ConnectionError("unreachable")


def _finish(key, conn, resp, complete: bool) -> None:
    if complete and not resp.will_close:
        try:
            resp.read()                       # http.client reuses a connection only once its response is closed
            resp.close()
        except (OSError, http.client.HTTPException):
            conn.close()
            return
        _give_back(key, conn)
    else:
        conn.close()


def post(url: str, body: bytes, headers: dict, timeout: float) -> bytes:
    """POST and return the whole response body. Raises OSError / HTTPException like urllib does."""
    key, conn, resp = _send(url, body, headers, timeout)
    try:
        data = resp.read()
    except BaseException:
        conn.close()
        raise
    _finish(key, conn, resp, True)
    return data


def post_lines(url: str, body: bytes, headers: dict, timeout: float):
    """POST and yield the response line by line (server-sent events) as it arrives."""
    key, conn, resp = _send(url, body, headers, timeout)
    complete = False
    try:
        while True:
            line = resp.readline()
            if not line:
                break
            yield line
        complete = True
    finally:
        _finish(key, conn, resp, complete)      # stopped early (caller interrupted): the connection cannot be reused
