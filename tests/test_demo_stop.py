"""PLAN_V4 Task B: after «✋ قاطعته هنا» the demo agent must be silent within 80 ms, whatever the chunk timing.

The playback AudioWorklet is taken out of each page and run in Node with a tiny stub of the Web Audio API.
Skipped when Node is not installed.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node") or str(Path.home() / ".local" / "node" / "bin" / "node")
PAGES = ["demo/index.html", "tools/live.html"]

HARNESS = r"""
globalThis.sampleRate = 48000;
let Proc = null;
class AudioWorkletProcessor { constructor() { this.msgs = []; this.port = {onmessage: null, postMessage: m => this.msgs.push(m)}; } }
function registerProcessor(name, cls) { Proc = cls; }
%SRC%
const BLOCK = 128, SR = 48000;
function make() { return new Proc(); }
function send(p, m) { p.port.onmessage({data: m}); }
function block(p) { const out = [new Float32Array(BLOCK)]; p.process([[new Float32Array(BLOCK)]], [out]); return Array.from(out[0]); }
function play(p, ms) { let o = []; const n = Math.ceil(ms / 1000 * SR / BLOCK); for (let i = 0; i < n; i++) o = o.concat(block(p)); return o; }
function chunk(sec) { return new Float32Array(Math.round(sec * 16000 * 3)).fill(0.5); }   // already at 48 kHz
function loudAfter(o, ms) { const k = Math.round(ms / 1000 * SR); return o.slice(k).some(x => Math.abs(x) > 1e-6); }
const res = {};
// (a) STOP in the middle of a chunk, two more queued
let p = make(); send(p, {start: 0}); ['a','b','c'].forEach((id, i) => send(p, {enqueue: chunk(0.2), id: i}));
play(p, 50); send(p, {stop: true}); res.mid_chunk = loudAfter(play(p, 600), 80);
// (b) STOP exactly between two chunks (nothing playing, the next one queued)
p = make(); send(p, {start: 0}); send(p, {enqueue: new Float32Array(BLOCK * 10).fill(0.5), id: 0}); play(p, 1000 * BLOCK * 10 / SR);
send(p, {enqueue: chunk(0.2), id: 1}); send(p, {enqueue: chunk(0.2), id: 2}); send(p, {stop: true});
res.between_chunks = loudAfter(play(p, 600), 80);
// (c) STOP, then 3 chunks that were already in flight arrive
p = make(); send(p, {start: 0}); send(p, {enqueue: chunk(0.2), id: 0}); play(p, 40); send(p, {stop: true});
[1, 2, 3].forEach(i => send(p, {enqueue: chunk(0.2), id: i})); res.in_flight = loudAfter(play(p, 600), 80);
// (d) chunks still in flight arrive 200 ms after the stop (after the fade has finished)
let q2 = make(); send(q2, {start: 0}); send(q2, {enqueue: chunk(0.2), id: 0}); play(q2, 40); send(q2, {stop: true}); play(q2, 200);
[1, 2].forEach(i => send(q2, {enqueue: chunk(0.2), id: i})); res.late_in_flight = play(q2, 400).some(x => Math.abs(x) > 1e-6);
// ...and a new reply after a new {start} plays again
send(p, {start: 1}); send(p, {enqueue: chunk(0.2), id: 9}); res.plays_after_start = play(p, 200).some(x => Math.abs(x) > 1e-6);
console.log(JSON.stringify(res));
"""


def worklet_source(page: str) -> str:
    html = (KIT / page).read_text(encoding="utf-8")
    m = re.search(r"const worklet = `(.*?)`;", html, re.S)
    assert m, f"no worklet in {page}"
    return m.group(1)


def run_worklet(src: str) -> dict:
    out = subprocess.run([NODE, "-e", HARNESS.replace("%SRC%", src)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not os.path.exists(NODE), reason="needs Node")
@pytest.mark.parametrize("page", PAGES)
def test_agent_is_silent_within_80_ms_after_stop(page):
    r = run_worklet(worklet_source(page))
    assert not r["mid_chunk"], "kept talking after a stop in the middle of a chunk"
    assert not r["between_chunks"], "kept talking after a stop between two chunks (B1)"
    assert not r["in_flight"], "played chunks that arrived during the fade (B2)"
    assert not r["late_in_flight"], "played chunks that arrived after the fade (B2)"
    assert r["plays_after_start"], "a new reply after the stop must play"


def test_engine_tags_audio_with_the_stop_epoch():
    src = (KIT / "convo" / "engine.py").read_text(encoding="utf-8")
    assert '"epoch": self.epoch' in src and "self.epoch += 1" in src and '"stop_vetoed"' in src
