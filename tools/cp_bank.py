"""Write the sentence banks for the checkpoint mirror and the stress probes (PLAN_V3 §3.2), blind to our system.

Gemini sees only the official category definitions from SCRIPTS.md (categories, the four time points, dialect
codes, fictional brands) and a short brief per category or probe. It never sees turn/lexicon.py, our tests or any
test-set sentence. The two stress banks are disjoint: the test bank is told not to reuse the dev bank's lines.

    python -m tools.cp_bank --out data/cp_bank.json            # mirror + stress-dev + stress-test
    python -m tools.cp_bank --only mirror --seed 7

These banks are for MEASUREMENT only: never copy their sentences into the lexicon, tests, teacher data or training.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from turn.config import load_env  # noqa: E402

URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODEL = "gemini-3.8-flash"

# ---------------------------------------------------------------- briefs (functions, not words)
MIRROR = {
    "GI": (6, "All 6 STOP. Vary the function: an objection, correcting a detail the agent got wrong, a question about "
              "what the agent just said, asking the agent to wait, a request to change something, and one that opens like "
              "an acknowledgement and turns into a request a moment later. One of them is a single firm word."),
    "BC": (6, "All 6 KEEP: the caller only shows they are listening. Vary them: one short token, one doubled token, one "
              "long agreement of 3-5 words with a falling tone, one dialect-specific form, one polite or blessing phrase, "
              "one agreeing with the content the agent just said."),
    "HT": (6, "5 KEEP during the agent's speech: a filled pause, thinking aloud to oneself, a half-started word that "
              "trails off, self-talk like 'let me see' not addressed to the agent, a question asked to oneself and "
              "answered by oneself. Plus 1 TAKE_TURN: the agent's last utterance is a question, the caller answers after it "
              "with pauses while searching (mark each pause as [pause N required] or [pause N optional])."),
    "BG": (6, "All 6 KEEP, and the caller stays silent (u1 = null). The sound comes from elsewhere: Arabic TV (MSA news "
              "or a series, put the TV line in 'tv'), another adult in the room talking to someone else, a child calling a "
              "parent, a person asking a question to someone in the room, street noise or a horn, cafe babble. For a person, "
              "put their words in 'side' and who they are in 'side_who'."),
    "NV": (6, "All 6 KEEP, the caller says no words (u1 = null). Sounds: a cough, a laugh at the agent's line, a sigh, a "
              "sneeze, a throat clear, and one speakerphone call where the agent's own voice echoes back (sound = echo)."),
    "AM": (6, "Ambiguous tokens («طيب», «خلاص», «أوكي», «إنزين» and dialect equivalents). 2 EITHER preferred KEEP: the token "
              "alone, falling, nothing after it. 2 STOP: the token, then the caller goes on and takes the floor (mark the "
              "word where it becomes clear). 2 EITHER preferred STOP: impatient «enough, I got it» while the agent "
              "repeats itself (mark the token as evident)."),
    "OV": (6, "Overlap. 3 STOP: the agent asked a question in its first utterance, waited, then starts a second utterance "
              "and the caller answers at the same moment (anchor_mode start, offset 0-150); the caller picks one of the "
              "options the agent is listing; the caller talks over the agent loudly and keeps going. 3 KEEP: the caller "
              "completes the agent's phrase softly by saying the agent's word with it; the caller repeats the word the agent "
              "is saying right now as support; a short acknowledgement right as the agent starts a new utterance."),
    "DV": (6, "Dialect variation: the same functions in non-Najdi speech. Use Hijazi, Southern, Eastern, Kuwaiti, Emirati, "
              "Bahraini or Qatari, each call a different one. 3 STOP (floor bids typical of that dialect) and 3 KEEP "
              "(acknowledgements typical of that dialect)."),
}
STRESS = {
    "S1": ("OV", 10, "Option answers. In the FIRST agent utterance the agent offers 2-3 concrete options (days, times, plans, "
                     "branches, delivery slots) and asks which one; it waits, then starts a SECOND utterance. 7 STOP: the "
                     "caller overlaps the second utterance by choosing one option, repeating the option's word (with or "
                     "without an acceptance word); mark the option word as evident. 3 KEEP controls: no question; the caller "
                     "softly repeats the word the agent is saying right now, only to show they follow (anchor_mode start "
                     "on that word, offset 80-150)."),
    "S2": ("GI", 10, "A pause after an acknowledgement. All 10 STOP: the caller starts with a short acknowledgement token, "
                     "pauses (write [pause N] with N between 150 and 400), then takes the floor: a change request, a choice, "
                     "an objection or a question. Mark the first word after the pause that makes the intent clear."),
    "S3": ("OV", 10, "Starting together. The caller starts speaking at almost the same moment as the agent's utterance. "
                     "7 STOP: an objection, correction or question of 3-6 words; mark the word where the intent is clear. "
                     "3 KEEP: a short acknowledgement. (We set the exact onset ourselves; use anchor = the first word of the "
                     "last agent utterance.)"),
    "S4": ("BG", 10, "Side talk with question and command words. 7 KEEP: the caller is silent (u1 = null); another person "
                     "close to the phone, as loud as the caller, talks to someone else in the room using questions, commands "
                     "or calling a name (where did you put..., look who is at the door, be quiet, what time is it, come "
                     "here); put it in 'side'. 3 STOP controls (category GI): the caller themself asks the agent a question "
                     "or gives a command about the call topic; mark the evident word."),
    "S5": ("AM", 10, "The functions of «طيب» and «خلاص» (and dialect forms like «إنزين», «أوكي»): acknowledgement (KEEP), "
                     "agreement at an agent pause with nothing after it (KEEP), a topic change or new question right after "
                     "the token (STOP), an objection (STOP), impatience with the token repeated fast (STOP), long thinking "
                     "token (KEEP), closing the call (STOP). Cover all seven functions at least once."),
    "S6": ("HT", 10, "Thinking aloud with question words. All 10 KEEP: while the agent talks, the caller thinks aloud to "
                     "themself using a question word (which one, what was it, where, how much) and then answers themself or "
                     "trails off; it is not addressed to the agent. Use … and fillers where natural."),
    "S7": ("BC", 10, "New backchannels. All 10 KEEP: acknowledgements and agreement in wording that is less common: blessings, "
                     "doubled tokens, agreement phrases, dialect forms, 1-5 words, falling tone, listening only."),
}

EXTRA = {   # banks added after Phase 1: hesitation variety (the mirror's weakest category)
    "S8": ("HT", 10, "Hesitation and thinking while the agent is still talking. 7 KEEP: a filled pause that goes on; a "
                     "half-started word that trails off; self-talk like 'let me see / let me remember' not addressed to the "
                     "agent; weighing aloud two options the agent just offered, to oneself, without choosing yet; recalling "
                     "a detail aloud; repeating a number or name the agent said while writing it down; a long thinking "
                     "«طيب…» or «يعني…». 3 STOP controls: after a short hesitation the caller clearly takes the floor with a "
                     "request or question to the agent (mark the evident word)."),
}

SCHEMA = """Return JSON: {"calls": [ ... ]}, one object per call, with exactly these keys:
- "use_case": short Arabic description of the call
- "dialect": one of sa-najdi, sa-hijazi, sa-southern, sa-eastern, sa-northern, gulf-kuwaiti, gulf-emirati, gulf-bahraini, gulf-qatari
- "gender": caller gender, "m" or "f"
- "a0": the agent's opener (greeting, name, fictional company, checks who it is talking to), ends with a question
- "u0": the caller's reply to a0 (1-5 words), in their dialect
- "a1": list of 1 or 2 agent utterances in Saudi "white" service dialect; the event happens during the LAST one, which has 22-40 words so the agent keeps talking at least 3 seconds after the caller's words
- "anchor": one exact word copied from the LAST agent utterance, in its first 60%, where the caller comes in
- "anchor_mode": "end" (caller starts after the agent finishes that word) or "start" (caller starts as the agent starts it)
- "offset_ms": integer -300..600, onset relative to the anchor
- "u1": the caller's words, or null when the caller says nothing. Mark the ONE word at whose end a native listener would say "they want the floor" with asterisks, like *كلمة* (only for STOP, and for EITHER when the token itself is the cue). Pauses as [pause 300]
- "expected": "STOP", "KEEP", "EITHER" or "TAKE_TURN"
- "preferred": for EITHER, "STOP" or "KEEP"; otherwise null
- "delivery": "soft", "normal", "loud" or "fast"
- "side": words said by another person in the room, or null
- "side_who": "child", "woman", "man" or null
- "sound": one of none, cough, laugh, sneeze, sigh, throat, echo, tv, horn, engine, babble, dog, baby
- "tv": an MSA TV line (news or a series) when sound = tv, else null
- "why": one English sentence: the human behaviour this call tests
Write natural, everyday Saudi and Gulf speech as people really talk on the phone. Every call must be different in wording.
Do not copy the example lines of the definitions."""


def scripts_context() -> str:
    """The official definitions only: §1 brands, §2 categories, §3 decisions and the four time points, §5 dialects."""
    s = (KIT / "SCRIPTS.md").read_text(encoding="utf-8")

    def section(start, end):
        a = s.find(start)
        b = s.find(end, a + 1)
        return s[a:b].strip() if a >= 0 else ""
    return "\n\n".join([section("### Brands and people", "---"), section("## 2. Categories", "## 3."),
                        section("### The four time points", "### Production"), section("### Dialect", "### Acoustic")])


def ask(sections: list[tuple[str, int, str]], seed: int, avoid: dict | None = None, retries: int = 2) -> dict[str, list[dict]]:
    """One request for several briefs (the free tier allows few requests a day). Returns {section: calls}."""
    key = os.environ["GEMINI_API_KEY"]
    system = ("You write test calls for an Arabic turn-taking benchmark: an AI phone agent talks, and the caller (Saudi "
              "or Gulf) sometimes takes the floor and sometimes only listens. You are a native speaker and an experienced "
              "script writer. Here are the official definitions:\n\n" + scripts_context())
    parts = []
    for name, n, brief in sections:
        part = f"### Section {name}: write {n} calls.\nBrief: {brief}"
        if avoid and avoid.get(name):
            part += "\nAnother writer already wrote these caller lines for this section. Do NOT reuse them or paraphrase them " \
                    "closely:\n" + "\n".join(f"- {x}" for x in avoid[name] if x)
        parts.append(part)
    user = ("\n\n".join(parts) + f"\n\nIn every section spread the dialects (at most 40% Najdi, include Hijazi and Gulf). "
            f"Variation seed: {seed}.\n\n" + SCHEMA +
            '\nAlso give each call a key "section" with the section name it belongs to, and write every section in full.')
    body = {"model": os.environ.get("CP_BANK_MODEL", MODEL), "temperature": 1.0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            out = json.load(urllib.request.urlopen(req, timeout=1200))
            text = out["choices"][0]["message"]["content"]
            text = re.sub(r"^```(json)?|```$", "", text.strip())
            calls = json.loads(text)["calls"]
            got: dict[str, list[dict]] = {}
            for c in calls:
                got.setdefault(str(c.get("section", "")).strip(), []).append(c)
            if all(len(got.get(name, [])) >= n - 1 for name, n, _ in sections):
                return {name: got[name][:n] for name, n, _ in sections}
            print("  incomplete:", {name: len(got.get(name, [])) for name, n, _ in sections}, flush=True)
        except Exception as e:  # noqa: BLE001  (network, quota, bad JSON)
            msg = str(e)
            print(f"  retry {attempt + 1}: {type(e).__name__} {msg[:120]}", flush=True)
            if "429" in msg:
                raise RuntimeError("quota") from e         # retries would only burn the daily quota
            time.sleep(5)
    raise RuntimeError("Gemini did not return a usable bank")


NT3 = {   # PLAN_V5 §4: the blind referee, hidden-set proportions (about 45 % STOP, 45 % KEEP, 10 % EITHER), 60 calls
    "GI": (8, "All STOP: objections, corrections, questions about what the agent said, asking it to wait, change requests, "
              "one opening like an acknowledgement then turning into a request, one single firm word."),
    "BC": (8, "All KEEP: listening only. Short, doubled, long agreeing, dialect forms, blessings, agreeing with the content."),
    "HT": (7, "6 KEEP while the agent talks (fillers, thinking aloud, self-talk, recalling, weighing options aloud, repeating a "
              "number while writing) and 1 TAKE_TURN (the agent asked a question; the caller answers with pauses, marked "
              "[pause N required] / [pause N optional])."),
    "BG": (7, "All KEEP, the caller silent (u1 = null): TV (put the line in 'tv'), people in the room talking to each other or "
              "calling someone (put it in 'side', with 'side_who'), street, cafe."),
    "NV": (7, "All KEEP, no caller words (u1 = null): cough, laugh, sigh, sneeze, throat clear, speakerphone echo."),
    "AM": (8, "Ambiguous tokens («طيب», «خلاص», «أوكي», «إنزين», dialect forms): 3 EITHER preferred KEEP (alone, falling), "
              "3 STOP (token then the caller takes the floor; mark the clear word), 2 EITHER preferred STOP (impatient)."),
    "OV": (8, "Overlap: 4 STOP (answering the agent's question as it starts again; picking an offered option; loud talk-over) "
              "and 4 KEEP (completing the agent's phrase, repeating its word as support, a quick acknowledgement at its start)."),
    "DV": (7, "Non-Najdi dialects, a different one each call: 4 STOP (floor bids in that dialect) and 3 KEEP (acknowledgements)."),
}


def ask_claude(sections, seed: int) -> dict[str, list[dict]]:
    """The same brief, written by Claude (claude -p) instead of Gemini: run in an empty folder with no tools, so it sees
    nothing of this repository."""
    import subprocess
    import tempfile
    name, n, brief = sections[0]
    prompt = ("You write test calls for an Arabic turn-taking benchmark: an AI phone agent talks, and the caller (Saudi or "
              "Gulf) sometimes takes the floor and sometimes only listens. You are a native speaker and an experienced script "
              "writer. Here are the official definitions:\n\n" + scripts_context() +
              f"\n\nWrite {n} calls for category {name}.\nBrief: {brief}\nSpread the dialects (at most 40% Najdi, include "
              f"Hijazi and Gulf). Variation seed: {seed}.\n\n" + SCHEMA +
              "\nAnswer with the JSON object only, no other text, and do not use any tools.")
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run(["claude", "-p", "--disallowedTools", "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task"],
                             input=prompt, capture_output=True, text=True, cwd=d, timeout=900)
    text = out.stdout.strip()
    text = text[text.find("{"): text.rfind("}") + 1]
    calls = json.loads(text)["calls"]
    return {name: calls[:n]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(KIT / "data" / "cp_bank.json"))
    ap.add_argument("--only", default="mirror,stress-dev,stress-test")
    ap.add_argument("--seed", type=int, default=20261119)
    a = ap.parse_args(argv)
    load_env()
    out = Path(a.out)
    bank = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    want = a.only.split(",")
    if "nt3" in want:                          # PLAN_V5 referee: one category per request, written by Claude
        for k, (n, brief) in NT3.items():
            if k in bank.get("nt3", {}):
                continue
            t0 = time.time()
            try:
                got = ask_claude([(k, n, brief)], a.seed + zlib.crc32(f"nt3/{k}".encode()) % 1000)
            except Exception as e:  # noqa: BLE001
                print(f"  nt3 {k}: failed ({type(e).__name__})", flush=True)
                continue
            for c in got[k]:
                c["category"], c["probe"] = k, None
            bank.setdefault("nt3", {})[k] = got[k]
            disk = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
            disk["nt3"] = bank["nt3"]
            out.write_text(json.dumps(disk, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  nt3 {k}: {len(got[k])} calls ({time.time() - t0:.0f}s)", flush=True)
    groups = {"mirror": [["GI", "BC", "HT", "BG"], ["NV", "AM", "OV", "DV"]],
              "stress": [["S1", "S2", "S3", "S4"], ["S5"], ["S6"], ["S7"]]}

    def save():
        """Merge into what is on disk: another run may have written batches meanwhile; never drop them."""
        disk = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        for name, d in bank.items():
            if isinstance(d, dict):
                disk.setdefault(name, {}).update(d)
        bank.update(disk)
        out.write_text(json.dumps(disk, ensure_ascii=False, indent=1), encoding="utf-8")

    def run(bank_name, keys, avoid=None):
        if all(k in bank.get(bank_name, {}) for k in keys):
            return                                                       # finished batches are kept
        t0 = time.time()
        if bank_name == "mirror":
            secs = [(k, MIRROR[k][0], MIRROR[k][1]) for k in keys]
        else:
            secs = [(k, STRESS[k][1], STRESS[k][2]) for k in keys]
        try:
            got = ask(secs, a.seed + zlib.crc32(f"{bank_name}/{'-'.join(keys)}".encode()) % 1000, avoid)
        except RuntimeError as e:
            print(f"  {bank_name} {keys}: gave up ({e})", flush=True)
            return
        for k, calls in got.items():
            for c in calls:
                c["category"] = k if bank_name == "mirror" else STRESS[k][0]
                c["probe"] = None if bank_name == "mirror" else k
            bank.setdefault(bank_name, {})[k] = calls
        save()                                                           # never lose a finished batch
        print(f"  {bank_name:11s} {keys}: {sum(len(v) for v in got.values())} calls ({time.time() - t0:.0f}s)", flush=True)

    if "ht-dev" in want or "ht-test" in want:
        for name in ("ht-dev", "ht-test"):
            if name not in want or "S8" in bank.get(name, {}):
                continue
            other = "ht-dev" if name == "ht-test" else "ht-test"           # the two banks stay disjoint either way
            avoid = {"S8": [c.get("u1") for c in bank.get(other, {}).get("S8", [])]} or None
            try:
                got = ask([("S8", EXTRA["S8"][1], EXTRA["S8"][2])], a.seed + zlib.crc32(name.encode()) % 1000, avoid)
            except RuntimeError as e:
                print(f"  {name}: gave up ({e})", flush=True)
                continue
            for c in got["S8"]:
                c["category"], c["probe"] = "HT", "S8"
            bank.setdefault(name, {})["S8"] = got["S8"]
            save()
            print(f"  {name}: {len(got['S8'])} calls", flush=True)
    with ThreadPoolExecutor(3) as ex:
        futs = []
        if "mirror" in want:
            futs += [ex.submit(run, "mirror", g) for g in groups["mirror"]]
        if "stress-dev" in want:
            futs += [ex.submit(run, "stress-dev", g) for g in groups["stress"]]
        for f in futs:
            f.result()
    if "stress-test" in want:                 # after the dev bank, so it can be told what not to reuse
        dev = bank.get("stress-dev", {})
        with ThreadPoolExecutor(2) as ex:
            for f in [ex.submit(run, "stress-test", g, {k: [c.get("u1") or c.get("side") for c in dev.get(k, [])] for k in g})
                      for g in groups["stress"]]:
                f.result()
    save()
    print("wrote", out, {k: sum(len(v) for v in d.values()) for k, d in bank.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
