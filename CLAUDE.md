# CLAUDE.md — instructions for Claude Code (Sarj turn-taking project)

Claude Code reads this file automatically whenever the project opens. It says **what we want, in what order, and when a task counts as done**. Read all of it before any change. If anything here conflicts with an explicit request from the user, the user's request wins.

## 0. Read first
1. `docs/PRD.md`: what we build and why, the current status, phases, and questions for the mentor.
2. `docs/TECH_SPEC.md`: how every module works, parameters, data partitioning, and gates G1–G7.
3. `INTERFACE.md`, `SCORING.md`, `DATA.md`, `SCRIPTS.md`: the official kit contract. These override everything.
4. `docs/EXPERIMENTS.md`: what has been tried, and what it scored.
5. **`docs/PLAN_V2.md`: the current plan (freeze, live-demo consistency and speed, persona "Tayf", the «طيب» work). Tasks T12–T24 below come from it. Start here.**
6. **`docs/PLAN_V3_SPEED.md`: reply speed without lower quality (demo only, branch `demo/day6`). S1–S5 are built; measure them, then S6–S12.**

## 1. Context in one paragraph
We are a student team in the Sarj AI hackathon. The harness streams a 2-channel call (agent + caller, 16 kHz, 20 ms frames) to `my_system.py` over WebSocket :8765. We decide only **STOP / RESUME / TAKE_TURN**. Full credit needs a STOP within 300 ms of the caller's intent becoming evident. Final rank = 80 % hidden human Arabic benchmark + 20 % judges.

Current scores (`runs/sim-history.csv` and `runs/final-dev-official`):

| dev | dev-aug | novel-test | blind novel-test-2 | real-time dev |
|---|---|---|---|---|
| 99.3 | 96.6 | **84.1** | **85.5** | 99.0 |

The gap is **generalisation**: lexicon errors on new phrasing, not ASR.

**Status (19 Nov):** T0–T10 are done and merged (`main` = `bf0ed81`). The live demo is fast on median but **inconsistent**: random long tails in stop latency and reply time, false stops on one-word acknowledgements, echo stops, and bad transcripts. Root causes RC1–RC7 are in `docs/PLAN_V2.md` §2.

**Code freeze: Day 5 at 18:00 (by our count, today, 19 Nov).** After the freeze tag `v2.0-freeze`, `turn/`, `my_system.py`, `models/` and `ttbench/` must stay byte-identical (`git diff v2.0-freeze -- turn my_system.py models ttbench` must be empty). Post-freeze work is demo-only, on branch `demo/day6`.

## 2. Rules (never break these)
1. **Never modify `ttbench/`.** It is the official harness.
2. **Measure every change under `turn/`** before you say it is done:
   ```bash
   python -m tools.loop --sets dev,dev-aug,novel-test,synth-test --note "<what changed>"
   ```
   Reject the change, or report it as a regression, if `dev-aug` or `novel-test` drops by more than 0.3, or false stops rise (gates G1–G7, TECH_SPEC §7.2).
3. **Never copy `novel-test` or `novel-test-2` sentences** into the lexicon, training data or tests. Fix the *category* of error instead. Open `novel-test-2` only once, when the user says so.
4. **Never delete files** without asking the user. Move them to `models/_archive/` or `_to_delete/` instead.
5. **No secrets in the repo.** API keys live only in `.env` (git-ignored). Never print a key, and never read `.env` aloud in output.
6. **No audio storage** from the checkpoint or final runs. Recordings in `data/live/` are git-ignored and need consent.
7. **Cross-platform.** Every change must run on Mac (Cohere/MLX) and Windows (faster-whisper). No hard-coded `~/Downloads/...`, and no Mac-only commands such as `open`.
8. **One heavy job at a time** (sim, training, generation). The laptop overloads otherwise.
9. **Git workflow:**
   - a new branch per task (`fix/...`, `exp/...`, `feat/...`);
   - small commits;
   - a PR into `main`;
   - **never push directly to `main`**;
   - never rewrite history;
   - after a measured improvement is merged, tag it: `git tag -a vX.Y -m "dev NN.N · dev-aug NN.N · novel NN.N"`, and update the score table in `README.md`.
10. **One module per function.** Fix a problem inside the module that owns it (see TECH_SPEC §4.1). Don't spread a fix across layers.
11. **The scored path stays local.** Cloud APIs are allowed only in the demo brain (`convo/`), data generation and offline analysis. They are optional, and they are declared in `WRITEUP.md`.
12. After each task, add an entry to `docs/EXPERIMENTS.md`: what changed, numbers before and after, decision.

## 3. How to work
- Do **one task at a time**, in the order below, unless the user picks one.
- At the start of each task:
  - create the branch;
  - restate the task's exit criteria;
  - run the baseline measurement if the task touches `turn/`.
- At the end:
  - show a before/after table;
  - say which gates passed;
  - open the PR (`gh pr create`), or say what is blocking.
- **Stop and ask** before anything irreversible: deleting, force-pushing, changing a threshold the gates depend on, or opening `novel-test-2`.
- If a fix fails twice, stop and explain. Don't keep trying random changes.

## 4. Task queue

T0–T10 are **done** (kept below for reference). The active queue is §4b.

| ID | Task | Phase | Exit criteria |
|---|---|---|---|
| **T0** | Decide `fix1` (uncommitted changes in `turn/lexicon.py`, `turn/policy.py`, `tools/synth_calls.py`). Measure stash vs applied on all 4 sets | P1 | Before/after table shown to the user; user decides keep or drop |
| **T1** | Cross-platform hygiene: (a) `tools/loop.py`: replace `open` with a platform-aware opener; add `--quick`; add `novel-test` to `SETS`. (b) `turn/config.py` with `TT_EXT_DIR`, replacing the hard-coded path in all 11 files (`grep -rn "sarj-ext-data"`). (c) `.env` loader + `.env.example`. (d) Share `novel-test` via a GitHub release `data-v1` plus a fetch step in `tools/fetch.py`. (e) Load `models/policy-{BACKEND}.json` for every backend, and train `policy-faster.json` | P1 | `pytest -q` green; `python -m tools.sim` works on Mac **and** Windows; tag `v1.0` |
| **T2** | Demo brain via the Gemini API: `TT_BRAIN=gemini` and `GEMINI_API_KEY` in `.env`, model `gemini-3.5-flash-lite`, handled in `convo/brain.py`. Add `--local-brain` to `setup.sh` so the 15 GB local model is not downloaded by default | P1 | Demo answers through Gemini; Claude Haiku takes over when Gemini fails |
| **T3** | Disk and models cleanup (**ask first**): mark the official models in `models/README.md`; move other policy and student variants to `models/_archive/` | P1 | User approved; `models/README.md` exists |
| **T4** | `tools/errors.py`: the 4 error types (false stop, missed, late, premature take-turn) × category × dialect × noise, with examples (call id, time, text, rule, `sem_p`) | P2 | Totals match `score.json` (9 false stops, 4 misses on `runs/sim-fix1-novel-test`) |
| **T5** | Generalisation (TECH_SPEC §5.7–5.8): split the lexicon into BID vs CUE; add the thinking-aloud rule; add teacher categories (openers in 14 dialects, thinking aloud, room talk) with **new** phrases; retrain the student and policy | P3 | `novel-test` ≥ 90, G1–G7 pass |
| **T6** | Early acoustic stop `EARLY_MS` (default 0) per TECH_SPEC §5.10. Also try `ASR_EVERY_MS=100` | P4 | Share of stops later than 300 ms lower, gates pass; otherwise left off |
| **T7** | Prosody for a lone ambiguous word («طيب»، «خلاص») | P4 | AM category up, no new false stops |
| **T8** | `novel-test-2` via `--bank data/novel_bank2.json` (blind bank). Measure **once** when the user says so | P5 | Result recorded; no tuning afterwards |
| **T9** | `WRITEUP.md` (2–4 pages) per TECH_SPEC §10: ablation, ≥ 5 failure cases with Arabic examples, declared services | P6 | Reviewed by the user |
| **T10** | `README.md`: setup for Mac and Windows, how to measure, the score table per tag | P6 | A new teammate can run `tools.sim` from README alone |
| T11 (optional) | Slow-brain RESUME confirmation `TT_SLOWBRAIN` (TECH_SPEC §5.11): measure the round trip first | — | Only if p90 RTT < 300 ms and the gates pass |

## 4b. Active queue (from `docs/PLAN_V2.md`; follow its section for the details)

**Track A: before the freeze.** Branch `fix/live-consistency`. Stop at 17:15 whatever is done; revert any task that fails a gate.

| ID | Task | Scored path? | Exit criteria |
|---|---|---|---|
| **T12** | `tools/replay_live.py`: replay `runs/live-calls/*/call.wav` (deduplicated) through the TurnTaker with real ASR timing and the demo echo gate; `--repeat 3`; report stop latency, ASR read + lock wait, single-backchannel stops, echo stops, decision agreement. PLAN §4 T12 | no | Baseline in `EXPERIMENTS.md` #48 |
| **T13** | Browser audio: `AudioContext({sampleRate:16000})`, no box filter; headset mode = AEC/AGC/NS **off** (fix the unused `aec` bug); speaker mode = AEC on, AGC off, plus a warning; send `getSettings()`; 3 s mic check sets U0. PLAN §4 T13 | no | No echo stops on headset replays; `pytest -q` green |
| **T14** | ASR priority lock (urgent interruption reads first, `wait_ms` logged); turn window 29 s → 12 s; reuse the TurnTaker transcript in `caller_finished()`; decoder cap 24 steps for urgent reads plus an early stop on repetition. PLAN §4 T14 | **yes** | Replay ASR p99 ≤ 300 ms, stop p99 ≤ 1000 ms; G1–G7 pass; real-time `dev` within 0.3 |
| **T15** | `lexicon.clean_asr()`: speaker labels anywhere, dash dialogue, repetition collapse, hallucination phrases, `fluent_msa` flag; used everywhere ASR text enters. PLAN §4 T15 | **yes** | `tests/test_clean_asr.py` with the real strings from PLAN §2 RC5; gates pass |
| **T16** | Promote guard: no "model is sure" stop for a single backchannel/filler/ambiguous/polite token, for hallucinated or `fluent_msa` text, or for echoed text. PLAN §4 T16 | **yes** | Replay shows 0 single-backchannel stops; gates pass (`novel-test` misses not higher) |
| **T17** | Preflight button: mic check, 5 warm ASR reads (p90), brain + voice time to first audio, thermal and free memory, headset reminder; saved per session. PLAN §4 T17 | no | Green on the demo laptop |
| **T18** | **Freeze** (17:15–18:00): PR merge, `ttbench.play --calls all`, `ttbench.ready`, paste the output into `SUBMISSION.md`, tag `v2.0-freeze` (ask before pushing), `tests/test_freeze_guard.py`. **Tell the user the hash.** PLAN §5 | — | The user has the hash before 18:00 |

**Track B: after the freeze, demo only.** Branch `demo/day6`. Never touch `turn/`, `my_system.py`, `models/`.

| ID | Task | Exit criteria |
|---|---|---|
| **T19** | Soak: 3× `ttbench.play --calls all` back to back on the frozen hash; RSS and ping log | No crash, all pings answered, RSS growth < 300 MB |
| **T20** | Update `WRITEUP.md` + `EXPERIMENTS.md` with Track A results and a short "Live demo engineering (demo-only)" section; list every external service | The user reads and approves it |
| **T21** | `tools/voice_audition.py`: same 3 lines × shortlisted voices (PLAN §6.2) × 2 presets on one A/B page. **Print the character cost and ask before rendering** | The team picks; `ELEVENLABS_VOICE_ID` set in `.env` |
| **T22** | Persona "Tayf" (`convo/persona.py`, PLAN §6.1) in all scenarios plus an energetic `GEMINI_STYLE`. ElevenLabs **WebSocket streaming** (flash v2.5 default; optional `eleven_v3_conversational` through the Text-to-Dialogue WebSocket); speculative brain start; component timing in the UI. PLAN §6.3–6.4 | Time to first audio p50 ≤ 1.0 s, p90 ≤ 1.8 s over 10 scripted calls |
| **T23** | `convo/vocab.py`: per-scenario entity vocabulary + fuzzy correction of demo transcripts, with unit tests; the brain is told transcripts may have errors. PLAN §6.5 | Tests pass; names come out right in 5 test calls |
| **T24** | «طيب» work (PLAN §7): demo "hold and duck" (−8 dB up to 400 ms, then decide on the continuation); `tools/ambig_word.py` features + LR, evaluated leave-one-speaker-out on the team recordings (Appendix B) | Beats the current rule on human recordings, otherwise stays off |

## 5. Commands
```bash
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m tools.sim                    # dev, fast
python -m tools.sim --data data/novel-test
python -m tools.loop --quick --note "..."
python -m ttbench.play --calls all     # official real-time run (~10 min, nothing else running)
python -m ttbench.play --data data/live --calls all
python -m ttbench.summary runs/<name> --failures
python -m ttbench.ready
pytest -q
```
```bash
python -m tools.replay_live --repeat 3 --note "<what changed>"   # live-demo consistency benchmark (after T12)
git diff v2.0-freeze -- turn my_system.py models ttbench           # must be empty after the freeze
```
Slash commands in `.claude/commands/`: `/measure`, `/task`, `/tag-version`. The source copies live in `claude-setup/`. If `.claude/` is missing, run `bash claude-setup/install.sh` once.

## 6. Starting prompt (for the user)
> Read CLAUDE.md and docs/PLAN_V2.md fully. It is Day 5; the code freeze is at 18:00. Create branch `fix/live-consistency` and do Track A in order (T12 → T17). After each task, show me the numbers. At 17:15 stop wherever you are and do T18 (freeze). Do not touch anything after the freeze tag except on `demo/day6`.
