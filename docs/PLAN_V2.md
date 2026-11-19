# PLAN_V2 — Freeze safely, then make the live demo fast, consistent, and give it a personality ("Tayf / طيف")

**Date:** 2026-11-19 · **Author:** planning pass (Claude, chat) · **Executor:** Claude Code in the terminal
**Reads with:** `CLAUDE.md` (rules + task queue), `docs/PRD.md`, `docs/TECH_SPEC.md`

---

## 0. Read this first: time and the code freeze

`CHALLENGE.md`: **"Day 5 — Code freeze at 18:00: hand in your commit hash and the list of external services. The final run uses exactly that code."**

By our count Day 3 (the checkpoint) was 17 Nov, so **Day 5 is today, 19 Nov, and the freeze is at 18:00**. The current candidate hash is `bf0ed81`, the head of `main`.

So the work has two tracks:

| Track | When | What may change | Goal |
|---|---|---|---|
| **A — Before freeze** | now → 17:15 | Only small, measured, low-risk changes. Every change under `turn/` must pass gates G1–G7 | Fix the root causes of the live-demo inconsistency that live in shared code, then freeze a **new hash** |
| **F — Freeze** | 17:15 → 18:00 | Nothing new | Full gates, real-time `dev`, `ttbench.ready`, tag `v2.0-freeze`, hand in the hash |
| **B — After freeze (demo only)** | after 18:00 → Day 6 | Only the demo path: `convo/`, `tools/live.py`, `tools/live.html`, `demo/`. **Never `turn/`, `my_system.py` or `models/`** | Persona "Tayf", voice, reply speed, polish for the judges' live demo |

> **Ask the mentor today, before 18:00** (the user does this, not Claude Code):
> "The final run uses our frozen hash. May we keep improving the *demo app only* (voice, persona, UI; no change to the decision code) on a separate branch for the Day-6 live demo, and declare it?"
> - If the answer is **no**, Track B must be finished before 18:00 too, or dropped.
> - If there is **no answer**, do Track B on branch `demo/day6`. Keep `turn/` byte-identical to the frozen hash, check it with `git diff v2.0-freeze -- turn my_system.py models` (must be empty), and say so in `WRITEUP.md`.

If the team decides the freeze is on another day, the order of work stays the same; only the deadlines move.

---

## 1. Where we are (verified 19 Nov, 14:00)

**Repo:**
- `main` = `origin/main` = `bf0ed81` ("Merge PR #11 docs/final-writeup").
- 11 PRs merged (#1–#11). T0–T10 are done; T11 (cloud RESUME check) was not built.
- Untracked, and must stay untracked: `data/novel-test*/`, `data/novel_bank*.json`, `data/eleven_plan.json`, `Claude outputs/`.

**Scores** (`runs/sim-history.csv`, `docs/EXPERIMENTS.md` #33–#47):

| Set | Score | Note |
|---|---|---|
| Official real-time run on `dev` | **99.0** | 0 false stops, 0 misses |
| `dev` (sim) | 99.3 | |
| `dev-aug` | 96.6 | |
| `novel-test` | 84.07 | target was 90, missed |
| `novel-test-2` (blind, opened once) | **85.46** | 14 false stops, 4 misses. Weakest categories: thinking aloud (S 0.33), ambiguous words (R 0.43) |
| `synth-test` | 99.5 | |

⚠️ The last row of `sim-history.csv` is `faster-dev, dev, 8.0` (0 stops, 7 misses, latency `None`). That is a **broken faster-whisper run**: the ASR did not load, so nothing ever stopped. It is not a real score. Do not quote it. Do not claim Windows works until a teammate's Windows run scores normally.

**Still open from the last session:**

| Item | Owner | Status |
|---|---|---|
| Hand in commit hash + external services list by Day 5 18:00 | User | **Today.** The hash will change after Track A |
| Read and approve `WRITEUP.md` | User | Claude Code updates it in task T20, then the user reads it |
| Slides, 7–8 min talk, 2 rehearsals, backup demo video | Team | Demo script in §9 |
| 30-min stability (soak) test | Claude Code | Task T19, after freeze, laptop otherwise idle |
| 20–30 consented human recordings | Team | Script in Appendix B; also feeds the «طيب» work |
| Windows run, then tag `v1.0` | Teammate | See the 8.0 warning above |
| Answers to the mentor questions in `docs/PRD.md` §14 | User | Add the demo-only question above |
| `.env.example` | **Done here** | Was blocked by our own permission rule `Read(.env.*)`; rule fixed in `claude-setup/settings.json` |

---

## 2. What the live demo actually does wrong: evidence from 108 saved demo calls

Source: `runs/live-calls/*/trace.json` + `call.wav` (108 calls ≥ 8 s, 17–18 Nov). Each session is saved twice, so there are duplicate rows. Numbers are from all saved reads.

| Metric (demo, laptop mic) | p50 | p90 | p99 | max |
|---|---|---|---|---|
| ASR read during agent speech (ms of compute) | 106 | 215 | **506** | 980 |
| ASR full-turn read (ms) | 171 | 415 | **1067** | 1274 |
| Stop latency (from candidate start, ms) | 440 | **1180** | **4830** | 5040 |
| Reply: caller end → agent audio (ms) | 1394 | **4537** | **9799** | 10414 |
| Caller level (p95 dBFS per call) | ranges −15 … −31 dB across calls | | | |
| Frames flagged "agent echo only" by the demo's echo gate | up to ~50 % of a call | | | |

The fast median hides a **long, random tail**: good and bad sessions come from the same code. The causes, ranked by impact:

### RC1. The browser mangles the microphone, on every run (bug)
`tools/live.html` `ensureAudio()` always calls:

```js
getUserMedia({audio:{echoCancellation:true, noiseSuppression:false, autoGainControl:true}})
```

The variable `aec = mode()==='spk'` is computed but **never used**, so even in "headphones" mode Chrome's AEC3 and AGC are on.

- **AEC3 attenuates the caller during double-talk**, which is exactly the moment of an interruption. Our own EXPERIMENTS #23 measured AEC3 removing **25–32 dB of the caller's voice while interrupting**, and rejected it server-side. The browser re-enabled it.
  - Result: interruption words arrive weak and garbled, so there is no stop or a late stop, and the transcripts are bad.
- **AGC** changes mic gain during the call.
  - Our level gates are *relative to the caller's first answer (U0)*: `STOP_REL_DB = 8`, `REL_DB = 10`.
  - When AGC moves the gain by 6–10 dB, the same voice passes in one run and fails in the next. This is the "سريع مرة وما يوقف مرة" symptom.
- Resampling in the AudioWorklet is a 3-tap box filter (48 kHz → 16 kHz), with weak anti-aliasing that slightly hurts ASR.

### RC2. Speakers instead of a headset → the agent hears itself
- **Evidence of false stops on the agent's own words**, transcribed from the speaker: «حياك الله حبيت أذكرك إن عندك دفعة بقيمة 3850 ريال» (p=1.00), «السلام عليكم. نعم. معك ريم من».
- The demo's echo gate flags up to half the frames as echo. That also **suppresses real interruptions**, which must be ≥ 9 dB above the echo.

### RC3. One ASR lock, three customers → priority inversion
`turn/asr.py` has a single `_LOCK`. Three things share it:
- (a) interruption reads (short, urgent);
- (b) the TurnTaker's own "turn" reads, up to 29 s of audio, repeated as the turn grows (we saw 10+ reads of one long turn);
- (c) `convo/engine.py::caller_finished()`, which **re-transcribes the whole turn again**.

So an urgent interruption read can wait behind a 1-second full-turn read. That explains the stop-latency tail (p99 4.8 s) and part of the reply latency.

### RC4. One-word acknowledgements trigger the learned "promote" path in live use
Live false stops with reason `"model is sure (p≥0.95) although the words are unclear"`: «أيوه», «أوكي», «أه», «ها», «شكراً», «أنا», «كان», «سلامتك», «تمثيل»…

On the synthetic sets the promote rule was safe (EXPERIMENTS #13). With browser AGC/AEC the acoustic features shift, and the LR model saturates. In `turn/policy.py` line ~299, promote fires with `n >= 1` and no lexical check.

### RC5. Cohere hallucinations that our filters miss
Cohere "is eager to transcribe, even non-speech sounds" (model card). Patterns seen in demo transcripts:

| Pattern | Examples seen |
|---|---|
| Speaker labels not at the start | `رتون:`, `رغد:`, `علاء:`, `- بدر: بدر: بدر:` |
| Dash-dialogue format | `- … - …` (only a leading label is stripped today) |
| Repetition loops up to the 48-token cap | `هههههه…`, `!!!!!!`, `؟؟؟؟`, `آآآآ`, `سنيييي`. These are also the **slowest** reads, because they run all 48 decode steps |
| MSA / YouTube / subtitle hallucinations on unclear audio | «شكراً جزيلاً على المشاهدة», «تصفيق», long fluent MSA sentences nobody said |
| Names misheard (Cohere has no keyword prompt) | «معاك الأستاذ سعود» heard as «مايكل ستارت», «ماك سالسود», «معاك للسادة يا سعود», «معاي كريستال سعود» |

### RC6. The reply path has no latency budget
EOT wait (600 ms, up to 2500) + a second full ASR (RC3) + brain + TTS over the network. Gemini live voice is used in most calls, so the network adds variance.
- ElevenLabs is called as **one HTTP POST per sentence** (`convo/voice.py::_eleven`): no streaming, no voice settings.
- The voice is "the first voice of the right gender in the account" when `ELEVENLABS_VOICE_ID` is unset. That is usually a generic premade voice reading Arabic, which explains the flat, "dead" sound.
- The Gemini style prompt literally asks for «نبرة دافئة **وهادئة** وسرعة كلام **عادية**».

### RC7. Environment drift
Sessions run with the browser, IDE and other apps on the same Mac, which shares the GPU with MLX. There is no preflight check, so a slow machine state is only discovered mid-demo.

---

## 3. Targets (how we will know it is fixed)

Measured with the new **live replay harness** (T12). It re-plays recorded demo calls through the real TurnTaker with real ASR timing, 3 repeats each, so human variability is removed and only system variance remains.

| Target | Now | Goal |
|---|---|---|
| Same decision on 3 repeats of the same call (consistency) | not measured | **≥ 97 %** of candidates |
| Stop latency p90 (replay, from candidate start) | 1180 ms | **≤ 600 ms** |
| Stop latency p99 | 4830 ms | **≤ 1000 ms** |
| False stops on a single backchannel word (أيوه/أوكي/أه/إيه/مم/نعم/شكراً/ها) | many | **0** |
| False stops on the agent's own words (headset mode) | several | **0** |
| ASR read p99 | 506 ms | **≤ 300 ms** |
| Reply time to first audio (demo) p50 / p90 | 1.39 s / 4.54 s | **≤ 1.0 s / ≤ 1.8 s** |
| Scored sets (G1–G7) | as in §1 | **no regression**: `dev` ≥ 99.0 real-time, `dev-aug` ≥ 96.3, `novel-test` ≥ 83.8, false stops not higher |

---

## 4. Track A — before the freeze (do in this order; stop at 17:15 whatever is done)

All on branch `fix/live-consistency`. Commit after each task. **Run `/measure` after every change under `turn/`.**

### T12 — Live replay harness (measurement first, ~30 min)
New file `tools/replay_live.py`:
- **Input:** `runs/live-calls/*/call.wav`. Deduplicate sessions by audio hash, because each call is saved twice.
- Feed frames through `policy.TurnTaker` exactly like `convo/engine.py::on_frame` does, including the demo `echo_gate`. Run ASR in a real background thread so timing is real (do **not** set `TT_SIM`).
- **Output:** `runs/replay-live-<note>/report.json` + a printed table with:
  - per call: candidates, STOPs (with text + reason), RESUMEs;
  - stop latency p50/p90/p99;
  - ASR read ms p50/p90/p99 including **lock wait** (measure wait separately);
  - STOPs whose text is a single backchannel token;
  - STOPs whose text matches the agent's own text (echo).
- `--repeat 3` runs each call 3× and reports the **decision agreement rate**.
- Privacy: read-only on these files. Never copy audio into git; `runs/` stays git-ignored.

**Exit:** baseline numbers recorded in `docs/EXPERIMENTS.md` (#48).

### T13 — Browser audio: raw mic, real resampling, headset by default (demo only, ~20 min)
In `tools/live.html` (and `demo/index.html` if it captures audio):
1. Create the context with `new AudioContext({sampleRate: 16000, latencyHint: 'interactive'})` and remove the box-filter decimation in the worklet (ratio becomes 1). Chrome's resampler is much better. Keep a fallback to the current path if the browser refuses 16 kHz.
2. getUserMedia constraints by mode:
   - **headset (default):** `{echoCancellation:false, noiseSuppression:false, autoGainControl:false, channelCount:1}`
   - **speakers:** `{echoCancellation:true, noiseSuppression:false, autoGainControl:false}`, plus a visible warning: «السماعة الخارجية تقلل دقة المقاطعة — استخدم سماعة رأس».
3. Read back `track.getSettings()` and send it to the server in `convo_start`, so every saved session records the real mic settings (needed to explain differences).
4. **Mic check** before a call: 3 s of the user saying «هلا، معك سعود».
   - Record the level and set U0 from it.
   - Warn if the level is < −40 dBFS or clipping.

**Exit:** a replay of 3 headset calls shows no echo stops; a manual call works. These files are not under `turn/`, so the gates don't apply, but run `pytest -q`.

### T14 — ASR scheduling: urgent reads first, no duplicate work (~45 min)
`turn/asr.py` + `convo/engine.py` + `turn/policy.py` (scored path, so gates are mandatory).
1. Replace the plain `_LOCK` with a **priority lock**: `transcribe(audio, priority="urgent"|"normal")`.
   - Interruption/candidate reads (`"cand"`, `"final"`) are urgent.
   - Turn reads and convo reads are normal.
   - A waiting urgent request is served before any waiting normal one.
   - MLX cannot run two decodes at once, so this is an ordering change, not parallelism.
   - Record `wait_ms` in the result.
2. **Turn reads:** in `policy.py` (~line 419), cap the window at **12 s**, not 29 s.
   - Phone turns are short.
   - A long turn is already covered by earlier reads, and the brain gets the joined text.
   - Measure that `dev` EOT and Core are unchanged.
3. **No second transcription.** In `convo/engine.py::caller_finished()`:
   - If `self.tt.turn` has a text whose last read ended within 400 ms of the end of speech, use that text.
   - Only re-transcribe when the last read is stale.
   - Run any re-read with `priority="normal"`.
4. Decoder caps in `_cohere()`:
   - `max_steps = 24` for urgent reads, 48 for normal.
   - **Early stop on repetition:** stop when the last 6 tokens repeat a 1–3-token pattern, or when one character repeats more than 4 times.

**Exit:** replay shows ASR read p99 ≤ 300 ms and stop p99 ≤ 1000 ms. `/measure` gates G1–G7 pass. Official real-time `dev` is unchanged within 0.3.

### T15 — ASR text sanitizer (~30 min)
`turn/lexicon.py` (scored path, gates mandatory). Add `clean_asr(text) -> (text, flags)` and use it everywhere ASR text enters (policy and convo):
- Strip speaker labels **anywhere**, not only at the start: `(^|\s|-)\S{1,12}\s?:` when followed by text.
- Drop dash-dialogue markers.
- Collapse runs: any character repeated more than 3 times → 2; any token repeated more than 3 times in a row → 2; strip `!!!`, `؟؟؟`.
- **Hallucination phrase list** (normalised match, whole-utterance or suffix): «شكرا على المشاهدة», «ترجمة نانسي قنقر», «اشتركوا في القناة», «تصفيق», «موسيقى».
  - If the whole text is one of these → treat as no speech.
- Flag `fluent_msa` when the text has ≥ 6 words, no Gulf markers, and MSA markers such as «إنها», «سأعود», «لماذا», «هل أنت». Use it only as a **veto feature for promote** (T16), never to force a stop.

**Exit:** unit tests with the real strings from §2 RC5 (create `tests/test_clean_asr.py`). Gates pass.

### T16 — Promote guard for one-word acknowledgements (~20 min)
`turn/policy.py` ~line 299 (scored path, gates mandatory). Promote ("model is sure although the words are unclear") is **not allowed** when:
- the cleaned text is a single token in `BC ∪ FILLER ∪ AMBIG ∪ POLITE ∪ {أنا, كان, ها, أوكي, اوكي, شكرا, نعم}`; or
- the text is flagged `fluent_msa` or a hallucination by T15; or
- the text is echoed from the agent (`echo_ratio ≥ 0.5`).

Real bid words still stop through the rules. Re-check that `PROMOTE_REL_DB` uses the U0 from the mic check (T13).

**Exit:** replay shows 0 single-backchannel STOPs. Gates pass: in particular `novel-test` misses are not higher, because promote also catches some real interruptions.

### T17 — Preflight page + latency probe (demo only, ~20 min)
Add a «فحص قبل العرض» button to the live page. It runs:
- the mic check (T13);
- 5 warm ASR reads on stored speech, with p90 shown (red if > 250 ms);
- semantic student warm;
- brain and voice round trip (time to first audio; red if > 1.5 s);
- `pmset -g therm` and free memory (red if under 3 GB free);
- a reminder that the headset is connected.

The result is saved into each session folder.

### Track A acceptance (17:00)
`/measure "track A"`, then:

```bash
python -m tools.replay_live --repeat 3 --note trackA
python -m ttbench.play --calls all
pytest -q
```

If any gate fails, **revert that task** (`git revert`) and freeze without it. A smaller, safe freeze beats a risky one.

---

## 5. F — Freeze (17:15 → 18:00)
1. Merge `fix/live-consistency` via PR. The real-time `dev` result is in the PR description.
2. `python -m ttbench.ready` prints the hash + external services. Copy that output into `SUBMISSION.md`/`WRITEUP.md`.
3. `git tag -a v2.0-freeze -m "dev RT NN.N · dev-aug NN.N · novel NN.N · novel-2 85.46"` and push the tag (ask the user first).
4. **The user hands in the hash before 18:00.**
5. From now on, `git diff v2.0-freeze -- turn my_system.py models ttbench` must stay empty on every demo branch. Add a test `tests/test_freeze_guard.py` that fails if it is not, active when the env `FREEZE_TAG` is set.

---

## 6. Track B — persona "Tayf / طيف" and a voice with energy (demo only, branch `demo/day6`)

### 6.1 Persona spec (put in `convo/persona.py`; used by every scenario)
- **Name:** طيف (Tayf). The agent introduces itself as «معك طيف من {company}».
  - Grammatical gender comes from `TAYF_GENDER` (`f` default, because «طيف» is mostly a female name in Saudi). Switch to `m` if the team picks a male voice.
  - Keep `as_male()` working.
- **Character:**
  - Energetic, warm, confident, quick.
  - A young Saudi professional who enjoys helping.
  - Light humour only when the caller jokes first.
  - Never sarcastic, never pushy.
- **Dialect:** white Saudi / Najdi. Avoid MSA, and avoid other dialects («ديال», «مش»).
- **Speech rules (these also make it faster):**
  1. Answer first, then explain. ≤ 12 words per sentence. One idea per sentence.
  2. Use natural energisers, sparingly (max one per reply): «أبشر», «ولا يهمك», «تمام», «على راسي», «حلو».
  3. Numbers the way people say them: «ثلاث مية وخمسين ريال», «يوم الخميس الجاي».
  4. After an interruption: acknowledge in 1–3 words («إيه تفضل», «أكيد»), answer, then return with «وزي ما كنت أقول…» only if the point still matters.
  5. If the caller sounds annoyed or says «خلاص»: shorten, offer the one next step, and close politely.
  6. Never invent facts. If unsure, say «بتأكد لك وأرجع لك».
  7. No emojis or symbols that TTS would read aloud.
- **Energy markup:**
  - For `eleven_v3_conversational`: allow at most one audio tag per reply, from `[excited]`, `[warmly]`, `[laughs softly]`, and only at sentence start.
  - For flash models: no tags. Energy comes from short sentences and an occasional «!».
  - The brain prompt states which mode is active, so it never writes tags for a model that would read them aloud.
- Replace `GEMINI_STYLE` with an energetic version: «طيف: موظفة خدمة عملاء سعودية شابة، لهجة بيضاء، نبرة حماسية ودودة وواثقة، سرعة كلام فوق المتوسط بشوي، ابتسامة في الصوت». Use the male form if `TAYF_GENDER=m`.

### 6.2 Voice: choose by ear, in 10 minutes
ElevenLabs Voice Library candidates (native Saudi, conversational; verified 19 Nov):

| Voice | voice_id | Why |
|---|---|---|
| ALI – Saudi Arabic & English (m) | `Hvlnv5DwiIO2CQ6oYMZ3` | Described as expressive, "shifts between energetic, thoughtful, humorous" |
| Hasawi – Conversational and Energize (m) | `kr4VZw8MSZMHE0y2m40n` | Energetic, Eastern Province warmth |
| Bandar (m, young) | `aLv8a8fbjjLsORNUDQ1J` | Young, warm, light-hearted |
| Ziyad – Natural Saudi, Najdi (m) | `pNGUEZc2WLum9bjP611g` | Authentic Najdi, confident |
| Fahad – Saudi Professional & Warm (m) | `YCN2uLvOoKxKJtRmeub3` | White Saudi, built for AI agents |
| Hana – Casual & Relatable (f, Hijazi) | `kdUY91gH5xyDHapxlthT` | Female, upbeat-casual |
| Abir – Arabic female (f, Saudi) | `VcSFj5ljnKA0aQcxeHpl` | Female option |

There are few energetic **female** Saudi voices in the library. If the team wants a female Tayf and none of the female voices fits, use **Voice Design**, which costs credits; the user approves first. Prompt:

> "A young Saudi woman in her mid-twenties, white Najdi Saudi dialect, bright, energetic and warm customer-service voice with a smile, confident and quick but clear, close-mic phone quality, no reverb."

Sample line: «هلا والله! معك طيف من متجر نوّار، حبيت أذكرك بدفعة بسيطة، دقيقة من وقتك؟»

**T21 — Voice audition page:** add `tools/voice_audition.py`.
- Render the same 3 lines (greeting, a number sentence, an apology after interruption) for each shortlisted voice.
- Use 2 settings presets: **lively** (`stability 0.35, similarity 0.8, style 0.35, speed 1.06, speaker_boost true`) and **steady** (`0.5, 0.8, 0.15, 1.0`).
- Show them on one page with A/B buttons.
- Cost is about 1.5k characters. Print the estimate and ask before rendering.
- The team picks one; save it in `.env` as `ELEVENLABS_VOICE_ID`.

### 6.3 Streaming TTS (fixes the "slow and flat" feel)
- Replace the per-sentence POST in `convo/voice.py::_eleven` with the **WebSocket stream-input** endpoint:
  - one connection per call, kept open, which reuses TLS;
  - `model_id = eleven_flash_v2_5` (~100 ms model latency, Arabic supported);
  - `output_format = pcm_24000`.
- Send brain sentences as they arrive. Forward audio chunks to the browser as the existing `pcm` messages.
- On barge-in: close the context/socket and open a new one in the background. ElevenLabs has no cancel message.
- **Optional expressive mode:**
  - `ELEVENLABS_MODEL=eleven_v3_conversational` through the **Text-to-Dialogue WebSocket** (`/v1/text-to-dialogue/stream-input`, one registered voice, ~165 ms model latency, audio tags supported).
  - On this path only `stability` applies; lower stability follows the tags more.
  - A/B it against flash in the audition page. Keep flash as the default if v3 time to first audio p90 is > 1.2 s.
- Pre-render the opening lines and fillers for the chosen voice and settings (cache already exists in `convo/voice.py`).
- Keep Gemini live voice and Edge as fallbacks. If ElevenLabs time to first audio is > 1.5 s twice in a call, switch to the fallback for the rest of the call and log it.

### 6.4 Reply speed (T22)
1. Reuse the TurnTaker transcript (from T14), saving 170–1000 ms.
2. **Speculative start:**
   - When silence reaches 300 ms and the student or lexicon says the sentence is complete (not ending in a filler or connective), start the brain request, but hold the audio.
   - At `TAKE_TURN` (600 ms), play immediately.
   - If the caller speaks again before 600 ms, cancel the request.
3. Keep the filler at 0.9 s, using the persona's fillers in the chosen voice.
4. One warm connection each to the brain and the TTS for the whole call.
5. Log time to first audio per reply with components: EOT wait, ASR, brain first token, TTS first chunk. Show the components in the demo page.

**Exit:** p50 ≤ 1.0 s and p90 ≤ 1.8 s over 10 scripted calls.

### 6.5 Key words and names (T23)
- `convo/vocab.py`: per scenario, the expected entities: agent name «طيف», caller name (from the mic-check sentence), company, amounts, days, branch, and app names.
- After ASR (demo transcripts only), fuzzy-correct tokens against this vocabulary:
  - normalise alef/ya/ta-marbuta/hamza;
  - character-level edit distance ≤ 30 % of the length, or a Soundex-like Arabic consonant skeleton match;
  - only replace when the score clearly wins.
- Examples to unit-test: «ماك سالسود» → «معك سعود», «مايكل ستارت» → *no change* (too far), «كم القيادة» → «كم القيمة».
- Tell the brain in the system prompt that transcripts can contain errors, and that it should ask a short clarifying question when an important entity is unclear.

---

## 7. Track C — «طيب» and other ambiguous words (data now, model after measurement)

Why it is hard: «طيب» is an acknowledgement, an agreement, a topic-change opener («طيب كم المبلغ؟»), an objection opener («طيب بس…»), impatience («طيب طيب»), and a closing («طيب مع السلامة»). The **word alone does not decide it**. Our T7 experiment (#43) used pitch slope (AUC 0.58) and effort (0.72) on synthetic data, and failed.

What actually separates them, from strongest to weakest:
1. **Continuation:** does the caller keep talking within ~350 ms after «طيب»? If yes, it is a floor bid (decide on what follows).
2. **Agent context:** did the agent just ask a yes/no question or pause at a clause boundary? Then «طيب» is an answer or acknowledgement. Mid-clause overlap leans towards a bid.
3. **Lengthening and repetition:** «طيييب» (thinking) vs fast «طيب طيب» (impatience, often a bid).
4. **Prosody relative to U0:** rising pitch, higher effort.

**Policy for the demo** (Track B; for the scored path only after the final):
- **"Hold and duck."** On a lone «طيب/خلاص/زين» during agent speech:
  - lower the agent volume by 8 dB for up to 400 ms (a browser gain change, not a STOP);
  - if speech continues → normal decision on the continuation;
  - if not → restore the volume and carry on.
- To the user this feels like a polite person who notices the caller without stopping dead.

**Data (T24, team + Claude Code):**
- Recording script (Appendix B): each teammate records «طيب» in the 6 functions × 3 prosodies, over an agent clip, with a headset. That is about 20 clips per person.
- Optional prosody variety: ElevenLabs `eleven_v3` with tags (`[hesitant]`, `[impatient]`, `[agreeing]`) in 3 Saudi voices.
  - Train/test use only. Human recordings are the test set; synthetic clips are training only.
- Label with the existing `tools/live.py` save flow, which uses the kit format.
- **Model:** a small logistic regression over `continuation_ms, agent_question, agent_boundary, dur_ms, lengthening, repeat, f0_slope_rel, effort_rel`, evaluated **leave-one-speaker-out**.
  - Only if it beats the current rule on human data does it go into the demo.
  - The scored path does not change after the freeze.

---

## 8. Track B/C acceptance and the soak test
- **T19 Soak** (after freeze, laptop otherwise idle):
  - Run `python -m ttbench.play --calls all` three times back to back (~30 min) against the frozen hash.
  - Log RSS memory every 30 s and every `ping`/`pong`.
  - **Pass:** no crash, every ping answered, RSS growth < 300 MB.
  - Record the result in `EXPERIMENTS.md`.
- **Scripted demo replay:** 10 calls from `runs/live-calls` + 10 new headset calls following §9.
  - Report the §3 targets.
  - Results go in `EXPERIMENTS.md` (#49+) and in a new short section of `WRITEUP.md`, "Live demo engineering", which states clearly that it is demo-only.

---

## 9. Day-6 live demo script (for the presenter; ~2 min)
Headset on, preflight green, scenario "payment".
1. Agent (Tayf) greets → presenter: «هلا، معك سعود» (sets U0).
2. While the agent explains: «إيه» … «مم» → it **keeps talking** (backchannel).
3. Play a TV clip from the phone near the mic → it **ignores** it (another voice, quieter).
4. «طيب» alone → **ducks and continues**. Then «طيب بس كم المبلغ؟» → **stops**, answers the amount, and returns with «وزي ما كنت أقول…».
5. «لا لا لحظة» → **stops within ~0.3 s**; the latency is shown on screen.
6. Close: «خلاص تمام، مع السلامة» → a short energetic goodbye.

Have the backup video ready, recorded on the same branch.

---

## 10. Risks
| Risk | Mitigation |
|---|---|
| A Track A change lowers the scored numbers | Gates per task; revert the single task; freeze the smaller set |
| Demo changes after the freeze are not allowed | Ask the mentor now; keep `turn/` identical (freeze-guard test); declare it |
| ElevenLabs latency or credits run out on demo day | Pre-render fixed lines; fallback to Gemini live, then Edge; preflight shows remaining credits |
| Headset not available | Speaker mode keeps AEC on (warned); expect lower interruption accuracy |
| Over-tuning to our own 108 calls | They are a **consistency/latency** benchmark only. Decision quality is still judged on `novel-test-2` and human recordings |

---

## Appendix A — files and owners
| Task | Files | Scored path? |
|---|---|---|
| T12 replay harness | `tools/replay_live.py` (new) | no |
| T13 browser audio | `tools/live.html`, `demo/index.html` | no |
| T14 ASR scheduling | `turn/asr.py`, `turn/policy.py`, `convo/engine.py` | **yes** |
| T15 sanitizer | `turn/lexicon.py`, `tests/test_clean_asr.py` | **yes** |
| T16 promote guard | `turn/policy.py` | **yes** |
| T17 preflight | `tools/live.py`, `tools/live.html` | no |
| T18 freeze | tags, `SUBMISSION.md`, `tests/test_freeze_guard.py` | — |
| T19 soak | `tools/soak.py` (new, optional wrapper) | no |
| T20 write-up | `WRITEUP.md`, `docs/EXPERIMENTS.md` | — |
| T21 voice audition | `tools/voice_audition.py` (new) | no |
| T22 reply speed + streaming TTS | `convo/voice.py`, `convo/engine.py` | no |
| T23 vocab correction | `convo/vocab.py` (new) | no |
| T24 «طيب» data + model | `tools/live.py` flow, `tools/ambig_word.py` (new) | no (demo only until after the final) |

## Appendix B — recording script for teammates (consented, headset, quiet + one noisy room)
For each line: play an agent clip (`tools/live.py` "single test" mode) and speak over it at a natural moment. Label it as you save.

| # | Say | Function | Expected |
|---|---|---|---|
| 1 | «طيب» (falling, short) | acknowledgement | KEEP |
| 2 | «طيب…» then silence | agreement at agent pause | KEEP |
| 3 | «طيب كم المبلغ؟» | topic change | STOP |
| 4 | «طيب بس أنا ما أقدر الخميس» | objection | STOP |
| 5 | «طيب طيب» (fast) | impatience | STOP |
| 6 | «طيييب» (long, thinking) | hesitation | KEEP |
| 7 | «طيب مع السلامة» | closing | STOP |
| 8–14 | same seven with «خلاص» | — | same |
| 15–20 | «إيه», «أها», «مم», «صح», «والله», «تمام» | backchannel | KEEP |
| 21–26 | «لا لا لحظة», «اسمع», «معليش بس», «وقف», «دقيقة», «كم؟» | interruption | STOP |

Each person also records one 2-minute free call in the demo, interrupting naturally. Everyone signs the consent line in `data/live/CONSENT.md`. Audio stays local and git-ignored.
