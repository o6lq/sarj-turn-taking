# Write-up: sarj-saud

## 1. The idea

**Approach, in a few words:** a layered hybrid of acoustic signals, local speech recognition, a Gulf Arabic lexicon, a small distilled meaning model and a learned decision layer.

Every 20 ms the system asks whether the sound on the caller's line is really this caller, and whether they want the floor. Cheap acoustic checks reject noise, television and the agent's own echo first. Speech recognition and meaning then separate «إيه والله صادق» (carry on) from «إيه… بس أبي أغيّر الموعد» (stop). Unlike the voice-activity baseline, every signal is measured relative to this caller's own first answer, and every decision carries a reason that a person can read.

## 2. Why this approach

The baseline stops at any voice and scores 41.0. In the calls, most caller sounds are not interruptions, and the same word can mean opposite things, so neither loudness nor words are enough alone.

What we tried and rejected, with measurements in `docs/EXPERIMENTS.md`:

| Idea | Result | Decision |
|---|---|---|
| WebRTC echo cancellation in the scored path | Removed the caller's voice by 25–32 dB during overlap | Rejected |
| Ignoring words the meaning model reads as thinking aloud | `novel-test` −3.19 and `dev-aug` −2.23, with more missed interruptions | Rejected |
| Retraining the meaning model on 829 harder examples | `novel-test` +1.71, but false stops rose from 11 to 23 | Rejected |
| Tone of voice for a lone «طيب» | No gain on ambiguous moments, `synth-test` −0.66 | Rejected |
| Reading speech every 100 ms | False stops rose from 2 to 5, late stops did not fall | Rejected |
| Stopping early from the voice alone | Late stops unchanged at 12 of 97 | Left off |

## 3. How it works

```
20 ms frame: agent channel + caller channel
  L0 echo suppression     spectral comparison with the last 800 ms of agent audio
  L1 voice and level      Silero VAD + loudness relative to this caller and the noise floor
  candidate opens         the caller speaks while the agent speaks
     every 150 ms, in a background worker:
  L4  speech recognition  Cohere Transcribe Arabic on Mac, faster-whisper on Windows
  L4a lexicon             floor bids, backchannels, fillers, agent echo
  L4b meaning             CAMeLBERT-DA student distilled from an LLM teacher (about 7 ms)
  L2  speaker match       CAM++ voiceprint against the caller's first answer
  L3  prosody             pitch, effort and rate relative to the caller
  L5  decision            hard gates, rules, learned logistic regression, meaning veto or promote
  output                  STOP, RESUME within 460 ms, or TAKE_TURN after the caller finishes
```

The frame loop never waits for the worker, so decisions are not queued behind recognition. The scored path makes no network calls. The speaking agent in our demo is separate from the scored system.

## 4. Speed and cost

| | |
|---|---|
| Median stop latency on the dev set (real-time run, from the report) | 0 ms after the intent became evident, with p90 at 772 ms |
| Where that time goes (each stage, roughly) | At least 100 ms of voiced speech, then a recognition read of 70 ms median and 200 ms p90. Frame processing averages 0.31 ms |
| Hardware it ran on, and how busy it was | MacBook Air, Apple M5, 24 GB. The decision process used about 6% of one CPU core during a call and peaked at 2.9 GB of memory |
| Model sizes | Cohere 8-bit 2.3 GB, meaning model 105 MB, CAM++ 27 MB, Silero VAD 2.2 MB, decision weights about 4 KB |
| Cloud calls per minute of audio, and their typical delay | 0 in the scored path |
| What it would cost to run on a real call | No per-minute API cost in the scored path. The cost is the machine running it |

## 5. Results on the development set

Official real-time run on all 24 calls, `runs/final-dev-official`:

| Measure | Baseline B1 | Our system |
|---|---|---|
| Core score | 41.0 | **99.0** |
| Responsiveness R | | 0.98 |
| Restraint S | | 1.00 |
| False stops | | 0 |
| Missed interruptions | | 0 |
| End-of-turn track | | 100 |

Unseen and harder sets, measured with the official scorer in our simulator:

| Set | What it tests | Core | False stops | Misses |
|---|---|---|---|---|
| `dev-aug` (96 calls) | Noise, phone line, speakerphone | 96.57 | 0 | 0 |
| `synth-test` (60 calls) | New voices, our sentences | 99.47 | 1 | 0 |
| `novel-test` (120 calls) | New phrasing, used for category-level diagnosis | 84.07 | 10 | 3 |
| **`novel-test-2` (120 calls)** | **Blind phrasing written by another model, measured once with no tuning afterwards** | **85.46** | 14 | 4 |

The blind test is our most honest estimate. Direct interruptions and talk-over scored 1.00, ambiguous words 0.43 on responsiveness, and hesitation 0.33 on restraint. Its dialect gap (Najdi against other dialects) is 0.05 in F, and its noise gap (clean against noisy) is 0.10.

**What each layer adds.** One layer is removed at a time:

| Layer removed | `novel-test` | `dev-aug` |
|---|---|---|
| Full system | 84.07 | 96.57 |
| Echo suppression | 83.70 | 93.56 |
| Level gate | 83.73 | 95.23 |
| Speaker check | 80.50 | 96.21 |
| Learned model (rules only) | 79.80 | 92.53 |
| Meaning model | 82.28 | 96.14 |
| RESUME | 84.07 | 96.57 |

## 5b. After the checkpoint: measure first, then fix

Sarj's Day-3 checkpoint (16 human calls) scored our freeze version `v2.0` at **91.9** (R 0.87, S 0.96): one missed
interruption in *overlapping starts* and one false stop in *background*. Two events cannot tell you what to fix, so we
built a mirror of the checkpoint before changing anything:

- **A checkpoint mirror** (`tools/cp_bank.py`, `tools/cp_gen.py`): 48 calls, 6 per category, written by another model
  that saw only the official category definitions in `SCRIPTS.md`, never our lexicon. Voices we never trained on
  (a Saudi VITS voice and Gulf voices outside our training pitch and rate), timings anchored to agent words as in
  `SCRIPTS.md`. The same sentences are rendered three times with different voices and noise.
- **Stress probes**: two disjoint banks (one to learn from, one held out) of targeted cases: option answers, a pause
  after «إيه»، onsets from −300 to +300 ms around the agent, side talk with question words, «طيب/خلاص», thinking aloud,
  new backchannels, and a hesitation probe added later.
- **A decision log without audio** (`TT_DECISION_LOG`): the rule that fired and its features for every read, so a real
  run can be explained. Text is off unless allowed.
- **Sarj-style reports with uncertainty** (`tools/cp_report.py`): the T01 tables plus a bootstrap interval, what a
  16-call checkpoint would show, and a 64-call projection. `tools/gates.py` runs every set against the last version.

The mirror reproduced the checkpoint: **92.0 in real time (R 0.92, S 0.95)**, with the same kind of overlap miss. A
16-call sample of it falls below 92 in 37 % of draws, so 91.9 was a typical result, not bad luck.

**What the evidence said** (and what it did not):

| Suspected weakness | Verdict | Evidence |
|---|---|---|
| An answer that repeats the agent's option is treated as echo | Partly | The real loss: a short answer to the agent's question (one content word) |
| RESUME over the caller | Rejected | 20/20 caught; RESUME fired twice in 140 calls and never cost a point |
| Caller starts just before the agent | Real, small | The first word is lost to recognition; it compounds the short-answer case |
| A floor-bid word from someone else stops the agent | Confirmed, wider | Side talk at the caller's level: 12/14 false stops; thinking aloud with question words: 11/20 |
| «طيب/خلاص» + more | Weak | 10/10 in one bank, 3/5 in the other |

Three things we did not expect: half of the official dev calls never build a voiceprint, and even when one exists
**22 % of real interruptions read below 0.25 similarity**, so speaker vetoes are only safe at extreme values; the
meaning model rates thinking aloud and side talk at ~1.0 «wants the floor»; and the learned model could stop on its
own on loud side talk.

**What we changed** (each on its own branch, kept only if every gate passed against the previous version):

| Version | Change | Mirror ×3 (sim) | False stops (all sets) |
|---|---|---|---|
| v2.0 | the checkpoint version | 92.8 | 66 |
| v2.1 | an answer that picks an option the agent offered is a floor bid | 94.0 | 66 |
| v2.2 | a question/call/command word stops only when the caller is talking to the agent (not calling someone in the room, not thinking aloud, not a lone question word); blessings are backchannels | 94.3 | 51 |
| v2.3 | hesitation structure: a negation inside a sentence, «عندي» alone, repeated digits and filler-led thinking are not bids | 94.3 | 58* |
| v2.4 | the learned model may veto a rule stop below p 0.3 (was 0.2) | 94.7 | 55 |
| v2.5 | weighing two options aloud («السبت ولا الأحد…») is thinking, not content | **95.1** | **54** |

\* from v2.3 on, two hesitation sets were added to the sum.

Rejected after measuring: safe RESUME (+4 false stops: RESUME was rescuing wrong stops), carrying the pre-onset turn
(+1 false stop, no gain), «طيب» + any continuation and question-context thresholds (no effect), a 0.4 model veto
(one real interruption missed), fuzzy blessing matching (a real interruption missed).

**v2.5 against v2.0, real time:** mirror **92.0 → 96.9**, dev 99.1 → 99.0, through a Cloudflare tunnel 96.4.
Two of the mirror's changed events are cases we studied while diagnosing, so the fairer evidence is on sets we never
tuned on: `dev-aug` 96.3 → 97.5, `novel-test` 84.0 → 84.7, `novel-test-2` 85.3 → 85.6, the held-out stress bank
76.4 → 79.2, the other two mirror renders 91.6 → 93.9 and 93.4 → 93.5. A 30-minute single-process soak (120 calls)
passed: no crash, every ping answered, decisions identical on 99 % of events across three runs.

## 6. Where it fails (at least five cases)

| # | Call | What happened | Listen | Why it went wrong (your analysis) |
|---|---|---|---|---|
| 1 | GI-D02 | Real-time run: stopped 772 ms after the intent was clear, for 0.76 credit | `python -m ttbench.render --run runs/final-dev-official --scenario GI-D02` | The caller says «أيوه… بس أبغى أغيّر الموعد». The opening word is a backchannel, so the system correctly waits, and the intent only appears at «بس» |
| 2 | AM-D02 | Real-time run: stopped 322 ms after the intent was clear, for 0.99 credit | `python -m ttbench.render --run runs/final-dev-official --scenario AM-D02` | «طيب؟ … بس كم المبلغ بالضبط؟» begins with an ambiguous word, and the stop waits for the following floor bid |
| 3 | OV-D01 | Car-noise version: stopped 1200 ms late, for 0.55 credit | `data/dev-aug/audio/OV-D01-q.wav` | In noise, recognition stretched «إيه الأحد» into a long «إييييي» and a wrong word, so the answer looked like a backchannel for over a second |
| 4 | GI-D02 | Speakerphone version: stopped 670 ms late, for 0.81 credit | `data/dev-aug/audio/GI-D02-e.wav` | Speakerphone audio lowered the voiceprint match to 0.27, so one early reading was treated as another speaker |
| 5 | AM-D01 | Car-noise version: stopped on a lone «طيب», for 0.60 credit | `data/dev-aug/audio/AM-D01-q.wav` | Recognition heard «طيب» as «بس», which is a strong floor bid, so a recognition error became a stop |
| 6 | AM-D03 | Phone and street version: kept talking through «خلاص خلاص… فهمت», for 0.60 credit | `data/dev-aug/audio/AM-D03-p.wav` | No recognition read was produced for this moment, so its words never reached the decision |
| 7 | cp-mirror HT-M02 (v2.5) | Thinking aloud («وين حطيته يا ربي») still stops the agent | `runs/v2.5-cp-mirror-rt/report.md` | The first read is only «مين»; by the time «يا ربي» shows it is thinking, the stop has fired. The model is sure (p 0.97), so its veto does not help |
| 8 | stress bank, side talk (v2.5) | A person near the phone, louder than the caller, says «كم الساعة الحين؟» | `runs/sim-hyb30-cp-stress-dev` | Recognition heard «كيف حالك؟», and the voice is +8 dB over the caller with no voiceprint yet: no cue says it is someone else |
| 9 | stress bank, blessings (v2.5) | «بيض الله وجهك» heard as «بيض اللهو» | `runs/sim-hyb30-cp-stress-dev` | Two unknown words count as content. Fuzzy phrase matching fixed it but cost a real interruption elsewhere, so we left it |

## 7. Strengths, weaknesses and trade-offs

**Strengths.** The system handles direct interruptions, talk-over, background noise and television well. It runs fully on a laptop, and each decision can be explained.

**Weaknesses.** New backchannels and thinking aloud cause most false stops. A lone «طيب» remains hard, because the words alone do not reveal the intent. Noisy and phone audio score lower than clean audio. All our benchmark sets use synthetic voices, so the hidden human set may score lower.

**Trade-offs.** We chose restraint over a faster but less careful stop, because a wrong stop interrupts the conversation. We rejected changes that raised one test set while hurting another.

**With another week,** we would collect consented human recordings, label them in the kit's format, and rebalance the meaning model with more examples of when not to stop.

## 8. External services, models and data

None of the cloud services below is used by the scored decision path.

| Kind | Name and version | What for |
|---|---|---|
| Model | Cohere Transcribe Arabic, MLX 8-bit (Apache-2.0) | Speech recognition on Mac |
| Model | Whisper large-v3-turbo through faster-whisper (MIT) | Speech recognition on Windows and Linux |
| Model | Silero VAD (MIT) | Voice activity detection |
| Model | 3D-Speaker CAM++ through sherpa-onnx (Apache-2.0) | Speaker match |
| Model | CAMeLBERT-DA (Apache-2.0) | Base of the local meaning model |
| Model | Smart Turn v3.2 (BSD-2) | Optional end-of-turn signal, off by default |
| API | Claude Opus 5.5 through Claude Code | Teacher labels for training, and a fallback demo brain |
| API | Gemini: `gemini-3.5-flash-lite`, `gemini-2.5-flash-native-audio`, `gemini-3.8-flash-tts` | Demo agent's brain and voice |
| API | Gemini text models (`gemini-3.8-flash`, `gemini-3.1-flash-lite`) | Writing the blind test sentences and the checkpoint-mirror and stress banks |
| Model | Lahja SA Huba (wasmdashai, VITS) | A Saudi voice we never trained on, for the checkpoint mirror only |
| API | ElevenLabs | Optional demo voice and synthetic test voices |
| Service | Microsoft Edge TTS | Synthetic test calls and a fallback demo voice |
| Data | Sarj development set | Evaluation |
| Data | ESC-50 (CC BY-NC) | Background noise for generated test calls only |

## 9. Team

أصايل، سعود، ابتسام، سيرين، عبدالله
