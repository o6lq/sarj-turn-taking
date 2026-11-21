# Write-up: sarj-team1

Hand-in: commit `2084030` (tag `v2.6`). Its decisions are identical to `v2.5`; `v2.6` only adds a label, a
confidence and the heard text to each decision (metadata, no change in timing or type).

## 1. The idea

**Approach, in a few words:** a layered hybrid: acoustic checks, local speech recognition, a Gulf Arabic lexicon, a
small distilled meaning model and a learned decision layer.

Every 20 ms the system asks two questions: is the sound on the caller's line really this caller, and do they want the
floor? Cheap acoustic checks reject noise, television and the agent's own echo first. Speech recognition and meaning
then separate «إيه والله صادق» (carry on) from «إيه… بس أبي أغيّر الموعد» (stop). Unlike the voice-activity baseline,
every signal is measured relative to this caller's own first answer, and every decision carries a reason a person can
read.

## 2. Why this approach

The baseline stops at any voice and scores 41.0. In the calls, most caller sounds are not interruptions, and the same
word can mean opposite things, so neither loudness nor words are enough alone.

What we tried and rejected (all measured, `docs/EXPERIMENTS.md`):

| Idea | Result | Decision |
|---|---|---|
| WebRTC echo cancellation in the scored path | Removed the caller's voice by 25–32 dB during overlap | Rejected |
| Retraining the meaning model on 829 harder examples | `novel-test` +1.71, but false stops 11 → 23 | Rejected |
| Tone of voice for a lone «طيب» | No gain on ambiguous moments, `synth-test` −0.66 | Rejected |
| Reading speech every 100 ms instead of 150 ms | False stops 2 → 5, late stops unchanged | Rejected |
| Stopping on ≥ 500 ms of voice alone | +23 false stops | Rejected |
| A second "confirmation" read before stopping | `novel-test-3` 84.9 → 80.9 and +160–190 ms | Rejected |

## 3. How it works

```
20 ms frame: agent channel + caller channel
  L0 echo suppression     spectral comparison with the last 800 ms of agent audio
  L1 voice and level      Silero VAD + loudness relative to this caller and the noise floor
  candidate opens         the caller speaks while the agent speaks
     every 150 ms, in a background worker:
  L4  speech recognition  Cohere Transcribe Arabic (MLX) on Mac, faster-whisper on Windows
  L4a lexicon             floor bids, backchannels, fillers, blessings, agent echo
  L4b meaning             CAMeLBERT-DA student distilled from an LLM teacher (about 7 ms)
  L2  speaker match       CAM++ voiceprint against the caller's first answer
  L3  prosody             pitch, effort and rate relative to the caller
  L5  decision            hard gates, rules, learned logistic regression (veto below p 0.3)
  output                  STOP, RESUME within 460 ms, or TAKE_TURN after the caller finishes
```

The frame loop never waits for the worker, so decisions are not queued behind recognition. The scored path makes no
network calls. Each decision is sent with its label (`interruption`, `backchannel`, `hesitation`, `echo`, `nonverbal`), a
confidence and the heard text. The talking agent in our demo is separate from the scored system.

## 4. Speed and cost

| | |
|---|---|
| Median stop latency on the dev set (real-time run, from the report) | 0 ms after the intent became evident; p90 772 ms (7 stops) |
| Where that time goes (each stage, roughly) | At least 100 ms of voiced speech, then a recognition read of 70 ms median and 200 ms p90. Frame processing averages 0.31 ms |
| Hardware it ran on, and how busy it was | MacBook Air, Apple M5, 24 GB. About 6 % of one CPU core during a call; peak memory 3.15 GB over 96 calls back to back |
| Model sizes | Cohere 8-bit 2.3 GB, meaning model 105 MB, CAM++ 27 MB, Silero VAD 2.2 MB, decision weights about 4 KB |
| Cloud calls per minute of audio, and their typical delay | 0 in the scored path |
| What it would cost to run on a real call | No per-minute API cost; the cost is the machine running it |

## 5. Results on the development set

Real-time run of the hand-in commit on all 24 calls (`runs/fc-dev-rt`, `system_version 2084030`):

| Measure | Baseline B1 | Our system |
|---|---|---|
| Core score | 41.0 | **99.0** |
| Responsiveness R | | 0.98 |
| Restraint S | | 1.00 |
| False stops / missed interruptions | | 0 / 0 |
| End-of-turn track (P, Q) | | 1.00, 1.00 |

Sets the system was not tuned on, scored with the official scorer in our simulator (`runs/gates-v2.5.json`):

| Set | What it tests | Core | False stops | Misses |
|---|---|---|---|---|
| `dev-aug` (96 calls) | Dev calls with noise, phone line, speakerphone | 97.51 | 0 | 0 |
| `synth-test` (60 calls) | New voices, our sentences | 99.55 | 0 | 0 |
| `novel-test` (120 calls) | New phrasing, used for category diagnosis | 84.70 | 9 | 3 |
| **`novel-test-2` (120 calls)** | **Phrasing written blind by another model; never tuned on** | **85.57** | 13 | 4 |

On the blind set, direct interruptions and talk-over score ≥ 0.99, ambiguous words 0.43 on responsiveness and
hesitation 0.33 on restraint. Najdi calls score F 0.95 against 0.89 for the other dialects.

**After the checkpoint.** Sarj's Day-3 checkpoint scored `v2.0` at 91.9. Before changing anything, we built a mirror
of it: 48 calls, 6 per category, written by another model from the definitions in `SCRIPTS.md` only, with voices we
never trained on. It reproduced the result (92.0 in real time, same kind of overlap miss). We then made five small
fixes, each kept only if every set passed against the previous version: option answers count as floor bids; a
question word stops only when the caller is talking to the agent; hesitation structure is not a bid; the learned
model may veto a rule stop below p 0.3; weighing two options aloud is thinking. Result: mirror **92.0 → 96.9** in
real time, `dev-aug` 96.3 → 97.5, `novel-test` 84.0 → 84.7, false stops across all sets 66 → 54.

**What each layer adds** (one removed at a time, measured on the 18 Nov build):

| Layer removed | `novel-test` | `dev-aug` |
|---|---|---|
| Full system | 84.07 | 96.57 |
| Echo suppression | 83.70 | 93.56 |
| Level gate | 83.73 | 95.23 |
| Speaker check | 80.50 | 96.21 |
| Learned model (rules only) | 79.80 | 92.53 |
| Meaning model | 82.28 | 96.14 |

## 6. Where it fails (at least five cases)

| # | Call | What happened | Listen | Why it went wrong (your analysis) |
|---|---|---|---|---|
| 1 | GI-D02 | Real-time run: stopped 772 ms after the intent was clear, 0.76 credit | `python -m ttbench.render --run runs/fc-dev-rt --scenario GI-D02` | The caller says «أيوه… بس أبغى أغيّر الموعد». The first word is a backchannel, so the system rightly waits; the intent only appears at «بس», and the stop waited for the word after it |
| 2 | AM-D02 | Real-time run: stopped 323 ms after the intent was clear, 0.99 credit | `python -m ttbench.render --run runs/fc-dev-rt --scenario AM-D02` | «طيب؟ … بس كم المبلغ بالضبط؟» opens with an ambiguous word, so the stop waits for the floor bid that follows |
| 3 | AM-D02 | Speakerphone version: stopped 781 ms late, 0.76 credit | `data/dev-aug/audio/AM-D02-e.wav` | The first read came in 15 dB below the caller's own level, so the level gate treated it as side speech or TV. The stop came only on the next read |
| 4 | GI-D02 | Speakerphone version: stopped 670 ms late, 0.81 credit | `data/dev-aug/audio/GI-D02-e.wav` | Same pattern as case 1, made later by the speakerphone: the floor bid after «أيوه» arrived in a later read |
| 5 | AM-D01 | Car-noise version: stopped on a lone «طيب», 0.60 credit | `data/dev-aug/audio/AM-D01-q.wav` | Recognition heard «طيب» as «بس», a strong floor bid, so a recognition error became a stop |
| 6 | AM-D03 | Phone + street version: kept talking through «خلاص خلاص… فهمت», 0.60 credit | `data/dev-aug/audio/AM-D03-p.wav` | No recognition read was produced for this moment (the only read in the call is the opening «إي نعم»), so its words never reached the decision |
| 7 | Mirror HT-M02 | Thinking aloud («وين حطيته يا ربي») stops the agent | `runs/v2.5-cp-mirror-rt/report.md` | The first read is only «مين»; by the time «يا ربي» shows it is thinking, the stop has fired. The model is sure (p 0.97), so its veto does not help |

## 7. Strengths, weaknesses and trade-offs

**Strengths.** Direct interruptions, talk-over, background noise, television and the agent's own echo are handled
well, with zero false stops on the dev set and its noisy versions. It runs fully on a laptop, and each decision can be
explained.

**Weaknesses.** New backchannels and thinking aloud cause most false stops. A lone «طيب» or «خلاص» remains hard: the
loss there comes from tone of voice or a misheard word, which rules on words cannot fix. Speakerphone and noisy audio
stop later. Our test sets use synthetic voices, so the hidden human set may score lower.

**Trade-offs.** We chose restraint over a faster but less careful stop, because a wrong stop breaks the conversation.
We rejected every change that raised one set while hurting another, even when it helped the checkpoint mirror.

**With another week,** we would collect consented human recordings in several dialects, label them in the kit's
format, and train the decision layer on tone for the ambiguous words.

## 8. External services, models and data

The scored decision path runs fully on the laptop. None of the APIs below is used by it.

| Kind | Name and version | What for |
|---|---|---|
| Model | Cohere Transcribe Arabic, MLX 8-bit (Apache-2.0) | Speech recognition on Mac |
| Model | Whisper large-v3-turbo through faster-whisper (MIT) | Speech recognition on Windows and Linux |
| Model | Silero VAD (MIT) | Voice activity detection |
| Model | 3D-Speaker CAM++ through sherpa-onnx (Apache-2.0) | Speaker match |
| Model | CAMeLBERT-DA (Apache-2.0) | Base of the local meaning model |
| Model | Smart Turn v3.2 (BSD-2) | Optional end-of-turn signal, off by default |
| API | Claude Opus 5.5 through Claude Code | Teacher labels for training; fallback demo brain |
| API | Gemini text, audio and TTS models | Demo agent's brain and voice; writing the blind test, mirror and stress sentences |
| API | ElevenLabs, HUMAIN Voice | Demo voices and synthetic test voices |
| Service | Microsoft Edge TTS | Synthetic test calls and a fallback demo voice |
| Model | Lahja SA Huba (VITS) | A Saudi voice we never trained on, checkpoint mirror only |
| Data | Sarj development set | Evaluation |
| Data | ESC-50 (CC BY-NC) | Background noise for generated test calls only |

## 9. Team

أصايل، سعود، ابتسام، سيرين، عبدالله
