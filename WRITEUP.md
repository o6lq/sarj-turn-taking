# Write-up: sarj-saud

**Frozen commit:** `84f9b21` (tag `v2.0-freeze`). Code: https://github.com/o6lq/sarj-turn-taking

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
| Never stopping on a single «أيوه/نعم» in the scored path | `novel-test` −0.31, because «نعم» and «طيب» sometimes open a real interruption | Kept in the demo only |

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

The frame loop never waits for the worker, so decisions are not queued behind recognition. Urgent reads during a possible interruption go before end-of-turn reads, and recogniser text is cleaned of speaker labels, loops and subtitle phrases before any layer sees it. The scored path makes no network calls. The speaking agent in our demo is separate from the scored system.

## 4. Speed and cost

| | |
|---|---|
| Median stop latency on the dev set (real-time run, from the report) | 0 ms after the intent became evident, with p90 at 772 ms |
| Stop latency on 55 replayed live-demo calls, 3 runs each | p50 260 ms, p90 583 ms, p99 726 ms (956 ms before the freeze fixes) |
| Where that time goes (each stage, roughly) | At least 100 ms of voiced speech, then a recognition read of 70 ms median and 200 ms p90. Frame processing averages 0.31 ms |
| Hardware it ran on, and how busy it was | MacBook Air, Apple M5, 24 GB. The decision process used about 6% of one CPU core during a call and peaked at 2.9 GB of memory |
| Model sizes | Cohere 8-bit 2.3 GB, meaning model 105 MB, CAM++ 27 MB, Silero VAD 2.2 MB, decision weights about 4 KB |
| Cloud calls per minute of audio, and their typical delay | 0 in the scored path |
| What it would cost to run on a real call | No per-minute API cost in the scored path. The cost is the machine running it |

## 5. Results on the development set

Official real-time run on all 24 calls, `runs/final-dev-official` (three more back-to-back runs on the pre-freeze commit `bf0ed81` scored 99.26, 99.11 and 98.98 with no crash and flat memory):

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
| `dev-aug` (96 calls) | Noise, phone line, speakerphone | 96.30 | 0 | 0 |
| `synth-test` (60 calls) | New voices, our sentences | 99.47 | 1 | 0 |
| `novel-test` (120 calls) | New phrasing, used for category-level diagnosis | 84.00 | 10 | 3 |
| **`novel-test-2` (120 calls)** | **Blind phrasing written by another model, measured once on `bf0ed81` with no tuning afterwards** | **85.46** | 14 | 4 |

Core scores are for the frozen commit. The freeze changes (urgent reads first, text cleaning) cost `dev-aug` 0.27 and `novel-test` 0.07, inside our 0.3 gate, and kept 11 false stops in total.

The blind test is our most honest estimate. Direct interruptions and talk-over scored 1.00, ambiguous words 0.43 on responsiveness, and hesitation 0.33 on restraint. Its dialect gap (Najdi against other dialects) is 0.05 in F, and its noise gap (clean against noisy) is 0.10.

**What each layer adds.** One layer is removed at a time (measured before the freeze changes):

| Layer removed | `novel-test` | `dev-aug` |
|---|---|---|
| Full system | 84.07 | 96.57 |
| Echo suppression | 83.70 | 93.56 |
| Level gate | 83.73 | 95.23 |
| Speaker check | 80.50 | 96.21 |
| Learned model (rules only) | 79.80 | 92.53 |
| Meaning model | 82.28 | 96.14 |
| RESUME | 84.07 | 96.57 |

## 6. Where it fails (at least five cases)

| # | Call | What happened | Listen | Why it went wrong (your analysis) |
|---|---|---|---|---|
| 1 | GI-D02 | Real-time run: stopped 772 ms after the intent was clear, for 0.76 credit | `python -m ttbench.render --run runs/final-dev-official --scenario GI-D02` | The caller says «أيوه… بس أبغى أغيّر الموعد». The opening word is a backchannel, so the system correctly waits, and the intent only appears at «بس» |
| 2 | AM-D02 | Real-time run: stopped 322 ms after the intent was clear, for 0.99 credit | `python -m ttbench.render --run runs/final-dev-official --scenario AM-D02` | «طيب؟ … بس كم المبلغ بالضبط؟» begins with an ambiguous word, and the stop waits for the following floor bid |
| 3 | OV-D01 | Car-noise version: stopped 1200 ms late, for 0.55 credit | `data/dev-aug/audio/OV-D01-q.wav` | In noise, recognition stretched «إيه الأحد» into a long «إييييي» and a wrong word, so the answer looked like a backchannel for over a second |
| 4 | GI-D02 | Speakerphone version: stopped 670 ms late, for 0.81 credit | `data/dev-aug/audio/GI-D02-e.wav` | Speakerphone audio lowered the voiceprint match to 0.27, so one early reading was treated as another speaker |
| 5 | AM-D01 | Car-noise version: stopped on a lone «طيب», for 0.60 credit | `data/dev-aug/audio/AM-D01-q.wav` | Recognition heard «طيب» as «بس», which is a strong floor bid, so a recognition error became a stop |
| 6 | AM-D03 | Phone and street version: kept talking through «خلاص خلاص… فهمت», for 0.60 credit | `data/dev-aug/audio/AM-D03-p.wav` | No recognition read was produced for this moment, so its words never reached the decision |
| 7 | Live demo, team recordings | Stopped on a lone «أيوه» or «نعم» 5 times in 55 calls | `python -m tools.replay_live --repeat 3` | The meaning model was sure about one polite word. A demo-only guard brought this to 0. We kept the scored path unchanged, because the same guard there lost 0.31 on `novel-test` |

## 7. Strengths, weaknesses and trade-offs

**Strengths.** The system handles direct interruptions, talk-over, background noise and television well. It runs fully on a laptop, and each decision can be explained.

**Weaknesses.** New backchannels and thinking aloud cause most false stops. A lone «طيب» remains hard, because the words alone do not reveal the intent. Noisy and phone audio score lower than clean audio. All our benchmark sets use synthetic voices, so the hidden human set may score lower. Speed depends on the machine: on a Windows laptop running recognition on the CPU, reads were too slow and every dev interruption was missed, so the final run uses the Mac.

**Trade-offs.** We chose restraint over a faster but less careful stop, because a wrong stop interrupts the conversation. We rejected changes that raised one test set while hurting another.

**Live demo engineering (demo only, not scored).** Microphone mode follows the output device: raw audio with a headset, echo cancellation with laptop speakers. A preflight check times the recogniser, brain and voice before a talk. The demo agent starts its reply 300 ms into the caller's silence and stays quiet until TAKE_TURN, and keeps its cloud connections open. None of this touches `turn/`, `my_system.py` or `models/`.

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
| API | Claude Opus 5.5 through Claude Code | Teacher labels for training |
| API | Claude Haiku 5.5 | Demo brain when Gemini fails |
| API | Gemini: `gemini-3.5-flash-lite`, `gemini-2.5-flash-native-audio`, `gemini-3.8-flash-tts` | Demo agent's brain and voice |
| API | Gemini text models | Writing the blind test sentences |
| API | ElevenLabs | Optional demo voice and synthetic test voices |
| Service | Microsoft Edge TTS | Synthetic test calls and a fallback demo voice |
| Data | Sarj development set | Evaluation |
| Data | ESC-50 (CC BY-NC) | Background noise for generated test calls only |

**Sources:** [Silero VAD](https://github.com/snakers4/silero-vad) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [its turbo weights](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) · [Cohere Arabic MLX build](https://huggingface.co/majentik/cohere-transcribe-arabic-07-2026-MLX-8bit) · [3D-Speaker](https://github.com/modelscope/3D-Speaker) and [CAM++ ONNX](https://huggingface.co/csukuangfj/speaker-embedding-models) via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) · [CAMeLBERT-DA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da) · [Smart Turn](https://github.com/pipecat-ai/smart-turn) · [ESC-50](https://github.com/karolpiczak/ESC-50)

## 9. Team

أصايل، سعود، ابتسام، سيرين، عبدالله
