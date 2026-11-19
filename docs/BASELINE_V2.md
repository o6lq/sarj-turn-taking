# BASELINE_V2 — the safe version and how to get back to it

**`v2.0` = commit `84f9b21`** (same commit as `v2.0-freeze`), branch `release/v2-safe` at the same commit.
It scored **91.9** on Sarj's checkpoint. Every later version is compared with the numbers below (PLAN_V3 §2, §4).

Scored code (`turn/`, `my_system.py`, `models/`, `ttbench/`) is byte-identical between `545a66e`, `01f830e` and
`84f9b21`, so the real-time run `runs/freeze-dev` (on `545a66e`) is a v2.0 run.

## 1. v2.0 on every set we have (measured 19 Nov, 23:06–23:10, Mac, Cohere ASR)

| Set | How | Run | Version | Core | R | S | A | False stops | Misses | Stop latency median / p90 (ms) | EOT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dev` | real time (`ttbench.play --calls all`) | `runs/freeze-dev` | 545a66e | **99.1** | 0.980 | 1.000 | 1.0 | 0 | 0 | 0 / 772 | 100.0 |
| `dev` | sim | `runs/sim-v2.0-dev` | v2.0 | **99.3** | 0.985 | 1.000 | 1.0 | 0 | 0 | 0 / 670 | 100.0 |
| `dev-aug` | sim | `runs/sim-v2.0-dev-aug` | v2.0 | **96.3** | 0.952 | 1.000 | 0.85 | 0 | 0 | 0 / 780 | 97.9 |
| `novel-test` | sim | `runs/sim-v2.0-novel-test` | v2.0 | **84.0** | 0.786 | 0.895 | 0.87 | 10 | 3 | 0 / 399 | 100.0 |
| `synth-test` | sim | `runs/sim-v2.0-synth-test` | v2.0 | **99.5** | 0.998 | 0.990 | 1.0 | 1 | 0 | 0 / 140 | — |
| `novel-test-2` (regression alarm only) | sim | `runs/sim-v2.0-novel-test-2` | v2.0 | **85.3** | 0.805 | 0.871 | 1.0 | 14 | 4 | 0 / 295 | 96.8 |

Notes:
- `novel-test-2` was 85.46 when it was first opened (EXPERIMENTS #46, before Track A). On v2.0 it is **85.3**.
  Gate G10 (≥ 85.0) uses this set as an alarm only; its sentences are never read to fix anything.
- Gate baselines (PLAN_V3 §4): G1 `novel-test` 84.0 · G2 `dev-aug` 96.3 · G3 false stops summed over these sets
  = 25 (0 + 0 + 10 + 1 + 14) · G4 `dev` sim 99.3, 0 misses · G5 median stop latency 0 ms on every set.

## 2. Sarj's checkpoint (official, 16 human calls, run on `84f9b21` over a Cloudflare tunnel)

| Core | R | S | Interruptions caught | Stop latency median / p90 | False stops | End-of-turn |
|---|---|---|---|---|---|---|
| **91.9** | 0.87 | 0.96 | 6 of 7 | 0 / 448 ms | 1 | 100 |

By category: overlapping starts (OV) 1 of 2 stopped (one missed interruption); background (BG) 2 of 3 kept (one
false stop); every other category perfect.

## 3. Checkpoint mirror and stress sets (Phase 1)

Filled in by Phase 1: see §5 below and `runs/v2.0-cp-*`.

## 4. Rollback procedure

- **A bad merge on `main`:** `git revert -m 1 <merge-commit>` on a branch, then a PR. Never rewrite history.
- **Emergency for the final** (anything wrong with the latest version):
  ```bash
  git fetch --tags
  git checkout v2.0            # or the last v2.N that passed every gate in PLAN_V3 §4
  source .venv/bin/activate
  python my_system.py --host 0.0.0.0
  ```
  `bash tools/checkpoint.sh` does the same with keep-awake, warm-up, readiness check and the tunnel.
- **What we hand in:** the hash of the last tag that passed all of PLAN_V3 §4. If none did, `84f9b21` (v2.0).
- Check after a checkout: `git rev-parse --short HEAD` must print the hash you hand in, with no `-dirty`
  (`ttbench.server.git_version` reports it in the hello message).
