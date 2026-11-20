# HANDOFF — PLAN_V6   (written 18:46, trigger: §2 self-estimate (~85 tool calls) + final check done)

## 1. Freeze answer
- Recommended freeze commit: main @ 2084030f628ac995e3feda118b29e8d736a8930d   (v2.6, tag `v2.6` pushed)
- Freeze-ready: yes — every PLAN_V6 §7.3 hand-in condition holds (table in §3)
- Safe fallback: v2-final @ aab8988 (scored code v2.5 @ 7c52fdd); it needs nothing else. Protected by tag `v2-final`, branch `release/v2-final`, and the offline bundle `~/Downloads/sarj-backups/sarj-v2-final-aab8988.bundle`.

## 2. Tasks
| Task | Status | Branch | Commit | Result vs v2.5 | Merged? |
|---|---|---|---|---|---|
| T1 dialect lexicon | dropped (no time for full gates before 18:00) | — | — | — | no |
| T2 ack + but | dropped (time) | — | — | — | no |
| T3 either-pref | dropped (17:15 go/no-go: no cp-am-dev) | — | — | — | no |
| T4 perf (identical decisions) | dropped (time) | — | — | identical? n/a, RT p90 n/a | no |
| T5 labels / confidence / transcript | done | feat/decision-labels | 08d4567 | identical decisions (same types/counts; Δt ≤ 1 frame = v2.5 self-noise) | yes, PR #28 → 2084030 |

## 3. Numbers (from files — give the path for each)
| Metric | v2.5 | candidate | file |
|---|---|---|---|
| novel-test-2 (sim, main tree) | 85.57 | 85.57 | runs/sim-fc-nt2/score.json |
| cp-mirror real time | 96.9 | 96.9 (R 1.00 · S 0.97) | runs/fc-mirror-rt/score.json |
| cp-mirror sim (same tree) | 95.7 | 95.7 | ~/Downloads/sarj-v25/runs/sim-{v25,t5}-cp-mirror/score.json |
| dev real time | 99.0 | 99.0 and 99.0 (2 runs) | runs/fc-dev-rt, runs/fc-dev-rt2 |
| worst dialect F (Omani) | 0.73 | 0.73 (decisions identical; not re-run) | PLAN_V6 §0 |
| false stops (total) | 54 | 54 (decisions identical) | EXPERIMENTS #77 |
| dev RT p90 (ms) | 772 (GI-D02) | same GI-D02 0.8 s late stop | runs/fc-dev-rt/report.md |
| G11 consistency | 99 % | dev RT 99.0 = 99.0 across 2 runs | runs/fc-dev-rt*, |
| G12 wire integrity | pass | pass | runs/logs/t5-g12.log |
| ttbench.ready | pass | pass (commit 2084030) | runs/logs/fc-ready.log |
| soak | pass | 96 calls, 17:53–18:38, no crash, 0 errors, RSS end < peak (3.04 GB load → 0.40 GB) | runs/logs/fc-rss.log, fc-system.log |

## 4. Numbers for the technical report (only if v2.6 is the freeze)
dev-aug 97.51 (0/0) · synth-test 99.55 (0/0) · novel-test 84.70 (9/3) · novel-test-2 85.57 (13/4) · cp-mirror RT 96.9 · dialect table changed? n (decisions identical to v2.5). New in v2.6: every decision carries `label` / `confidence` / `transcript` (INTERFACE.md optional fields).

## 5. External services used by the frozen version
Unchanged from v2.5 (scored path is local). Demo-only addition, not in the frozen scored path: HUMAIN Voice TTS on branch `demo/humain-voice` (not merged).

## 6. Still running
Nothing. The demo server `python -m tools.live` (pid 99279, port 8790) is still up from before. The main tree is back on `demo/stop-hard`.

## 7. Resume here
1. Send Sarj the hash 2084030 (tag v2.6) + the services line from §5.
2. Optional docs: update SUBMISSION.md / WRITEUP.md with the §3 numbers and a labels paragraph, on a docs branch (after the hash, does not change it).
3. Saud's decision: hand in v2.6 (recommended; same score, adds the optional labels) or v2-final.

## 8. Risks / do-not-do
- Do not merge `demo/humain-voice` or `demo/stop-hard` into the scored path; they are demo only.
- Never move or delete `v2-final`, `v2.0`, `release/v2-safe`, `release/v2-final`.
- The sim in a fresh worktree (no .cache/asr) scores dev 98.34 vs 99.3 in the main tree: that is the ASR cache, not the code. Compare only within one tree.
