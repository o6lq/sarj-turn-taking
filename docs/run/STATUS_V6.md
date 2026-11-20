# STATUS — PLAN_V6 run (Fri 20 Nov)

## CURRENT STATE  (updated 18:47)
scored-path HEAD : main @ 2084030 = tag v2.6 (pushed)   freeze-ready: yes
baseline         : v2.5 @ 7c52fdd · fallback v2-final @ aab8988 (protected 3 ways)
T1 dialect-lexicon : dropped  (time)
T2 ack-but         : dropped  (time)
T3 either-pref     : dropped  (17:15, no cp-am-dev)
T4 perf (identical): dropped  (time)
T5 labels          : merged  PR #28 → 2084030
best numbers vs v2.5: nt2 85.57=85.57 / mirror RT 96.9=96.9 / worst-dialect F 0.73 (unchanged) / false stops 54 (identical) / dev RT 99.0 ×2 / G11 pass / G12 pass
running now      : nothing (demo server pid 99279 still up; main tree on demo/stop-hard)
next step        : HANDOFF written (docs/run/HANDOFF_V6.md). Holding the reserve.

## LOG
17:23  —   run start: read RUN_PROTOCOL.md + PLAN_V6.md (Downloads); no T1–T5 branches exist; 17:15 checkpoint missed → T3 dropped
17:25  —   CLAUDE.md protocol pointer → demo/stop-hard @ c5a5bf5 (pushed)
17:26  —   STATUS_V6.md created (untracked)
17:44  T5  screen: identical decisions dev/nt2/mirror (same tree) · G12 pass (runs/logs/t5-g12.log) · pytest 90 pass (runs/logs/t5-pytest.log)
17:47  T5  PR #28 opened (feat/decision-labels @ 08d4567); EXPERIMENTS #77
17:48  T5  merged → main @ 2084030 · T1/T2/T4 dropped (time)
17:53  FC  final check started on main @ 2084030 → runs/logs/fc-*.log, runs/fc-{dev-rt,mirror-rt,dev-rt2}
18:13  —   SAFETY: v2-final aab8988 protected: tags v2-final/v2.0/v2.5 on GitHub + branch release/v2-final (pushed) + offline bundle ~/Downloads/sarj-backups/sarj-v2-final-aab8988.bundle (verified). Saud approved tagging/pushing (18:12).
18:38  FC  final check done: ready pass · dev RT 99.0 · mirror RT 96.9 · dev RT2 99.0 · 96 calls no crash (runs/fc-*, runs/logs/fc-*)
18:44  FC  nt2 sim main tree 85.57 (runs/sim-fc-nt2) = v2.5
18:45  —   tag v2.6 → 2084030 pushed (Saud approved 18:12)
18:47  —   HANDOFF_V6.md written
