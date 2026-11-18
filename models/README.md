# Models

These are the official model files. The system loads them automatically.

| File | Used for | Used on |
|---|---|---|
| `policy-cohere.json` | Decision weights for the Cohere speech engine | Mac |
| `policy-faster.json` | Decision weights for the faster-whisper speech engine | Windows and Linux |
| `policy.json` | Fallback decision weights for any other engine | Any |
| `student/` | Meaning model (CAMeLBERT-DA student) | All |
| `silero_vad.onnx` | Voice activity detection | All |

`policy-faster.json` is created by this command:

```bash
TT_ASR=faster .venv/bin/python -m tools.train_policy --train data/synth
```

## Archive

`_archive/` holds older experiments that the system does not load. It is git-ignored and stays on the machine that created it.

| Item | What it was |
|---|---|
| `policy-big-lr.json`, `policy-big-lgbm.json` | Weights trained on the larger generated set |
| `policy-lr-ac.json`, `policy-v12-lr.json` | Earlier decision weights |
| `student-camel/`, `student-saudi/` | Earlier meaning-model variants |
