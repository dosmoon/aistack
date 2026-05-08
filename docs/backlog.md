# aistack Backlog

Single page for "what could come next." Items below are NOT scheduled —
they are candidates that compete for attention each day. The right-most
column is whether picking it up next makes sense; if blocked or
deferred, the reason lives there too.

Format:
- **`now`** — active or about to start
- **`ready`** — well-scoped, no blockers, can pick up any session
- **`later`** — known want, but waiting on signal / gated on something
- **`parked`** — explicit "not now," with reason

Daily progress is in `docs/progress/aistack-YYYY-MM-DD.md`. When an
item moves to "done", it gets retired here and recorded in that day's
chronicle.

---

## Engine / models

| Item | State | Note |
|---|---|---|
| Mandarin ASR routing — SenseVoice as default for `zh*` | done | Already wired (`_select_for_auto`) — no work, just remember it exists. |
| Parakeet VAD threshold tuning | parked | Deferred until repeated real-world signal. See `memory/project_parakeet_vad_deferred.md`. |
| 720 s-seam Parakeet word drop (chunked mode) | parked | Documented limitation — Parakeet can miss short crosstalk segments mid-chunk regardless of attention size or overlap. Not chunking's fault. Mitigation requires switching backend (Whisper handles crosstalk better). |
| Qwen3-TTS `gpu_memory_utilization` cap | later | The unified GPU lock made it less urgent. Paper config is still wrong. |
| First-token latency for LLM streaming | later | Observability records total latency; first-token is the meaningful metric for streaming chat UX. Add when a consumer asks. |

## Bench / evaluation

| Item | State | Note |
|---|---|---|
| Mandarin ASR dataset (Common Voice zh-CN or AISHELL dev) | ready | Needed for SenseVoice quality regression. Mirror `bench/asr_eval.py` shape. |
| Multi-length WER reporting | later | Currently flat-average across all clip lengths. Bucketing by duration (< 5 s / 5–15 s / > 15 s) reveals length-dependent regressions. |
| Historical bench comparison | later | Save `bench/output/<run>/summary.json` with timestamp, write a `bench/diff.py` that compares two runs. |
| Single-pass-vs-chunked A/B (25-min only) | later | Confirms chunked-mode adds zero overall recall loss vs single-pass. Gated on whether single-pass actually completes — VRAM may OOM. |

## Admin / observability

| Item | State | Note |
|---|---|---|
| `POST /admin/api/reset-asr-state` | ready | Manual "drop loaded ASR models, free VRAM" button. Useful between bench runs without restarting uvicorn. |
| Prometheus `/metrics` OpenMetrics exporter | later | Add when there's a Grafana on the other end. JSON `/admin/api/metrics` already covers self-written scripts. |
| D4 — install / uninstall actions in admin UI | ready | Closes the loop on D4 (read-only model browser → full mgmt). Concrete next product slice. |

## Architecture / cleanup

| Item | State | Note |
|---|---|---|
| `aistack/backends/` refactor | parked | Gated on a real second LLM backend existing (today only Ollama). |
| Doc-string vs code env-name drift check | later | One-shot script that scans aistack/ for `AISTACK_*` mentions in comments and verifies they exist in `aistack/config.py`. Cheap; catches future drift. |

---

## Recently done (last 7 days, retired here)

| Date | What |
|---|---|
| 2026-05-08 | Centralized env-driven config in `aistack/config.py`; user-facing `docs/configuration.md`. |
| 2026-05-08 | Parakeet long-audio chunked transcription (12-min windows, 2-min overlap, word-LCS stitch) + bench experiment harness. Solves long-audio OOM, 1-segment-for-50-min, 4× wall-time variance. |
| 2026-05-08 | Bench reference data: 4 long mp3s (12/25/50/97 min) + en/zh transcripts as ground truth. |
| 2026-05-08 | LibriSpeech dev-clean WER bench (`bench/asr_eval.py`). |
| 2026-05-07 | D5 observability — metrics + JSONL access log + payload capture + X-Request-ID. |
| 2026-05-07 | SSE streaming for ASR — real for Whisper/SenseVoice, downgrade for Parakeet. |
| 2026-05-07 | `/v1/models` advertises `languages`, `supports_streaming`, `auto` routing alias. |
| 2026-05-07 | Unified error envelope — slot-busy 503 stops leaking `{detail: ...}`. |
