# Transcription Pipeline

Audio file(s) -> clean transcript -> structured output (JSON, SRT) with keywords, summary and action items.

## Quick start (Windows PowerShell)

Run these commands from the `transcription_pipeline` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python pipeline.py --mock
python -m unittest discover -s . -p "test_*.py" -v
```

The mock command needs neither a model nor audio. For real audio, install
[FFmpeg](https://ffmpeg.org/) and make sure `ffmpeg` is on `PATH`, then run:

```powershell
python pipeline.py .\recordings -o .\out --model small
```

The pipeline writes one JSON and one SRT file per supported audio file. Supported
formats are WAV, MP3, M4A, FLAC, OGG, WEBM, and MP4. The first real run downloads
the selected faster-whisper model.

To run the mock HTTP service:

```powershell
$env:STT_BACKEND = "mock"
python service.py
```

For a real service, leave `STT_BACKEND` unset. Upload an audio file to
`http://localhost:8000/transcribe` as multipart field `file`; add `?format=srt`
for subtitles. `GET /health` reports service and FFmpeg status.

On macOS/Linux, replace the virtual-environment activation command with
`source .venv/bin/activate` and use platform-appropriate paths.

## Preparing for GitHub

Create the Git repository in this `transcription_pipeline` folder. Do not upload
`.venv`, generated `out` files, or private recordings. A `.gitignore` is included.
The project brief, engineering decisions, and known limitations are documented
below for discussion during review.

## Flow
`validate -> ffmpeg to 16k mono wav -> transcribe (retry/backoff) -> clean -> enrich -> export`

## Engineering decisions
| Decision | Why |
|---|---|
| **faster-whisper, local** | Open source, no per-minute cost, audio never leaves the machine (matters for call recordings/PII), int8 runs on CPU. |
| **Backend behind a 1-method Protocol** | Swap in a hosted API or another model without touching the pipeline. Also makes tests fast and deterministic (MockBackend). |
| **Normalize with ffmpeg first** | One code path for mp3/m4a/webm/etc.; also surfaces corrupt files early with a clear error. |
| **Clean-up stage** | Raw ASR output is not "downstream ready": fillers, silence hallucinations, and Whisper's repeated-segment loops are removed; duplicates are merged rather than dropped so timestamps stay correct. |
| **Confidence + no-speech filtering** | Cheap guard against garbage text from noise. Thresholds are configurable and drops are reported in `warnings`, not silent. |
| **Retry only transient errors** | Exponential backoff for backend failures; validation errors fail fast. |
| **Batch isolation** | One bad file is recorded in the report and the run continues; exit code is non-zero if any failed. |
| **Segments with timestamps preserved** | Downstream consumers (subtitles, search-to-timestamp, speaker alignment) need them; flat text is derived. |
| **Rule-based enrichment** | Keywords, extractive summary and action items are deterministic, free, and testable. Good enough to show the interface. |

## Known limitations / what I'd do next
- **No speaker diarization** (pyannote would be the next stage; the `Segment` model has room for a `speaker` field).
- **Enrichment is heuristic.** In production I'd replace `summarize`/`action_items` with an LLM call using structured output, keeping the rule-based version as a fallback and for evals.
- **Sequential processing.** For volume: a queue (SQS/Celery) with one model-loaded worker per GPU/core; the model is already loaded once per process.
- **Very long audio** relies on faster-whisper's internal VAD/windowing; I'd add explicit chunking with overlap if memory became a problem.
- **Quality measurement:** add a small labelled set and track WER per release; the mock-based tests only verify plumbing, not accuracy.
- **Language:** auto-detected; no handling for code-switching (e.g. Hindi/English), where a larger model or forced language per chunk helps.
- Cleanup regexes are English-only.

## Review talking points

- **What does it do?** Accepts audio files, normalizes them to 16 kHz mono WAV, transcribes them, cleans and enriches the transcript, then exports JSON and SRT with timestamps.
- **Why faster-whisper?** It is an open-source local backend, so recordings do not need to be sent to a hosted transcription API. The trade-off is that users need FFmpeg, a compatible runtime, and a model download on first use.
- **Why define an `STTBackend` protocol?** It keeps transcription separate from orchestration. A different model or a deterministic mock can be substituted without changing validation, cleanup, enrichment, or export.
- **How are failures handled?** Invalid files fail validation immediately. Transient backend errors get bounded exponential retries. Batch processing records per-file failures and continues, while the command exits non-zero if any file failed.
- **How is it tested?** Tests use the mock backend to check output shape, cleanup, retry behavior, batch isolation, silence handling, SRT formatting, and HTTP request validation. This verifies pipeline behavior, not recognition accuracy.
- **What would you improve for production?** Measure word error rate against labeled recordings, add diarization and stronger language-aware cleanup, process long recordings asynchronously, and replace or evaluate heuristic enrichment against structured LLM output.

## Service details

The service returns JSON by default; use `?format=srt` for subtitles. Status
codes are 400 for invalid requests, 413 for oversized uploads, 415 for
unsupported types, 422 for unreadable audio, and 500 for unexpected failures.
Client filenames are never used on disk, temporary files are always cleaned up,
and a process handles one transcription at a time. For production scale-out,
use multiple worker processes; for long recordings, move work to an async job
queue instead of holding the request open.
