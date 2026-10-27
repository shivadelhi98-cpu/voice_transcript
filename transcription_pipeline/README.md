# Local Voice Transcript

Upload an audio recording, transcribe it locally with a pretrained Hugging Face
Whisper model, and get a cleaned transcript with timestamps, keywords, a short
summary, action items, and JSON/SRT downloads.

Audio stays on your machine for transcription. Optional LLM correction is
disabled by default; when enabled, only transcript text (never the audio) is sent
to the configured text model. The raw local transcript is preserved.

## Architecture

```text
Streamlit UI
    -> FastAPI upload /transcribe
        -> local audio validation and FFmpeg normalization
        -> faster-whisper model downloaded from Hugging Face and run locally
        -> cleanup, timestamps, keywords, summary, action items
        -> optional text-only correction (OpenAI API; disabled unless configured)
    <- JSON transcript
    -> JSON and SRT downloads
```

FastAPI keeps the UI separate from transcription. The ASR backend is replaceable;
optional local ASR fallbacks are tried in order. Cloud ASR is not configured and
audio is never silently sent to a cloud transcription service. If correction is
unavailable or fails, the app keeps the original transcript and reports a warning.

## Windows setup

Install Python and [FFmpeg](https://ffmpeg.org/) first. Confirm `ffmpeg -version`
works in a new PowerShell window, then open PowerShell in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Put local credentials in `transcription_pipeline/.env`. The checked-in
`.env.example` is only a template; `.env` is ignored by Git and must never be
committed. Settings from the shell take precedence over values in the file.
`OPENAI_API_KEY` and `HF_TOKEN` are optional; leave their example lines
commented unless you need those providers.

Start the API in the first terminal:

```powershell
python -m uvicorn service:app --host 127.0.0.1 --port 8000
```

Start the UI in a second terminal:

```powershell
cd C:\path\to\voice_transcript\transcription_pipeline
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Open the Streamlit URL it prints (normally <http://localhost:8501>). FastAPI's
interactive API docs are at <http://127.0.0.1:8000/docs>.

The default ASR model is `small`, a multilingual model downloaded from Hugging
Face the first time it is used. Change model and fallbacks in the API terminal
before starting FastAPI:

```powershell
$env:WHISPER_MODEL = "small"
$env:WHISPER_FALLBACK_MODELS = "base"
python -m uvicorn service:app --host 127.0.0.1 --port 8000
```

Model fallbacks are local and tried only if the previous model errors. They use
additional disk space and may download extra weights. Pick models that fit the
machine and benchmark them on representative recordings before production use.
`large-v3-turbo` is another multilingual candidate; English-only Distil-Whisper
models should not be selected for Hindi or mixed Hindi-English recordings
without evaluating them first.

## Optional text-only correction

Local transcription and the UI work without an LLM account or API key. To enable
the optional correction checkbox, uncomment `OPENAI_API_KEY` in `.env`, add your
key, and restart FastAPI:

Never commit the key or put it in Streamlit code. The correction step is
text-only, opt-in per request, and asks for conservative spelling, punctuation,
and grammar edits without changing facts or segment timestamps. The app keeps
the raw local transcript alongside any corrected text. Review corrections,
especially for names, numbers, dates, and code-switched speech.

As an example of the cost scale, the published OpenAI API price for GPT-5 nano
is $0.05 per million input tokens and $0.40 per million output tokens at the time
this was documented. A 15,000-token input and 15,000-token output would be about
$0.00675; actual usage varies by transcript and edits. Prices and account
availability change. See [OpenAI API pricing](https://developers.openai.com/api/docs/pricing).
No LLM call happens unless correction is configured and selected.

Public Hugging Face Whisper models download without a token. If you later choose
a private or gated Hugging Face model, create a read-only token and set
`HF_TOKEN` in `.env`; `huggingface_hub` reads this standard environment
variable. Never put either token in Python source, Streamlit settings, or Git.

## Run and test

```powershell
python -m unittest discover -s . -p "test_*.py" -v
```

Run the API with deterministic sample output (no model download):

```powershell
$env:STT_BACKEND = "mock"
python -m uvicorn service:app --host 127.0.0.1 --port 8000
```

Tests cover audio validation, local-backend fallback, correction validation and
failure behavior, API request validation, and response formats. Mock tests do
not measure speech recognition quality. Evaluate the ASR models using recordings
you have permission to process and calculate WER/CER against verified transcripts.

## API

- `GET /health`: API, FFmpeg, and optional correction-provider readiness.
- `POST /transcribe`: multipart `file`; optional `format=json|srt` and
  `correct=true|false` form fields.

Typical errors are 400 for invalid options, 413 for an upload over 500 MB, 415
for an unsupported extension, 422 for an unreadable file, and 503 when correction
is requested but not configured.

## Limitations and next steps

- ASR quality and latency depend on model size, recording quality, language,
  and available hardware. The default has not been benchmarked against your own
  speech data.
- Whisper's multilingual capability is not a guarantee of accurate Hindi-English
  code-switching. Keep raw transcripts and evaluate corrections before relying
  on them.
- Rule-based summaries and action items are heuristic, not verified facts.
- The API currently handles each transcription in a synchronous worker thread.
  Add job IDs and a queue if concurrent or lengthy production workloads require
  background processing.
- Review licenses and model terms for the selected Hugging Face checkpoint and
  dependencies before commercial deployment.
