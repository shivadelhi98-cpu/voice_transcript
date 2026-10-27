# Voice Transcription Application

A local-first voice transcription application using a Streamlit UI,
FastAPI service, and Faster-Whisper. It returns transcript segments with
timestamps and supports JSON and SRT output.

## Architecture

``` text
Streamlit UI
    -> FastAPI API
        -> input validation
        -> FFmpeg audio normalization when needed
        -> local Faster-Whisper transcription
        -> transcript post-processing (timestamps, keywords, summary, action items)
        -> optional text-only correction (opt-in)
    <- JSON response or SRT output
```

## Design decisions

### 1. Local-first speech recognition

Faster-Whisper runs locally, so audio is not silently sent to a cloud
transcription service. This gives users more control over audio privacy
and keeps cloud speech-to-text credentials out of the core workflow. The
trade-off is that speed and accuracy depend on model size, hardware,
language, and recording quality.

### 2. Separate API and UI

FastAPI handles HTTP requests and transcription, while Streamlit
provides the user interface. This separation allows other clients to
call the API without using Streamlit and makes the components easier to
test independently.

### 3. FFmpeg for audio compatibility

Audio files can differ in codec, sample rate, and channel layout. FFmpeg
is used to decode or normalize audio where needed, including converting
OGG/Opus-family inputs to mono 16 kHz WAV before transcription. FFmpeg
must be installed and available on the system path.

### 4. Segment timestamps and output formats

The transcript includes segment-level start and end times. These
timestamps help align text with the recording and generate subtitles.
JSON supports programmatic integration; SRT supports subtitle workflows.

### 5. Optional text correction

Text correction is optional and disabled unless configured and selected.
Only transcript text---not the audio---is sent to the configured text
model. The raw local transcript is preserved. This keeps core
transcription independent of an LLM API key, while recognizing that
corrected names, dates, numbers, and mixed-language speech may need
review.

### 6. Configurable ASR backend and fallbacks

The backend and local fallback models can be configured to suit
available hardware and language requirements. Fallbacks may require
additional disk space and have different speed/accuracy trade-offs, so
they should be evaluated on representative recordings.

### 7. Tests that do not require model inference

Automated tests cover validation, backend fallback behavior, correction
handling, and response formats. A mock backend enables predictable API
tests without downloading model weights or running inference. Mock tests
validate application behavior; they do not measure real-world
speech-recognition accuracy. Evaluate accuracy separately using verified
transcripts and metrics such as Word Error Rate (WER) and Character
Error Rate (CER).

## API overview

-   `GET /health` --- reports API and dependency readiness.
-   `POST /transcribe` --- accepts an audio upload as multipart form
    data.
-   `format` --- selects `json` or `srt` output.
-   `correct` --- optional flag for text correction when configured.

Typical errors include invalid options, oversized uploads, unsupported
file extensions, unreadable audio, and correction requested without a
configured provider. See the actual API response for exact details.

## Run locally

Install Python dependencies from `requirements.txt` and install FFmpeg
separately. Configure optional credentials in a local `.env` file based
on `.env.example`; never commit `.env` or API keys.

Start the API from the project directory:

``` powershell
python -m uvicorn service:app --host 127.0.0.1 --port 8000
```

Start the UI in a second terminal:

``` powershell
python -m streamlit run app.py
```

FastAPI documentation is normally available at
`http://127.0.0.1:8000/docs`.

Run tests:

``` powershell
python -m unittest discover -s . -p "test_*.py" -v
```

To exercise the API with deterministic mock output:

``` powershell
$env:STT_BACKEND = "mock"
python -m uvicorn service:app --host 127.0.0.1 --port 8000
```

## Current limitations

-   Accuracy and latency depend on audio quality, language, model size,
    and hardware.
-   Multilingual capability does not guarantee accurate Hindi-English
    code-switching.
-   Rule-based summaries, keywords, and action items are heuristic and
    may need review.
-   Transcription is handled synchronously in a worker thread; there is
    no persistent background-job system.
-   Mock tests do not measure speech-recognition quality.
-   Persistent transcript history is not part of the current workflow
    described here.

## Possible future improvements

These are proposals, not claims about the current implementation:

1.  Add background jobs with job IDs and status polling for long
    recordings or higher traffic.
2.  Add authentication, rate limiting, and per-user quotas before
    exposing the API publicly.
3.  Add structured logs, latency metrics, error monitoring, and
    model-quality evaluation.
4.  If transcript history is needed, persist transcript metadata and
    segments in a database; store large audio files in object storage
    only when retention is required.
5.  Add configurable retention and deletion policies.
6.  Benchmark models on representative, consented recordings using
    WER/CER and latency.
7.  Review dependency and model licenses before commercial deployment.

## Privacy and credentials

Keep API keys and private tokens in environment variables or a local
`.env` file. Never commit credentials to source control. Process
recordings only with permission, and review privacy implications before
enabling an external text-correction provider.
