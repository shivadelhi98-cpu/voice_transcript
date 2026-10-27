"""FastAPI service for local audio transcription and optional text correction."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from correction import OpenAITextCorrector, TextCorrector, apply_correction
from settings import load_settings

load_settings()

from pipeline import (
    MAX_BYTES,
    SUPPORTED_EXT,
    FasterWhisperBackend,
    FallbackBackend,
    MockBackend,
    PipelineError,
    STTBackend,
    enrich,
    process_file,
    to_json,
    to_srt,
)

log = logging.getLogger("service")


def create_app(
    backend: STTBackend,
    normalize: bool = True,
    corrector: TextCorrector | None = None,
) -> FastAPI:
    app = FastAPI(title="Local Voice Transcription API", version="2.0.0")

    @app.get("/health")
    def health():
        backends = backend.backends if isinstance(backend, FallbackBackend) else [backend]
        return {
            "status": "ok",
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "text_correction_available": corrector is not None,
            "asr_mode": "mock" if isinstance(backend, MockBackend) else "local",
            "asr_models": [
                getattr(item, "model_size", "mock" if isinstance(item, MockBackend) else type(item).__name__)
                for item in backends
            ],
        }

    @app.post("/transcribe")
    def transcribe(
        file: UploadFile = File(...),
        output_format: str = Form("json", alias="format"),
        correct: bool = Form(False),
    ):
        if not file.filename:
            raise HTTPException(status_code=400, detail="upload must have a filename")

        filename = Path(file.filename.replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXT:
            raise HTTPException(
                status_code=415,
                detail={"message": f"unsupported format {suffix!r}", "supported": sorted(SUPPORTED_EXT)},
            )
        if output_format not in ("json", "srt"):
            raise HTTPException(status_code=400, detail="format must be json or srt")
        if correct and corrector is None:
            raise HTTPException(
                status_code=503,
                detail="text correction is not configured; set OPENAI_API_KEY to enable it",
            )

        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                audio_path = Path(temporary_directory) / f"upload{suffix}"
                total_bytes = 0
                with audio_path.open("wb") as saved_file:
                    while chunk := file.file.read(1024 * 1024):
                        total_bytes += len(chunk)
                        if total_bytes > MAX_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"file exceeds {MAX_BYTES // 2**20} MB",
                            )
                        saved_file.write(chunk)

                if total_bytes == 0:
                    raise HTTPException(status_code=422, detail="file is empty")

                transcript = process_file(audio_path, backend, normalize)

            transcript.source = filename
            if correct and corrector is not None:
                try:
                    apply_correction(transcript, corrector)
                    transcript.enrichment = enrich(transcript)
                except Exception as error:
                    log.warning("text correction failed; retaining local transcript: %s", error)
                    transcript.warnings.append(
                        f"text correction failed; original transcript kept: {error}"
                    )

            if output_format == "srt":
                return Response(to_srt(transcript), media_type="text/plain")
            return JSONResponse(content=json.loads(to_json(transcript)))
        except HTTPException:
            raise
        except PipelineError as error:
            log.warning("audio processing failed for %s: %s", filename, error)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            log.exception("unexpected transcription failure")
            raise HTTPException(status_code=500, detail="internal transcription error") from error

    return app


def create_default_app() -> FastAPI:
    if os.getenv("STT_BACKEND", "").lower() == "mock":
        backend: STTBackend = MockBackend()
        normalize = False
    else:
        model_name = os.getenv("WHISPER_MODEL", "small")
        models = [model_name]
        models.extend(
            name.strip()
            for name in os.getenv("WHISPER_FALLBACK_MODELS", "").split(",")
            if name.strip() and name.strip() != model_name
        )
        backend = FallbackBackend([FasterWhisperBackend(name) for name in models])
        normalize = True

    corrector = (
        OpenAITextCorrector(os.getenv("OPENAI_TEXT_MODEL", "gpt-5-nano"))
        if os.getenv("OPENAI_API_KEY")
        else None
    )
    return create_app(backend, normalize=normalize, corrector=corrector)


app = create_default_app()
