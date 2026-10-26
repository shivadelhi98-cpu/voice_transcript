"""HTTP service: upload an audio file, get a transcript back.

  POST /transcribe          multipart field `file`; optional ?format=json|srt (default json)
  GET  /health

Run:  python service.py                       (real model, needs ffmpeg + faster-whisper)
      STT_BACKEND=mock python service.py      (no model needed)
"""
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request

from pipeline import (FasterWhisperBackend, MockBackend, PipelineError, STTBackend,
                      SUPPORTED_EXT, MAX_BYTES, process_file, to_json, to_srt)

log = logging.getLogger("service")


def create_app(backend: STTBackend, normalize: bool = True) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    # Whisper inference is CPU/GPU bound and the model object is shared; run one job
    # at a time per process and scale out with more worker processes instead.
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "ffmpeg": shutil.which("ffmpeg") is not None}

    @app.post("/transcribe")
    def transcribe():
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify(error="send the audio as multipart field 'file'"), 400
        suffix = Path(f.filename).suffix.lower()
        if suffix not in SUPPORTED_EXT:
            return jsonify(error=f"unsupported format {suffix!r}", supported=sorted(SUPPORTED_EXT)), 415
        fmt = request.args.get("format", "json")
        if fmt not in ("json", "srt"):
            return jsonify(error="format must be json or srt"), 400

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"  # never use the client's filename on disk
            f.save(path)
            try:
                with lock:
                    t = process_file(path, backend, normalize)
            except PipelineError as e:
                log.warning("rejected %s: %s", f.filename, e)
                return jsonify(error=str(e)), 422
            except Exception:
                log.exception("unexpected failure")
                return jsonify(error="internal error"), 500

        t.source = f.filename  # report the original name, not the temp path
        if fmt == "srt":
            return Response(to_srt(t), mimetype="text/plain")
        return Response(to_json(t), mimetype="application/json")

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"file exceeds {MAX_BYTES // 2**20} MB"), 413

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if os.getenv("STT_BACKEND") == "mock":
        app = create_app(MockBackend(), normalize=False)
    else:
        app = create_app(FasterWhisperBackend(os.getenv("WHISPER_MODEL", "base")))
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
