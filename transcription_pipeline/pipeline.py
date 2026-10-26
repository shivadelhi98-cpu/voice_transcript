"""Audio -> text -> structured downstream output.

Stages:  validate -> normalize audio -> transcribe (pluggable backend, with retry)
         -> clean up text -> enrich (keywords, summary, action items) -> export (JSON/SRT)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol

log = logging.getLogger("pipeline")

SUPPORTED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
MAX_BYTES = 500 * 1024 * 1024


# ----------------------------------------------------------------- data model
@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float | None = None  # 0..1, backend-dependent


@dataclass
class Transcript:
    source: str
    language: str
    duration: float
    segments: list[Segment]
    enrichment: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


# ------------------------------------------------------------------- backends
class STTBackend(Protocol):
    def transcribe(self, wav_path: Path) -> tuple[list[Segment], str, float]:
        """Return (segments, language, duration_seconds)."""


class FasterWhisperBackend:
    """Local, open-source, CPU-friendly (CTranslate2). Model loaded once, lazily."""

    def __init__(self, model_size: str = "base", compute_type: str = "int8"):
        self.model_size, self.compute_type, self._model = model_size, compute_type, None

    def transcribe(self, wav_path):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy: keeps import cost/test deps out
            self._model = WhisperModel(self.model_size, compute_type=self.compute_type)
        raw, info = self._model.transcribe(
            str(wav_path), vad_filter=True, beam_size=5,
            condition_on_previous_text=False,  # reduces repetition loops on long audio
        )
        segs = []
        for s in raw:
            if s.no_speech_prob > 0.8:  # probably silence/noise
                continue
            # avg_logprob is a log-prob; exp() gives a rough 0..1 confidence
            segs.append(Segment(s.start, s.end, s.text.strip(), round(2.718281828 ** s.avg_logprob, 3)))
        return segs, info.language, info.duration


class MockBackend:
    """Deterministic fake for tests/demos. `fail_times` simulates transient errors."""

    def __init__(self, segments=None, fail_times: int = 0):
        self.segments = DEMO_SEGMENTS if segments is None else segments
        self.fail_times, self.calls = fail_times, 0

    def transcribe(self, wav_path):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated transient backend failure")
        duration = self.segments[-1].end if self.segments else 0.0
        return list(self.segments), "en", duration


DEMO_SEGMENTS = [
    Segment(0.0, 4.2, "Um, so welcome everyone to the Q3 planning call.", 0.93),
    Segment(4.2, 9.0, "Priya will send the revised budget to finance by Friday.", 0.91),
    Segment(9.0, 12.5, "We need to fix the login latency issue before the release.", 0.88),
    Segment(12.5, 15.0, "We need to fix the login latency issue before the release.", 0.88),  # dup
    Segment(15.0, 19.5, "Uh the budget review and the release date are the two main risks.", 0.84),
    Segment(19.5, 20.0, "hmm", 0.2),
]


# ------------------------------------------------------------ stage: ingestion
class PipelineError(Exception):
    pass


def validate(path: Path) -> None:
    if not path.is_file():
        raise PipelineError(f"not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXT:
        raise PipelineError(f"unsupported format {path.suffix!r}; expected one of {sorted(SUPPORTED_EXT)}")
    size = path.stat().st_size
    if size == 0:
        raise PipelineError("file is empty")
    if size > MAX_BYTES:
        raise PipelineError(f"file too large ({size} bytes)")


def normalize_audio(src: Path, workdir: Path) -> Path:
    """Convert to 16 kHz mono WAV (what Whisper-family models expect)."""
    if shutil.which("ffmpeg") is None:
        raise PipelineError("ffmpeg not installed")
    out = workdir / (src.stem + ".16k.wav")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise PipelineError(f"ffmpeg failed (corrupt audio?): {r.stderr.strip()[:200]}")
    return out


# --------------------------------------------------- stage: transcribe + retry
def transcribe_with_retry(backend: STTBackend, wav: Path, retries: int = 2, base_delay: float = 0.5):
    for attempt in range(retries + 1):
        try:
            return backend.transcribe(wav)
        except PipelineError:
            raise  # deterministic, retrying won't help
        except Exception as e:
            if attempt == retries:
                raise PipelineError(f"transcription failed after {retries + 1} attempts: {e}") from e
            delay = base_delay * 2 ** attempt
            log.warning("attempt %d failed (%s); retrying in %.1fs", attempt + 1, e, delay)
            time.sleep(delay)


# ------------------------------------------------------------ stage: cleanup
FILLERS = re.compile(r"\b(um+|uh+|erm|hmm+|you know,?|i mean,?)\b[,.]?\s*", re.I)


def clean_segments(segs: list[Segment], min_conf: float = 0.4) -> tuple[list[Segment], list[str]]:
    """Remove fillers, low-confidence fragments and consecutive duplicates (a known Whisper failure mode)."""
    out, warnings, dropped_low = [], [], 0
    for s in segs:
        text = FILLERS.sub("", s.text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        if len(text.split()) < 2 or (s.confidence is not None and s.confidence < min_conf):
            dropped_low += 1
            continue
        if out and out[-1].text.lower() == text.lower():
            out[-1].end = s.end  # extend previous instead of repeating
            continue
        out.append(Segment(s.start, s.end, text, s.confidence))
    if dropped_low:
        warnings.append(f"dropped {dropped_low} short/low-confidence segment(s)")
    return out, warnings


# ----------------------------------------------------------- stage: enrichment
STOP = set("""a an the and or but if so to of in on at for with we i you he she it they is are was were be been
will would can could should need must this that these those by from as before after about just also not no
by two main""".split())
ACTION = re.compile(r"\b(will|needs? to|need to|must|should|has to|have to|going to)\b", re.I)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def keywords(text: str, n: int = 8) -> list[str]:
    words = [w for w in re.findall(r"[a-z][a-z'-]{2,}", text.lower()) if w not in STOP]
    return [w for w, _ in Counter(words).most_common(n)]


def summarize(text: str, k: int = 2) -> list[str]:
    """Extractive: pick the k sentences with the highest keyword density, keep original order."""
    sents = sentences(text)
    kw = set(keywords(text, 10))
    scored = sorted(range(len(sents)),
                    key=lambda i: -sum(w in kw for w in re.findall(r"[a-z']+", sents[i].lower())) / (len(sents[i].split()) or 1))
    return [sents[i] for i in sorted(scored[:k])]


def action_items(segs: list[Segment]) -> list[dict]:
    return [{"text": s.text, "at": round(s.start, 1)} for s in segs if ACTION.search(s.text)]


def enrich(t: Transcript) -> dict:
    return {"keywords": keywords(t.text), "summary": summarize(t.text), "action_items": action_items(t.segments)}


# ----------------------------------------------------------------- exporters
def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000); m, ms = divmod(ms, 60_000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_srt(t: Transcript) -> str:
    return "\n".join(f"{i}\n{_ts(s.start)} --> {_ts(s.end)}\n{s.text}\n" for i, s in enumerate(t.segments, 1))


def to_json(t: Transcript) -> str:
    d = asdict(t); d["text"] = t.text
    return json.dumps(d, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------ orchestrator
def process_file(path: Path, backend: STTBackend, normalize: bool = True) -> Transcript:
    validate(path)
    with tempfile.TemporaryDirectory() as tmp:
        wav = normalize_audio(path, Path(tmp)) if normalize else path
        segs, lang, dur = transcribe_with_retry(backend, wav)
    segs, warns = clean_segments(segs)
    if not segs:
        warns.append("no speech detected")
    t = Transcript(str(path), lang, round(dur, 2), segs, warnings=warns)
    t.enrichment = enrich(t)
    return t


def process_batch(paths: list[Path], backend: STTBackend, out_dir: Path, normalize: bool = True) -> dict:
    """One bad file never kills the batch; failures are collected and reported."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"ok": [], "failed": []}
    for p in paths:
        try:
            t = process_file(p, backend, normalize)
            (out_dir / f"{p.stem}.json").write_text(to_json(t), encoding="utf-8")
            (out_dir / f"{p.stem}.srt").write_text(to_srt(t), encoding="utf-8")
            report["ok"].append(str(p))
        except PipelineError as e:
            log.error("%s: %s", p, e)
            report["failed"].append({"file": str(p), "error": str(e)})
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", type=Path, help="audio files or directories")
    ap.add_argument("-o", "--out", type=Path, default=Path("out"))
    ap.add_argument("--model", default="base", help="faster-whisper model size")
    ap.add_argument("--mock", action="store_true", help="use mock backend (no model/audio needed)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.mock:
        # demo: skip ffmpeg and file validation by feeding a tiny placeholder file
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "demo.wav"; f.write_bytes(b"RIFF")
            t = process_file(f, MockBackend(), normalize=False)
        t.source = "demo.wav"
        print(to_json(t)); return

    if not a.inputs:
        ap.error("provide at least one audio file or directory, or use --mock")

    files, input_failures = [], []
    for path in a.inputs:
        if path.is_dir():
            candidates = sorted(
                f for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
            )
            if not candidates:
                input_failures.append({
                    "file": str(path),
                    "error": "directory contains no supported audio files",
                })
            files.extend(candidates)
        else:
            files.append(path)

    report = process_batch(files, FasterWhisperBackend(a.model), a.out)
    report["failed"].extend(input_failures)
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report["failed"] else 0)


if __name__ == "__main__":
    main()
