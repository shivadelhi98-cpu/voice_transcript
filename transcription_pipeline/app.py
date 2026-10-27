"""Streamlit interface for the FastAPI transcription service."""
from __future__ import annotations

import json
import os

import requests
import streamlit as st

from settings import load_settings

load_settings()

API_URL = os.getenv("TRANSCRIPTION_API_URL", "http://127.0.0.1:8000").rstrip("/")


def srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def transcript_to_srt(transcript: dict) -> str:
    entries = [
        f"{index}\n{srt_timestamp(segment['start'])} --> {srt_timestamp(segment['end'])}\n"
        f"{segment['text']}\n"
        for index, segment in enumerate(transcript["segments"], start=1)
    ]
    return "\n".join(entries)


st.set_page_config(page_title="Voice Transcript", layout="wide")
st.title("Voice Transcript")
st.caption("Local Hugging Face speech recognition with optional text-only correction.")

try:
    health_response = requests.get(f"{API_URL}/health", timeout=3)
    health_response.raise_for_status()
    health = health_response.json()
    st.success(
        f"API ready · FFmpeg: {'available' if health.get('ffmpeg') else 'not found'} · "
        f"Text correction: {'configured' if health.get('text_correction_available') else 'off'}"
    )
    if health.get("asr_mode") == "mock":
        st.warning("The API is in mock mode and returns sample text; no audio recognition is performed.")
    elif health.get("asr_models"):
        st.caption("Local ASR model fallback order: " + " -> ".join(health["asr_models"]))
except requests.RequestException as error:
    health = {}
    st.error(f"Cannot reach the transcription API at {API_URL}: {error}")

uploaded_audio = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg", "oga", "opus", "webm", "mp4"],
)

correction_available = health.get("text_correction_available", False)
correct_text = st.checkbox(
    "Optionally correct spelling and punctuation using a text-only LLM",
    value=False,
    disabled=not correction_available,
    help="Audio is transcribed locally. Only the resulting transcript text is sent for correction.",
)

if not correction_available:
    st.caption("Text correction is off. Set OPENAI_API_KEY on the API server to enable this optional feature.")

if st.button("Transcribe", type="primary", disabled=uploaded_audio is None):
    st.session_state.pop("transcript_result", None)
    with st.spinner("Transcribing locally. The first run may download the selected model..."):
        try:
            response = requests.post(
                f"{API_URL}/transcribe",
                files={
                    "file": (
                        uploaded_audio.name,
                        uploaded_audio.getvalue(),
                        uploaded_audio.type or "application/octet-stream",
                    )
                },
                data={"format": "json", "correct": str(correct_text).lower()},
                timeout=(10, 3600),
            )
            response.raise_for_status()
            st.session_state["transcript_result"] = response.json()
        except requests.RequestException as error:
            detail = error.response.text if error.response is not None else str(error)
            st.error(f"Transcription request failed: {detail}")

transcript = st.session_state.get("transcript_result")
if transcript:
    st.subheader(transcript["source"])
    st.caption(
        f"Detected language: {transcript['language']} · "
        f"Duration: {transcript['duration']:.1f} seconds · "
        f"Text correction applied: {'yes' if transcript.get('correction_applied') else 'no'}"
    )

    if transcript.get("raw_text") is not None:
        with st.expander("Original local transcript"):
            st.write(transcript["raw_text"])

    st.text_area("Transcript", value=transcript["text"], height=220)

    columns = st.columns(2)
    with columns[0]:
        st.markdown("**Keywords**")
        st.write(", ".join(transcript["enrichment"]["keywords"]) or "None detected")
        st.markdown("**Summary**")
        for sentence in transcript["enrichment"]["summary"]:
            st.write(f"- {sentence}")
    with columns[1]:
        st.markdown("**Action items**")
        for item in transcript["enrichment"]["action_items"]:
            st.write(f"- {item['text']} *(at {item['at']:.1f}s)*")
        if not transcript["enrichment"]["action_items"]:
            st.write("None detected")

    for warning in transcript["warnings"]:
        st.warning(warning)

    st.download_button(
        "Download JSON",
        data=json.dumps(transcript, ensure_ascii=False, indent=2),
        file_name="transcript.json",
        mime="application/json",
    )
    st.download_button(
        "Download SRT",
        data=transcript_to_srt(transcript),
        file_name="transcript.srt",
        mime="text/plain",
    )
