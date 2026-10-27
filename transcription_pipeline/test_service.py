import io
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from pipeline import FasterWhisperBackend, FallbackBackend, MockBackend
from service import create_app


def wav_bytes(seconds=1, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(struct.pack("<h", 0) * rate * seconds)
    return buf.getvalue()


class FakeCorrector:
    def correct_segments(self, segments):
        return [
            "Welcome everyone to the Q3 planning call."
            if text.startswith("So welcome")
            else text
            for text in segments
        ]


class FailingCorrector:
    def correct_segments(self, segments):
        raise RuntimeError("provider unavailable")


class ServiceTests(unittest.TestCase):
    def make_client(self, corrector=None):
        return TestClient(create_app(MockBackend(), normalize=False, corrector=corrector))

    @staticmethod
    def post(client, data, name="a.wav", content_type="audio/wav", **form_data):
        return client.post(
            "/transcribe",
            files={"file": (name, data, content_type)},
            data=form_data,
        )

    def setUp(self):
        self.client = self.make_client()

    def test_health(self):
        result = self.client.get("/health").json()
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["text_correction_available"])
        self.assertEqual(result["asr_mode"], "mock")

    def test_health_reports_local_fallback_order(self):
        backend = FallbackBackend([FasterWhisperBackend("small"), FasterWhisperBackend("base")])
        client = TestClient(create_app(backend, normalize=False))

        result = client.get("/health").json()

        self.assertEqual(result["asr_mode"], "local")
        self.assertEqual(result["asr_models"], ["small", "base"])

    def test_wav_upload_json(self):
        response = self.post(self.client, wav_bytes(), "meeting.wav")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"], "meeting.wav")
        self.assertIn("action_items", response.json()["enrichment"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for Ogg integration tests")
    def test_ogg_family_uploads_are_normalized_and_transcribed(self):
        client = TestClient(create_app(MockBackend(), normalize=True))
        with tempfile.TemporaryDirectory() as directory:
            for suffix, codec in ((".OGG", "libopus"), (".opus", "libopus"), (".oga", "libvorbis")):
                with self.subTest(suffix=suffix):
                    audio_path = Path(directory) / f"meeting{suffix}"
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-loglevel", "error",
                            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                            "-t", "0.5", "-c:a", codec, "-f", "ogg", str(audio_path),
                        ],
                        check=True,
                        capture_output=True,
                    )

                    response = self.post(
                        client,
                        audio_path.read_bytes(),
                        audio_path.name,
                        content_type="audio/ogg",
                    )

                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["source"], audio_path.name)
                    self.assertTrue(response.json()["segments"])

    def test_srt_format(self):
        response = self.post(self.client, wav_bytes(), **{"format": "srt"})
        self.assertTrue(response.text.startswith("1\n00:00:00,000 --> 00:00:04,200"))

    def test_missing_file(self):
        self.assertEqual(self.client.post("/transcribe").status_code, 422)

    def test_unsupported_type(self):
        response = self.post(self.client, b"hello", "notes.txt")
        self.assertEqual(response.status_code, 415)

    def test_empty_file(self):
        response = self.post(self.client, b"", "a.mp3")
        self.assertEqual(response.status_code, 422)

    def test_bad_format_param(self):
        response = self.post(self.client, wav_bytes(), **{"format": "xml"})
        self.assertEqual(response.status_code, 400)

    def test_correction_is_opt_in_and_preserves_original(self):
        client = self.make_client(FakeCorrector())
        response = self.post(client, wav_bytes(), correct="true")

        self.assertEqual(response.status_code, 200)
        transcript = response.json()
        self.assertTrue(transcript["correction_applied"])
        self.assertTrue(transcript["raw_text"].startswith("So welcome"))
        self.assertTrue(transcript["text"].startswith("Welcome"))
        self.assertEqual(transcript["segments"][0]["start"], 0.0)

    def test_correction_failure_keeps_local_transcript(self):
        client = self.make_client(FailingCorrector())
        response = self.post(client, wav_bytes(), correct="true")

        self.assertEqual(response.status_code, 200)
        transcript = response.json()
        self.assertFalse(transcript["correction_applied"])
        self.assertIsNone(transcript["raw_text"])
        self.assertTrue(transcript["text"].startswith("So welcome"))
        self.assertTrue(any("original transcript kept" in item for item in transcript["warnings"]))

    def test_correction_requires_configured_provider(self):
        response = self.post(self.client, wav_bytes(), correct="true")
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
