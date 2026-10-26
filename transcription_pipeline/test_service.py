import io, struct, unittest, wave
from pipeline import MockBackend
from service import create_app


def wav_bytes(seconds=1, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * rate * seconds)
    return buf.getvalue()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.c = create_app(MockBackend(), normalize=False).test_client()

    def post(self, data, name="a.wav", **kw):
        return self.c.post("/transcribe", data={"file": (io.BytesIO(data), name)},
                           content_type="multipart/form-data", **kw)

    def test_health(self):
        self.assertEqual(self.c.get("/health").json["status"], "ok")

    def test_wav_upload_json(self):
        r = self.post(wav_bytes(), "meeting.wav")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json["source"], "meeting.wav")
        self.assertIn("action_items", r.json["enrichment"])

    def test_srt_format(self):
        r = self.post(wav_bytes(), query_string={"format": "srt"})
        self.assertTrue(r.get_data(as_text=True).startswith("1\n00:00:00,000"))

    def test_missing_file(self):
        self.assertEqual(self.c.post("/transcribe").status_code, 400)

    def test_unsupported_type(self):
        self.assertEqual(self.post(b"hello", "notes.txt").status_code, 415)

    def test_empty_file(self):
        self.assertEqual(self.post(b"", "a.mp3").status_code, 422)

    def test_bad_format_param(self):
        self.assertEqual(self.post(wav_bytes(), query_string={"format": "xml"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
