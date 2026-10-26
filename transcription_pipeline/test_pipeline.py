import tempfile, unittest
from pathlib import Path
from pipeline import *


def fake_audio(suffix=".wav", data=b"RIFFxxxx"):
    d = tempfile.mkdtemp(); p = Path(d) / f"a{suffix}"; p.write_bytes(data); return p


class Tests(unittest.TestCase):
    def test_end_to_end_mock(self):
        t = process_file(fake_audio(), MockBackend(), normalize=False)
        self.assertEqual(len(t.segments), 4)                      # dup merged, "hmm" dropped
        self.assertNotIn("Um", t.text); self.assertNotIn("Uh", t.text)
        self.assertEqual(t.segments[2].end, 15.0)                 # duplicate extended previous
        self.assertEqual(len(t.enrichment["action_items"]), 2)
        self.assertIn("budget", t.enrichment["keywords"])

    def test_retry_recovers(self):
        b = MockBackend(fail_times=2)
        segs, *_ = transcribe_with_retry(b, Path("x"), retries=2, base_delay=0)
        self.assertEqual(b.calls, 3); self.assertTrue(segs)

    def test_retry_gives_up(self):
        with self.assertRaises(PipelineError):
            transcribe_with_retry(MockBackend(fail_times=9), Path("x"), retries=1, base_delay=0)

    def test_validation(self):
        with self.assertRaises(PipelineError): validate(Path("nope.wav"))
        with self.assertRaises(PipelineError): validate(fake_audio(".txt"))
        with self.assertRaises(PipelineError): validate(fake_audio(data=b""))

    def test_batch_isolates_failures(self):
        good, bad = fake_audio(), fake_audio(".txt")
        r = process_batch([good, bad], MockBackend(), Path(tempfile.mkdtemp()), normalize=False)
        self.assertEqual((len(r["ok"]), len(r["failed"])), (1, 1))

    def test_silence_warns(self):
        b = MockBackend(segments=[Segment(0, 1, "uh", 0.9)])
        t = process_file(fake_audio(), b, normalize=False)
        self.assertIn("no speech detected", t.warnings)

    def test_empty_mock_transcript(self):
        t = process_file(fake_audio(), MockBackend(segments=[]), normalize=False)
        self.assertEqual(t.duration, 0)
        self.assertEqual(t.segments, [])
        self.assertIn("no speech detected", t.warnings)

    def test_srt_format(self):
        t = process_file(fake_audio(), MockBackend(), normalize=False)
        self.assertTrue(to_srt(t).startswith("1\n00:00:00,000 --> 00:00:04,200"))


if __name__ == "__main__":
    unittest.main()
