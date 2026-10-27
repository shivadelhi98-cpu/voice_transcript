import unittest

from correction import apply_correction
from pipeline import Segment, Transcript


class FakeCorrector:
    def __init__(self, corrected):
        self.corrected = corrected

    def correct_segments(self, segments):
        return self.corrected


class CorrectionTests(unittest.TestCase):
    def make_transcript(self):
        return Transcript(
            source="meeting.wav",
            language="en",
            duration=4.0,
            segments=[Segment(0.0, 4.0, "we has a plan", 0.9)],
        )

    def test_applies_correction_and_preserves_timestamps(self):
        transcript = self.make_transcript()

        applied = apply_correction(transcript, FakeCorrector(["We have a plan."]))

        self.assertTrue(applied)
        self.assertEqual(transcript.raw_text, "we has a plan")
        self.assertTrue(transcript.correction_applied)
        self.assertEqual(transcript.segments[0].text, "We have a plan.")
        self.assertEqual(transcript.segments[0].start, 0.0)
        self.assertEqual(transcript.segments[0].end, 4.0)

    def test_rejects_wrong_segment_count(self):
        with self.assertRaisesRegex(ValueError, "different number"):
            apply_correction(self.make_transcript(), FakeCorrector([]))

    def test_rejects_empty_correction(self):
        with self.assertRaisesRegex(ValueError, "empty segment"):
            apply_correction(self.make_transcript(), FakeCorrector([" "]))

    def test_noop_does_not_replace_transcript(self):
        transcript = self.make_transcript()

        applied = apply_correction(transcript, FakeCorrector(["we has a plan"]))

        self.assertFalse(applied)
        self.assertIsNone(transcript.raw_text)
        self.assertFalse(transcript.correction_applied)


if __name__ == "__main__":
    unittest.main()
