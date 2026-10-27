import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from settings import load_settings


class SettingsTests(unittest.TestCase):
    def test_loads_values_from_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("VOICE_TRANSCRIPT_TEST_SETTING=from-file\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                load_settings(env_file)
                self.assertEqual(os.environ["VOICE_TRANSCRIPT_TEST_SETTING"], "from-file")

    def test_process_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("VOICE_TRANSCRIPT_TEST_SETTING=from-file\n", encoding="utf-8")

            with patch.dict(os.environ, {"VOICE_TRANSCRIPT_TEST_SETTING": "from-process"}):
                load_settings(env_file)
                self.assertEqual(os.environ["VOICE_TRANSCRIPT_TEST_SETTING"], "from-process")


if __name__ == "__main__":
    unittest.main()
