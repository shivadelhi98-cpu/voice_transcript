"""Load local project settings from the ignored .env file."""
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent / ".env"


def load_settings(env_file: Path = ENV_FILE) -> None:
    """Load project environment variables without overriding process settings."""
    load_dotenv(dotenv_path=env_file, override=False)
