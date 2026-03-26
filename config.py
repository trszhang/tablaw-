import os
from pathlib import Path

def load_settings():
    settings_path = Path(__file__).parent / "setting.txt"
    settings = {}
    if settings_path.exists():
        # Always read settings as UTF-8 to avoid Windows locale decode issues.
        with open(settings_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    settings[key.strip()] = value.strip().strip('"').strip("'")
    return settings

_s = load_settings()
API_KEY = os.getenv("API_KEY") or _s.get("API_KEY", "")
BASE_URL = os.getenv("BASE_URL") or _s.get("BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("MODEL_NAME") or _s.get("MODEL_NAME", "gpt-4o")
