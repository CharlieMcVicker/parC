from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent

for env_file in (ROOT / ".test.env", ROOT / ".env"):
    if env_file.exists():
        load_dotenv(env_file, override=False)
        break
