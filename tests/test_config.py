from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_api_url, get_chat_model, get_embedding_model, get_persist_dir


def test_configuration_defaults():
    assert get_api_url().startswith("http")
    assert get_chat_model()
    assert get_embedding_model()
    assert get_persist_dir()
