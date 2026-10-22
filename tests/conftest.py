import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SAMPLE = ROOT / "data" / "sample"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOGUARD_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def toyota_sample() -> Path:
    return SAMPLE / "toyota_carobd_ufpe7.csv"


@pytest.fixture
def diesel_sample() -> Path:
    return SAMPLE / "dieselobd_test_excerpt_3000rows.xlsx"
