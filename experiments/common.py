"""Shared loading helpers for experiment scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data_sources.files import RecordedFileSource  # noqa: E402
from core.preprocessing.resample import resample_1hz  # noqa: E402
from core.preprocessing.splits import load_split, make_trip_split, save_split  # noqa: E402

TOYOTA_DIR = ROOT / "data" / "raw" / "eron93br_carOBD" / "obdiidata"
SPLIT_PATH = ROOT / "datasets" / "splits" / "toyota_carobd_split.json"
PROCESSED = ROOT / "data" / "processed"


def toyota_split() -> dict[str, list[str]]:
    if SPLIT_PATH.exists():
        return load_split(SPLIT_PATH)
    split = make_trip_split([p.name for p in TOYOTA_DIR.glob("*.csv")])
    save_split(split, SPLIT_PATH)
    return split


def load_trip_1hz(name: str) -> pd.DataFrame:
    s = RecordedFileSource(TOYOTA_DIR / name).read_session()
    df, _ = resample_1hz(s.data, s.signals)
    return df


def load_partition(names: list[str]) -> dict[str, pd.DataFrame]:
    return {n: load_trip_1hz(n) for n in names}
