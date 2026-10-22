"""Recorded file sources: CSV, Excel and JSON."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from core.data_sources.base import DataSourceError, SourceCapabilities, VehicleDataSource
from core.mapping import ColumnMapping, auto_mapping, clean_column_name, detect_mapping
from core.preprocessing.normalize import normalize
from core.schema import DataOrigin, NormalizedSession
from core.validation.quality import QualityReport

MAX_FILE_BYTES = 500 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=clean_column_name)
    # Drop fully empty 'Unnamed: N' columns produced by trailing delimiters.
    drop = [c for c in df.columns if str(c).startswith("Unnamed") and df[c].isna().all()]
    return df.drop(columns=drop)


def read_csv_robust(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        head = f.read(65536)
    text = head.decode("utf-8-sig", errors="replace")
    try:
        delim = csv.Sniffer().sniff(text.split("\n", 1)[0] + "\n", delimiters=",;\t|").delimiter
    except csv.Error:
        delim = ","
    decimal = ","
    if delim == ",":
        decimal = "."
    else:
        # ';' separated files may still use '.' decimals - check the data lines.
        body = "\n".join(text.splitlines()[1:20])
        if body.count(".") > body.count(","):
            decimal = "."
    # index_col=False: carOBD rows end with a trailing delimiter; without this
    # pandas silently shifts every column by one (verified on drive1.csv).
    return pd.read_csv(path, sep=delim, decimal=decimal, skipinitialspace=True,
                       index_col=False, encoding="utf-8-sig", low_memory=False)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataSourceError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise DataSourceError(f"File is empty: {path.name}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DataSourceError(f"File is larger than {MAX_FILE_BYTES >> 20} MB: {path.name}")
    ext = path.suffix.lower()
    try:
        if ext in (".csv", ".txt", ".tsv"):
            df = read_csv_robust(path)
        elif ext in (".xlsx", ".xlsm", ".xls"):
            df = pd.read_excel(path, sheet_name=0)
        elif ext == ".json":
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(obj, dict) and "data" in obj and "columns" in obj:
                df = pd.DataFrame(obj["data"], columns=obj["columns"])
            elif isinstance(obj, dict) and "readings" in obj:
                df = pd.json_normalize(obj["readings"])
            elif isinstance(obj, list):
                df = pd.json_normalize(obj)
            else:
                raise DataSourceError("Unsupported JSON layout: expected a list of records, "
                                      "{'readings': [...]} or {'columns': [...], 'data': [...]}.")
        else:
            raise DataSourceError(f"Unsupported file type '{ext}'. Use CSV, XLSX or JSON.")
    except DataSourceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise DataSourceError(f"{path.name} is not valid text/JSON: {e}") from e
    except Exception as e:  # pandas/openpyxl raise many types on corrupt input
        raise DataSourceError(f"Could not parse {path.name}: {type(e).__name__}: {e}") from e
    df = _clean_frame(df)
    if df.empty or df.shape[1] == 0:
        raise DataSourceError(f"{path.name} contains no data rows.")
    return df


def read_preview(path: str | Path, n: int = 200) -> pd.DataFrame:
    return read_table(path).head(n)


class RecordedFileSource(VehicleDataSource):
    """CSV/XLSX/JSON reader. The concrete format is chosen by file extension;
    CSVDataSource / ExcelDataSource / JSONDataSource are aliases kept for
    readability where a specific type is expected."""

    def __init__(self, path: str | Path, mapping: ColumnMapping | None = None,
                 playback_interval_s: float | None = None):
        self.path = Path(path)
        self.name = self.path.name
        self._mapping = mapping
        self._interval = playback_interval_s
        self._raw: pd.DataFrame | None = None
        self._session: NormalizedSession | None = None
        self.quality: QualityReport | None = None

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None:
            self._raw = read_table(self.path)
        return self._raw

    @property
    def mapping(self) -> ColumnMapping:
        if self._mapping is None:
            cols = list(self.raw.columns)
            self._mapping = detect_mapping(cols) or auto_mapping(cols)
        return self._mapping

    def set_mapping(self, mapping: ColumnMapping) -> None:
        self._mapping = mapping
        self._session = None

    def read_session(self) -> NormalizedSession:
        if self._session is None:
            try:
                self._session, self.quality = normalize(
                    self.raw, self.mapping, self.name, DataOrigin.RECORDED,
                    source_sha256=sha256_file(self.path), playback_interval_s=self._interval)
            except ValueError as e:
                raise DataSourceError(f"{self.name}: {e}") from e
        return self._session

    def capabilities(self) -> SourceCapabilities:
        s = self.read_session()
        return SourceCapabilities(s.meta.origin, s.meta.time_basis, tuple(s.signals))


CSVDataSource = ExcelDataSource = JSONDataSource = RecordedFileSource


def session_from_dataframe(df: pd.DataFrame, name: str, mapping: ColumnMapping | None = None,
                           origin: DataOrigin = DataOrigin.RECORDED
                           ) -> tuple[NormalizedSession, QualityReport]:
    df = _clean_frame(df)
    mapping = mapping or detect_mapping(list(df.columns)) or auto_mapping(list(df.columns))
    return normalize(df, mapping, name, origin)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
