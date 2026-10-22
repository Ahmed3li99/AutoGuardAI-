"""File parsing, column mapping, units, missing/corrupt inputs."""
import json

import numpy as np
import pandas as pd
import pytest

from core.data_sources.base import DataSourceError
from core.data_sources.files import RecordedFileSource, read_table
from core.mapping import ColumnMapping, ColumnRule, auto_mapping, convert_units
from core.schema import DataOrigin, TimeBasis


def test_toyota_trailing_delimiter_alignment(toyota_sample):
    # Regression: rows end with ',' -> without index_col=False columns shift by one.
    s = RecordedFileSource(toyota_sample).read_session()
    assert s.meta.mapping_id == "toyota_carobd_v1"
    assert s.meta.time_basis == TimeBasis.RELATIVE_RECORDED
    assert s.data["t_s"].is_monotonic_increasing
    assert s.data["rpm"].max() < 5000            # would be ~2000+ run-time seconds if shifted
    assert 13 < s.data["battery_voltage"].median() < 15
    assert s.data["coolant_temp_c"].between(-40, 130).all()


def test_dieselobd_inferred_units_and_synthetic_clock(diesel_sample):
    src = RecordedFileSource(diesel_sample)
    s = src.read_session()
    assert s.meta.time_basis == TimeBasis.SYNTHETIC_CLOCK
    assert s.meta.vehicle_id is None and s.meta.trip_id is None   # never fabricated
    assert "inferred" in s.meta.unit_conversions["coolant_temp_c"]
    assert 50 < s.data["coolant_temp_c"].median() < 110            # degF converted to degC
    assert 90 < s.data["baro_pressure_kpa"].median() < 105          # psi converted to kPa
    assert any("synthetic playback clock" in w for w in src.quality.warnings)


def test_csv_semicolon_decimal_comma(tmp_path):
    p = tmp_path / "eu.csv"
    p.write_text("time;Speed (mph);Coolant [F];rpm\n0;10,0;180,0;900\n1;20,0;190,4;1500\n", encoding="utf-8")
    s = RecordedFileSource(p).read_session()
    assert s.meta.time_basis == TimeBasis.RELATIVE_RECORDED
    assert s.data["speed_kmh"].tolist() == pytest.approx([16.09344, 32.18688])
    assert s.data["coolant_temp_c"].iloc[0] == pytest.approx(82.222, abs=1e-3)


def test_json_layouts(tmp_path):
    recs = [{"timestamp": "2024-05-01T10:00:00", "rpm": 800, "speed": 0},
            {"timestamp": "2024-05-01T10:00:02", "rpm": 1200, "speed": 12}]
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps(recs))
    s = RecordedFileSource(p1).read_session()
    assert s.meta.time_basis == TimeBasis.ABSOLUTE
    assert s.meta.start_timestamp.startswith("2024-05-01T10:00:00")
    assert s.data["t_s"].tolist() == [0.0, 2.0]
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps({"columns": ["t_s", "rpm"], "data": [[0, 700], [1, 720]]}))
    assert RecordedFileSource(p2).read_session().data["rpm"].tolist() == [700, 720]


def test_missing_values_stay_missing_not_zero(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("t_s,rpm,coolant_temp_c\n0,800,\n1,,85\n2,820,86\n")
    src = RecordedFileSource(p)
    s = src.read_session()
    assert np.isnan(s.data.loc[0, "coolant_temp_c"]) and np.isnan(s.data.loc[1, "rpm"])
    q = {x.name: x for x in src.quality.signals}
    assert q["rpm"].missing == 1 and q["coolant_temp_c"].missing == 1


def test_out_of_range_and_non_numeric_counted(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("t_s,rpm,speed_kmh\n0,800,10\n1,abc,999\n2,-5,20\n")
    src = RecordedFileSource(p)
    s = src.read_session()
    q = {x.name: x for x in src.quality.signals}
    assert q["rpm"].parse_failures == 1
    assert q["rpm"].invalid_out_of_range == 1          # -5 rpm is impossible
    assert q["speed_kmh"].invalid_out_of_range == 1    # 999 km/h > PID max 255
    assert s.data["speed_kmh"].isna().sum() == 1


def test_unsorted_time_is_reported_and_sorted(tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("t_s,rpm\n0,800\n2,820\n1,810\n")
    src = RecordedFileSource(p)
    s = src.read_session()
    assert s.data["t_s"].tolist() == [0, 1, 2]
    assert src.quality.non_monotonic_time == 1


@pytest.mark.parametrize("content,suffix", [("", ".csv"), ("not,a\x00binary", ".xlsx"), ("{broken", ".json")])
def test_corrupt_files_fail_gracefully(tmp_path, content, suffix):
    p = tmp_path / f"c{suffix}"
    p.write_text(content)
    with pytest.raises(DataSourceError):
        RecordedFileSource(p).read_session()


def test_unmappable_file_fails_with_message(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("foo,bar\n1,2\n")
    with pytest.raises(DataSourceError, match="manual column mapping"):
        RecordedFileSource(p).read_session()


def test_user_defined_mapping(tmp_path):
    p = tmp_path / "custom.csv"
    p.write_text("secs,MotorDrehzahl,Tempo\n0,900,0\n1,1500,30\n")
    src = RecordedFileSource(p)
    src.set_mapping(ColumnMapping("manual", None, "", [], "secs", TimeBasis.RELATIVE_RECORDED,
                                  columns={"MotorDrehzahl": ColumnRule("rpm", "rpm"),
                                           "Tempo": ColumnRule("speed_kmh", "mph")}))
    s = src.read_session()
    assert s.meta.column_map == {"MotorDrehzahl": "rpm", "Tempo": "speed_kmh"}
    assert s.data["speed_kmh"].iloc[1] == pytest.approx(48.28, abs=0.01)


def test_unit_conversion_rejects_unknown():
    with pytest.raises(ValueError):
        convert_units(pd.Series([1.0]), "furlongs", "speed_kmh")


def test_auto_mapping_aliases():
    m = auto_mapping(["Timestamp", "Engine RPM", "Vehicle Speed (mph)", "ECT (F)", "Voltage"])
    assert m.time_basis == TimeBasis.ABSOLUTE
    cols = {k: (v.to, v.unit) for k, v in m.columns.items()}
    assert cols["Engine RPM"] == ("rpm", "rpm")
    assert cols["Vehicle Speed (mph)"] == ("speed_kmh", "mph")
    assert cols["ECT (F)"] == ("coolant_temp_c", "degF")


def test_reimported_synthetic_export_stays_synthetic(tmp_path):
    from core.data_sources.synthetic_demo import SyntheticDemoDataSource
    from reporting.csv_export import export_session_csv
    s = SyntheticDemoDataSource("normal_drive", duration_s=200).read_session()
    p = export_session_csv(s, tmp_path / "syn.csv")
    s2 = RecordedFileSource(p).read_session()
    assert s2.meta.origin == DataOrigin.SYNTHETIC


def test_read_table_unsupported_extension(tmp_path):
    p = tmp_path / "x.parquet"
    p.write_bytes(b"123")
    with pytest.raises(DataSourceError, match="Unsupported"):
        read_table(p)
