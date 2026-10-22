"""Data Import & Quality: file selection, preview, column mapping, validation, provenance."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSplitter, QTableWidget, QTabWidget, QVBoxLayout, QWidget,
                               QMessageBox, QPlainTextEdit)
from PySide6.QtCore import Qt

from app.state import AppState, Worker
from app.widgets.common import fill_table, h1, muted, readonly_table
from core.data_sources.base import DataSourceError
from core.data_sources.files import RecordedFileSource
from core.data_sources.synthetic_demo import SCENARIOS, SyntheticDemoDataSource
from core.mapping import UNITS_BY_FIELD, ColumnMapping, ColumnRule
from core.paths import SAMPLE_DIR
from core.schema import CANONICAL_FIELDS, NUMERIC_FIELDS, TimeBasis

IGNORE = "— ignore —"


class ImportPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.source: RecordedFileSource | SyntheticDemoDataSource | None = None
        self.pending = None   # (session, quality) validated but not yet saved

        lay = QVBoxLayout(self)
        lay.addWidget(h1("Data Import & Quality"))
        lay.addWidget(muted("Import a recorded CSV/XLSX/JSON file, a bundled sample trip, or a clearly labelled "
                            "synthetic demo. Nothing is uploaded anywhere; all processing is local."))
        row = QHBoxLayout()
        b_open = QPushButton("Open file…")
        b_open.setObjectName("primary")
        b_open.clicked.connect(self.open_file)
        self.samples = QComboBox()
        self._fill_samples()
        b_sample = QPushButton("Load sample")
        b_sample.clicked.connect(self.load_sample)
        self.scenario = QComboBox()
        self.scenario.addItems(list(SCENARIOS))
        b_syn = QPushButton("Load synthetic demo")
        b_syn.clicked.connect(self.load_synthetic)
        for w in (b_open, QLabel("  Bundled sample:"), self.samples, b_sample, QLabel("  Synthetic:"),
                  self.scenario, b_syn):
            row.addWidget(w)
        row.addStretch()
        lay.addLayout(row)

        self.info = QLabel("No source loaded.")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)

        split = QSplitter(Qt.Vertical)
        tabs = QTabWidget()
        self.preview = readonly_table()
        tabs.addTab(self.preview, "Raw preview (first 200 rows)")
        mapw = QWidget()
        ml = QVBoxLayout(mapw)
        trow = QHBoxLayout()
        self.time_col = QComboBox()
        self.time_basis = QComboBox()
        self.time_basis.addItem("Recorded relative time (s)", TimeBasis.RELATIVE_RECORDED.value)
        self.time_basis.addItem("Recorded timestamps (date/time)", TimeBasis.ABSOLUTE.value)
        self.time_unit = QComboBox()
        self.time_unit.addItems(["s", "ms", "min"])
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.01, 3600)
        self.interval.setValue(1.0)
        self.interval.setSuffix(" s/row")
        for w in (QLabel("Time column:"), self.time_col, self.time_basis, self.time_unit,
                  QLabel("  If no time column, synthetic playback interval:"), self.interval):
            trow.addWidget(w)
        trow.addStretch()
        ml.addLayout(trow)
        self.map_table = QTableWidget()
        self.map_table.setColumnCount(4)
        self.map_table.setHorizontalHeaderLabels(["Source column", "Canonical signal", "Source unit", "Note"])
        self.map_table.verticalHeader().setVisible(False)
        ml.addWidget(self.map_table)
        self.map_desc = muted("")
        ml.addWidget(self.map_desc)
        tabs.addTab(mapw, "Column mapping")
        split.addWidget(tabs)

        qbox = QGroupBox("Validation report")
        ql = QVBoxLayout(qbox)
        self.q_summary = QLabel("Not validated.")
        self.q_summary.setWordWrap(True)
        self.q_table = readonly_table()
        self.q_warn = QPlainTextEdit()
        self.q_warn.setReadOnly(True)
        self.q_warn.setMaximumHeight(110)
        ql.addWidget(self.q_summary)
        ql.addWidget(self.q_table)
        ql.addWidget(self.q_warn)
        split.addWidget(qbox)
        lay.addWidget(split, 1)

        brow = QHBoxLayout()
        self.b_validate = QPushButton("Apply mapping && validate")
        self.b_validate.clicked.connect(self.validate)
        self.b_use = QPushButton("Save session && open replay")
        self.b_use.setObjectName("primary")
        self.b_use.clicked.connect(self.use_session)
        self.b_validate.setEnabled(False)
        self.b_use.setEnabled(False)
        brow.addStretch()
        brow.addWidget(self.b_validate)
        brow.addWidget(self.b_use)
        lay.addLayout(brow)

    # ---- loading --------------------------------------------------------
    def _fill_samples(self) -> None:
        self.samples.clear()
        manifest = SAMPLE_DIR / "SAMPLES.json"
        labels = {}
        if manifest.exists():
            labels = {d["file"]: d["label"] for d in json.loads(manifest.read_text(encoding="utf-8"))["samples"]}
        for p in sorted(SAMPLE_DIR.glob("*")):
            if p.suffix.lower() in (".csv", ".xlsx", ".json") and p.name != "SAMPLES.json":
                self.samples.addItem(labels.get(p.name, p.name), str(p))

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open recorded vehicle data", "",
                                              "Vehicle data (*.csv *.xlsx *.xls *.json *.txt)")
        if path:
            self._load_path(Path(path))

    def load_sample(self) -> None:
        p = self.samples.currentData()
        if p:
            self._load_path(Path(p))

    def _load_path(self, path: Path) -> None:
        self.source = RecordedFileSource(path)
        self.info.setText(f"Reading {path.name}…")
        self.state.busy_changed.emit(True, f"Reading {path.name}…")
        w = Worker(lambda: self.source.raw)
        w.signals.finished.connect(self._raw_loaded)
        w.signals.failed.connect(self._load_failed)
        self.state.pool.start(w)

    def _load_failed(self, msg: str) -> None:
        self.state.busy_changed.emit(False, "Import failed.")
        self.info.setText("Import failed: " + msg)
        QMessageBox.warning(self, "Import failed", msg)
        self.source = None

    def _raw_loaded(self, raw: pd.DataFrame) -> None:
        self.state.busy_changed.emit(False, "File read.")
        src = self.source
        fill_table(self.preview, raw, 200)
        m = src.mapping
        self.info.setText(f"<b>{src.name}</b> — {len(raw)} rows × {raw.shape[1]} columns. "
                          f"Detected mapping: <b>{m.mapping_id}</b>"
                          + (f" (dataset: {m.dataset_id})" if m.dataset_id else " (unknown source — review!)"))
        self._show_mapping(list(raw.columns), m)
        self.b_validate.setEnabled(True)
        self.validate()

    def load_synthetic(self) -> None:
        self.source = SyntheticDemoDataSource(self.scenario.currentText())
        s = self.source.read_session()
        fill_table(self.preview, s.data, 200)
        self.map_table.setRowCount(0)
        self.map_desc.setText("Synthetic demo uses canonical signals directly.")
        self.info.setText(f"<b style='color:#ab47bc'>SYNTHETIC DEMO</b> — scenario '{self.source.scenario}'. "
                          "Generated values, not a real vehicle. Injected events are listed in the notes.")
        self.b_validate.setEnabled(False)
        self._show_quality(s, self.source.quality)

    # ---- mapping editor -------------------------------------------------
    def _show_mapping(self, columns: list[str], m: ColumnMapping) -> None:
        self.time_col.clear()
        self.time_col.addItem("(none — synthetic playback clock)", None)
        for c in columns:
            self.time_col.addItem(c, c)
        if m.time_column in columns:
            self.time_col.setCurrentIndex(columns.index(m.time_column) + 1)
            self.time_basis.setCurrentIndex(0 if m.time_basis == TimeBasis.RELATIVE_RECORDED else 1)
            self.time_unit.setCurrentText(m.time_unit)
        self.interval.setValue(m.playback_interval_s)
        rules = m.resolved_columns(columns)
        self.map_table.setRowCount(len(columns))
        for i, c in enumerate(columns):
            self.map_table.setItem(i, 0, _ro_item(c))
            fbox = QComboBox()
            fbox.addItem(IGNORE)
            fbox.addItems(NUMERIC_FIELDS)
            ubox = QComboBox()
            note = ""
            if c in rules:
                fbox.setCurrentText(rules[c].to)
                ubox.addItems(UNITS_BY_FIELD.get(rules[c].to, [rules[c].unit]))
                unit = rules[c].unit
                if ubox.findText(unit) < 0:
                    ubox.addItem(unit)
                ubox.setCurrentText(unit)
                note = "unit INFERRED — verify" if rules[c].unit_inferred else ""
            fbox.currentTextChanged.connect(lambda f, ub=ubox: self._field_changed(f, ub))
            self.map_table.setCellWidget(i, 1, fbox)
            self.map_table.setCellWidget(i, 2, ubox)
            self.map_table.setItem(i, 3, _ro_item(note))
        self.map_table.resizeColumnsToContents()
        self.map_desc.setText(m.description)

    @staticmethod
    def _field_changed(field: str, ubox: QComboBox) -> None:
        ubox.clear()
        if field != IGNORE:
            ubox.addItems(UNITS_BY_FIELD.get(field, [CANONICAL_FIELDS[field].unit]))
            ubox.setCurrentText(CANONICAL_FIELDS[field].unit)

    def _mapping_from_ui(self) -> ColumnMapping:
        base = self.source.mapping
        rules = {}
        used = set()
        for i in range(self.map_table.rowCount()):
            col = self.map_table.item(i, 0).text()
            f = self.map_table.cellWidget(i, 1).currentText()
            u = self.map_table.cellWidget(i, 2).currentText()
            if f == IGNORE:
                continue
            if f in used:
                raise ValueError(f"Signal '{f}' is mapped from more than one column.")
            used.add(f)
            prev = base.resolved_columns([col]).get(col)
            inferred = bool(prev and prev.to == f and prev.unit == u and prev.unit_inferred)
            rules[col] = ColumnRule(f, u, inferred)
        tcol = self.time_col.currentData()
        changed = (rules != base.resolved_columns(list(self.source.raw.columns)) or tcol != base.time_column)
        return ColumnMapping(
            mapping_id=base.mapping_id + ("+manual" if changed else ""), dataset_id=base.dataset_id,
            description=base.description, detect_columns=base.detect_columns, time_column=tcol,
            time_basis=TimeBasis(self.time_basis.currentData()) if tcol else TimeBasis.SYNTHETIC_CLOCK,
            time_unit=self.time_unit.currentText(), playback_interval_s=self.interval.value(), columns=rules)

    # ---- validation -----------------------------------------------------
    def validate(self) -> None:
        if not isinstance(self.source, RecordedFileSource):
            return
        try:
            self.source.set_mapping(self._mapping_from_ui())
            s = self.source.read_session()
        except (DataSourceError, ValueError) as e:
            self.q_summary.setText(f"<span style='color:#e53935'>Validation failed: {e}</span>")
            self.b_use.setEnabled(False)
            return
        self._show_quality(s, self.source.quality)

    def _show_quality(self, session, q) -> None:
        self.pending = (session, q)
        d = q.to_dict()
        color = {"OK": "#43a047", "Warnings": "#fb8c00"}.get(d["status"], "#e53935")
        self.q_summary.setText(
            f"Status <b style='color:{color}'>{d['status']}</b> · {session.meta.origin.label} · "
            f"{session.meta.time_basis.label} · {d['rows']} rows · {d['duration_s']:.0f} s · "
            f"median Δt {d['median_dt_s']} s · gaps {d['n_gaps']} · duplicate rows {d['duplicate_rows']}")
        rows = pd.DataFrame([{"signal": s["name"], "unit": s["unit"], "present": s["present"],
                              "missing %": round(100 * s["missing_rate"], 1),
                              "invalid": s["invalid_out_of_range"], "parse errors": s["parse_failures"],
                              "min": s["min"], "median": s["median"], "max": s["max"]} for s in d["signals"]])
        fill_table(self.q_table, rows, float_fmt="{:.4g}")
        self.q_warn.setPlainText("\n".join(["Warnings:"] + [f"• {w}" for w in d["warnings"]]
                                           + ["", "Notes:"] + [f"• {n}" for n in session.meta.notes]
                                           + ([f"• Unmapped columns: {', '.join(d['unmapped_columns'])}"]
                                              if d["unmapped_columns"] else [])))
        self.b_use.setEnabled(d["status"] != "Unusable")

    def use_session(self) -> None:
        if not self.pending:
            return
        session, q = self.pending
        sid = self.state.repo.save_session(session, q.to_dict(), self.state.repo.get_setting("active_vehicle_id"))
        self.state.set_session(session, q.to_dict(), sid)
        self.state.history_changed.emit()
        self.window().navigate("Replay")  # type: ignore[attr-defined]


def _ro_item(text: str):
    from PySide6.QtWidgets import QTableWidgetItem
    it = QTableWidgetItem(text)
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    return it
