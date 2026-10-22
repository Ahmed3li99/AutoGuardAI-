"""Offline PDF report (ReportLab + Matplotlib). Content comes only from the
persisted session, quality report and analysis run - unavailable items are
omitted, never filled with zeros."""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

from ai.classifier_optional import UNAVAILABLE_MESSAGE  # noqa: E402
from ai.events import AnalysisResult  # noqa: E402
from ai.pipeline import overall_status  # noqa: E402
from core.schema import CANONICAL_FIELDS, PRIMARY_SIGNALS, TIME_COL, DataOrigin, NormalizedSession  # noqa: E402

APP_VERSION = "0.1.0"

LIMITATIONS = [
    "AutoGuard AI is an educational prototype. It does not diagnose mechanical faults and is not a safety system.",
    "Rule thresholds are configurable heuristics for a generic passenger car, not manufacturer specifications.",
    "Anomaly scores indicate statistical unusualness relative to calibration trips recorded from ONE vehicle "
    "(Toyota Etios 2014, carOBD dataset). On other vehicles the score scale is uncalibrated.",
    "No real-world fault labels were available, so detection rates on genuine faults are unknown.",
    "Live OBD-II connection is a future, adapter-required feature and has not been tested on a real vehicle.",
    "If an alert concerns temperature or charging voltage, have the vehicle inspected by a qualified mechanic; "
    "do not rely on this report to decide whether a vehicle is safe to drive.",
]


def _fig_to_image(fig, width_cm: float = 17.0) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_cm * cm, height=width_cm * cm * h / w)


def _signals_chart(session: NormalizedSession, result: AnalysisResult | None) -> Image | None:
    sigs = [s for s in PRIMARY_SIGNALS if s in session.signals][:6]
    if not sigs:
        return None
    df = session.data
    fig, axes = plt.subplots(len(sigs), 1, figsize=(10, 1.6 * len(sigs)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, sigs):
        ax.plot(df[TIME_COL], df[s], lw=0.7, color="#1f77b4")
        ax.set_ylabel(f"{s}\n[{CANONICAL_FIELDS[s].unit}]", fontsize=7)
        ax.tick_params(labelsize=7)
        if result:
            for a in result.alerts:
                ax.axvspan(a.t_start_s, a.t_end_s + 1, color="#d62728" if a.severity == "warning" else "#ff7f0e",
                           alpha=0.18, lw=0)
    axes[-1].set_xlabel(f"time [s] ({session.meta.time_basis.label})", fontsize=8)
    fig.suptitle("Signals (shaded: alert intervals)", fontsize=9)
    return _fig_to_image(fig)


def _scores_chart(result: AnalysisResult) -> Image | None:
    dets = [d for d in result.scores if result.scores[d].scores]
    if not dets:
        return None
    fig, axes = plt.subplots(len(dets), 1, figsize=(10, 1.8 * len(dets)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, dets):
        sc = result.scores[d]
        y = np.array([np.nan if v is None else v for v in sc.scores], dtype=float)
        ax.plot(sc.t_s, y, lw=0.7, color="#2ca02c")
        if sc.threshold:
            ax.axhline(sc.threshold["attention"], color="#ff7f0e", ls="--", lw=0.8, label="attention threshold")
            ax.axhline(sc.threshold["warning"], color="#d62728", ls="--", lw=0.8, label="warning threshold")
        ax.set_ylabel(d, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
    axes[-1].set_xlabel("time [s]", fontsize=8)
    fig.suptitle("Anomaly scores vs calibrated thresholds", fontsize=9)
    return _fig_to_image(fig)


def _table(rows, col_widths=None, header=True) -> Table:
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
             ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t


def build_pdf_report(path: str | Path, session: NormalizedSession, quality: dict | None,
                     result: AnalysisResult | None, session_db_id: int | None = None,
                     vehicle_profile: dict | None = None) -> Path:
    path = Path(path)
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=8, leading=10)
    banner = ParagraphStyle("banner", parent=ss["Heading2"], textColor=colors.HexColor("#b71c1c"))
    m = session.meta
    story = [Paragraph("AutoGuard AI - Telemetry Analysis Report", ss["Title"])]
    if m.origin == DataOrigin.SYNTHETIC:
        story.append(Paragraph("SYNTHETIC DEMO DATA - generated values, not a real vehicle", banner))
    story.append(Paragraph(f"Generated {dt.datetime.now().isoformat(timespec='seconds')} by AutoGuard AI "
                           f"{APP_VERSION} (offline).", small))
    story.append(Spacer(1, 6))

    status = overall_status(result, session)
    ident = [["Item", "Value"],
             ["Overall status", status],
             ["Data origin", m.origin.label],
             ["Source", m.source_name],
             ["Session id (local DB)", str(session_db_id) if session_db_id is not None else "not saved"]]
    if m.source_sha256:
        ident.append(["Source SHA-256", m.source_sha256])
    if m.dataset_id:
        ident.append(["Recognized dataset", m.dataset_id])
    if m.mapping_id:
        ident.append(["Column mapping", m.mapping_id])
    ident.append(["Time basis", m.time_basis.label])
    if m.start_timestamp:
        ident.append(["Recorded start", m.start_timestamp])
    ident.append(["Duration", f"{session.duration_s:.0f} s ({len(session.data)} rows)"])
    if m.vehicle_id:
        ident.append(["Vehicle id (from source)", m.vehicle_id])
    if m.trip_id:
        ident.append(["Trip id (from source)", m.trip_id])
    if vehicle_profile:
        ident.append(["Vehicle profile (user-entered)",
                      " ".join(str(vehicle_profile.get(k) or "") for k in ("name", "make", "model", "year")).strip()])
    if m.unit_conversions:
        ident.append(["Unit conversions", "; ".join(f"{k}: {v}" for k, v in m.unit_conversions.items())])
    story += [Paragraph("1. Source and provenance", ss["Heading2"]),
              _table([[Paragraph(str(a), small), Paragraph(str(b), small)] for a, b in ident],
                     [5 * cm, 12 * cm])]
    for n in m.notes:
        story.append(Paragraph("Note: " + n, small))

    if quality:
        story.append(Paragraph("2. Data quality", ss["Heading2"]))
        q = quality
        story.append(Paragraph(
            f"Status: <b>{q.get('status')}</b>. Rows {q['rows']} (raw {q['raw_rows']}), "
            f"median sample interval {q['median_dt_s'] if q['median_dt_s'] is not None else 'n/a'} s, "
            f"gaps {q['n_gaps']}, duplicate rows {q['duplicate_rows']}.", small))
        rows = [["Signal", "Unit", "Present", "Missing %", "Invalid", "Min", "Median", "Max"]]
        for s in q["signals"]:
            if s["present"] == 0:
                rows.append([s["name"], s["unit"], "0", "100", str(s["invalid_out_of_range"]), "N/A", "N/A", "N/A"])
            else:
                rows.append([s["name"], s["unit"], str(s["present"]), f"{100 * s['missing_rate']:.1f}",
                             str(s["invalid_out_of_range"]), f"{s['min']:.2f}", f"{s['median']:.2f}",
                             f"{s['max']:.2f}"])
        story.append(_table(rows))
        for w in q.get("warnings", []):
            story.append(Paragraph("• " + w, small))

    chart = _signals_chart(session, result)
    if chart:
        story.append(KeepTogether([Paragraph("3. Signals", ss["Heading2"]), chart]))

    story.append(PageBreak())
    story.append(Paragraph("4. Analysis", ss["Heading2"]))
    if result is None:
        story.append(Paragraph("No analysis has been run for this session.", small))
    else:
        rows = [["Detector", "Status", "Version", "Threshold / note"]]
        for det, st in result.detector_status.items():
            thr = result.scores.get(det).threshold if det in result.scores else None
            thr_txt = (f"attention {thr['attention']:.4g} (q={thr['attention_quantile']}), warning "
                       f"{thr['warning']:.4g}; {thr['calibration_source']}" if thr else "")
            note = st.get("reason") or st.get("note") or ""
            rows.append([det, st.get("status", ""), st.get("version", "") or "",
                         Paragraph((thr_txt + " " + note).strip(), small)])
        rows.append(["fault_classifier", "unavailable", "", Paragraph(UNAVAILABLE_MESSAGE, small)])
        story.append(_table(rows, [3.2 * cm, 2.5 * cm, 4.8 * cm, 6.5 * cm]))
        if result.timing_s:
            story.append(Paragraph("Measured run time: " + ", ".join(
                f"{k} {v:.2f} s" for k, v in result.timing_s.items()), small))
        sc = _scores_chart(result)
        if sc:
            story.append(sc)
        story.append(Paragraph(f"5. Alert timeline ({len(result.alerts)})", ss["Heading2"]))
        if not result.alerts:
            story.append(Paragraph("No alerts were raised by the available detectors.", small))
        else:
            rows = [["Start [s]", "End [s]", "Severity", "Detector", "Signals", "Explanation"]]
            for a in result.alerts:
                rows.append([f"{a.t_start_s:.0f}", f"{a.t_end_s:.0f}", a.severity, a.detector,
                             Paragraph(", ".join(a.signals), small), Paragraph(a.explanation, small)])
            story.append(_table(rows, [1.4 * cm, 1.4 * cm, 1.7 * cm, 3.2 * cm, 2.8 * cm, 6.5 * cm]))
    story.append(KeepTogether([Paragraph("6. Limitations", ss["Heading2"])]
                              + [Paragraph("• " + t, small) for t in LIMITATIONS]))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                      title="AutoGuard AI report", author="AutoGuard AI").build(story)
    return path
