"""Offline PDF report in Arabic or English (ReportLab + Matplotlib).

Content comes only from the persisted session, quality report and analysis
run. Arabic text is shaped (arabic-reshaper) and laid out right-to-left
(python-bidi) line by line; the font is taken from the Windows fonts folder.
If no Arabic-capable font is available, the report falls back to English."""
from __future__ import annotations

import datetime as dt
import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT, TA_RIGHT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

from ai.classifier_optional import UNAVAILABLE_MESSAGE  # noqa: E402
from ai.events import AnalysisResult  # noqa: E402
from ai.explain import coverage, display_name, findings, mmss, trip_summary, verdict  # noqa: E402
from core.paths import resource_root  # noqa: E402
from core.schema import CANONICAL_FIELDS, TIME_COL, DataOrigin, NormalizedSession  # noqa: E402

APP_VERSION = "0.3.0"
PAGE_W = A4[0] - 3.6 * cm

LIMITATIONS = [
    ("AutoGuard AI نموذج تعليمي، مش جهاز تشخيص ولا نظام أمان.",
     "AutoGuard AI is an educational prototype. It does not diagnose mechanical faults and is not a safety system."),
    ("حدود القواعد قيم تقريبية لعربية ملاكي عادية، مش مواصفات المصنع.",
     "Rule thresholds are configurable heuristics for a generic passenger car, not manufacturer specifications."),
    ("الموديلات اتعلمت واتظبطت على رحلات عربية واحدة (تويوتا إتيوس 2014). على عربيات تانية دقتها أقل.",
     "Models were trained and calibrated on trips of ONE car (Toyota Etios 2014). On other cars they are less reliable."),
    ("مكانش متاح داتا فيها أعطال حقيقية مؤكدة، فدقة اكتشاف الأعطال الحقيقية مش معروفة.",
     "No data with confirmed real faults was available, so detection performance on real faults is unknown."),
    ("الاتصال المباشر بالعربية اتجرب على محوّل وهمي بس.", "The live OBD-II connection was tested with a simulated adapter only."),
    ("لو فيه أي شك في الحرارة أو الشحن، اعرض العربية على ميكانيكي.",
     "If an alert concerns temperature or charging, have the car checked by a qualified mechanic."),
]

T = {  # report labels
    "title": ("تقرير تحليل رحلة — AutoGuard AI", "Trip Analysis Report — AutoGuard AI"),
    "synthetic": ("عرض تجريبي: الأعطال في التقرير ده مصطنعة ومش حقيقية", "DEMO DATA: the faults in this report are artificial"),
    "generated": ("اتعمل في {d} بواسطة AutoGuard AI {v} (بدون إنترنت)", "Generated {d} by AutoGuard AI {v} (offline)"),
    "summary": ("ملخص الرحلة", "Trip summary"),
    "findings": ("اللي لقيناه", "What we found"),
    "none": ("مفيش ملاحظات — كل القراءات طبيعية.", "No findings — all readings look normal."),
    "low": ("كمان فيه {n} ملاحظات ضعيفة (لقاها موديل إحصائي واحد بس) ومش معروضة هنا.",
            "{n} weak observations (one statistical model only) are not listed."),
    "what": ("إيه اللي حصل", "What happened"), "normal": ("الطبيعي", "What is normal"),
    "meaning": ("ممكن يعني إيه", "What it may mean"), "action": ("تعمل إيه", "What to do"),
    "evidence": ("إزاي عرفنا", "How we know"),
    "sev_warning": ("مهم", "Important"), "sev_attention": ("ملاحظة", "Observation"),
    "conf_high": ("ثقة عالية", "high confidence"), "conf_medium": ("ثقة متوسطة", "medium confidence"),
    "conf_low": ("ثقة قليلة", "low confidence"),
    "charts": ("الرسومات (المظلل = أماكن الملاحظات)", "Charts (shaded = findings)"),
    "source": ("مصدر الداتا وجودتها", "Data source and quality"),
    "tech": ("تفاصيل فنية", "Technical details"),
    "limits": ("حدود البرنامج", "Limitations"),
    "not_analysed": ("الرحلة لسه ماتحللتش.", "This trip has not been analysed."),
}


# ---------------------------------------------------------------- fonts / text
def _register_fonts() -> tuple[str, str] | None:
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for regular, bold in (("arial.ttf", "arialbd.ttf"), ("tahoma.ttf", "tahomabd.ttf"), ("segoeui.ttf", "segoeuib.ttf")):
        if (fonts / regular).exists() and (fonts / bold).exists():
            if "AG" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("AG", str(fonts / regular)))
                pdfmetrics.registerFont(TTFont("AG-Bold", str(fonts / bold)))
            return "AG", "AG-Bold"
    return None


class _Text:
    def __init__(self, lang: str):
        reg = _register_fonts()
        if reg is None:
            lang = "en"
            self.font, self.bold = "Helvetica", "Helvetica-Bold"
        else:
            self.font, self.bold = reg
        self.lang = lang
        self.rtl = lang == "ar"
        if self.rtl:
            import arabic_reshaper
            from bidi.algorithm import get_display
            self._reshape, self._display = arabic_reshaper.reshape, get_display

    def t(self, key: str, **kw) -> str:
        ar, en = T[key]
        return (ar if self.rtl else en).format(**kw)

    def pick(self, ar: str, en: str) -> str:
        return ar if self.rtl else en

    def style(self, size: float = 9.5, bold: bool = False, color=colors.black, space: float = 2) -> ParagraphStyle:
        return ParagraphStyle("s", fontName=self.bold if bold else self.font, fontSize=size, leading=size * 1.45,
                              textColor=color, alignment=TA_RIGHT if self.rtl else TA_LEFT, spaceAfter=space)

    def para(self, text: str, size: float = 9.5, bold: bool = False, color=colors.black,
             width: float = PAGE_W, space: float = 2):
        """Paragraph for LTR text; for Arabic, a stack of pre-wrapped visual lines."""
        st = self.style(size, bold, color, space)
        text = text.replace("⚠️", "(!)").replace("️", "")   # PDF fonts have no emoji glyphs
        if not self.rtl:
            return Paragraph(_esc(text), st)
        font = st.fontName
        lines, cur = [], ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if pdfmetrics.stringWidth(self._reshape(trial), font, size) > width - 4 and cur:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            lines.append(cur)
        rows = [[Paragraph(_esc(self._display(self._reshape(ln))), st)] for ln in lines] or [[""]]
        tbl = Table(rows, colWidths=[width])
        tbl.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                 ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return tbl

    def mpl(self, text: str) -> str:
        return self._display(self._reshape(text)) if self.rtl else text


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- charts
def _fig_to_image(fig, width_cm: float = 17.4) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_cm * cm, height=width_cm * cm * h / w)


LABELS = {"speed_kmh": ("السرعة", "Speed"), "rpm": ("لفات المحرك", "RPM"), "coolant_temp_c": ("حرارة المحرك", "Coolant"),
          "battery_voltage": ("الفولت", "Voltage"), "engine_load_pct": ("حمل المحرك", "Load"),
          "throttle_pct": ("البوابة", "Throttle"), "intake_air_temp_c": ("هوا السحب", "Intake air"),
          "manifold_pressure_kpa": ("ضغط السحب", "MAP")}


def _signals_chart(session: NormalizedSession, fs, tx: _Text) -> Image | None:
    order = ["speed_kmh", "rpm", "coolant_temp_c", "battery_voltage", "engine_load_pct", "throttle_pct"]
    sigs = [s for s in order if s in session.signals][:5]
    if not sigs:
        return None
    plt.rcParams["font.family"] = ["Arial", "Tahoma", "DejaVu Sans"]
    df = session.data
    fig, axes = plt.subplots(len(sigs), 1, figsize=(10, 1.55 * len(sigs)), sharex=True)
    axes = np.atleast_1d(axes)
    cols = ["#1C4FD8", "#0F7B55", "#B45309", "#C2283A", "#6D4AE0"]
    for ax, s, c in zip(axes, sigs, cols):
        ax.plot(df[TIME_COL] / 60.0, df[s], lw=0.8, color=c)
        ax.set_ylabel(tx.mpl(tx.pick(*LABELS[s])) + f"\n[{CANONICAL_FIELDS[s].unit}]", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for f in fs:
            if f.confidence != "low":
                ax.axvspan(f.t_start_s / 60, (f.t_end_s + 1) / 60, lw=0, alpha=0.18,
                           color="#C2283A" if f.severity == "warning" else "#B45309")
    axes[-1].set_xlabel(tx.mpl(tx.pick("الوقت (دقيقة)", "time (min)")), fontsize=8)
    return _fig_to_image(fig)


# ---------------------------------------------------------------- tables
def _kv_table(rows: list[tuple[str, str]], tx: _Text, widths=(5.2 * cm, PAGE_W - 5.2 * cm)) -> Table:
    data = []
    for k, v in rows:
        kp = tx.para(k, 8.5, True, colors.HexColor("#4A5568"), widths[0] - 8)
        vp = tx.para(v, 8.5, False, colors.black, widths[1] - 8)
        data.append([vp, kp] if tx.rtl else [kp, vp])
    t = Table(data, colWidths=list(widths[::-1]) if tx.rtl else list(widths))
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9DEE7")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (1 if tx.rtl else 0, 0), (1 if tx.rtl else 0, -1), colors.HexColor("#F4F6FA"))]))
    return t


def _finding_block(f, tx: _Text) -> KeepTogether:
    t = f.text(tx.lang)
    col = colors.HexColor("#C2283A" if f.severity == "warning" else "#B45309")
    head = f"{t['title']}  —  {tx.t('sev_' + f.severity)} · {tx.t('conf_' + f.confidence)} · {mmss(f.t_start_s)}–{mmss(f.t_end_s)}"
    rows = [(tx.t(k), t[k]) for k in ("what", "normal", "meaning", "action", "evidence") if t.get(k)]
    inner = [tx.para(head, 10.5, True, col, PAGE_W - 0.6 * cm, 4), _kv_table(rows, tx, (3.4 * cm, PAGE_W - 4.0 * cm))]
    box = Table([[inner]], colWidths=[PAGE_W])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, col),
                             ("LINEBEFORE" if not tx.rtl else "LINEAFTER", (0, 0), (0, -1), 4, col),
                             ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                             ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return KeepTogether([box, Spacer(1, 8)])


# ---------------------------------------------------------------- report
def build_pdf_report(path: str | Path, session: NormalizedSession, quality: dict | None,
                     result: AnalysisResult | None, session_db_id: int | None = None,
                     vehicle_profile: dict | None = None, lang: str = "en") -> Path:
    path = Path(path)
    tx = _Text(lang)
    m = session.meta
    fs = findings(result, session)
    main = [f for f in fs if f.confidence != "low"]
    # brand header: logo + title (logo on the reading-start side)
    logo = resource_root() / "app" / "assets" / "logo_128.png"
    title = tx.para(tx.t("title"), 17, True, colors.HexColor("#0B1626"), PAGE_W - 2.2 * cm, space=0)
    if logo.exists():
        mark = Image(str(logo), width=1.5 * cm, height=1.5 * cm)
        cells = [[title, mark]] if tx.rtl else [[mark, title]]
        widths = [PAGE_W - 1.9 * cm, 1.9 * cm] if tx.rtl else [1.9 * cm, PAGE_W - 1.9 * cm]
        head = Table(cells, colWidths=widths)
        head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                  ("LINEBELOW", (0, 0), (-1, 0), 2, colors.HexColor("#22D3EE"))]))
        story = [head, Spacer(1, 6)]
    else:
        story = [title]
    if m.origin == DataOrigin.SYNTHETIC:
        story.append(tx.para(tx.t("synthetic"), 11, True, colors.HexColor("#6D4AE0")))
    story.append(tx.para(tx.t("generated", d=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), v=APP_VERSION),
                         8, False, colors.HexColor("#5B6779")))
    story.append(Spacer(1, 8))

    v = verdict(fs, result, session)
    vc = {"ok": "#0F7B55", "attention": "#B45309", "warning": "#C2283A", "none": "#5B6779"}[v["level"]]
    vbox = Table([[[tx.para(tx.pick(v["ar"], v["en"]), 14, True, colors.HexColor(vc), PAGE_W - 1 * cm),
                    tx.para(tx.pick(v["ar_sub"], v["en_sub"]), 9.5, False, colors.black, PAGE_W - 1 * cm),
                    tx.para(display_name(m.source_name, tx.lang), 8.5, False, colors.HexColor("#5B6779"),
                            PAGE_W - 1 * cm)]]], colWidths=[PAGE_W])
    vbox.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor(vc)),
                              ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFBFD")),
                              ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                              ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [vbox, Spacer(1, 6)]
    story.append(tx.para(tx.pick("النتيجة دي موثوقة لأي درجة؟", "How reliable is this result?"), 11, True, space=3))
    story.append(_kv_table([(("✓ " if c["ok"] else "! ") + tx.pick(c["ar"], c["en"]),
                             tx.pick(c["value_ar"], c["value_en"])) for c in coverage(result, session, quality)], tx))
    story.append(Spacer(1, 10))

    story.append(tx.para(tx.t("summary"), 12.5, True, space=4))
    story.append(_kv_table([(tx.pick(ar, en), val) for ar, en, val in trip_summary(session)], tx))
    story.append(Spacer(1, 10))

    story.append(tx.para(tx.t("findings"), 12.5, True, space=4))
    if result is None:
        story.append(tx.para(tx.t("not_analysed")))
    elif not main:
        story.append(tx.para(tx.t("none")))
    for f in main:
        story.append(_finding_block(f, tx))
    low = len(fs) - len(main)
    if low:
        story.append(tx.para(tx.t("low", n=low), 8.5, False, colors.HexColor("#5B6779")))

    chart = _signals_chart(session, fs, tx)
    if chart:
        story.append(KeepTogether([tx.para(tx.t("charts"), 12.5, True, space=4), chart]))

    story.append(PageBreak())
    story.append(tx.para(tx.t("source"), 12.5, True, space=4))
    rows = [(tx.pick("نوع الداتا", "Data origin"), {"recorded": tx.pick("رحلة مسجّلة", "Recorded"),
                                                    "synthetic": tx.pick("تجريبية (مصطنعة)", "Synthetic demo"),
                                                    "live_obd": tx.pick("مباشر من العربية", "Live OBD-II")}[m.origin.value]),
            (tx.pick("الملف / المصدر", "Source"), m.source_name),
            (tx.pick("رقم الجلسة", "Session id"), str(session_db_id) if session_db_id is not None else "-"),
            (tx.pick("محور الوقت", "Time basis"), m.time_basis.label)]
    if m.source_sha256:
        rows.append(("SHA-256", m.source_sha256))
    if m.dataset_id:
        rows.append((tx.pick("الداتاسِت", "Dataset"), m.dataset_id))
    if vehicle_profile:
        rows.append((tx.pick("بيانات العربية (من المستخدم)", "Vehicle profile (user-entered)"),
                     " ".join(str(vehicle_profile.get(k) or "") for k in ("name", "make", "model", "year")).strip()))
    if m.unit_conversions:
        rows.append((tx.pick("تحويل وحدات", "Unit conversions"), "; ".join(f"{k}: {v}" for k, v in m.unit_conversions.items())))
    if quality:
        rows.append((tx.pick("جودة الداتا", "Data quality"),
                     f"{quality.get('status')} · {quality['rows']} rows · gaps {quality['n_gaps']} · "
                     f"duplicates {quality['duplicate_rows']}"))
        miss = [s for s in quality["signals"] if s["missing_rate"] > 0.2 or s["invalid_out_of_range"]]
        if miss:
            rows.append((tx.pick("إشارات فيها نقص", "Signals with gaps"),
                         ", ".join(f"{s['name']} ({100 * s['missing_rate']:.0f}% missing)" for s in miss)))
    story.append(_kv_table(rows, tx))
    for n in m.notes:
        story.append(tx.para("• " + n, 7.5, False, colors.HexColor("#4A5568")))
    story.append(Spacer(1, 10))

    if result is not None:
        story.append(tx.para(tx.t("tech"), 12.5, True, space=4))
        trows = []
        for det, st in result.detector_status.items():
            thr = result.scores[det].threshold if det in result.scores else None
            txt = f"{st.get('status')} · {st.get('version', '') or ''}"
            if st.get("device"):
                txt += f" · {st['device']}"
            if thr:
                txt += f" · threshold q={thr['attention_quantile']} ({thr['calibration_source']})"
            if st.get("reason") or st.get("note"):
                txt += " · " + (st.get("reason") or st.get("note"))
            if det in result.timing_s:
                txt += f" · {result.timing_s[det]:.2f} s"
            trows.append((det, txt))
        trows.append(("fault_classifier", UNAVAILABLE_MESSAGE))
        story.append(_kv_table(trows, tx, (3.6 * cm, PAGE_W - 3.6 * cm)))
        story.append(Spacer(1, 10))

    story.append(tx.para(tx.t("limits"), 12.5, True, space=4))
    for ar, en in LIMITATIONS:
        story.append(tx.para("• " + tx.pick(ar, en), 8.5))

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.4 * cm,
                      bottomMargin=1.4 * cm, title="AutoGuard AI report", author="AutoGuard AI").build(story)
    return path
