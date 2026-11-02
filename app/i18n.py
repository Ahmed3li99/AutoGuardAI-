"""Arabic / English UI strings. tr(key) returns the text in the current language."""
from __future__ import annotations

_LANG = "ar"

S: dict[str, tuple[str, str]] = {
    # app / navigation
    "app_title": ("AutoGuard AI — مراقبة وتحليل بيانات العربية", "AutoGuard AI — Vehicle Data Analysis"),
    "tagline": ("حلّل رحلة عربيتك واعرف لو فيه حاجة غير طبيعية — من غير نت، وكل الداتا على جهازك.",
                "Analyse a car trip and see if anything is unusual — offline, all data stays on this PC."),
    "nav_home": ("الرئيسية", "Home"),
    "nav_trip": ("الرحلة", "Trip"),
    "nav_results": ("النتيجة", "Results"),
    "nav_report": ("التقرير", "Report"),
    "settings": ("الإعدادات", "Settings"),
    "lang_toggle": ("English", "عربي"),
    "offline": ("بدون إنترنت", "Offline"),
    "no_trip": ("مفيش رحلة مفتوحة", "No trip loaded"),
    # origins
    "origin_recorded": ("رحلة مسجّلة (إعادة تشغيل)", "Recorded trip (replay)"),
    "origin_synthetic": ("عرض تجريبي — أعطال مصطنعة", "Demo — artificial faults"),
    "origin_live_obd": ("توصيل مباشر بالعربية", "Live from the car"),
    # home
    "start_title": ("ابدأ من هنا", "Start here"),
    "card_file_title": ("افتح ملف رحلة", "Open a trip file"),
    "card_file_text": ("ملف CSV أو Excel أو JSON متسجّل من العربية.", "A CSV, Excel or JSON file recorded from a car."),
    "card_file_btn": ("اختار ملف…", "Choose file…"),
    "card_sample_title": ("جرّب رحلة حقيقية جاهزة", "Try a real sample trip"),
    "card_sample_text": ("رحلات حقيقية متسجلة من عربية تويوتا إتيوس.", "Real trips recorded from a Toyota Etios."),
    "card_sample_btn": ("افتح الرحلة", "Open trip"),
    "card_demo_title": ("عرض تجريبي بأعطال", "Demo with faults"),
    "card_demo_text": ("رحلة حقيقية حطّينا فيها أعطال مصطنعة (سخونة، فولت واطي، قراءة واقفة) عشان تشوف البرنامج بيكتشفها إزاي.",
                       "A real trip with artificial faults injected (overheating, low voltage, frozen reading) "
                       "to show how they are detected."),
    "card_demo_btn": ("شغّل العرض", "Run demo"),
    "card_live_title": ("اتصل بالعربية", "Connect to the car"),
    "card_live_text": ("بمحوّل OBD-II (ELM327). البرنامج بيدوّر على المحوّل ويتعرّف على العربية لوحده.",
                       "With an OBD-II (ELM327) adapter. The app finds the adapter and the car automatically."),
    "card_live_btn": ("اتصل", "Connect"),
    "recent": ("الرحلات اللي فاتت", "Previous trips"),
    "open": ("افتح", "Open"),
    "delete": ("امسح", "Delete"),
    "delete_all": ("امسح كل السجل", "Delete all history"),
    "confirm_delete": ("متأكد إنك عايز تمسح الرحلة دي؟", "Delete this trip?"),
    "confirm_delete_all": ("متأكد إنك عايز تمسح كل الرحلات المحفوظة على الجهاز؟", "Delete ALL saved trips on this PC?"),
    "advanced_import": ("استيراد متقدم (ربط الأعمدة)…", "Advanced import (column mapping)…"),
    "loading": ("بيحمّل…", "Loading…"),
    "analysing": ("بيحلّل الرحلة…", "Analysing trip…"),
    "file_ok": ("الملف اتقرا تمام", "File read successfully"),
    "file_warn": ("الملف اتقرا، بس فيه ملاحظات على الداتا", "File read, with data notes"),
    "file_bad": ("الملف ده مش قابل للاستخدام", "This file cannot be used"),
    "quality_line": ("{rows} قراءة · {dur} دقيقة · {n} إشارة متاحة", "{rows} readings · {dur} min · {n} signals"),
    "details": ("التفاصيل", "Details"),
    # trip
    "play": ("تشغيل", "Play"),
    "pause": ("إيقاف مؤقت", "Pause"),
    "resume": ("كمّل", "Resume"),
    "stop": ("إيقاف", "Stop"),
    "speed": ("السرعة:", "Speed:"),
    "replay_note": ("ده تشغيل لرحلة متسجلة، مش اتصال مباشر بالعربية.", "This replays a recorded trip — not a live connection."),
    "time_basis_relative": ("الوقت من تشغيل المحرك", "time since engine start"),
    "time_basis_synthetic": ("ساعة تشغيل افتراضية (الملف مفيهوش وقت)", "playback clock (no time in file)"),
    "time_basis_absolute": ("الوقت الفعلي المسجل", "recorded time"),
    "signals_to_show": ("الإشارات المعروضة", "Signals shown"),
    "na": ("مش متاحة", "N/A"),
    # results
    "analyse": ("حلّل الرحلة", "Analyse trip"),
    "reanalyse": ("حلّل تاني", "Analyse again"),
    "trip_summary": ("ملخص الرحلة", "Trip summary"),
    "findings": ("اللي لقيناه", "What we found"),
    "less_important": ("ملاحظات أقل أهمية ({n}) — لقاها موديل إحصائي واحد بس", "Less important observations ({n}) — found by one statistical model only"),
    "no_findings": ("مفيش حاجة تستاهل — كل القراءات طبيعية.", "Nothing to report — all readings look normal."),
    "what": ("إيه اللي حصل", "What happened"),
    "normal": ("الطبيعي للعربية", "What is normal"),
    "meaning": ("ممكن يعني إيه", "What it may mean"),
    "action": ("تعمل إيه", "What to do"),
    "evidence": ("إزاي عرفنا", "How we know"),
    "show_chart": ("وريني على الرسم", "Show on chart"),
    "conf_high": ("ثقة عالية", "High confidence"),
    "conf_medium": ("ثقة متوسطة", "Medium confidence"),
    "conf_low": ("ثقة قليلة", "Low confidence"),
    "sev_warning": ("مهم", "Important"),
    "sev_attention": ("ملاحظة", "Observation"),
    "tech_details": ("تفاصيل فنية للمتخصصين…", "Technical details…"),
    "disclaimer": ("البرنامج ده نموذج تعليمي، مش جهاز تشخيص. في أي شك اعرض العربية على ميكانيكي.",
                   "Educational prototype, not a diagnostic tool. When in doubt, see a qualified mechanic."),
    "how_it_works": ("البرنامج بيفحص الرحلة بـ 5 طرق", "The trip is checked with 5 methods"),
    "how_rules": ("قواعد حدود: زي «الحرارة فوق 105°م».", "Limit rules: e.g. “temperature above 105 °C”."),
    "how_eva": ("المتوقع مقابل الفعلي: الموديل اتعلم من رحلات العربية العادية القراءة المفروض تبقى كام، ويقارنها بالحقيقية.",
                "Expected vs actual: learned from normal trips what each reading should be, then compares."),
    "how_if": ("Isolation Forest: بيدوّر على لحظات شكلها مختلف عن العادي.", "Isolation Forest: finds moments that look different from normal."),
    "how_moment": ("MOMENT: موديل ذكاء اصطناعي جاهز للسلاسل الزمنية.", "MOMENT: a pretrained AI model for time series."),
    "how_consistency": ("اتساق الحساسات: بيراجع إن القراءات بتحكي قصة متوافقة (مفيش قفزات مستحيلة، مفيش عربية ماشية واللفات صفر، مفيش حساس اختفى).",
                        "Sensor consistency: checks the readings tell a physically consistent story (no impossible jumps, no moving car at 0 RPM, no sensor dropping out)."),
    # report
    "report_title": ("التقرير", "Report"),
    "report_text": ("اطلع تقرير PDF فيه ملخص الرحلة، النتيجة، الشرح، الرسومات، ومصدر الداتا — جاهز تطبعه أو تبعته.",
                    "Export a PDF with the trip summary, verdict, explanations, charts and data source — ready to print or share."),
    "export_pdf": ("اطلع تقرير PDF", "Export PDF report"),
    "export_csv_findings": ("اطلع النتايج CSV", "Export findings CSV"),
    "export_csv_data": ("اطلع الداتا CSV", "Export data CSV"),
    "report_lang": ("لغة التقرير:", "Report language:"),
    "saved_to": ("اتحفظ في:", "Saved to:"),
    "open_now": ("تفتحه دلوقتي؟", "Open it now?"),
    # live
    "live_title": ("توصيل مباشر بالعربية", "Live connection"),
    "live_steps": ("1) ركّب المحوّل في فتحة OBD تحت الدركسيون  2) دوّر الكونتاكت أو المحرك والعربية واقفة  3) دوس «اتصل»",
                   "1) Plug the adapter into the OBD port under the dashboard  2) Ignition on / engine idling, car parked  3) Press “Connect”"),
    "live_safety": ("للأمان: استخدمه والعربية راكنة. ماتستخدمش البرنامج وإنت سايق.", "Safety: use while parked. Never operate the app while driving."),
    "live_untested": ("الخاصية دي اتجربت على محوّل وهمي بس، لسه ماتجربتش على عربية حقيقية.", "Tested with a simulated adapter only — not yet on a real car."),
    "connect": ("اتصل", "Connect"),
    "disconnect": ("افصل", "Disconnect"),
    "simulate": ("جرّب بمحوّل وهمي (محاكاة)", "Try with a simulated adapter"),
    "searching": ("بيدوّر على المحوّل…", "Searching for adapter…"),
    "connected": ("متصل ✓", "Connected ✓"),
    "not_connected": ("مش متصل", "Not connected"),
    "supported_signals": ("القراءات اللي العربية بتدعمها:", "Signals this car supports:"),
    "record": ("● ابدأ تسجيل", "● Start recording"),
    "stop_record": ("■ وقّف وحلّل", "■ Stop & analyse"),
    "recording": ("بيسجّل… {n} قراءة", "Recording… {n} samples"),
    "obd_missing": ("مكتبة python-OBD مش متسطبة. شغّل: pip install -r requirements-obd-optional.txt",
                    "python-OBD is not installed. Run: pip install -r requirements-obd-optional.txt"),
    "close": ("قفل", "Close"),
    "error": ("حصلت مشكلة", "Something went wrong"),
}


def set_lang(lang: str) -> None:
    global _LANG
    _LANG = "ar" if lang == "ar" else "en"


def lang() -> str:
    return _LANG


def tr(key: str, **kw) -> str:
    ar, en = S.get(key, (key, key))
    text = ar if _LANG == "ar" else en
    return text.format(**kw) if kw else text


def pick(ar: str, en: str) -> str:
    return ar if _LANG == "ar" else en
