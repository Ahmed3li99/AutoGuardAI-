"""Bilingual workbench for graph, probabilistic, active, fusion and federated research."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from ai.active_diagnosis import simulate_acquisition
from ai.advanced_service import default_paths, personal_training, reference_frames
from ai.federation import ClientUpdate, aggregate, local_update, new_round
from ai.graph_network import GraphModel
from ai.multimodal import analyse_files, export_event_clip
from ai.probabilistic import ForecastModel
from app.i18n import pick
from app.state import Worker
from app.widgets.common import signal_label
from core.preprocessing.resample import resample_1hz


class AdvancedDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state=state
        # A dialog owns an immutable trip snapshot even if a background load completes.
        self.session=state.session
        self.vid=state.repo.session_vehicle(state.session_id) if state.session_id is not None else None
        self.graph_path,self.forecast_path=default_paths(self.vid,self.session.signals if self.session is not None else None)
        self.busy=False; self.buttons=[]; self.last_output=None
        self.setWindowTitle(pick("التحليل المتقدم", "Advanced analysis")); self.resize(1180,850)
        root=QVBoxLayout(self)
        intro=QLabel(pick("خمس أدوات بحثية تعمل محلياً. نتائجها منفصلة عن الحكم الأساسي على العربية؛ "
                         "العلاقات ليست أسباباً مؤكدة، ونطاقات التوقع تحتاج معايرة مناسبة للعربية.",
                         "Five local research tools. Results are separate from the main vehicle verdict; "
                         "associations are not causes and forecast bands need appropriate vehicle calibration."))
        intro.setWordWrap(True); root.addWidget(intro)
        tabs=QTabWidget(); root.addWidget(tabs,1)
        self.tabs=tabs
        self._models_tab(); self._graph_tab(); self._forecast_tab(); self._active_tab(); self._fusion_tab(); self._federated_tab()
        self.status=QLabel(""); self.status.setWordWrap(True); root.addWidget(self.status)
        row=QHBoxLayout()
        export=QPushButton(pick("تصدير نتيجة الأداة JSON", "Export tool result as JSON")); export.clicked.connect(self.export)
        row.addWidget(export); row.addStretch()
        close=QPushButton(pick("إغلاق", "Close")); close.clicked.connect(self.close); row.addWidget(close)
        root.addLayout(row)
        self._model_text()

    def _tab(self,ar,en):
        widget=QWidget(); layout=QVBoxLayout(widget); self.tabs.addTab(widget,pick(ar,en)); return layout

    def _button(self,layout,ar,en,fn):
        b=QPushButton(pick(ar,en)); b.clicked.connect(fn); layout.addWidget(b); self.buttons.append(b); return b

    def _text(self,layout):
        text=QTextBrowser(); layout.addWidget(text); return text

    def _plot_widget(self,layout):
        plot=pg.PlotWidget(); plot.setBackground("#FFFFFF"); plot.showGrid(x=True,y=True,alpha=.15)
        layout.addWidget(plot,1); return plot

    def _grid(self):
        if self.session is None: raise ValueError("Open a trip first.")
        return resample_1hz(self.session.data,self.session.signals)[0]

    def _run(self,kind,fn,*args):
        if self.busy: return
        self.busy=True; self._kind=kind
        for b in self.buttons: b.setEnabled(False)
        self.status.setText(pick("جارٍ الحساب…", "Working…"))
        worker=Worker(fn,*args)
        worker.signals.finished.connect(self._done); worker.signals.failed.connect(self._failed)
        self.state.pool.start(worker)

    def _guard(self,fn):
        try: fn()
        except Exception as exc: self._failed(str(exc))

    def _models_tab(self):
        lay=self._tab("الموديلات","Models")
        self._button(lay,"اختيار موديل شبكة حساسات NPZ","Choose graph model NPZ",self.choose_graph)
        self._button(lay,"اختيار موديل توقع NPZ","Choose forecast model NPZ",self.choose_forecast)
        self._button(lay,"تدريب الموديلين من رحلات عربيتي المرجعية","Train both on my car's reference trips",self.train)
        self.models_text=self._text(lay)

    def _model_text(self):
        self.models_text.setPlainText(pick("موديل الشبكة: ","Graph model: ")+str(self.graph_path)+"\n"
            +pick("موديل المستقبل: ","Forecast model: ")+str(self.forecast_path)+"\n\n"
            +pick("التدريب الشخصي يحتاج 5 رحلات مرجعية على الأقل. آخر رحلتين منفصلتان للمعايرة والاختبار. "
                  "الموديلات المرفقة مدربة على تويوتا واحدة؛ نتائج عربية مختلفة بحثية وغير معايرة.",
                  "Personal training needs at least 5 approved trips. The last two are separate calibration and test trips. "
                  "Bundled models are trained on one Toyota; other-car results are research-only and uncalibrated."))

    def choose_graph(self):
        path,_=QFileDialog.getOpenFileName(self,"Graph model","","Models (*.npz)")
        if path: self._guard(lambda:self._set_graph(path))

    def _set_graph(self,path):
        GraphModel.load(path); self.graph_path=Path(path); self._model_text()

    def choose_forecast(self):
        path,_=QFileDialog.getOpenFileName(self,"Forecast model","","Models (*.npz)")
        if path: self._guard(lambda:self._set_forecast(path))

    def _set_forecast(self,path):
        ForecastModel.load(path); self.forecast_path=Path(path); self._model_text()

    def train(self):
        def start():
            ids,frames=reference_frames(self.state.repo,self.vid)
            self._run("train",personal_training,ids,frames,self.vid)
        self._guard(start)

    def _graph_tab(self):
        lay=self._tab("شبكة الحساسات","Sensor graph")
        self._button(lay,"تحليل العلاقات وترتيب القراءات غير المعتادة","Analyse relations and rank deviations",self.graph)
        self.graph_plot=self._plot_widget(lay); self.graph_text=self._text(lay)

    def graph(self):
        def start():
            grid=self._grid(); path=self.graph_path
            def work():
                model=GraphModel.load(path)
                return {**model.inspect(grid),"signals":model.signals,"model":model.meta,"t_s":grid.t_s.to_numpy()}
            self._run("graph",work)
        self._guard(start)

    def _forecast_tab(self):
        lay=self._tab("مستقبلات محتملة","Possible futures")
        self.forecast_signal=QComboBox(); lay.addWidget(self.forecast_signal)
        self.forecast_signal.currentIndexChanged.connect(self._draw_forecast)
        self._button(lay,"توقع 10 و30 و60 ثانية","Forecast 10, 30 and 60 seconds",self.forecast)
        self.forecast_plot=self._plot_widget(lay); self.forecast_text=self._text(lay)
        self.forecast_result=None

    def forecast(self):
        def start():
            grid=self._grid(); path=self.forecast_path
            def work():
                model=ForecastModel.load(path)
                return {**model.forecast(grid,limits={"coolant_temp_c":105.,"battery_voltage":15.,"rpm":4500.}),
                        "signals":model.signals,"model":model.meta}
            self._run("forecast",work)
        self._guard(start)

    def _draw_forecast(self):
        r=self.forecast_result
        if r is None or self.forecast_signal.currentIndex()<0: return
        i=self.forecast_signal.currentIndex(); p=self.forecast_plot; p.clear()
        for trajectory in r["paths"][:30]: p.plot(r["t_s"],trajectory[:,i],pen=pg.mkPen("#A5B4FC55"))
        low=p.plot(r["t_s"],r["lower"][:,i],pen=pg.mkPen("#60A5FA"))
        high=p.plot(r["t_s"],r["upper"][:,i],pen=pg.mkPen("#60A5FA"))
        p.addItem(pg.FillBetweenItem(low,high,brush=pg.mkBrush("#60A5FA33")))
        p.plot(r["t_s"],r["center"][:,i],pen=pg.mkPen("#1C4FD8",width=3))
        p.setLabel("bottom",pick("زمن الرحلة بالثانية","Trip time (seconds)"))
        p.setLabel("left",signal_label(r["signals"][i]))

    def _active_tab(self):
        lay=self._tab("الفحص النشط","Active acquisition")
        self._button(lay,"محاكاة اختيار القراءات على الرحلة","Simulate adaptive queries on this trip",self.active)
        self.active_text=self._text(lay)
        self.active_text.setPlainText(pick("لتطبيق الاختيار على المحوّل فعلياً: افتح الاتصال بالعربية وفعّل الفحص النشط. "
                                          "هنا مقارنة عدد الاستعلامات على تسجيل سابق، وليست تجربة جهاز حقيقي.",
                                          "For actual adapter scheduling, enable Active acquisition in Connect to car. "
                                          "This tab replays recorded data; it is not a hardware benchmark."))

    def active(self):
        self._guard(lambda:self._run("active",simulate_acquisition,self._grid()))

    def _fusion_tab(self):
        lay=self._tab("صوت + اهتزاز + OBD","Audio + vibration + OBD")
        form=QFormLayout(); lay.addLayout(form)
        self.audio_path=QLineEdit(); self.vibration_path=QLineEdit()
        form.addRow("WAV (16-bit PCM)",self.audio_path); form.addRow("CSV: t_s, ax, ay, az (m/s²)",self.vibration_path)
        self._button(lay,"اختيار تسجيل الصوت","Choose audio recording",lambda:self._choose_media(self.audio_path,"WAV (*.wav)"))
        self._button(lay,"اختيار تسجيل الاهتزاز","Choose vibration recording",lambda:self._choose_media(self.vibration_path,"CSV (*.csv)"))
        self.audio_offset=QDoubleSpinBox(); self.vibration_offset=QDoubleSpinBox(); self.reference_end=QDoubleSpinBox()
        for w in (self.audio_offset,self.vibration_offset): w.setRange(-86400,86400); w.setDecimals(2)
        self.reference_end.setRange(0,86400); self.reference_end.setValue(120)
        form.addRow(pick("بداية الصوت على ساعة الرحلة (ث)","Audio start on trip clock (s)"),self.audio_offset)
        form.addRow(pick("إزاحة وقت الاهتزاز (ث)","Vibration clock offset (s)"),self.vibration_offset)
        form.addRow(pick("نهاية الفترة المرجعية السليمة (ث)","Known-normal reference ends at (s)"),self.reference_end)
        self._button(lay,"مزامنة وتحليل التسجيلات","Synchronize and analyse recordings",self.fusion)
        self.fusion_result=None
        self.fusion_events=QComboBox(); lay.addWidget(self.fusion_events)
        self._button(lay,"تصدير مقطع الحدث مع القراءات","Export event audio clip and readings",self.export_fusion_event)
        self.fusion_plot=self._plot_widget(lay); self.fusion_text=self._text(lay)

    def _choose_media(self,edit,filt):
        path,_=QFileDialog.getOpenFileName(self,pick("اختر التسجيل","Choose recording"),"",filt)
        if path: edit.setText(path)

    def fusion(self):
        def start():
            if self.session is None: raise ValueError("Open a trip first.")
            self._fusion_audio=self.audio_path.text()
            self._fusion_offset=self.audio_offset.value()
            self._run("fusion",analyse_files,self.session,self.audio_path.text(),self.vibration_path.text(),
                      self.audio_offset.value(),self.vibration_offset.value(),self.reference_end.value())
        self._guard(start)

    def export_fusion_event(self):
        def start():
            if self.fusion_result is None or self.fusion_events.currentIndex()<0:
                raise ValueError("Analyse recordings and select an event first.")
            path,_=QFileDialog.getSaveFileName(self,"Event clip","event.wav","WAV (*.wav)")
            if path:
                info=export_event_clip(self._fusion_audio,path,self.fusion_result["timeline"],
                                       float(self.fusion_events.currentData()),self._fusion_offset)
                self.status.setText(json.dumps(info,ensure_ascii=False))
        self._guard(start)

    def _federated_tab(self):
        lay=self._tab("تعلّم اتحادي","Federated learning")
        form=QFormLayout(); lay.addLayout(form)
        self.exchange_url=QLineEdit(); self.exchange_url.setPlaceholderText("https://coordinator:8443")
        self.exchange_id=QLineEdit(); self.exchange_secret=QLineEdit()
        self.exchange_secret.setEchoMode(QLineEdit.Password)
        self.exchange_ca=QLineEdit()
        form.addRow(pick("عنوان المنسق","Coordinator URL"),self.exchange_url)
        form.addRow(pick("معرّف العميل","Client identity"),self.exchange_id)
        form.addRow(pick("مفتاح العميل (لا يتم حفظه)","Client key (not saved)"),self.exchange_secret)
        form.addRow(pick("ملف شهادة CA (اختياري)","CA certificate file (optional)"),self.exchange_ca)
        self._button(lay,"تنزيل موديل الجولة من المنسق","Download round from coordinator",self.fetch_round)
        self._button(lay,"إرسال تحديث للمنسق","Upload update to coordinator",self.send_update)
        self._button(lay,"تصدير موديل بدء جولة جديدة","Export base model for a new round",self.export_round)
        self._button(lay,"تدريب عميل وتصدير تحديث فقط","Train client and export update only",self.export_update)
        self._button(lay,"دمج تحديثات العملاء ومعايرة النتيجة","Aggregate client updates and recalibrate",self.combine)
        self.federated_text=self._text(lay)
        self.federated_text.setPlainText(pick(
            "تبادل ملفات بين أجهزة أو عربيات: وزّع موديل بدء الجولة، وكل عميل يصدّر تحديث أوزان فقط؛ "
            "الرحلات لا تدخل ملف التحديث. اختَر موديل الجولة من تبويب الموديلات قبل تدريب العميل أو الدمج. "
            "يتطلب الدمج عميلين مختلفين ومعايرة محلية. للمشاركين الموثوقين فقط: التحديثات ليست خصوصية تفاضلية أو تجميعاً مشفراً.",
            "File-based federation across PCs/cars: distribute the round's base model; each client exports weight deltas only, "
            "without trips. Select the round model in Models before client training/aggregation. "
            "Aggregation needs two distinct clients and local calibration. Trusted participants only: "
            "updates are not differentially private or securely aggregated."))

    def fetch_round(self):
        from ai.federated_transport import download_round
        def start():
            path,_=QFileDialog.getSaveFileName(self,"Round model","downloaded_round.npz","Models (*.npz)")
            if path:
                self._run("download",download_round,self.exchange_url.text(),self.exchange_id.text(),
                          self.exchange_secret.text(),path,self.exchange_ca.text() or None)
        self._guard(start)

    def send_update(self):
        from ai.federated_transport import upload_update
        def start():
            path,_=QFileDialog.getOpenFileName(self,"Client update","","Updates (*.npz)")
            if path:
                self._run("federated",upload_update,self.exchange_url.text(),self.exchange_id.text(),
                          self.exchange_secret.text(),path,self.exchange_ca.text() or None)
        self._guard(start)

    def export_round(self):
        def start():
            model=new_round(GraphModel.load(self.graph_path))
            path,_=QFileDialog.getSaveFileName(self,"Round model","round.npz","Models (*.npz)")
            if path:
                model.save(path); self.graph_path=Path(path); self._model_text()
                self.status.setText(pick("تم تصدير موديل الجولة: ","Round model exported: ")+model.meta["round_id"])
        self._guard(start)

    def export_update(self):
        def start():
            model=GraphModel.load(self.graph_path); rid=model.meta.get("round_id")
            if not rid: raise ValueError("Select a new-round model first.")
            ids,frames=reference_frames(self.state.repo,self.vid)
            if len(frames)<4: raise ValueError("Need four approved trips; the newest two are reserved for calibration and testing.")
            path,_=QFileDialog.getSaveFileName(self,"Client update","update.npz","Updates (*.npz)")
            if not path: return
            # A persistent installation id prevents vehicle 1 on two PCs being mistaken for one client.
            import uuid
            device=self.state.repo.get_setting("federated_device_id")
            if not device:
                device=uuid.uuid4().hex; self.state.repo.set_setting("federated_device_id",device)
            client_id=self.exchange_id.text().strip() or f"{device}:{self.vid}"
            def work():
                update=local_update(model,frames[:-2],client_id,rid)
                update.save(path)
                return {"path":path,"client_id":client_id,"round_id":rid,"samples":update.samples,"export":"weights only; no trip rows"}
            self._run("federated",work)
        self._guard(start)

    def combine(self):
        def start():
            model=GraphModel.load(self.graph_path)
            paths,_=QFileDialog.getOpenFileNames(self,"Client updates","","Updates (*.npz)")
            if not paths: return
            ids,frames=reference_frames(self.state.repo,self.vid)
            if len(frames)<2: raise ValueError("Need separate local calibration and evaluation trips.")
            path,_=QFileDialog.getSaveFileName(self,"Aggregated model","federated_graph.npz","Models (*.npz)")
            if not path: return
            if Path(path).resolve()==self.graph_path.resolve():
                raise ValueError("Save the new round to a different file to preserve the base model.")
            def work():
                result=aggregate(model,[ClientUpdate.load(p) for p in paths],model.meta.get("round_id"))
                result.calibrate(frames[-2:-1]); result.meta["test_metrics"]=result.evaluate(frames[-1:])
                result.save(path)
                return {"path":path,"audit":result.meta["update_audit"],"test":result.meta["test_metrics"]}
            self._run("federated",work)
        self._guard(start)

    def _done(self,result):
        try:
            self.last_output={"tool":self._kind,"result":result}
            if self._kind=="train":
                self.graph_path=Path(result["graph_path"]); self.forecast_path=Path(result["forecast_path"])
                self._model_text(); self.models_text.append(json.dumps(result,indent=2))
            elif self._kind=="graph":
                self._draw_graph(result)
                self.graph_text.setPlainText(pick("الترتيب حسب الانحراف، وليس احتمال عطل:\n","Ranked by deviation, not fault probability:\n")+
                    "\n".join(f"{signal_label(r['signal'])}: {r['peak_ratio']:.2f} ×; {r['fraction_above']:.1%}"
                              for r in result["ranking"]))
            elif self._kind=="forecast":
                self.forecast_result=result; self.forecast_signal.clear()
                for s in result["signals"]: self.forecast_signal.addItem(signal_label(s))
                self._draw_forecast()
                notices=[pick(f"تنبيه بحثي: احتمال تجاوز {signal_label(s)} للحد {p['limit']} بلغ 50% أو أكثر.\n",
                              f"Research alert: {s} exceeds {p['limit']} in at least 50% of residual scenarios.\n")
                         for s,p in result["probabilities"].items() if max(p["above"],default=0)>=.5]
                self.forecast_text.setPlainText("".join(notices)+pick("النطاق 90% لكل نقطة منفردة على بيانات مشابهة للمعايرة؛ ليس ضماناً للمسار كله.\n",
                                                     "90% marginal calibration bands on similar data; not a whole-path guarantee.\n")+
                                               json.dumps(result["probabilities"],indent=2))
            elif self._kind=="active":
                self.active_text.setPlainText(pick("محاكاة تسجيل — ليست قياس دقة تشخيص:\n","Recorded replay — not diagnostic accuracy:\n")+
                    f"{result['query_count']} / {result['fixed_scan_count']} queries\n"+json.dumps(result["final"],indent=2))
            elif self._kind=="fusion":
                self.fusion_result=result; self.fusion_events.clear()
                for row in result["events"].itertuples():
                    self.fusion_events.addItem(f"{row.t_s:.0f} s — {row.feature}",float(row.t_s))
                d=result["timeline"]; self.fusion_plot.clear()
                self.fusion_plot.plot(d.t_s.to_numpy(),d.fusion_score.to_numpy(),pen=pg.mkPen("#1C4FD8",width=2),connect="finite")
                self.fusion_plot.addItem(pg.InfiniteLine(pos=1,angle=0,pen=pg.mkPen("#C2283A")))
                self.fusion_text.setPlainText(pick("ثواني متزامنة / الإجمالي: ","Aligned / total seconds: ")+
                    f"{result['aligned_seconds']} / {result['total_seconds']}\n"+result["events"][["t_s","fusion_score","feature"]].to_string(index=False))
            elif self._kind in ("federated","download"):
                if self._kind=="download":
                    self.graph_path=Path(result["path"]); self._model_text()
                self.federated_text.setPlainText(json.dumps(result,indent=2))
            self.status.setText(pick("اكتمل الحساب. النتائج البحثية متاحة للتصدير.","Completed. Research results can be exported."))
        except Exception as exc:
            self.status.setText(str(exc))
        finally:
            self.busy=False
            for b in self.buttons: b.setEnabled(True)

    def _draw_graph(self,r):
        p=self.graph_plot; p.clear(); n=len(r["signals"])
        angle=np.arange(n)*2*np.pi/n; xy=np.column_stack([np.cos(angle),np.sin(angle)])
        for edge in r["edges"]:
            i=r["signals"].index(edge["from"]); j=r["signals"].index(edge["to"])
            p.plot(xy[[i,j],0],xy[[i,j],1],pen=pg.mkPen("#94A3B8",width=1+3*edge["weight"]))
        p.plot(xy[:,0],xy[:,1],pen=None,symbol="o",symbolSize=20,symbolBrush="#22D3EE")
        for i,s in enumerate(r["signals"]):
            text=pg.TextItem(signal_label(s),color="#0F1B2D",anchor=(.5,-.4)); text.setPos(*xy[i]); p.addItem(text)
        p.setAspectLocked(); p.hideAxis("left"); p.hideAxis("bottom")

    def _failed(self,msg):
        self.busy=False
        for b in self.buttons: b.setEnabled(True)
        self.status.setText(pick("تعذر إكمال العملية: ","Could not complete: ")+msg)

    def export(self):
        if self.last_output is None: return
        path,_=QFileDialog.getSaveFileName(self,"Export result","advanced_result.json","JSON (*.json)")
        if not path: return
        def convert(value):
            import pandas as pd
            if isinstance(value,pd.DataFrame): return convert(value.to_dict(orient="records"))
            if isinstance(value,np.ndarray): return convert(value.tolist())
            if isinstance(value,dict): return {str(k):convert(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)): return [convert(v) for v in value]
            if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
            if isinstance(value,np.integer): return int(value)
            return value
        self._guard(lambda:Path(path).write_text(json.dumps(convert(self.last_output),ensure_ascii=False,indent=2,
                                                          allow_nan=False),encoding="utf-8"))

    def reject(self):
        if not self.busy: super().reject()

    def closeEvent(self,event):
        if self.busy:
            self.status.setText(pick("انتظر اكتمال العملية قبل الإغلاق.","Wait for the operation to finish before closing."))
            event.ignore()
        else: super().closeEvent(event)
