"""Training/application services for the advanced analysis workbench."""
from __future__ import annotations

from pathlib import Path

from ai.graph_network import GraphModel
from ai.probabilistic import ForecastModel
from ai.telemetry_learning import SIGNALS
from core.paths import MODELS_DIR, user_data_dir
from core.preprocessing.resample import resample_1hz


def reference_frames(repo, vehicle_id):
    if vehicle_id is None:
        raise ValueError("Assign the current trip to a car and approve reference trips in My cars first.")
    ids=repo.reference_sessions(vehicle_id)
    frames=[]
    for sid in ids:
        session,_=repo.load_session(sid)
        if session.meta.origin.value=="synthetic":
            raise ValueError("Synthetic trips cannot train a personal production reference model.")
        frames.append(resample_1hz(session.data,session.signals)[0])
    return ids,frames


def train_models(train, calibration, test, directory, provenance=None, epochs=35):
    if not train or not calibration or not test:
        raise ValueError("Separate train, calibration and test trips are required.")
    signals=[s for s in SIGNALS if all(s in f and f[s].notna().sum()>=90 for f in train+calibration)]
    if len(signals)<2:
        raise ValueError("Need two shared sensors and at least 90 observed seconds in each split.")
    graph=GraphModel.initialize(train,signals)
    graph.train(train,epochs=epochs)
    graph.calibrate(calibration)
    graph.meta.update(provenance or {})
    graph.meta["test_metrics"]=graph.evaluate(test)
    forecast=ForecastModel.fit(train,calibration,signals)
    forecast.meta.update(provenance or {})
    forecast.meta["test_metrics"]=forecast.evaluate(test)
    directory=Path(directory)
    graph.save(directory/"graph.npz"); forecast.save(directory/"forecast.npz")
    return {"graph_path":str(directory/"graph.npz"),"forecast_path":str(directory/"forecast.npz"),
            "signals":signals,"graph_test":graph.meta["test_metrics"],"forecast_test":forecast.meta["test_metrics"],
            "train_trips":len(train),"calibration_trips":len(calibration),"test_trips":len(test)}


def personal_training(ids, frames, vehicle_id, epochs=35):
    if len(frames)<5:
        raise ValueError("Advanced personal training needs five approved trips: 3+ train, 1 calibration, 1 test.")
    return train_models(frames[:-2],frames[-2:-1],frames[-1:],
                        user_data_dir()/"advanced"/str(vehicle_id),
                        {"scope":"vehicle","vehicle_id":vehicle_id,"train_ids":ids[:-2],
                         "calibration_ids":ids[-2:-1],"test_ids":ids[-1:]},epochs)


def default_paths(vehicle_id=None, signals=None):
    directory=MODELS_DIR/"advanced"
    if signals is not None and not set(SIGNALS)<=set(signals):
        directory=directory/"drivetrain"
    if vehicle_id is not None:
        personal=user_data_dir()/"advanced"/str(vehicle_id)
        if (personal/"graph.npz").exists() and (personal/"forecast.npz").exists():
            try:
                graph=GraphModel.load(personal/"graph.npz")
                forecast=ForecastModel.load(personal/"forecast.npz")
                if signals is None or set(graph.signals+forecast.signals)<=set(signals):
                    directory=personal
            except Exception:
                import logging
                logging.getLogger(__name__).warning("Personal advanced model is unreadable; using bundled model.")
    return directory/"graph.npz",directory/"forecast.npz"
