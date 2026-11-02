"""Advanced algorithms: held-out predictions, missing data, exchange and fusion."""
import copy
import wave

import numpy as np
import pandas as pd
import pytest

from ai.graph_network import GraphModel
from ai.probabilistic import ForecastModel
from ai.telemetry_learning import examples, features, load_npz, save_npz
from ai.federation import ClientUpdate, aggregate, local_update, new_round

SIGNALS = ["rpm", "engine_load_pct"]


def trip(seed=1, n=240):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    v = np.sin(t / 23 + seed / 10)
    return pd.DataFrame({"t_s": t, "rpm": 1400 + 300*v + rng.normal(0, 2, n),
                         "engine_load_pct": 40 + 12*v + rng.normal(0, .2, n)})


@pytest.fixture
def graph():
    model = GraphModel.initialize([trip()], SIGNALS)
    model.train([trip()], epochs=12)
    model.calibrate([trip(2)])
    return model


def test_graph_heldout_anomaly_and_roundtrip(graph, tmp_path):
    clean = trip(3)
    bad = clean.copy(); bad.loc[140:160, "rpm"] += 2500
    normal = graph.inspect(clean); anomaly = graph.inspect(bad)
    assert np.nanmedian(anomaly["ratios"][140:160, 0]) > np.nanmedian(normal["ratios"])*3
    assert anomaly["ranking"][0]["signal"] == "rpm"
    path = tmp_path / "graph.npz"; graph.save(path)
    loaded = GraphModel.load(path)
    assert loaded.fingerprint() == graph.fingerprint()
    np.testing.assert_allclose(loaded.inspect(clean)["ratios"], normal["ratios"], equal_nan=True)
    assert "persistence_mae" in loaded.evaluate([clean])["rpm"]


def test_causal_features_and_nonfinite_targets(graph):
    a = trip(); b = a.copy(); b.loc[120:, "rpm"] = 999999
    np.testing.assert_array_equal(features(a, SIGNALS, graph.mean, graph.scale)[:120],
                                  features(b, SIGNALS, graph.mean, graph.scale)[:120])
    a.loc[50, "rpm"] = np.inf
    _, y, mask = examples([a], SIGNALS, graph.mean, graph.scale)
    assert np.isfinite(y).all() and (~mask).any()
    a.loc[80:, SIGNALS] = np.nan
    assert np.isnan(graph.inspect(a)["ratios"][81:]).all()


def test_graph_rejects_untrusted_arrays(graph, tmp_path):
    path = tmp_path / "bad.npz"; graph.save(path)
    meta, arrays = load_npz(path)
    arrays["adjacency"][0, 1] = -1
    save_npz(path, meta, arrays)
    with pytest.raises(ValueError, match="adjacency"): GraphModel.load(path)
    arrays["mean"][0] = np.inf
    save_npz(path, meta, arrays)
    with pytest.raises(ValueError, match="numerical"): GraphModel.load(path)


def test_forecast_joint_paths_bands_roundtrip_and_missing(tmp_path):
    model = ForecastModel.fit([trip()], [trip(2)], SIGNALS)
    frame = trip(3)
    result = model.forecast(frame, paths=20, limits={"rpm": 1600})
    assert result["paths"].shape == (20, 3, 2)
    np.testing.assert_array_equal(result["t_s"], [249, 269, 299])
    assert np.all(result["lower"] <= result["center"]) and np.all(result["center"] <= result["upper"])
    assert result["probabilities"] == model.forecast(frame, seed=99, limits={"rpm": 1600})["probabilities"]
    assert 0 <= model.evaluate([frame])["rpm"]["10"]["coverage"] <= 1
    path = tmp_path / "f.npz"; model.save(path)
    np.testing.assert_allclose(ForecastModel.load(path).forecast(frame)["center"], result["center"])
    frame.loc[len(frame)-1, "rpm"] = np.nan
    with pytest.raises(ValueError, match="current reading"): model.forecast(frame)


def test_federated_two_clients_file_exchange_recalibration(graph, tmp_path):
    base = new_round(graph); rid = base.meta["round_id"]
    original = base.fingerprint()
    updates = []
    for client, seed in [("car-a", 3), ("car-b", 4)]:
        update = local_update(base, [trip(seed)], client, rid, epochs=2)
        path = tmp_path / (client + ".npz"); update.save(path)
        updates.append(ClientUpdate.load(path))
        _, arrays = load_npz(path)
        assert set(arrays) == set(base.params)  # exchange contains parameters, no telemetry rows
    combined = aggregate(base, updates, rid)
    assert base.fingerprint() == original and combined.fingerprint() != original
    with pytest.raises(ValueError, match="Calibrate"): combined.inspect(trip(5))
    combined.calibrate([trip(5)])
    assert combined.evaluate([trip(6)])["rpm"]["mae"] >= 0
    with pytest.raises(ValueError, match="Duplicate"): aggregate(base, [updates[0], updates[0]], rid)
    bad = copy.deepcopy(updates[1]); bad.base_hash = "wrong"
    with pytest.raises(ValueError, match="another"): aggregate(base, [updates[0], bad], rid)
    bad = copy.deepcopy(updates[1]); bad.delta["bias"][0] = np.nan
    with pytest.raises(ValueError, match="values"): aggregate(base, [updates[0], bad], rid)


def test_federated_clips_and_weights(graph):
    base = new_round(graph); rid = base.meta["round_id"]
    updates = [ClientUpdate(str(i), rid, base.fingerprint(), count,
                           {k: np.full_like(v, .001*i) for k, v in base.params.items()})
               for i, count in [(1, 100), (2, 300)]]
    combined = aggregate(base, updates, rid)
    np.testing.assert_allclose(combined.params["bias"]-base.params["bias"], .00175)
    clipped = aggregate(base, updates, rid, clip_norm=.001)
    norm = np.sqrt(sum(np.sum((clipped.params[k]-v)**2) for k,v in base.params.items()))
    assert norm <= .00100001


def test_active_scheduler_fairness_and_evidence():
    from ai.active_diagnosis import ActiveScheduler
    scheduler = ActiveScheduler(max_queries=2, budget_s=.25)
    supported = ["rpm", "speed_kmh", "battery_voltage", "coolant_temp_c", "engine_load_pct"]
    seen = set()
    for now in range(20):
        plan = scheduler.plan(supported, now)
        assert len(plan) <= 2 and sum(p["estimated_seconds"] for p in plan) <= .25
        for item in plan:
            seen.add(item["signal"]); scheduler.observe(item["signal"], None, now)
    assert seen == set(supported)
    scheduler = ActiveScheduler()
    for now in range(10): scheduler.observe("battery_voltage", 10, now)
    assert scheduler.summary()["patterns"]["voltage_pattern"] > .5


def test_active_obd_only_supported_read_queries():
    from test_obd_optional import FakeConn, fake_obd, Q
    from core.data_sources.obd_optional import OBDDataSource, READ_ONLY_PIDS
    conn = FakeConn(set(READ_ONLY_PIDS), {"RPM": Q(1000, "rpm")})
    clock = iter([0., 1., 2.]).__next__
    source = OBDDataSource(enabled=True, active=True, obd_module=fake_obd([conn]), clock=clock)
    source.open(); reading = source.poll_once()
    assert reading.t_s == 1 and 1 <= len(conn.queries) <= 4
    assert set(conn.queries) <= set(READ_ONLY_PIDS)
    assert any("ACTIVE_SAMPLING" in n for n in source.read_session().meta.notes)
    source.close()


def test_multimodal_files_alignment_and_event(tmp_path):
    from ai.multimodal import analyse_files, audio_features, export_event_clip
    from core.schema import NormalizedSession, SessionMetadata, DataOrigin, TimeBasis
    rate, seconds = 2000, 160
    t = np.arange(rate*seconds) / rate
    amplitude = np.where(t >= 130, .5, .05)
    mono = (32767*amplitude*np.sin(2*np.pi*100*t)).astype("<i2")
    wav = tmp_path / "sound.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(np.column_stack([mono, -mono]).astype("<i2").tobytes())
    assert audio_features(wav).audio_rms.iloc[0] > .03  # stereo must not cancel
    vt = np.arange(seconds*20) / 20
    vibration = tmp_path / "accel.csv"
    pd.DataFrame({"t_s": vt, "ax": .1*np.sin(2*np.pi*3*vt), "ay": np.zeros(len(vt)),
                  "az": np.full(len(vt), 9.81)}).to_csv(vibration, index=False)
    data = pd.DataFrame({"t_s": np.arange(seconds, dtype=float), "rpm": 1400., "engine_load_pct": 40.})
    session = NormalizedSession(data, SessionMetadata("test", DataOrigin.RECORDED, TimeBasis.RELATIVE_RECORDED))
    result = analyse_files(session, wav, vibration, reference_end=100)
    assert result["aligned_seconds"] == seconds
    assert not result["events"].empty and result["events"].t_s.min() >= 130
    clip=export_event_clip(wav,tmp_path/"event.wav",result["timeline"],135)
    with wave.open(clip["audio"], "rb") as w:
        assert w.getnframes()/w.getframerate() == 10
    readings=pd.read_csv(clip["readings"])
    assert readings.t_s.min()==130 and "rpm" in readings and readings.clip_time_s.min()==0
    with pytest.raises(ValueError, match="preserve"):
        export_event_clip(wav,wav,result["timeline"],135)
    with pytest.raises(ValueError, match="aligned"):
        analyse_files(session, wav, vibration, audio_offset=1000, reference_end=100)


def test_advanced_dialog_renders_graph_forecast(graph, tmp_path):
    from PySide6.QtWidgets import QApplication
    from database.connection import connect
    from database.repositories import Repository
    from app.state import AppState
    from app.pages.advanced_dialog import AdvancedDialog
    app = QApplication.instance() or QApplication([])
    state = AppState(Repository(connect(tmp_path / "ui.db")))
    dialog = AdvancedDialog(state)
    assert dialog.tabs.count() == 6
    dialog._kind = "graph"
    dialog._done({**graph.inspect(trip()), "signals": SIGNALS})
    assert "Completed" in dialog.status.text() or "اكتمل" in dialog.status.text()
    model = ForecastModel.fit([trip()], [trip(2)], SIGNALS)
    dialog._kind = "forecast"
    dialog._done({**model.forecast(trip()), "signals": SIGNALS})
    assert dialog.forecast_signal.count() == 2
    dialog.show(); app.processEvents()
    assert dialog.grab().save(str(tmp_path / "advanced.png"))
    dialog.close()
