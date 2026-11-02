import zipfile

import numpy as np
import pandas as pd
import pytest

from ai.graph_network import GraphModel
from ai.telemetry_learning import load_npz, examples
from core.preprocessing.resample import resample_1hz
from test_advanced import trip, SIGNALS


@pytest.mark.parametrize("times", [[0,1e10],[0,float('nan')],[0,float('inf')]])
def test_bad_timestamps_fail_before_grid_allocation(times):
    with pytest.raises(ValueError): resample_1hz(pd.DataFrame({"t_s":times,"rpm":[900,900]}))


def test_eight_hour_analysis_is_chunked_and_training_bounded():
    frame=trip(n=8*3600)
    model=GraphModel.initialize([trip()],SIGNALS); model.train([trip()],epochs=1); model.calibrate([trip(2)])
    output=model.inspect(frame)
    assert output["expected"].shape==(8*3600,2) and np.isfinite(output["ratios"][10:]).all()
    x,_,_=examples([frame]*3,SIGNALS,model.mean,model.scale,limit=100)
    assert len(x)<=100


def test_malformed_and_duplicate_model_archives(tmp_path):
    broken=tmp_path/"broken.npz"; broken.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises((ValueError,zipfile.BadZipFile)): load_npz(broken)
    with zipfile.ZipFile(broken,"w") as archive:
        archive.writestr("metadata.npy",b"x")
        with pytest.warns(UserWarning): archive.writestr("metadata.npy",b"y")
    with pytest.raises(ValueError,match="duplicate"): load_npz(broken)


def test_disconnect_during_poll_then_reconnect_preserves_recording():
    from core.data_sources.obd_optional import OBDDataSource
    from test_obd_optional import FakeConn, fake_obd, Q
    first=FakeConn({"RPM","SPEED"},{"RPM":Q(900,"rpm"),"SPEED":Q(10,"kph")})
    second=FakeConn({"RPM"},{"RPM":Q(1000,"rpm")})
    clock=iter([10.,11.,12.,13.]).__next__
    source=OBDDataSource(enabled=True,obd_module=fake_obd([first,second]),clock=clock)
    source.open(); a=source.poll_once(); first.connected=False
    b=source.poll_once()
    assert len(source.recorded)==2 and b.t_s>a.t_s and b.values["rpm"]==1000
    assert source.read_session().data.rpm.tolist()==[900,1000]
    source.close()


def test_zero_fill_keeps_missing_seconds():
    result,_=resample_1hz(pd.DataFrame({"t_s":[0.,2.],"rpm":[900.,1000.]}),ffill_limit_s=0)
    assert np.isnan(result.rpm.iloc[1])
