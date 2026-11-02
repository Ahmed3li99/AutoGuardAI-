"""Loopback integration tests for the actual signed HTTP/TLS protocol."""
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from ai.federation import new_round, local_update
from ai.graph_network import GraphModel
from ai.federated_transport import Coordinator, download_round, upload_update, signature, MAX_BYTES
from test_advanced import trip, SIGNALS


@pytest.fixture
def peer(tmp_path):
    base=new_round(GraphModel.initialize([trip()],SIGNALS))
    path=tmp_path/"round.npz"; base.save(path)
    secret=secrets.token_hex(32)
    server=Coordinator(("127.0.0.1",0),path,{"car-a":secret},tmp_path/"inbox")
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try: yield server,base,secret,f"http://127.0.0.1:{server.server_port}"
    finally: server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_authenticated_fetch_upload_idempotent_retry(peer,tmp_path):
    server,base,secret,url=peer
    fetched=tmp_path/"fetched.npz"
    download_round(url,"car-a",secret,fetched)
    assert GraphModel.load(fetched).fingerprint()==base.fingerprint()
    update=local_update(base,[trip(3)],"car-a",base.meta["round_id"],epochs=1)
    path=tmp_path/"update.npz"; update.save(path)
    assert upload_update(url,"car-a",secret,path)["status"]=="accepted"
    assert upload_update(url,"car-a",secret,path)["status"]=="already_received"
    assert len(list(server.directory.glob("*.npz")))==1
    update.delta["bias"]+=.1; update.save(path)
    with pytest.raises(HTTPError) as exc: upload_update(url,"car-a",secret,path)
    assert exc.value.code==409


def test_reject_wrong_key_tamper_and_replay(peer):
    _,_,secret,url=peer
    with pytest.raises(HTTPError) as exc:
        download_round(url,"car-a","x"*64,Path("unused.npz"))
    assert exc.value.code==401
    stamp=str(int(time.time())); nonce=secrets.token_hex(16)
    headers={"X-Client":"car-a","X-Time":stamp,"X-Nonce":nonce,
             "X-Signature":signature(secret,"GET","/round",stamp,nonce,b"")}
    req=Request(url+"/round",headers=headers)
    with urlopen(req,timeout=5) as response: assert response.status==200
    with pytest.raises(HTTPError) as exc: urlopen(req,timeout=5)
    assert exc.value.code==409
    headers["X-Nonce"]=secrets.token_hex(16) # invalidates the signature
    with pytest.raises(HTTPError) as exc: urlopen(Request(url+"/round",headers=headers),timeout=5)
    assert exc.value.code==401


def test_reject_wrong_round_and_oversized_request(peer,tmp_path):
    _,base,secret,url=peer
    update=local_update(base,[trip()],"car-a","wrong-round",epochs=1)
    path=tmp_path/"bad.npz"; update.save(path)
    with pytest.raises(HTTPError) as exc: upload_update(url,"car-a",secret,path)
    assert exc.value.code==400
    with pytest.raises(HTTPError) as exc:
        urlopen(Request(url+"/update",data=b"",headers={"Content-Length":str(MAX_BYTES+1)},method="POST"),timeout=5)
    assert exc.value.code==413


def test_remote_plaintext_refused(tmp_path):
    with pytest.raises(ValueError,match="HTTPS"):
        download_round("http://192.0.2.1:8443","car-a","x"*64,tmp_path/"never.npz")


def test_tls_certificate_verification(tmp_path):
    openssl=shutil.which("openssl") or "C:/Program Files/Git/usr/bin/openssl.exe"
    if not Path(openssl).exists(): pytest.skip("OpenSSL needed to generate a temporary test certificate")
    cert=tmp_path/"cert.pem"; key=tmp_path/"key.pem"
    subprocess.run([openssl,"req","-x509","-newkey","rsa:2048","-nodes","-days","1",
                    "-subj","/CN=localhost","-addext","subjectAltName=IP:127.0.0.1,DNS:localhost",
                    "-keyout",str(key),"-out",str(cert)],check=True,capture_output=True)
    model=new_round(GraphModel.initialize([trip()],SIGNALS)); path=tmp_path/"base.npz"; model.save(path)
    secret=secrets.token_hex(32)
    with Coordinator(("127.0.0.1",0),path,{"car-a":secret},tmp_path/"tls",str(cert),str(key)) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            url=f"https://127.0.0.1:{server.server_port}"
            with pytest.raises(URLError): download_round(url,"car-a",secret,tmp_path/"untrusted.npz")
            result=download_round(url,"car-a",secret,tmp_path/"trusted.npz",str(cert))
            assert result["round_id"]==model.meta["round_id"]
        finally: server.shutdown(); thread.join(timeout=5)
