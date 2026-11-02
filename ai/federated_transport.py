"""Authenticated, bounded HTTPS exchange. Raw trips never enter this protocol.

Per-client HMAC keys authenticate requests; TLS authenticates the coordinator and
encrypts transport. This does not implement secure aggregation or differential privacy.
The coordinator must protect its key file and configure trusted participants.
"""
from __future__ import annotations

import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import sqlite3
import ssl
import tempfile
import threading
import time
from urllib.parse import urlsplit
from urllib.request import Request

from ai.federation import ClientUpdate, validate_update
from ai.graph_network import GraphModel

MAX_BYTES = 2 * 1024 * 1024


def signature(secret, method, path, stamp, nonce, body):
    message = "\n".join([method,path,stamp,nonce,hashlib.sha256(body).hexdigest()]).encode()
    return hmac.new(secret.encode(),message,hashlib.sha256).hexdigest()


def request(base_url, client_id, secret, method="GET", body=b"", ca_file=None):
    parsed=urlsplit(base_url)
    local=parsed.hostname in ("127.0.0.1","localhost","::1")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("Use a server origin URL without credentials, path or query.")
    if parsed.scheme!="https" and not (local and parsed.scheme=="http"):
        raise ValueError("Remote exchange requires HTTPS with certificate verification.")
    if len(secret)<32 or len(body)>MAX_BYTES:
        raise ValueError("Use a client secret of at least 32 characters and an update below 2 MB.")
    path="/round" if method=="GET" else "/update"
    stamp=str(int(time.time())); nonce=secrets.token_hex(16)
    headers={"X-Client":client_id,"X-Time":stamp,"X-Nonce":nonce,
             "X-Signature":signature(secret,method,path,stamp,nonce,body)}
    req=Request(base_url.rstrip("/")+path,data=body if method=="POST" else None,headers=headers,method=method)
    context=ssl.create_default_context(cafile=ca_file or None)
    # Redirects must not forward credentials to a different endpoint.
    from urllib.request import build_opener, HTTPSHandler, HTTPRedirectHandler
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs): return None
    opener=build_opener(HTTPSHandler(context=context),NoRedirect())
    with opener.open(req,timeout=20) as response:
        data=response.read(MAX_BYTES+1)
        if len(data)>MAX_BYTES: raise ValueError("Coordinator response is too large.")
        return data


def download_round(url, client_id, secret, output, ca_file=None):
    data=request(url,client_id,secret,ca_file=ca_file)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temp=Path(directory)/"round.npz"; temp.write_bytes(data)
        model=GraphModel.load(temp)
        if not model.meta.get("round_id"): raise ValueError("Coordinator did not provide a round model.")
        model.save(output)
    return {"path":str(output),"round_id":model.meta["round_id"]}


def upload_update(url, client_id, secret, path, ca_file=None):
    path=Path(path)
    if path.stat().st_size>MAX_BYTES: raise ValueError("Update exceeds 2 MB.")
    update=ClientUpdate.load(path)
    if update.client_id!=client_id: raise ValueError("Credentials must belong to the client that trained this update.")
    return json.loads(request(url,client_id,secret,"POST",path.read_bytes(),ca_file))


class Coordinator(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,round_path,clients,directory,cert=None,key=None):
        if address[0] not in ("127.0.0.1","localhost","::1") and not (cert and key):
            raise ValueError("A network listener requires a TLS certificate and private key.")
        if not clients or any(not isinstance(k,str) or not re.fullmatch(r"[\w:.-]{1,150}",k)
                              or not isinstance(v,str) or len(v)<32 for k,v in clients.items()):
            raise ValueError("Provide named clients with secrets of at least 32 characters.")
        self.base=GraphModel.load(round_path); self.round_id=self.base.meta.get("round_id")
        if not self.round_id: raise ValueError("Export a new round model first.")
        self.round_bytes=Path(round_path).read_bytes()
        if len(self.round_bytes)>MAX_BYTES: raise ValueError("Round exceeds exchange size limit.")
        self.clients=dict(clients); self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock(); self.db=self.directory/"receipts.sqlite3"
        self.slots=threading.BoundedSemaphore(8)
        with sqlite3.connect(self.db) as db:
            db.execute("CREATE TABLE IF NOT EXISTS nonces(client TEXT, nonce TEXT, expiry INTEGER, PRIMARY KEY(client,nonce))")
            db.execute("CREATE TABLE IF NOT EXISTS receipts(round TEXT,client TEXT,digest TEXT,PRIMARY KEY(round,client))")
        ctx=None
        if cert and key:
            ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.minimum_version=ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(cert,key)
        self.tls=ctx
        super().__init__(address,Handler)

    def get_request(self):
        connection,address=super().get_request()
        connection.settimeout(10)
        if self.tls is not None:
            try: connection=self.tls.wrap_socket(connection,server_side=True)
            except Exception:
                connection.close(); raise
        return connection,address

    def process_request(self,request,address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request); return
        try: super().process_request(request,address)
        except Exception:
            self.slots.release(); raise

    def process_request_thread(self,request,address):
        try: super().process_request_thread(request,address)
        finally: self.slots.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup(); self.connection.settimeout(10)

    def log_message(self,*args): pass  # Do not leak participant credentials into request logs.

    def reply(self,code,data):
        if not isinstance(data,bytes): data=json.dumps(data).encode()
        self.send_response(code); self.send_header("Content-Length",str(len(data)))
        self.send_header("Content-Type","application/octet-stream")
        self.end_headers(); self.wfile.write(data)

    def do_GET(self): self.handle_request()
    def do_POST(self): self.handle_request()

    def handle_request(self):
        try:
            if (self.command,self.path) not in (("GET","/round"),("POST","/update")):
                return self.reply(404,{"error":"Unknown endpoint"})
            if self.headers.get("Transfer-Encoding"):
                return self.reply(400,{"error":"Chunked requests are not supported"})
            size=int(self.headers.get("Content-Length","0"))
            if not 0<=size<=MAX_BYTES: return self.reply(413,{"error":"Request too large"})
            client=self.headers.get("X-Client",""); secret=self.server.clients.get(client)
            stamp=self.headers.get("X-Time",""); nonce=self.headers.get("X-Nonce","")
            if not secret or not re.fullmatch(r"[a-f0-9]{32}",nonce) or abs(time.time()-int(stamp))>120:
                return self.reply(401,{"error":"Invalid credentials or clock skew"})
            body=self.rfile.read(size)
            if len(body)!=size: return self.reply(400,{"error":"Incomplete request"})
            expected=signature(secret,self.command,self.path,stamp,nonce,body)
            if not hmac.compare_digest(expected,self.headers.get("X-Signature","")):
                return self.reply(401,{"error":"Invalid signature"})
            with self.server.lock, sqlite3.connect(self.server.db) as db:
                db.execute("DELETE FROM nonces WHERE expiry<?",(int(time.time()),))
                if db.execute("SELECT COUNT(*) FROM nonces WHERE client=?",(client,)).fetchone()[0]>=240:
                    return self.reply(429,{"error":"Client request quota exceeded; retry later"})
                try: db.execute("INSERT INTO nonces VALUES(?,?,?)",(client,nonce,int(time.time())+240))
                except sqlite3.IntegrityError: return self.reply(409,{"error":"Replayed request"})
            if self.command=="GET": return self.reply(200,self.server.round_bytes)
            with tempfile.TemporaryDirectory(dir=self.server.directory) as directory:
                incoming=Path(directory)/"update.npz"; incoming.write_bytes(body)
                update=ClientUpdate.load(incoming)
                if update.client_id!=client: return self.reply(403,{"error":"Client identity mismatch"})
                validate_update(self.server.base,update,self.server.round_id)
                digest=hashlib.sha256(body).hexdigest()
                filename=hashlib.sha256((self.server.round_id+":"+client).encode()).hexdigest()+".npz"
                with self.server.lock, sqlite3.connect(self.server.db) as db:
                    row=db.execute("SELECT digest FROM receipts WHERE round=? AND client=?",(self.server.round_id,client)).fetchone()
                    if row:
                        return self.reply(200 if row[0]==digest else 409,{"status":"already_received" if row[0]==digest else "conflicting_update"})
                    update.save(self.server.directory/filename)
                    db.execute("INSERT INTO receipts VALUES(?,?,?)",(self.server.round_id,client,digest))
            self.reply(201,{"status":"accepted","round_id":self.server.round_id})
        except (ConnectionError,TimeoutError):
            return  # A disconnected client can retry; no receipt was acknowledged.
        except (ValueError,KeyError,TypeError,OverflowError):
            self.reply(400,{"error":"Invalid request or update"})
        except Exception:
            self.reply(400,{"error":"Unreadable or incomplete update"})
