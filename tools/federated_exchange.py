"""Explicit coordinator/client CLI; never starts a listener automatically."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ai.federated_transport import Coordinator, download_round, upload_update


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    server=sub.add_parser("serve")
    server.add_argument("--round",required=True); server.add_argument("--clients",required=True)
    server.add_argument("--directory",required=True); server.add_argument("--bind",default="127.0.0.1")
    server.add_argument("--port",type=int,default=8443); server.add_argument("--cert"); server.add_argument("--key")
    for command in ("fetch","upload"):
        p=sub.add_parser(command); p.add_argument("--url",required=True); p.add_argument("--credentials",required=True)
        p.add_argument("--file",required=True); p.add_argument("--ca")
    args=parser.parse_args()
    if args.command=="serve":
        clients=json.loads(Path(args.clients).read_text(encoding="utf-8"))
        with Coordinator((args.bind,args.port),args.round,clients,args.directory,args.cert,args.key) as server:
            print(f"Coordinator ready on {args.bind}:{server.server_port}; press Ctrl+C to stop.",flush=True)
            try: server.serve_forever()
            except KeyboardInterrupt: pass
    else:
        credentials=json.loads(Path(args.credentials).read_text(encoding="utf-8"))
        fn=download_round if args.command=="fetch" else upload_update
        print(json.dumps(fn(args.url,credentials["client_id"],credentials["secret"],args.file,args.ca)))


if __name__=="__main__": main()
