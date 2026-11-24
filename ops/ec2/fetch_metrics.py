"""Continuously fetch small EC2 report artifacts; never download raw NPZ arrays."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ops.ec2.monitoring import read_archived_status, write_observation


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("folder",type=Path)
    ap.add_argument("--interval",type=int,default=30)
    args=ap.parse_args();spec=json.loads((args.folder/"manifest.json").read_text())
    destination=args.folder/"metrics";destination.mkdir(exist_ok=True)
    prefix=["aws","--profile",spec["profile"],"--region",spec["region"],"--output","json"]
    while True:
        observed={"updated":datetime.now(timezone.utc).isoformat()}
        try:
            result=subprocess.run([*prefix,"s3","sync",f's3://{spec["bucket"]}/{spec["run_id"]}/metrics/',str(destination),
                                   "--exact-timestamps","--only-show-errors","--no-progress"],capture_output=True,text=True,timeout=180)
            observed["fetch_returncode"]=result.returncode
            if result.returncode:observed["error"]=result.stderr[-1000:]
        except (subprocess.TimeoutExpired, OSError) as error:
            observed.update(fetch_returncode=None,error=str(error))
        # Query EC2 on every poll, even when archived application status says
        # running. An OS shutdown need not publish a final heartbeat.
        instance_state="unknown";query_error=None
        try:
            state=subprocess.run([*prefix,"ec2","describe-instances","--instance-ids",spec["resources"]["instance_id"],
                                  "--query","Reservations[0].Instances[0].State.Name"],capture_output=True,text=True,timeout=30)
            if state.returncode:
                query_error=state.stderr[-1000:]
            else:
                value=json.loads(state.stdout)
                instance_state=value if isinstance(value,str) else "unknown"
        except (subprocess.TimeoutExpired, ValueError, OSError) as error:
            query_error=str(error)
        overlay=write_observation(args.folder,read_archived_status(args.folder),instance_state,query_error=query_error)
        observed.update(remote_status=overlay["archived_status"],
                        effective_status=overlay["effective_status"],instance_state=instance_state)
        (args.folder/"fetch_status.json").write_text(json.dumps(observed,indent=2))
        print(json.dumps(observed),flush=True)
        if instance_state in ["stopped","terminated"]:
            break
        time.sleep(args.interval)


if __name__=="__main__":main()
