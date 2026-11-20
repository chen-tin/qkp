"""Continuously fetch small EC2 report artifacts; never download raw NPZ arrays."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("folder",type=Path)
    ap.add_argument("--interval",type=int,default=30)
    args=ap.parse_args();spec=json.loads((args.folder/"manifest.json").read_text())
    destination=args.folder/"metrics";destination.mkdir(exist_ok=True)
    prefix=["aws","--profile",spec["profile"],"--region",spec["region"],"--output","json"]
    terminal=False
    while True:
        result=subprocess.run([*prefix,"s3","sync",f's3://{spec["bucket"]}/{spec["run_id"]}/metrics/',str(destination),
                               "--only-show-errors","--no-progress"],capture_output=True,text=True,timeout=180)
        observed={"updated":datetime.now(timezone.utc).isoformat(),"fetch_returncode":result.returncode}
        if result.returncode:observed["error"]=result.stderr[-1000:]
        p=destination/"status.json"
        if p.exists():
            status=json.loads(p.read_text());observed["remote_status"]=status.get("status")
            terminal=status.get("status") in ["failed","interrupted","numerical_baseline_failed_or_inconclusive","all_scheduled_observations_completed"]
        (args.folder/"fetch_status.json").write_text(json.dumps(observed,indent=2))
        print(json.dumps(observed),flush=True)
        if terminal:
            state=subprocess.run([*prefix,"ec2","describe-instances","--instance-ids",spec["resources"]["instance_id"],
                                  "--query","Reservations[0].Instances[0].State.Name"],capture_output=True,text=True,timeout=30)
            if state.returncode==0 and json.loads(state.stdout) in ["stopped","terminated"]:break
        time.sleep(args.interval)


if __name__=="__main__":main()
