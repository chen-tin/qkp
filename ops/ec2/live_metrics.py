"""Read-only live export of per-decoder rows before a whole job finishes."""
import argparse
from datetime import datetime,timezone
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True);ap.add_argument("--bucket",required=True)
    ap.add_argument("--run-id",required=True)
    args=ap.parse_args()
    os.environ.setdefault("MPLBACKEND","Agg")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=args.root/"live_export";output.mkdir(exist_ok=True,parents=True)
    previous=None
    while True:
        paths=sorted((args.root/"plans").glob("*/jobs/*/*/*_summary.json"))
        signature=tuple((str(p),p.stat().st_mtime_ns) for p in paths)
        status_path=args.root/"metrics/status.json"
        state=json.loads(status_path.read_text()) if status_path.exists() else {"status":"preparing_remote_environment"}
        page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Live GKP production metrics</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;color:#24334a}img{max-width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse;width:100%}</style><h1>Live GKP production metrics</h1>'
        page+="<p>EC2 status: <strong>"+html.escape(state["status"])+"</strong>. Updated "+datetime.now(timezone.utc).isoformat()+".</p>"
        page+="<p>Curves appear as each decoder finishes, before the full suite completes. Production numerical/physics/noise runs have8cycles plus initialization; finite-gate controls have their own2cycle budget. Failed numerical gates mark diagnostic curves.</p>"
        rows=[(p,json.loads(p.read_text())) for p in paths]
        stages={}
        for path,row in rows:stages.setdefault(path.parents[3].name,[]).append((path,row))
        if not rows:page+="<p>Worker is preparing or computing its first decoder. No production observations have finished yet.</p>"
        for preset,group in stages.items():
            page+="<h2>"+html.escape(preset)+f" — {len(group)} decoder/cutoff observations</h2>"
            if signature!=previous:
                fig,axes=plt.subplots(1,2,figsize=(13,4.7))
                for path,row in group[:12]:
                    values=np.array(row["finite_ancilla"]["mean"]);t=np.arange(len(values))
                    label=f'{row["decoder"]} / N{row["cutoff"]} / {path.parents[1].name[:6]}'
                    line,=axes[0].plot(t,values,"o-",ms=3,label=label)
                    if not row["numerical_edge_gate_pass"]:line.set_linestyle("--")
                    axes[0].fill_between(t,row["finite_ancilla"]["ci95_low"],row["finite_ancilla"]["ci95_high"],color=line.get_color(),alpha=.1)
                    raw_path=path.parent/(row["decoder"]+"_finite_ancilla.npz")
                    with np.load(raw_path) as raw:
                        energy=raw["energy"]
                        axes[1].plot(t,energy.mean(0),color=line.get_color())
                        axes[1].plot(t,energy.max(0),":",color=line.get_color(),alpha=.7)
                axes[0].set(title="Logical fidelity: pointwise95% intervals",xlabel="Cycle",ylabel="Six-state mean fidelity",ylim=(0,1.02))
                axes[1].set(title="Energy: solid mean / dotted maximum",xlabel="Cycle",ylabel="Photons")
                for axis in axes:axis.grid(alpha=.2)
                fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,-.15),ncol=2,fontsize=7)
                fig.suptitle(preset+" — first12observations; dashed fidelity fails numerical gates")
                fig.tight_layout()
                for ext in ["png","svg"]:fig.savefig(output/(preset+"."+ext),dpi=150,bbox_inches="tight")
                plt.close(fig)
            page+='<img src="'+preset+'.png" alt="Production fidelity and energy by cycle"><table><tr><th>Decoder</th><th>Cutoff</th><th>Trajectories</th><th>Cycles</th><th>Stage gates</th><th>Final fidelity</th></tr>'
            for _,row in group:
                page+=f'<tr><td>{html.escape(row["decoder"])}</td><td>{row["cutoff"]}</td><td>{row["total_trajectories"]}</td><td>{row["rounds"]}</td><td>{"pass" if row["numerical_edge_gate_pass"] else "failed; diagnostic"}</td><td>{100*row["finite_ancilla"]["mean"][-1]:.4f}%</td></tr>'
            page+="</table>"
        page+='<p><a href="../index.html">Suite progress and full reports</a> · <a href="observations.json">Observed metrics JSON</a></p><p>These observations do not certify scoring refinement, cutoff equivalence, or independent instrument coverage. Those checks are separate in the full suite report.</p>'
        (output/"index.html").write_text(page)
        (output/"observations.json").write_text(json.dumps({"status":state,"rows":[row for _,row in rows],
            "exporter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
        subprocess.run(["aws","s3","sync",str(output),f"s3://{args.bucket}/{args.run_id}/metrics/live/","--only-show-errors","--no-progress"],timeout=180,check=True)
        previous=signature
        if state["status"] in ("failed","interrupted","numerical_baseline_failed_or_inconclusive","all_scheduled_observations_completed"):break
        time.sleep(30)


if __name__=="__main__":main()
