"""Combine the reference run with independently rerun numerical refinements."""
import argparse
import html
import json
from pathlib import Path

import numpy as np

from experiments.analyze_qutip_reference import analyze,logical_refinement,numerical_comparison,LABELS
from experiments.qutip_reference import save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path);ap.add_argument('refined',type=Path)
    args=ap.parse_args();run,refined=args.run,args.refined
    analyze(run)
    logical=logical_refinement(refined)
    result=json.loads((run/'result.json').read_text());config=result['config']
    fine=json.loads((refined/'result.json').read_text());assert fine['complete']
    n=max(config['cutoffs']);audit=json.loads((run/'audit.json').read_text())
    comparisons=[]
    for row in result['rows']:
        if row['cutoff']!=n:continue
        point,name=row['point'],row['decoder'];key=point+'/'+name
        dx=numerical_comparison(run,refined,point,n,n,name,tolerance=.005)
        cutoff=audit['cutoff_comparisons'][key]
        pop=audit['population_gates'][key+f'/N{n}']
        logical_key=key+'/finite_ancilla'
        gate=bool(all(cutoff['within_tolerance']) and all(dx['within_tolerance'])
                  and logical[logical_key]['within_tolerance']
                  and pop['valid_population_horizon']>=config['rounds']
                  and row['max_norm_error']<1e-8)
        delta=row['reference_minus_classical'];lo,hi=delta['ci95_low'][-1],delta['ci95_high'][-1]
        numerical_budget=max(abs(cutoff['ci95_low'][-1]),abs(cutoff['ci95_high'][-1]))+max(abs(dx['ci95_low'][-1]),abs(dx['ci95_high'][-1]))+logical[logical_key]['max_absolute_difference']
        inflated=[lo-numerical_budget,hi+numerical_budget]
        if not gate:assessment='numerically_inconclusive'
        elif inflated[1]<-.01 or inflated[0]>.01:assessment='difference_exceeds_one_percentage_point'
        elif inflated[0]>=-.01 and inflated[1]<=.01:assessment='within_one_percentage_point'
        else:assessment='inconclusive_at_one_percentage_point'
        comparison={'point':point,'decoder':name,'cycles':config['rounds'],
            'classical':row['classical']['mean'][-1],
            'finite_energy_replayed':row['finite_energy_replayed']['mean'][-1],
            'reference':row['finite_ancilla']['mean'][-1],
            'reference_ci95':[row['finite_ancilla']['ci95_low'][-1],row['finite_ancilla']['ci95_high'][-1]],
            'difference':delta['mean'][-1],'paired_statistical_ci95':[lo,hi],
            'numerical_sensitivity_budget':numerical_budget,
            'sensitivity_expanded_interval':inflated,'all_numerical_gates_pass':gate,
            'cutoff_comparison':cutoff,'dx_comparison':dx,'logical_refinement':logical[logical_key],
            'assessment':assessment}
        comparisons.append(comparison)
    report={'scope':'Two-quadrature finite-energy GKP data + ideal finite-ancilla SUM/homodyne; no loss/dephasing/gate faults.',
            'run':str(run),'refinement_run':str(refined),'cycles':config['rounds'],
            'trajectories_per_input_state':config['per_state'],'input_states':6,
            'initial_reference_fidelity':result['rows'][0]['finite_ancilla']['mean'][0],
            'comparisons':comparisons,'source_sha256':sha(Path(__file__)),
            'interpretation':'Expanded intervals add empirical numerical-sensitivity bounds to sampling intervals; they are a conservative sensitivity summary, not a rigorous bound on the infinite-cutoff limit.'}
    save(run/'comparison_report.json',report)
    trs=[]
    for r in comparisons:
        pct=lambda x:f'{100*x:.2f}%'
        ci=r['paired_statistical_ci95']
        trs.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in [r['point'],LABELS[r['decoder']],pct(r['classical']),pct(r['reference']),
              f"{100*r['difference']:+.2f} pp [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]",'pass' if r['all_numerical_gates_pass'] else 'inconclusive'])+'</tr>')
    page='''<!doctype html><meta charset="utf-8"><title>QuTiP reference comparison</title>
<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#20273c}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}img{width:100%}code{background:#eef;padding:2px 4px}</style>
<h1>Finite-energy QuTiP reference</h1>'''
    page+=f'<p>{config["rounds"]} correction cycles; {6*config["per_state"]:,} trajectories per decoder/noise setting, equally split among six logical input states. Fock cutoffs '+', '.join(map(str,config['cutoffs']))+'.</p>'
    page+=f'<p>The finite-energy input starts at {100*report["initial_reference_fidelity"]:.3f}% average logical fidelity under modular-subsystem extraction; the classical model starts at100%. The replay control isolates encoding/readout effects; the finite-ancilla arm also includes conditional syndrome-extraction backaction.</p>'
    page+='<img src="comparison.png" alt="Logical fidelity curves and paired differences"><table><tr><th>Noise</th><th>Decoder</th><th>Classical</th><th>Finite ancilla</th><th>Difference, statistical95% CI</th><th>Numerical gates</th></tr>'+''.join(trs)+'</table>'
    page+='<p>These results validate a bounded ideal SUM/homodyne model. They do not reproduce a hardware experiment or establish50-cycle decoder rankings. The eight-cycle extension remains numerically unresolved at the tested sample sizes. Sampling intervals are pointwise. The JSON report includes separate cutoff, ancilla-bin and logical-quadrature checks and conservative sensitivity-expanded intervals.</p><p><a href="comparison_report.json">Full report JSON</a> · <a href="comparison.csv">Per-cycle CSV</a> · <a href="comparison.svg">SVG figure</a></p>'
    (run/'report.html').write_text(page)
    print(json.dumps([{k:r[k] for k in ['point','decoder','classical','reference','difference','all_numerical_gates_pass','assessment']} for r in comparisons],indent=2))


if __name__=='__main__':main()
