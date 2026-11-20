"""Refresh the three-paper reproduction report from actual saved results.

--watch refreshes every30s while either registered training process is alive.
No training, simulation, model selection or test evaluation happens here.
"""
import argparse
import base64
import html
import io
import json
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).with_name('gkp-walkthrough')
RUNS={
 'convy':ROOT/'results/reproduce_convy2022/anchor/bfcba380fc8e8c55',
 'baireuther':ROOT/'results/reproduce_baireuther2018/figure3/aaacc7ab3e3a280d',
 'wan':ROOT/'results/reproduce_wan2020/components/91b306a688bce429',
}
COLORS=['#087e82','#7960bd','#ca634b']


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def figure(fig,name):
    folder=OUT/'reproduction-data';folder.mkdir(exist_ok=True)
    for ext in ['svg','png']:fig.savefig(folder/f'{name}.{ext}',bbox_inches='tight',dpi=160)
    content=(folder/f'{name}.svg').read_bytes();plt.close(fig)
    return '<img class="chart" alt="'+html.escape(name.replace('-',' '))+'" src="data:image/svg+xml;base64,'+base64.b64encode(content).decode()+'">'


def axis(ax,ylabel):
    ax.grid(axis='y',color='#e5edf2');ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False);ax.set_ylabel(ylabel)


def convy_assessment():
    """Apply the predeclared comparisons; never choose seeds or retrain."""
    run=RUNS['convy'];base=read(run/'bayesian_metrics.json')
    def compatible(metric,target):
        paper_sem=np.sqrt(target*(1-target)/30000)
        tolerance=1.96*np.sqrt(metric['sem']**2+paper_sem**2)+.0002
        return bool(abs(metric['final_fidelity']-target)<=tolerance)
    result={'bayesian_published_compatible':compatible(base,.8868),
            'interpretation':'Inconclusive: published Bayesian control mismatch.',
            'seed_comparisons':{}}
    for s in range(3):
        metric=read(run/f'lstm_s{s}_metrics.json')
        if metric is None:continue
        cache=OUT/'reproduction-data'/f'convy-seed{s}-comparison.json'
        comparison=read(cache)
        if comparison is None or comparison['predictions_sha256']!=metric['predictions_sha256']:
            with np.load(run/f'lstm_s{s}_predictions.npz') as f:pred=f['predictions'][:,-1]
            with np.load(run/'bayesian_predictions.npz') as f:control=f['predictions'][:,-1]
            labels=np.load(run/'test_labels.npy',mmap_mode='r')[:,-1]
            diff=(pred==labels).astype(float)-(control==labels).astype(float)
            mean=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
            comparison={'seed':s,'final_fidelity':metric['final_fidelity'],
                'published_lstm_compatible':compatible(metric,.8835667),
                'paired_lstm_minus_bayes':mean,'paired_normal95':[mean-1.96*se,mean+1.96*se],
                'n':len(diff),'predictions_sha256':metric['predictions_sha256'],
                'metric':'Final exact-state correctness, paired by trajectory.'}
            cache.parent.mkdir(exist_ok=True);cache.write_text(json.dumps(comparison,indent=2)+'\n')
        result['seed_comparisons'][str(s)]=comparison
    return result


def status():
    processes=subprocess.check_output(['ps','-axo','command'],text=True)
    states={}
    for key,module in [('convy','reproduce_convy2022'),('baireuther','reproduce_baireuther2018')]:
        active=any(f'-m experiments.{module} ' in x and '--smoke' not in x and not x.startswith('/bin/') for x in processes.splitlines())
        histories={str(s):read(RUNS[key]/f'lstm_s{s}_history.json',[]) for s in range(3)}
        finished=[s for s in range(3) if (RUNS[key]/f'lstm_s{s}.pt').exists()]
        complete=(all((RUNS[key]/f'lstm_s{s}_metrics.json').exists() for s in range(3)) if key=='convy'
                  else (RUNS[key]/'result.json').exists())
        states[key]={'active':active,'histories':histories,'finished_seeds':finished,
                     'training_complete':complete,'state':'complete' if complete else 'running' if active else 'interrupted',
                     'run':str(RUNS[key].relative_to(ROOT))}
    states['wan']={'components':read(RUNS['wan']/'result.json'),'full_figure3_reproduced':False,
                   'run':str(RUNS['wan'].relative_to(ROOT))}
    paths=sorted((ROOT/'results/reproduce_wan2020/fidelity_audit').glob('*/result.json'))
    states['wan']['fidelity_audit']=read(paths[-1]) if paths else None
    states['convy']['test_metrics']={name:read(RUNS['convy']/f'{name}_metrics.json')
        for name in ['bayesian','lstm_s0','lstm_s1','lstm_s2']}
    states['convy']['assessment']=convy_assessment()
    states['baireuther']['test_result']=read(RUNS['baireuther']/'result.json')
    return states


def seed_table(state,key):
    rows=[]
    for s in range(3):
        hist=state['histories'][str(s)];last=hist[-1] if hist else {}
        done=s in state['finished_seeds'];label='Finished' if done else 'Training' if hist and state['active'] else 'Queued' if state['active'] else 'Not running'
        metric=last.get('val_ce') if key=='convy' else last.get('validation_epsilon')
        rows.append(f'<tr><td>{s}</td><td>{label}</td><td>{last.get("epoch","—")}</td><td>{metric:.6f}</td></tr>' if metric is not None else f'<tr><td>{s}</td><td>{label}</td><td>—</td><td>—</td></tr>')
    metric='Validation cross entropy' if key=='convy' else 'Validation error / cycle'
    return '<table><thead><tr><th>Seed</th><th>Status</th><th>Epoch</th><th>'+metric+'</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'


def render():
    OUT.mkdir(exist_ok=True);states=status();stamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
    payload={'updated_utc':stamp,**states}
    folder=OUT/'reproduction-data';folder.mkdir(exist_ok=True)
    tmp=folder/'status.tmp';tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(folder/'status.json')
    parts=[]
    # Convy: final published target and actual model metrics only.
    run=RUNS['convy'];base=read(run/'bayesian_metrics.json')
    fig,ax=plt.subplots(figsize=(8,3.5));names=[];values=[];intervals=[]
    for name,file in [('Bayes','bayesian_metrics.json')]+[(f'LSTM seed{s}',f'lstm_s{s}_metrics.json') for s in range(3)]:
        metric=read(run/file)
        if metric:
            names.append(name);values.append(metric['final_fidelity']);intervals.append(metric['wilson95'])
    if values:
        vals=np.array(values);ints=np.array(intervals);ax.errorbar(np.arange(len(vals)),vals,yerr=np.stack([vals-ints[:,0],ints[:,1]-vals]),fmt='o',capsize=5,color=COLORS[0],label='Our test estimate · Wilson95')
    ax.axhline(.8868,color=COLORS[2],ls='--',label='Published Bayes ≈.8868')
    ax.axhline(.8835667,color=COLORS[1],ls=':',label='Published LSTM ≈.88357')
    ax.set_xticks(range(len(names)),names);ax.set_ylim(min(.78,min(values,default=.8)-.025),.94);axis(ax,'Final exact-state fidelity');ax.legend(fontsize=8,loc='lower left');fig.tight_layout()
    convyplot=figure(fig,'convy-final-comparison')
    parts.append(f'''<section id="convy"><div class="tag warning">Published control mismatch · LSTM {states['convy']['state']}</div><h2>Convy2022 · continuous three-qubit readout</h2>
    <p>Our Bayesian control scores <b>{base['final_fidelity']:.2%}</b> on30000 trajectories; the published target is88.68%. Independent execution of the authors' filter produces the same final predictions on our data. The remaining simulation or scoring discrepancy is unresolved.</p>
    <div class="split"><div>{convyplot}</div><div><h3>Three unchanged training seeds</h3>{seed_table(states['convy'],'convy')}<p>100000 training trajectories ·625 steps ·2LSTM layers×32units ·20–100epochs with validation stopping. CPU training. Original neural notebook unavailable.</p><p>These runs test whether an independently implemented LSTM approaches our verified Bayes control. They cannot establish agreement with the paper while that control mismatch remains.</p></div></div>
    <p class="links"><a href="https://doi.org/10.1088/1367-2630/ac66f9">Published paper</a> · <a href="../../../docs/convy2022_reproduction.md">Registered protocol</a> · <a href="../../../docs/convy2022_control_audit.md">Discrepancy audit</a></p></section>''')
    # Baireuther: validation progress does not masquerade as test performance.
    state=states['baireuther'];fig,ax=plt.subplots(figsize=(8,3.5))
    plotted=False
    for s in range(3):
        hist=state['histories'][str(s)]
        if hist:
            ax.plot([r['epoch'] for r in hist],[r['validation_epsilon'] for r in hist],color=COLORS[s],label=f'Seed{s}');plotted=True
    ax.set_yscale('log');ax.set_xlabel('Training epoch');axis(ax,'Fitted validation error / cycle')
    if plotted:ax.legend(fontsize=8)
    fig.tight_layout();plot=figure(fig,'baireuther-validation')
    result=read(RUNS['baireuther']/'result.json')
    if result:
        lo,hi=result['epsilon_bootstrap95'];target=.00209
        compat=lo-.000005<=target<=hi+.000005
        testnote=f"Held-out test: error/cycle={result['epsilon_per_cycle']:.6f}, bootstrap95[{lo:.6f},{hi:.6f}], selected seed{result['selected_seed']}. Published .00209 {'is' if compat else 'is not'} compatible under the registered criterion."
        fig,axs=plt.subplots(1,2,figsize=(10,3.5))
        t=np.array(result['cycles']);f=np.array(result['fidelities'])
        axs[0].plot(t,f,'o',ms=3,color=COLORS[0],label='Held-out fidelity')
        axs[0].plot(t,.5+.5*(1-2*result['epsilon_per_cycle'])**(t-result['t0']),color=COLORS[1],label='Fitted decay')
        axs[0].set_xlabel('Circuit cycles');axis(axs[0],'Final parity fidelity');axs[0].legend(fontsize=8)
        val=result['epsilon_per_cycle'];axs[1].errorbar([0],[val],yerr=[[max(0,val-lo)],[max(0,hi-val)]],fmt='o',capsize=5,color=COLORS[0],label='Trajectory bootstrap95')
        axs[1].axhline(.00209,color=COLORS[2],ls='--',label='Published .00209')
        axs[1].set_xticks([0],['Selected LSTM']);axis(axs[1],'Logical error / cycle');axs[1].legend(fontsize=8)
        fig.tight_layout();plot=figure(fig,'baireuther-test-comparison')+plot
    else:testnote='Held-out test has not been evaluated. The published target is.00209 logical errors/cycle; validation progress is not a reproduction result.'
    parts.append(f'''<section id="baireuther"><div class="tag">{state['state']} · {len(state['finished_seeds'])}/3seeds finished</div><h2>Baireuther2018 · Surface-17 circuit decoding</h2>
    <p>4million training sequences ·10000validation sequences ·50000held-out test sequences. The simulator matches the authors' circuit under identical injected faults. The PyTorch port also matches the source's recurrent equations and optimizer conventions.</p>
    <div class="split"><div>{plot}</div><div><h3>Training progress</h3>{seed_table(state,'baireuther')}<p>Two recurrent branches, each2layers×64units. Each epoch has10000updates. A seed finishes only after100epochs without improvement; improvements restart that counter. GPU training.</p><p class="callout">{testnote}</p></div></div>
    <p class="links"><a href="https://doi.org/10.22331/q-2018-01-29-48">Published paper</a> · <a href="../../../docs/baireuther2018_reproduction.md">Registered protocol</a> · <a href="../../../reproductions/baireuther2018/validation.json">Source-equivalence audit</a></p></section>''')
    # Wan completed component checks.
    wan=states['wan']['components'];audit=states['wan']['fidelity_audit'];wanplots=''
    if wan:
        mc=read(RUNS['wan']/'tracking_mc.json');rows=read(RUNS['wan']/'variance_curve.json')
        fig,axs=plt.subplots(1,2,figsize=(11,3.7));ax=axs[0]
        val=mc['probability'];lo,hi=mc['wilson95']
        ax.errorbar([0],[val*1e5],yerr=[[(val-lo)*1e5],[(hi-val)*1e5]],fmt='o',capsize=5,color=COLORS[0],label='20M samples · Wilson95')
        ax.axhline(wan['tracking_integral']*1e5,color=COLORS[1],label='Independent integral')
        ax.set_xticks([0],['Tracking failure']);axis(ax,'Probability ×10⁻⁵');ax.legend(fontsize=8)
        ax=axs[1];xs=[r['round'] for r in rows]
        for field,label,color in [('control_mse','No estimator',COLORS[2]),('bayesian_mse','Bayesian residual',COLORS[0]),('posterior_variance','Exact Gaussian variance',COLORS[1])]:
            ax.plot(xs,[r[field]*1e4 for r in rows],label=label,color=color,linewidth=1.3)
        ax.set_xlabel('Round');axis(ax,'Mean squared displacement ×10⁻⁴');ax.legend(fontsize=8);fig.tight_layout();wanplots=figure(fig,'wan-components')
        numbers=f"{mc['failures']}failures /{mc['n']:,}samples = {val:.3g};95%interval[{lo:.3g},{hi:.3g}]. Integral={wan['tracking_integral']:.8g}."
    else:numbers='Component results not available.'
    if audit:
        rows=audit['curves'];fig,ax=plt.subplots(figsize=(9,4))
        ax.plot([r['round'] for r in rows],[r['no_qec_plus'] for r in rows],label=r'Our uncorrected $|+\rangle$',color=COLORS[0])
        ax.plot([r['round'] for r in rows],[r['no_qec_worst']['fidelity'] for r in rows],label='Our minimum over all pure qubits',color=COLORS[1])
        refs=[r for r in audit['published_no_qec_digitized'] if r['fidelity'] is not None]
        ax.scatter([r['round'] for r in refs],[r['fidelity'] for r in refs],s=14,color='#39b3df',label='Published no-QEC curve · digitized')
        ax.set_xlabel('Rounds');axis(ax,'Logical-qubit fidelity');ax.legend(fontsize=8);fig.tight_layout();wanplots+=figure(fig,'wan-fidelity-scoring')
    parts.append(f'''<section id="wan"><div class="tag success">Numerical components confirmed · full Figure3 incomplete</div><h2>Wan2020 · finite-energy GKP memory</h2>
    <p>{numbers} The journal's approximate tracking-error scale is10⁻⁵. Its older preprint says10⁻¹¹; we use the published journal version. Matrix algebra, recursive filtering and200000sampled trajectories also agree on posterior variance.</p>{wanplots}
    <p class="callout">The finite-energy scoring audit reproduces initial fidelity≈.98165. At200rounds the uncorrected |+〉state gives≈.91163, matching the plotted curve, while the minimum over all pure qubits is≈.90194. The paper labels its axis F<sub>worst</sub>; the original scoring convention is unresolved. The full memoryless and memory-assisted curves have not been reproduced.</p>
    <p class="links"><a href="https://doi.org/10.1103/PhysRevResearch.2.043280">Published paper</a> · <a href="../../../docs/wan2020_reproduction.md">Components protocol</a> · <a href="../../../docs/wan2020_fidelity_audit.md">Fidelity audit protocol</a> · <a href="../../../reproductions/wan2020/validation.json">Independent validation</a></p></section>''')
    active=states['convy']['active'] or states['baireuther']['active']
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'''+('<meta http-equiv="refresh" content="60">' if active else '')+'''<title>Published decoder reproductions · QKP</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#172b42;font:15px/1.55 system-ui,sans-serif}header{background:#142a40;color:white;padding:20px max(24px,calc((100vw - 1280px)/2));display:flex;gap:26px;align-items:center}header a{color:#c8e4df;text-decoration:none}main{max-width:1320px;margin:auto;padding:32px 24px}h1{font:48px/1.1 Georgia,serif;margin:12px 0 20px}h2{font:29px/1.2 Georgia,serif;margin:12px 0}h3{font-size:16px}p{max-width:1100px;color:#526477}section{background:white;padding:26px;margin:25px 0;border:1px solid #dfe7ec;border-radius:12px;scroll-margin-top:20px}.split{display:grid;grid-template-columns:1.3fr 1fr;gap:22px;align-items:center}.chart{width:100%;height:auto}.tag{display:inline-block;background:#e9eef8;color:#3a5687;padding:5px 10px;border-radius:5px;font-size:12px}.warning{background:#fff0da;color:#885b20}.success{background:#e5f3ee;color:#126651}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;border-bottom:1px solid #e0e8ec;padding:9px 7px}.callout{border-left:3px solid #bc8631;padding:10px 14px;background:#fbf7ef}.links{font-size:13px}a{color:#087e82}.intro{font-size:17px}.stamp{font-size:12px;color:#617285}nav{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:850px){.split{grid-template-columns:1fr}h1{font-size:36px}section{padding:18px}header{flex-wrap:wrap}}
    </style><header><b>QKP / reproductions</b><a href="viewer.html">← GKP walkthrough</a><a href="reproduction-data/status.json" download>Download run data</a></header><main><h1>Published decoder results,<br>checked against matched experiments.</h1><p class="intro">Three separate physical models and protocols. Our earlier GKP noise sweep is not a reproduction of these papers. Completed checks, active training and discrepancies are shown explicitly.</p><nav><a href="#convy">Convy2022</a><a href="#baireuther">Baireuther2018</a><a href="#wan">Wan2020</a></nav>'''+f'<p class="stamp">Updated {stamp}. '+('Training is active; this page reloads every60seconds while the local report updater runs.' if active else 'No registered training process is currently active. Status is based on saved artifacts.')+'</p>'+''.join(parts)+'''<footer class="stamp">Uncertainty: Wilson95 for Bernoulli outcomes; trajectory bootstrap for the completed surface-code test. Published points are references, never substituted for missing measurements. SVG and PNG charts are saved in reproduction-data/.</footer></main></html>'''
    # Keep labels and units readable, without touching embedded chart data.
    for before,after in {
        'Convy2022':'Convy 2022','Baireuther2018':'Baireuther 2018','Wan2020':'Wan 2020',
        'Figure3':'Figure 3','on30000':'on 30,000','is88.68%':'is 88.68%',
        '100000 training':'100,000 training','625 steps':'625 steps',
        '2LSTM layers×32units':'2 LSTM layers × 32 units','20–100epochs':'20–100 epochs',
        '4million':'4 million','10000validation':'10,000 validation','50000held-out':'50,000 held-out',
        'each2layers×64units':'each 2 layers × 64 units','has10000updates':'has 10,000 updates',
        'after100epochs':'after 100 epochs','is.00209':'is .00209',
        'seeds finished':' seeds finished','/3seeds':'/3 seeds','failures /':' failures /',
        'samples =':' samples =',';95%interval':' ; 95% interval',
        'is10⁻⁵':'is 10⁻⁵','says10⁻¹¹':'says 10⁻¹¹','and200000sampled':'and 200,000 sampled',
        'At200rounds':'At 200 rounds','every60seconds':'every 60 seconds',
        'seed0':'seed 0','seed1':'seed 1','seed2':'seed 2',
        'Seed0':'Seed 0','Seed1':'Seed 1','Seed2':'Seed 2',
        ' on30000':' on 30,000',' /20,000,000':' / 20,000,000',
        'Wilson95':'Wilson 95%','bootstrap95':'bootstrap 95%',
        'fidelity≈':'fidelity ≈','gives≈':'gives ≈','qubits is≈':'qubits is ≈',
        '|+〉state':'|+〉 state',
    }.items():page=page.replace(before,after)
    tmp=OUT/'reproductions.tmp';tmp.write_text(page);tmp.replace(OUT/'reproductions.html')
    print(json.dumps({'updated_utc':stamp,'convy':states['convy']['state'],'baireuther':states['baireuther']['state'],'page':str(OUT/'reproductions.html')}),flush=True)
    return active


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        active=render()
        if not args.watch or not active:break
        time.sleep(30)
    if args.watch:
        from writeup.artifact.refresh_gkp_viewer import refresh
        refresh(package=True)


if __name__=='__main__':main()
