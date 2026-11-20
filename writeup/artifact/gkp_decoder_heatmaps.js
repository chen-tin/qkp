/* Pure result helpers are also exported for numerical checks outside the browser. */
const GKPHeatmaps = (() => {
  'use strict';
  function wilson(k,n) {
    if(!Number.isInteger(k)||!Number.isInteger(n)||n<1||k<0||k>n)throw new Error('Invalid binomial counts');
    const z=1.959963984540054, p=k/n, den=1+z*z/n;
    const center=(p+z*z/(2*n))/den, half=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;
    return [Math.max(0,center-half),Math.min(1,center+half)];
  }
  function stats(record,n,t) {
    if(!record||record.status!=='complete'||t<1||t>record.counts.length)return null;
    const k=record.counts[t-1], [lo,hi]=wilson(k,n);
    return {k,n,p:k/n,lo,hi,delta:(record.worse_than_nl[t-1]-record.better_than_nl[t-1])/n};
  }
  const classify=(s,threshold)=>!s?'missing':s.lo>threshold?'above':s.hi<threshold?'below':'uncertain';
  const key=p=>['sigma_q','sigma_m','rho'].map(k=>k+p[k].toFixed(3)).join('_');
  function architectures(data) {
    const sweep=data.size_sweep;
    if(!sweep||sweep.evaluated!==sweep.expected_evaluations)return [];
    const cfg=data.config,cells=Object.values(data.cells),t=cfg.rounds-1,n=cfg.trials;
    const nominal=data.cells[key({sigma_q:.2,sigma_m:.05,rho:.9})];
    const mean=xs=>xs.reduce((a,b)=>a+b,0)/xs.length,rows=[];
    for(const hidden of sweep.config.widths)for(const layers of sweep.config.depths){
      const models=cfg.decoders.filter(m=>m.group==='size_sweep'&&m.hidden===hidden&&m.layers===layers);
      if(models.length!==sweep.config.seeds.length||new Set(models.map(m=>m.seed)).size!==models.length||models.some(m=>cells.some(c=>c.models[m.id]?.status!=='complete')))return [];
      const rates=models.map(m=>nominal.models[m.id].counts[t]/n);
      const grid=models.map(m=>mean(cells.map(c=>c.models[m.id].counts[t]/n)));
      rows.push({hidden,layers,parameters:models[0].n_params,seeds:models.length,
        nominal_mean:mean(rates),nominal_seed_min:Math.min(...rates),nominal_seed_max:Math.max(...rates),
        grid_mean:mean(grid),grid_seed_min:Math.min(...grid),grid_seed_max:Math.max(...grid)});
    }
    return rows;
  }
  return {wilson,stats,classify,key,architectures};
})();
if(typeof module!=='undefined'&&module.exports)module.exports=GKPHeatmaps;

if(typeof document!=='undefined') (()=>{
  'use strict';
  const $=id=>document.getElementById(id), node=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;};
  const source=$('decoder-heatmaps-data');
  let data;try{data=source?JSON.parse(source.textContent):null;}catch{data=null;}
  if(!data?.config?.decoders){$('heatmap-unavailable').hidden=false;$('heatmap-content').hidden=true;return;}
  const cfg=data.config, labels={sigma_q:'Displacement σq / L',sigma_m:'Readout noise σm / L',rho:'Correlation ρ'}, short={sigma_q:'σq',sigma_m:'σm',rho:'ρ'};
  let plane=cfg.planes[0],metric='error',filter=data.size_sweep?'sizes0':'representative',horizon=cfg.rounds,threshold=.10,selected={sigma_q:.20,sigma_m:.05,rho:.9},frame=0;
  const fmt=v=>v.toFixed(2),pct=v=>v===0?'0%':(100*v).toFixed(v<.01?2:1)+'%',pp=v=>(v>0?'+':'')+(100*v).toFixed(1)+' pp';
  const unit=(k,v)=>fmt(v)+(k==='rho'?'':'L');
  const value=(k,v)=>`${short[k]} = ${unit(k,v)}`;
  const cellButtons=new Map();
  const getStat=(pid,id)=>GKPHeatmaps.stats(data.cells[pid]?.models[id],cfg.trials,horizon);
  function visibleModels(){return cfg.decoders.filter(m=>filter==='all'||filter==='representative'&&m.primary||filter==='classical'&&m.kind!=='lstm'&&m.id!=='no_correction'||filter==='learned'&&m.kind==='lstm'||filter.startsWith('sizes')&&m.group==='size_sweep'&&(filter==='sizes'||m.seed===Number(filter.slice(5)))).sort((a,b)=>filter.startsWith('sizes')?a.layers-b.layers||a.hidden-b.hidden||a.seed-b.seed:0);}
  function cellParams(x,y){return {[plane.x]:x,[plane.y]:y,[plane.fixed]:plane.value};}
  function blend(a,b,t){return a.map((v,i)=>Math.round(v+(b[i]-v)*t));}
  function color(v) {
    let rgb;
    if(metric==='difference') {
      const w=Math.min(1,Math.abs(v)/.25);rgb=blend([247,246,242],v<0?[35,116,130]:[170,53,42],w);
    } else {
      const stops=[[237,246,242],[177,219,204],[243,215,144],[219,134,87],[157,44,39]],t=Math.min(1,Math.max(0,v/.5))*4,i=Math.min(3,Math.floor(t));rgb=blend(stops[i],stops[i+1],t-i);
    }
    return {bg:`rgb(${rgb.join(',')})`,ink:rgb[0]*.299+rgb[1]*.587+rgb[2]*.114<145?'#ffffff':'#213c46'};
  }
  function statusText(s){const c=GKPHeatmaps.classify(s,threshold);return c==='above'?'Above threshold':c==='below'?'Below threshold':c==='uncertain'?'Uncertain':'Not evaluated';}
  function describe(params){return ['sigma_q','sigma_m','rho'].map(k=>value(k,params[k])).join(' · ');}
  function buildCard(m) {
    const card=node('article','heatmap-card');card.append(node('h3','',m.name),node('p','model-detail',m.detail));
    card.append(node('p','heatmap-y-label','↑ '+labels[plane.y]));
    const table=node('table');table.setAttribute('aria-label',`${m.name}: ${labels[plane.x]} by ${labels[plane.y]}`);
    const caption=node('caption','sr-only',`${m.name}, logical error at round ${horizon}. ${describe(selected)} is selected.`);table.append(caption);
    const body=node('tbody');let above=0,uncertain=0,complete=0;
    for(const y of [...cfg.values[plane.y]].reverse()) {
      const row=node('tr'),head=node('th','',fmt(y));head.scope='row';row.append(head);
      for(const x of cfg.values[plane.x]) {
        const params=cellParams(x,y),pid=GKPHeatmaps.key(params),s=getStat(pid,m.id),td=node('td');
        const status=GKPHeatmaps.classify(s,threshold),b=node('button','heatmap-cell '+status,s?(metric==='error'?pct(s.p).replace('%',''):pp(s.delta).replace(' pp','')):'—');
        b.type='button';b.setAttribute('aria-pressed',String(pid===GKPHeatmaps.key(selected)));
        if(s){const c=color(metric==='error'?s.p:s.delta);b.style.background=c.bg;b.style.color=c.ink;complete++;if(status==='above')above++;if(status==='uncertain')uncertain++;}
        const desc=s?`${m.name}; ${describe(params)}; round ${horizon}: ${s.k} errors in ${s.n} trials, ${pct(s.p)}; 95% interval ${pct(s.lo)} to ${pct(s.hi)}; difference from nearest lattice ${pp(s.delta)}; ${statusText(s)}.`:`${m.name}; ${describe(params)}; not evaluated.`;
        b.setAttribute('aria-label',desc);b.title=desc;
        b.addEventListener('click',()=>{selected=params;render();cellButtons.get(m.id+'/'+pid)?.focus({preventScroll:true});});
        cellButtons.set(m.id+'/'+pid,b);td.append(b);row.append(td);
      }
      body.append(row);
    }
    const foot=node('tfoot'),tr=node('tr');tr.append(node('th'));
    for(const x of cfg.values[plane.x]){const th=node('th','',fmt(x));th.scope='col';tr.append(th);}foot.append(tr);table.append(body,foot);card.append(table,node('p','heatmap-x-label',labels[plane.x]+' →'));
    const summary=node('div','map-summary'),s=getStat(GKPHeatmaps.key(selected),m.id);
    summary.append(node('strong','',s?`Selected: ${pct(s.p)} · 95% interval ${pct(s.lo)}–${pct(s.hi)}`:m.training_status==='planned'?'Training pending; results will appear after evaluation.':'Selected setting has not been evaluated.'));
    summary.append(node('div','',complete?`${above}/${complete} cells confidently above ${pct(threshold)}; ${uncertain} uncertain.`:'No measured cells yet.'));card.append(summary);return card;
  }
  function details(models) {
    const box=$('heatmap-detail'),pid=GKPHeatmaps.key(selected);box.replaceChildren(node('h3','',`Selected setting · ${describe(selected)}`));
    box.append(node('p','',`Readout after round ${horizon}. Each error rate uses ${cfg.trials.toLocaleString()} paired trials. The 95% intervals describe Monte Carlo uncertainty for a fixed model; they do not include training-seed or grid-discretization uncertainty.`));
    const wrap=node('div','table-scroll'),table=node('table'),head=node('thead'),headrow=node('tr');
    for(const name of ['Decoder / checkpoint','Logical errors','Pₗ(t)','95% interval','Δ vs nearest lattice','Threshold status']){const th=node('th','',name);th.scope='col';headrow.append(th);}head.append(headrow);table.append(head);
    const body=node('tbody');
    for(const m of models){const s=getStat(pid,m.id),tr=node('tr');tr.append(node('th','',m.name));tr.firstChild.scope='row';
      for(const text of s?[`${s.k} / ${s.n}`,pct(s.p),`${pct(s.lo)}–${pct(s.hi)}`,pp(s.delta)]:['—','—','—','—'])tr.append(node('td','',text));
      tr.append(node('td',s?GKPHeatmaps.classify(s,threshold)==='above'?'failure':GKPHeatmaps.classify(s,threshold):'',statusText(s)));body.append(tr);}
    table.append(body);wrap.append(table);box.append(wrap);
    const zeros=models.some(m=>getStat(pid,m.id)?.k===0);
    if(zeros)box.append(node('p','',`Zero observed errors does not mean zero risk. With ${cfg.trials.toLocaleString()} trials, a zero-error cell has a 95% Wilson upper bound of ${pct(GKPHeatmaps.wilson(0,cfg.trials)[1])}.`));
  }
  function render() {
    const models=visibleModels();cellButtons.clear();
    $('heatmap-horizon-value').value=String(horizon);$('heatmap-threshold-value').value=pct(threshold);
    $('heatmap-horizon').setAttribute('aria-valuetext','Round '+horizon);$('heatmap-threshold').setAttribute('aria-valuetext',pct(threshold)+' logical error');
    $('heatmap-context').replaceChildren(document.createTextNode(`${labels[plane.x]} × ${labels[plane.y]} · held fixed: ${value(plane.fixed,plane.value)} · ${models.length} heatmaps. `));
    const link=node('a','',`Compare the selected setting ↓`);link.href='#heatmap-detail';$('heatmap-context').append(link);
    const planeKeys=cfg.values[plane.y].flatMap(y=>cfg.values[plane.x].map(x=>GKPHeatmaps.key(cellParams(x,y))));
    const available=models.reduce((n,m)=>n+planeKeys.filter(pid=>data.cells[pid]?.models[m.id]?.status==='complete').length,0);
    $('heatmap-sampling').textContent=`${available}/${models.length*planeKeys.length} measured model/settings in this view · ${cfg.trials.toLocaleString()} trials per setting · shared random draws across decoders and settings · ${cfg.rounds} rounds · exploratory development sweep. Border classifications always use logical error, even when colors show a difference.`;
    $('heatmap-scale-title').textContent=metric==='error'?`Logical error at round ${horizon} · % · lower is better`:'Difference from nearest lattice · percentage points';
    const stops=Array.from({length:21},(_,i)=>color(metric==='error'?i/40:(i/20-.5)*.5).bg);
    $('heatmap-colorbar').style.background='linear-gradient(to right,'+stops.join(',')+')';
    $('heatmap-scale-ticks').replaceChildren(...(metric==='error'?['0%','12.5%','25%','37.5%','≥50%']:['≤−25 pp','−12.5','0','+12.5','≥+25 pp']).map(t=>node('span','',t)));
    $('heatmap-cards').replaceChildren(...models.map(buildCard));details(models);
  }
  function save(name,text,type){const u=URL.createObjectURL(new Blob([text],{type})),a=node('a');a.href=u;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),3000);}
  $('heatmap-plane').addEventListener('change',()=>{plane=cfg.planes.find(p=>p.id===$('heatmap-plane').value);selected={sigma_q:.2,sigma_m:.05,rho:.9};render();});
  $('heatmap-metric').addEventListener('change',()=>{metric=$('heatmap-metric').value;render();});
  $('heatmap-models').addEventListener('change',()=>{filter=$('heatmap-models').value;render();});
  function scheduleRender(){if(frame)cancelAnimationFrame(frame);frame=requestAnimationFrame(()=>{frame=0;render();});}
  $('heatmap-horizon').max=String(cfg.rounds);$('heatmap-horizon').value=String(horizon);
  $('heatmap-horizon').addEventListener('input',()=>{horizon=Number($('heatmap-horizon').value);scheduleRender();});
  $('heatmap-threshold').addEventListener('input',()=>{threshold=Number($('heatmap-threshold').value)/100;scheduleRender();});
  $('heatmap-export').addEventListener('click',()=>{
    const rows=[['run_id','plane','decoder','sigma_q_L','sigma_m_L','rho','round','errors','trials','logical_error_probability','wilson95_lo','wilson95_hi','paired_difference_from_nl','threshold','threshold_status']];
    for(const m of visibleModels())for(const y of cfg.values[plane.y])for(const x of cfg.values[plane.x]){
      const p=cellParams(x,y),s=getStat(GKPHeatmaps.key(p),m.id);rows.push([data.run_id,plane.id,m.id,p.sigma_q,p.sigma_m,p.rho,horizon,s?.k??'',cfg.trials,s?.p??'',s?.lo??'',s?.hi??'',s?.delta??'',threshold,statusText(s)]);}
    const quote=v=>'"'+String(v).replaceAll('"','""')+'"';save(`gkp-decoder-heatmaps-${plane.id}-round${horizon}.csv`,rows.map(r=>r.map(quote).join(',')).join('\n'),'text/csv;charset=utf-8');
  });
  $('heatmap-json').addEventListener('click',()=>save('gkp-decoder-heatmaps-'+data.run_id+'.json',JSON.stringify(data,null,2),'application/json'));
  const methods=$('heatmap-methods-text');
  const sweep=data.size_sweep;
  const architectureRows=GKPHeatmaps.architectures(data);
  if(architectureRows.length){
    const box=$('benchmark-results');box.hidden=false;
    const nominalBest=architectureRows.reduce((a,b)=>a.nominal_mean<=b.nominal_mean?a:b);
    const gridBest=architectureRows.reduce((a,b)=>a.grid_mean<=b.grid_mean?a:b);
    const exactPct=v=>(v*100).toFixed(3)+'%';
    box.append(node('div','eyebrow','Completed benchmark · '+new Date(data.updated).toLocaleDateString(undefined,{year:'numeric',month:'long',day:'numeric'})),node('h3','','LSTM size results are ready'));
    box.append(node('p','benchmark-lead',`${sweep.models} trained models · ${sweep.evaluated.toLocaleString()} measured model/settings · ${architectureRows.length} architectures × ${sweep.config.seeds.length} seeds. Every model was evaluated on all ${data.expected_cells} stationary noise settings, with ${cfg.trials.toLocaleString()} shared trials per setting.`));
    const findings=node('div','benchmark-findings');
    for(const [best,field,label,context] of [[nominalBest,'nominal','Lowest observed nominal mean','σq/L = 0.20 · σm/L = 0.05 · ρ = 0.90'],[gridBest,'grid','Lowest observed noise-grid mean',`${data.expected_cells} settings, weighted equally`]]){
      const card=node('article');card.append(node('span','benchmark-label',label),node('strong','benchmark-value',exactPct(best[field+'_mean'])),node('p','',`${best.hidden} units × ${best.layers} layer${best.layers===1?'':'s'} · mean across ${best.seeds} seeds`),node('small','',`Seed range ${exactPct(best[field+'_seed_min'])}–${exactPct(best[field+'_seed_max'])}`),node('small','',context));findings.append(card);
    }
    box.append(findings,node('p','benchmark-note',`These summaries use logical error after round ${cfg.rounds}. The observed rankings are exploratory; seed ranges are not confidence intervals. The grid mean is not an expected error under a physical operating distribution. All models trained at nominal noise, and additional depth did not guarantee better performance.`));
    const comparison=node('details','benchmark-comparison');comparison.append(node('summary','',`Compare all ${architectureRows.length} architectures · means and seed ranges`));
    const wrap=node('div','table-scroll'),table=node('table'),head=node('thead'),hr=node('tr');
    for(const label of ['Width × depth','Parameters','Nominal Pₗ(50)','Mean across 61 settings']){const th=node('th','',label);th.scope='col';hr.append(th);}head.append(hr);table.append(head);
    const body=node('tbody');
    for(const row of architectureRows){const tr=node('tr'),name=node('th','',`${row.hidden} × ${row.layers}`);name.scope='row';tr.append(name,node('td','',row.parameters.toLocaleString()));for(const field of ['nominal','grid']){const td=node('td');td.append(node('strong','',exactPct(row[field+'_mean'])),node('small','',`Seed range ${exactPct(row[field+'_seed_min'])}–${exactPct(row[field+'_seed_max'])}`));tr.append(td);}body.append(tr);}table.append(body);wrap.append(table);comparison.append(wrap);box.append(comparison);
    const downloads=node('div','actions');
    const csvButton=node('button','button','Download architecture comparison · CSV ↓');csvButton.type='button';
    csvButton.addEventListener('click',()=>{const keys=Object.keys(architectureRows[0]);save('gkp-lstm-architecture-comparison.csv',[keys,...architectureRows.map(r=>keys.map(k=>r[k]))].map(row=>row.join(',')).join('\r\n')+'\r\n','text/csv;charset=utf-8');});
    const jsonButton=node('button','button','Download full benchmark · JSON ↓');jsonButton.type='button';jsonButton.addEventListener('click',()=>save('gkp-decoder-heatmaps-'+data.run_id+'.json',JSON.stringify(data,null,2),'application/json'));
    downloads.append(csvButton,jsonButton);box.append(downloads);
    const banner=$('benchmark-update-banner');if(banner){banner.hidden=false;banner.append(node('strong','',`Benchmark complete · ${sweep.models}/${sweep.models} LSTMs · ${sweep.evaluated.toLocaleString()} evaluations`));const jump=node('a','','View the completed results →');jump.href='#benchmark-results';banner.append(jump);}
  }
  for(const option of $('heatmap-models').options)if(option.value.startsWith('sizes'))option.hidden=!sweep;
  $('heatmap-models').value=filter;
  if(sweep){
    const box=$('heatmap-size-coverage');box.hidden=false;
    box.append(node('h3','',`LSTM size sweep · ${sweep.trained}/36 trained · ${sweep.evaluated.toLocaleString()}/2,196 evaluations`));
    box.append(node('p','',`4 widths × 3 depths × 3 seeds. All sizes share nominal-noise training, ${sweep.config.n_train.toLocaleString()} training sequences and ${sweep.config.n_val.toLocaleString()} validation sequences per seed, with up to ${sweep.config.max_epochs} epochs and validation-based early stopping. No model is retrained at individual heatmap cells. ${sweep.evaluated===sweep.expected_evaluations?'All planned size models have measured results.':'Reload this offline page to see newly completed models.'}`));
    const table=node('table'),head=node('tr');head.append(node('th','','Evaluated seeds'));
    for(const h of sweep.config.widths)head.append(node('th','',h+' units'));table.append(head);
    for(const depth of sweep.config.depths){const tr=node('tr');tr.append(node('th','',depth+' layer'+(depth===1?'':'s')));
      for(const width of sweep.config.widths){const models=cfg.decoders.filter(m=>m.group==='size_sweep'&&m.layers===depth&&m.hidden===width);const complete=models.filter(m=>Object.values(data.cells).every(c=>c.models[m.id]?.status==='complete')).length;tr.append(node('td',complete===3?'complete':'',complete+'/3'));}table.append(tr);}
    box.append(table);
    const training=node('details','heatmap-methods');
    training.append(node('summary','','How these LSTMs are trained · method and literature'));
    const steps=node('ol');
    for(const text of [
      `Simulate ${sweep.config.n_train.toLocaleString()} training and ${sweep.config.n_val.toLocaleString()} validation sequences per seed, each ${sweep.config.rounds} rounds long, at nominal noise: σq/L = 0.20, σm/L = 0.05, ρ = 0.90. Architectures within each seed share the same data and minibatch order.`,
      'Feed the causal LSTM the measured syndrome features s/L, sin(2πs/L) and cos(2πs/L). Its memory starts fresh for each sequence; it cannot see future syndromes or hidden simulator noise.',
      'Use simulator truth to supervise five parity outputs: windows of 1, 2, 3 and 5 rounds, plus the full history. Minimize binary cross-entropy averaged across sequences, rounds and heads by backpropagation through time.',
      `Optimize with AdamW, learning rate ${sweep.config.lr}, weight decay ${sweep.config.weight_decay}, batch size ${sweep.config.batch.toLocaleString()}, gradient clipping and a cosine schedule. Train for up to ${sweep.config.max_epochs} epochs, with patience ${sweep.config.patience}. Restore the best validation-loss checkpoint and choose its head by validation mean logical error.`,
      `Freeze the model and evaluate all ${data.expected_cells} noise settings on ${cfg.trials.toLocaleString()} shared development trials per setting. These heatmaps do not select its checkpoint or output head.`
    ])steps.append(node('li','',text));
    training.append(steps);
    const recurrent=node('p','','Literature precedent: Baireuther et al. (2018) train an LSTM surface-code decoder on syndrome histories and parity labels using cross-entropy, validation selection and separate evaluation. Their code and architecture differ from this GKP experiment. ');
    const recurrentLink=node('a','','Read Appendix A');recurrentLink.href='https://ar5iv.labs.arxiv.org/html/1705.07855';recurrentLink.target='_blank';recurrentLink.rel='noopener';recurrent.append(recurrentLink);training.append(recurrent);
    const memory=node('p','','GKP motivation: Wan, Neville and Kolthammer (2020) show benefits from combining syndrome information across rounds with Bayesian estimation. This supports using history, not this specific LSTM design. ');
    const memoryLink=node('a','','Read the paper');memoryLink.href='https://arxiv.org/abs/1912.00829';memoryLink.target='_blank';memoryLink.rel='noopener';memory.append(memoryLink);training.append(memory);
    training.append(node('p','','Interpretation: the features, five-head design and numerical training settings are project choices being tested. Off-nominal failures measure noise mismatch as well as capacity; they do not prove that an architecture could not learn that regime. A fixed epoch budget does not establish best attainable performance or convergence. This displacement/parity simulator is not a full finite-energy oscillator or hardware validation.'));
    box.append(training);
  }
  for(const text of [
    `Each cell is a stationary Gaussian AR(1) displacement process plus independent Gaussian readout noise. The three 5 × 5 planes intersect at shared settings, giving ${data.expected_cells} unique configurations. The third parameter is fixed at its nominal value; this is not a full three-dimensional sweep or a test of combined drift, steps and bursts.`,
    `Pₗ(t) is the fraction of trials whose residual is in the wrong logical parity after round t. It is not a flip count or a per-round error rate. Errors at different rounds can be correlated, and a decoder can revise an earlier parity estimate. The heatmaps therefore use the selected readout horizon directly.`,
    `Threshold borders use 95% Wilson intervals: solid when the entire interval is above the threshold, dashed when it crosses it. These are pointwise, exploratory intervals, not simultaneous guarantees over all cells. Cell colors always show measured estimates without interpolation. Colors saturate at 50% error or ±25 percentage points; the printed numbers retain the actual values.`,
    `LSTM checkpoints are evaluated separately with their validation-selected heads. The seed selectors expose training-seed variation; no seed is pooled into a larger binomial trial count. Planned models have no measured cells until training and evaluation finish. None is retrained at individual cells. Matched Bayes receives the true cell parameters; the other Bayesian models keep their stated assumptions fixed.`,
    `Bayesian filters use the repository’s reference implementation with ${cfg.n_x} grid points over 2L. The grid approximation is not included in the Monte Carlo intervals. Control and strong-noise cases were checked against unblocked closed-loop decoders before the sweep. This initial view covers analog-syndrome parity decoders; binary-readout, continuous-action and RL policies require separate evaluation.`,
    `Reproducibility: run ${data.run_id}, development split, seed root ${cfg.seed_root}, stream ${cfg.stream}. Paired raw logical-error records, configuration, source hashes, checkpoint hashes and validation checks are saved under ${data.artifact}${sweep?'; original decoder records remain under '+sweep.parent_artifact:''}. Results updated ${data.updated}. Additional values require a new measured sweep; moving the display controls does not run or fabricate new decoder evaluations.`
  ])methods.append(node('p','',text));
  render();
})();
