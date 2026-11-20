/* Browser teaching model. Shifts are normalized by L; see simulator/nonstationary.py. */
const GKPNoise = (() => {
  'use strict';
  const L = Math.sqrt(Math.PI);
  const defaults = Object.freeze({family:'stationary',target:'sigma_q',sigma_q:.2,sigma_m:.05,rho:.9,
    amp:.3,period:200,phase:0,randomPhase:false,start:.16,end:.26,before:.2,after:.26,
    levelLo:.2,levelHi:.26,p_switch:.005,kick:.4,duration:1,tauLo:.25,tauHi:.75,
    rel:.25,rateMin:-4,rateMax:Math.log10(.3),n_ou:6,rounds:400});
  const clamp = (x,lo,hi) => Math.max(lo,Math.min(hi,x));
  function random(seed) {
    let a = seed >>> 0;
    return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a;
      t = Math.imul(t ^ t >>> 15,t | 1); t ^= t + Math.imul(t ^ t >>> 7,t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function normalSource(seed) {
    const uniform = random(seed); let spare = null;
    return () => { if (spare !== null) { const v = spare; spare = null; return v; }
      const r = Math.sqrt(-2*Math.log(Math.max(uniform(),1e-12))), angle = 2*Math.PI*uniform();
      spare = r*Math.sin(angle); return r*Math.cos(angle); };
  }
  const normals = (seed,n) => { const next=normalSource(seed); return Array.from({length:n},next); };
  function draws(seed,n) {
    return {z:normals(seed ^ 0x243f6a88,n),zm:normals(seed ^ 0x85a308d3,n),
      zk:normals(seed ^ 0x13198a2e,n),switches:Array.from({length:n},random(seed ^ 0x03707344)),
      event:random(seed ^ 0xa4093822)(),phase:random(seed ^ 0x299f31d0)()*2*Math.PI,
      initial:random(seed ^ 0x082efa98)() < .5 ? 0 : 1,
      ou:Array.from({length:12},(_,k)=>normals((seed ^ 0xec4e6c89)+k*7919,n))};
  }
  function schedule(p,r) {
    const n=p.rounds;
    const s={sigma_q:Array(n).fill(p.sigma_q),sigma_m:Array(n).fill(p.sigma_m),
      rho:Array(n).fill(p.rho),kick:Array(n).fill(0),tau:-1,clipped:0};
    const target=p.family==='one_over_f'?'sigma_q':p.target;
    const first=Math.max(1,Math.floor(p.tauLo*n));
    const last=Math.max(first,Math.min(n-1,Math.ceil(p.tauHi*n)-1));
    const event=first+Math.floor(r.event*(last-first+1));
    if (p.family==='sinusoid') {
      const phase=p.randomPhase?r.phase:p.phase*Math.PI/180;
      for(let t=0;t<n;t++) { const w=Math.sin(2*Math.PI*t/p.period+phase);
        s[target][t]=target==='rho'?p[target]+p.amp*w:p[target]*(1+p.amp*w); }
    } else if (p.family==='ramp') {
      s[target]=Array.from({length:n},(_,t)=>p.start+(p.end-p.start)*t/Math.max(1,n-1));
    } else if (p.family==='step') {
      s.tau=event; s[target]=Array.from({length:n},(_,t)=>t>=event?p.after:p.before);
    } else if (p.family==='telegraph') {
      let state=r.initial;
      for(let t=0;t<n;t++) { if(t && r.switches[t]<p.p_switch)state=1-state;
        s[target][t]=state?p.levelHi:p.levelLo; }
    } else if (p.family==='burst') {
      s.tau=event;
      for(let k=0;k<p.duration;k++)s.kick[Math.min(event+k,n-1)]=p.kick;
    } else if (p.family==='one_over_f') {
      const count=p.n_ou, states=Array.from({length:count},(_,k)=>r.ou[k][0]);
      const rates=states.map((_,k)=>10**(p.rateMin+(p.rateMax-p.rateMin)*k/Math.max(1,count-1)));
      for(let t=0;t<n;t++) {
        for(let k=0;k<count;k++)if(t) { const a=Math.exp(-rates[k]); states[k]=a*states[k]+Math.sqrt(1-a*a)*r.ou[k][t]; }
        const x=clamp(states.reduce((a,b)=>a+b,0)/Math.sqrt(count),-3,3);
        s.sigma_q[t]=p.sigma_q*Math.exp(p.rel*x-p.rel*p.rel/2);
      }
    }
    for(let t=0;t<n;t++) {
      if(Math.abs(s.rho[t])>.995 || s.sigma_q[t]<1e-6/L || s.sigma_m[t]<0)s.clipped++;
      s.rho[t]=clamp(s.rho[t],-.995,.995);s.sigma_q[t]=Math.max(1e-6/L,s.sigma_q[t]);s.sigma_m[t]=Math.max(0,s.sigma_m[t]);
    }
    return s;
  }
  const wrap = x => ((x+.5)%1+1)%1-.5;
  function simulate(p,r) {
    const s=schedule(p,r), d=[],nu=[],syndrome=[],residual=[],innovation=[];
    let eps=0,x=0;
    for(let t=0;t<p.rounds;t++) {
      const fresh=s.sigma_q[t]*Math.sqrt(1-s.rho[t]**2)*r.z[t];
      eps=t?s.rho[t]*s.sigma_q[t]/s.sigma_q[t-1]*eps+fresh:s.sigma_q[0]*r.z[0];
      d.push(eps+s.kick[t]*r.zk[t]);nu.push(s.sigma_m[t]*r.zm[t]);innovation.push(t?fresh:null);
      x+=d[t];const y=wrap(x+nu[t]);syndrome.push(y);x-=y;residual.push(x);
    }
    return {s,d,nu,syndrome,residual,innovation};
  }
  function coverage(p) {
    const missed=[];
    const check=(name,x,lo,hi)=>{if(x<lo-1e-9 || x>hi+1e-9)missed.push(`${name} ${x.toFixed(3)} outside ${lo}–${hi}`);};
    if(['ramp','one_over_f','telegraph','burst'].includes(p.family))return {heldOut:true,text:'Held-out shape: this schedule family is absent from the drift-LSTM training mixture.'};
    check('σm / L',p.sigma_m,.05,.05);
    if(p.family==='stationary') {check('σq / L',p.sigma_q,.14,.26);check('ρ',p.rho,.5,.97);}
    else if(p.target==='sigma_m')return {heldOut:true,text:'Held-out target: readout-noise schedules are absent from the drift-LSTM training mixture (σm / L is fixed at 0.05 there).'};
    else if(p.family==='sinusoid') {
      check('period',p.period,100,1000);
      if(p.target==='sigma_q') {check('baseline σq / L',p.sigma_q,.14,.24);check('amplitude',p.amp,.1,.4);check('ρ',p.rho,.5,.97);}
      else {check('baseline ρ',p.rho,.6,.85);check('amplitude',p.amp,.05,.12);check('σq / L',p.sigma_q,.14,.26);}
    } else if(p.family==='step') {
      if(p.target==='sigma_q') {check('before / L',p.before,.14,.22);check('after / L',p.after,.18,.28);check('ρ',p.rho,.5,.97);}
      else {check('before ρ',p.before,.6,.97);check('after ρ',p.after,.2,.6);check('σq / L',p.sigma_q,.14,.26);}
      check('event window start',p.tauLo,.25,.75);check('event window end',p.tauHi,.25,.75);
    }
    return {heldOut:false,text:missed.length?'Outside the training ranges: '+missed.join('; ')+'.':'These parameter values fall within the drift-LSTM training ranges. Training varies them across trials; this is one fixed setting.'};
  }
  return {defaults,draws,schedule,simulate,coverage,wrap};
})();
if(typeof module!=='undefined' && module.exports)module.exports=GKPNoise;

if(typeof document!=='undefined') (() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const node=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;};
  const labels={stationary:'Constant noise',sinusoid:'Sinusoidal drift',ramp:'Linear ramp',one_over_f:'1/f-like drift',step:'One sudden step',telegraph:'Repeated switching',burst:'Displacement burst'};
  const targets={sigma_q:'σq / L',sigma_m:'σm / L',rho:'ρ'};
  const modes={stationary:['stationary'],drift:['sinusoid','ramp','one_over_f'],sudden:['step','telegraph','burst'],unseen:['sinusoid','step','stationary','ramp','one_over_f','telegraph','burst']};
  let p={...GKPNoise.defaults},reference={...p},mode='stationary',seed=20261117,focus='sigma_q',pinned=false,view=null,raf=0;
  const controls=new Map(), colors={schedule:'#467fc3',d:'#087e82',nu:'#7960bd',syndrome:'#ca634b'};
  const fmt=(v,d=2)=>Number(v).toFixed(d).replace(/^-0\.00$/,'0.00');
  function range(parent,key,label,min,max,step,hint,format=v=>fmt(v),disabled=false) {
    const box=node('div','noise-control'),row=node('label','',label),output=node('output'),input=node('input');
    input.id='noise-'+key;input.type='range';input.min=min;input.max=max;input.step=step;input.value=p[key];input.disabled=disabled;
    row.htmlFor=input.id;output.htmlFor=input.id;output.id=input.id+'-value';row.append(output);
    const help=node('small','',hint);help.id=input.id+'-help';input.setAttribute('aria-describedby',help.id);
    box.append(row,input,help);parent.append(box);controls.set(key,{input,output,box,format});
    input.addEventListener('input',()=>{
      p[key]=Number(input.value);focus=key;
      if(key==='tauLo' && p.tauLo>p.tauHi)p.tauHi=p.tauLo;
      if(key==='tauHi' && p.tauHi<p.tauLo)p.tauLo=p.tauHi;
      if(key==='rateMin' && p.rateMin>p.rateMax)p.rateMax=p.rateMin;
      if(key==='rateMax' && p.rateMax<p.rateMin)p.rateMin=p.rateMax;
      updateOutputs();scheduleRender();
    });
  }
  function updateOutputs() {
    for(const [key,c] of controls) { c.input.value=p[key];c.output.value=c.format(p[key]);
      c.input.setAttribute('aria-valuetext',c.format(p[key]));c.box.classList.toggle('is-active',key===focus); }
  }
  function setTargetDefaults() {
    if(p.target==='rho')Object.assign(p,{start:.5,end:.95,before:.9,after:.3,levelLo:.3,levelHi:.97,amp:.1});
    else if(p.target==='sigma_m')Object.assign(p,{start:.02,end:.12,before:.05,after:.12,levelLo:.02,levelHi:.12,amp:.3});
    else Object.assign(p,{start:.16,end:.26,before:.2,after:.26,levelLo:.2,levelHi:.26,amp:.3});
  }
  function buildControls() {
    controls.clear();for(const id of ['noise-base-controls','noise-shape-controls','noise-run-controls'])$(id).replaceChildren();
    $('noise-family-row').hidden=mode==='stationary';$('noise-family').replaceChildren(...modes[mode].map(f=>{const o=node('option','',labels[f]);o.value=f;return o;}));$('noise-family').value=p.family;
    const varying=['sinusoid','ramp','step','telegraph'].includes(p.family);
    $('noise-target-row').hidden=!varying;$('noise-target').value=p.target;
    const base=$('noise-base-controls'),shape=$('noise-shape-controls');
    const replaces=['ramp','step','telegraph'].includes(p.family);
    range(base,'sigma_q','Displacement · σq',.01,.6,.01,replaces && p.target==='sigma_q'?'Set by the shape levels below.':'Larger values make the physical kicks wider.',v=>fmt(v)+' L',replaces && p.target==='sigma_q');
    range(base,'sigma_m','Readout error · σm',0,.3,.005,replaces && p.target==='sigma_m'?'Set by the shape levels below.':'Changes the measurement; physical kicks stay the same.',v=>fmt(v,3)+' L',replaces && p.target==='sigma_m');
    range(base,'rho','Correlation · ρ',-.99,.99,.01,replaces && p.target==='rho'?'Set by the shape levels below.':'Positive: persistent. Zero: independent. Negative: alternating.',v=>fmt(v),replaces && p.target==='rho');
    if(p.family!=='stationary')shape.append(node('h4','',labels[p.family]+' controls'));
    const isRho=p.target==='rho',min=isRho?-.99:p.target==='sigma_q'?.01:0,max=isRho?.99:p.target==='sigma_q'?.6:.3;
    const units=v=>fmt(v)+(isRho?'':' L');
    if(p.family==='sinusoid') {
      range(shape,'amp','Amplitude',0,isRho?.9:.95,.01,isRho?'An additive change in ρ; extreme values are clipped.':'Fraction of the baseline: 0.30 means ±30%.',v=>isRho?'±'+fmt(v):'±'+Math.round(v*100)+'%');
      range(shape,'period','Period',20,1000,10,'Fewer rounds per cycle means faster drift.',v=>Math.round(v)+' rounds');
      range(shape,'phase','Starting phase',0,360,5,'Moves the peaks and troughs along the time axis.',v=>Math.round(v)+'°',p.randomPhase);
      const lab=node('label','noise-check'),check=node('input');check.type='checkbox';check.checked=p.randomPhase;check.id='noise-random-phase';lab.append(check,document.createTextNode('Draw starting phase from the seed'));shape.append(lab);
      check.addEventListener('change',()=>{p.randomPhase=check.checked;focus='phase';buildControls();render();});
    }
    if(p.family==='ramp') {
      range(shape,'start','Starting level',min,max,.01,'The selected property at the first round.',units);
      range(shape,'end','Ending level',min,max,.01,'The selected property at the last round.',units);
    }
    if(p.family==='step') {
      range(shape,'before','Before the change',min,max,.01,'Noise level before the event.',units);
      range(shape,'after','After the change',min,max,.01,'Noise level from the event onward.',units);
    }
    if(p.family==='telegraph') {
      range(shape,'levelLo','Regime A',min,max,.01,'First of the two switching levels.',units);
      range(shape,'levelHi','Regime B',min,max,.01,'Second of the two switching levels.',units);
      range(shape,'p_switch','Switch probability',0,1,.001,'Chance of switching per round. Mean dwell ≈ 1 / p.',v=>fmt(v,3));
    }
    if(p.family==='burst') {
      range(shape,'kick','Burst strength',0,1.2,.02,'Standard deviation of extra kicks, outside the AR state.',v=>fmt(v)+' L');
      range(shape,'duration','Burst duration',1,100,1,'Independent extra kick in each affected round.',v=>Math.round(v)+(v===1?' round':' rounds'));
    }
    if(['step','burst'].includes(p.family)) {
      range(shape,'tauLo','Event window begins',0,.98,.01,'Earliest event location as a fraction of the run.',v=>Math.round(v*100)+'%');
      range(shape,'tauHi','Event window ends',0,.99,.01,'Latest event window edge. A seeded draw selects the event.',v=>Math.round(v*100)+'%');
    }
    if(p.family==='one_over_f') {
      range(shape,'rel','Fluctuation strength',0,.6,.01,'Controls the log-scale variation of σq.',v=>fmt(v));
      range(shape,'rateMin','Slowest rate',-5,0,'any','Logarithmic slider. Slower components produce longer-lived variations.',v=>(10**v).toPrecision(2)+'/round');
      range(shape,'rateMax','Fastest rate',-5,0,'any','Logarithmic slider. Faster components add finer time structure.',v=>(10**v).toPrecision(2)+'/round');
      range(shape,'n_ou','Component processes',1,12,1,'More log-spaced rates give a denser range of time scales.',v=>Math.round(v));
    }
    range($('noise-run-controls'),'rounds','Displayed rounds',100,1000,100,'Also sets the duration of ramps and event windows.',v=>Math.round(v));
    updateOutputs();
  }
  const presets={
    stationary:[['Larger physical kicks',{sigma_q:.35},'sigma_q'],['Noisy measurement',{sigma_m:.15},'sigma_m'],['No memory',{rho:0},'rho'],['Alternating kicks',{rho:-.8},'rho']],
    drift:[['Slow σq drift',{family:'sinusoid',target:'sigma_q',amp:.3,period:1000},'period'],['Fast σq drift',{family:'sinusoid',target:'sigma_q',amp:.3,period:100},'period'],['Drifting correlation',{family:'sinusoid',target:'rho',rho:.75,amp:.1,period:200},'amp'],['Drifting readout',{family:'sinusoid',target:'sigma_m',sigma_m:.1,amp:.6,period:200},'amp']],
    sudden:[['Stronger noise step',{family:'step',target:'sigma_q',before:.2,after:.34},'after'],['Memory switches',{family:'telegraph',target:'rho',levelLo:.3,levelHi:.97,p_switch:.02},'p_switch'],['One-round burst',{family:'burst',kick:.6,duration:1},'kick'],['Longer burst',{family:'burst',kick:.4,duration:20},'duration']],
    unseen:[['Amplitude 0.6',{family:'sinusoid',target:'sigma_q',amp:.6,period:1000},'amp'],['Period 50',{family:'sinusoid',target:'sigma_q',amp:.3,period:50},'period'],['Step to 0.34L',{family:'step',target:'sigma_q',before:.2,after:.34},'after'],['Step to ρ = 0',{family:'step',target:'rho',before:.9,after:0},'after'],['Constant σq = 0.32L',{family:'stationary',sigma_q:.32},'sigma_q']]
  };
  function buildPresets() {
    $('noise-presets').replaceChildren(...presets[mode].map(([label,values,key])=>{const b=node('button','',label);b.type='button';b.addEventListener('click',()=>{const rounds=p.rounds;p={...GKPNoise.defaults,rounds,...values};focus=key;buildControls();render();});return b;}));
  }
  function setMode(next) {
    mode=next;p={...GKPNoise.defaults,family:modes[mode][0]};focus='sigma_q';
    if(mode==='unseen'){p.amp=.6;p.period=1000;focus='amp';}
    for(const b of document.querySelectorAll('[data-noise-mode]'))b.setAttribute('aria-pressed',String(b.dataset.noiseMode===mode));
    buildControls();buildPresets();render();
  }
  function explanation() {
    const n=p.rounds,unit=targets[p.target];
    const shared={sigma_q:`σq sets the width of the physical kicks. At the current baseline of ${fmt(p.sigma_q)}L, the shaded band shows ±1 standard deviation. The same underlying random draws make the change in kick size visible.`,
      sigma_m:`σm changes the readout error without changing the physical kicks. A noisier readout changes the wrapped syndrome and therefore the nearest-lattice correction. The readout errors remain independent between rounds.`,
      rho:`ρ controls memory. Positive values keep nearby kicks pointing in similar directions; negative values favor alternation. Changing ρ rescales the fresh innovations so the intended displacement width stays fixed.`};
    let text=shared[focus];
    if(!text) {
      const descriptions={stationary:'The noise settings are constant through time. Try σq for larger kicks, σm for a less reliable measurement, and ρ for longer memory.',
        sinusoid:`${unit} follows a sine wave. Amplitude sets the excursion, period sets the speed, and phase sets where the wave starts. ${p.target==='rho'?'Amplitude is added to ρ.':'Amplitude is a fraction of the baseline.'}`,
        ramp:`${unit} moves from ${fmt(p.start)} to ${fmt(p.end)} across ${n} rounds. Changing an endpoint changes the slope; a longer run spreads the same change over more rounds.`,
        one_over_f:'σq wanders as a combination of slower and faster random processes. Fluctuation strength sets its variation; the rate bounds set the time scales; the component count sets how densely those time scales are sampled.',
        step:`${unit} jumps from ${fmt(p.before)} to ${fmt(p.after)} at the marked round. The two event-window sliders control where that round can fall. No reset is applied to the correlated noise state.`,
        telegraph:`${unit} switches between ${fmt(p.levelLo)} and ${fmt(p.levelHi)}. ${p.p_switch>0?'The expected stay in a regime is about '+fmt(1/p.p_switch,1)+' rounds.':'With zero switch probability it stays in the starting regime.'} The first regime is chosen from the seed.`,
        burst:`Extra displacement kicks with width ${fmt(p.kick)}L are added for ${p.duration} round${p.duration===1?'':'s'}. They do not feed back into the AR noise state, although their effect on the oscillator can remain after the burst.`};
      text=descriptions[p.family];
    }
    if(p.family!=='stationary' && shared[focus])text+=' The selected '+labels[p.family].toLowerCase()+' schedule is also applied.';
    if(focus==='phase')text=p.randomPhase?'The starting phase is drawn from the seed. Draw another path to change it; its amplitude and period stay fixed.':'Starting phase slides the sinusoidal pattern in time. The independent noise draws stay fixed, so the changed envelope acts on the same random path.';
    if(focus==='rounds')text='The time window changes how much of the path you see. Sinusoidal periods stay in rounds; ramps stretch across the window, and event times follow the selected fractions of its length.';
    return text;
  }
  const ns='http://www.w3.org/2000/svg';
  function svgNode(tag,attrs,text) {const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
  function plot(id,current,other,{color,ymin,ymax,band=null,event=-1,label='',guide=false,step=false,refEvent=-1}) {
    const svg=$(id),W=460,H=210,left=49,right=13,top=12,bottom=34,n=current.length;
    const X=t=>left+(W-left-right)*t/Math.max(1,n-1),Y=v=>top+(H-top-bottom)*(ymax-v)/(ymax-ymin);
    svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.replaceChildren(svgNode('title',{},label),svgNode('desc',{},'Solid colored line: current settings. Gray dashed line: reference. Both use the same underlying random draws.'));
    const clipId=id+'-clip',defs=svgNode('defs',{}),clip=svgNode('clipPath',{id:clipId});clip.append(svgNode('rect',{x:left,y:top,width:W-left-right,height:H-top-bottom}));defs.append(clip);svg.append(defs);
    for(let i=0;i<=4;i++){const v=ymin+(ymax-ymin)*i/4;svg.append(svgNode('line',{x1:left,x2:W-right,y1:Y(v),y2:Y(v),stroke:'#e5edf2'}),svgNode('text',{x:left-7,y:Y(v)+4,'text-anchor':'end'},Math.abs(v)<1e-9?'0':fmt(v,Math.abs(ymax-ymin)<.1?3:2)));}
    for(let i=0;i<=4;i++){const t=Math.round((n-1)*i/4);svg.append(svgNode('text',{x:X(t),y:H-17,'text-anchor':'middle'},t+1));}
    svg.append(svgNode('text',{x:(left+W-right)/2,y:H-2,'text-anchor':'middle'},'Round'));
    const traces=svgNode('g',{'clip-path':`url(#${clipId})`});
    if(band){const pts=band.map((v,t)=>`${X(t)},${Y(v)}`).concat([...band].reverse().map((v,j)=>`${X(n-1-j)},${Y(-v)}`));traces.append(svgNode('polygon',{points:pts.join(' '),fill:color,opacity:.09}));}
    if(guide)for(const v of [-.5,.5])if(v>=ymin && v<=ymax)traces.append(svgNode('line',{x1:left,x2:W-right,y1:Y(v),y2:Y(v),stroke:'#b58c78','stroke-dasharray':'2 4',opacity:.7}));
    if(ymin<0 && ymax>0)traces.append(svgNode('line',{x1:left,x2:W-right,y1:Y(0),y2:Y(0),stroke:'#c1cdd6'}));
    for(const [t,c,dash] of [[refEvent,'#93a2b2','2 5'],[event,color,'4 4']])if(t>=0)traces.append(svgNode('line',{x1:X(t),x2:X(t),y1:top,y2:H-bottom,stroke:c,'stroke-dasharray':dash,opacity:.8}));
    const path=arr=>arr.map((v,t)=>(t?(step?`H${X(t).toFixed(2)}V${Y(v).toFixed(2)}`:`L${X(t).toFixed(2)},${Y(v).toFixed(2)}`):`M${X(t).toFixed(2)},${Y(v).toFixed(2)}`)).join(' ');
    traces.append(svgNode('path',{d:path(other),fill:'none',stroke:'#9ba9b8','stroke-width':1.5,'stroke-dasharray':'5 4',opacity:.9}),svgNode('path',{d:path(current),fill:'none',stroke:color,'stroke-width':1.7,'stroke-linejoin':'round'}));svg.append(traces);
    if(event>=0)svg.append(svgNode('text',{x:Math.max(left+4,Math.min(X(event)+5,W-100)),y:top+11,fill:color},'Event '+(event+1)));
  }
  const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
  const maxAbs=a=>a.reduce((m,x)=>Math.max(m,Math.abs(x)),0);
  function render() {
    const r=GKPNoise.draws(seed,p.rounds),a=GKPNoise.simulate(p,r),b=GKPNoise.simulate({...reference,rounds:p.rounds},r);
    view={a,b};$('noise-explanation').textContent=explanation();$('noise-seed').textContent=`Path seed ${seed} · ${p.rounds} rounds`;
    $('noise-coverage').hidden=mode!=='unseen';$('noise-coverage').textContent=GKPNoise.coverage(p).text;
    $('noise-reference-label').textContent=pinned?'Reference: pinned settings':'Reference: nominal';
    let target=p.family==='burst'?'kick':p.family==='one_over_f'?'sigma_q':p.family==='stationary' && ['sigma_q','sigma_m','rho'].includes(focus)?focus:p.target;
    const unit=target==='kick'?'burst width / L':targets[target];
    $('noise-schedule-caption').textContent=unit+' · '+labels[p.family];
    let lo=Math.min(0,...a.s[target],...b.s[target]),hi=Math.max(.01,...a.s[target],...b.s[target]);
    if(target==='rho'){lo=-1;hi=1;}else hi=Math.max(hi*1.15,.05);
    const event=a.s.tau,refEvent=b.s.tau;
    plot('noise-schedule-chart',a.s[target],b.s[target],{color:colors.schedule,ymin:lo,ymax:hi,event,refEvent,label:unit+' schedule',step:['step','telegraph','burst'].includes(p.family)});
    const dq=a.s.sigma_q.map((q,t)=>Math.hypot(q,a.s.kick[t]));
    // A fixed minimum span preserves perceived width; expand together only to avoid clipping.
    const ds=Math.max(.8,Math.ceil(Math.max(maxAbs(a.d),maxAbs(b.d),maxAbs(dq))*10)/10),ms=Math.max(.5,Math.ceil(Math.max(maxAbs(a.nu),maxAbs(b.nu))*10)/10);
    plot('noise-displacement-chart',a.d,b.d,{color:colors.d,ymin:-ds,ymax:ds,band:dq,event,refEvent,guide:true,label:'Physical displacement in units of L'});
    plot('noise-readout-chart',a.nu,b.nu,{color:colors.nu,ymin:-ms,ymax:ms,band:a.s.sigma_m,event,refEvent,guide:true,label:'Readout error in units of L'});
    plot('noise-syndrome-chart',a.syndrome,b.syndrome,{color:colors.syndrome,ymin:-.55,ymax:.55,event,refEvent,guide:true,label:'Observed syndrome in units of L, with nearest-lattice correction each round'});
    const metrics=[['RMS kick width',Math.sqrt(mean(dq.map(x=>x*x))),Math.sqrt(mean(b.s.sigma_q.map((q,t)=>q*q+b.s.kick[t]**2))),' L'],['RMS readout width',Math.sqrt(mean(a.s.sigma_m.map(x=>x*x))),Math.sqrt(mean(b.s.sigma_m.map(x=>x*x))),' L'],['Mean AR correlation',mean(a.s.rho.slice(1)),mean(b.s.rho.slice(1)),'']];
    $('noise-metrics').replaceChildren(...metrics.map(([name,v,ref,suffix])=>{const e=node('div','noise-metric');e.append(node('span','',name),node('strong','',fmt(v,3)+suffix),node('small','',`Reference ${fmt(ref,3)}${suffix}`));return e;}));
    $('noise-scale-note').textContent=`Both traces share each panel’s scale. Displacement: ±${fmt(ds,1)}L; readout: ±${fmt(ms,1)}L; syndrome always lies within ±½L. Shading: ±1σ of your noise. `+(a.s.clipped?`${a.s.clipped} schedule rounds reached the simulator’s parameter bounds. `:'')+(p.family==='burst' && event+p.duration>p.rounds?'The burst is cut short by the end of the displayed run. ':'')+'One trajectory illustrates behavior; it is not a decoder accuracy estimate.';
  }
  function scheduleRender(){if(raf)cancelAnimationFrame(raf);raf=requestAnimationFrame(()=>{raf=0;render();});}
  for(const b of document.querySelectorAll('[data-noise-mode]'))b.addEventListener('click',()=>setMode(b.dataset.noiseMode));
  $('noise-family').addEventListener('change',()=>{p.family=$('noise-family').value;if(p.family==='one_over_f')p.target='sigma_q';focus='shape';buildControls();render();});
  $('noise-target').addEventListener('change',()=>{p.target=$('noise-target').value;setTargetDefaults();focus='shape';buildControls();render();});
  $('noise-reset').addEventListener('click',()=>{reference={...GKPNoise.defaults};pinned=false;seed=20261117;setMode(mode);});
  $('noise-new-path').addEventListener('click',()=>{seed=(seed+1)>>>0;render();});
  $('noise-pin').addEventListener('click',()=>{reference={...p};pinned=true;render();});
  $('noise-nominal').addEventListener('click',()=>{reference={...GKPNoise.defaults};pinned=false;render();});
  $('noise-download').addEventListener('click',()=>{
    const {a,b}=view,fields=['sigma_q','sigma_m','rho','kick'];
    const rows=[['round','seed',...fields.map(k=>'current_'+k+(k==='rho'?'':'_L')),...fields.map(k=>'reference_'+k+(k==='rho'?'':'_L')),'current_d_L','reference_d_L','current_nu_L','reference_nu_L','current_s_L','reference_s_L','current_residual_L','reference_residual_L']];
    for(let t=0;t<p.rounds;t++)rows.push([t+1,seed,...fields.map(k=>a.s[k][t]),...fields.map(k=>b.s[k][t]),a.d[t],b.d[t],a.nu[t],b.nu[t],a.syndrome[t],b.syndrome[t],a.residual[t],b.residual[t]]);
    const url=URL.createObjectURL(new Blob([rows.map(row=>row.join(',')).join('\n')],{type:'text/csv;charset=utf-8'}));
    const link=node('a');link.href=url;link.download='gkp-noise-'+p.family+'-'+seed+'.csv';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),3000);
  });
  setMode('stationary');
})();
