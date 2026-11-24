"""Resource-only reservation; actual memory remains separately measured."""
import subprocess
from pathlib import Path

RELEASE=Path(__file__).with_name('DENSITY_FINAL_RELEASED')
RESERVATION=4*1024**3

def density_reservation():
    rows={}
    for line in subprocess.run(['ps','-axo','pid=,ppid=,rss=,command='],capture_output=True,text=True,check=True).stdout.splitlines():
        fields=line.strip().split(None,3)
        if len(fields)==4:rows[int(fields[0])]={'ppid':int(fields[1]),'rss':int(fields[2])*1024,'command':fields[3]}
    roots={pid for pid,row in rows.items() if ' -m experiments.local_finite_gate_audit run ' in row['command']}
    tree=set(roots)
    while True:
        expanded=tree|{pid for pid,row in rows.items() if row['ppid'] in tree}
        if expanded==tree:break
        tree=expanded
    sampled=sum(rows[pid]['rss'] for pid in tree)
    released=RELEASE.exists()
    committed=0 if released else RESERVATION*max(1,len(roots))
    if released and roots:raise ValueError('Density release marker conflicts with active density process')
    readback_roots={pid for pid,row in rows.items() if 'readback_completed_density.py' in row['command'] and str(Path('/Users/christopherbabayan/programs/QUANTUM/qkp/results/simulator_accuracy_actions/20261124/finite_gate/v8')) in row['command']}
    readback_tree=set(readback_roots)
    while True:
        expanded=readback_tree|{pid for pid,row in rows.items() if row['ppid'] in readback_tree}
        if expanded==readback_tree:break
        readback_tree=expanded
    readback_rss=sum(rows[pid]['rss'] for pid in readback_tree)
    readback_commit=1024**3*len(readback_roots)
    return {'readback_root_pids':sorted(readback_roots),'readback_sampled_tree_rss_bytes':readback_rss,
            'readback_committed_bytes':readback_commit,'readback_remaining_growth_bytes':max(0,readback_commit-readback_rss),
            'density_root_pids':sorted(roots),'density_tree_pids':sorted(tree),'density_sampled_tree_rss_bytes':sampled,
            'density_committed_bytes':committed,'density_remaining_growth_bytes':max(0,committed-sampled),
            'density_final_release_marker':released,'scope':'Sampled process-tree RSS; full4GiB reservation persists between v7/v8.'}
