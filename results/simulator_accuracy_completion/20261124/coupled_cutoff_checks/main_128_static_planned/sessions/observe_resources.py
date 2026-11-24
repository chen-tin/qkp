"""Read-only postlaunch RSS samples; never labels a sampled maximum a true peak."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,time
F=Path(__file__).resolve().parent.parent
log=F/'sessions/resource_samples.jsonl';summary=F/'sessions/resource_samples_summary.json'
start=datetime.now(timezone.utc).isoformat();maximum=0;count=0
with log.open('x') as stream:
    while True:
        registry=json.loads((F/'sessions/registry.json').read_text());parent=registry.get('active_parent_pid')
        rows={}
        for line in subprocess.run(['ps','-axo','pid=,ppid=,rss=,stat='],capture_output=True,text=True,check=True).stdout.splitlines():
            fields=line.split()
            if len(fields)==4:rows[int(fields[0])]=(int(fields[1]),int(fields[2])*1024,fields[3])
        tree={parent} if parent in rows and not rows[parent][2].startswith('Z') else set()
        while True:
            more=tree|{pid for pid,(ppid,rss,state) in rows.items() if ppid in tree and not state.startswith('Z')}
            if more==tree:break
            tree=more
        rss=sum(rows[pid][1] for pid in tree);maximum=max(maximum,rss);count+=1
        event={'utc':datetime.now(timezone.utc).isoformat(),'parent_pid':parent,'tree_rss_bytes':rss,'members':{str(pid):rows[pid][1] for pid in sorted(tree)},'manager_status':registry['status']}
        stream.write(json.dumps(event)+'\n');stream.flush()
        value={'started_utc':start,'latest_utc':event['utc'],'samples':count,'largest_retained_tree_rss_sample_bytes':maximum,'scope':'Postlaunch5secondprocess-treeRSSsamples,notcertifiedwhole-sessionpeak. ScientificRSSwatchdogseparatelysamplesactivejobchildagainst8GiB.'}
        temp=summary.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(summary)
        if registry['status'] in ['all_fixed_study_jobs_processed','blocked_non_timeout_terminal','three_session_budget_exhausted','manager_error','external_deadline_or_stop_no_auto_retry','stop_requested']:break
        time.sleep(5)
