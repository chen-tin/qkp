"""Enforce the independent read-only audit's small separate reservation."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,signal,subprocess,time
folder=Path(__file__).resolve().parent
pid=32514
limit=1024**3
wall_limit=1200.
def process():
    result=subprocess.run(['ps','-o','etime=,rss=','-p',str(pid)],text=True,capture_output=True)
    values=result.stdout.split()
    if not values:return None
    fields=[int(x) for x in values[0].split(':')]
    elapsed=sum(value*60**index for index,value in enumerate(reversed(fields)))
    return elapsed,int(values[1])*1024
samples=[];reason=None
while True:
    observed=process()
    if observed is None:break
    elapsed,rss=observed;samples.append({'process_elapsed_seconds':elapsed,'rss_bytes':rss})
    if rss>limit:reason='hard_rss_cap'
    if elapsed>wall_limit:reason='total_wall_time_cap'
    if reason:
        os.kill(pid,signal.SIGINT)
        break
    time.sleep(1)
receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'pid':pid,'scope':'Read-only raw/CQ audit; no dynamics or trajectories','hard_rss_bytes':limit,'total_wall_seconds':wall_limit,'interruption_reason':reason,'max_sampled_rss_bytes':max((x['rss_bytes'] for x in samples),default=0),'samples':samples,'scientific_reservation_released':True}
(folder/'readback_resource_guard.json').write_text(json.dumps(receipt,indent=2)+'\n')
