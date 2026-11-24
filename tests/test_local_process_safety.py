import json
import os
from pathlib import Path
import signal
import sys
import time
import pytest
from experiments.local_process_safety import (active_group_members,spawn_owned,stop_owned,
                                              supervise_owned,validated_budget_history)


def history():
    return [{'number':i,'parent_pid':100+i,'terminal_status':{'elapsed_seconds':20.}} for i in (1,2)]


def test_restart_preserves_all_launches_and_measured_cleanup_time():
    records=history()+[{'number':3,'parent_pid':103,'terminal_status':{'status':'interrupted'},'parent_wall_seconds':3601.5}]
    saved=json.loads(json.dumps(records))
    assert validated_budget_history(saved)==3661.5
    assert len(saved)==3


@pytest.mark.parametrize('alter',[lambda x:x[0].update(number=2),lambda x:x.append({'number':3,'parent_pid':103}),
                                  lambda x:x[1].update(parent_pid=101),lambda x:x[1]['terminal_status'].update(elapsed_seconds=float('nan'))])
def test_uncertain_or_duplicate_history_refuses_restart(alter):
    records=history();alter(records)
    with pytest.raises(ValueError):validated_budget_history(records)


def test_group_cleanup_reaches_grandchild_after_supervisor_exits(tmp_path):
    marker=tmp_path/'grandchild'
    code=('import subprocess,sys; p=subprocess.Popen([sys.executable,"-c",'
          '"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"]);'
          f'open({str(marker)!r},"w").write(str(p.pid))')
    process=spawn_owned([sys.executable,'-c',code])
    deadline=time.monotonic()+5
    while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
    process.wait(timeout=5);assert marker.exists()
    # Wait until grandchild installs SIGTERM handler before triggering cleanup.
    time.sleep(.1)
    receipt=stop_owned(process,grace_seconds=.1)
    assert receipt['escalated_to_sigkill'] and not active_group_members(process.owned_pgid)


def test_external_deadline_covers_uncooperative_finalization():
    process=spawn_owned([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'])
    start=time.monotonic()
    result=supervise_owned(process,.4,started_monotonic=start,
                           grace_seconds=.1,poll_seconds=.02)
    assert result['external_stop_reason']=='external_parent_deadline'
    assert result['parent_wall_seconds']<2 and not active_group_members(process.owned_pgid)


def test_cleanup_refuses_unowned_group():
    class Unowned:pid=os.getpid()
    with pytest.raises(ValueError,match='unverified'):stop_owned(Unowned(),grace_seconds=.1)


def test_external_stop_never_becomes_retryable_from_later_clean_timeout_status():
    records=history()+[{'number':3,'parent_pid':103,'parent_wall_seconds':5.,
                       'external_stop_reason':'external_parent_deadline',
                       'terminal_status':{'status':'interrupted','error':"TimeoutError('Registered local deadline reached')"}}]
    with pytest.raises(ValueError,match='external stop'):validated_budget_history(records)
