"""Local orchestration safety; this module does not select scientific settings."""
import math
import os
import signal
import subprocess
import time


def active_group_members(pgid):
    """Return live (non-zombie) members even after the original leader exits."""
    result=subprocess.run(['ps','-axo','pid=,pgid=,stat='],capture_output=True,text=True,check=True)
    return [int(fields[0]) for line in result.stdout.splitlines()
            if len(fields:=line.split())==3 and int(fields[1])==pgid and not fields[2].startswith('Z')]


def spawn_owned(argv,**kwargs):
    if kwargs.pop('start_new_session',True) is not True:
        raise ValueError('Owned supervisor requires its own session/process group')
    process=subprocess.Popen(argv,start_new_session=True,**kwargs)
    try:
        pgid=os.getpgid(process.pid)
        if pgid!=process.pid:raise RuntimeError('Supervisor process group ownership verification failed')
    except ProcessLookupError:
        # setsid occurs before Popen returns; an already-exited short command
        # cannot have inherited our group. Its assigned group ID remains pid.
        pgid=process.pid
    process.owned_pgid=pgid
    return process


def stop_owned(process,grace_seconds=50.,kill_wait_seconds=5.):
    """Stop the whole owned group, including an orphaned numerical grandchild."""
    pgid=getattr(process,'owned_pgid',None)
    if pgid!=process.pid or pgid==os.getpgrp():raise ValueError('Refuse cleanup of an unverified process group')
    def send(sig):
        try:os.killpg(pgid,sig)
        except ProcessLookupError:pass
    # Give the supervisor enough time to complete its nested30+10s cleanup.
    # Its exit alone does not discharge ownership of surviving descendants.
    if process.poll() is None:
        try:process.send_signal(signal.SIGINT)
        except ProcessLookupError:pass
    deadline=time.monotonic()+grace_seconds
    while active_group_members(pgid) and time.monotonic()<deadline:time.sleep(.02)
    escalated=bool(active_group_members(pgid))
    if escalated:send(signal.SIGKILL)
    deadline=time.monotonic()+kill_wait_seconds
    while active_group_members(pgid) and time.monotonic()<deadline:time.sleep(.02)
    remaining=active_group_members(pgid)
    if process.poll() is not None:process.wait()
    if remaining:raise RuntimeError(f'Owned process group still has live compute: {remaining}')
    return {'pgid':pgid,'escalated_to_sigkill':escalated,'live_group_members_after_cleanup':remaining}


def supervise_owned(process,maximum_work_seconds,started_monotonic=None,stop_requested=lambda:False,
                    on_poll=lambda:None,poll_seconds=.5,grace_seconds=50.):
    """Bound work/finalization time externally, then allow bounded group cleanup.

    Polls external to the scientific program, so analysis/finalization hangs are
    covered. Callers supply an even shorter cooperative scientific deadline.
    """
    if maximum_work_seconds<=0 or grace_seconds<0:
        raise ValueError('Positive work deadline and nonnegative graceful interval required')
    start=time.monotonic() if started_monotonic is None else started_monotonic
    reason=None;cleanup=None
    try:
        while process.poll() is None:
            if stop_requested():reason='stop_requested';break
            if time.monotonic()-start>=maximum_work_seconds:
                reason='external_parent_deadline';break
            on_poll();time.sleep(poll_seconds)
        # Descendants may survive an already-exited supervisor.
        if reason is not None or active_group_members(process.owned_pgid):
            cleanup=stop_owned(process,grace_seconds=grace_seconds)
        return {'returncode':process.wait(),'parent_wall_seconds':time.monotonic()-start,
                'external_stop_reason':reason,'group_cleanup':cleanup,
                'external_work_deadline_seconds':maximum_work_seconds,'graceful_cleanup_allowance_seconds':grace_seconds,'kill_verification_allowance_seconds':5.}
    except BaseException:
        stop_owned(process,grace_seconds=grace_seconds)
        raise


def validated_budget_history(records,maximum_launches=6,maximum_seconds=10800):
    """Refuse uncertain history rather than resetting a previous launch budget."""
    if not isinstance(records,list) or not 2<=len(records)<=maximum_launches:
        raise ValueError('Invalid durable parent-launch history')
    pids=[];charged=0.
    for number,record in enumerate(records,1):
        if record.get('number')!=number or not isinstance(record.get('parent_pid'),int):
            raise ValueError('Parent history must be sequential and identified')
        pids.append(record['parent_pid'])
        if record.get('external_stop_reason') is not None:
            raise ValueError('Prior external stop requires review; refuse automatic restart')
        if not isinstance(record.get('terminal_status'),dict):
            raise ValueError('Unsettled historical parent; refuse automatic restart')
        if number<=2:
            value=record['terminal_status'].get('elapsed_seconds');allowance=10.
        else:
            value=record.get('parent_wall_seconds');allowance=0.
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('Uncertain historical parent wall time; refuse restart')
        charged+=value+allowance
    if len(set(pids))!=len(pids) or charged>maximum_seconds:
        raise ValueError('Duplicate parent identity or exhausted cumulative budget')
    return charged
