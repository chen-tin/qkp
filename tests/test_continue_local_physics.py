"""Directed local continuation planning/safety checks; no production simulation."""
import json
from pathlib import Path
import numpy as np
import pytest
from experiments import continue_local_physics as local
from experiments import quality_benchmark as qb

ORIGINAL=Path('results/ec2_runs/production_20261119/status_checks/20261123T222312Z/metrics_snapshot/physics/plan.json')


def test_migration_preserves_every_scientific_field_except_source_hashes():
    old=json.loads(ORIGINAL.read_text());plan=local.build_continuation(ORIGINAL)
    assert len(plan['jobs'])==4
    selected=local.source_jobs(old)
    for job in plan['jobs']:
        match=next(j for j in selected if (j['variant'],j['cutoff'])==(job['variant'],job['cutoff']))
        assert local.semantic_config(job['config'])==local.semantic_config(match['config'])
        assert job['id']!=match['id'] and job['base_seed']==26112001
    assert plan['metric_contract']==old['metric_contract']


def test_model_change_is_not_ignored_by_semantic_compare():
    a={'source_hashes':{'old':'x'},'physics':{'loss_rate':.001}}
    b={'source_hashes':{'new':'y'},'physics':{'loss_rate':.002}}
    assert local.semantic_config(a)!=local.semantic_config(b)
    b['physics']['loss_rate']=.001
    assert local.semantic_config(a)==local.semantic_config(b)


def test_draft_does_not_freeze_or_execute_and_freeze_needs_final_sources(tmp_path):
    out=tmp_path/'draft'
    result=local.prepare(ORIGINAL,out)
    assert result['status']=='draft_sources_not_frozen'
    assert not (out/'plan.json').exists() and not (out/'jobs').exists()
    with pytest.raises(ValueError,match='final'):
        local.prepare(ORIGINAL,out,freeze=True)


def test_tiny_negative_control_changes_only_declared_software_budget():
    plan=local.build_continuation(ORIGINAL,software_smoke=True)
    assert plan['preset']=='smoke' and plan['classification']=='infrastructure_smoke'
    assert len(plan['jobs'])==4
    for job in plan['jobs']:
        assert job['cutoff'] in [16,24] and job['config']['per_state']==2 and job['config']['rounds']==1
        assert job['config']['points']=={'nominal':[.2,.05,.9]}
    assert all(j['declared_software_budget_departures'] for j in plan['migration']['selected_original_jobs'])


def test_capacity_rejects_memory_and_disk_shortfall_without_compute(tmp_path):
    plan=local.build_continuation(ORIGINAL)
    with pytest.raises(ValueError,match='memory'):
        local.preflight(tmp_path,plan,2,1024,2*1024**3,free_memory=32*1024**3,free_disk=40*1024**3)
    with pytest.raises(ValueError,match='disk'):
        local.preflight(tmp_path,plan,2,16*1024**3,2*1024**3,free_memory=32*1024**3,free_disk=1024)
    result=local.preflight(tmp_path,plan,2,16*1024**3,2*1024**3,free_memory=32*1024**3,free_disk=40*1024**3)
    assert result['estimated_peak_bytes']<16*1024**3


def test_frozen_manifest_and_source_tampering_are_detected(tmp_path,monkeypatch):
    out=tmp_path/'new'
    local.prepare(ORIGINAL,out,freeze=True,sources_final=True)
    assert len(local.validate(out)['jobs'])==4
    manifest=json.loads((out/'local_manifest.json').read_text());manifest['remote_sync']=True
    (out/'local_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='local-only'):local.validate(out)


def test_partial_job_raw_is_checkpointed_before_job_result_exists(tmp_path):
    root=tmp_path/'jobs'/'job'/'nominal_N16_smooth';root.mkdir(parents=True)
    raw=root/'nearest_finite_ancilla.npz';np.savez(raw,states=np.eye(2))
    qb.save(raw.with_suffix('.sha256.json'),{'sha256':qb.sha(raw)})
    local.checkpoint_seals(tmp_path,{'id':'job'},tmp_path/'receipts')
    receipts=list((tmp_path/'receipts/job').glob('*.json'))
    assert len(receipts)==1 and json.loads(receipts[0].read_text())['sha256']==qb.sha(raw)
    assert not (root.parent/'result.json').exists()
    before=receipts[0].read_bytes()
    local.checkpoint_seals(tmp_path,{'id':'job'},tmp_path/'receipts')
    assert receipts[0].read_bytes()==before


def test_local_run_requires_no_sync_no_shutdown_and_bounded_runtime(tmp_path):
    with pytest.raises(ValueError,match='no-sync'):
        local.run(tmp_path,10,1,10**10,0,False,True)
    with pytest.raises(ValueError,match='max-seconds'):
        local.run(tmp_path,0,1,10**10,0,True,True)


def test_sigterm_becomes_catchable_interruption_and_original_handler_is_restored():
    import os,signal
    previous=signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt,match='supervisor'):
        with local.interrupt_signals():os.kill(os.getpid(),signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM)==previous
