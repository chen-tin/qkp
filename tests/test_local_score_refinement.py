"""Shared-basis scoring must retain the original finite-operator convention."""
import numpy as np
import pytest
from quantum import scoring


def banks():
    rng=np.random.default_rng(14)
    states=[];labels=[]
    for count in [3,5,7]:
        x=rng.normal(size=(24,count))+1j*rng.normal(size=(24,count))
        x/=np.linalg.norm(x,axis=0)
        states.append(x);labels.append(np.arange(count)%6)
    return states,labels


@pytest.mark.parametrize('budget',[24*16*2,24*16*100])
def test_shared_basis_matches_individual_scoring_through_capacity_splits(budget):
    states,labels=banks()
    expected=[scoring.logical_fidelity(x,l,96,37) for x,l in zip(states,labels)]
    actual=scoring.logical_fidelity_many(states,labels,96,37,max_batch_bytes=budget)
    for a,b in zip(actual,expected):
        for key in a: np.testing.assert_allclose(a[key],b[key],atol=2e-13,rtol=2e-13)


def test_basis_is_evaluated_once_for_several_banks(monkeypatch):
    states,labels=banks();calls=[]
    old=scoring.hermite_functions
    def counted(x,n): calls.append(len(x));return old(x,n)
    monkeypatch.setattr(scoring,'hermite_functions',counted)
    scoring.logical_fidelity_many(states,labels,96,37)
    shared=len(calls);calls.clear()
    for x,l in zip(states,labels):scoring.logical_fidelity(x,l,96,37)
    assert len(calls)==3*shared


def test_batches_reject_cutoff_truncation_and_invalid_capacity():
    states,labels=banks()
    with pytest.raises(ValueError,match='full-cutoff'):scoring.logical_fidelity_many(states,labels,max_batch_bytes=1)
    with pytest.raises(ValueError,match='common full cutoff'):
        scoring.logical_fidelity_many([states[0],states[1][:-1]],[labels[0],labels[1]])


def saved_snapshots(path,seed):
    rng=np.random.default_rng(seed);raw={'fidelity':np.empty((12,3)),'snapshot_cycles':np.arange(3)}
    for cycle in range(3):
        psi=rng.normal(size=(24,12))+1j*rng.normal(size=(24,12));psi/=np.linalg.norm(psi,axis=0)
        raw[f'snapshot_cycle_{cycle}']=psi;raw[f'snapshot_indices_{cycle}']=np.arange(12)
        raw[f'snapshot_representative_{cycle}']=np.ones(12,dtype=bool)
        raw['fidelity'][:,cycle]=scoring.logical_fidelity(psi,np.repeat(np.arange(6),2),96)['fidelity']
    np.savez(path,**raw)


def test_all_file_cycle_refinement_metrics_match_original(tmp_path):
    from experiments.refine_logical_scores import refine_raw,refine_raw_many
    paths=[tmp_path/'a.npz',tmp_path/'b.npz']
    for i,path in enumerate(paths):saved_snapshots(path,41+i)
    expected=[refine_raw(path,2,bins=[96,192],chunk=37) for path in paths]
    actual=refine_raw_many(paths,2,bins=[96,192],chunk=37,max_batch_bytes=24*16*16)
    for a,b in zip(actual,expected):
        assert a['coverage']==b['coverage']=='all_cycles'
        assert a['all_cycle_snapshots_pass']==b['all_cycle_snapshots_pass']
        for ca,cb in zip(a['checks'],b['checks']):
            assert ca['pass']==cb['pass']
            for ra,rb in zip(ca['refinements'],cb['refinements']):
                for key in ['max_absolute_difference','trace_error','minimum_logical_eigenvalue']:
                    assert ra[key]==pytest.approx(rb[key],abs=2e-13)
                np.testing.assert_allclose(ra['difference']['mean'],rb['difference']['mean'],atol=2e-13)


def test_shared_refinement_preserves_negative_score_control_and_partial_scope(tmp_path):
    from experiments.refine_logical_scores import refine_raw_many
    path=tmp_path/'raw.npz';saved_snapshots(path,42)
    with np.load(path) as z: raw={k:z[k] for k in z.files}
    raw['fidelity'][:,1]-=.01;np.savez(path,**raw)
    check=refine_raw_many([path],2,bins=[96,192])[0]
    assert not check['checks'][1]['pass'] and not check['all_cycle_snapshots_pass']
    partial=refine_raw_many([path],2,bins=[96,192],cycles=[0,2])[0]
    assert partial['sample_checks_pass'] and not partial['all_cycle_snapshots_pass']


def test_external_analysis_output_never_changes_original_result_or_raw(tmp_path):
    import hashlib,json
    from experiments.refine_logical_scores import refine_run
    run=tmp_path/'input';case=run/'nominal_N24_smooth';case.mkdir(parents=True)
    hashes={}
    for arm in ['finite_ancilla','replayed_corrections']:
        path=case/f'nearest_{arm}.npz';saved_snapshots(path,42)
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    result={'complete':True,'config':{'per_state':2,'dx':None,'measurement_resource':'smooth'},
            'rows':[{'point':'nominal','cutoff':24,'decoder':'nearest','raw_hashes':hashes}]}
    (run/'result.json').write_text(json.dumps(result));before=hashlib.sha256((run/'result.json').read_bytes()).hexdigest()
    target,report=refine_run(run,bins=[96,192],shared_basis=True,output_directory=tmp_path/'analysis')
    assert target.parent==tmp_path/'analysis' and not (run/'quality').exists()
    assert report['all_cycles_checked'] and len(report['checks'])==2
    assert hashlib.sha256((run/'result.json').read_bytes()).hexdigest()==before
    assert all(hashlib.sha256((case/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())


@pytest.mark.parametrize('shared',[False,True])
@pytest.mark.parametrize('hot_delta',[.0005,.002])
def test_quadrature_deltas_are_deterministic_without_confidence_intervals(tmp_path,monkeypatch,shared,hot_delta):
    from experiments import refine_logical_scores as refinement
    from experiments.qutip_reference import estimate
    indices=np.r_[np.concatenate([np.arange(3*i,3*i+2) for i in range(6)]),2]
    representative=np.r_[np.ones(12,dtype=bool),False]
    baseline=np.full((18,1),.5)
    deltas=np.r_[np.linspace(-.0004,.0003,12),hot_delta]
    scored={'fidelity':baseline[indices,0]+deltas,'trace':np.ones(13),'minimum_eigenvalue':np.full(13,.1)}
    monkeypatch.setattr(refinement,'logical_fidelity',lambda *args,**kwargs:scored)
    monkeypatch.setattr(refinement,'logical_fidelity_many',lambda *args,**kwargs:[scored])
    path=tmp_path/'fixed_states.npz'
    np.savez(path,fidelity=baseline,snapshot_cycles=np.array([0]),
             snapshot_cycle_0=np.zeros((4,13),dtype=complex),snapshot_indices_0=indices,
             snapshot_representative_0=representative)
    report=(refinement.refine_raw_many([path],3,bins=[2,4])[0] if shared else
            refinement.refine_raw(path,3,bins=[2,4]))
    first,second=report['checks'][0]['refinements']
    for row in [first,second]:
        difference=row['difference']
        assert not {'se','ci95_low','ci95_high'} & difference.keys()
        assert difference['kind']=='deterministic_quadrature_delta'
        assert difference['representative_snapshot_count']==12
        assert difference['selected_hot_snapshot_count']==1
    actual_delta=scored['fidelity']-baseline[indices,0]
    # Retain the previous representative mean exactly; no statistical SE/CI.
    assert first['difference']['mean']==estimate(actual_delta[representative],2)['mean']
    assert first['difference']['range_including_hot']==[float(actual_delta.min()),float(actual_delta.max())]
    assert first['max_absolute_difference']==float(np.abs(actual_delta).max())
    assert first['pass']==(np.abs(actual_delta).max()<=.001)
    assert second['max_absolute_difference']==0 and second['pass']
    assert report['all_cycle_snapshots_pass']==first['pass']
