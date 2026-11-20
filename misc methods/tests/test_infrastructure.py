import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'_shared'))
from metrics import compare, per_round, wilson
from runner import papers, score, validate, write, digest
from analyze import decay_fit


class Contracts(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'requires optional analysis dependencies')
    def test_decay_fit_recovers_boundary_contrast_and_error_rate(self):
        rows=[]
        for r in [1,3,7,13,25]:
            p=(1-.93*(1-2*.017)**r)/2
            rows.append({'rounds':r,'errors':round(p*1000000),'shots':1000000})
        fit=decay_fit(rows)
        self.assertAlmostEqual(fit['value'],.017,places=5)
        self.assertAlmostEqual(fit['amplitude'],.93,places=4)

    def test_all_papers_have_sourced_quantitative_targets(self):
        self.assertEqual(len(papers()),11)
        for folder in papers().values():
            p=validate(folder)
            self.assertTrue((folder/'README.md').exists())
            self.assertTrue((folder/'reproduce.py').exists())
            if p['supported_modes']: self.assertTrue((folder/'adapter.py').exists())

    def test_wilson_zero_errors_does_not_mean_zero_upper_bound(self):
        rate,ci=wilson(0,100)
        self.assertEqual(rate,0)
        self.assertAlmostEqual(ci[1],.0369934982,places=8)
        for values in [(1,0),(-1,10),(11,10),(1.5,10)]:
            with self.assertRaises(ValueError): wilson(*values)

    def test_per_round_is_parity_conversion_not_division(self):
        eps,_=per_round(180,1000,2)
        self.assertAlmostEqual(eps,.1)
        self.assertNotEqual(eps,.18/2)

    def test_metric_units_and_conditions_are_enforced(self):
        target={'kind':'per_round_ler','unit':'fraction_per_round','conditions':{'distance':3},'value':.03}
        obs={'kind':'shot_failure_rate','unit':'fraction','conditions':{'distance':3},'value':.03}
        with self.assertRaises(ValueError): compare(target,obs)
        obs.update(kind=target['kind'],unit=target['unit'],conditions={'distance':5})
        with self.assertRaises(ValueError): compare(target,obs)
        obs['conditions']=target['conditions'];obs['value']=float('nan')
        with self.assertRaises(ValueError): compare(target,obs)

    def test_metric_physical_bounds(self):
        target={'kind':'frame_error_rate','unit':'fraction','conditions':{'distance':3},'value':.03}
        for value,ci in [(-.01,[-.02,0]),(1.2,[1.1,1.3]),(.01,[-.01,.03])]:
            with self.assertRaises(ValueError):
                compare(target,dict(target,value=value,ci95=ci))
        target['kind']='relative_ler_reduction'
        self.assertEqual(compare(target,dict(target,value=-.2))['observed'],-.2)

    def test_smoke_cannot_be_scored_as_reproduction(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'evidence.json'
            write(path,{'paper_id':'nbp4','classification':'infrastructure_smoke'})
            with self.assertRaises(ValueError): score(papers()['nbp4'],path)

    def test_hash_verification_and_comparison(self):
        folder=papers()['nbp4'];target=json.loads((folder/'targets.json').read_text())[0]
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'raw.txt';raw.write_text('3 errors / 3000 shots\n')
            obs={'target_id':target['id'],'kind':target['kind'],'unit':target['unit'],
                 'conditions':target['conditions'],'value':.001}
            evidence={'paper_id':'nbp4','classification':'candidate_reproduction','provenance':{'seed':1},
                      'artifacts':[{'path':'raw.txt','sha256':digest(raw)}],'observations':[obs]}
            path=Path(d)/'evidence.json';write(path,evidence)
            self.assertEqual(score(folder,path)[target['id']]['delta'],0)
            raw.write_text('modified\n')
            with self.assertRaises(ValueError): score(folder,path)


if __name__=='__main__': unittest.main()
