"""Dependency-free orchestration for independently documented paper reproductions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request

from metrics import compare

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def papers():
    return {read(p)['id']: p.parent for p in sorted(ROOT.glob('*/paper.json'))}


def validate(folder):
    p, targets, protocol = [read(folder/n) for n in ('paper.json', 'targets.json', 'protocol.json')]
    assert p['title'] and p['source_url'] and p['methods'] and p['scope']
    assert protocol['status'] == 'planned'
    ids = set()
    for t in targets:
        assert t['id'] not in ids
        ids.add(t['id'])
        assert t['source_url'] and t['locator'] and t['kind'] and t['unit'] and t['conditions']
        assert 'value' in t or 'reference_interval' in t
        if 'reference_interval' in t:
            assert t['reference_interval'][0] < t['reference_interval'][1]
    for s in p['sources']:
        assert s['url'].startswith('https://')
    repo = p.get('upstream')
    if repo:
        assert len(repo['commit']) == 40 and all(c in '0123456789abcdef' for c in repo['commit'])
    return p


def git(*args, cwd):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def fetch(folder, kind):
    p = validate(folder)
    lockpath = folder/'artifacts'/'fetch-lock.json'
    lock = read(lockpath) if lockpath.exists() else {}
    if kind in ('paper', 'all'):
        for s in p['sources']:
            if not s.get('download'):
                continue
            path = folder/'artifacts'/s['name']
            if not path.exists():
                req = urllib.request.Request(s['url'], headers={'User-Agent': 'QKP-reproduction/1.0'})
                with urllib.request.urlopen(req, timeout=90) as r:
                    payload = r.read()
                if path.suffix == '.pdf' and not payload.startswith(b'%PDF'):
                    raise ValueError(f"Not a PDF: {s['url']}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            sha = digest(path)
            expected = s.get('sha256') or lock.get(s['name'], {}).get('sha256')
            if expected and sha != expected:
                raise ValueError(f"Source hash changed: {path}")
            lock[s['name']] = {'url': s['url'], 'sha256': sha}
            write(lockpath, lock)
    if kind in ('code', 'all'):
        repo = p.get('upstream')
        if not repo:
            raise ValueError('No verified public author repository. See blockers in paper.json.')
        path = folder/'upstream'
        if not path.exists():
            path.mkdir()
            git('init', cwd=path)
            git('remote', 'add', 'origin', repo['url'], cwd=path)
            git('fetch', '--depth', '1', '--filter=blob:none', 'origin', repo['commit'], cwd=path)
            if repo.get('checkout_paths'):
                git('config', 'core.sparseCheckout', 'true', cwd=path)
                (path/'.git/info/sparse-checkout').write_text('\n'.join(repo['checkout_paths'])+'\n')
            git('checkout', '--detach', 'FETCH_HEAD', cwd=path)
        if git('rev-parse', 'HEAD', cwd=path) != repo['commit']:
            raise ValueError('Existing upstream checkout is not the pinned commit; refusing to overwrite it')
        lock['upstream'] = repo
        write(lockpath, lock)
    return lock


def provenance(folder):
    repo = folder/'upstream'
    p = read(folder/'paper.json')
    if p.get('upstream') and not repo.exists():
        raise ValueError('Run fetch --kind code first')
    rev = git('rev-parse', 'HEAD', cwd=repo) if repo.exists() else None
    if rev and rev != p['upstream']['commit']:
        raise ValueError('Upstream commit differs from registered version')
    if repo.exists() and git('diff', '--name-only', 'HEAD', cwd=repo):
        raise ValueError('Tracked upstream files changed; use a separate documented patch')
    return {'python': sys.version, 'platform': platform.platform(), 'upstream_commit': rev,
            'protocol_sha256': digest(folder/'protocol.json'),
            'targets_sha256': digest(folder/'targets.json'),
            'adapter_sha256': digest(folder/'adapter.py') if (folder/'adapter.py').exists() else None,
            'runner_sha256': digest(Path(__file__)), 'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


def run(folder, mode, python):
    p = validate(folder)
    if mode not in p['supported_modes']:
        raise ValueError(f"{mode} unavailable. " + '; '.join(p['blockers']))
    prov = provenance(folder)
    output = folder/'runs'/(time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-'+mode+'-'+str(time.time_ns()%1000000))
    output.mkdir(parents=True)
    prov.update({'paper_id': p['id'], 'mode': mode, 'status': 'running',
                 'classification': 'infrastructure_smoke' if mode == 'smoke' else 'author_data_reanalysis' if mode == 'replay' else 'candidate_reproduction'})
    write(output/'provenance.json', prov)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', WANDB_MODE='disabled')
    cmd = [python, str(folder/'adapter.py'), mode, '--output', str(output)]
    prov['command'] = cmd
    environment = json.loads(subprocess.check_output([python, '-c',
        'import sys,json,platform,importlib.metadata as m; '
        'print(json.dumps({"executable":sys.executable,"version":sys.version,"platform":platform.platform(),'
        '"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True))
    prov['execution_environment'] = environment
    prov['shared_source_sha256'] = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    write(output/'environment.json', environment)
    (output/'environment.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(environment['packages'].items()))+'\n')
    with (output/'console.log').open('w') as log:
        rc = subprocess.call(cmd, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
    prov.update(status='completed' if rc == 0 else 'failed', returncode=rc)
    write(output/'provenance.json', prov)
    print(output/'console.log')
    if rc:
        raise RuntimeError((output/'console.log').read_text()[-2500:])
    return {'run': str(output), 'classification': prov['classification'], 'status': prov['status']}


def score(folder, path):
    evidence = read(path)
    p = read(folder/'paper.json')
    if evidence['paper_id'] != p['id'] or evidence['classification'] not in ('candidate_reproduction','author_data_reanalysis'):
        raise ValueError('Wrong paper or synthetic/smoke observations are not admissible')
    if not evidence.get('provenance') or not evidence.get('artifacts'):
        raise ValueError('Provenance and hashed raw artifacts are required')
    for artifact in evidence['artifacts']:
        artifact_path = (Path(path).resolve().parent/artifact['path']).resolve()
        if digest(artifact_path) != artifact['sha256']:
            raise ValueError('Raw artifact hash mismatch')
    targets = {t['id']: t for t in read(folder/'targets.json')}
    if not evidence.get('observations'):
        raise ValueError('No observations')
    return {o['target_id']: compare(targets[o['target_id']],o) for o in evidence['observations']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['list','check','plan','fetch','run','score'])
    ap.add_argument('--paper')
    ap.add_argument('--kind', choices=['paper','code','all'], default='paper')
    ap.add_argument('--mode', choices=['smoke','replay','train','evaluate'], default='smoke')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--input')
    args = ap.parse_args(argv)
    index = papers()
    try:
        if args.command in ('list','check'):
            for key, folder in index.items():
                if args.paper and key != args.paper: continue
                p = validate(folder)
                print(f"{key:18} {','.join(p['supported_modes']) or 'protocol/scoring only':24} {p['title']}")
            if args.paper and args.paper not in index: raise ValueError('Unknown paper')
            return
        if args.paper not in index: raise ValueError('Choose --paper from list')
        folder = index[args.paper]
        if args.command == 'plan': result = read(folder/'protocol.json')
        elif args.command == 'fetch': result = fetch(folder, args.kind)
        elif args.command == 'run': result = run(folder,args.mode,args.python)
        elif args.command == 'score':
            if not args.input: raise ValueError('Provide --input evidence.json')
            result = score(folder,args.input)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError,RuntimeError,KeyError,AssertionError,OSError,subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
