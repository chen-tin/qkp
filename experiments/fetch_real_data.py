"""Download the open T1 time series used by experiments/benchmark_real.py.

Zhu et al., "Disentangling the impact of quasiparticles and two-level systems
on the statistics of superconducting qubit lifetime" (arXiv:2409.09926),
data doi:10.5281/zenodo.16754689, CC-BY-4.0. Three transmons x 11 fridge
temperatures; each key holds T1 (microseconds) and its time stamps (seconds).

Run from the repo root:  python -m experiments.fetch_real_data
Files go to data/zhu_t1/ (gitignored) and are checked against SHA-256.
"""

import hashlib
import urllib.request

from experiments.common import ROOT

RECORD = "https://zenodo.org/api/records/16754689/files/{name}/content"
FILES = {
    "B7Q1_T1_timeseries.npz": "6cf702b05a2299eb982b545c0cf359d23004b93d0a80b4842fff8c8f9474d953",
    "B7Q5_T1_timeseries.npz": "73932850836032feacc2a4bfd8fedb76189e90cd570b44f35d434c287915acb6",
    "B8Q5_T1_timeseries.npz": "1f1c9957f5117abaf03b314e64250ef54ca81518f6a47bff12ca2c485e90479c",
}
DATA_DIR = ROOT / "data" / "zhu_t1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = DATA_DIR / name
        if path.exists() and sha256(path) == digest:
            continue
        req = urllib.request.Request(RECORD.format(name=name),
                                     headers={"User-Agent": "research-client"})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
        if sha256(path) != digest:
            path.unlink()
            raise RuntimeError(f"{name}: checksum mismatch, file removed")
        print(f"fetched {name}")
    return DATA_DIR


if __name__ == "__main__":
    print(fetch())
