"""Rebuild the offline viewer from existing illustrations and latest heatmaps.

Run: .venv/bin/python -m writeup.artifact.refresh_gkp_viewer
No figures or decoder evaluations are recomputed.
"""
import json
import re
import zipfile
from pathlib import Path

from writeup.artifact.paper_guide import include_paper_guide


def refresh(package=True):
    root = Path(__file__).resolve().parent
    out = root / "gkp-walkthrough"
    page = out / "viewer.html"
    match = re.search(r'<script id="walkthrough-data"[^>]*>(.*?)</script>', page.read_text(), re.S)
    if match is None:
        raise ValueError("Existing viewer has no illustration payload; run build_gkp_walkthrough first")
    template = (root / "gkp_walkthrough_template.html").read_text()
    template = include_paper_guide(template, root, out)
    for name, prefix in (("gkp_noise_playground", "NOISE"), ("gkp_decoder_heatmaps", "HEATMAP")):
        for ext in ("html", "css", "js"):
            template = template.replace(f"__{prefix}_{ext.upper()}__", (root / f"{name}.{ext}").read_text())
    source = out / "data" / "decoder-heatmaps.json"
    heatmaps = json.loads(source.read_text()) if source.exists() else None
    template = template.replace("__HEATMAP_DATA__", json.dumps(heatmaps, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"))
    template = template.replace("__PAYLOAD__", match.group(1))
    if re.search(r'__(?:NOISE|HEATMAP|PAPER|PAYLOAD)[A-Z_]*__', template):
        raise ValueError("Unresolved template placeholder")
    page.write_text(template)
    if package:
        with zipfile.ZipFile(root / "qkp-gkp-walkthrough.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(out))
    return page


if __name__ == "__main__":
    print(refresh())
