"""Embed the written paper guide and create its standalone reading view."""
from pathlib import Path


def include_paper_guide(template: str, root: Path, out: Path) -> str:
    body = (root / "gkp_paper_guide.html").read_text()
    css = (root / "gkp_paper_guide.css").read_text()
    template = template.replace("__PAPER_HTML__", body).replace("__PAPER_CSS__", css)
    base = """
    :root{--ink:#172b42;--muted:#617285;--teal:#087e82;--line:#dfe7ec}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f6f8;color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}a{color:var(--teal)}a:focus-visible{outline:3px solid #82bac4;outline-offset:4px}.shell{max-width:1280px;margin:auto;padding:30px 24px 60px}.topbar{background:#142a40;color:white;padding:18px 25px;font-size:13px;display:flex;justify-content:space-between;gap:20px}.topbar a{color:#c8d8e5;text-decoration:none}.section-intro{margin:20px 0 28px;max-width:940px}.eyebrow{font-size:10px;letter-spacing:1.8px;text-transform:uppercase;color:var(--teal);font-weight:750;margin-bottom:13px}h2{font:clamp(30px,4vw,46px)/1.17 Georgia,serif;letter-spacing:-1px;margin:0}.section-intro p{font-size:14px;color:var(--muted);line-height:1.8;margin:16px 0 0}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid var(--line);text-align:left}code{font-family:ui-monospace,monospace}@media(max-width:760px){.shell{padding:20px 15px 40px}.topbar{padding:15px;font-size:11px}}
    """
    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<meta name='description' content='A referenced guide to reinforcement learning control of quantum error correction and its connection to QKP.'>"
            "<title>QKP · The Nature paper explained</title><style>" + base + css + "</style></head>"
            "<body class='paper-standalone'><header class='topbar'><strong>QKP / research notes</strong>"
            "<a href='viewer.html#paper-guide'>Back to the illustrated viewer →</a></header>"
            "<main class='shell'>" + body + "</main></body></html>")
    (out / "paper.html").write_text(page)
    (out / "paper-references.bib").write_text((root / "gkp_paper_references.bib").read_text())
    return template
