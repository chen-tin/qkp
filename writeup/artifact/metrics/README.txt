QKP metrics overview

Open viewer.html in any modern browser; no server or internet is required.
The HTML embeds all charts and chart data. PNG, SVG and CSV downloads also work offline.
figures/ contains standalone Matplotlib plots. data/ contains plotted tables.
sources/ contains the source snapshots; metrics_data.json records SHA-256 hashes.

Regenerate inside the qkp repository:
  .venv/bin/python -m writeup.artifact.build_metrics
This only renders saved data, including completed models in the active size sweep. It does not train models.
Use --watch-size-sweep to rebuild when another model finishes; reload the offline page to see updates.
Source commit: 4152d253c0360dd50620c23f936070ba4a1a8d19
Built: 2026-11-18T09:47:23+00:00
