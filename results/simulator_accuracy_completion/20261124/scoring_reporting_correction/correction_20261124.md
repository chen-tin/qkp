# 2026-11-24: deterministic quadrature reporting correction

The active `experiments/refine_logical_scores.py` producer now reports fixed-state quadrature changes as deterministic sensitivities. It no longer emits `difference.se`, `difference.ci95_low` or `difference.ci95_high` in either serial or shared-basis reports. These fields in frozen historical refinement reports must not be interpreted as Monte Carlo confidence intervals for quadrature error.

The representative-only, equal-cardinal-weight `difference.mean` and its list shape are preserved exactly. Selected hot states remain excluded from that mean. The new deterministic range and the existing maximum absolute difference include every selected state, including hot states. Scoring calls, trace/positivity checks, tolerances, gate expressions and all-cycle coverage are unchanged.

- Old active producer SHA256: `eb80abca3ab464605298492cd11d0af9e5f254fb0798e2922e628167d185837f`.
- New active producer SHA256: `5f63badfcc3a7d732e4b8c5f2b2439cac530edb2254916597cd80225f6eb0b02`.
- Tests: four new cases failed on the old uncertainty fields; the corrected scorer/control suite passed 33 tests in 3.08 seconds. Exact commands and logs are recorded in `correction_20261124.json`.

No historical scientific job was rerun or rescored. All 21 checked completed refinement reports, both scientific plans and their frozen producer snapshots remain unchanged. Completed results retain their original source identities; the new reporting version has a separate identity. Current-source validation of an old plan should reject the deliberately changed active producer rather than silently rewriting provenance.

The machine-readable correction identifies the exact superseded field paths, before/after hashes, AST checks showing unchanged scorer calls and gates, report-preservation checks, and source freeze for fresh integration.
