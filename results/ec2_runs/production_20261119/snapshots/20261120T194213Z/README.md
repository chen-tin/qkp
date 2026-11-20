# EC2 production metrics snapshot

Captured 2026-11-20T19:42:13.690567+00:00 from the local read-only metrics mirror. This is a **partial production snapshot**, not a completed benchmark or a live GitHub feed.

The numerical baseline passed its registered stage, scoring-refinement and cutoff-equivalence sample checks. Finite-gate controls failed population gates. Physics and noise sweeps are incomplete; parameter coverage and rare-event grids remain queued. Supplied-state independent instrument validation remains separate and incomplete.

See [root heartbeat](status.json), [numerical summary](numerical/summary.json), [physics summary](physics/summary.json), [noise summary](noise/summary.json), and [snapshot checksums](snapshot_manifest.json). Root heartbeat and suite reports may reflect adjacent exporter checkpoints.

Serve the repository locally and open this directory or `live/index.html` to review the captured plots. Refresh markup in the original HTML reloads these same saved files; it does not fetch AWS data. Raw observations remain in the private run archive, outside Git.
