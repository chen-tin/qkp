"""Local operational observations, kept separate from archived scientific status."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path

TERMINAL = {"failed", "interrupted", "numerical_baseline_failed_or_inconclusive",
            "all_scheduled_observations_completed"}


def observation(archived, instance_state, checked=None, stale_after=180, query_error=None):
    checked = checked or datetime.now(timezone.utc)
    age = None
    try:
        stamp = datetime.fromisoformat(archived.get("updated", "").replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Heartbeat lacks timezone")
        age = (checked - stamp).total_seconds()
    except (ValueError, TypeError, AttributeError):
        pass
    stale = age is None or age < -30 or age > stale_after
    status = archived.get("status", "missing_archived_status")
    if status in TERMINAL:
        effective = status
    elif instance_state in {"stopped", "terminated", "stopping", "shutting-down"}:
        effective = "interrupted_instance_" + instance_state.replace("-", "_")
    elif instance_state != "running":
        effective = "instance_status_unknown" if instance_state == "unknown" else "instance_" + instance_state
    elif age is None or age < -30:
        effective = "invalid_heartbeat"
    elif stale:
        effective = "stale_heartbeat"
    else:
        effective = status
    result = {"checked_utc": checked.isoformat(), "instance_state": instance_state,
              "effective_status": effective, "archived_status": status,
              "archived_updated": archived.get("updated"), "heartbeat_age_seconds": age,
              "heartbeat_stale": stale, "stale_after_seconds": stale_after,
              "scope": "Local EC2/heartbeat observation; archived metrics are unchanged. A recent heartbeat is not a direct process inspection."}
    if query_error:
        result["query_error"] = str(query_error)
    return result


def write_observation(folder, archived, instance_state, checked=None, query_error=None):
    folder = Path(folder)
    report = observation(archived, instance_state, checked=checked, query_error=query_error)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "observed_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    status = html.escape(report["effective_status"])
    links = "".join(f'<li><a href="metrics/{suite}/">{suite}</a></li>' for suite in
                    ("numerical", "noise", "physics", "finite_gate", "coverage", "rare"))
    page = ('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
            '<title>GKP run observation</title><style>body{font:16px system-ui;max-width:960px;margin:40px auto;padding:0 20px}'
            'pre{white-space:pre-wrap}</style><h1>GKP run observation</h1>'
            f'<p>Observed status: <strong>{status}</strong></p>'
            f'<p>Checked {html.escape(report["checked_utc"])}. Instance: {html.escape(instance_state)}.</p>'
            '<p>Scientific reports below are archived observations. Their application status labels may be stale.</p>'
            f'<ul>{links}</ul><p><a href="metrics/">Archived dashboard</a> · '
            '<a href="observed_status.json">Local observation JSON</a></p>'
            f'<pre>{html.escape(json.dumps(report, indent=2))}</pre>')
    temporary = folder / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(folder / "index.html")
    return report


def read_archived_status(folder):
    try:
        return json.loads((Path(folder) / "metrics/status.json").read_text())
    except (OSError, ValueError):
        return {}
