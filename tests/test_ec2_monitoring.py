"""Operational status must not turn archived application labels into live claims."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

NOW = datetime(2026, 11, 23, 22, 0, tzinfo=timezone.utc)


def test_stopped_instance_overrides_archived_running_label_without_mutating_evidence(tmp_path):
    from ops.ec2 import monitoring
    archived = {"status": "running_production_sweeps", "updated": "2026-11-21T00:31:45+00:00"}
    path = tmp_path / "metrics/status.json"
    path.parent.mkdir()
    path.write_text(json.dumps(archived))
    original = path.read_bytes()
    report = monitoring.write_observation(tmp_path, archived, "stopped", checked=NOW)
    assert report["effective_status"] == "interrupted_instance_stopped"
    assert report["archived_status"] == "running_production_sweeps"
    assert report["heartbeat_stale"] is True
    assert path.read_bytes() == original
    assert "interrupted_instance_stopped" in (tmp_path / "index.html").read_text()


@pytest.mark.parametrize("stamp", ["2026-11-21T00:31:45+00:00", None, "invalid", "2026-11-24T00:00:00+00:00"])
def test_stale_missing_invalid_or_future_heartbeat_never_reports_running(stamp):
    from ops.ec2 import monitoring
    observed = monitoring.observation({"status": "running_production_sweeps", "updated": stamp},
                                      "running", checked=NOW)
    assert observed["effective_status"] in {"stale_heartbeat", "invalid_heartbeat"}


def test_api_failure_does_not_claim_instance_still_running():
    from ops.ec2 import monitoring
    observed = monitoring.observation({"status": "running_production_sweeps", "updated": NOW.isoformat()},
                                      "unknown", checked=NOW, query_error="expired credentials")
    assert observed["effective_status"] == "instance_status_unknown"
    assert observed["query_error"] == "expired credentials"


def test_completed_science_preserved_when_instance_stopped():
    from ops.ec2 import monitoring
    observed = monitoring.observation({"status": "all_scheduled_observations_completed", "updated": NOW.isoformat()},
                                      "stopped", checked=NOW)
    assert observed["effective_status"] == "all_scheduled_observations_completed"


def test_fetch_queries_ec2_even_if_archived_label_says_running_and_uses_timestamps(tmp_path, monkeypatch):
    from ops.ec2 import fetch_metrics
    from argparse import Namespace
    manifest = {"profile": "test", "region": "test-region", "bucket": "bucket", "run_id": "run",
                "resources": {"instance_id": "i-test"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    dest = tmp_path / "metrics";dest.mkdir()
    (dest / "status.json").write_text(json.dumps({"status": "running_production_sweeps", "updated": NOW.isoformat()}))
    commands = []
    def run(argv, **kwargs):
        commands.append(argv)
        if "describe-instances" in argv:
            return Namespace(returncode=0, stdout='"stopped"', stderr="")
        return Namespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(fetch_metrics.subprocess, "run", run)
    monkeypatch.setattr(fetch_metrics.time, "sleep", lambda *a: pytest.fail("Poller ignored stopped EC2"))
    monkeypatch.setattr("sys.argv", ["fetch_metrics", str(tmp_path)])
    fetch_metrics.main()
    assert any("--exact-timestamps" in c for c in commands if "sync" in c)
    assert any("describe-instances" in c for c in commands)
    obs=json.loads((tmp_path / "observed_status.json").read_text())
    assert obs["effective_status"] == "interrupted_instance_stopped"


def test_manage_observation_api_failure_keeps_archive_unchanged(tmp_path, monkeypatch):
    from ops.ec2 import manage
    status = tmp_path / "metrics/status.json";status.parent.mkdir()
    status.write_text('{"status":"running_production_sweeps","updated":"2026-11-21T00:31:45+00:00"}\n')
    original = status.read_bytes()
    def unavailable(*args):
        raise RuntimeError("expired credentials")
    monkeypatch.setattr(manage, "aws", unavailable)
    result = manage.observe(tmp_path, {"resources": {"instance_id": "i-test"}})
    assert result["effective_status"] == "instance_status_unknown"
    assert status.read_bytes() == original


def test_fresh_heartbeat_on_running_instance_is_explicitly_scoped():
    from ops.ec2 import monitoring
    result = monitoring.observation({"status": "running_production_sweeps", "updated": NOW.isoformat()},
                                    "running", checked=NOW)
    assert result["effective_status"] == "running_production_sweeps"
    assert result["heartbeat_stale"] is False
    assert "not a direct process inspection" in result["scope"]


def test_sync_timeout_still_observes_stopped_ec2(tmp_path, monkeypatch):
    from ops.ec2 import fetch_metrics
    from argparse import Namespace
    manifest = {"profile": "test", "region": "test-region", "bucket": "bucket", "run_id": "run",
                "resources": {"instance_id": "i-test"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    def run(argv, **kwargs):
        if "sync" in argv:
            raise fetch_metrics.subprocess.TimeoutExpired(argv, 180)
        return Namespace(returncode=0, stdout='"stopped"', stderr="")
    monkeypatch.setattr(fetch_metrics.subprocess, "run", run)
    monkeypatch.setattr("sys.argv", ["fetch_metrics", str(tmp_path)])
    fetch_metrics.main()
    result=json.loads((tmp_path / "observed_status.json").read_text())
    assert result["effective_status"] == "interrupted_instance_stopped"
    assert json.loads((tmp_path / "fetch_status.json").read_text())["fetch_returncode"] is None
