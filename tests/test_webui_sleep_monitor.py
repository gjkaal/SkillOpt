"""Tests for the WebUI Sleep monitor (evidence-log parsing and rendering).

Pure functions only: no Gradio, no network, no real Sleep state directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

from skillopt_webui import sleep_monitor as sm

CHECKS = [
    {"op": "contains", "arg": "invoicerwe", "description": "Name the registry as app plus region code."},
    {"op": "regex", "arg": "(?m)^\\s*pr:", "description": "Declare an explicit PR trigger block."},
]


def _events():
    return [
        {"seq": 1, "ts": "2026-10-26T11:16:20", "stage": "cycle", "event": "start",
         "backend": "azure:gpt-5.6-sol", "project": "/p"},
        {"seq": 2, "ts": "2026-10-26T11:16:20", "stage": "mine", "event": "task_ready",
         "task_id": "t02", "split": "train", "intent": "Write a pipeline", "checks": CHECKS},
        {"seq": 3, "ts": "2026-10-26T11:17:00", "stage": "gate", "event": "baseline",
         "hard": 0.5, "soft": 0.9, "score": 0.7},
        {"seq": 4, "ts": "2026-10-26T11:19:14", "stage": "replay", "event": "model_call",
         "kind": "attempt", "phase": "train", "cache_hit": False,
         "response": "trigger:\n  - main\nfull tail", "error": ""},
        {"seq": 5, "ts": "2026-10-26T11:19:14", "stage": "replay", "event": "result",
         "phase": "train", "task_id": "t02", "split": "train", "hard": 0.0, "soft": 0.5,
         "why": "failed: contains=invoicerwe", "response_head": "trigger:\n  - main",
         "tokens": 4746, "latency_ms": 41394.2},
        {"seq": 6, "stage": "reflect", "event": "edits_returned", "target": "skill",
         "edits": [{"op": "add", "content": "<b>x</b>", "anchor": "", "rationale": "r"}]},
        {"seq": 7, "stage": "gate", "event": "trial", "target": "skill", "mode": "gated",
         "cand_score": 0.8, "accepted": True},
    ]


def test_summarize_folds_events():
    s = sm.summarize(_events(), path="x.jsonl", mtime=1000.0, now=1010.0)
    assert s.status == "running"
    assert s.start["backend"] == "azure:gpt-5.6-sol"
    assert list(s.tasks) == ["t02"]
    assert len(s.results) == 1 and s.results[0].tokens == 4746
    assert s.gate_baseline["score"] == 0.7
    assert s.n_calls == 1 and s.last_phase == "train"


def test_status_stopped_and_completed():
    stale = sm.summarize(_events(), mtime=0.0, now=10_000.0)
    assert stale.status == "stopped"
    done = sm.summarize(_events() + [{"seq": 9, "stage": "cycle", "event": "end", "outcome": "completed"}])
    assert done.status == "completed"


def test_parse_events_skips_partial_line():
    lines = [json.dumps({"stage": "cycle", "event": "start"}), '{"stage": "rep']
    events, bad = sm.parse_events(lines)
    assert len(events) == 1 and bad == 1


def test_failed_checks_maps_to_descriptions():
    task = sm.TaskInfo("t02", checks=CHECKS)
    assert sm.failed_checks(task, "failed: contains=invoicerwe") == [CHECKS[0]["description"]]
    assert sm.failed_checks(task, "all checks passed") == []
    assert sm.failed_checks(None, "failed: odd") == ["odd"]


def test_full_response_matches_by_prefix():
    s = sm.summarize(_events())
    assert sm.full_response(s, s.results[0]).endswith("full tail")


def test_trial_for_batch_pairs_by_order():
    s = sm.summarize(_events())
    assert sm.trial_for_batch(s, s.reflect_edits[0])["accepted"] is True


def test_rendering_escapes_model_output():
    events = _events()
    events[4]["response_head"] = "<script>alert(1)</script>"
    s = sm.summarize(events)
    for out in (sm.render_overview(s), sm.render_results_table(s),
                sm.render_edits(s), sm.render_task_detail(s, "t02")):
        assert "<script>alert" not in out
    assert "&lt;b&gt;x&lt;/b&gt;" in sm.render_edits(s)


def test_resolve_choice_only_accepts_discovered_files(tmp_path):
    state = tmp_path / "state"
    (state / "evidence").mkdir(parents=True)
    log = state / "evidence" / "dryrun-1.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    with mock.patch.object(sm, "sleep_state_dir", return_value=state), \
            mock.patch.dict(os.environ, {sm.EXTRA_PROJECTS_ENV: ""}):
        assert sm.resolve_evidence_choice(str(log.resolve()), project) == log.resolve()
        outside = tmp_path / "secret.jsonl"
        outside.write_text("{}\n", encoding="utf-8")
        assert sm.resolve_evidence_choice(str(outside), project) is None
        assert sm.resolve_evidence_choice(str(Path("..") / "x"), project) is None
