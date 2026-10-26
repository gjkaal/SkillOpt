"""SkillOpt-Sleep run monitor: read a Sleep evidence log and render it as HTML.

Sleep writes one JSON object per line to an evidence log while a night runs
(``<state_dir>/evidence/dryrun-*.jsonl`` for dry-runs,
``<project>/.skillopt-sleep/staging/<night>/evidence.jsonl`` for staged runs).
This module turns that log into a readable progress view for the WebUI.

Everything rendered here comes from model output and task files, so every
value passes through ``html.escape`` before it reaches the page.
"""
from __future__ import annotations

import html
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A log that has not been written for this long, and has no cycle/end event,
# is reported as stopped rather than running.
RUNNING_WINDOW_SECONDS = 180

PHASE_LABELS = {
    "baseline_val": "Baseline (validation)",
    "train": "Train replay",
    "gate_trial:skill": "Gate trial (skill edit)",
    "gate_trial:memory": "Gate trial (memory edit)",
    "train_post_skill": "Train replay (after skill edit)",
    "final_val": "Final validation",
}

SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}

EXTRA_PROJECTS_ENV = "SKILLOPT_WEBUI_SLEEP_PROJECTS"


@dataclass
class TaskInfo:
    task_id: str
    split: str = ""
    intent: str = ""
    checks: list[dict] = field(default_factory=list)


@dataclass
class ReplayResult:
    seq: int
    ts: str
    phase: str
    task_id: str
    split: str
    hard: float
    soft: float
    why: str
    response_head: str
    tokens: int
    latency_ms: float


@dataclass
class RunSummary:
    path: str
    status: str = "unknown"
    start: dict = field(default_factory=dict)
    end: dict = field(default_factory=dict)
    tasks: dict[str, TaskInfo] = field(default_factory=dict)
    results: list[ReplayResult] = field(default_factory=list)
    responses: list[dict] = field(default_factory=list)
    gate_baseline: dict = field(default_factory=dict)
    gate_trials: list[dict] = field(default_factory=list)
    gate_decision: dict = field(default_factory=dict)
    test_score: dict = field(default_factory=dict)
    reflect_edits: list[dict] = field(default_factory=list)
    unmatched_edits: list[dict] = field(default_factory=list)
    budget_skips: list[str] = field(default_factory=list)
    staged: dict = field(default_factory=dict)
    call_errors: list[str] = field(default_factory=list)
    n_calls: int = 0
    n_cache_hits: int = 0
    last_ts: str = ""
    last_phase: str = ""
    bad_lines: int = 0


# ---------------------------------------------------------------- discovery

def sleep_state_dir() -> Path:
    """Sleep's state directory, honoring ~/.skillopt-sleep/config.json."""
    try:
        from skillopt_sleep.config import load_config
        return Path(load_config().state_dir)
    except Exception:
        return Path(os.path.expanduser("~/.skillopt-sleep"))


def evidence_roots(project_root: Path) -> list[Path]:
    """Directories that may hold evidence logs, in display order."""
    roots = [sleep_state_dir() / "evidence"]
    projects = [project_root]
    for raw in os.environ.get(EXTRA_PROJECTS_ENV, "").split(os.pathsep):
        if raw.strip():
            projects.append(Path(os.path.expanduser(raw.strip())))
    roots.extend(p / ".skillopt-sleep" / "staging" for p in projects)
    return roots


def discover_evidence_files(project_root: Path) -> list[Path]:
    """All evidence logs under the known roots, newest first."""
    found: dict[Path, float] = {}
    for root in evidence_roots(project_root):
        if not root.is_dir() or root.is_symlink():
            continue
        candidates = list(root.glob("*.jsonl")) + list(root.glob("*/evidence.jsonl"))
        for path in candidates:
            if path.is_file() and not path.is_symlink():
                found[path.resolve()] = path.stat().st_mtime
    return sorted(found, key=found.get, reverse=True)


def resolve_evidence_choice(choice: str, project_root: Path) -> Path | None:
    """Map a UI value back to a discovered file; never trust it as a path."""
    if not choice:
        return None
    for path in discover_evidence_files(project_root):
        if str(path) == choice:
            return path
    return None


def evidence_choices(project_root: Path) -> list[tuple[str, str]]:
    """(label, value) pairs for the file dropdown."""
    choices = []
    for path in discover_evidence_files(project_root):
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
        name = path.name if path.name != "evidence.jsonl" else f"{path.parent.name}/evidence.jsonl"
        choices.append((f"{stamp}  {name}", str(path)))
    return choices


# ---------------------------------------------------------------- parsing

def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_events(lines: list[str]) -> tuple[list[dict], int]:
    """Decode JSONL lines; a partial last line (still being written) is skipped."""
    events, bad = [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events, bad


def summarize(events: list[dict], *, path: str = "", mtime: float | None = None,
              now: float | None = None) -> RunSummary:
    """Fold evidence events into one RunSummary."""
    s = RunSummary(path=path)
    for ev in events:
        stage, event = ev.get("stage", ""), ev.get("event", "")
        s.last_ts = ev.get("ts", s.last_ts)
        if ev.get("phase"):
            s.last_phase = ev["phase"]
        if (stage, event) == ("cycle", "start"):
            s.start = ev
        elif (stage, event) == ("cycle", "end"):
            s.end = ev
        elif (stage, event) == ("mine", "task_ready"):
            tid = str(ev.get("task_id", ""))
            s.tasks[tid] = TaskInfo(tid, ev.get("split", ""), ev.get("intent", ""),
                                    list(ev.get("checks") or []))
        elif (stage, event) == ("replay", "result"):
            s.results.append(ReplayResult(
                seq=int(ev.get("seq", 0) or 0), ts=ev.get("ts", ""),
                phase=ev.get("phase", ""), task_id=str(ev.get("task_id", "")),
                split=ev.get("split", ""), hard=_float(ev.get("hard")),
                soft=_float(ev.get("soft")), why=ev.get("why", ""),
                response_head=ev.get("response_head", ""),
                tokens=int(_float(ev.get("tokens"))),
                latency_ms=_float(ev.get("latency_ms"))))
        elif (stage, event) == ("replay", "model_call"):
            s.n_calls += 1
            if ev.get("cache_hit"):
                s.n_cache_hits += 1
            if ev.get("error"):
                s.call_errors.append(str(ev["error"]))
            if ev.get("kind") == "attempt" and ev.get("response"):
                s.responses.append({"phase": ev.get("phase", ""), "response": ev["response"]})
        elif (stage, event) == ("gate", "baseline"):
            s.gate_baseline = ev
        elif (stage, event) == ("gate", "trial"):
            s.gate_trials.append(ev)
        elif (stage, event) == ("gate", "decision"):
            s.gate_decision = ev
        elif (stage, event) == ("test", "held_out_score"):
            s.test_score = ev
        elif (stage, event) == ("reflect", "edits_returned"):
            s.reflect_edits.append(ev)
        elif (stage, event) == ("reflect", "edits_unmatched"):
            s.unmatched_edits.append(ev)
        elif (stage, event) == ("budget", "skip"):
            s.budget_skips.append(str(ev.get("step", "")))
        elif (stage, event) == ("stage", "staged"):
            s.staged = ev

    if s.end:
        s.status = "completed"
    elif mtime is not None and (now or time.time()) - mtime <= RUNNING_WINDOW_SECONDS:
        s.status = "running"
    elif events:
        s.status = "stopped"
    return s


def load_summary(path: Path) -> RunSummary:
    with open(path, encoding="utf-8", errors="replace") as f:
        events, bad = parse_events(f.readlines())
    summary = summarize(events, path=str(path), mtime=path.stat().st_mtime)
    summary.bad_lines = bad
    return summary


def failed_checks(task: TaskInfo | None, why: str) -> list[str]:
    """Plain-language descriptions of the checks a result's ``why`` names."""
    if not why or why == "all checks passed":
        return []
    if not why.startswith("failed: "):
        return [why]
    detail = why[len("failed: "):]
    missed = []
    for check in (task.checks if task else []):
        token = f"{check.get('op')}={check.get('arg')}"
        if token in detail:
            missed.append(check.get("description") or token)
    return missed or [detail]


def full_response(summary: RunSummary, result: ReplayResult) -> str:
    """The untruncated reply for a result, matched by phase and response prefix."""
    head = (result.response_head or "").strip()
    for call in reversed(summary.responses):
        if call["phase"] == result.phase and head and call["response"].strip().startswith(head[:200]):
            return call["response"]
    return result.response_head


# ---------------------------------------------------------------- rendering

_E = html.escape

_CSS = """
<style>
.som { font-size: 0.92rem; color: var(--body-text-color); }
.som h3 { margin: 0.6rem 0 0.4rem; font-size: 1.05rem; }
.som .cards { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 8px; }
.som .card { border: 1px solid var(--border-color-primary); border-radius: 8px;
             padding: 8px 12px; min-width: 140px; background: var(--background-fill-secondary); }
.som .card .k { font-size: 0.75rem; opacity: 0.75; }
.som .card .v { font-size: 1.1rem; font-weight: 600; }
.som table { border-collapse: collapse; width: 100%; }
.som th, .som td { border-bottom: 1px solid var(--border-color-primary);
                   padding: 4px 8px; text-align: left; vertical-align: top; }
.som th { font-size: 0.8rem; opacity: 0.8; }
.som .pass { color: #15803d; font-weight: 600; }
.som .fail { color: #b91c1c; font-weight: 600; }
.som .muted { opacity: 0.7; }
.som pre { white-space: pre-wrap; word-break: break-word; max-height: 480px; overflow: auto;
           border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 8px;
           background: var(--background-fill-secondary); }
.som ul { margin: 0.2rem 0 0.2rem 1.1rem; padding: 0; }
@media (prefers-color-scheme: dark) { .som .pass { color: #4ade80; } .som .fail { color: #f87171; } }
</style>
"""


def _wrap(body: str) -> str:
    return f'{_CSS}<div class="som">{body}</div>'


def _pct(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return f"{_float(value):.2f}"


def _card(key: str, value: str) -> str:
    return f'<div class="card"><div class="k">{_E(key)}</div><div class="v">{_E(value)}</div></div>'


def phase_label(phase: str) -> str:
    return PHASE_LABELS.get(phase, phase or "-")


def render_overview(summary: RunSummary | None) -> str:
    if summary is None:
        return _wrap('<p class="muted">No evidence log selected. Start a Sleep run '
                     '(<code>skillopt-sleep dry-run ...</code>) and press Refresh list.</p>')
    start, end = summary.start, summary.end
    tokens = end.get("tokens_used") if end else sum(r.tokens for r in summary.results)
    cards = [
        _card("Status", summary.status),
        _card("Current phase", "done" if end else phase_label(summary.last_phase)),
        _card("Backend", str(start.get("backend", "-"))),
        _card("Tasks", str(len(summary.tasks))),
        _card("Replays scored", str(len(summary.results))),
        _card("Model calls", f"{summary.n_calls} ({summary.n_cache_hits} cached)"),
        _card("Tokens", f"{int(_float(tokens)):,}"),
        _card("Last event", summary.last_ts or "-"),
    ]
    body = f'<h3>Run</h3><div class="cards">{"".join(cards)}</div>'
    body += f'<div class="muted">{_E(summary.path)} | project: {_E(str(start.get("project", "-")))}</div>'

    score_cards = []
    if summary.gate_baseline:
        b = summary.gate_baseline
        score_cards.append(_card("Baseline (val)", f"{_pct(b.get('score'))}  (hard {_pct(b.get('hard'))} / soft {_pct(b.get('soft'))})"))
    for i, t in enumerate(summary.gate_trials, 1):
        verdict = "accepted" if t.get("accepted") else "rejected"
        if t.get("mode") == "gated":
            score_cards.append(_card(f"Trial {i} ({t.get('target')})", f"{_pct(t.get('cand_score'))}  {verdict}"))
        else:
            score_cards.append(_card(f"Trial {i} ({t.get('target')})", f"{verdict} (gate off)"))
    if summary.gate_decision:
        d = summary.gate_decision
        score_cards.append(_card("Gate decision", f"{d.get('action', '-')}: {_pct(d.get('baseline_score'))} -> {_pct(d.get('candidate_score'))}"))
    if summary.test_score:
        t = summary.test_score
        score_cards.append(_card("Held-out test", f"hard {_pct(t.get('hard'))} / soft {_pct(t.get('soft'))}"))
    if score_cards:
        body += f'<h3>Scores</h3><div class="cards">{"".join(score_cards)}</div>'

    notes = []
    if summary.budget_skips:
        notes.append("Skipped by token budget: " + ", ".join(summary.budget_skips))
    if summary.call_errors:
        notes.append(f"{len(summary.call_errors)} model call error(s); last: {summary.call_errors[-1][:300]}")
    if summary.staged:
        notes.append(f"Staged proposal: {summary.staged.get('staging_dir', '')}")
    if end and end.get("outcome") != "completed":
        notes.append(f"Run ended with outcome: {end.get('outcome')}")
    if summary.bad_lines:
        notes.append(f"{summary.bad_lines} unreadable line(s) skipped")
    if notes:
        body += "<h3>Notes</h3><ul>" + "".join(f"<li>{_E(n)}</li>" for n in notes) + "</ul>"
    return _wrap(body)


def render_results_table(summary: RunSummary | None) -> str:
    if summary is None or not summary.results:
        return _wrap('<p class="muted">No replay results yet.</p>')
    rows = []
    for r in sorted(summary.results, key=lambda r: -r.seq):
        task = summary.tasks.get(r.task_id)
        ok = r.hard >= 1.0
        missed = failed_checks(task, r.why)
        missed_html = ("<ul>" + "".join(f"<li>{_E(m)}</li>" for m in missed) + "</ul>") if missed else ""
        rows.append(
            f"<tr><td>{_E(r.ts[11:19])}</td><td>{_E(phase_label(r.phase))}</td>"
            f"<td>{_E(r.task_id)}</td><td>{_E(r.split)}</td>"
            f'<td class="{"pass" if ok else "fail"}">{"PASS" if ok else "FAIL"}</td>'
            f"<td>{r.soft:.2f}</td><td>{missed_html}</td>"
            f"<td>{r.tokens:,}</td><td>{r.latency_ms / 1000:.1f}s</td></tr>")
    head = ("<tr><th>Time</th><th>Phase</th><th>Task</th><th>Split</th><th>Result</th>"
            "<th>Soft</th><th>Missed</th><th>Tokens</th><th>Latency</th></tr>")
    return _wrap(f"<h3>Replay results (newest first)</h3><table>{head}{''.join(rows)}</table>")


def render_edits(summary: RunSummary | None) -> str:
    if summary is None or not (summary.reflect_edits or summary.unmatched_edits):
        return _wrap('<p class="muted">No skill edits proposed yet.</p>')
    body = "<h3>Proposed edits</h3>"
    for i, batch in enumerate(summary.reflect_edits, 1):
        trial = trial_for_batch(summary, batch)
        if not batch.get("edits"):
            verdict = "no edits returned"
        elif trial:
            verdict = "accepted" if trial.get("accepted") else "rejected"
        else:
            verdict = "pending gate (or not applicable)"
        body += f"<p><b>Batch {i}</b> ({_E(str(batch.get('target', '')))}): {_E(verdict)}</p>"
        for edit in batch.get("edits") or []:
            anchor = edit.get("anchor") or ""
            body += (f"<p><b>{_E(str(edit.get('op', '')))}</b>"
                     + (f" at <code>{_E(anchor[:120])}</code>" if anchor else "")
                     + f'<br><span class="muted">{_E(str(edit.get("rationale", "")))}</span></p>'
                     + f"<pre>{_E(str(edit.get('content', '')))}</pre>")
    if summary.unmatched_edits:
        n = sum(int(_float(u.get("n_edits"))) for u in summary.unmatched_edits)
        body += f'<p class="muted">{n} edit(s) could not be applied (anchor not found in the skill).</p>'
    return _wrap(body)


def trial_for_batch(summary: RunSummary, batch: dict) -> dict:
    """The gate trial that judged an edit batch: same target, before the next batch."""
    start = int(_float(batch.get("seq")))
    later = [int(_float(b.get("seq"))) for b in summary.reflect_edits
             if int(_float(b.get("seq"))) > start]
    stop = min(later) if later else float("inf")
    for trial in summary.gate_trials:
        seq = int(_float(trial.get("seq")))
        if start < seq < stop and trial.get("target") == batch.get("target"):
            return trial
    return {}


def task_choices(summary: RunSummary | None) -> list[str]:
    if summary is None:
        return []
    return sorted(summary.tasks, key=lambda t: (SPLIT_ORDER.get(summary.tasks[t].split, 9), t))


def render_task_detail(summary: RunSummary | None, task_id: str) -> str:
    if summary is None or not task_id or task_id not in summary.tasks:
        return _wrap('<p class="muted">Select a task to see its checks and latest response.</p>')
    task = summary.tasks[task_id]
    results = [r for r in summary.results if r.task_id == task_id]
    latest = max(results, key=lambda r: r.seq) if results else None
    missed = set(failed_checks(task, latest.why)) if latest else set()
    body = f"<h3>{_E(task_id)} <span class='muted'>({_E(task.split)})</span></h3>"
    body += f"<p>{_E(task.intent)}</p><h3>Checks</h3><ul>"
    for check in task.checks:
        desc = check.get("description") or f"{check.get('op')}={check.get('arg')}"
        if latest is None:
            mark = '<span class="muted">-</span>'
        elif desc in missed:
            mark = '<span class="fail">missed</span>'
        else:
            mark = '<span class="pass">ok</span>'
        body += f"<li>{mark} {_E(desc)}</li>"
    body += "</ul>"
    if results:
        body += "<h3>Scores per phase</h3><ul>" + "".join(
            f"<li>{_E(phase_label(r.phase))}: {'PASS' if r.hard >= 1.0 else 'FAIL'} (soft {r.soft:.2f})</li>"
            for r in sorted(results, key=lambda r: r.seq)) + "</ul>"
        body += (f"<h3>Latest response ({_E(phase_label(latest.phase))})</h3>"
                 f"<pre>{_E(full_response(summary, latest))}</pre>")
    return _wrap(body)
