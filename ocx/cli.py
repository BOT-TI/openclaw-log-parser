"""
ocx

CLI companion for OpenClaw logs.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# =============================================================================
# CLI wiring
# =============================================================================

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

# =============================================================================
# Domain model
# =============================================================================


class EventType(str, Enum):
    """A minimal taxonomy for agent/run log events."""

    plan = "plan"
    tool = "tool"
    error = "error"
    output = "output"
    meta = "meta"
    unknown = "unknown"


@dataclass(frozen=True)
class Event:
    """
    A normalized representation of a single log event.

    - ts: timestamp in ISO-ish format if present (e.g. 2026-02-17T13:48:47.365Z)
    - run_id: a run correlation id if present
    - type: coarse event type (error/tool/...)
    - name: more specific classifier label (e.g. "gateway.conflict")
    - payload: arbitrary extra data (raw line, subsystem, level, parsed json, ...)
    """

    ts: Optional[str]
    run_id: Optional[str]
    type: EventType
    name: str
    payload: Dict[str, object]


# =============================================================================
# Parsing: regex + helpers
# =============================================================================

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"^\[?(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(\.\d+)?Z?)\]?\s*")
LEVEL_RE = re.compile(r"\b(trace|debug|info|warn|error)\b", re.I)
SUBSYS_RE = re.compile(r'"subsystem"\s*:\s*"([^"]+)"')
RUN_RE = re.compile(r"\b(runId|run_id|run|traceId|trace_id)\s*[:=]\s*([a-f0-9\-]{8,})\b", re.I)


def strip_ansi(s: str) -> str:
    """Remove ANSI color escape sequences from a string."""
    return ANSI_RE.sub("", s)


def try_parse_json(line: str) -> Optional[dict]:
    """
    Best-effort attempt to parse a full line as JSON.

    We only attempt JSON parsing if the line looks like a JSON object.
    """
    s = line.strip()
    if not s:
        return None
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_ts(clean_line: str) -> Optional[str]:
    m = TS_RE.search(clean_line)
    return m.group(1) if m else None


def extract_run_id(clean_line: str) -> Optional[str]:
    m = RUN_RE.search(clean_line)
    return m.group(2) if m else None


def extract_level(clean_line: str) -> str:
    m = LEVEL_RE.search(clean_line)
    return m.group(1).lower() if m else "unknown"


def extract_subsystem(clean_line: str) -> Optional[str]:
    m = SUBSYS_RE.search(clean_line)
    return m.group(1) if m else None


# =============================================================================
# Parsing: classification
# =============================================================================


def classify_json(obj: dict) -> Event:
    """
    Normalize a JSON event into our Event model.

    Supports a few common key variants (ts/timestamp, run_id/runId, etc).
    """
    ts = obj.get("ts") or obj.get("timestamp") or obj.get("time")
    run_id = obj.get("run_id") or obj.get("runId") or obj.get("trace_id") or obj.get("traceId")

    et = (obj.get("type") or obj.get("event") or "").lower()
    name = obj.get("name") or obj.get("event_name") or et or "event"

    if et in ("plan", "agent.plan"):
        t = EventType.plan
    elif "tool" in et or "function" in et:
        t = EventType.tool
    elif et in ("error", "exception", "failed"):
        t = EventType.error
    elif et in ("output", "result", "success", "completed"):
        t = EventType.output
    else:
        msg = json.dumps(obj).lower()
        t = EventType.error if ("schema mismatch" in msg or "exception" in msg) else EventType.meta

    return Event(
        ts=str(ts) if ts is not None else None,
        run_id=str(run_id) if run_id is not None else None,
        type=t,
        name=str(name),
        payload=obj,
    )


def classify_openclaw_text_line(line: str) -> Event:
    """
    Classify an OpenClaw log line (structured text with optional JSON fragments).

    OpenClaw output often looks like:
      [ts] info gateway/ws {"subsystem":"gateway/ws"} ⇄ res ✓ config.get ...

    We:
    - strip ANSI codes
    - extract ts, runId, level, subsystem
    - apply high-signal rules first
    - fall back to level-based classification
    """
    raw = line.rstrip("\n")
    clean = strip_ansi(raw)

    ts = extract_ts(clean)
    run_id = extract_run_id(clean)
    level = extract_level(clean)
    subsystem = extract_subsystem(clean)

    low = clean.lower()

    # ---- High-signal classifications ----

    # Gateway conflicts
    if "gateway already running" in low or ("port" in low and "already in use" in low):
        return Event(
            ts=ts,
            run_id=run_id,
            type=EventType.error,
            name="gateway.conflict",
            payload={"raw": clean, "level": level, "subsystem": subsystem},
        )

    # Embedded run lifecycle
    if "embedded run start" in low:
        return Event(ts, run_id, EventType.meta, "run.start", {"raw": clean, "level": level, "subsystem": subsystem})
    if "embedded run prompt start" in low:
        return Event(ts, run_id, EventType.meta, "run.prompt_start", {"raw": clean, "level": level, "subsystem": subsystem})
    if "embedded run prompt end" in low:
        return Event(ts, run_id, EventType.meta, "run.prompt_end", {"raw": clean, "level": level, "subsystem": subsystem})
    if "embedded run done" in low:
        return Event(ts, run_id, EventType.output, "run.done", {"raw": clean, "level": level, "subsystem": subsystem})
    if "embedded run timeout" in low or ("timeoutms" in low and "embedded run" in low):
        return Event(ts, run_id, EventType.error, "run.timeout", {"raw": clean, "level": level, "subsystem": subsystem})

    # Provider issues (rate limits / cooldowns / connection errors)
    if "rate_limit" in low or ("provider" in low and "cooldown" in low):
        return Event(ts, run_id, EventType.error, "provider.rate_limit", {"raw": clean, "level": level, "subsystem": subsystem})
    if "failovererror" in low or "connection error" in low:
        return Event(ts, run_id, EventType.error, "provider.connection_error", {"raw": clean, "level": level, "subsystem": subsystem})

    # gateway/ws tends to be noisy
    if subsystem and subsystem.startswith("gateway/ws"):
        return Event(ts, run_id, EventType.meta, "gateway.ws", {"raw": clean, "level": level, "subsystem": subsystem})

    # ---- Generic fallbacks ----
    if level == "error":
        return Event(ts, run_id, EventType.error, "error", {"raw": clean, "level": level, "subsystem": subsystem})
    if level == "warn":
        return Event(ts, run_id, EventType.meta, "warn", {"raw": clean, "level": level, "subsystem": subsystem})

    return Event(ts, run_id, EventType.meta, "log", {"raw": clean, "level": level, "subsystem": subsystem})


def parse_events(lines: Iterable[str]) -> Iterator[Event]:
    """
    Convert raw lines to normalized Events.

    Strategy:
    - If the line is a JSON object, parse it into an Event.
    - Otherwise, treat it as OpenClaw-ish text and classify it.
    """
    for line in lines:
        if not line.strip():
            continue
        obj = try_parse_json(line)
        yield classify_json(obj) if obj is not None else classify_openclaw_text_line(line)


# =============================================================================
# IO: reading logs from stdin / file / openclaw --follow
# =============================================================================


def iter_lines_from_source(source: Optional[Path]) -> Iterator[str]:
    """Yield lines from either stdin (default) or a file path."""
    if source is None:
        for line in sys.stdin:
            yield line.rstrip("\n")
    else:
        with source.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n")


def iter_lines_from_openclaw_follow() -> Iterator[str]:
    """Spawn `openclaw logs --follow` and yield its output lines."""
    proc = subprocess.Popen(
        ["openclaw", "logs", "--follow"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for ln in proc.stdout:
        yield ln.rstrip("\n")


# =============================================================================
# Presentation (rich)
# =============================================================================


def format_event(ev: Event) -> Tuple[str, str]:
    """Convert Event into (style, text) for Rich output."""
    ts = f"[{ev.ts}] " if ev.ts else ""
    rid = f" run={ev.run_id}" if ev.run_id else ""
    subs = ev.payload.get("subsystem")
    subs_s = f" [{subs}]" if subs else ""
    msg = ev.payload.get("raw", "")
    text = f"{ts}{ev.name}{subs_s}{rid} {msg}"

    if ev.type == EventType.error:
        return ("red", text)
    if ev.type == EventType.tool:
        return ("cyan", text)
    if ev.type == EventType.plan:
        return ("violet", text)
    if ev.type == EventType.output:
        return ("green", text)
    return ("dim", text)


# =============================================================================
# De-duplication (always on)
# =============================================================================


def event_key(ev: Event) -> str:
    """
    Key used for deduplication.

    We intentionally ignore timestamps so repeating the same message becomes
    one summarized line.
    """
    subs = ev.payload.get("subsystem")
    raw = str(ev.payload.get("raw", ""))
    return f"{ev.type.value}|{ev.name}|{subs}|{raw}"


def dedup_consecutive(events: Iterable[Event]) -> Iterator[Tuple[Event, int]]:
    """Yield (event, count) for consecutive duplicates."""
    last_key: Optional[str] = None
    last_event: Optional[Event] = None
    count = 0

    for ev in events:
        k = event_key(ev)
        if last_key is None:
            last_key = k
            last_event = ev
            count = 1
            continue

        if k == last_key:
            count += 1
            continue

        assert last_event is not None
        yield (last_event, count)
        last_key = k
        last_event = ev
        count = 1

    if last_event is not None:
        yield (last_event, count)


def print_event_deduped(events: Iterable[Event]) -> None:
    """Always-print with consecutive dedup + (xN)."""
    for ev, count in dedup_consecutive(events):
        style, text = format_event(ev)
        if count > 1:
            text = f"{text}  [dim](x{count})[/dim]"
        console.print(f"[{style}]{text}[/{style}]")


# =============================================================================
# Commands
# =============================================================================


@app.command()
def tail(
    follow: bool = typer.Option(False, "--follow", "-f", help="Run `openclaw logs --follow` and pretty-print."),
    source: Optional[Path] = typer.Argument(None, help="Log file path (defaults to stdin)."),
    errors_only: bool = typer.Option(False, "--errors-only", help="Only print error-classified events."),
    hide_ws: bool = typer.Option(True, "--hide-ws/--show-ws", help="Hide noisy gateway/ws lines (default: hide)."),
):
    """
    Pretty tail for OpenClaw agent runs.

    Examples:
      openclaw logs --follow | ocx tail
      ocx tail --follow
      ocx tail ./openclaw.log
      ocx tail --follow --errors-only
    """
    if follow and source is not None:
        raise typer.BadParameter("Use either --follow OR a source file OR stdin, not both.")

    lines = iter_lines_from_openclaw_follow() if follow else iter_lines_from_source(source)

    event_stream: Iterable[Event] = parse_events(lines)

    if hide_ws:
        event_stream = (ev for ev in event_stream if ev.name != "gateway.ws")
    if errors_only:
        event_stream = (ev for ev in event_stream if ev.type == EventType.error)

    # Dedup is ALWAYS on
    print_event_deduped(event_stream)


@app.command()
def analyze(
    source: Optional[Path] = typer.Argument(None, help="Log file path (defaults to stdin)."),
    top_errors: int = typer.Option(5, "--top-errors", help="How many error samples to show."),
):
    """
    Summarize failures and signals from OpenClaw logs.

    Prints:
    - number of distinct runs (run_id)
    - event type counts (volume-aware even though output is deduped)
    - a few representative error samples (deduped with xN)
    """
    counts: Dict[str, int] = {t.value: 0 for t in EventType}
    run_event_counts: Dict[str, int] = {}
    err_samples: List[str] = []

    # Dedup is ALWAYS on (but counts keep real volume via count)
    for ev, n in dedup_consecutive(parse_events(iter_lines_from_source(source))):
        counts[ev.type.value] += n
        if ev.run_id:
            run_event_counts[ev.run_id] = run_event_counts.get(ev.run_id, 0) + n

        if ev.type == EventType.error and len(err_samples) < top_errors:
            raw = str(ev.payload.get("raw", ""))
            sample = raw[:220]
            if n > 1:
                sample = f"{sample} (x{n})"
            err_samples.append(sample)

    table = Table(title="ocx analyze")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Runs (distinct run_id)", str(len(run_event_counts)))
    for k, v in counts.items():
        table.add_row(f"Events: {k}", str(v))

    console.print(table)

    if err_samples:
        console.print(
            Panel(
                "\n".join(f"- {s}" for s in err_samples),
                title="Error samples",
                border_style="red",
            )
        )


@app.command()
def export(
    fmt: str = typer.Option("jsonl", "--format", help="Export format (jsonl only for now)."),
    source: Optional[Path] = typer.Argument(None, help="Log file path (defaults to stdin)."),
):
    """
    Export parsed events as JSONL.

    Deduplication is ALWAYS on:
    - consecutive duplicates are collapsed
    - `count` tells how many duplicates were collapsed into this row
    """
    if fmt.lower() != "jsonl":
        raise typer.BadParameter("Only --format jsonl is supported in v0.1")

    for ev, n in dedup_consecutive(parse_events(iter_lines_from_source(source))):
        out = {
            "ts": ev.ts,
            "run_id": ev.run_id,
            "type": ev.type.value,
            "name": ev.name,
            "count": n,
            "payload": ev.payload,
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")


# =============================================================================
# Entrypoint
# =============================================================================


def main() -> None:
    app()


if __name__ == "__main__":
    main()
