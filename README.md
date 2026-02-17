# ocx

> A CLI companion for OpenClaw logs.

[PyPI](https://pypi.org/project/ocx-tail/)

`ocx` turns noisy `openclaw logs` output into a structured, high-signal
stream.

It is designed for: 
- debugging gateway conflicts 
- tracking agent run lifecycles 
- spotting provider rate limits & connection failures 
-reducing log spam during live sessions 
- exporting normalized events for further analysis

------------------------------------------------------------------------

## Features

-   Pretty `tail` for OpenClaw logs
-   Error-only filtering
-   Hide websocket noise (`gateway/ws`)
-   Automatic de-duplication of repeated identical errors
-   Run-level summary (`analyze`)
-   JSONL export (`export`)
-   Clean internal event model (extensible, rule-based classifier)

------------------------------------------------------------------------

## Installation

### Recommended: pipx

``` bash
pip install --user pipx
pipx ensurepath
```

Then:

``` bash
pipx install ocx-tail
```

Verify:

``` bash
ocx --help
```

------------------------------------------------------------------------

## Usage

### Tail OpenClaw logs

``` bash
openclaw logs --follow | ocx tail
```

Or:

``` bash
ocx tail --follow
```

### Errors only

``` bash
openclaw logs --follow | ocx tail --errors-only
```

### Disable duplicate collapsing

``` bash
openclaw logs --follow | ocx tail --no-dedup
```

------------------------------------------------------------------------

## Analyze a saved log file

``` bash
ocx analyze /tmp/openclaw/openclaw-2026-02-17.log
```

------------------------------------------------------------------------

## Export normalized JSONL

``` bash
openclaw logs --follow | ocx export > events.jsonl
```

------------------------------------------------------------------------

## Event Model

All log lines are normalized into:

``` python
Event(
    ts: Optional[str],
    run_id: Optional[str],
    type: EventType,
    name: str,
    payload: Dict[str, object],
)
```

High-signal classifications include:

-   `gateway.conflict`
-   `run.start` / `run.done` / `run.timeout`
-   `provider.connection_error`
-   `provider.rate_limit`

------------------------------------------------------------------------

## Architecture

The codebase is intentionally structured:

-   Parsing (regex + classification)
-   IO (stdin, file, openclaw subprocess)
-   Presentation (Rich formatting)
-   CLI wiring (Typer)

------------------------------------------------------------------------

## License

MIT
