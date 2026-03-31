#!/usr/bin/env python3
"""
app.py — Local web UI for Obsidian Vault Adapter

Run:  python app.py
Open: http://localhost:8080
"""

import asyncio
import json
import logging
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Patch the vault_builder logger so its output feeds into our SSE queue
# ---------------------------------------------------------------------------
import vault_builder as vb

app = FastAPI(title="Obsidian Vault Adapter")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Active run state: job_id → {"queue": Queue, "done": bool}
_runs: dict[str, dict] = {}


class QueueHandler(logging.Handler):
    """Forwards log records into a queue for SSE streaming."""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        level = record.levelname.lower()
        self.q.put({"type": "log", "level": level, "msg": self.format(record)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.get("/config")
async def get_config():
    """Return current config.yaml so the UI can pre-fill the form."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return {
            "source_path": cfg.get("source_path", ""),
            "vault_path": cfg.get("vault_path", ""),
            "model": cfg.get("lm_studio", {}).get("model", "qwen3"),
            "endpoint": cfg.get("lm_studio", {}).get("endpoint", "http://localhost:1234/v1/chat/completions"),
            "batch_size": cfg.get("batch_size", 10),
            "enrichment_word_limit": cfg.get("enrichment_word_limit", 2000),
            "split_word_threshold": cfg.get("split_word_threshold", 5000),
        }
    return {}


@app.post("/run")
async def start_run(request: Request):
    """Accept form params, start the pipeline in a background thread, return a job ID."""
    body = await request.json()

    source = body.get("source_path", "").strip()
    vault = body.get("vault_path", "").strip()
    model = body.get("model", "qwen3").strip()
    endpoint = body.get("endpoint", "http://localhost:1234/v1/chat/completions").strip()
    batch_size = int(body.get("batch_size", 10))
    enrichment_word_limit = int(body.get("enrichment_word_limit", 2000))
    split_word_threshold = int(body.get("split_word_threshold", 5000))
    dry_run = bool(body.get("dry_run", False))

    if not source or not vault:
        return {"error": "source_path and vault_path are required"}

    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    _runs[job_id] = {"queue": q, "done": False}

    cfg = {
        "source_path": source,
        "vault_path": vault,
        "lm_studio": {
            "endpoint": endpoint,
            "model": model,
            "timeout_seconds": 120,
            "max_retries": 3,
            "retry_delay_seconds": 5,
        },
        "batch_size": batch_size,
        "enrichment_word_limit": enrichment_word_limit,
        "split_word_threshold": split_word_threshold,
        "folders": {
            "inbox": "Inbox",
            "pdfs": "Documents/PDFs",
            "word": "Documents/Word",
            "moc": "MOC",
        },
        "state_file": "processed.json",
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, cfg, dry_run, q),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream(job_id: str):
    """SSE endpoint — streams log messages for a running job."""
    if job_id not in _runs:
        return HTMLResponse("Unknown job", status_code=404)

    return StreamingResponse(
        _event_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in _runs:
        return {"error": "unknown job"}
    return {"done": _runs[job_id]["done"]}


# ---------------------------------------------------------------------------
# Pipeline runner (runs in a background thread)
# ---------------------------------------------------------------------------


def _run_pipeline(job_id: str, cfg: dict, dry_run: bool, q: queue.Queue) -> None:
    # Attach a queue handler to the vault_builder logger for this run
    handler = QueueHandler(q)
    handler.setFormatter(logging.Formatter("%(message)s"))
    vb_log = logging.getLogger("vault_builder")
    vb_log.addHandler(handler)

    try:
        source_path = Path(cfg["source_path"]).expanduser().resolve()
        vault_path = Path(cfg["vault_path"]).expanduser().resolve()

        if not source_path.exists():
            q.put({"type": "log", "level": "error", "msg": f"Source path not found: {source_path}"})
            return

        vault_path.mkdir(parents=True, exist_ok=True)
        state_file = vault_path / cfg["state_file"]
        state = vb.load_state(state_file)

        q.put({"type": "log", "level": "info", "msg": f"Scanning {source_path} ..."})
        queue_ = vb.scan_files(source_path, state)

        if not queue_:
            q.put({"type": "log", "level": "info", "msg": "Nothing new to process. Vault is up to date."})
            q.put({"type": "done", "stats": {"found": 0, "written": 0, "total_state": len(state)}})
            return

        q.put({"type": "log", "level": "info", "msg": f"Found {len(queue_)} file(s) to process."})
        vault_paths = vb.ensure_vault_structure(vault_path, cfg["folders"])
        all_enriched: list[vb.EnrichedDoc] = []
        batch_size = cfg["batch_size"]

        for i, file_path in enumerate(queue_, 1):
            q.put({"type": "progress", "current": i, "total": len(queue_), "file": file_path.name})
            q.put({"type": "log", "level": "info", "msg": f"[{i}/{len(queue_)}] Extracting {file_path.name}"})

            try:
                doc = vb.extract(file_path)
            except Exception as exc:
                q.put({"type": "log", "level": "error", "msg": f"Extraction failed: {exc}"})
                continue

            if dry_run:
                q.put({
                    "type": "log", "level": "info",
                    "msg": f"  dry-run — {doc.word_count} words, {doc.page_count} pages"
                })
                continue

            q.put({"type": "log", "level": "info", "msg": f"  Enriching with {cfg['lm_studio']['model']} ..."})
            try:
                enriched = vb.enrich(doc, cfg)
            except Exception as exc:
                q.put({"type": "log", "level": "warning", "msg": f"  Enrichment failed: {exc}. Using empty metadata."})
                enriched = vb.EnrichedDoc(extracted=doc, inferred_title=doc.title)

            try:
                written = vb.write_note(enriched, vault_paths, cfg)
                all_enriched.append(enriched)
                h = vb.file_hash(file_path)
                from datetime import date
                state[h] = {
                    "source": str(file_path),
                    "notes": [str(p) for p in written],
                    "processed_date": date.today().isoformat(),
                }
                vb.save_state(state_file, state)
                q.put({"type": "log", "level": "info", "msg": f"  Written: {', '.join(p.name for p in written)}"})
            except Exception as exc:
                q.put({"type": "log", "level": "error", "msg": f"  Write failed: {exc}"})

        if not dry_run and all_enriched:
            q.put({"type": "log", "level": "info", "msg": "Regenerating MOC files ..."})
            vb.generate_moc_all(all_enriched, vault_paths["moc"])
            vb.generate_moc_tags(all_enriched, vault_paths["moc"])

        q.put({
            "type": "done",
            "stats": {
                "found": len(queue_),
                "written": len(all_enriched),
                "total_state": len(state),
            },
        })

    except Exception as exc:
        q.put({"type": "log", "level": "error", "msg": f"Unexpected error: {exc}"})
        q.put({"type": "done", "stats": {}})
    finally:
        vb_log.removeHandler(handler)
        _runs[job_id]["done"] = True


async def _event_generator(job_id: str) -> AsyncGenerator[str, None]:
    run = _runs[job_id]
    q: queue.Queue = run["queue"]

    while True:
        try:
            msg = q.get(timeout=0.1)
            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("type") == "done":
                break
        except queue.Empty:
            if run["done"]:
                break
            # Keepalive ping
            yield ": ping\n\n"
            await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    port = 8080
    print(f"\n  Obsidian Vault Adapter")
    print(f"  Open: http://localhost:{port}\n")

    Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
