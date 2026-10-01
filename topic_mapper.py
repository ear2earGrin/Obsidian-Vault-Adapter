#!/usr/bin/env python3
"""
topic_mapper.py — Group the notes already in your vault into topics, locally

vault_builder.py creates notes from source files. This works on the notes that
are already in the vault, whoever wrote them, and gives the graph structure:
every note gets topic tags in its frontmatter, and every topic gets a Map of
Content linking its members. Embedding, clustering and naming all run on this
machine through Ollama, so no note leaves it.

Commands:
  scan     Embed notes, cluster them into topics and name each topic. Writes a
           proposal to <vault>/.topic_mapper/ and changes nothing else.
  apply    Write the approved proposal: tags into frontmatter (note bodies are
           never edited) and one MOC per topic under MOC/Topics/.
  assign   Incremental run: give notes that have no topic yet the closest
           topic from the last apply. Safe to schedule.

Typical flow:
  python topic_mapper.py scan
  python topic_mapper.py apply --dry-run
  python topic_mapper.py apply --limit 1
  python topic_mapper.py apply
  python topic_mapper.py assign --write

Between scan and apply, edit <vault>/.topic_mapper/topics.yaml to rename,
skip or merge topics.

Tags this writes:
  topic/<name>   From clustering. topic_mapper owns this namespace: apply
                 replaces any existing topic/ tags on the notes it edits.
  anything else  From the folder and title rules (topics.rules in
                 config.yaml). These are only ever added, never removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import requests
import yaml
from rich.table import Table

import vault_builder as vb

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is not installed. Run: pip install -r requirements.txt")

log = vb.log
console = vb.console

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

STATE_DIR = ".topic_mapper"
TOPIC_PREFIX = "topic/"
MOC_SUBDIR = "Topics"
INDEX_NAME = "Topics Index"
GENERATED_BY = "topic_mapper"
EMBED_BATCH = 16

# Tags for the adapter's own output folders. Rules for your own folders go in
# config.yaml under topics.rules, which replaces this list. A folder rule
# matches that folder anywhere in a note's path.
DEFAULT_RULES = [
    {"folder": "Documents/ChatGPT", "tag": "type/chatgpt"},
    {"folder": "Documents/PDFs", "tag": "type/document"},
    {"folder": "Documents/Word", "tag": "type/document"},
]

DEFAULT_SETTINGS = {
    "embed_model": "bge-m3",
    "name_model": None,              # falls back to ollama_model
    "granularity": "broad",          # broad: 12-24 topics, fine: 30-60
    "exclude": ["MOC", "Keys"],      # folders skipped anywhere in the path
    "rules": None,                   # None means DEFAULT_RULES
    "embed_chars": 3000,             # characters of each note sent to the embedder
    "min_words": 20,                 # shorter notes are listed, not clustered
    "min_topic_size": 5,             # smaller clusters are dissolved
    "low_confidence_pct": 10,        # weakest fits get no topic tag by default
    "related_min_similarity": 0.5,   # floor for `apply --related`
}


def topic_settings(cfg: dict) -> dict:
    settings = {**DEFAULT_SETTINGS, **(cfg.get("topics") or {})}
    if settings["rules"] is None:
        settings["rules"] = DEFAULT_RULES
    settings["name_model"] = settings["name_model"] or cfg.get("ollama_model", "qwen3:8b")
    return settings


def ollama_base_url(cfg: dict) -> str:
    parts = urlsplit(cfg.get("ollama_endpoint", "http://localhost:11434/v1/chat/completions"))
    return f"{parts.scheme}://{parts.netloc}"


def _in_folder(rel: str, folder: str) -> bool:
    """True when `folder` (one or more path segments) appears anywhere in `rel`."""
    folder = folder.strip("/").lower()
    return bool(folder) and f"/{folder}/" in f"/{rel.lower()}"


# ---------------------------------------------------------------------------
# Reading notes
# ---------------------------------------------------------------------------


@dataclass
class Note:
    rel: str               # vault-relative path, forward slashes
    path: Path
    title: str             # file name without .md, as Obsidian shows it
    text: str              # full text, minus any byte-order mark
    bom: bool
    fm_raw: Optional[str]  # raw YAML between the fences, None if there is none
    fm: dict
    fm_valid: bool         # False when the frontmatter is not a YAML mapping
    body: str              # everything after the closing fence, verbatim
    file_hash: str


def split_frontmatter(text: str) -> tuple[Optional[str], str]:
    """Return (raw frontmatter, body). The body is kept byte for byte."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip() != "---":
        return None, text
    for i in range(1, len(lines)):
        if lines[i].rstrip() in ("---", "..."):
            return "".join(lines[1:i]), "".join(lines[i + 1:])
    return None, text


def read_note(path: Path, vault: Path) -> Optional[Note]:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None

    bom = text.startswith("﻿")
    if bom:
        text = text[1:]

    fm_raw, body = split_frontmatter(text)
    fm: dict = {}
    fm_valid = True
    if fm_raw is not None and fm_raw.strip():
        try:
            parsed = yaml.safe_load(fm_raw)
        except yaml.YAMLError:
            parsed = None
            fm_valid = False
        if isinstance(parsed, dict):
            fm = parsed
        elif parsed is not None:
            fm_valid = False

    return Note(
        rel=path.relative_to(vault).as_posix(),
        path=path,
        title=path.stem,
        text=text,
        bom=bom,
        fm_raw=fm_raw,
        fm=fm,
        fm_valid=fm_valid,
        body=body,
        file_hash=hashlib.sha256(data).hexdigest(),
    )


def discover_notes(vault: Path, settings: dict, moc_dir: str) -> tuple[list[Note], list[str]]:
    """All markdown notes outside dot-folders and excluded folders."""
    excluded = list(settings["exclude"]) + [moc_dir]
    notes: list[Note] = []
    unreadable: list[str] = []

    for path in sorted(vault.rglob("*.md")):
        rel = path.relative_to(vault).as_posix()
        if any(part.startswith(".") for part in rel.split("/")):
            continue
        if any(_in_folder(rel, folder) for folder in excluded):
            continue
        if not path.is_file():
            continue
        note = read_note(path, vault)
        if note is None:
            unreadable.append(rel)
        else:
            notes.append(note)

    return notes, unreadable


# ---------------------------------------------------------------------------
# Tags and rules
# ---------------------------------------------------------------------------


def as_tag_list(value) -> list[str]:
    """Normalise a frontmatter `tags` value (list, string or missing) to a list."""
    if value is None:
        return []
    items = re.split(r"[,\s]+", value) if isinstance(value, str) else value
    if not isinstance(items, list):
        items = [items]
    tags = []
    for item in items:
        tag = str(item).strip().lstrip("#")
        if tag:
            tags.append(tag)
    return tags


def has_topic_tag(note: Note) -> bool:
    return any(t.lower().startswith(TOPIC_PREFIX) for t in as_tag_list(note.fm.get("tags")))


def merged_tags(existing: list[str], additions: list[str]) -> list[str]:
    """Keep existing tags except topic/ ones, then append new tags once each."""
    tags = [t for t in existing if not t.lower().startswith(TOPIC_PREFIX)]
    seen = {t.lower() for t in tags}
    for tag in additions:
        if tag.lower() not in seen:
            tags.append(tag)
            seen.add(tag.lower())
    return tags


def rule_tags(note: Note, rules: list[dict]) -> list[str]:
    tags: list[str] = []
    title = note.title.lower()
    for rule in rules:
        tag = str(rule.get("tag") or "").strip().lstrip("#")
        if not tag or tag in tags:
            continue
        if "folder" in rule and _in_folder(note.rel, str(rule["folder"])):
            tags.append(tag)
        elif "title_prefix" in rule and title.startswith(str(rule["title_prefix"]).lower()):
            tags.append(tag)
        elif "title_contains" in rule and str(rule["title_contains"]).lower() in title:
            tags.append(tag)
    return tags


def sanitize_slug(value: str) -> str:
    """Turn a model's or user's topic name into a tag-safe slug (nesting allowed)."""
    value = value.strip().lower().lstrip("#")
    if value.startswith(TOPIC_PREFIX):
        value = value[len(TOPIC_PREFIX):]
    value = re.sub(r"[^a-z0-9/]+", "-", value)
    parts = [re.sub(r"-{2,}", "-", p).strip("-") for p in value.split("/")]
    return "/".join(p for p in parts if p)[:60]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def clean_body(body: str) -> str:
    body = re.sub(r"!\[\[[^\]]*\]\]", " ", body)               # embeds
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)          # images
    body = re.sub(r"https?://\S+", " ", body)
    body = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", body)  # [[a|b]] -> b
    return re.sub(r"\s+", " ", body).strip()


def note_summary(note: Note) -> str:
    summary = note.fm.get("summary")
    return re.sub(r"\s+", " ", summary).strip() if isinstance(summary, str) else ""


def too_short(note: Note, settings: dict) -> bool:
    return not note_summary(note) and len(clean_body(note.body).split()) < settings["min_words"]


def embed_text(note: Note, settings: dict) -> str:
    """Title, the adapter's summary and key concepts when present, then the body."""
    concepts = note.fm.get("key_concepts")
    concepts = ", ".join(str(c) for c in concepts) if isinstance(concepts, list) else ""
    parts = [note.title, note_summary(note), concepts, clean_body(note.body)]
    return "\n".join(p for p in parts if p)[: settings["embed_chars"]]


def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


class EmbeddingCache:
    """Vectors keyed by a hash of model + embedded text, so unchanged notes are free."""

    def __init__(self, path: Path):
        self.path = path
        self.vectors: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as data:
                for key, vec in zip(data["keys"], data["vectors"]):
                    self.vectors[str(key)] = vec

    def save(self, keep: Optional[set] = None) -> None:
        items = [(k, v) for k, v in self.vectors.items() if keep is None or k in keep]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        if items:
            keys = np.array([k for k, _ in items])
            vectors = np.stack([v for _, v in items]).astype(np.float32)
        else:
            keys = np.array([], dtype=str)
            vectors = np.zeros((0, 0), dtype=np.float32)
        np.savez(tmp, keys=keys, vectors=vectors)
        os.replace(tmp, self.path)


class ModelMissing(RuntimeError):
    pass


def embed_batch(texts: list[str], base_url: str, model: str) -> list:
    for attempt in range(1, 4):
        try:
            resp = requests.post(
                f"{base_url}/api/embed", json={"model": model, "input": texts}, timeout=300
            )
            if resp.status_code == 404:
                body = resp.text.lower()
                if "model" in body and "not found" in body:
                    raise ModelMissing(f"Ollama has no model '{model}'. Run: ollama pull {model}")
                # Ollama before 0.3 only has the single-input route
                return [_embed_one(t, base_url, model) for t in texts]
            resp.raise_for_status()
            vectors = resp.json()["embeddings"]
            if len(vectors) != len(texts):
                raise ValueError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
            return vectors
        except ModelMissing:
            raise
        except (requests.RequestException, ValueError, KeyError) as exc:
            log.warning(f"Embedding attempt {attempt}/3 failed: {exc}")
            if attempt == 3:
                raise RuntimeError(f"Embedding failed after 3 attempts: {exc}") from exc
            time.sleep(3)
    return []


def _embed_one(text: str, base_url: str, model: str) -> list:
    resp = requests.post(
        f"{base_url}/api/embeddings", json={"model": model, "prompt": text}, timeout=300
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_notes(
    notes: list[Note], settings: dict, cfg: dict, cache: EmbeddingCache
) -> tuple[list[Note], np.ndarray, list[str]]:
    """
    Embed `notes`, reusing cached vectors. Returns the notes that embedded
    cleanly with their normalised vectors and cache keys; any note whose
    vector came back non-finite or all zeros is dropped from the cache and
    left out, so it gets a fresh attempt on the next run.
    """
    model = settings["embed_model"]
    texts = [embed_text(n, settings) for n in notes]
    keys = [hashlib.sha256(f"{model}\n{t}".encode("utf-8")).hexdigest() for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in cache.vectors]

    if missing:
        log.info(f"Embedding {len(missing)} note(s) with {model} ({len(notes) - len(missing)} cached).")
        base_url = ollama_base_url(cfg)
        with vb.build_progress() as progress:
            task = progress.add_task("Embedding", total=len(missing))
            for n_batch, start in enumerate(range(0, len(missing), EMBED_BATCH)):
                batch = missing[start : start + EMBED_BATCH]
                vectors = embed_batch([texts[i] for i in batch], base_url, model)
                for i, vec in zip(batch, vectors):
                    cache.vectors[keys[i]] = np.asarray(vec, dtype=np.float32)
                progress.advance(task, len(batch))
                if n_batch % 10 == 9:
                    cache.save()  # checkpoint, so an interrupted run resumes here
    else:
        log.info(f"All {len(notes)} embeddings cached.")

    X = np.stack([cache.vectors[k] for k in keys]).astype(np.float32)
    good = np.isfinite(X).all(axis=1) & (np.abs(X).sum(axis=1) > 0)
    if not good.all():
        bad = np.flatnonzero(~good)
        log.warning(f"{len(bad)} note(s) got an unusable embedding and were left out; rerun to retry them.")
        for i in bad:
            cache.vectors.pop(keys[i], None)
        notes = [n for n, ok in zip(notes, good) if ok]
        keys = [k for k, ok in zip(keys, good) if ok]
        X = X[good]
    return notes, l2_normalize(X), keys


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Matrix product without numpy's floating-point warnings. numpy builds that
    use Apple's Accelerate library can raise spurious divide-by-zero and
    overflow warnings in matmul even for normal inputs, so the result is
    checked directly instead.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = a @ b
    if not np.isfinite(out).all():
        raise FloatingPointError("similarity matrix contains non-finite values")
    return out


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def default_k(n: int, granularity: str) -> int:
    if granularity == "fine":
        return max(30, min(60, round(math.sqrt(n))))
    return max(12, min(24, round(math.sqrt(n / 4))))


def cluster(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        sys.exit("scikit-learn is not installed. Run: pip install -r requirements.txt")
    # Same spurious Accelerate warnings as in _dot; inputs are already checked
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    return km.labels_, l2_normalize(km.cluster_centers_)


_PARENT_LINK = re.compile(r"^\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]$")


def note_families(notes: list[Note]) -> list[list[int]]:
    """
    Group the adapter's split notes with their parent, so a long document or
    conversation split into 40 section notes counts as one item instead of
    forming a topic of its own. Section notes carry `parent: "[[Name]]"`; the
    parent is the note called Name in the same folder. Everything else is a
    family of one.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for i, note in enumerate(notes):
        folder = note.rel.rsplit("/", 1)[0] if "/" in note.rel else ""
        parent = note.fm.get("parent")
        match = _PARENT_LINK.match(parent.strip()) if isinstance(parent, str) else None
        name = match.group(1).split("/")[-1] if match else note.title
        groups.setdefault((folder, name.strip().lower()), []).append(i)
    return list(groups.values())


def related_candidates(
    X: np.ndarray, rels: list[str], family_of: np.ndarray, top: int = 5
) -> list[list]:
    """
    The `top` most similar other notes for each note, as [rel, similarity].
    Sections of the same split document are skipped, since they'd always win.
    """
    top = min(top, len(rels) - 1)
    if top <= 0:
        return [[] for _ in rels]
    out = []
    for start in range(0, len(X), 512):
        block = _dot(X[start : start + 512], X.T)
        for row_i, row in enumerate(block):
            row[family_of == family_of[start + row_i]] = -1.0
            idx = np.argpartition(-row, top - 1)[:top]
            idx = idx[np.argsort(-row[idx])]
            out.append([[rels[j], round(float(row[j]), 4)] for j in idx if row[j] > -1.0])
    return out


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

NAMING_PROMPT = """\
You are organising a personal knowledge base. The notes below were grouped together because their content is similar. Name the subject they share.

Return ONLY a JSON object with exactly these keys:
- "slug": 1-4 lowercase English words joined by hyphens, naming the shared subject (e.g. "bitcoin-market", "ai-agents", "game-design"). Name the subject, never the format: do not use words like "notes", "transcript", "video", "summary" or "document".
- "label": a short human-readable English title for the subject (2-5 words).
- "description": one English sentence saying what these notes have in common.

Names already taken by other topics, so choose something distinct: {taken}

Most representative notes in this group:
{notes}
"""


def name_topics(
    notes: list[Note], topic_of: np.ndarray, fit: np.ndarray, count: int, cfg: dict, settings: dict
) -> list[dict]:
    topics: list[dict] = []
    taken: list[str] = []
    failures = 0

    with vb.build_progress() as progress:
        task = progress.add_task("Naming topics", total=count)
        for tid in range(count):
            members = sorted(np.flatnonzero(topic_of == tid), key=lambda i: -fit[i])
            examples = []
            for i in members[:10]:
                # Titles alone are often uninformative ("Untitled", a date), so
                # fall back to the opening of the body when there's no summary
                line = f"- {notes[i].title}"
                gist = note_summary(notes[i]) or clean_body(notes[i].body)
                if gist:
                    line += f" — {gist[:160]}"
                examples.append(line)

            result = None
            # Stop asking after two straight failures rather than retrying a
            # model that can't answer for every remaining topic.
            if failures < 2:
                prompt = NAMING_PROMPT.format(
                    taken=", ".join(taken) or "none yet", notes="\n".join(examples)
                )
                result = vb.ollama_chat_json(prompt, cfg, model=settings["name_model"])
                failures = 0 if result else failures + 1

            slug = sanitize_slug(str(result.get("slug", ""))) if result else ""
            named = bool(slug)
            slug = slug or f"topic-{tid + 1:02d}"
            base, n = slug, 2
            while slug in taken:
                slug, n = f"{base}-{n}", n + 1
            taken.append(slug)

            label = str(result.get("label", "")).strip() if result else ""
            topics.append({
                "id": tid,
                "tag": TOPIC_PREFIX + slug,
                "label": label or slug.replace("-", " ").title(),
                "description": str(result.get("description", "")).strip() if result else "",
                "named": named,
                "members": [int(i) for i in members],
            })
            progress.advance(task)

    return topics


# ---------------------------------------------------------------------------
# Writing frontmatter
# ---------------------------------------------------------------------------

_LINK_UNSAFE = re.compile(r"[\[\]|#^]")


def make_link(rel: str, title: str) -> Optional[str]:
    """A path-qualified wikilink, so notes with the same name never collide."""
    target = rel[:-3] if rel.lower().endswith(".md") else rel
    if _LINK_UNSAFE.search(target):
        return None
    alias = _LINK_UNSAFE.sub("", title).strip() or target
    return f"[[{target}|{alias}]]"


def _remove_key(lines: list[str], key: str) -> tuple[list[str], Optional[int]]:
    """Drop `key:` and its indented or `- ` continuation lines; return where it was."""
    key_re = re.compile(rf"^{re.escape(key)}\s*:")
    out: list[str] = []
    position: Optional[int] = None
    skipping = False
    for line in lines:
        bare = line.rstrip("\r\n")
        if skipping:
            if bare == "" or bare[0] in " \t" or bare == "-" or bare.startswith("- "):
                continue
            skipping = False
        if key_re.match(bare):
            if position is None:
                position = len(out)
            skipping = True
            continue
        out.append(line)
    return out, position


def update_frontmatter(note: Note, tags: list[str], related: Optional[list[str]]) -> Optional[str]:
    """
    The note's new text with `tags` (and `related`, when given) replaced, or None
    when nothing would change. Every other property keeps its exact original
    lines, and the body is untouched.
    """
    tags_changed = as_tag_list(note.fm.get("tags")) != tags
    related_changed = related is not None and (note.fm.get("related") or []) != related
    if not tags_changed and not related_changed:
        return None

    first = note.text.splitlines(keepends=True)[:1]
    nl = "\r\n" if first and first[0].endswith("\r\n") else "\n"
    lines = (note.fm_raw or "").splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += nl

    changes = []
    if tags_changed:
        changes.append(("tags", tags))
    if related_changed:
        changes.append(("related", related))

    for key, values in changes:
        lines, position = _remove_key(lines, key)
        block = [f"{key}:{nl}"] + [f"  - {json.dumps(v, ensure_ascii=False)}{nl}" for v in values]
        if not values:
            block = []
        if position is None:
            lines.extend(block)
        else:
            lines[position:position] = block

    new_fm = "".join(lines)
    if not new_fm.strip():
        # Nothing left, e.g. a note whose only property was `related`: drop
        # the empty fences too, so the note goes back to having no frontmatter
        return note.body
    return f"---{nl}{new_fm}---{nl}{note.body}"


def verify_update(note: Note, new_text: str, tags: list[str], related: Optional[list[str]]) -> bool:
    """Re-parse the edit: body identical, other properties identical, new values exact."""
    fm_raw, body = split_frontmatter(new_text)
    if body != note.body:
        return False
    try:
        parsed = (yaml.safe_load(fm_raw) or {}) if fm_raw is not None else {}
    except yaml.YAMLError:
        return False
    if not isinstance(parsed, dict) or as_tag_list(parsed.get("tags")) != tags:
        return False
    owned = {"tags"}
    if related is not None:
        if (parsed.get("related") or []) != related:
            return False
        owned.add("related")
    before = {k: v for k, v in note.fm.items() if k not in owned}
    after = {k: v for k, v in parsed.items() if k not in owned}
    return before == after


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file and rename, so a crash never leaves half a note."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tm-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        if path.exists():
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Topic MOCs
# ---------------------------------------------------------------------------


def collect_topic_members(vault: Path, settings: dict, moc_dir: str) -> dict[str, list[tuple[str, str]]]:
    """Read topic/ tags back from the vault, so MOCs match what is really there."""
    notes, _ = discover_notes(vault, settings, moc_dir)
    members: dict[str, list[tuple[str, str]]] = {}
    for note in notes:
        for tag in as_tag_list(note.fm.get("tags")):
            if tag.lower().startswith(TOPIC_PREFIX):
                members.setdefault(tag.lower(), []).append((note.rel, note.title))
    return members


def write_topic_mocs(
    vault: Path, moc_dir: str, members: dict[str, list[tuple[str, str]]], meta: dict[str, dict]
) -> tuple[int, int]:
    folder = vault / moc_dir / MOC_SUBDIR
    folder.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    written: set[str] = set()
    rows: list[tuple[str, str, int]] = []

    for tag in sorted(members, key=lambda t: (-len(members[t]), t)):
        info = meta.get(tag, {})
        slug = tag[len(TOPIC_PREFIX):]
        label = info.get("label") or slug.replace("-", " ").replace("/", " / ").title()
        name = vb._safe_filename(label)
        if f"{name}.md".lower() in written or name.lower() == INDEX_NAME.lower():
            name = vb._safe_filename(f"{label} ({slug.replace('/', ' ')})")
        written.add(f"{name}.md".lower())

        notes = sorted(members[tag], key=lambda m: m[1].lower())
        lines = [
            "---",
            f"title: {json.dumps(label, ensure_ascii=False)}",
            f"topic_tag: {json.dumps(tag)}",
            f"generated_by: {GENERATED_BY}",
            f'generated: "{today}"',
            f"note_count: {len(notes)}",
            "tags:",
            '  - "type/moc"',
            "---",
            "",
            f"# {label}",
            "",
        ]
        if info.get("description"):
            lines += [f"> {info['description']}", ""]
        lines += [
            f"*{len(notes)} notes tagged `#{tag}`. Generated by topic_mapper.py; "
            "edits here are replaced on the next run.*",
            "",
        ]
        lines += [f"- {make_link(rel, title) or title}" for rel, title in notes]
        write_atomic(folder / f"{name}.md", "\n".join(lines) + "\n")
        rows.append((name, label, len(notes)))

    index = [
        "---",
        f'title: "{INDEX_NAME}"',
        f"generated_by: {GENERATED_BY}",
        f'generated: "{today}"',
        "tags:",
        '  - "type/moc"',
        "---",
        "",
        f"# {INDEX_NAME}",
        "",
        f"*{len(rows)} topics. Generated by topic_mapper.py.*",
        "",
    ]
    for name, label, count in rows:
        link = make_link(f"{moc_dir}/{MOC_SUBDIR}/{name}.md", label) or label
        index.append(f"- {link} ({count})")
    write_atomic(folder / f"{INDEX_NAME}.md", "\n".join(index) + "\n")
    written.add(f"{INDEX_NAME}.md".lower())

    # Remove MOCs from topics that no longer exist, but only ones this tool made
    removed = 0
    for path in folder.glob("*.md"):
        if path.name.lower() in written:
            continue
        note = read_note(path, vault)
        if note is not None and note.fm.get("generated_by") == GENERATED_BY:
            path.unlink()
            removed += 1

    return len(rows), removed


# ---------------------------------------------------------------------------
# Credential check
# ---------------------------------------------------------------------------

SECRET_PATTERNS = [
    ("Anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("OpenAI/DeepSeek-style key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{20,}")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("xAI key", re.compile(r"xai-[A-Za-z0-9]{20,}")),
    ("Groq key", re.compile(r"gsk_[A-Za-z0-9]{20,}")),
    ("GitHub token", re.compile(r"(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("JWT (e.g. Supabase key)", re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.eyJ[A-Za-z0-9_-]{20,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Telegram bot token", re.compile(r"\b[0-9]{8,10}:AA[A-Za-z0-9_-]{30,}")),
    ("Private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]


def find_secrets(vault: Path) -> list[tuple[str, list[str]]]:
    """
    Notes and plugin settings that contain credential-shaped strings. Reports the
    kind only, never the value. Covers excluded folders too, since that's where
    keys tend to live.
    """
    paths = [
        p for p in vault.rglob("*.md")
        if not any(part in (".git", ".trash", STATE_DIR) for part in p.relative_to(vault).parts)
    ]
    paths += list((vault / ".obsidian" / "plugins").glob("*/data.json"))

    hits = []
    for path in sorted(paths):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        kinds = [name for name, pattern in SECRET_PATTERNS if pattern.search(text)]
        if kinds:
            hits.append((path.relative_to(vault).as_posix(), kinds))
    return hits


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def _file_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def cmd_scan(args, cfg: dict, vault: Path, settings: dict) -> None:
    state = vault / STATE_DIR
    moc_dir = cfg["folders"]["moc"]
    min_size = int(settings["min_topic_size"])

    console.rule("[bold]Topic Mapper — scan[/bold]")
    console.print(f"  Vault       : [cyan]{vault}[/cyan]")
    console.print(f"  Embeddings  : [cyan]{settings['embed_model']}[/cyan] @ {ollama_base_url(cfg)}")
    console.print(f"  Naming      : [cyan]{settings['name_model']}[/cyan]")
    console.print(f"  Granularity : [cyan]{settings['granularity']}[/cyan]")
    console.print()

    notes, unreadable = discover_notes(vault, settings, moc_dir)
    short = [n for n in notes if too_short(n, settings)]
    usable = [n for n in notes if not too_short(n, settings)]
    log.info(f"Found {len(notes)} notes: {len(usable)} to cluster, {len(short)} too short.")
    if len(usable) < 2 * min_size:
        log.error(f"Need at least {2 * min_size} notes with content to form topics.")
        sys.exit(1)

    cache = EmbeddingCache(state / f"embeddings-{_file_slug(settings['embed_model'])}.npz")
    try:
        usable, X, keys = embed_notes(usable, settings, cfg, cache)
    except (ModelMissing, RuntimeError) as exc:
        log.error(f"{exc}. Embeddings done so far are cached; rerun to continue.")
        sys.exit(1)
    finally:
        cache.save()
    cache.save(keep=set(keys))  # forget notes that were deleted or have changed

    # Cluster whole documents rather than their split sections: each family's
    # vector is the mean of its members, and every member inherits the result.
    families = note_families(usable)
    family_of = np.empty(len(usable), dtype=np.int64)
    for f, members in enumerate(families):
        family_of[members] = f
    U = l2_normalize(np.stack([X[members].mean(axis=0) for members in families]))
    # The parent note (no `parent` property) speaks for its family in naming
    reps = [next((i for i in members if "parent" not in usable[i].fm), members[0]) for members in families]
    split = sum(1 for members in families if len(members) > 1)
    if split:
        log.info(f"Grouped {sum(len(m) for m in families if len(m) > 1)} split notes into {split} documents.")

    n = len(families)
    k = max(2, min(args.k or default_k(n, settings["granularity"]), n // min_size))
    log.info(f"Clustering {n} documents into {k} topics ...")
    labels, centers = cluster(U, k)

    # Dissolve clusters too small to be a topic, renumber the rest by size, and
    # give every document its nearest surviving topic.
    sizes = np.bincount(labels, minlength=k)
    kept = [int(c) for c in np.argsort(-sizes, kind="stable") if sizes[c] >= min_size]
    if not kept:
        log.error("No cluster reached min_topic_size. Lower it in config.yaml or pass a smaller --k.")
        sys.exit(1)
    centers = centers[kept]
    sims = _dot(U, centers.T)
    family_topic = sims.argmax(axis=1)
    family_fit = sims.max(axis=1)
    threshold = float(np.percentile(family_fit, settings["low_confidence_pct"]))

    topic_of = family_topic[family_of]
    fit = family_fit[family_of]
    low = fit < threshold

    related = related_candidates(X, [note.rel for note in usable], family_of)
    topics = name_topics([usable[r] for r in reps], family_topic, family_fit, len(kept), cfg, settings)
    for t in topics:
        # name_topics works on families; expand back to individual notes
        t["examples"] = [usable[reps[f]].title for f in t["members"][:5]]
        t["members"] = [i for f in t["members"] for i in families[f]]

    # ── Proposal files ─────────────────────────────────────────────────────
    state.mkdir(parents=True, exist_ok=True)
    rules = settings["rules"]
    assignment_notes = {}
    for i, note in enumerate(usable):
        assignment_notes[note.rel] = {
            "hash": note.file_hash,
            "topic": int(topic_of[i]),
            "fit": round(float(fit[i]), 4),
            "low_confidence": bool(low[i]),
            "rule_tags": rule_tags(note, rules),
            "related": related[i],
        }
    for note in short:
        assignment_notes[note.rel] = {
            "hash": note.file_hash,
            "topic": None,
            "fit": None,
            "low_confidence": False,
            "too_short": True,
            "rule_tags": rule_tags(note, rules),
            "related": [],
        }

    (state / "assignments.json").write_text(json.dumps({
        "generated": date.today().isoformat(),
        "embed_model": settings["embed_model"],
        "threshold": threshold,
        "notes": assignment_notes,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(state / "scan.npz", centers=centers.astype(np.float32))

    topic_yaml = [{
        "id": t["id"],
        "tag": t["tag"],
        "label": t["label"],
        "description": t["description"],
        "skip": False,
        "notes": len(t["members"]),
        "low_confidence": int(sum(low[i] for i in t["members"])),
        "examples": t["examples"],
    } for t in topics]
    header = (
        "# Proposed topics from `python topic_mapper.py scan`.\n"
        "# Edit this file, then run `python topic_mapper.py apply`.\n"
        "#   rename:  change `tag` (keep the topic/ prefix) and `label`\n"
        "#   drop:    set `skip: true`; its notes get no topic tag\n"
        "#   merge:   give two topics the same `tag`\n"
        "# `notes`, `low_confidence` and `examples` are for reference only.\n\n"
    )
    (state / "topics.yaml").write_text(
        header + yaml.safe_dump({"topics": topic_yaml}, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )

    secrets = find_secrets(vault)
    invalid = [note.rel for note in notes if not note.fm_valid]
    rule_counts = Counter(t for e in assignment_notes.values() for t in e["rule_tags"])
    top1 = [r[0][1] for r in related if r]
    review = _review_markdown(
        vault, settings, usable, short, topics, fit, low, threshold, rule_counts,
        secrets, invalid, unreadable, float(np.median(top1)) if top1 else None,
    )
    (state / "review.md").write_text(review, encoding="utf-8")

    # ── Terminal summary ──────────────────────────────────────────────────
    table = Table(title="Proposed topics", show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Tag")
    table.add_column("Notes", justify="right")
    table.add_column("Examples", overflow="fold")
    for t in topics:
        examples = "; ".join(t["examples"][:3])
        table.add_row(str(t["id"]), t["tag"], str(len(t["members"])), examples)
    console.print()
    console.print(table)

    unnamed = sum(1 for t in topics if not t["named"])
    console.print(f"  Clustered        : {n} notes into {len(topics)} topics")
    console.print(f"  Low confidence   : {int(low.sum())} (no topic tag unless --include-low-confidence)")
    console.print(f"  Too short        : {len(short)}")
    if rule_counts:
        console.print("  Rule tags        : " + ", ".join(f"{t} ({c})" for t, c in rule_counts.most_common()))
    if invalid:
        console.print(f"  [yellow]Invalid YAML     : {len(invalid)} notes will be left untouched[/yellow]")
    if unnamed:
        console.print(
            f"  [yellow]Unnamed topics   : {unnamed} got placeholder names because "
            f"{settings['name_model']} didn't return JSON. Rename them in topics.yaml, or rerun "
            "with --name-model qwen2.5:latest[/yellow]"
        )
    if secrets:
        console.print(f"  [red]Credentials      : {len(secrets)} file(s) contain key-shaped strings[/red]")
        for rel, kinds in secrets:
            console.print(f"     [red]•[/red] {rel}  [dim]({', '.join(kinds)})[/dim]")

    console.print()
    console.print("Nothing in your notes was changed. Review, then edit topic names:")
    console.print(f'  open -e "{state / "review.md"}"')
    console.print(f'  open -e "{state / "topics.yaml"}"')
    console.print("Then: [bold]python topic_mapper.py apply --dry-run[/bold]")


def _review_markdown(
    vault, settings, usable, short, topics, fit, low, threshold, rule_counts,
    secrets, invalid, unreadable, median_top1,
) -> str:
    lines = [
        f"# Topic proposal — {date.today().isoformat()}",
        "",
        f"Vault: `{vault}`  ",
        f"Embeddings: `{settings['embed_model']}` · naming: `{settings['name_model']}`  ",
        f"Notes clustered: {len(usable)} · topics: {len(topics)} · "
        f"low confidence: {int(low.sum())} (fit below {threshold:.3f}) · too short: {len(short)}",
        "",
    ]
    if median_top1 is not None:
        lines += [
            f"Median similarity between a note and its closest neighbour: {median_top1:.3f}. "
            f"`apply --related N` only links pairs at or above {settings['related_min_similarity']} "
            "(topics.related_min_similarity in config.yaml).",
            "",
        ]

    lines += ["## Topics", ""]
    for t in topics:
        lines += [f"### {t['id']}. `{t['tag']}` — {t['label']} ({len(t['members'])} notes)", ""]
        if t["description"]:
            lines += [f"> {t['description']}", ""]
        for i in t["members"]:
            flag = "  *(low confidence)*" if low[i] else ""
            lines.append(f"- {usable[i].rel}  `{fit[i]:.3f}`{flag}")
        lines.append("")

    if rule_counts:
        lines += ["## Folder and title rules", ""]
        lines += [f"- `{tag}`: {count} notes" for tag, count in rule_counts.most_common()]
        lines.append("")
    if short:
        lines += ["## Too short to classify", ""] + [f"- {n.rel}" for n in short] + [""]
    if invalid:
        lines += ["## Invalid frontmatter (will not be edited)", ""] + [f"- {r}" for r in invalid] + [""]
    if unreadable:
        lines += ["## Not UTF-8 (skipped)", ""] + [f"- {r}" for r in unreadable] + [""]
    if secrets:
        lines += ["## Possible credentials (values not shown)", ""]
        lines += [f"- {rel} — {', '.join(kinds)}" for rel, kinds in secrets]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def load_proposal(state: Path) -> tuple[dict[int, dict], dict]:
    topics_path, assign_path = state / "topics.yaml", state / "assignments.json"
    if not topics_path.exists() or not assign_path.exists():
        log.error("No proposal found. Run: python topic_mapper.py scan")
        sys.exit(1)
    try:
        data = yaml.safe_load(topics_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        log.error(f"{topics_path} is not valid YAML after editing: {exc}")
        sys.exit(1)

    topics: dict[int, dict] = {}
    for entry in data.get("topics") or []:
        try:
            tid = int(entry["id"])
        except (KeyError, TypeError, ValueError):
            continue
        slug = sanitize_slug(str(entry.get("tag") or ""))
        topics[tid] = {
            "id": tid,
            "tag": TOPIC_PREFIX + slug if slug else "",
            "label": str(entry.get("label") or slug.replace("-", " ").title()).strip(),
            "description": str(entry.get("description") or "").strip(),
            "skip": bool(entry.get("skip", False)) or not slug,
        }
    return topics, json.loads(assign_path.read_text(encoding="utf-8"))


def _topic_meta(topics) -> dict[str, dict]:
    """Label and description per tag; the first one wins when topics are merged."""
    meta: dict[str, dict] = {}
    for t in sorted(topics, key=lambda t: t["id"]):
        if not t["skip"] and t["tag"]:
            meta.setdefault(t["tag"].lower(), t)
    return meta


def _print_problems(problems: list[tuple[str, str]]) -> None:
    if not problems:
        return
    console.print(f"  [yellow]Skipped {len(problems)} note(s):[/yellow]")
    for rel, reason in problems[:20]:
        console.print(f"     [yellow]•[/yellow] {rel}  [dim]({reason})[/dim]")
    if len(problems) > 20:
        console.print(f"     … and {len(problems) - 20} more")


def cmd_apply(args, cfg: dict, vault: Path, settings: dict) -> None:
    state = vault / STATE_DIR
    moc_dir = cfg["folders"]["moc"]
    topics, assignment = load_proposal(state)
    min_sim = float(settings["related_min_similarity"])

    mode = "dry run" if args.dry_run else (f"first {args.limit} note(s)" if args.limit else "write")
    console.rule(f"[bold]Topic Mapper — apply ({mode})[/bold]")

    counts: Counter = Counter()
    problems: list[tuple[str, str]] = []
    samples: list[tuple[str, str, str]] = []

    for rel, entry in sorted(assignment["notes"].items()):
        if args.limit and counts["changed"] >= args.limit:
            break
        path = vault / rel
        if not path.is_file():
            counts["missing"] += 1
            continue
        note = read_note(path, vault)
        if note is None:
            problems.append((rel, "not valid UTF-8"))
            continue
        if note.file_hash != entry["hash"]:
            problems.append((rel, "changed since scan; rerun scan to include it"))
            continue
        if not note.fm_valid:
            problems.append((rel, "frontmatter is not valid YAML"))
            continue

        additions = list(entry.get("rule_tags") or [])
        topic = topics.get(entry["topic"]) if entry.get("topic") is not None else None
        if topic and not topic["skip"] and (args.include_low_confidence or not entry.get("low_confidence")):
            additions.append(topic["tag"])
        tags = merged_tags(as_tag_list(note.fm.get("tags")), additions)

        related = None
        if args.clear_related:
            related = []  # an empty list removes the property
        elif args.related > 0:
            related = []
            for other, score in entry.get("related") or []:
                if score < min_sim or len(related) >= args.related:
                    break
                link = make_link(other, Path(other).stem)
                if link:
                    related.append(link)

        new_text = update_frontmatter(note, tags, related)
        if new_text is None:
            counts["unchanged"] += 1
            continue
        if not verify_update(note, new_text, tags, related):
            problems.append((rel, "edit could not be verified, left untouched"))
            continue

        counts["changed"] += 1
        if args.dry_run:
            if len(samples) < 3:
                samples.append((rel, note.fm_raw or "", split_frontmatter(new_text)[0] or ""))
            continue
        output = ("﻿" if note.bom else "") + new_text
        write_atomic(path, output)
        # Record the new hash so a later apply (after --limit) doesn't see our
        # own edit as "changed since scan"
        entry["hash"] = hashlib.sha256(output.encode("utf-8")).hexdigest()

    if counts["changed"] and not args.dry_run:
        (state / "assignments.json").write_text(
            json.dumps(assignment, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    if samples:
        for rel, before, after in samples:
            console.print(f"\n[bold]{rel}[/bold]")
            console.print("[dim]before:[/dim]")
            console.print(before.rstrip() or "  (no frontmatter)", markup=False, highlight=False)
            console.print("[dim]after:[/dim]")
            console.print(after.rstrip(), markup=False, highlight=False)

    verb = "Would change" if args.dry_run else "Changed"
    console.print()
    console.print(f"  {verb:<16}: {counts['changed']} notes")
    console.print(f"  Already up to date: {counts['unchanged']}")
    if counts["missing"]:
        console.print(f"  Missing          : {counts['missing']} (moved or deleted since scan)")
    _print_problems(problems)

    if args.dry_run:
        console.print("\nDry run: nothing was written. Next: [bold]python topic_mapper.py apply --limit 1[/bold]")
        return
    if args.limit:
        console.print(
            "\nOpen the changed note in Obsidian and check its properties. "
            "MOCs are only generated on a full run: [bold]python topic_mapper.py apply[/bold]"
        )
        return

    meta = _topic_meta(topics.values())
    members = collect_topic_members(vault, settings, moc_dir)
    count, removed = write_topic_mocs(vault, moc_dir, members, meta)
    console.print(f"  Topic MOCs       : {count} in {moc_dir}/{MOC_SUBDIR}/" + (f" ({removed} stale removed)" if removed else ""))

    # Remember the applied topics so `assign` can place new notes later
    with np.load(state / "scan.npz") as scan:
        np.savez(state / "centroids.npz", centers=scan["centers"])
    (state / "applied.json").write_text(json.dumps({
        "applied": date.today().isoformat(),
        "embed_model": assignment["embed_model"],
        "threshold": assignment["threshold"],
        "topics": [topics[tid] for tid in sorted(topics)],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(f"[green]Done.[/green] Open {moc_dir}/{MOC_SUBDIR}/{INDEX_NAME} in Obsidian to browse the topics.")


# ---------------------------------------------------------------------------
# assign
# ---------------------------------------------------------------------------


def cmd_assign(args, cfg: dict, vault: Path, settings: dict) -> None:
    state = vault / STATE_DIR
    moc_dir = cfg["folders"]["moc"]
    applied_path, centroids_path = state / "applied.json", state / "centroids.npz"
    if not applied_path.exists() or not centroids_path.exists():
        log.error("No applied topics yet. Run scan and apply first.")
        sys.exit(1)

    applied = json.loads(applied_path.read_text(encoding="utf-8"))
    if applied["embed_model"] != settings["embed_model"]:
        log.error(
            f"Topics were built with {applied['embed_model']} but embed_model is now "
            f"{settings['embed_model']}. Run a full scan and apply."
        )
        sys.exit(1)
    with np.load(centroids_path) as saved:
        centers = saved["centers"]
    topics = {int(t["id"]): t for t in applied["topics"]}
    threshold = float(applied["threshold"])

    console.rule(f"[bold]Topic Mapper — assign ({'write' if args.write else 'dry run'})[/bold]")
    notes, _ = discover_notes(vault, settings, moc_dir)
    candidates = [n for n in notes if n.fm_valid and not has_topic_tag(n)]
    if not candidates:
        console.print("[green]Every note already has a topic.[/green]")
        return

    usable = [n for n in candidates if not too_short(n, settings)]
    plan: list[tuple[Note, Optional[str], Optional[float]]] = []
    if usable:
        cache = EmbeddingCache(state / f"embeddings-{_file_slug(settings['embed_model'])}.npz")
        try:
            usable, X, _ = embed_notes(usable, settings, cfg, cache)
        except (ModelMissing, RuntimeError) as exc:
            log.error(str(exc))
            sys.exit(1)
        finally:
            cache.save()
        sims = _dot(X, centers.T)
        for note, best, score in zip(usable, sims.argmax(axis=1), sims.max(axis=1)):
            topic = topics.get(int(best))
            ok = topic and not topic.get("skip") and topic.get("tag") and score >= threshold
            plan.append((note, topic["tag"] if ok else None, float(score)))
    plan += [(n, None, None) for n in candidates if too_short(n, settings)]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Note", overflow="fold")
    table.add_column("Topic")
    table.add_column("Fit", justify="right")
    for note, tag, score in plan:
        table.add_row(note.rel, tag or "[dim]none[/dim]", f"{score:.3f}" if score is not None else "short")
    console.print(table)

    if not args.write:
        console.print("Dry run: nothing written. Add --write to tag these notes.")
        return

    changed, problems = 0, []
    for note, tag, _ in plan:
        additions = rule_tags(note, settings["rules"]) + ([tag] if tag else [])
        tags = merged_tags(as_tag_list(note.fm.get("tags")), additions)
        new_text = update_frontmatter(note, tags, None)
        if new_text is None:
            continue
        if not verify_update(note, new_text, tags, None):
            problems.append((note.rel, "edit could not be verified, left untouched"))
            continue
        write_atomic(note.path, ("﻿" if note.bom else "") + new_text)
        changed += 1

    if changed:
        members = collect_topic_members(vault, settings, moc_dir)
        write_topic_mocs(vault, moc_dir, members, _topic_meta(applied["topics"]))
    console.print(f"  Changed: {changed} notes")
    _print_problems(problems)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    common.add_argument("--vault", help="Obsidian vault root (overrides config.yaml)")

    parser = argparse.ArgumentParser(
        description="Group existing vault notes into topics with local models.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", parents=[common], help="Propose topics; changes nothing")
    scan.add_argument("--k", type=int, help="Exact number of topics (default: based on note count)")
    scan.add_argument("--granularity", choices=["broad", "fine"], help="broad: 12-24 topics, fine: 30-60")
    scan.add_argument("--name-model", help="Ollama model that names topics (default: ollama_model)")

    apply = sub.add_parser("apply", parents=[common], help="Write the approved proposal")
    apply.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    apply.add_argument("--limit", type=int, help="Only edit the first N notes, as a test")
    apply.add_argument("--related", type=int, default=0, metavar="N",
                       help="Also add up to N similar notes as a `related` property")
    apply.add_argument("--clear-related", action="store_true",
                       help="Remove the `related` property that --related added")
    apply.add_argument("--include-low-confidence", action="store_true",
                       help="Tag weak fits with their closest topic too")

    assign = sub.add_parser("assign", parents=[common], help="Tag new notes with existing topics")
    assign.add_argument("--write", action="store_true", help="Write the tags (default is a dry run)")

    args = parser.parse_args()
    if args.command == "apply" and args.clear_related and args.related:
        parser.error("use either --related N or --clear-related, not both")

    config_path = Path(args.config) if args.config else None
    cfg = vb.load_config(config_path, None, args.vault)
    vault = Path(cfg["vault_path"]).expanduser().resolve()
    if not vault.is_dir():
        log.error(f"Vault folder not found: {vault}")
        sys.exit(1)

    settings = topic_settings(cfg)
    if getattr(args, "granularity", None):
        settings["granularity"] = args.granularity
    if getattr(args, "name_model", None):
        settings["name_model"] = args.name_model

    {"scan": cmd_scan, "apply": cmd_apply, "assign": cmd_assign}[args.command](args, cfg, vault, settings)


if __name__ == "__main__":
    main()
