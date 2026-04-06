#!/usr/bin/env python3
"""
cleanup.py — Post-processing cleanup for Obsidian Vault Adapter

What this does:
  1. Finds all ChatGPT notes that were incorrectly split into section files
  2. Deletes the orphaned section files
  3. Removes the split conversations from processed.json so they get
     re-processed as proper single notes on the next run
  4. Reports a summary of what was cleaned

Run:
  python3 cleanup.py --vault /path/to/vault
  python3 cleanup.py --vault /path/to/vault --dry-run   (preview only)
"""

import argparse
import json
import re
import sys
from pathlib import Path

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
DIM    = "\033[2m"


def ok(text):   print(f"  \033[32m✓\033[0m  {text}")
def info(text): print(f"  \033[2m·\033[0m  {text}")
def warn(text): print(f"  \033[33m⚠\033[0m  {text}")
def err(text):  print(f"  \033[31m✗\033[0m  {text}")


def read_frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter from a markdown file."""
    try:
        content = path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return {}
        end = content.find("\n---", 4)
        if end == -1:
            return {}
        fm_text = content[4:end]
        result = {}
        for line in fm_text.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                result[k.strip()] = v.strip().strip('"')
        return result
    except Exception:
        return {}


def find_split_parents(chatgpt_folder: Path) -> list[Path]:
    """Find all parent index notes (ChatGPT notes that were split)."""
    parents = []
    for path in chatgpt_folder.glob("*.md"):
        fm = read_frontmatter(path)
        if fm.get("type") == "index" and fm.get("file_type") == "chatgpt":
            parents.append(path)
    return sorted(parents)


def find_section_files(parent: Path, chatgpt_folder: Path) -> list[Path]:
    """Find all section files belonging to a parent note."""
    stem = parent.stem
    sections = []
    for path in chatgpt_folder.glob("*.md"):
        if path == parent:
            continue
        # Section files are named: "{parent_stem} — {section_title}.md"
        if path.stem.startswith(stem + " \u2014 ") or path.stem.startswith(stem + " - "):
            sections.append(path)
    return sorted(sections)


def remove_from_state(state: dict, source_filename: str) -> list[str]:
    """Remove all state entries whose source matches this filename."""
    removed = []
    keys_to_remove = []
    for key, val in state.items():
        source = val.get("source", "")
        # Match by conversation ID embedded in the fake path
        if source_filename in source or f"chatgpt__{source_filename}" in source:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del state[key]
        removed.append(key)
    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Clean up split ChatGPT notes from Obsidian vault."
    )
    parser.add_argument("--vault", required=True, help="Obsidian vault root path")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no changes made")
    args = parser.parse_args()

    vault_path = Path(args.vault).expanduser().resolve()
    if not vault_path.exists():
        err(f"Vault not found: {vault_path}")
        sys.exit(1)

    chatgpt_folder = vault_path / "Documents" / "ChatGPT"
    if not chatgpt_folder.exists():
        err(f"ChatGPT folder not found: {chatgpt_folder}")
        sys.exit(1)

    state_file = vault_path / "processed.json"
    state = {}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    print(f"\n{BOLD}Obsidian Vault Adapter — Cleanup{RESET}")
    if args.dry_run:
        print(f"  {YELLOW}DRY RUN — no files will be deleted{RESET}")
    print(f"  Vault: {vault_path}\n")

    # Find all split parent notes
    parents = find_split_parents(chatgpt_folder)
    print(f"  Found {len(parents)} split ChatGPT note(s) to clean up.\n")

    if not parents:
        ok("Nothing to clean up.")
        return

    total_sections_deleted = 0
    total_state_cleared = 0
    conversations_to_reprocess = []

    for parent in parents:
        sections = find_section_files(parent, chatgpt_folder)
        fm = read_frontmatter(parent)
        conv_id = fm.get("source", parent.stem)

        info(f"{parent.name} → {len(sections)} section file(s)")

        if args.dry_run:
            for s in sections:
                print(f"       would delete: {s.name}")
            print(f"       would reset state for: {conv_id}")
            conversations_to_reprocess.append(parent.stem)
            continue

        # Delete section files
        for s in sections:
            try:
                s.unlink()
                total_sections_deleted += 1
            except Exception as e:
                err(f"Could not delete {s.name}: {e}")

        # Delete parent index note too — will be recreated as proper single note
        try:
            parent.unlink()
        except Exception as e:
            err(f"Could not delete parent {parent.name}: {e}")

        # Clear from processed.json so it gets re-enriched
        # Match by conversation_id in state entries
        removed = []
        for key in list(state.keys()):
            val = state[key]
            notes = val.get("notes", [])
            if any(parent.stem in n for n in notes):
                del state[key]
                removed.append(key)
                total_state_cleared += 1

        conversations_to_reprocess.append(parent.stem)
        ok(f"Cleaned: {parent.stem[:70]}")

    if not args.dry_run:
        # Save updated state
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

        print(f"\n  {BOLD}Summary{RESET}")
        print(f"  Section files deleted : {total_sections_deleted}")
        print(f"  State entries cleared : {total_state_cleared}")
        print(f"  Conversations queued  : {len(conversations_to_reprocess)}")
        print(f"\n  {GREEN}Next step:{RESET} run the vault adapter again to re-process")
        print(f"  these {len(conversations_to_reprocess)} conversations as single notes.\n")
    else:
        print(f"\n  {YELLOW}Dry run complete.{RESET}")
        print(f"  {len(parents)} parent notes and their sections would be cleaned.")
        print(f"  Run without --dry-run to apply.\n")


if __name__ == "__main__":
    main()
