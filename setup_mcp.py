#!/usr/bin/env python3
"""
setup_mcp.py — Wire your Obsidian vault into Claude via MCP

What this does:
  1. Installs qmd  (hybrid BM25 + vector semantic search over your vault)
  2. Installs obsidian-mcp  (read/write access to vault notes)
  3. Indexes your vault with qmd so semantic search works immediately
  4. Writes the MCP server config into ~/.claude/settings.json

Run:
  python3 setup_mcp.py

After this, every Claude Code session can search and read your vault.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# ── Colours ─────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
YELLOW = "\033[33m"
RED   = "\033[31m"
DIM   = "\033[2m"


def h(text):  print(f"\n{BOLD}{CYAN}{text}{RESET}")
def ok(text): print(f"  {GREEN}✓{RESET}  {text}")
def info(text): print(f"  {DIM}·{RESET}  {text}")
def warn(text): print(f"  {YELLOW}⚠{RESET}  {text}")
def err(text):  print(f"  {RED}✗{RESET}  {text}")


def ask(prompt, default=""):
    display_default = f" [{default}]" if default else ""
    try:
        val = input(f"  {BOLD}{prompt}{RESET}{display_default}: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)
    return val or default


def run(cmd: list[str], capture=False, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=cwd,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def check_node() -> bool:
    r = run(["node", "--version"], capture=True)
    if r.returncode == 0:
        ok(f"Node.js {r.stdout.strip()}")
        return True
    err("Node.js not found. Install it with: brew install node")
    return False


def check_npx() -> bool:
    r = run(["npx", "--version"], capture=True)
    if r.returncode == 0:
        ok(f"npx {r.stdout.strip()}")
        return True
    err("npx not found.")
    return False


def install_qmd() -> bool:
    h("Installing qmd (hybrid search engine)")
    info("Running: npm install -g @tobilu/qmd")
    r = run(["npm", "install", "-g", "@tobilu/qmd"])
    if r.returncode == 0:
        ok("qmd installed")
        return True
    err("qmd install failed. Try manually: npm install -g @tobilu/qmd")
    return False


def index_vault(vault_path: Path) -> bool:
    h("Indexing vault with qmd")
    info(f"Adding collection: {vault_path}")

    # Add vault as a qmd collection named 'vault'
    r = run(["qmd", "collection", "add", "vault", str(vault_path)], capture=True)
    if r.returncode != 0:
        # Collection may already exist
        if "already" in (r.stderr or "").lower() or "exists" in (r.stderr or "").lower():
            ok("Collection 'vault' already registered")
        else:
            warn(f"qmd collection add: {r.stderr.strip() or 'unknown error'}")

    # Trigger initial index build
    info("Building search index (this may take a minute for large vaults)...")
    r = run(["qmd", "search", "test"], capture=True)
    if r.returncode == 0:
        ok("Index built and verified")
        return True
    else:
        warn("Index may still be building in the background — that's fine.")
        return True


def write_mcp_config(vault_path: Path, api_key: str) -> None:
    h("Writing MCP config to ~/.claude/settings.json")

    settings_path = Path.home() / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing settings
    if settings_path.exists():
        with open(settings_path) as f:
            try:
                settings = json.load(f)
            except json.JSONDecodeError:
                settings = {}
    else:
        settings = {}

    # Build MCP server entries
    mcp_servers = settings.get("mcpServers", {})

    # qmd — hybrid semantic search
    mcp_servers["qmd"] = {
        "command": "qmd",
        "args": ["mcp"],
        "env": {
            "HOME": str(Path.home())
        }
    }

    # obsidian-mcp — read/write vault notes
    obsidian_entry: dict = {
        "command": "npx",
        "args": ["-y", "obsidian-mcp", str(vault_path)],
    }
    if api_key:
        obsidian_entry["env"] = {"OBSIDIAN_API_KEY": api_key}

    mcp_servers["obsidian"] = obsidian_entry

    settings["mcpServers"] = mcp_servers

    # Back up existing settings before overwriting
    if settings_path.exists():
        backup = settings_path.with_suffix(".json.bak")
        backup.write_text(settings_path.read_text())
        info(f"Backed up existing settings → {backup}")

    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    ok(f"Written: {settings_path}")
    info("Active MCP servers:")
    for name in mcp_servers:
        print(f"       • {name}")


def print_obsidian_plugin_instructions(api_key_provided: bool) -> None:
    h("Obsidian setup (one-time)")
    print(f"""
  obsidian-mcp uses Obsidian's {BOLD}Local REST API{RESET} community plugin
  to read and write notes while Obsidian is open.

  Steps:
    1. Open Obsidian → Settings → Community Plugins → Browse
    2. Search for {BOLD}Local REST API{RESET} and install it
    3. Enable it and copy the API key it shows you
    4. Re-run this script with the key when prompted, OR add it manually:

       {DIM}"env": {{"OBSIDIAN_API_KEY": "your-key-here"}}{RESET}

       in the "obsidian" block of ~/.claude/settings.json
""")
    if not api_key_provided:
        warn("No API key provided — obsidian-mcp will work in read-only mode until you add it.")


def print_usage_guide(vault_path: Path) -> None:
    h("You're wired in. Here's how to use it.")
    print(f"""
  In any Claude Code session, Claude can now:

  {BOLD}Search your vault semantically{RESET}
    "Find everything in my vault about prompt injection"
    "What notes do I have on Bitcoin self-custody?"

  {BOLD}Retrieve specific notes{RESET}
    "Get the note called [[MOC-All]] from my vault"
    "Read the summary section of the Bitcoin note"

  {BOLD}Read/write via Obsidian{RESET} (when Obsidian is open + Local REST API enabled)
    "Create a new note in my vault called X"
    "Update the tags on the prompt-injection note"

  {BOLD}Vault path:{RESET} {vault_path}
  {BOLD}MCP config:{RESET} ~/.claude/settings.json

  Restart Claude Code for the MCP servers to activate.
""")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print(f"\n{BOLD}Obsidian Vault Adapter — MCP Setup{RESET}")
    print(f"{DIM}Wires your Obsidian vault into Claude as a live memory layer{RESET}\n")

    # ── Preflight ────────────────────────────────────────────────────────────
    h("Checking dependencies")
    if not check_node() or not check_npx():
        err("Missing dependencies. Install Node.js first: brew install node")
        sys.exit(1)

    # ── Vault path ───────────────────────────────────────────────────────────
    h("Vault location")

    # Try to pre-fill from config.yaml
    default_vault = ""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            default_vault = cfg.get("vault_path", "")
        except ImportError:
            pass

    vault_str = ask("Obsidian vault path", default_vault)
    vault_path = Path(vault_str).expanduser().resolve()

    if not vault_path.exists():
        err(f"Path not found: {vault_path}")
        create = ask("Create it? (y/n)", "y")
        if create.lower() == "y":
            vault_path.mkdir(parents=True)
            ok(f"Created: {vault_path}")
        else:
            sys.exit(1)

    ok(f"Vault: {vault_path}")

    # ── Obsidian API key (optional) ──────────────────────────────────────────
    h("Obsidian Local REST API key (optional)")
    info("Leave blank if you haven't installed the plugin yet — you can add it later.")
    api_key = ask("API key", "")
    api_key_provided = bool(api_key)

    # ── Install qmd ──────────────────────────────────────────────────────────
    install_qmd()

    # ── Index vault ──────────────────────────────────────────────────────────
    index_vault(vault_path)

    # ── Write MCP config ─────────────────────────────────────────────────────
    write_mcp_config(vault_path, api_key)

    # ── Instructions ─────────────────────────────────────────────────────────
    print_obsidian_plugin_instructions(api_key_provided)
    print_usage_guide(vault_path)


if __name__ == "__main__":
    main()
