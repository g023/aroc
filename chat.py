#!/usr/bin/env python3
"""
AROC - Agentic Read-Only Chat
chat.py — Self-contained agentic chat 
Uses: g023/Qwen3.5-9B-IQ2_M (https://huggingface.co/g023/g023-Qwen3.5-9B-GGUF)
Requires: locally installed llama-server (https://github.com/ggerganov/llama.cpp) by Georgi Gerganov

Author: g023
License: MIT

**Note:** Attempts are made to make this read only, but I might not have stopped everything, so only consider this for testing in a container or sandbox environment.

A rich terminal chat interface powered by llama.cpp with:
  • Auto-managed llama-server backend with anti-looping optimizations
  • Reasoning (/think) and non-reasoning (/nothink) modes
  • 19 built-in tools: read_file, head, tail, list_dir, find_files, grep,
    grep_context, file_info, python_outline, diff_files, get_time,
    scratch_pad, analyze_file (subagent), todo_add, todo_list, todo_done,
    todo_remove, memory_append, memory_read
  • In-session scratch pad, todo list, and memory for planning and tracking
  • Streaming token output with interleaved thinking display
  • Multi-turn tool calling chains with subagent delegation
  • Context-aware token management and session save/load

Usage:
  python3 chat.py                      Start with defaults
  python3 chat.py --port 8080          Custom port
  python3 chat.py --no-server          Connect to existing server
  python3 chat.py --think              Start in reasoning mode
  python3 chat.py --no-color           Disable ANSI colors
"""

import os
import sys
import json
import time
import re
import signal
import socket
import subprocess
import shutil
import stat
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

try:
    import readline  # noqa: F401 — enables input() history
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "g023-Qwen3.5-9B-IQ2_M.gguf" # grab from https://huggingface.co/g023/g023-Qwen3.5-9B-GGUF
MODEL_PATH = SCRIPT_DIR / MODEL_NAME

LLAMA_SERVER_SEARCH = [
    "llama-server",
    str(SCRIPT_DIR / "llama-server"),
    str(SCRIPT_DIR.parent / "llama.cpp" / "build" / "bin" / "llama-server"),
]

# Optimal sampling — prevents looping at 2-bit quantization
SAMPLING = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.05,
    "repeat_penalty": 1.15,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.0,
}

DEFAULT_PORT = 19300 # for llama-server API
CONTEXT_SIZE = 64000
KV_CACHE_TYPE = "q4_0"
N_GPU_LAYERS = 36
MAX_GEN_TOKENS = 16384
MAX_TOOL_TURNS = 12
SUBAGENT_MAX_TOKENS = 16384
CONTEXT_PRUNE_RATIO = 0.75
VERSION = "2.0.0"

# Reasoning format for proper think/content separation
REASONING_FORMAT = "deepseek"

# ═══════════════════════════════════════════════════════════════════════
# ANSI helpers
# ═══════════════════════════════════════════════════════════════════════

_NO_COLOR = False  # Set by --no-color or if not a tty


def _c(code):
    """Return ANSI code or empty string when colours are off."""
    return "" if _NO_COLOR else code


class C:
    """Lazy ANSI codes — read through _c() so --no-color works."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"
    BRED = "\033[1;31m"
    BGREEN = "\033[1;32m"
    BYELLOW = "\033[1;33m"
    BBLUE = "\033[1;34m"
    BMAGENTA = "\033[1;35m"
    BCYAN = "\033[1;36m"
    # semantic
    THINK = "\033[2;3;36m"
    TOOL = "\033[36m"
    ERR = "\033[1;31m"
    PROMPT = "\033[1;32m"
    SYS = "\033[1;35m"


def _w(code, text):
    """Wrap *text* with an ANSI *code* (and reset)."""
    return f"{_c(code)}{text}{_c(C.RESET)}"


# ═══════════════════════════════════════════════════════════════════════
# Small utilities
# ═══════════════════════════════════════════════════════════════════════

def _resolve_path(p):
    """Resolve a user-supplied path to an absolute Path."""
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _safe_json(raw):
    """Try to parse *raw* as JSON; return {} on failure."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


# ═══════════════════════════════════════════════════════════════════════
# Tool definitions (OpenAI function-calling schema)
# ═══════════════════════════════════════════════════════════════════════

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. Supports optional line range "
                "(1-based). Returns at most 200 lines per call. Use head/tail for quick previews."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (absolute or relative to cwd)",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line (1-based, default 1)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line (1-based, default start+499)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List a directory's contents with sizes and dates. "
                "Set recursive=true for a tree view up to max_depth."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "recursive": {
                        "type": "boolean",
                        "description": "Tree view (default false)",
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Include dot-files (default false)",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Max tree depth (default 3)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files matching a glob pattern (e.g. '**/*.py').",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "root": {
                        "type": "string",
                        "description": "Root directory (default cwd)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default 50)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search for a regex pattern in files. Returns matching lines "
                "with file:line prefixes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {
                        "type": "string",
                        "description": "File or directory to search",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Recurse into subdirectories (default true)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max matches (default 30)",
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive (default true)",
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_info",
            "description": "Get metadata: size, type, permissions, modified time, line count.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File or directory path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "head",
            "description": "Read the first N lines of a file (default 20). Quick file preview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "lines": {"type": "integer", "description": "Number of lines (default 20)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tail",
            "description": "Read the last N lines of a file (default 20). Good for logs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "lines": {"type": "integer", "description": "Number of lines (default 20)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_context",
            "description": (
                "Search for a regex pattern and return matches with surrounding "
                "context lines. Like grep -C."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "File to search"},
                    "context": {
                        "type": "integer",
                        "description": "Lines of context before and after (default 3)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max matches (default 10)",
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_outline",
            "description": (
                "Extract the structure of a Python file: classes, methods, functions, "
                "and their line numbers. Great for understanding code organization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Python file path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff_files",
            "description": "Show differences between two text files in unified diff format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file1": {"type": "string", "description": "First file path"},
                    "file2": {"type": "string", "description": "Second file path"},
                    "context_lines": {
                        "type": "integer",
                        "description": "Context lines around changes (default 3)",
                    },
                },
                "required": ["file1", "file2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratch_pad",
            "description": (
                "Your working memory. Read or write your scratch pad for plans, "
                "notes, task tracking, and findings. Call with content to overwrite. "
                "Call without content to read current notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "New content to write (omit to read current pad)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date, time, timezone, and system uptime.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_file",
            "description": (
                "Delegate deep file analysis to a focused sub-agent so the main "
                "conversation context is not overloaded with raw file content. "
                "The sub-agent reads the file and answers the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to analyze"},
                    "question": {
                        "type": "string",
                        "description": "What to find or analyze",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Max lines to read (default 300)",
                    },
                },
                "required": ["path", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_add",
            "description": "Add a task to your todo list. Returns the task ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task description"},
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Priority level (default: medium)",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_list",
            "description": "List all todo items with their status and priority.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_done",
            "description": "Mark a todo item as done by its ID number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Todo item ID to mark done"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_remove",
            "description": "Remove a todo item by its ID number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Todo item ID to remove"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_append",
            "description": (
                "Append a note to persistent session memory. Unlike scratch_pad "
                "(which overwrites), memory_append accumulates entries. "
                "Use for logging findings, decisions, and key facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "Note to append"},
                    "tag": {
                        "type": "string",
                        "description": "Optional category tag (e.g. 'finding', 'decision', 'question')",
                    },
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read all accumulated session memory notes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ═══════════════════════════════════════════════════════════════════════
# Tool implementations  (all read-only, stdlib only)
# ═══════════════════════════════════════════════════════════════════════


def tool_read_file(path, start_line=1, end_line=None):
    fp = _resolve_path(path)
    if not fp.exists():
        return f"Error: not found: {fp}"
    if not fp.is_file():
        return f"Error: not a file: {fp}"
    if not os.access(fp, os.R_OK):
        return f"Error: permission denied: {fp}"
    try:
        size = fp.stat().st_size
        if size > 10_000_000:
            return (
                f"Error: file too large ({_human_size(size)}). "
                "Use start_line/end_line to read a portion."
            )
        with open(fp, "r", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        s = max(1, int(start_line)) - 1
        e = min(int(end_line), total) if end_line else min(s + 200, total)
        sel = lines[s:e]
        hdr = f"[{fp}  lines {s+1}–{e} of {total}]"
        if e < total:
            hdr += f"  ({total - e} more)"
        return hdr + "\n" + "".join(sel)
    except Exception as exc:
        return f"Error: {exc}"


def _tree(root, hidden, maxd, prefix="", depth=0):
    if depth > maxd:
        return prefix + "…\n"
    try:
        entries = sorted(root.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return prefix + "(permission denied)\n"
    if not hidden:
        entries = [e for e in entries if not e.name.startswith(".")]
    out = ""
    if depth == 0:
        out = f"{root}/\n"
    for i, entry in enumerate(entries):
        last = i == len(entries) - 1
        conn = "└── " if last else "├── "
        if entry.is_dir():
            out += f"{prefix}{conn}{entry.name}/\n"
            ext = "    " if last else "│   "
            if depth < maxd:
                out += _tree(entry, hidden, maxd, prefix + ext, depth + 1)
        else:
            out += f"{prefix}{conn}{entry.name}\n"
    return out


def tool_list_dir(path=".", recursive=False, show_hidden=False, max_depth=3):
    dp = _resolve_path(path)
    if not dp.exists():
        return f"Error: not found: {dp}"
    if not dp.is_dir():
        return f"Error: not a directory: {dp}"
    if recursive:
        return _tree(dp, show_hidden, max_depth)
    try:
        entries = sorted(dp.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        rows = [f"[{dp}]"]
        for e in entries:
            if not show_hidden and e.name.startswith("."):
                continue
            try:
                st = e.stat()
                mt = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                if e.is_dir():
                    rows.append(f"  {e.name}/  {mt}")
                else:
                    rows.append(f"  {e.name}  ({_human_size(st.st_size)})  {mt}")
            except OSError:
                rows.append(f"  {e.name}  (unreadable)")
        if len(rows) == 1:
            rows.append("  (empty)")
        return "\n".join(rows)
    except PermissionError:
        return f"Error: permission denied: {dp}"


def tool_find_files(pattern, root=".", max_results=50):
    rp = _resolve_path(root)
    if not rp.is_dir():
        return f"Error: not a directory: {rp}"
    try:
        matches = sorted(rp.glob(pattern))[:max_results + 1]
        trunc = len(matches) > max_results
        matches = matches[:max_results]
        rows = [f"[find '{pattern}' in {rp}]"]
        for m in matches:
            try:
                rel = m.relative_to(rp)
            except ValueError:
                rel = m
            suf = "/" if m.is_dir() else f"  ({_human_size(m.stat().st_size)})"
            rows.append(f"  {rel}{suf}")
        if not matches:
            rows.append("  (no matches)")
        elif trunc:
            rows.append(f"  … truncated at {max_results}")
        else:
            rows.append(f"  [{len(matches)} match{'es' if len(matches) != 1 else ''}]")
        return "\n".join(rows)
    except Exception as exc:
        return f"Error: {exc}"


def tool_grep(pattern, path, recursive=True, max_results=30, ignore_case=True):
    fp = _resolve_path(path)
    if not fp.exists():
        return f"Error: not found: {fp}"
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        return f"Error: bad regex: {exc}"

    hits = []
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}

    def _search(fpath):
        if len(hits) >= max_results:
            return
        try:
            with open(fpath, "r", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if len(hits) >= max_results:
                        return
                    if rx.search(line):
                        try:
                            rel = fpath.relative_to(Path.cwd())
                        except ValueError:
                            rel = fpath
                        hits.append(f"  {rel}:{i}: {line.rstrip()}")
        except (PermissionError, IsADirectoryError):
            pass

    if fp.is_file():
        _search(fp)
    else:
        walker = os.walk(fp) if recursive else [(str(fp), [], [f.name for f in fp.iterdir() if f.is_file()])]
        for dirpath, dirs, files in walker:
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fname in files:
                _search(Path(dirpath) / fname)

    hdr = f"[grep '{pattern}' in {fp}]"
    if not hits:
        return f"{hdr}\n  (no matches)"
    tail = f"  … (truncated at {max_results})" if len(hits) >= max_results else f"  [{len(hits)} match{'es' if len(hits)!=1 else ''}]"
    return "\n".join([hdr] + hits + [tail])


def tool_file_info(path):
    fp = _resolve_path(path)
    if not fp.exists():
        return f"Error: not found: {fp}"
    try:
        st = fp.stat()
        info = {
            "path": str(fp),
            "type": "directory" if fp.is_dir() else ("symlink" if fp.is_symlink() else "file"),
            "size": _human_size(st.st_size),
            "size_bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "permissions": stat.filemode(st.st_mode),
        }
        if fp.is_file():
            try:
                with open(fp, "r", errors="replace") as fh:
                    info["lines"] = sum(1 for _ in fh)
            except Exception:
                pass
        elif fp.is_dir():
            try:
                children = list(fp.iterdir())
                info["entries"] = len(children)
                info["subdirs"] = sum(1 for c in children if c.is_dir())
                info["files"] = sum(1 for c in children if c.is_file())
            except Exception:
                pass
        return json.dumps(info, indent=2)
    except Exception as exc:
        return f"Error: {exc}"


def tool_head(path, lines=20):
    fp = _resolve_path(path)
    if not fp.exists():
        return f"Error: not found: {fp}"
    if not fp.is_file():
        return f"Error: not a file: {fp}"
    try:
        with open(fp, "r", errors="replace") as f:
            all_lines = f.readlines()
        n = min(int(lines), len(all_lines))
        hdr = f"[{fp}  first {n} of {len(all_lines)} lines]"
        return hdr + "\n" + "".join(all_lines[:n])
    except Exception as exc:
        return f"Error: {exc}"


def tool_tail(path, lines=20):
    fp = _resolve_path(path)
    if not fp.exists():
        return f"Error: not found: {fp}"
    if not fp.is_file():
        return f"Error: not a file: {fp}"
    try:
        with open(fp, "r", errors="replace") as f:
            all_lines = f.readlines()
        n = min(int(lines), len(all_lines))
        start = len(all_lines) - n
        hdr = f"[{fp}  last {n} of {len(all_lines)} lines]"
        return hdr + "\n" + "".join(all_lines[start:])
    except Exception as exc:
        return f"Error: {exc}"


def tool_grep_context(pattern, path, context=3, max_results=10):
    fp = _resolve_path(path)
    if not fp.exists():
        return f"Error: not found: {fp}"
    if not fp.is_file():
        return f"Error: not a file: {fp}"
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Error: bad regex: {exc}"
    try:
        with open(fp, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as exc:
        return f"Error: {exc}"
    ctx = int(context)
    hits = []
    matched_lines = set()
    for i, line in enumerate(lines):
        if rx.search(line):
            matched_lines.add(i)
    if not matched_lines:
        return f"[grep_context '{pattern}' in {fp}]\n  (no matches)"
    # Build context blocks
    blocks = []
    count = 0
    for i in sorted(matched_lines):
        if count >= max_results:
            break
        count += 1
        start = max(0, i - ctx)
        end = min(len(lines), i + ctx + 1)
        block = [f"--- match at line {i+1} ---"]
        for j in range(start, end):
            marker = ">>" if j == i else "  "
            block.append(f"{marker} {j+1}: {lines[j].rstrip()}")
        blocks.append("\n".join(block))
    hdr = f"[grep_context '{pattern}' in {fp} — {len(matched_lines)} matches]"
    return hdr + "\n" + "\n".join(blocks)


def tool_python_outline(path):
    fp = _resolve_path(path)
    if not fp.exists():
        return f"Error: not found: {fp}"
    if not fp.is_file():
        return f"Error: not a file: {fp}"
    try:
        with open(fp, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as exc:
        return f"Error: {exc}"
    outline = [f"[{fp} — {len(lines)} lines]"]
    indent_stack = []  # track current class for methods
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        # Match class/def/async def
        m = re.match(r'^(\s*)(class|async\s+def|def)\s+(\w+)', line)
        if m:
            spaces, kind, name = m.group(1), m.group(2), m.group(3)
            kind_clean = kind.replace('async ', 'async_')
            # Extract signature for functions
            sig = ""
            if 'def' in kind:
                sig_m = re.match(r'^\s*(?:async\s+)?def\s+(\w+\([^)]*\))', line)
                if sig_m:
                    sig = sig_m.group(1)
                else:
                    sig = name + "(...)"
            else:
                sig = name
            prefix = "  " * (indent // 4) if indent > 0 else ""
            outline.append(f"  {prefix}{kind_clean} {sig}  (line {i})")
    if len(outline) == 1:
        outline.append("  (no classes or functions found)")
    return "\n".join(outline)


def tool_diff_files(file1, file2, context_lines=3):
    import difflib
    fp1 = _resolve_path(file1)
    fp2 = _resolve_path(file2)
    if not fp1.exists():
        return f"Error: not found: {fp1}"
    if not fp2.exists():
        return f"Error: not found: {fp2}"
    try:
        with open(fp1, "r", errors="replace") as f:
            lines1 = f.readlines()
        with open(fp2, "r", errors="replace") as f:
            lines2 = f.readlines()
    except Exception as exc:
        return f"Error: {exc}"
    diff = list(difflib.unified_diff(
        lines1, lines2,
        fromfile=str(fp1), tofile=str(fp2),
        n=int(context_lines)
    ))
    if not diff:
        return f"Files are identical ({len(lines1)} lines each)"
    result = "".join(diff)
    if len(result) > 8000:
        result = result[:8000] + "\n… (truncated)"
    return result


def tool_get_time():
    now = datetime.now()
    info = {
        "datetime": now.isoformat(),
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": time.tzname[0],
        "utc_offset": time.strftime("%z"),
    }
    try:
        with open("/proc/uptime", "r") as f:
            up = float(f.read().split()[0])
        d, rem = divmod(int(up), 86400)
        h, rem = divmod(rem, 3600)
        m, _ = divmod(rem, 60)
        info["uptime"] = f"{d}d {h}h {m}m"
    except Exception:
        pass
    return json.dumps(info, indent=2)


# dispatch (analyze_file and scratch_pad are handled by the Agent directly)
TOOL_DISPATCH = {
    "read_file": lambda a: tool_read_file(a.get("path", ""), a.get("start_line", 1), a.get("end_line")),
    "head": lambda a: tool_head(a.get("path", ""), a.get("lines", 20)),
    "tail": lambda a: tool_tail(a.get("path", ""), a.get("lines", 20)),
    "list_dir": lambda a: tool_list_dir(a.get("path", "."), a.get("recursive", False), a.get("show_hidden", False), a.get("max_depth", 3)),
    "find_files": lambda a: tool_find_files(a.get("pattern", "*"), a.get("root", "."), a.get("max_results", 50)),
    "grep": lambda a: tool_grep(a.get("pattern", ""), a.get("path", "."), a.get("recursive", True), a.get("max_results", 30), a.get("ignore_case", True)),
    "grep_context": lambda a: tool_grep_context(a.get("pattern", ""), a.get("path", "."), a.get("context", 3), a.get("max_results", 10)),
    "file_info": lambda a: tool_file_info(a.get("path", ".")),
    "python_outline": lambda a: tool_python_outline(a.get("path", "")),
    "diff_files": lambda a: tool_diff_files(a.get("file1", ""), a.get("file2", ""), a.get("context_lines", 3)),
    "get_time": lambda _: tool_get_time(),
}

# ═══════════════════════════════════════════════════════════════════════
# LlamaServer  — manage the llama-server process
# ═══════════════════════════════════════════════════════════════════════


class LlamaServer:
    def __init__(self, port=DEFAULT_PORT, model=None, ngl=N_GPU_LAYERS, ctx=CONTEXT_SIZE):
        self.port = port
        self.model = str(model or MODEL_PATH)
        self.ngl = ngl
        self.ctx = ctx
        self.base_url = f"http://127.0.0.1:{port}"
        self.process = None
        self._bin = None

    # -- binary discovery --------------------------------------------------
    def find_binary(self):
        if self._bin:
            return self._bin
        w = shutil.which("llama-server")
        if w:
            self._bin = w
            return w
        for p in LLAMA_SERVER_SEARCH:
            if p and os.path.isfile(p) and os.access(p, os.X_OK):
                self._bin = p
                return p
        return None

    # -- health ------------------------------------------------------------
    def is_healthy(self):
        try:
            r = urllib.request.urlopen(f"{self.base_url}/health", timeout=3)
            return json.loads(r.read()).get("status") == "ok"
        except Exception:
            return False

    # -- start / stop ------------------------------------------------------
    def start(self):
        if self.is_healthy():
            return True
        binary = self.find_binary()
        if not binary:
            return False
        cmd = [
            binary,
            "-m", self.model,
            "-ngl", str(self.ngl),
            "-fa", "on",
            "-ctk", KV_CACHE_TYPE,
            "-ctv", KV_CACHE_TYPE,
            "-c", str(self.ctx),
            "-b", "512", "-ub", "512",
            "-t", str(min(os.cpu_count() or 4, 8)),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-np", "1",
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        for _ in range(180):
            time.sleep(0.5)
            if self.is_healthy():
                return True
            if self.process.poll() is not None:
                return False
        return False

    def stop(self):
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.process = None

    # -- API ---------------------------------------------------------------
    def _request(self, messages, *, tools=None, stream=True,
                 max_tokens=MAX_GEN_TOKENS, extra_sampling=None):
        """Build and send a chat-completion request.  Returns HTTPResponse."""
        body = {
            "model": "local",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            **SAMPLING,
        }
        if extra_sampling:
            body.update(extra_sampling)
        if tools:
            body["tools"] = tools
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=600)

    def chat(self, messages, **kw):
        """Convenience: non-streaming → parsed dict."""
        resp = self._request(messages, stream=False, **kw)
        return json.loads(resp.read())

    def chat_stream(self, messages, **kw):
        """Convenience: streaming → raw HTTPResponse."""
        return self._request(messages, stream=True, **kw)


# ═══════════════════════════════════════════════════════════════════════
# Streaming parsers
# ═══════════════════════════════════════════════════════════════════════


class StreamParser:
    """Iterate over SSE chunks from an HTTPResponse (or any line-iterable)."""

    def __init__(self, response):
        self.resp = response

    def __iter__(self):
        for raw in self.resp:
            if isinstance(raw, bytes):
                line = raw.decode("utf-8", errors="replace").strip()
            else:
                line = raw.strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


class ThinkParser:
    """Incrementally separate <think>…</think> from content."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.in_think = False
        self.pending = ""

    def feed(self, text):
        """Return list of (type, text) where type is 'think' or 'content'."""
        buf = self.pending + text
        self.pending = ""
        results = []

        while buf:
            if self.in_think:
                idx = buf.find(self.CLOSE)
                if idx >= 0:
                    if idx > 0:
                        results.append(("think", buf[:idx]))
                    buf = buf[idx + len(self.CLOSE):]
                    self.in_think = False
                else:
                    # hold back to avoid splitting </think>
                    hold = len(self.CLOSE) - 1
                    if len(buf) > hold:
                        results.append(("think", buf[:-hold]))
                        self.pending = buf[-hold:]
                    else:
                        self.pending = buf
                    buf = ""
            else:
                idx = buf.find(self.OPEN)
                if idx >= 0:
                    if idx > 0:
                        results.append(("content", buf[:idx]))
                    buf = buf[idx + len(self.OPEN):]
                    self.in_think = True
                else:
                    hold = len(self.OPEN) - 1
                    if len(buf) > hold:
                        results.append(("content", buf[:-hold]))
                        self.pending = buf[-hold:]
                    else:
                        self.pending = buf
                    buf = ""
        return results

    def flush(self):
        """Emit anything still buffered."""
        if self.pending:
            kind = "think" if self.in_think else "content"
            out = [(kind, self.pending)]
            self.pending = ""
            return out
        return []


# ═══════════════════════════════════════════════════════════════════════
# Agent — conversation loop with tool calling
# ═══════════════════════════════════════════════════════════════════════


class Agent:
    def __init__(self, server: LlamaServer, thinking=False):
        self.server = server
        self.thinking = thinking
        self.messages: list[dict] = []
        self.total_prompt_tokens = 0
        self.total_comp_tokens = 0
        self.tool_calls_made = 0
        self.session_start = time.time()
        self.scratch_pad = ""  # in-session working memory
        self.todos = []  # list of {id, task, priority, done}
        self._todo_counter = 0
        self.memory = []  # list of {note, tag, time}

    # -- system prompt -----------------------------------------------------
    def _system(self):
        parts = [
            "You are AROC, an expert AI assistant with read-only filesystem tools "
            "and planning/memory tools for managing complex tasks.\n\n",
            "TOOL STRATEGY:\n"
            "- Use head (first N lines) or tail (last N lines) for quick previews.\n"
            "- Use python_outline to map code structure before diving into details.\n"
            "- Use grep/grep_context to find specific code sections efficiently.\n"
            "- Use read_file with start_line/end_line for targeted reading (avoid reading entire large files).\n"
            "- Use scratch_pad to save/overwrite working notes.\n"
            "- Use memory_append to log key findings and decisions (accumulates).\n"
            "- Use todo_add/todo_list/todo_done to track multi-step plans.\n"
            "- Use analyze_file to delegate deep file analysis to a sub-agent.\n"
            "- Be concise. Cite line numbers when discussing code.\n\n",
            f"Working directory: {Path.cwd()}\n",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        ]
        return "".join(parts)

    # -- context management ------------------------------------------------
    def _est_tokens(self):
        total = len(self._system()) // 4
        for m in self.messages:
            c = m.get("content") or ""
            if isinstance(c, str):
                total += len(c) // 4
            if "tool_calls" in m:
                total += len(json.dumps(m["tool_calls"], default=str)) // 4
        return total

    def _prune(self):
        limit = int(CONTEXT_SIZE * CONTEXT_PRUNE_RATIO)
        while self._est_tokens() > limit and len(self.messages) > 4:
            self.messages.pop(0)

    # -- build messages list -----------------------------------------------
    def _build_messages(self):
        return [{"role": "system", "content": self._system()}] + self.messages

    # -- tool execution ----------------------------------------------------
    def _exec_tool(self, name, raw_args):
        self.tool_calls_made += 1
        args = _safe_json(raw_args)

        if name == "analyze_file":
            return self._subagent(args)

        if name == "scratch_pad":
            return self._handle_scratch_pad(args)

        if name in ("todo_add", "todo_list", "todo_done", "todo_remove"):
            return self._handle_todo(name, args)

        if name in ("memory_append", "memory_read"):
            return self._handle_memory(name, args)

        handler = TOOL_DISPATCH.get(name)
        if not handler:
            return f"Error: unknown tool '{name}'"
        try:
            result = handler(args)
        except Exception as exc:
            result = f"Error: {exc}"
        # truncate very large results
        if len(result) > 12000:
            result = result[:12000] + "\n… (truncated)"
        return result

    # -- scratch pad -------------------------------------------------------
    def _handle_scratch_pad(self, args):
        content = args.get("content")
        if content is not None:
            self.scratch_pad = content
            return f"Scratch pad updated ({len(content)} chars)"
        if self.scratch_pad:
            return f"[Scratch Pad]\n{self.scratch_pad}"
        return "(scratch pad is empty)"

    def _handle_todo(self, name, args):
        if name == "todo_add":
            self._todo_counter += 1
            item = {
                "id": self._todo_counter,
                "task": args.get("task", ""),
                "priority": args.get("priority", "medium"),
                "done": False,
            }
            self.todos.append(item)
            return f"Added todo #{item['id']}: {item['task']} [{item['priority']}]"

        if name == "todo_list":
            if not self.todos:
                return "(todo list is empty)"
            lines = []
            for t in self.todos:
                mark = "✓" if t["done"] else "○"
                lines.append(f"  {mark} #{t['id']} [{t['priority']}] {t['task']}")
            done = sum(1 for t in self.todos if t["done"])
            lines.append(f"\n{done}/{len(self.todos)} completed")
            return "\n".join(lines)

        if name == "todo_done":
            tid = int(args.get("id", 0))
            for t in self.todos:
                if t["id"] == tid:
                    t["done"] = True
                    return f"Marked todo #{tid} as done: {t['task']}"
            return f"Error: todo #{tid} not found"

        if name == "todo_remove":
            tid = int(args.get("id", 0))
            for i, t in enumerate(self.todos):
                if t["id"] == tid:
                    self.todos.pop(i)
                    return f"Removed todo #{tid}: {t['task']}"
            return f"Error: todo #{tid} not found"

        return f"Error: unknown todo operation '{name}'"

    def _handle_memory(self, name, args):
        if name == "memory_append":
            entry = {
                "note": args.get("note", ""),
                "tag": args.get("tag", ""),
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            self.memory.append(entry)
            return f"Memory #{len(self.memory)} saved: [{entry['tag'] or 'note'}] {entry['note'][:80]}"

        if name == "memory_read":
            if not self.memory:
                return "(session memory is empty)"
            lines = []
            for i, m in enumerate(self.memory, 1):
                tag = f"[{m['tag']}] " if m["tag"] else ""
                lines.append(f"  #{i} ({m['time']}) {tag}{m['note']}")
            return "\n".join(lines)

        return f"Error: unknown memory operation '{name}'"

    # -- subagent ----------------------------------------------------------
    def _subagent(self, args):
        path = args.get("path", "")
        question = args.get("question", "Summarize this file.")
        max_lines = int(args.get("max_lines", 300))

        content = tool_read_file(path, 1, max_lines)
        if content.startswith("Error:"):
            return content

        msgs = [
            {"role": "system", "content": (
                "You are a focused file-analysis assistant. Analyze the provided "
                "file and answer concisely. Cite line numbers when relevant."
            )},
            {"role": "user", "content": f"File: {path}\n\n{content}\n\n---\nQuestion: {question}"},
        ]
        try:
            body = self.server.chat(
                msgs, max_tokens=SUBAGENT_MAX_TOKENS,
                extra_sampling={"reasoning_format": REASONING_FORMAT},
            )
            msg = body["choices"][0]["message"]
            result = msg.get("content") or ""
            # If server uses native reasoning, content may be empty during thinking;
            # fall back to reasoning_content if no actual content produced
            if not result.strip() and msg.get("reasoning_content"):
                result = msg["reasoning_content"]
            return result or "(no response)"
        except Exception as exc:
            return f"Sub-agent error: {exc}"

    # -- main chat entry point ---------------------------------------------
    def chat(self, user_input, *, on_think=None, on_content=None,
             on_tool=None, on_tool_result=None):
        """Process one user turn; may invoke multiple tool rounds.

        Returns (content_text, think_text, usage_dict).
        """
        self._prune()
        self.messages.append({"role": "user", "content": user_input})

        all_content = ""
        all_think = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}

        for _turn in range(MAX_TOOL_TURNS):
            try:
                resp = self.server.chat_stream(
                    self._build_messages(), tools=TOOL_DEFS,
                    extra_sampling={"reasoning_format": REASONING_FORMAT},
                )
            except (urllib.error.URLError, OSError) as exc:
                err = f"[server error: {exc}]"
                if on_content:
                    on_content(err)
                self.messages.append({"role": "assistant", "content": err})
                return err, all_think, usage

            # ---- stream parsing ------------------------------------------
            content_buf = ""
            think_buf = ""
            tc_accum = {}           # {index: {id, function:{name, arguments}}}
            tp = ThinkParser()     # fallback for servers without native reasoning
            _has_native_reasoning = False

            try:
                for chunk in StreamParser(resp):
                    ch = (chunk.get("choices") or [{}])[0]
                    delta = ch.get("delta", {})

                    # reasoning_content (native llama.cpp reasoning support)
                    if delta.get("reasoning_content"):
                        _has_native_reasoning = True
                        think_buf += delta["reasoning_content"]
                        all_think += delta["reasoning_content"]
                        if self.thinking and on_think:
                            on_think(delta["reasoning_content"])

                    # content tokens
                    if delta.get("content"):
                        content_buf += delta["content"]
                        if _has_native_reasoning:
                            # Server already separated thinking — content is clean
                            all_content += delta["content"]
                            if on_content:
                                on_content(delta["content"])
                        else:
                            # Fallback: parse <think> tags from content
                            for kind, txt in tp.feed(delta["content"]):
                                if kind == "think":
                                    all_think += txt
                                    if self.thinking and on_think:
                                        on_think(txt)
                                else:
                                    all_content += txt
                                    if on_content:
                                        on_content(txt)

                    # tool-call tokens
                    for tc in delta.get("tool_calls", []):
                        idx = tc.get("index", 0)
                        if idx not in tc_accum:
                            tc_accum[idx] = {
                                "id": tc.get("id", f"call_{idx}"),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if "id" in tc:
                            tc_accum[idx]["id"] = tc["id"]
                        fn = tc.get("function", {})
                        if "name" in fn and fn["name"]:
                            tc_accum[idx]["function"]["name"] = fn["name"]
                        if "arguments" in fn:
                            tc_accum[idx]["function"]["arguments"] += fn["arguments"]

                    # usage
                    if "usage" in chunk:
                        for k in usage:
                            usage[k] = max(usage[k], chunk["usage"].get(k, 0))

            except KeyboardInterrupt:
                try:
                    resp.close()
                except Exception:
                    pass
                break

            # flush remaining think-buffer (only for fallback parser)
            if not _has_native_reasoning:
                for kind, txt in tp.flush():
                    if kind == "think":
                        all_think += txt
                        if self.thinking and on_think:
                            on_think(txt)
                    else:
                        all_content += txt
                        if on_content:
                            on_content(txt)

            # ---- tool dispatch -------------------------------------------
            tool_calls = [tc_accum[k] for k in sorted(tc_accum)]
            if tool_calls:
                asst_msg = {"role": "assistant"}
                if content_buf:
                    asst_msg["content"] = content_buf
                asst_msg["tool_calls"] = tool_calls
                self.messages.append(asst_msg)

                for tc in tool_calls:
                    fn = tc["function"]
                    name = fn["name"]
                    args = _safe_json(fn["arguments"])
                    if on_tool:
                        on_tool(name, args)
                    result = self._exec_tool(name, fn["arguments"])
                    if on_tool_result:
                        on_tool_result(name, result)
                    self.messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc["id"],
                    })
                continue  # next turn — model sees tool results
            else:
                # no tool calls → final answer
                # Fallback: if model produced only reasoning with no content,
                # use reasoning as content (common with 2-bit models)
                if not content_buf.strip() and think_buf.strip():
                    # Clean up reasoning: remove any leaked tool_call tags
                    fallback = think_buf.strip()
                    # Strip malformed tool_call blocks that leaked into reasoning
                    fallback = re.sub(
                        r'<tool_call>.*?</tool_call>',
                        '', fallback, flags=re.DOTALL
                    ).strip()
                    # Also remove unclosed tool_call blocks at the end
                    fallback = re.sub(
                        r'<tool_call>.*$',
                        '', fallback, flags=re.DOTALL
                    ).strip()
                    if fallback:
                        all_content += fallback
                        if on_content:
                            on_content(fallback)
                        self.messages.append({"role": "assistant", "content": fallback})
                    else:
                        self.messages.append({"role": "assistant", "content": content_buf})
                else:
                    self.messages.append({"role": "assistant", "content": content_buf})
                break

        self.total_prompt_tokens += usage["prompt_tokens"]
        self.total_comp_tokens += usage["completion_tokens"]
        return all_content, all_think, usage

    # -- convenience -------------------------------------------------------
    def clear(self):
        self.messages.clear()

    def set_thinking(self, on):
        self.thinking = on


# ═══════════════════════════════════════════════════════════════════════
# Chat UI
# ═══════════════════════════════════════════════════════════════════════


class ChatUI:
    def __init__(self):
        self._thinking_shown = False
        self._after_tool = False

    # -- banners -----------------------------------------------------------
    def banner(self, server, agent):
        bar = "═" * 60
        print(f"\n{_w(C.BMAGENTA, bar)}")
        print(f"{_w(C.BMAGENTA, '  AROC • Agentic Read-Only Chat')}")
        print(f"{_w(C.BMAGENTA, bar)}")
        print(_w(C.GRAY, f"  Model   : {MODEL_NAME}"))
        print(_w(C.GRAY, f"  Server  : {server.base_url}"))
        print(_w(C.GRAY, f"  Context : {server.ctx} tokens  (KV {KV_CACHE_TYPE})"))
        print(_w(C.GRAY, f"  Thinking: {'ON' if agent.thinking else 'OFF'}"))
        print(_w(C.GRAY, f"  Tools   : {len(TOOL_DEFS)} available"))
        print(_w(C.GRAY, f"  Version : {VERSION}"))
        print(_w(C.GRAY, "  Type /help for commands"))
        print(f"{_w(C.BMAGENTA, '─' * 60)}\n")

    # -- prompting ---------------------------------------------------------
    def prompt(self):
        try:
            return input(f"{_c(C.PROMPT)}You ▸ {_c(C.RESET)}").strip()
        except (EOFError, KeyboardInterrupt):
            return None

    # -- streaming display callbacks ---------------------------------------
    def start_response(self):
        self._thinking_shown = False
        self._after_tool = False
        print(f"\n{_c(C.BCYAN)}AI ▸{_c(C.RESET)} ", end="", flush=True)

    def on_think(self, text):
        if self._after_tool:
            print(f"\n{_c(C.BCYAN)}AI ▸{_c(C.RESET)} ", end="", flush=True)
            self._after_tool = False
        if not self._thinking_shown:
            print(f"\n{_c(C.THINK)}  💭 ", end="", flush=True)
            self._thinking_shown = True
        formatted = text.replace("\n", f"\n{_c(C.THINK)}     ")
        print(f"{_c(C.THINK)}{formatted}", end="", flush=True)

    def on_content(self, text):
        if self._thinking_shown:
            print(f"{_c(C.RESET)}")
            print(f"{_c(C.BCYAN)}AI ▸{_c(C.RESET)} ", end="", flush=True)
            self._thinking_shown = False
        if self._after_tool:
            print(f"\n{_c(C.BCYAN)}AI ▸{_c(C.RESET)} ", end="", flush=True)
            self._after_tool = False
        print(text, end="", flush=True)

    def on_tool(self, name, args):
        # end any open thinking block
        if self._thinking_shown:
            print(_c(C.RESET))
            self._thinking_shown = False
        arg_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
        print(f"\n{_c(C.TOOL)}  🔧 {name}({arg_str}){_c(C.RESET)}", flush=True)

    def on_tool_result(self, name, result):
        lines = result.split("\n")
        max_show = 8
        if len(lines) > max_show:
            print(f"{_c(C.GRAY)}  ┌─ result ({len(lines)} lines)")
            for ln in lines[:max_show - 2]:
                print(f"{_c(C.GRAY)}  │ {ln}")
            print(f"{_c(C.GRAY)}  │ …")
            print(f"{_c(C.GRAY)}  └─ ({len(lines) - max_show + 2} more lines){_c(C.RESET)}")
        else:
            print(f"{_c(C.GRAY)}  ┌─ result")
            for ln in lines:
                print(f"{_c(C.GRAY)}  │ {ln}")
            print(f"{_c(C.GRAY)}  └─{_c(C.RESET)}")
        self._after_tool = True

    def end_response(self, usage=None):
        print(_c(C.RESET))
        if usage:
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            if pt or ct:
                print(f"{_c(C.GRAY)}  [{pt}+{ct} tokens]{_c(C.RESET)}")
        print()

    # -- system / error messages -------------------------------------------
    def sys(self, msg):
        print(f"{_c(C.SYS)}  ⚙ {msg}{_c(C.RESET)}")

    def err(self, msg):
        print(f"{_c(C.ERR)}  ✗ {msg}{_c(C.RESET)}")

    # -- help & info -------------------------------------------------------
    def show_help(self):
        print(f"""
{_w(C.BYELLOW, 'Commands:')}
  {_w(C.BOLD, '/think')}      Enable reasoning mode (step-by-step thinking)
  {_w(C.BOLD, '/nothink')}    Disable reasoning mode (faster)
  {_w(C.BOLD, '/clear')}      Clear conversation history
  {_w(C.BOLD, '/save FILE')}  Save session to JSON file
  {_w(C.BOLD, '/load FILE')}  Load session from JSON file
  {_w(C.BOLD, '/pad')}        Show scratch pad contents
  {_w(C.BOLD, '/clearpad')}   Clear the scratch pad
  {_w(C.BOLD, '/todos')}      Show todo list
  {_w(C.BOLD, '/memory')}     Show session memory notes
  {_w(C.BOLD, '/tokens')}     Show token usage statistics
  {_w(C.BOLD, '/tools')}      List available tools
  {_w(C.BOLD, '/model')}      Show model & sampling info
  {_w(C.BOLD, '/help')}       Show this help
  {_w(C.BOLD, '/quit')}       Exit

{_w(C.BYELLOW, 'Tips:')}
  • Ask about files:   "Read /etc/hostname"
  • Browse code:       "What functions are in chat.py?" or use python_outline
  • Search:            "Find all .py files under RELEASE/"
  • Context search:    "Search for 'def main' with surrounding code"
  • Analyze:           "Analyze the main() function in iq2m_stress_test.py"
  • Compare:           "Diff chat.py and chat_test.py"
  • Track work:        "Add a todo to check all error handlers"
  • Multi-line:        End a line with \\ to continue on the next line
  • Interrupt:         Ctrl+C during generation stops the response
""")

    def show_tools(self):
        print(f"\n{_w(C.BYELLOW, 'Available Tools:')}")
        for td in TOOL_DEFS:
            fn = td["function"]
            props = fn["parameters"].get("properties", {})
            req = set(fn["parameters"].get("required", []))
            parts = []
            for pname in props:
                parts.append(f"{pname}{'*' if pname in req else ''}")
            print(f"  {_w(C.BOLD, fn['name'])}({', '.join(parts)})")
            desc = fn["description"]
            if len(desc) > 90:
                desc = desc[:87] + "…"
            print(f"    {_c(C.GRAY)}{desc}{_c(C.RESET)}")
        print()

    def show_tokens(self, agent):
        elapsed = time.time() - agent.session_start
        m, s = divmod(int(elapsed), 60)
        est = agent._est_tokens()
        pct = est / CONTEXT_SIZE * 100
        print(f"""
{_w(C.BYELLOW, 'Session Stats:')}
  Uptime:          {m}m {s}s
  Messages:        {len(agent.messages)}
  Tool calls:      {agent.tool_calls_made}
  Tokens (est):    {est:,} / {CONTEXT_SIZE:,} ({pct:.1f}%)
  Total generated: {agent.total_comp_tokens:,}
  Thinking:        {'ON' if agent.thinking else 'OFF'}
""")

    def show_model(self):
        print(f"""
{_w(C.BYELLOW, 'Model Info:')}
  Name:      {MODEL_NAME}
  Arch:      Qwen3.5-9B (Hybrid Mamba2-Attention, 32 layers)
  Quant:     IQ2_M (2-bit, 3.4 GB)
  Context:   {CONTEXT_SIZE} tokens
  KV cache:  {KV_CACHE_TYPE}
  GPU:       {N_GPU_LAYERS} layers offloaded

{_w(C.BYELLOW, 'Sampling:')}
  temperature:       {SAMPLING['temperature']}
  top_p:             {SAMPLING['top_p']}
  top_k:             {SAMPLING['top_k']}
  min_p:             {SAMPLING['min_p']}
  repeat_penalty:    {SAMPLING['repeat_penalty']}
  frequency_penalty: {SAMPLING['frequency_penalty']}
  presence_penalty:  {SAMPLING['presence_penalty']}
""")


# ═══════════════════════════════════════════════════════════════════════
# Slash-command dispatch
# ═══════════════════════════════════════════════════════════════════════


def _save_session(agent, path, ui):
    data = {
        "version": VERSION,
        "messages": agent.messages,
        "thinking": agent.thinking,
        "scratch_pad": agent.scratch_pad,
        "todos": agent.todos,
        "todo_counter": agent._todo_counter,
        "memory": agent.memory,
        "stats": {
            "prompt_tokens": agent.total_prompt_tokens,
            "comp_tokens": agent.total_comp_tokens,
            "tool_calls": agent.tool_calls_made,
        },
        "saved_at": datetime.now().isoformat(),
    }
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        ui.sys(f"Session saved → {path}")
    except Exception as exc:
        ui.err(f"Save failed: {exc}")


def _load_session(agent, path, ui):
    try:
        with open(path) as f:
            data = json.load(f)
        agent.messages = data.get("messages", [])
        agent.thinking = data.get("thinking", False)
        agent.scratch_pad = data.get("scratch_pad", "")
        agent.todos = data.get("todos", [])
        agent._todo_counter = data.get("todo_counter", 0)
        agent.memory = data.get("memory", [])
        st = data.get("stats", {})
        agent.total_prompt_tokens = st.get("prompt_tokens", 0)
        agent.total_comp_tokens = st.get("comp_tokens", 0)
        agent.tool_calls_made = st.get("tool_calls", 0)
        ui.sys(f"Loaded {len(agent.messages)} messages from {path}")
    except FileNotFoundError:
        ui.err(f"File not found: {path}")
    except Exception as exc:
        ui.err(f"Load failed: {exc}")


def handle_command(text, agent, ui, server):
    """Process a /command.  Returns 'quit' to exit, else None."""
    parts = text.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit", "/q"):
        return "quit"
    elif cmd == "/help":
        ui.show_help()
    elif cmd == "/think":
        agent.set_thinking(True)
        ui.sys("Reasoning mode ON — model will think step-by-step")
    elif cmd in ("/nothink", "/no_think"):
        agent.set_thinking(False)
        ui.sys("Reasoning mode OFF — faster responses")
    elif cmd == "/clear":
        agent.clear()
        ui.sys("Conversation cleared")
    elif cmd == "/pad":
        if agent.scratch_pad:
            print(f"\n{_w(C.BYELLOW, 'Scratch Pad:')}\n{agent.scratch_pad}\n")
        else:
            ui.sys("Scratch pad is empty")
    elif cmd == "/clearpad":
        agent.scratch_pad = ""
        ui.sys("Scratch pad cleared")
    elif cmd == "/todos":
        if agent.todos:
            lines = []
            for t in agent.todos:
                mark = "✓" if t["done"] else "○"
                lines.append(f"  {mark} #{t['id']} [{t['priority']}] {t['task']}")
            done = sum(1 for t in agent.todos if t["done"])
            lines.append(f"\n{done}/{len(agent.todos)} completed")
            print(f"\n{_w(C.BYELLOW, 'Todo List:')}\n" + "\n".join(lines) + "\n")
        else:
            ui.sys("Todo list is empty")
    elif cmd == "/memory":
        if agent.memory:
            lines = []
            for i, m in enumerate(agent.memory, 1):
                tag = f"[{m['tag']}] " if m.get("tag") else ""
                lines.append(f"  #{i} ({m['time']}) {tag}{m['note']}")
            print(f"\n{_w(C.BYELLOW, 'Session Memory:')}\n" + "\n".join(lines) + "\n")
        else:
            ui.sys("Session memory is empty")
    elif cmd == "/tokens":
        ui.show_tokens(agent)
    elif cmd == "/tools":
        ui.show_tools()
    elif cmd == "/model":
        ui.show_model()
    elif cmd == "/save":
        if not arg:
            ui.err("Usage: /save <filename>")
        else:
            _save_session(agent, arg, ui)
    elif cmd == "/load":
        if not arg:
            ui.err("Usage: /load <filename>")
        else:
            _load_session(agent, arg, ui)
    else:
        ui.err(f"Unknown command: {cmd}  (type /help)")
    return None


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    global _NO_COLOR

    ap = argparse.ArgumentParser(
        description="Agentic chat interface for Qwen3.5-9B-IQ2_M"
    )
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    ap.add_argument("--model", type=str, default=str(MODEL_PATH), help="GGUF model path")
    ap.add_argument("--ngl", type=int, default=N_GPU_LAYERS, help="GPU layers")
    ap.add_argument("--ctx", type=int, default=CONTEXT_SIZE, help="Context size")
    ap.add_argument("--no-server", action="store_true",
                     help="Don't start server; connect to an existing one")
    ap.add_argument("--think", action="store_true", help="Start in reasoning mode")
    ap.add_argument("--no-color", action="store_true", help="Disable colours")
    args = ap.parse_args()

    _NO_COLOR = args.no_color or not sys.stdout.isatty()

    ui = ChatUI()
    server = LlamaServer(port=args.port, model=args.model,
                         ngl=args.ngl, ctx=args.ctx)

    # ---- server setup ----------------------------------------------------
    if args.no_server:
        if not server.is_healthy():
            ui.err(f"No server on port {args.port}")
            sys.exit(1)
        ui.sys(f"Connected to existing server on port {args.port}")
    else:
        if server.is_healthy():
            ui.sys(f"Reusing running server on port {args.port}")
        else:
            binary = server.find_binary()
            if not binary:
                ui.err("llama-server not found! Searched:")
                for p in LLAMA_SERVER_SEARCH:
                    if p:
                        ui.err(f"  {p}")
                sys.exit(1)
            if not Path(args.model).exists():
                ui.err(f"Model file not found: {args.model}")
                sys.exit(1)
            ui.sys(f"Starting llama-server ({Path(args.model).name})…")
            if not server.start():
                ui.err("Server failed to start (check VRAM / model path)")
                sys.exit(1)
            ui.sys("Server ready ✓")

    agent = Agent(server, thinking=args.think)
    managed = not args.no_server and server.process is not None

    def cleanup(sig=None, frame=None):
        if managed:
            ui.sys("Stopping server…")
            server.stop()
        print()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    ui.banner(server, agent)

    # ---- main loop -------------------------------------------------------
    while True:
        try:
            raw = ui.prompt()
        except (KeyboardInterrupt, EOFError):
            cleanup()
            break

        if raw is None:
            cleanup()
            break
        if not raw:
            continue

        # multi-line: end with backslash
        while raw.endswith("\\"):
            raw = raw[:-1] + "\n"
            try:
                raw += input(f"{_c(C.GRAY)}  … {_c(C.RESET)}")
            except (KeyboardInterrupt, EOFError):
                break

        # slash commands
        if raw.startswith("/"):
            if handle_command(raw, agent, ui, server) == "quit":
                cleanup()
                break
            continue

        # chat turn
        ui.start_response()
        try:
            content, thinking, usage = agent.chat(
                raw,
                on_think=ui.on_think,
                on_content=ui.on_content,
                on_tool=ui.on_tool,
                on_tool_result=ui.on_tool_result,
            )
        except KeyboardInterrupt:
            print(f"\n{_c(C.GRAY)}  (interrupted){_c(C.RESET)}")
            continue
        ui.end_response(usage)


if __name__ == "__main__":
    main()
