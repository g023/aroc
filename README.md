<p align="center">
  <h1 align="center">AROC — Agentic Read-Only Chat</h1>
  <p align="center">
    A self-contained agentic terminal chat with 19 tools, powered by Qwen3.5-9B and llama.cpp
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/version-2.0.0-blue" alt="Version">
    <img src="https://img.shields.io/badge/python-3.8+-green" alt="Python">
    <img src="https://img.shields.io/badge/dependencies-stdlib_only-brightgreen" alt="Dependencies">
    <img src="https://img.shields.io/badge/license-MIT-orange" alt="License">
    <img src="https://img.shields.io/badge/GPU-CUDA-76B900" alt="CUDA">
  </p>
</p>

---

**AROC** is an offline-first, read-only agentic chat application that runs entirely on local hardware. It combines a 2-bit quantized 9B parameter model with 19 built-in tools for filesystem exploration, code analysis, task management, and working memory — all in a single Python file with zero pip dependencies.

## Highlights

- **19 built-in tools** — filesystem ops, code analysis, todo tracking, session memory
- **Single file** — one `chat.py`, pure Python stdlib, no installs
- **Offline** — runs fully local on consumer GPUs (tested: RTX 3060 12GB)
- **64K context** — large conversations with automatic pruning
- **Agentic** — multi-turn tool chains up to 12 rounds per message
- **Persistent state** — scratch pad, todos, and memory survive session save/load

## Quick Start

```bash
# Prerequisites: llama-server binary + model file
# Get llama.cpp: https://github.com/ggerganov/llama.cpp
# Get model:     https://huggingface.co/g023/g023-Qwen3.5-9B-GGUF

# Place g023-Qwen3.5-9B-IQ2_M.gguf alongside chat.py, then:
python3 chat.py
```

AROC auto-starts a llama-server, loads the model, and drops you into an interactive chat. To connect to an existing server:

```bash
python3 chat.py --no-server --port 19300
```

## Tools

<details>
<summary><b>Filesystem (8 tools)</b></summary>

| Tool | Description |
|------|-------------|
| `read_file` | Read with optional line ranges (default max 200 lines) |
| `head` | First N lines — quick file previews |
| `tail` | Last N lines — logs and recent content |
| `list_dir` | Directory listing with sizes and dates |
| `find_files` | Glob pattern file search |
| `grep` | Regex search across files |
| `grep_context` | Regex search with surrounding context lines |
| `file_info` | Size, permissions, timestamps, line count |

</details>

<details>
<summary><b>Code Analysis (3 tools)</b></summary>

| Tool | Description |
|------|-------------|
| `python_outline` | AST-based class/function extraction with line numbers |
| `diff_files` | Unified diff between two files |
| `analyze_file` | Subagent delegation for deep file analysis |

</details>

<details>
<summary><b>Task Management (4 tools)</b></summary>

| Tool | Description |
|------|-------------|
| `todo_add` | Add task with priority (high/medium/low) |
| `todo_list` | List all todos with completion status |
| `todo_done` | Mark task as done |
| `todo_remove` | Remove task |

</details>

<details>
<summary><b>Memory & State (4 tools)</b></summary>

| Tool | Description |
|------|-------------|
| `scratch_pad` | Overwrite-style working notes |
| `memory_append` | Append timestamped notes (accumulates) |
| `memory_read` | Read all session memory |
| `get_time` | Current date, time, and uptime |

</details>

## Architecture

```
┌───────────────────────────────────────────────────┐
│  ChatUI  (terminal rendering, slash commands)      │
│    ↕                                               │
│  Agent  (tool dispatch, state, conversation loop)  │
│    ↕                                               │
│  LlamaServer  (HTTP → llama-server process)        │
│    ↕                                               │
│  llama-server  (GGUF inference, CUDA)              │
└───────────────────────────────────────────────────┘
```

The Agent runs a loop: send messages → stream response → if tool calls, execute and loop (up to 12 turns) → if no tool calls, deliver final answer.

Three in-session state mechanisms complement each other:

| State | Behavior | Use Case |
|-------|----------|----------|
| **Scratch pad** | Overwrite | Current plan / working notes |
| **Memory** | Append-only | Findings, decisions, facts |
| **Todos** | Structured list | Multi-step task tracking |

## Commands

| Command | Action |
|---------|--------|
| `/think` | Enable reasoning display |
| `/nothink` | Disable reasoning (faster) |
| `/clear` | Clear conversation |
| `/save FILE` | Save session (messages + state) |
| `/load FILE` | Restore session |
| `/pad` | Show scratch pad |
| `/clearpad` | Clear scratch pad |
| `/todos` | Show todo list |
| `/memory` | Show memory notes |
| `/tokens` | Token usage stats |
| `/tools` | List tools |
| `/model` | Model info |
| `/quit` | Exit |

**Multi-line input:** end a line with `\`  
**Interrupt:** `Ctrl+C` during generation

## Configuration

### CLI Options

```
--port PORT     Server port (default: 19300)
--model PATH    GGUF model path
--ngl LAYERS    GPU layers (default: 36)
--ctx TOKENS    Context window (default: 64000)
--no-server     Connect to existing server
--think         Start in reasoning mode
--no-color      Disable ANSI colors
```

### Model

| Spec | Value |
|------|-------|
| Model | g023-Qwen3.5-9B-IQ2_M.gguf (3.4GB) |
| Quantization | IQ2_M (2-bit) |
| Architecture | Qwen3.5-9B Hybrid Mamba2-Attention |
| Context | 64,000 tokens |
| KV Cache | q4_0 quantized |

### Sampling

Tuned specifically to prevent repetition loops at 2-bit quantization:

```
temperature=0.3  top_p=0.9  top_k=40  min_p=0.05
repeat_penalty=1.15  frequency_penalty=0.2  presence_penalty=0.0
```

## Key Technical Detail: reasoning_format

> **Critical for IQ2_M models**: The `reasoning_format: "deepseek"` server parameter is required for the model to produce actual response content. Without it, all tokens go to internal reasoning and `content` stays empty. This is a server-level parameter — prompt engineering cannot fix it.

```python
# Applied to all API calls:
extra_sampling = {"reasoning_format": "deepseek"}
```

## Examples

```
You ▸ What's the structure of chat.py?
  🔧 python_outline(path="chat.py")
AI ▸ chat.py has 1887 lines with 5 main classes: LlamaServer, StreamParser, ...

You ▸ Compare line counts of chat.py and chat_test.py
  🔧 file_info(path="chat.py")
  🔧 file_info(path="chat_test.py")
AI ▸ chat.py: 1,887 lines. chat_test.py: 709 lines.

You ▸ Create a review plan for error handling
  🔧 todo_add(task="Find all try/except blocks", priority="high")
  🔧 todo_add(task="Check error messages", priority="medium")
AI ▸ Created 2 todos. Use /todos to track progress.
```

## Requirements

| Requirement | Details |
|-------------|---------|
| Python | 3.8+ (stdlib only) |
| GPU | CUDA-compatible, 12GB+ VRAM |
| RAM | 16GB+ (64GB recommended) |
| OS | Linux |
| llama.cpp | With `reasoning_format` support |

## Testing

```bash
python3 chat_test.py
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **2.0.0** | 2026-04-19 | 19 tools, reasoning_format fix, 64K context, in-session state, reasoning fallback |
| 0.1 | 2026-04-18 | Initial release (7 tools, 16K context) |

## License

MIT License — see [LICENSE](LICENSE)

## Author

**g023** — [HuggingFace](https://huggingface.co/g023)
