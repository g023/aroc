# AROC - Agentic Read-Only Chat

## chat.py — agentic read only chat 

> Uses: g023/Qwen3.5-9B-IQ2_M (https://huggingface.co/g023/g023-Qwen3.5-9B-GGUF) (download and toss it in the same folder as chat.py - designed for the IQ2_M model)

> Requires: locally installed llama-server (https://github.com/ggerganov/llama.cpp) by Georgi Gerganov

> Author: g023

> License: MIT

**Note:** Attempts are made to make this read only, but I might not have stopped everything, so only consider this for testing in a container or sandbox environment.

```
A rich terminal chat interface powered by llama.cpp with:
  • Auto-managed llama-server backend with anti-looping optimizations
  • Reasoning (/think) and non-reasoning (/nothink) modes
  • 7 built-in tools: read_file, list_dir, find_files, grep, file_info,
    get_time, analyze_file (subagent)
  • Streaming token output with interleaved thinking display
  • Multi-turn tool calling chains with subagent delegation
  • Context-aware token management and session save/load
```

```
Usage:
  python3 chat.py                      Start with defaults
  python3 chat.py --port 8080          Custom port
  python3 chat.py --no-server          Connect to existing server
  python3 chat.py --think              Start in reasoning mode
  python3 chat.py --no-color           Disable ANSI colors
```
