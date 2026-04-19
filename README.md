# AROC - Agentic Read-Only Chat

## chat.py — agentic read only chat 

> Uses: g023/Qwen3.5-9B-IQ2_M (https://huggingface.co/g023/g023-Qwen3.5-9B-GGUF) (download and toss it in the same folder as chat.py - designed for the IQ2_M model)

> Requires: locally installed llama-server (https://github.com/ggerganov/llama.cpp) by Georgi Gerganov

> Author: g023

> License: MIT

---

# AROC - chat.py

**Version 0.1** — Self-contained agentic chat interface for Qwen3.5-9B-IQ2_M

A rich terminal-based chat application powered by llama.cpp, featuring an auto-managed server backend, reasoning modes, and 7 built-in filesystem tools for interactive AI assistance.

## Features

- **Auto-managed llama-server**: Automatically starts and manages a llama.cpp server with optimized anti-looping sampling parameters
- **Reasoning modes**: Toggle between step-by-step thinking display (`/think`) and faster responses (`/nothink`)
- **7 built-in tools**: Read files, list directories, find files, grep search, file metadata, get time, and subagent file analysis
- **Streaming output**: Real-time token streaming with interleaved thinking display
- **Multi-turn conversations**: Context-aware with automatic token pruning and session persistence
- **Tool calling chains**: Models can invoke multiple tools in sequence with subagent delegation for complex tasks
- **Session management**: Save/load conversations to/from JSON files
- **Zero dependencies**: Pure Python stdlib implementation

## Quick Start

1. **Prerequisites**:
   - Linux system with CUDA-compatible GPU (tested on RTX 3060 12GB)
   - llama.cpp binary in PATH 
   - Qwen3.5-9B-IQ2_M.gguf model file in the same directory

2. **Run**:
   ```bash
   python3 chat.py
   ```

The application will automatically start a llama-server on port 19300 and launch the interactive chat interface.

## Usage

### Command Line Options

```bash
python3 chat.py [OPTIONS]

Options:
  --port PORT           Server port (default: 19300)
  --model PATH          GGUF model file path (default: g023-Qwen3.5-9B-IQ2_M.gguf)
  --ngl LAYERS          GPU layers to offload (default: 40)
  --ctx TOKENS          Context window size (default: 16384)
  --no-server           Connect to existing server instead of starting one
  --think               Start in reasoning mode (thinking displayed)
  --no-color            Disable ANSI color output
  --help                Show help message
```

### Interactive Commands

Type these commands during chat:

- `/think`      — Enable reasoning mode (step-by-step thinking)
- `/nothink`    — Disable reasoning mode (faster responses)
- `/clear`      — Clear conversation history
- `/save FILE`  — Save session to JSON file
- `/load FILE`  — Load session from JSON file
- `/tokens`     — Show token usage statistics
- `/tools`      — List available tools
- `/model`      — Show model and sampling info
- `/help`       — Show help
- `/quit`       — Exit

### Multi-line Input

End a line with `\` to continue input on the next line.

## Tools

The AI has access to 7 read-only filesystem tools:

1. **read_file** — Read file contents with optional line ranges
2. **list_dir** — List directory contents with sizes and dates
3. **find_files** — Find files matching glob patterns
4. **grep** — Search for regex patterns in files
5. **file_info** — Get file metadata (size, permissions, etc.)
6. **get_time** — Get current date, time, and system uptime
7. **analyze_file** — Delegate deep file analysis to a focused subagent

## Configuration

### Model Settings

- **Model**: Qwen3.5-9B-IQ2_M.gguf (3.4GB, 2-bit quantization) (vroom vroom)
- **Architecture**: Hybrid Mamba2-Attention (32 layers, 8 attention layers)
- **Context**: 16384 tokens
- **KV Cache**: q4_0 quantization

### Sampling Parameters (Anti-looping optimized)

- Temperature: 0.3
- Top-p: 0.9
- Top-k: 40
- Min-p: 0.05
- Repeat penalty: 1.15
- Frequency penalty: 0.2
- Presence penalty: 0.0

### Server Configuration

- Batch size: 512
- Threads: min(8, CPU count)
- Host: 127.0.0.1
- Max concurrent requests: 1

## Examples

### Basic Chat
```
You ▸ What is 2+2?
AI ▸ 4
```

### Using Tools
```
You ▸ Read the first 5 lines of /etc/hostname
AI ▸ 🔧 read_file(path="/etc/hostname", start_line=1, end_line=5)
     ┌─ result
     │ megabox
     └─
AI ▸ The file contains: megabox
```

### Reasoning Mode
```
You ▸ /think
⚙ Reasoning mode ON — model will think step-by-step

You ▸ Explain how addition works
AI ▸ 💭 Thinking process:
     1. **Understand the request**: The user asked to explain how addition works...
AI ▸ Addition is the mathematical operation of combining two or more numbers...
```

### Session Management
```
You ▸ /save my_session.json
⚙ Session saved → my_session.json

You ▸ /load my_session.json
⚙ Loaded 12 messages from my_session.json
```

## Requirements

- **Python**: 3.8+ (stdlib only)
- **Hardware**: CUDA GPU with 12GB+ VRAM (tested on RTX 3060 w/12GB)
- **Memory**: 64GB+ RAM recommended
- **OS**: Linux
- **llama.cpp**: Build 8826+ with reasoning support

## Testing

Run the comprehensive test suite:

```bash
python3 chat_test.py
```

Includes 70 unit tests and 5 integration tests (requires running server).

## License

MIT License

## Contributing

Contributions welcome! Please test thoroughly and ensure all tests pass.

## Version History

- **0.1** (2026-04-18): Initial release with full agentic chat functionality
