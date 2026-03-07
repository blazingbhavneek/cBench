# Ag-LiveCodeBench-X

This repository contains scripts used in the [Agnostics project](https://agnostics.abgru.me) to evaluate models on **Ag-LiveCodeBench-X**, a multi-PL variant of LiveCodeBench which is more of a challenge than MultiPL-E.

You can find out more about the Agnostics project, including related artifacts, on [its website](https://agnostics.abgru.me).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
   - [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
   - [MCP (Model Context Protocol)](#mcp-model-context-protocol)
5. [CLI Reference](#cli-reference)
6. [Docker Verifier](#docker-verifier)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Using uv (recommended)
uv run main.py completions \
    --model-name openai/qwen3_8b_awq \
    --completions-path completions.jsonl \
    --temperature 0.2 \
    --num-concurrent 50 \
    --max-tokens 2048 \
    --language "C /nothink"

uv run main.py executions \
    --container-name ghcr.io/nuprl/agnostics:c \
    --timeout-seconds 15 \
    --generations-path completions.jsonl \
    --executions-path executions.jsonl \
    --num-concurrent 50

uv run main.py pass1 executions.jsonl
```

---

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Using pip

```bash
pip install -r requirements.txt
```

### Required Dependencies

- `datasets` - For loading Ag-LiveCodeBench-X from Hugging Face
- `openai` - For OpenAI-compatible API calls
- `chromadb` - For RAG vector storage (optional, for RAG feature)
- `mcp` - For MCP tool integration (optional, for MCP feature)
- `transformers` - For tokenization (optional, for thinking budget)

---

## Basic Usage

### Commands

The `main.py` script supports several subcommands:

#### 1. `completions` - Generate Solutions

```bash
python main.py completions \
    --model-name "your-model" \
    --base-url "http://localhost:8000/v1" \
    --api-key "your-api-key" \
    --completions-path ./completions.jsonl \
    --language "C" \
    --temperature 0.6 \
    --num-concurrent 20 \
    --max-tokens 5000
```

#### 2. `executions` - Run Test Cases

```bash
python main.py executions \
    --container-name "cbench" \
    --timeout-seconds 10 \
    --generations-path ./completions.jsonl \
    --executions-path ./executions.jsonl \
    --num-concurrent 20
```

#### 3. `refinements` - Fix Failed Solutions

```bash
python main.py refinements \
    --model-name "your-model" \
    --base-url "http://localhost:8000/v1" \
    --api-key "your-api-key" \
    --executions-path ./executions.jsonl \
    --refinements-path ./refinements.jsonl \
    --completions-path ./refined_completions.jsonl \
    --language "C"
```

#### 4. `iterative` - Full Refinement Pipeline

```bash
python main.py iterative \
    --model-name "your-model" \
    --base-url "http://localhost:8000/v1" \
    --api-key "your-api-key" \
    --container-name "cbench" \
    --timeout-seconds 10 \
    --output-dir ./results \
    --language "C" \
    --num-problems 20 \
    --max-refinement-iterations 3
```

#### 5. `pass1` - Summarize Results

```bash
python main.py pass1 ./executions.jsonl
```

---

## Advanced Features

### RAG (Retrieval-Augmented Generation)

RAG allows the LLM to search through your documentation files and retrieve relevant context during generation.

#### Setup

```bash
# Install ChromaDB
pip install chromadb

# Create documents directory
mkdir -p ./client/data

# Add your markdown documentation
cp /path/to/c_reference.md ./client/data/
```

#### Usage

```bash
python main.py completions \
    --model-name "your-model" \
    --base-url "http://localhost:8000/v1" \
    --api-key "your-api-key" \
    --completions-path ./results.jsonl \
    --language "C" \
    --use-rag \
    --rag-data-dir "./client/data" \
    --rag-embedding-base-url "http://localhost:8000/v1" \
    --rag-embedding-model "bge-m3"
```

#### How It Works

1. On startup, all `.md` files in `--rag-data-dir` are loaded and chunked
2. Embeddings are created via your embedding API
3. Stored in in-memory ChromaDB
4. During generation, relevant context is retrieved and added to prompts

**Note:** RAG is disabled by default. Enable with `--use-rag`.

---

### MCP (Model Context Protocol)

MCP allows the LLM to connect to external tools and services (filesystem, databases, APIs, etc.).

#### Setup

```bash
# Install MCP library
pip install mcp
```

#### Configuration

Create a JSON configuration file. You can save it anywhere, for example:
- `./mcp_config.json` (project root)
- `~/.config/mcp_config.json` (home directory)
- `/path/to/project/mcp_config.json` (custom path)

**Example: Filesystem Server**

Save as `./mcp_config.json`:

```json
{
    "servers": [
        {
            "name": "filesystem",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        }
    ],
    "timeout": 30
}
```

**Example: Multiple Servers**

```json
{
    "servers": [
        {
            "name": "filesystem",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"]
        },
        {
            "name": "database",
            "transport": "sse",
            "url": "http://localhost:8080/sse"
        },
        {
            "name": "git",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-git"]
        }
    ],
    "timeout": 60
}
```

#### Usage

```bash
python main.py completions \
    --model-name "your-model" \
    --base-url "http://localhost:8000/v1" \
    --api-key "your-api-key" \
    --completions-path ./results.jsonl \
    --language "C" \
    --use-mcp \
    --mcp-config-path ./mcp_config.json
```

#### Available MCP Servers

| Server | Command | Description |
|--------|---------|-------------|
| Filesystem | `npx -y @modelcontextprotocol/server-filesystem /path` | Read/write files |
| PostgreSQL | `npx -y @modelcontextprotocol/server-postgres postgres://...` | Query databases |
| Git | `npx -y @modelcontextprotocol/server-git` | Git operations |
| GitHub | `npx -y @modelcontextprotocol/server-github` | GitHub API |
| Puppeteer | `npx -y @modelcontextprotocol/server-puppeteer` | Browser automation |
| SQLite | `npx -y @modelcontextprotocol/server-sqlite /path/to/db` | SQLite queries |

See [MCP Servers](https://github.com/modelcontextprotocol/servers) for more.

**Note:** MCP is disabled by default. Enable with `--use-mcp`.

---

### Using RAG and MCP Together

You can enable both features simultaneously:

```bash
python main.py completions \
    --model-name "your-model" \
    --base-url "http://localhost:8000/v1" \
    --api-key "your-api-key" \
    --completions-path ./results.jsonl \
    --language "C" \
    --use-rag \
    --rag-data-dir "./client/data" \
    --rag-embedding-base-url "http://localhost:8000/v1" \
    --rag-embedding-model "bge-m3" \
    --use-mcp \
    --mcp-config-path ./mcp_config.json
```

---

## CLI Reference

### Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-name` | Model name (required) | - |
| `--base-url` | API base URL | `http://localhost:8000/v1` |
| `--api-key` | API key | `None` |
| `--temperature` | Sampling temperature | `0.6` |
| `--num-concurrent` | Concurrent requests | `20` |
| `--max-tokens` | Max tokens in response | `5000` |
| `--top-p` | Top-p sampling | `0.95` |
| `--language` | Programming language (required) | - |
| `--reasoning-effort` | Reasoning effort level | `medium` |
| `--max-retries` | Max retry attempts | `3` |
| `--use-thinking-budget` | Enable thinking budget | `False` |
| `--tokenizer-name-or-path` | Tokenizer for thinking budget | `None` |
| `--max-thinking-budget` | Max thinking tokens | `512` |
| `--max-agent-iterations` | Max RAG/Web search iterations | `0` |
| `--summarize-context` | Summarize retrieved context | `False` |
| `--cache-dir` | Dataset cache directory | `None` |

### RAG Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--use-rag` | Enable RAG | `False` |
| `--rag-data-dir` | Directory with markdown files | `None` |
| `--rag-embedding-base-url` | Embedding API URL | `http://localhost:8000/v1` |
| `--rag-embedding-api-key` | Embedding API key | `None` |
| `--rag-embedding-model` | Embedding model | `bge-m3` |

### MCP Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--use-mcp` | Enable MCP | `False` |
| `--mcp-config-path` | Path to JSON config | `None` |
| `--mcp-timeout` | Tool call timeout (seconds) | `30` |

### Command-Specific Arguments

#### completions
- `--completions-path` - Output file path (required)
- `--num-completions` - Solutions per problem (default: 1)
- `--num-problems` - Limit number of problems (default: all)

#### executions
- `--container-name` - Docker container name (required)
- `--timeout-seconds` - Execution timeout (required)
- `--generations-path` - Input solutions file (required)
- `--executions-path` - Output results file (required)

#### refinements
- `--executions-path` - Input executions file (required)
- `--refinements-path` - Output training data file (required)
- `--completions-path` - Output refined solutions file (required)

#### iterative
- `--container-name` - Docker container name (required)
- `--timeout-seconds` - Execution timeout (required)
- `--output-dir` - Output directory (required)
- `--num-completions` - Solutions per problem (default: 1)
- `--max-refinement-iterations` - Max refinement rounds (default: 3)
- `--num-problems` - Number of problems (default: 20)

---

## Docker Verifier

The Docker verifier (`Docker/verify.py`) executes C code with GMP and uthash support.

### Building the Verifier

```bash
cd Docker
docker build -t cbench .
```

The Dockerfile verifies that all required libraries (GMP, math, uthash) are installed by compiling and running a test program.

### Verifier Protocol

The verifier communicates over JSON on stdin/stdout:

**Input:**
```json
{
    "code": "int main() { return 0; }",
    "timeout_s": 10,
    "test_cases": [
        {"input": "1\n2", "output": "3"}
    ]
}
```

**Output:**
```json
{"result": "success", "stderr": ""}
```

Or on failure:
```json
{"result": "fail:error", "exit_code": 1, "stdout": "", "stderr": "..."}
```

---

## Troubleshooting

### Common Issues

#### "No module named 'datasets'"
```bash
pip install datasets
# or
uv sync
```

#### "ChromaDB not installed"
```bash
pip install chromadb
```

#### "MCP library not installed"
```bash
pip install mcp
```

#### "Failed to connect to Docker"
```bash
# Start Docker daemon
sudo systemctl start docker
# or use sudo
sudo docker run ...
```

#### "LLM response was truncated"
- Increase `--max-tokens`
- Add `/nothink` to `--language` to disable thinking for models that support it

#### "Tool call timed out"
```bash
--mcp-timeout 60
```

#### "No markdown files found" (RAG)
- Check that `--rag-data-dir` points to the correct directory
- Ensure files have `.md` or `.markdown` extension

### Getting Help

- Check logs for detailed error messages
- Ensure all required dependencies are installed
- Verify API endpoints and keys are correct
- See [RAG and MCP Setup](#rag-retrieval-augmented-generation) for feature-specific troubleshooting

---

## Additional Resources

- [Agnostics Project](https://agnostics.abgru.me)
- [LiveCodeBench](https://livecodebench.github.io/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [MCP Servers](https://github.com/modelcontextprotocol/servers)
- [ChromaDB Documentation](https://docs.trychroma.com/)
