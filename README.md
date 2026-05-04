# context-forge (ctxf)

**Self-hostable Agentic Context Engineering** — evolve your AI playbook automatically.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)

Based on the paper: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618) (Stanford + SambaNova, 2025).

---

## What is this?

ACE (Agentic Context Engineering) is a framework that treats LLM context — system prompts, agent memory, instructions — as an **evolving playbook** that self-improves through a structured loop:

```
Generator → Reflector → Curator → Merge → Refine
```

Instead of rewriting prompts from scratch (which causes *context collapse*), ACE performs small **delta updates** — like git commits for your prompts. Each bullet in the playbook tracks helpful/harmful counters, gets deduplicated, and evolves over time.

**context-forge** makes ACE production-ready:
- Self-hostable (Docker, any cloud, bare metal)
- REST API + CLI + MCP server
- Works with any OpenAI-compatible LLM provider (OpenAI, Anthropic via proxy, Ollama, etc.)
- SQLite storage — zero external dependencies
- Hybrid retrieval (BM25 keyword + vector similarity)

## Quick Start

### Install

```bash
pip install ctxf
# or with uv:
uv pip install ctxf
```

### Set up

```bash
# Initialize a playbook
ctxf init --task my-project

# Set your LLM provider
export CTXF_LLM_BASE_URL="https://api.openai.com/v1"
export CTXF_LLM_API_KEY="sk-..."
export CTXF_LLM_MODEL="gpt-4o"
```

### Use the CLI

```bash
# Search playbook for relevant context
ctxf retrieve "how to handle database errors"

# Analyze a task outcome and extract insights
ctxf reflect --doc task_result.json

# Convert insights into playbook updates
ctxf curate --reflection reflection.json

# Apply updates to the playbook
ctxf commit --delta delta.json

# Or run the full loop in one command
ctxf evolve --doc task_result.json

# View playbook statistics
ctxf stats
```

### Start the API Server

```bash
ctxf serve --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /v1/retrieve` — Search playbook for relevant bullets
- `POST /v1/feedback` — Submit task outcome for reflection
- `POST /v1/evolve` — Run full reflect→curate→commit loop
- `GET /v1/playbook` — Get current playbook state
- `GET /v1/playbook/{id}` — Get specific playbook version
- `GET /v1/stats` — Playbook statistics

### MCP Server (for IDE integration)

```bash
ctxf mcp
```

Exposes these MCP tools:
- `retrieve` — Search playbook for relevant context
- `reflect` — Analyze task outcome
- `curate` — Generate delta operations from reflection
- `evolve` — Run full evolution loop
- `stats` — View playbook statistics

### Docker

```bash
docker run -p 8000:8000 \
  -e CTXF_LLM_BASE_URL=https://api.openai.com/v1 \
  -e CTXF_LLM_API_KEY=sk-... \
  -v ctxf-data:/data \
  ghcr.io/samirius/context-forge:latest
```

Or with docker-compose:

```bash
curl -O https://raw.githubusercontent.com/Samirius/context-forge/main/docker-compose.yml
docker-compose up -d
```

## Architecture

```
┌─────────────────────────────────────────────┐
│                CLI / API / MCP               │
│           (Click / FastAPI / stdio)          │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              Engine Layer                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │Generator │ │Reflector │ │ Curator  │    │
│  └──────────┘ └──────────┘ └──────────┘    │
│  ┌──────────┐ ┌──────────────────┐         │
│  │ Refiner  │ │ Hybrid Retrieval │         │
│  └──────────┘ └──────────────────┘         │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           Storage & LLM                     │
│  ┌──────────┐ ┌───────────────────────┐    │
│  │  SQLite  │ │ OpenAI-compat / Ollama│    │
│  │ + Vector │ │ (any LLM provider)    │    │
│  └──────────┘ └───────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `CTXF_DB_PATH` | `~/.ctxf/playbook.db` | SQLite database path |
| `CTXF_LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API base URL |
| `CTXF_LLM_API_KEY` | — | LLM API key |
| `CTXF_LLM_MODEL` | `gpt-4o` | Default model for all roles |
| `CTXF_GENERATOR_MODEL` | — | Model for generator (overrides LLM_MODEL) |
| `CTXF_REFLECTOR_MODEL` | — | Model for reflector |
| `CTXF_CURATOR_MODEL` | — | Model for curator |
| `CTXF_RETRIEVAL_TOP_K` | `10` | Number of bullets to retrieve |
| `CTXF_REFINE_THRESHOLD` | `0.85` | Similarity threshold for dedup |
| `CTXF_MAX_BULLETS` | `500` | Max bullets per playbook |
| `CTXF_HOST` | `127.0.0.1` | API server host |
| `CTXF_PORT` | `8000` | API server port |

## How It Works

### The ACE Loop

1. **Generator** executes a task using relevant playbook bullets as context
2. **Reflector** analyzes the outcome — what worked, what failed, what was missing
3. **Curator** converts reflections into structured delta operations (ADD/PATCH/INCR/DEPRECATE/MERGE)
4. **Merge** applies deltas deterministically to the playbook
5. **Refiner** periodically deduplicates and prunes low-value bullets

### Delta Operations

| Operation | Description |
|-----------|-------------|
| `ADD` | Add a new bullet to a section |
| `PATCH` | Update an existing bullet's content |
| `INCR` | Increment helpful/harmful counter |
| `DEPRECATE` | Soft-delete a bullet |
| `MERGE` | Combine similar bullets into one |

### Playbook Format

```
## STRATEGIES
[STR-00001] helpful=5 harmful=0 :: Always validate input before processing
[STR-00002] helpful=3 harmful=1 :: Consider edge cases in async code

## COMMON MISTAKES
[MIS-00003] helpful=8 harmful=0 :: Don't forget to handle connection timeouts

## INSIGHTS
[INS-00004] helpful=6 harmful=0 :: Batch operations are 10x faster than individual calls
```

## Comparison with Official ACE

| Feature | ace-agent/ace (Official) | context-forge (This) |
|---------|-------------------------|---------------------|
| Language | Python (research) | Python (production) |
| CLI | ❌ | ✅ Click-based |
| REST API | ❌ | ✅ FastAPI |
| MCP Server | ❌ | ✅ stdio |
| Storage | File-based | SQLite + versioning |
| LLM Providers | SambaNova only | Any OpenAI-compatible |
| Self-hosting | ❌ | ✅ Docker |
| Retrieval | Basic | Hybrid (BM25 + vector) |
| Web Dashboard | ❌ | Planned |

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

- [ACE Paper](https://arxiv.org/abs/2510.04618) — Zhang et al., Stanford + SambaNova
- [DannyMac180/ACE](https://github.com/DannyMac180/ACE) — Reference MCP implementation
