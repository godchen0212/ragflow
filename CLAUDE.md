# CLAUDE.md

Guidance for Claude Code when working with this RAGFlow repository.

## Project Overview

RAGFlow: Open-source RAG engine with deep document understanding.
- Backend: Python (Quart async API server)
- Frontend: React/TypeScript (Vite + UmiJS)
- Data stores: MySQL, Elasticsearch/Infinity/OceanBase, Redis, MinIO

## Architecture

### Two-Process Model
- **API Server** (`api/ragflow_server.py`) - Quart HTTP server
- **Task Executors** (`rag/svr/task_executor.py`) - Workers for document ingestion via Redis streams

### Backend (`/api/`)
- Blueprint auto-registration: `api/apps/__init__.py` discovers `*_app.py` files
- Routes: `/{API_VERSION}/{page_name}` (e.g., `/v1/kb`)
- Database: Peewee ORM with MySQL/Postgres/OceanBase
- Services in `api/db/services/` extend `CommonService`

### Agent System (`/agent/`)
- DAG-based workflow: `agent/canvas.py` loads JSON DSL
- Components auto-discovered in `agent/component/`
- Tools in `agent/tools/` (Tavily, Wikipedia, GitHub, SQL, etc.)

### Task Executor
- Reads tasks from Redis → fetches files from MinIO → parses → chunks → stores
- Parser types: `naive`, `paper`, `book`, `laws`, etc.
- Task types: `dataflow`, `raptor`, `graphrag`, `mindmap`, `memory`

### Memory System (`/memory/`)
- Extracts semantic/episodic/procedural memories from conversations
- **Keywords**: User preferences/opinions stored as structured metadata (not vectorized)
- **Filtering**: Exact keyword matching via `filter_by_keywords()` with OR logic
- **Backends**: Native (ES/Infinity/OceanBase) or mem0
- **API**: `GET /v1/messages/search?filter_keywords=python,fastapi`
- See `MEMORY_KEYWORDS_DESIGN.md` for details

### Core Processing (`/rag/`)
- `rag/llm/` - LLM abstractions (chat, embedding, rerank, vision, TTS, STT)
- `rag/flow/` - Chunking, parsing, tokenization
- `rag/graphrag/` - Knowledge graph construction

### Configuration

`common/settings.py` merges: `conf/service_conf.yaml` → environment variables → Docker env vars
- `conf/service_conf.yaml` - Primary config (DB, Redis, MinIO, ES, LLM)
- `docker/.env` - Environment overrides

### Frontend (`/web/`)
React 18 + TypeScript, Ant Design + Tailwind, Zustand + React Query

### SDK (`/sdk/`)
Python SDK in `sdk/python/ragflow_sdk/`

## Development Commands

### Backend
```bash
uv sync --python 3.12 --all-extras
docker compose -f docker/docker-compose-base.yml up -d
# Add to /etc/hosts: 127.0.0.1 es01 infinity mysql minio redis
bash docker/launch_backend_service.sh
pkill -f "ragflow_server.py|task_executor.py"  # Stop
```

### Testing
```bash
uv run pytest                                    # all tests
uv run pytest test/testcases/test_xxx.py        # specific file
uv run pytest -m p1                             # by priority
```

### Linting
```bash
ruff check && ruff format
pre-commit run --all-files
```

### Frontend
```bash
cd web && npm install && npm run dev
```

### Docker
```bash
docker compose -f docker/docker-compose.yml up -d
docker build --platform linux/amd64 -f Dockerfile -t infiniflow/ragflow:nightly .
```

## Key Notes

- **Database Engine**: Set `DOC_ENGINE` in `docker/.env` (elasticsearch/infinity/oceanbase)
- **Python**: >=3.12, <3.15
- **Node.js**: >=18.20.4
- **Docker**: >=24.0.0, Compose >=v2.26.1
- **HuggingFace Mirror (China)**: `export HF_ENDPOINT=https://hf-mirror.com`
