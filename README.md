# Facility Research Pipeline

Stateless FastAPI control API + asyncio worker that researches supplier facilities via grounded Gemini, geocodes them, and persists to Supabase. See [PLAN.md](PLAN.md) for the full design.

## Quick start

```bash
cp .env.example .env   # fill in keys; set SUPABASE_SCHEMA if not using public
# Apply migrations/001_init.sql in Supabase (schema must match SUPABASE_SCHEMA)
docker compose up --build
```

API: `http://localhost:8000` — Worker health: `http://localhost:8080/healthz`

```bash
# Local (without Docker for app processes)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uvicorn api.main:app --reload --port 8000
python -m worker.main
pytest
```

## Catalog (discovered data)

Read-only Supabase-backed endpoints (API needs `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SUPABASE_SCHEMA`):

- `GET /suppliers` — list; `q`, `limit`, `offset`
- `GET /suppliers/{id}`
- `GET /facilities` — list; `supplier_id`, `product`, `facility_type`, `limit`, `offset`
- `GET /facilities/{id}`

## Cloud Run notes

- **API**: request-based CPU, min instances 0, concurrency 80, 512 MiB / 1 vCPU
- **Worker**: CPU always allocated, min/max instances 1 initially, 512 MiB / 1 vCPU; listens on `WORKER_HEALTH_PORT` (default 8080)
- Secrets via Secret Manager; both services use VPC connector `cloudrun-connector`
- `REDIS_URL` secret must be the Memorystore private IP (`redis://10.x.x.x:6379/0`), not localhost
- API and worker set `SUPABASE_SCHEMA=supply_res_dev` (must match migrations)
