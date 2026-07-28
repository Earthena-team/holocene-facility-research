# Implementation Plan: Supplier Facility Research Pipeline

**Audience:** coding agent. Build exactly what is specified here. Where a decision is marked *(decision)*, it is final — do not substitute alternatives without flagging.

---

## 1. System Overview

Two deployables plus existing infrastructure:

```
┌─────────────┐   HTTP    ┌─────────────┐   BRPOPLPUSH   ┌─────────────┐
│   Clients   │ ───────►  │ Control API │ ──► Redis ──►  │   Worker    │
└─────────────┘           │ (stateless) │    (queue +    │ (pipeline   │
                          └─────────────┘     job state)  │  graph)     │
                                                          └──────┬──────┘
                                                                 │
                                              ┌──────────────────┼──────────────┐
                                              ▼                  ▼              ▼
                                        Gemini API      Google Geocoding    Supabase
                                        (grounded)           API           (Postgres)
```

- **Control API** — stateless FastAPI service. Owns nothing; reads/writes Redis only. Never calls Gemini, Geocoding, or Supabase.
- **Worker** — long-running process. Pulls jobs from Redis, executes the research pipeline (a small explicit state machine — this is "where the graph lives"), writes results to Supabase, writes status/metrics back to Redis.
- **Redis** — job queue, job state, dedup cache, geocode cache, budget counters, halt flags. Existing instance; namespace all keys under `frp:` (facility research pipeline).
- **Supabase (Postgres)** — durable store for suppliers, facilities, and research run audit records.

**Language/stack** *(decision)*: Python 3.12, FastAPI + uvicorn (API), plain asyncio worker (no Celery/RQ — the queue is simple enough that a thin Redis client keeps dependencies and failure modes minimal), `redis-py` (async), `httpx` for outbound calls, `pydantic` v2 for all schemas, `google-genai` SDK for Gemini.

**Repo layout:**

```
/facility-research
  /shared          # pydantic models, redis client, config, constants
    models.py
    redis_queue.py # the "simple Redis client" — enqueue/dequeue/ack/status
    config.py
  /api             # control API
    main.py
    routes.py
  /worker          # pipeline
    main.py        # loop: dequeue → run graph → ack
    graph.py       # state machine: RESEARCH → EXTRACT → GEOCODE → PERSIST
    gemini.py
    geocode.py
    persist.py
  /migrations      # Supabase SQL
  docker-compose.yml (local dev: api, worker, redis)
  Dockerfile.api / Dockerfile.worker
```

---

## 2. Data Model

### 2.1 Facility schema (pydantic + Postgres)

```python
class FacilityType(str, Enum):
    manufacturing = "manufacturing"
    logistics = "logistics"
    raw_material = "raw_material"

class Facility(BaseModel):
    facility_name: str                    # non-empty, ≤ 300 chars
    facility_address: str                 # non-empty, full postal address
    product: str                          # product this facility was researched for; stamped
                                          # from the job's resolved product, NOT returned by Gemini
    latitude: float | None                # -90..90; None until geocoded
    longitude: float | None               # -180..180
    facility_type: FacilityType
    confidence: Literal["high","medium","low"]   # model's own assessment
    source_url: str | None                # grounding citation if available
    geocode_status: Literal["ok","approximate","failed","pending"]
```

`product` is deterministic bookkeeping, not a research output: a job is scoped to exactly one product (either the caller's or the one DISCOVER_PRODUCT resolved), so every facility it finds relates to that product. EXTRACT stamps it (§5.3), so it stays out of `FACILITY_LIST_SCHEMA` and costs no output tokens. It is also the axis a supplier's products hang off of (§2.3).

### 2.2 Supabase tables (write migration SQL)

```sql
create table suppliers (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  normalized_name text not null unique,  -- COMPANY identity, product-independent (§2.3, §5.1)
  created_at timestamptz default now()
);

create table facilities (
  id uuid primary key default gen_random_uuid(),
  supplier_id uuid references suppliers(id) on delete cascade,
  facility_name text not null,
  facility_address text not null,
  product text not null default '',        -- product this facility was researched for (§2.1)
  latitude double precision,
  longitude double precision,
  facility_type text not null check (facility_type in ('manufacturing','logistics','raw_material')),
  confidence text not null,
  source_url text,
  geocode_status text not null default 'pending',
  created_at timestamptz default now(),
  unique (supplier_id, product, facility_name, facility_address)  -- product is part of the key
);

create table research_runs (
  id uuid primary key,                    -- = job_id
  supplier_id uuid references suppliers(id),
  product text not null default '',       -- which product this run researched (§2.3)
  status text not null,
  input_tokens int, output_tokens int, thinking_tokens int,
  search_queries int, geocode_calls int,
  est_cost_usd numeric(10,5),
  error text,
  started_at timestamptz, finished_at timestamptz
);
```

`research_runs` is the durable audit/cost record; Redis job state is operational and expires.

### 2.3 Supplier ↔ product relationship *(a supplier supplies many products)*

A supplier is a **company**, not a (company, product) pair. `suppliers` holds exactly one row per company, keyed on `normalized_name` (slug of the name alone). The products a supplier supplies are not a column — they are the distinct `product` values across its `facilities` and `research_runs`. Researching "Acme" for steel and again for bearings yields **one** supplier row with facilities partitioned by `product`, plus two `research_runs`.

This splits two concepts that used to be fused into `normalized_key`:
- **`normalized_name`** (`slug(name)`) — supplier identity in Supabase. One company, one row.
- **`normalized_key`** (`slug(name):slug(product)`) — the *per-run* dedupe/cache key (§5.1). Still exactly one (supplier, product) research inside the refresh window; the cost invariant is unchanged, it just keys on `normalized_name + product` rather than a single fused string.

Migration 003 performs this split on existing data: it backfills `normalized_name`, collapses any duplicate same-company rows into one canonical row (repointing their facilities and runs first), and swaps the constraints. Freshness (the refresh-window check) moves from `suppliers.created_at` — meaningless once a row is reused across products and runs — to the latest successful `research_runs.finished_at` for that (supplier, product).

---

## 3. Redis Design (shared/redis_queue.py)

All keys prefixed `frp:`. Implement a single class `JobQueue` used by both API and worker.

| Key | Type | Purpose | TTL |
|---|---|---|---|
| `frp:queue:pending` | LIST | main job queue (job_ids) | — |
| `frp:queue:processing` | LIST | in-flight jobs (reliable-queue pattern) | — |
| `frp:queue:dead` | LIST | jobs failed after max retries | — |
| `frp:job:{id}` | HASH | full job record (see below) | 7 days after terminal state |
| `frp:halt:{id}` | STRING "1" | halt flag, checked between pipeline steps | 24 h |
| `frp:dedupe:{normalized_key}` | STRING job_id | prevents duplicate research | 90 days *(config)* |
| `frp:geocode:{sha1(address)}` | STRING json | geocode result cache | 180 days |
| `frp:budget:{YYYY-MM-DD}` | STRING float | running est. spend today (USD) | 48 h |

**Job hash fields:** `id, supplier_name, product, normalized_key, status, step, attempts, priority, enqueued_at, started_at, finished_at, input_tokens, output_tokens, thinking_tokens, search_queries, geocode_calls, est_cost_usd, facilities_found, error, worker_id, heartbeat_at, discovered_products`. `discovered_products` is a JSON-encoded ranked list, populated only when DISCOVER_PRODUCT ran; `product` is always `discovered_products[0]` in that case (or `"general operations"` if discovery found nothing).

**Statuses:** `queued → running → succeeded | failed | halted | dead`. `step` ∈ `research | extract | geocode | persist` while running.

**Queue semantics** *(decision)*: use the reliable-queue pattern — worker does `BLMOVE frp:queue:pending frp:queue:processing RIGHT LEFT timeout=5`. On success/terminal-failure, `LREM` from processing (ack). On worker start, run a **reaper**: any job in `processing` whose `heartbeat_at` is older than 10 min is re-queued (increment `attempts`) or moved to `dead` if `attempts >= 3`.

**Priority ("job sorting")**: two lists, `frp:queue:pending:high` and `frp:queue:pending:normal`; worker polls high first (`BLMOVE` high with 0.1s timeout, then normal). Enqueue accepts `priority` param. Do not build a sorted-set scheduler unless asked.

---

## 4. Control API (stateless)

FastAPI, no local state, no DB connection — Redis only. All endpoints return JSON; errors as RFC 7807 problem details.

| Method | Path | Behavior |
|---|---|---|
| POST | `/jobs` | Body: `{supplier_name, product?, priority?, force?}`. `product` is optional — if omitted/blank, the worker runs DISCOVER_PRODUCT first (§5.1a) before RESEARCH. Normalizes key (§5.1). If dedupe hit and `force` is false → **200** with `{status:"duplicate", existing_job_id}` and do **not** enqueue. Else create job hash, push id to queue, set dedupe key. → **201** `{job_id}` |
| GET | `/jobs/{id}` | Full job hash + derived fields (elapsed, cost so far). 404 if unknown. |
| GET | `/jobs?status=&limit=&cursor=` | List jobs by scanning a small secondary index: maintain `frp:index:jobs` (ZSET, score = enqueued_at). Paginate by score. |
| DELETE | `/jobs/{id}` | Only allowed if status = `queued` (LREM from pending, mark `deleted`) — a running job must be halted instead. 409 otherwise. |
| POST | `/jobs/{id}/halt` | Set `frp:halt:{id}`. Returns 202. Worker honors between steps (≤ one step of latency). Idempotent. |
| POST | `/jobs/{id}/retry` | Allowed on `failed | dead | halted`: reset attempts if desired flag set, re-enqueue. |
| GET | `/stats` | Queue depths (pending high/normal, processing, dead), jobs by status in last 24h, today's `frp:budget` value, tokens and search queries consumed today. |
| GET | `/healthz` | Redis PING. |

**Observability** *(requirements)*:
- Structured JSON logs (one line per request: method, path, status, latency ms, job_id if any).
- `/stats` is the human dashboard endpoint; keep it cheap (all values are O(1) Redis reads plus small ZSET counts).
- Worker writes `heartbeat_at` on the job hash every 15 s and increments daily counters (`frp:metrics:{date}:tokens_in`, `:tokens_out`, `:searches`, `:geocodes`) — `/stats` reads these.

---

## 5. Worker Pipeline (the graph)

Implement `graph.py` as an explicit, resumable state machine. Each step is a pure-ish async function `(JobState) -> JobState`; the runner persists `step` to the job hash before executing it and checks the halt flag **before every step**.

```
(DISCOVER_PRODUCT ──►) RESEARCH ──► EXTRACT ──► GEOCODE ──► PERSIST ──► done
        │                  │            │           │           │
        └──────────────────┴────────────┴───────────┴───────────┴──► on exception: retry step ≤2 (backoff 2s/8s), then fail job
```

DISCOVER_PRODUCT only runs when the job was submitted without a `product`; it is skipped entirely (and absent from the job's `step` history) for normal explicit-product jobs.

### 5.1 Pre-step: dedup & normalization

- Two normalized keys, split because a supplier supplies many products (§2.3):
  - `normalized_name = slugify(lower(supplier_name))` — **supplier identity** in Supabase (one row per company).
  - `normalized_key = normalized_name + ":" + slugify(lower(product))` — **per-run dedupe/cache key** for Redis enqueue and the Supabase cache lookup. When `product` is blank at enqueue time, `"__auto__"` stands in for it so repeat no-product submissions for the same supplier still dedupe onto one discovery job.
- The API already checks dedupe on enqueue; the worker re-checks Supabase before spending money by looking up the supplier on `normalized_name` and its facilities **for this specific product**: if facilities exist and the latest successful `research_runs` for that (supplier, product) is newer than `REFRESH_DAYS` (default 90), mark job `succeeded` with `facilities_found` from DB and note `"served_from_cache": true`. **This is the single biggest cost-saving control — never pay Gemini twice for the same (supplier, product) inside the refresh window.** Serving is per-product, so a fresh steel result never suppresses a first-time bearings request for the same company. Skipped when `product` is still unknown (nothing real to key the cache on yet) — see §5.1a.

### 5.1a DISCOVER_PRODUCT — one small grounded call, only when `product` is omitted

**Why:** requiring a product at submission time is fine when the caller already knows it, but forces a manual lookup step otherwise. Rather than adding a second, separate research pass, this reuses the same one-call-plus-conditional-retry shape as RESEARCH, just with a much smaller output cap — it asks Gemini for a short ranked list of the supplier's most popular/most-sourced products (`PRODUCT_LIST_SCHEMA`: a bare JSON array of strings, no object wrapper) and takes the top entry as `state.product`.

- `max_output_tokens=1024` (this is a name list, not a facility search) — negligible added cost per job (well under the RESEARCH call's own budget).
- Conditional retry (max 1): if the list comes back empty, retry once asking the model to infer from industry sector/filings rather than give up — mirrors RESEARCH's retry pattern.
- After resolving `state.product`, recompute `normalized_key` with the real product, persist both to the job hash (`discovered_products` holds the full ranked list for visibility/audit, even though only the top entry is used), claim the resolved dedupe key, and **re-run the Supabase cache check** — if this exact (supplier, discovered product) was already researched under some prior explicit-product job, skip RESEARCH entirely and serve from cache.
- If discovery genuinely finds nothing (rare — a real company with no inferable product), fall back to `"general operations"` as `state.product` rather than failing the job.
- Same budget circuit breaker as RESEARCH: checked before the call, so a blank-product job can't slip past the daily cap.

### 5.2 RESEARCH — one grounded Gemini call *(cost-critical)*

**Model** *(decision)*: `gemini-3.5-flash` with Google Search grounding. Do not use Pro; do not build a multi-turn agent loop. **One grounded call per supplier**, plus at most **one** conditional retry (below).

Request configuration:

```python
config = GenerateContentConfig(
    tools=[Tool(google_search=GoogleSearch())],
    response_mime_type="application/json",
    response_schema=FACILITY_LIST_SCHEMA,      # array of Facility, max 40 items
    max_output_tokens=12288,  # raised from 8192 — the original cap was already tight for ~40 facilities at the documented ~150-220 tokens/facility, and truncation silently caps recall before EXTRACT even runs
    thinking_config=ThinkingConfig(thinking_budget=1024),  # cap thinking; this task is retrieval-heavy, not reasoning-heavy
    temperature=0.1,
)
```

System prompt (keep under ~400 tokens; store as a constant, do not template large text in):

> You are a supply-chain research assistant. Given a supplier company and a product, find every real, currently operating facility involved in producing that product for the supplier — including facilities the supplier owns directly AND third-party sites operating on its behalf (contract manufacturers, toll manufacturers, joint-venture plants, third-party logistics providers, co-packers). Do not include the customer's own facilities, resellers, or pure sales/admin offices with no production, storage, or processing role. For each facility return: facility_name, full postal facility_address, latitude/longitude if a source states them (else null — never guess coordinates), facility_type as exactly one of manufacturing | logistics | raw_material, confidence (high/medium/low), and source_url. Manufacturing = production/assembly/fabrication plants, contract or toll manufacturing sites. Logistics = warehouses, distribution centers, fulfillment hubs, ports, freight terminals. Raw_material = mines, quarries, wells, smelters, refineries, mills, tanneries, farms, plantations, or other primary-material extraction/processing sites. Search broadly and check multiple source types: the company's own factory/locations pages, annual reports, ESG/CSRD disclosures, investor filings, published supplier or factory lists (e.g. OpenSupplyHub, Sourcemap), industry directories, and trade/customs records. Find as many distinct real facilities as the evidence supports — do not stop after just one or two hits if more exist. If you cannot verify an address, set confidence low. Return JSON only.
>
> (See `worker/gemini.py:PRODUCT_DISCOVERY_SYSTEM_PROMPT` for the separate, much smaller DISCOVER_PRODUCT prompt — §5.1a.)

User message: `Supplier: {supplier_name}\nProduct: {product}`.

**Why these choices save money:**
- `response_schema` + JSON-only removes prose padding; expected output ≈ 150–220 tokens/facility.
- `thinking_budget=1024` caps the dominant hidden output cost.
- "never guess coordinates" pushes lat/lng work to the Geocoding API (cheap, accurate) instead of paying output tokens for hallucinated numbers.
- No agent loop means search-query fan-out stays at whatever a single grounded call does (typically 3–6 queries) instead of 15+.

**Conditional retry (max 1)** *(revised — see note below)*: if the call returns fewer than `MIN_FACILITIES_BEFORE_RETRY` (default 3) facilities, retry once with a user message that lists the facilities already found (so the retry doesn't re-surface them) and steers the grounded search toward a different angle — contract/toll manufacturer relationships, third-party logistics, factory-audit databases (OpenSupplyHub, Sourcemap, Better Work, FLA), import/customs records, local trade press. The retry's results are **appended** to the first call's, not used to replace it. Otherwise accept the result — a genuinely low result for an obscure or small supplier is a valid outcome, not an error.

> **Deviation from the original decision:** the original trigger was "0 facilities AND <2 search queries," which let most low-yield-but-nonzero jobs through uncorrected — a major source of the "can't find many facilities" complaint this pipeline was built to solve. Broadening the trigger to a facility-count threshold, and changing the retry's search angle from two bolted-on query strings to a genuinely different source class, increases recall without adding architecture (still exactly one call + at most one retry, no agent loop, no extra Gemini call type). Worst-case cost impact is bounded to 2x a single call, and only for jobs that would otherwise return a handful of facilities or fewer — the same order of magnitude the original design already budgeted for.

**Capture usage:** read `usage_metadata` (prompt/candidates/thoughts token counts) and `grounding_metadata` (number of web search queries) from the response; write to the job hash and increment daily metrics and `frp:budget:{date}` using the pricing constants in config (`PRICE_IN`, `PRICE_OUT`, `PRICE_PER_SEARCH` — make these config values, not hardcoded literals, so price changes don't need a deploy).

### 5.3 EXTRACT — validate & clean (no LLM)

Pure Python. Parse the JSON against the pydantic schema. Then:
- Drop items with empty name or address.
- Deduplicate within the batch on `(lower(name), lower(address))` and near-duplicates on address similarity ≥ 0.9 (use `rapidfuzz`, it's cheap).
- Coerce/validate `facility_type` against the enum; if the model produced anything else, map obvious synonyms (`factory→manufacturing`, `warehouse/DC→logistics`, `mine/smelter/refinery→raw_material`), else drop the record and log it.
- Clamp lat/lng to valid ranges; if out of range, null them (geocoding will fix).
- Stamp each surviving facility's `product` with the job's resolved product (§2.1) — deterministic, no LLM.

No tokens spent here. Do **not** add a second LLM "cleanup" call.

### 5.4 GEOCODE — Google Geocoding API with cache

For each facility lacking coordinates (or whose model-provided coordinates are > 50 km from the geocoded address — trust the geocoder):
1. `sha1(normalized_address)` → check `frp:geocode:{hash}`. Hit → use cached.
2. Miss → call Geocoding API. Store result (lat, lng, `location_type`) in cache with 180-day TTL.
3. Map `location_type`: `ROOFTOP/RANGE_INTERPOLATED → geocode_status=ok`, `GEOMETRIC_CENTER/APPROXIMATE → approximate`, no result → `failed` (keep the facility, coordinates null).

Concurrency: ≤ 5 parallel geocode calls; respect 429s with backoff. The cache matters at scale — suppliers share industrial parks and repeated re-onboards shouldn't re-pay.

### 5.5 PERSIST

Upsert supplier by `normalized_name` (one row per company — §2.3); upsert facilities on the `(supplier_id, product, facility_name, facility_address)` unique constraint (update coordinates/confidence on conflict). Insert the `research_runs` row (including `product`) with full token/cost accounting. Mark job `succeeded`, set `facilities_found`.

### 5.6 Worker main loop

```python
while not shutdown:
    job_id = await queue.dequeue(timeout=5)        # BLMOVE high→normal
    if job_id is None: continue
    try:
        await run_graph(job_id)                     # steps + halt checks + heartbeats
        await queue.ack(job_id)
    except HaltedError:
        mark halted; ack
    except Exception as e:
        attempts += 1
        if attempts >= MAX_ATTEMPTS(3): move to dead list; mark dead
        else: re-enqueue with backoff; mark queued
```

- Handle SIGTERM: finish current step, re-enqueue job, exit (Cloud Run sends SIGTERM on scale-down/deploy).
- `WORKER_CONCURRENCY` env (default 2): run N graph coroutines against the queue. Keep default low — Gemini rate limits, not CPU, are the constraint.

---

## 6. Cost Guardrails *(build all of these)*

1. **Daily budget circuit breaker.** Before RESEARCH, read `frp:budget:{today}`; if ≥ `DAILY_BUDGET_USD` (env, default 25), do not call Gemini — re-enqueue the job with status note `budget_paused` and sleep the worker 15 min. `/stats` exposes this state.
2. **Dedup at three layers** (API enqueue, worker Supabase check, geocode cache) — already specified above.
3. **Hard caps per job:** `max_output_tokens=12288`, thinking budget 1024, max 1 conditional retry, max 40 facilities parsed.
4. **Cost estimate written per job** so anomalies are visible immediately in `/stats` (e.g., a job that triggered 12 search queries).
5. **Config-driven pricing constants** so estimates stay honest.

Expected steady-state cost ≈ **$0.10–0.20 per new supplier** (Gemini tokens + ~5 grounding queries + ~20 geocodes with cache misses), ~$0 for duplicates.

---

## 7. Configuration (env vars)

```
REDIS_URL, SUPABASE_URL, SUPABASE_SERVICE_KEY,
GEMINI_API_KEY, GEMINI_MODEL=gemini-3.5-flash,
GEOCODING_API_KEY,
DAILY_BUDGET_USD=25, REFRESH_DAYS=90, MAX_ATTEMPTS=3,
WORKER_CONCURRENCY=2, THINKING_BUDGET=1024, MAX_OUTPUT_TOKENS=12288,
MIN_FACILITIES_BEFORE_RETRY=3,
PRICE_IN_PER_M=1.50, PRICE_OUT_PER_M=9.00, PRICE_PER_1K_SEARCH=14.00, PRICE_PER_1K_GEOCODE=5.00
```

Secrets via Cloud Run secret manager references, never in the image.

---

## 8. Deployment (GCP Cloud Run)

- **API**: Cloud Run service, request-based CPU, min instances 0, concurrency 80. Tiny footprint (512 MiB / 1 vCPU).
- **Worker**: Cloud Run service with **CPU always allocated**, min instances 1, max 1 initially (scale later by raising max + relying on the reliable-queue semantics; the BLMOVE pattern is multi-worker safe). 1 vCPU / 512 MiB.
- Both connect to Redis over VPC connector (or Upstash-style TLS URL — match the existing instance).
- Health: API `/healthz`; worker exposes a minimal `/healthz` HTTP listener (Cloud Run requires a port) that returns 200 if the loop heartbeat is < 60 s old.

---

## 9. Testing & Acceptance Criteria

**Unit tests** (pytest, mock all external calls):
- Queue client: enqueue/dequeue/ack, priority ordering, reaper re-queues stale processing jobs, dead-letter after 3 attempts.
- Dedup: second POST /jobs with same (supplier, product) returns `duplicate` and does not grow the queue; `force=true` bypasses.
- Supplier identity: same company + two products → one supplier row, facilities partitioned by `product`, one `research_runs` per product.
- Extract: type-synonym mapping, near-duplicate collapse, invalid lat/lng nulled.
- Halt: flag set mid-run → job ends `halted` before the next step, is ack'd, no Supabase write occurs if halted before PERSIST.
- Budget breaker: with budget exceeded, no Gemini call is made (assert mock not called).

**Integration test** (against real Redis, mocked Gemini/Geocode with canned fixtures): full job lifecycle queued→succeeded, verify Supabase rows, verify `research_runs` cost math matches fixture usage numbers.

**Acceptance:**
1. POST /jobs → job completes → facilities in Supabase with all six attributes; every facility has either valid coordinates or `geocode_status=failed`.
2. Re-POSTing the same supplier within REFRESH_DAYS costs $0 (no Gemini/geocode calls; verify via metrics counters).
3. Halting a running job stops it within one pipeline step.
4. Killing the worker mid-job (SIGKILL) → reaper re-queues it; job eventually succeeds; no duplicate facility rows (upsert constraint).
5. `/stats` shows accurate queue depths, today's spend, and token counters.
6. POST /jobs with no `product` → DISCOVER_PRODUCT resolves one, RESEARCH runs against it, and the persisted facilities/`research_runs` carry that product; `discovered_products` on the job hash holds the full ranked list. If that resolved (supplier, product) pair is already cached, RESEARCH is skipped (0 extra Gemini calls beyond DISCOVER_PRODUCT itself).
7. Researching the same company for two different products yields **one** supplier row (keyed on `normalized_name`) with facilities partitioned by `product`, and two `research_runs`. Caching is per-product: a fresh result for product A does not suppress a first-time request for product B on the same company.

---

## 10. Explicit Non-Goals (do not build)

- No multi-agent / iterative research loops. One grounded call (+1 conditional retry) per supplier for RESEARCH, and — only when `product` is omitted — one small additional grounded call (+1 conditional retry) for DISCOVER_PRODUCT. This is a fixed extra step, not a loop: it runs at most once per job, never branches into multiple product-specific research passes.
- No Celery, RQ, Kafka, or Pub/Sub — Redis lists only.
- No frontend. `/stats` JSON is the dashboard.
- No automatic scheduled refresh of suppliers (manual re-enqueue with `force=true` covers it for now).
- No embedding/vector search, no scraping beyond what Gemini grounding returns.