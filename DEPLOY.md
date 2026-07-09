# Deploying GridWatch

GridWatch is a five-service stack (TimescaleDB, Qdrant, API, ingestion, dashboard).
Locally it runs with one command:

```bash
docker compose up --build
```

For a hosted deployment you have two realistic paths. **Railway** (per the project
plan) is documented first; a **single-VM / compose-native** option is simpler if you
want the whole stack up with minimal wiring.

---

## Option A — Railway

Railway deploys each service separately (it does not run `docker-compose.yml`
natively), but all app services share this repo and its `Dockerfile` — you just give
each a different **start command**. Two stateful services run as public Docker images.

### 1. Data services (Docker-image services)

| Service | Image | Notes |
|---|---|---|
| `db` | `timescale/timescaledb:latest-pg15` | Add a **volume** mounted at `/var/lib/postgresql/data`. Set `POSTGRES_USER/PASSWORD/DB = gridwatch`. |
| `qdrant` | `qdrant/qdrant:latest` | Add a volume at `/qdrant/storage`. |

Railway gives each service a private hostname like `db.railway.internal` and
`qdrant.railway.internal`.

### 2. App services (from this repo's Dockerfile)

Create three services from the GitHub repo. Each uses the same image; override the
**start command**:

| Service | Start command |
|---|---|
| `api` | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| `ingestion` | `python -m ingestion.scheduler` |
| `dashboard` | `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0` |

Expose `api` and `dashboard` publicly; `ingestion` is a background worker (no domain).

### 3. Environment variables (set on each app service)

```
DATABASE_URL=postgresql://gridwatch:gridwatch@db.railway.internal:5432/gridwatch
QDRANT_HOST=qdrant.railway.internal
QDRANT_PORT=6333
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
EMBED_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...            # set as a secret
API_BASE_URL=https://<your-api-service>.up.railway.app   # dashboard only
LOG_LEVEL=INFO
ENVIRONMENT=production
```

> The `dashboard` talks to the API over the public URL, so set `API_BASE_URL` to the
> deployed `api` domain (not `http://api:8000`, which only exists in Compose).

### 4. First-run notes

- `ingestion` performs the 90-day backfill, trains the ML models, and seeds Qdrant
  on first boot — this runs for a couple of minutes and makes outbound calls to the
  Carbon Intensity API, Open-Meteo, and OpenAI (embeddings). Watch its logs.
- ML model artifacts live in `ml/artifacts/` inside the ingestion container. On
  Railway that's ephemeral, so models are retrained on each redeploy (nightly retrain
  still applies). For persistence, mount a volume at `/app/ml/artifacts`.
- Bring services up in order: `db` + `qdrant` → `ingestion` → `api` → `dashboard`.

---

## Option B — Single VM / compose-native (simplest)

Any host with Docker (a cheap VPS, or a service that runs a Compose file directly
such as **Render Blueprints**) can run the stack as-is:

```bash
git clone <your-repo> && cd gridwatch
cp .env.example .env      # set OPENAI_API_KEY; DATABASE_URL/QDRANT_HOST already
                          # point at the compose service names
docker compose up -d --build
```

Put a reverse proxy (Caddy/Nginx) in front of ports `8501` (dashboard) and `8000`
(API) for TLS. This keeps the exact local topology, so there's no service-name or
private-networking rewiring — the fastest way to a working hosted demo.

---

## Security checklist before going public

- [ ] `.env` is **git-ignored** (it holds your OpenAI key) — confirm it's not committed.
- [ ] Change the default `gridwatch/gridwatch` Postgres credentials.
- [ ] Rotate the OpenAI key if it was ever committed or shared.
- [ ] The NL layer already executes generated SQL **read-only** (single `SELECT`, in a
      `READ ONLY` transaction). Keep the DB user least-privileged as a second layer.
- [ ] Consider basic auth on the dashboard if the demo is public.
