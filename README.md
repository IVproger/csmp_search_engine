# csmp_search_engine

Microservice workspace for spectrum-to-molecule annotation.

## Structure

- `services/fastapi_service/` — FastAPI orchestrator service
  - `app/main.py` — API entry with `POST /annotate-spectrum`
  - `app/models.py` — Pydantic data models
  - `Dockerfile` — service container image
  - `requirements.txt` — service Python dependencies
- `services/postgres_service/` — PostgreSQL + pgvector database service
  - `initdb/` — one-time initialization scripts (schema, seed load, indexes)
  - `scripts/reload_from_csv.sql` — manual CSV reload/upsert script

## Run with Docker Compose

```bash
docker compose up --build
```

This starts:

- `postgres_service` (`5432`) with persistent DB storage in `data/postgres_molecular_search_db/postgres_data`
- `fastapi_service` (`8000`)
- `triton_service` (`8001/8002/8003`)

## PostgreSQL + pgvector setup

The DB container is configured for persistent host-mounted storage:

- Host path: `data/postgres_molecular_search_db/postgres_data`
- Container path: `/var/lib/postgresql/data`

Initialization scripts in `services/postgres_service/initdb` run automatically **only on first DB initialization** (when `postgres_data` is empty):

1. Create extension `vector`
2. Create table `molecular_search` with fields:
  - `formula`
  - `smiles`
  - `inchikey` (PRIMARY KEY / unique criterion)
  - `monoisotopic_mass`
  - `mol_embedding` (`vector(256)`)
3. Load data from:
  - `data/postgres_molecular_search_db/molecules_with_embeddings.csv`
4. Create indexes:
  - B-tree on `monoisotopic_mass`
  - HNSW on `mol_embedding` with `vector_cosine_ops`

Because DB files are persisted in the host mount, restarting the container is fast and data remains available.

## How to reload/update data manually (inside container)

When you need to refresh the table from CSV:

```bash
docker compose exec postgres_service psql -U csmp_user -d molecular_search_db -f /opt/db-scripts/reload_from_csv.sql
```

This script:

- Loads CSV into staging table
- Upserts rows into `molecular_search` by `inchikey`
- Rebuilds both indexes

## How to force full re-initialization

If you want to recreate DB from scratch and re-run all init scripts:

```bash
docker compose down
rm -rf data/postgres_molecular_search_db/postgres_data/*
docker compose up --build
```

## Helper scripts (recommended)

Use project scripts instead of long inline shell snippets:

```bash
./scripts/reset_postgres_cluster.sh
./scripts/check_postgres_access.sh
./scripts/reload_molecular_csv.sh
```

If your interactive `zsh` prompt breaks on `set -u` (e.g. `RPROMPT: parameter not set`), do not run `set -euo pipefail` manually in your interactive shell. These scripts already handle strict mode safely inside `bash`.
