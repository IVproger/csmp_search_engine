# csmp_search_engine

Microservice workspace for spectrum-to-molecule annotation.

## Structure

- `services/fastapi_service/` — FastAPI orchestrator service
  - `app/main.py` — API entry with `POST /annotate-spectrum`
  - `app/models.py` — Pydantic data models
  - `Dockerfile` — service container image
  - `requirements.txt` — service Python dependencies

## Run with Docker Compose

```bash
docker compose up --build
```

This starts `fastapi_service` now and keeps placeholders in `docker-compose.yml` for Streamlit, Triton, and Postgres.
