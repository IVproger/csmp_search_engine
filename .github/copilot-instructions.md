# Copilot instructions — csmp_search_engine

Concise guidance for AI coding assistants. For the full brief see
[`AGENTS.md`](../AGENTS.md); for human docs see [`README.md`](../README.md).

## Project in one line

A containerized **spectrum → molecule annotation engine**: parse an MS/MS
spectrum, encode it into a 256-dim vector with a neural encoder, and retrieve
candidate molecules via **mass-window filtering + `pgvector` cosine similarity**.

## Architecture (4 services, Docker Compose)

- **`streamlit_service`** (`:8501`) — Web UI: upload, Plotly spectrum preview,
  RDKit structure rendering of candidates.
- **`fastapi_service`** (`:8000`) — Orchestrator and the only business-logic
  layer. Endpoints: `GET /health`, `POST /annotate-spectrum`.
- **`triton_service`** (gRPC `:8001`) — NVIDIA Triton serving the ONNX spectrum
  encoder.
- **`postgres_service`** (`:5432`) — PostgreSQL + `pgvector` molecular database.

## Supported inputs

Spectrum files: `.mzML`, `.MGF`, `.MSP`, `.JSON` (mzML via `pymzml`; the rest via
`matchms`). A file may contain many spectra.

## Request pipeline (`fastapi_service/app/main.py`)

1. Validate extension and parse all spectra (`spectrum_parser.py`); normalize
   peak intensities (max-norm to 100).
2. Batch-encode spectra that have a precursor m/z via Triton
   (`spectrum_encoder_client.py`); L2-normalize embeddings.
3. Per spectrum: reconstruct neutral mass from precursor m/z + adduct + proton
   (`utils.py`), mass-window filter, then cosine NN search (`db_search_client.py`),
   return top-K candidates.
4. Spectra without a precursor m/z are returned with a message and no candidates.

## Response schema (`models.py`)

`AnnotateSpectrumResponse { status, file_name, file_type, message, results[] }`,
where each result is
`{ spectrum_id, precursor_mz, candidates[] | null, message }` and each candidate
is `{ smiles, mass, similarity_score }` with `similarity_score` in `0–100`
(`(1 − cosine_distance) × 100`).

## Conventions to follow

- Put real logic in `fastapi_service`; keep services stateless.
- Always **mass-filter before vector search**; keep the escalating-ppm strategy.
- Degrade gracefully per spectrum — never let one failure 500 the request.
- Read config from env vars with defaults (see README tables); don't hard-code
  hosts/ports/tolerances. Treat the encoder model as swappable.
- Python 3.11+, full type hints, `snake_case`, `lru_cache` singletons for clients.

## Don't

- Don't assume `models/`, `data/`, `notebooks/`, `scripts/` exist or commit them
  — they are git-ignored, locally-provided assets (multi-GB weights, DB cluster,
  seed CSV).
- Don't change the `VECTOR(256)` embedding contract or the
  `mzs`/`intens`/`num_peaks` Triton inputs without updating both sides.
