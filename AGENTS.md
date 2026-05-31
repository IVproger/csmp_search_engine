# AGENTS.md — brief for AI coding agents

Orientation for any coding agent (Claude Code, Copilot, Cursor, …) working in
this repository. Read this first; see [`README.md`](README.md) for the full,
human-facing documentation.

## What this project is

A **spectrum → molecule annotation engine** for MS/MS mass spectrometry. A user
uploads a spectrum file; the system parses it, encodes each spectrum into a
256-dim vector with a neural encoder, and retrieves candidate molecules from a
`pgvector` database by combining **exact mass-window filtering** with **cosine
embedding similarity**. Delivered as four Docker Compose microservices.

## Service map

| Service             | Path                          | Tech            | Role |
|---------------------|-------------------------------|-----------------|------|
| `streamlit_service` | `services/streamlit_service/` | Streamlit/Plotly/RDKit | Web UI: upload, preview, render results |
| `fastapi_service`   | `services/fastapi_service/`   | FastAPI         | **Orchestrator** — the only place business logic lives |
| `triton_service`    | `services/triton_service/`    | NVIDIA Triton   | Serves the ONNX spectrum encoder over gRPC |
| `postgres_service`  | `services/postgres_service/`  | PostgreSQL + pgvector | Molecular database + init/reload SQL |

## Where things live (`fastapi_service/app/`)

- `main.py` — endpoints: `GET /health`, `POST /annotate-spectrum`. Orchestrates
  parse → encode → search and builds the response (incl. per-spectrum messages).
- `spectrum_parser.py` — file → `ParsedSpectrum` list (mzML via `pymzml`;
  MGF/MSP/JSON via `matchms`); intensity max-normalization.
- `spectrum_encoder_client.py` — Triton gRPC client; batches/chunks spectra,
  L2-normalizes embeddings.
- `db_search_client.py` — mass-window pre-filter + `pgvector` cosine search,
  escalating ppm windows, optional vector-only fallback, scoring.
- `utils.py` — adduct normalization + neutral-mass reconstruction.
- `models.py` — Pydantic schemas (`ParsedSpectrum`, `MoleculeCandidate`,
  `AnnotateSpectrumResponse`, …). `file_formats.py` / `inference_config.py` —
  small config modules.

## Data flow (one request)

1. UI `POST`s the file to `/annotate-spectrum`.
2. Validate extension → parse all spectra → normalize intensities.
3. Spectra **with** a precursor m/z are batch-encoded by Triton; embeddings are
   L2-normalized.
4. For each spectrum: reconstruct neutral mass (precursor + adduct + proton),
   mass-window filter, then cosine NN search; take top-K.
5. Return one result block per spectrum. Spectra **without** a precursor m/z are
   returned with an explanatory message and no candidates.

## Conventions

- **FastAPI is the only orchestration layer.** Keep services stateless. The UI
  and the test runner re-implement lightweight parsing for previews only — real
  logic belongs in `fastapi_service`.
- **Always mass-filter before vector search.** Preserve the
  filter-then-rank order.
- **Fail gracefully per spectrum.** Inference/DB errors must not 500 the whole
  request — attach a message and continue (see `main.py`).
- **Treat the encoder as swappable.** Don't hard-code embedding specifics beyond
  the `VECTOR(256)` contract and the `mzs`/`intens`/`num_peaks` inputs.
- Config is read from environment variables with sane defaults (see the tables
  in `README.md`); don't hard-code hosts, ports, or tolerances.
- Python 3.11+, type hints, `snake_case`, module-level singletons via
  `functools.lru_cache` (e.g. `get_db_search_client`).

## Build / run / test

```bash
docker compose up --build                        # full stack
docker compose --profile test up --build api_tests   # integration tests
python services/fastapi_service/tests/run_api_tests.py   # tests against a live API
```

## Important: git-ignored, locally-provided assets

`models/`, `data/`, `notebooks/`, `scripts/` are **not** in version control
(multi-GB weights, DB cluster, seed CSV, research notebooks). Do not assume they
exist in a fresh clone, and never try to commit them. Required paths:

- `models/spectrum_encoder/<version>/csu-ms-2-spec-encoder.onnx(.data)` + `config.pbtxt`
- `data/postgres_molecular_search_db/molecules_with_embeddings.csv`

## Database schema (`molecular_search`)

`formula TEXT`, `smiles TEXT`, `inchikey TEXT PRIMARY KEY`,
`monoisotopic_mass DOUBLE PRECISION`, `mol_embedding VECTOR(256)`.
Indexes: B-tree on `monoisotopic_mass` and `formula`; HNSW
(`vector_cosine_ops`) on `mol_embedding`. Init scripts run **only** on first DB
startup; use `reload_from_csv.sql` to refresh data afterwards.
