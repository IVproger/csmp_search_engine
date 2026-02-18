# Copilot Instructions for csmp_search_engine

## Project Overview

- This project implements a hybrid AI-powered search engine for spectrum → molecule annotation.
- The system is fully containerized and built as a set of modular microservices.

---

## Core Services

### Streamlit Web App (UI)

- Accepts spectrum files in mzML / MGF formats
- Sends annotation requests to backend API 
- Displays ranked molecule candidates

Runs in Docker on port `8501`.

---

### FastAPI Backend (Orchestrator)

- Central control layer for all requests
- Handles spectrum parsing, inference calls, and DB search
- Exposes main endpoint:

POST /annotate-spectrum which accepts spectrum data in mzML / MGF formats and returns ranked molecule candidates with metadata for each spectrum instance in file.

Runs in Docker on port `8000`.

---

### Spectrum Parser

- Converts MS/MS spectra files mzML / MGF into structured numeric tensors
- Extracts:
  - peak m/z values List[float]
  - intensities List[float]
  - precursor m/z float
  - adduct string (if available)
  - formula string (if available)

implemented inside FastAPI service for tight integration with inference and search pipelines.

---

### Embedding Service (ML Inference)

- Runs Triton Inference Server
- Serves ONNX spectrum encoder model
- Receives batches of parsed spectrum data via gRPC
- Outputs dense vector embeddings, which send back to FastAPI service inside db search client for downstream search

Isolated in its own container and cab accessed inside the docker network on port `8001` via gRPC.

---

### Molecular Search Database

- PostgreSQL with pgvector extension
- Stores:
  - molecule metadata
  - molecular embeddings
- Supports:
  - mass window filtering
  - nearest-neighbor vector search

Runs on port `5432` with persistent volume.

---

## Data Flow

1. User uploads spectrum file in formar mzML / MGF in Streamlit UI. File contains one or more spectrum instances, each with precursor_m/z, adduct, formula, and peak lists (mz and intencity values).
2. Streamlit calls FastAPI POST `/annotate-spectrum` and send a exact file.
3. File parsed into numeric form. For each spectrum instance, extract data according to schema:
  - precursor_m/z float
  - adduct string (if available)
  - formula string (if available)
  - peaks List[Tuple[float, float]] (m/z, intensity pairs)
4. Parsed data sent as batch to Triton for embeddings generation. Also the batch of precursor m/z values sent to db client for mass filtering.
5. Embedding returned into to FastAPI service into db client, which performs search in PostgreSQL database.  
6. Precursor m/z filters candidate molecules and pgvector ranks by similarity  
8. Ranked results returned to UI in approximate json format 

```
{ spectrum №1
  "candidates": [
    {"smiles": "O=C1CCCN1CC#CCN1CCCC1", "mass": 162.1157, "score": 0.91},
    {"smiles": "CC(C[N+](C)(C)C)OC(=O)N", "mass": 166.0629, "score": 0.74}
  ]
spectrum №2
  "candidates": [
    {"smiles": "O=C1CCCN1CC#CCN1CCCC1", "mass": 162.1157, "score": 0.91},
    {"smiles": "CC(C[N+](C)(C)C)OC(=O)N", "mass": 166.0629, "score": 0.74}
  ]
}
```

---

## Tech Stack

- Python 3.11+
- FastAPI
- Streamlit
- Triton Inference Server
- ONNX Runtime, PyTorch for model development
- PostgreSQL + pgvector
- Docker & Docker Compose
- gRPC for model inference
- pymzml, matchms for spectrum parsing

---

## Conventions & Patterns

- Keep services stateless where possible
- Use FastAPI as the only orchestration layer
- Always apply mass filtering before vector search
- Treat embedding models as swappable components
- Keep data schemas explicit and versioned

---

## Development Workflow

- Run services via Docker Compose
- Update ML models by replacing ONNX artifacts
- Extend DB schema via migrations
- Add new pipelines as FastAPI modules

---

## Extending the System

- Add new encoders as separate Triton models
- Implement ensemble ranking modules
- Add spectrum clustering or caching layers
- Introduce feedback-driven re-ranking

---

## Integration Points

- gRPC → Triton inference
- REST → FastAPI API
- SQL + vector search → Postgres

---