# Triton Inference Service

This service runs NVIDIA Triton Inference Server and serves ONNX models from the shared model repository mounted at `/models`.

## Ports

- `8001` gRPC (used by FastAPI)
- `8002` HTTP/REST
- `8003` metrics

## Startup

Run with Docker Compose from workspace root:

```bash
docker compose up --build triton_service fastapi_service
```

Triton readiness endpoint:

- `http://localhost:8002/v2/health/ready`

FastAPI depends on Triton health in `docker-compose.yml`.
