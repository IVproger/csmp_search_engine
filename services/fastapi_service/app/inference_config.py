import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SpectrumInferenceConfig:
    triton_url: str = os.getenv("TRITON_GRPC_URL", "triton_service:8001")
    model_name: str = os.getenv("TRITON_MODEL_NAME", "spectrum_encoder")
    model_version: str = os.getenv("TRITON_MODEL_VERSION", "")
    max_peaks: int = int(os.getenv("SPECTRUM_MAX_PEAKS", "1024"))
    infer_timeout_seconds: float = float(os.getenv("TRITON_INFER_TIMEOUT_SECONDS", "30"))
    infer_chunk_size: int = int(os.getenv("SPECTRUM_INFER_CHUNK_SIZE", "32"))
