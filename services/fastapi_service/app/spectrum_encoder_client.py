from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import tritonclient.grpc as grpcclient

from app.inference_config import SpectrumInferenceConfig
from app.models import ParsedSpectrum


logger = logging.getLogger(__name__)


class SpectrumInferenceError(RuntimeError):
    pass


class SpectrumEncoderClient:
    def __init__(self, config: SpectrumInferenceConfig):
        self._config = config
        self._client = grpcclient.InferenceServerClient(url=config.triton_url)
        self._output_name = self._resolve_output_name()

    def encode(self, spectra: list[ParsedSpectrum]) -> np.ndarray:
        if not spectra:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings_chunks: list[np.ndarray] = []
        for start_index in range(0, len(spectra), self._config.infer_chunk_size):
            chunk = spectra[start_index : start_index + self._config.infer_chunk_size]
            mzs, intensities, num_peaks = self._build_inputs(chunk)
            embeddings_chunks.append(self._infer_chunk(mzs, intensities, num_peaks))

        # L2-normalize embeddings
        embeddings = np.concatenate(embeddings_chunks, axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) # Avoid division by zero
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        
        return embeddings

    def _resolve_output_name(self) -> str:
        try:
            metadata = self._client.get_model_metadata(
                model_name=self._config.model_name,
                model_version=self._config.model_version,
            )
            outputs = getattr(metadata, "outputs", None)
            if outputs:
                return outputs[0].name
        except Exception as error:
            logger.warning("Failed to fetch Triton model metadata: %s", error)

        return "output"

    def _build_inputs(self, spectra: list[ParsedSpectrum]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = len(spectra)
        max_peaks = self._config.max_peaks
        mzs = np.zeros((batch_size, max_peaks), dtype=np.float32)
        intensities = np.zeros((batch_size, max_peaks), dtype=np.float32)
        num_peaks = np.zeros((batch_size, 1), dtype=np.int64)

        for row_index, spectrum in enumerate(spectra):
            spectrum_peaks = spectrum.peaks[:max_peaks]
            peaks_count = len(spectrum_peaks)
            num_peaks[row_index, 0] = peaks_count

            if peaks_count == 0:
                continue

            mzs[row_index, :peaks_count] = np.array([peak.mz for peak in spectrum_peaks], dtype=np.float32)
            intensities[row_index, :peaks_count] = np.array(
                [peak.intensity for peak in spectrum_peaks],
                dtype=np.float32,
            )

        return mzs, intensities, num_peaks

    def _infer_chunk(self, mzs: np.ndarray, intensities: np.ndarray, num_peaks: np.ndarray) -> np.ndarray:
        infer_inputs = [
            self._build_infer_input("mzs", mzs),
            self._build_infer_input("intens", intensities),
            self._build_infer_input("num_peaks", num_peaks),
        ]

        try:
            result = self._client.infer(
                model_name=self._config.model_name,
                model_version=self._config.model_version,
                inputs=infer_inputs,
                outputs=[grpcclient.InferRequestedOutput(self._output_name)],
                client_timeout=self._config.infer_timeout_seconds,
            )
        except Exception as error:
            raise SpectrumInferenceError(f"Triton inference request failed: {error}") from error

        try:
            output = result.as_numpy(self._output_name)
        except Exception as error:
            raise SpectrumInferenceError(f"Failed to read Triton output '{self._output_name}': {error}") from error

        if output is None:
            raise SpectrumInferenceError("Triton returned empty output for spectrum encoder.")

        return output.astype(np.float32)

    @staticmethod
    def _build_infer_input(name: str, values: np.ndarray) -> grpcclient.InferInput:
        infer_input = grpcclient.InferInput(name, values.shape, np_to_triton_dtype(values.dtype))
        infer_input.set_data_from_numpy(values)
        return infer_input


def np_to_triton_dtype(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "FP32"
    if dtype == np.int32:
        return "INT32"
    if dtype == np.int64:
        return "INT64"
    raise SpectrumInferenceError(f"Unsupported numpy dtype for Triton input: {dtype}")


@lru_cache(maxsize=1)
def get_spectrum_encoder_client() -> SpectrumEncoderClient:
    config = SpectrumInferenceConfig()
    return SpectrumEncoderClient(config)
