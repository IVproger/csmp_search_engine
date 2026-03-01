from __future__ import annotations
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import numpy as np
from fastapi import UploadFile
from matchms.importing import load_from_json, load_from_mgf, load_from_msp
from pymzml.run import Reader
from app.file_formats import SUPPORTED_EXTENSIONS
from app.models import ParsedSpectrum, Peak

class SpectrumParserError(ValueError):
    pass

def get_file_extension(file_name: str) -> str:
    return Path(file_name).suffix.lower()

async def parse_uploaded_spectra(file: UploadFile) -> list[ParsedSpectrum]:
    file_name = file.filename or ""
    extension = get_file_extension(file_name)

    if extension not in SUPPORTED_EXTENSIONS:
        raise SpectrumParserError("Unsupported file extension.")

    file_bytes = await file.read()
    if not file_bytes:
        raise SpectrumParserError("Uploaded file is empty.")

    with NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(file_bytes)
        temp_path = Path(temp_file.name)

    try:
        parsed_spectra = _parse_file(temp_path=temp_path, extension=extension)
    finally:
        temp_path.unlink(missing_ok=True)

    if not parsed_spectra:
        raise SpectrumParserError("No spectra were parsed from file.")

    return parsed_spectra


def _parse_file(temp_path: Path, extension: str) -> list[ParsedSpectrum]:
    try:
        if extension == ".mzml":
            return _parse_mzml(temp_path)
        if extension == ".mgf":
            return _parse_matchms_spectra(load_from_mgf(str(temp_path)), source_format="MGF")
        if extension == ".msp":
            return _parse_matchms_spectra(load_from_msp(str(temp_path)), source_format="MSP")
        if extension == ".json":
            return _parse_matchms_spectra(load_from_json(str(temp_path)), source_format="JSON")
    except Exception as error:
        raise SpectrumParserError(f"Failed to parse file: {error}") from error


def _parse_mzml(file_path: Path) -> list[ParsedSpectrum]:
    parsed: list[ParsedSpectrum] = []

    try:
        run = Reader(str(file_path))
    except Exception as error:
        raise SpectrumParserError(f"Failed to read mzML file: {error}") from error

    try:
        for index, spec in enumerate(run):
            try:
                peaks_raw = spec.peaks(peak_type="centroided")
                precursor_mz = _extract_mzml_precursor_mz(spec)
                parsing_message = None
                if precursor_mz is None:
                    parsing_message = "Missing precursor_mz in input spectrum. Candidate search is unavailable."

                mz_values = np.array([float(mz) for mz, _ in peaks_raw], dtype=np.float32)
                intensity_values = np.array([float(intensity) for _, intensity in peaks_raw], dtype=np.float32)
                normalized_intensities = normalize_spectrum_intensities(intensity_values, method="max_norm")
                peaks = [
                    Peak(mz=float(mz), intensity=float(intensity))
                    for mz, intensity in zip(mz_values, normalized_intensities)
                ]

                parsed.append(
                    ParsedSpectrum(
                        spectrum_id=str(spec.ID or index),
                        precursor_mz=precursor_mz,
                        charge=_parse_charge(
                            _first_non_empty(
                                spec.get("charge state"),
                                spec.get("charge"),
                                _extract_mzml_precursor_charge(spec),
                            )
                        ),
                        adduct=_first_non_empty(spec.get("adduct"), spec.get("precursor type")),
                        formula=_first_non_empty(
                            spec.get("molecular_formula"),
                            spec.get("formula"),
                            spec.get("molecular formula"),
                        ),
                        peaks=peaks,
                        parsing_message=parsing_message,
                    )
                )
            except Exception:
                continue
    except Exception as error:
        raise SpectrumParserError(f"Failed to parse mzML file content: {error}") from error

    return parsed


def _extract_mzml_precursor_mz(spec: Any) -> float | None:
    selected_precursors = getattr(spec, "selected_precursors", None) or []
    if selected_precursors:
        precursor_mz = selected_precursors[0].get("mz")
        if precursor_mz is not None:
            return _parse_precursor_mz_value(precursor_mz)

    fallback_value = _first_non_empty(
        spec.get("precursor_mz"),
        spec.get("precursor m/z"),
        spec.get("selected ion m/z"),
        spec.get("isolation window target m/z"),
    )
    return _parse_precursor_mz_value(fallback_value)


def _extract_mzml_precursor_charge(spec: Any) -> int | None:
    selected_precursors = getattr(spec, "selected_precursors", None) or []
    if not selected_precursors:
        return None

    return _parse_charge(
        _first_non_empty(
            selected_precursors[0].get("charge"),
            selected_precursors[0].get("charge state"),
        )
    )


def _parse_precursor_mz_value(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]

    if isinstance(value, str):
        normalized = value.strip().replace(",", " ")
        if not normalized:
            return None
        token = normalized.split()[0]
        try:
            return float(token)
        except ValueError:
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_matchms_spectra(spectra: Any, source_format: str) -> list[ParsedSpectrum]:
    parsed: list[ParsedSpectrum] = []

    try:
        for index, spectrum in enumerate(spectra):
                metadata = spectrum.metadata
                precursor_mz = _extract_matchms_precursor_mz(metadata)
                parsing_message = None
                if precursor_mz is None:
                    parsing_message = "Missing precursor_mz in input spectrum. Candidate search is unavailable."

                mz_values = np.asarray(spectrum.peaks.mz, dtype=np.float32)
                intensity_values = np.asarray(spectrum.peaks.intensities, dtype=np.float32)
                normalized_intensities = normalize_spectrum_intensities(intensity_values, method="max_norm")
                peaks = [
                    Peak(mz=float(mz), intensity=float(intensity))
                    for mz, intensity in zip(mz_values, normalized_intensities)
                ]

                parsed.append(
                    ParsedSpectrum(
                        spectrum_id=str(
                            _first_non_empty(
                                metadata.get("spectrum_id"),
                                metadata.get("id"),
                                metadata.get("scans"),
                                metadata.get("title"),
                                index,
                            )
                        ),
                        precursor_mz=precursor_mz,
                        charge=_parse_charge(
                            _first_non_empty(
                                metadata.get("charge"),
                                metadata.get("precursor_charge"),
                                metadata.get("charge state"),
                                metadata.get("charge_state"),
                            )
                        ),
                        adduct=_first_non_empty(metadata.get("adduct"), metadata.get("precursor_type")),
                        formula=_first_non_empty(
                            metadata.get("formula"),
                            metadata.get("molecular_formula"),
                        ),
                        peaks=peaks,
                        parsing_message=parsing_message,
                    )
                )
            
    except Exception as error:
        raise SpectrumParserError(
            f"Failed to parse {source_format} file content: {error}"
        ) from error

    return parsed


def _extract_matchms_precursor_mz(metadata: dict[str, Any]) -> float | None:
    for key in (
        "precursor_mz",
        "precursor mz",
        "precursor m/z",
        "pepmass",
        "parentmass",
        "parent_mass",
    ):
        value = metadata.get(key)
        parsed = _parse_precursor_mz_value(value)
        if parsed is not None:
            return parsed

    return None


def _parse_charge(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None

    if isinstance(value, str):
        normalized = value.strip().replace(" ", "")
        if not normalized:
            return None

        if normalized.endswith("+") or normalized.endswith("-"):
            sign = 1 if normalized.endswith("+") else -1
            digits = normalized[:-1]
            if not digits:
                return sign
            if digits.isdigit():
                return sign * int(digits)

        if normalized.startswith("+") or normalized.startswith("-"):
            sign = 1 if normalized.startswith("+") else -1
            digits = normalized[1:]
            if not digits:
                return sign
            if digits.isdigit():
                return sign * int(digits)

        if normalized.isdigit():
            return int(normalized)

    return None


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value in (None, ""):
            continue
        return value
    return None


def normalize_spectrum_intensities(
    intensity_values: np.ndarray,
    *,
    method: str = "max_norm",
) -> np.ndarray:
    values = np.asarray(intensity_values, dtype=np.float32)
    if values.size == 0:
        return values

    if method == "max_norm":
        max_intensity = float(np.max(values))
        if max_intensity > 0:
            normalized_intensities = (values / max_intensity) * 100.0
        else:
            normalized_intensities = values
        return normalized_intensities.astype(np.float32, copy=False)

    raise SpectrumParserError(f"Unsupported intensity normalization method: {method}")
