from pathlib import Path
import logging
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from app.db_search_client import DatabaseSearchError, get_db_search_client
from app.file_formats import SUPPORTED_EXTENSIONS, SUPPORTED_FORMATS_MESSAGE
from app.models import AnnotateSpectrumResponse, MoleculeCandidate, SpectrumAnnotationResult
from app.spectrum_parser import SpectrumParserError, parse_uploaded_spectra
from app.spectrum_encoder_client import (
    SpectrumInferenceError,
    get_spectrum_encoder_client,
)

def _configure_logger() -> logging.Logger:
    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(logging.INFO)

    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        app_logger.addHandler(handler)

    app_logger.propagate = False
    return app_logger


logger = _configure_logger()

app = FastAPI(
    title="CSMP Spectrum Annotation API",
    version="0.1.0",
    description="FastAPI backend for spectrum-to-molecule annotation.",
)

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post(
    "/annotate-spectrum",
    response_model=AnnotateSpectrumResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def annotate_spectrum(file: UploadFile = File(...)) -> AnnotateSpectrumResponse:
    file_name = file.filename or ""
    extension = Path(file_name).suffix.lower()
    logger.info("Received /annotate-spectrum request for file: %s", file_name)

    # Check if the file extension is supported
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=SUPPORTED_FORMATS_MESSAGE,
        )

    # Parse the uploaded file and extract spectra
    try:
        parsed_spectra = await parse_uploaded_spectra(file)
    
    except SpectrumParserError as error:
        logger.warning("Spectrum parsing error for file %s: %s", file_name, error)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    
    except Exception as error:
        logger.exception("Unexpected parser failure for file %s", file_name)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Failed to parse uploaded spectrum file. "
                "The file may be corrupted or does not match its extension format."
            ),
        ) from error

    # TODO: send parsed_spectra to inference + DB search pipeline.
    # Prepare response with parsed spectra and placeholder messages for candidates
    results: list[SpectrumAnnotationResult] = []
    valid_spectra = [spectrum for spectrum in parsed_spectra if spectrum.precursor_mz is not None]
    embeddings_by_spectrum_id: dict[str, list[float]] = {}
    candidates_by_spectrum_id: dict[str, list[MoleculeCandidate]] = {}
    db_errors_by_spectrum_id: dict[str, str] = {}

    if valid_spectra:
        try:
            encoder_client = get_spectrum_encoder_client()
            embeddings = encoder_client.encode(valid_spectra)

            if embeddings.shape[0] != len(valid_spectra):
                raise SpectrumInferenceError(
                    "Encoder output batch size does not match parsed spectra batch size."
                )

            for index, spectrum in enumerate(valid_spectra):
                embeddings_by_spectrum_id[spectrum.spectrum_id] = embeddings[index].astype(float).tolist()

            logger.info(
                "Generated embeddings for %d spectra using Triton model",
                len(valid_spectra),
            )

            db_search_client = get_db_search_client()
            for spectrum in valid_spectra:
                spectrum_embedding = embeddings_by_spectrum_id.get(spectrum.spectrum_id)
                if spectrum_embedding is None or spectrum.precursor_mz is None:
                    logger.warning(
                        "Missing embedding or precursor_mz for spectrum %s. Skipping DB search.",
                        spectrum.spectrum_id,
                    )
                    continue

                try:
                    candidates_by_spectrum_id[spectrum.spectrum_id] = db_search_client.search_candidates(
                        precursor_mz=spectrum.precursor_mz,
                        embedding=spectrum_embedding,
                    )
                except DatabaseSearchError as error:
                    logger.warning(
                        "Molecular DB search failed for spectrum %s: %s",
                        spectrum.spectrum_id,
                        error,
                    )
                    db_errors_by_spectrum_id[spectrum.spectrum_id] = str(error)
        except SpectrumInferenceError as error:
            logger.warning("Spectrum encoder inference failed: %s", error)
        except DatabaseSearchError as error:
            logger.warning("Molecular DB client initialization failed: %s", error)
        except Exception as error:
            logger.exception("Unexpected spectrum encoder failure")
    
    # If precursor_mz is missing, include parsing message and skip candidate generation for that spectrum.
    for spectrum in parsed_spectra:
        if spectrum.precursor_mz is None:
            results.append(
                SpectrumAnnotationResult(
                    spectrum_id=spectrum.spectrum_id,
                    precursor_mz=None,
                    candidates=None,
                    message=spectrum.parsing_message
                    or "Missing precursor_mz in input spectrum. Candidate search is unavailable.",
                )
            )
            logger.info("Parsed spectrum %s is missing precursor_mz. Added parsing message to response.", spectrum.spectrum_id)
            continue
        
        # For spectra with precursor_mz, we would normally run the inference + DB search pipeline to generate candidates.
        results.append(
            SpectrumAnnotationResult(
                spectrum_id=spectrum.spectrum_id,
                precursor_mz=spectrum.precursor_mz,
                candidates=candidates_by_spectrum_id.get(spectrum.spectrum_id),
                message=(
                    "Spectrum parsed and encoded successfully. No molecular candidates found in the configured mass window."
                    if (
                        spectrum.spectrum_id in embeddings_by_spectrum_id
                        and spectrum.spectrum_id not in db_errors_by_spectrum_id
                        and not candidates_by_spectrum_id.get(spectrum.spectrum_id)
                    )
                    else (
                        "Spectrum parsed, encoded, and searched successfully."
                    )
                    if (
                        spectrum.spectrum_id in embeddings_by_spectrum_id
                        and spectrum.spectrum_id not in db_errors_by_spectrum_id
                        and bool(candidates_by_spectrum_id.get(spectrum.spectrum_id))
                    )
                    else (
                        "Spectrum parsed successfully, but molecular DB search is unavailable. "
                        "Candidate search is unavailable."
                    )
                    if spectrum.spectrum_id in db_errors_by_spectrum_id
                    else (
                        "Spectrum parsed successfully, but encoder inference is unavailable. "
                        "Candidate search is unavailable."
                    )
                ),
            )
        )

    logger.info("Prepared %d spectrum results for response", len(results))

    return AnnotateSpectrumResponse(
        status="accepted",
        file_name=file_name,
        file_type=SUPPORTED_EXTENSIONS[extension],
        message=(
            f"Successfully parsed {len(parsed_spectra)} spectra. "
            "Spectrum encoder inference and molecular DB candidate search were attempted."
        ),
        results=results,
    )
