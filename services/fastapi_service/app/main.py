from pathlib import Path
import logging
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from app.file_formats import SUPPORTED_EXTENSIONS, SUPPORTED_FORMATS_MESSAGE
from app.models import AnnotateSpectrumResponse, SpectrumAnnotationResult
from app.spectrum_parser import SpectrumParserError, parse_uploaded_spectra

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

    # Prepare response with parsed spectra and placeholder messages for candidates
    results: list[SpectrumAnnotationResult] = []
    
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
                candidates=None,
                message=(
                    "Spectrum parsed successfully."
                ),
            )
        )

    logger.info("Prepared %d spectrum results for response", len(results))

    # TODO: send parsed_spectra to inference + DB search pipeline.
    return AnnotateSpectrumResponse(
        status="accepted",
        file_name=file_name,
        file_type=SUPPORTED_EXTENSIONS[extension],
        message=(
            f"Successfully parsed {len(parsed_spectra)} spectra. "
            "Inference pipeline is not implemented yet."
        ),
        results=results,
    )
