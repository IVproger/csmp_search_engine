from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from app.models import AnnotateSpectrumResponse

app = FastAPI(
    title="CSMP Spectrum Annotation API",
    version="0.1.0",
    description="FastAPI backend for spectrum-to-molecule annotation.",
)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".mzml": "mzML",
    ".mgf": "MGF",
}

@app.post(
    "/annotate-spectrum",
    response_model=AnnotateSpectrumResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def annotate_spectrum(file: UploadFile = File(...)) -> AnnotateSpectrumResponse:
    file_name = file.filename or ""
    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Use .mzML or .MGF files.",
        )

    return AnnotateSpectrumResponse(
        status="accepted",
        file_name=file_name,
        file_type=SUPPORTED_EXTENSIONS[extension],
        message="Endpoint contract is ready. Parsing/inference pipeline is not implemented yet.",
        results=[],
    )
