from typing import Literal
from pydantic import BaseModel, Field


class Peak(BaseModel):
    mz: float = Field(..., description="Peak m/z value")
    intensity: float = Field(..., description="Peak intensity")

class ParsedSpectrum(BaseModel):
    spectrum_id: str = Field(..., description="Unique spectrum identifier within source file")
    precursor_mz: float | None = Field(default=None, description="Precursor m/z")
    charge: int | None = Field(default=None, description="Precursor charge")
    adduct: str | None = Field(default=None, description="Adduct string when available")
    formula: str | None = Field(default=None, description="Molecular formula when available")
    peaks: list[Peak] = Field(default_factory=list, description="Spectrum peaks")
    parsing_message: str | None = Field(default=None, description="Parsing status for this spectrum")

class MoleculeCandidate(BaseModel):
    smiles: str
    mass: float
    similarity_score: float = Field(..., ge=0.0, le=100.0)

class SpectrumAnnotationResult(BaseModel):
    spectrum_id: str
    precursor_mz: float | None = Field(default=None)
    candidates: list[MoleculeCandidate] | None = Field(default=None)
    message: str | None = Field(default=None)

class AnnotateSpectrumResponse(BaseModel):
    status: Literal["accepted", "processed"]
    file_name: str
    file_type: Literal["mzML", "MGF", "JSON", "MSP"]
    message: str
    results: list[SpectrumAnnotationResult] = Field(default_factory=list)
