from typing import Literal
from pydantic import BaseModel, Field


class Peak(BaseModel):
    mz: float = Field(..., description="Peak m/z value")
    intensity: float = Field(..., description="Peak intensity")

class ParsedSpectrum(BaseModel):
    spectrum_id: str = Field(..., description="Unique spectrum identifier within source file")
    precursor_mz: float = Field(..., description="Precursor m/z")
    adduct: str | None = Field(default=None, description="Adduct string when available")
    formula: str | None = Field(default=None, description="Molecular formula when available")
    peaks: list[Peak] = Field(default_factory=list, description="Spectrum peaks")

class MoleculeCandidate(BaseModel):
    smiles: str
    mass: float
    score: float = Field(..., ge=0.0, le=1.0)

class SpectrumAnnotationResult(BaseModel):
    spectrum_id: str
    candidates: list[MoleculeCandidate] = Field(default_factory=list)

class AnnotateSpectrumResponse(BaseModel):
    status: Literal["accepted", "processed"]
    file_name: str
    file_type: Literal["mzML", "MGF"]
    message: str
    results: list[SpectrumAnnotationResult] = Field(default_factory=list)
