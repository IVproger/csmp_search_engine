from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
import psycopg
from app.models import MoleculeCandidate
from app.utils import _get_mass_candidates

class DatabaseSearchError(RuntimeError):
    pass

@dataclass(frozen=True)
class DatabaseSearchConfig:
    host: str = os.getenv("POSTGRES_HOST", "postgres_service")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname: str = os.getenv("POSTGRES_DB", "molecular_search_db")
    user: str = os.getenv("POSTGRES_USER", "csmp_user")
    password: str = os.getenv("POSTGRES_PASSWORD", "csmp_password")
    table_name: str = os.getenv("POSTGRES_MOLECULAR_TABLE", "molecular_search")
    ppm_tolerance: float = float(os.getenv("SEARCH_PPM_TOLERANCE", "100"))
    top_k: int = int(os.getenv("SEARCH_TOP_K", "10"))
    min_mass_window_da: float = float(os.getenv("SEARCH_MIN_MASS_WINDOW_DA", "0.01"))
    allow_vector_only_fallback: bool = os.getenv("SEARCH_ALLOW_VECTOR_ONLY_FALLBACK", "true").lower() == "true"
    connect_timeout_seconds: int = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5"))

class DbSearchClient:
    def __init__(self, config: DatabaseSearchConfig):
        self._config = config

    def search_candidates(
        self,
        *,
        precursor_mz: float,
        adduct: str | None = None,
        charge: int | None = None,
        embedding: list[float],
        ppm_tolerance: float | None = None,
        top_k: int | None = None,
    ) -> list[MoleculeCandidate]:
        if precursor_mz <= 0:
            return []

        if not embedding:
            raise DatabaseSearchError("Empty query embedding received for molecular DB search.")

        if not all(math.isfinite(value) for value in embedding):
            raise DatabaseSearchError("Query embedding contains non-finite values.")

        effective_ppm = ppm_tolerance if ppm_tolerance is not None else self._config.ppm_tolerance
        effective_top_k = top_k if top_k is not None else self._config.top_k

        if effective_ppm <= 0:
            effective_ppm = self._config.ppm_tolerance

        neutral_mass_candidates = _get_mass_candidates(
            precursor_mz=precursor_mz,
            adduct=adduct,
            charge=charge,
        )
        ppm_windows = _get_ppm_windows(effective_ppm)

        vector_literal = _to_pgvector_literal(embedding)

        mass_filtered_query = f"""
            WITH mass_filtered AS (
                SELECT
                    smiles,
                    monoisotopic_mass,
                    mol_embedding
                FROM {self._config.table_name}
                WHERE (monoisotopic_mass BETWEEN %s AND %s)
            )
            SELECT
                smiles,
                monoisotopic_mass,
                (mol_embedding <=> CAST(%s AS vector)) AS cosine_distance
            FROM mass_filtered
            ORDER BY mol_embedding <=> CAST(%s AS vector)
            LIMIT %s
        """

        vector_only_query = f"""
            SELECT
                smiles,
                monoisotopic_mass,
                (mol_embedding <=> CAST(%s AS vector)) AS cosine_distance
            FROM {self._config.table_name}
            ORDER BY mol_embedding <=> CAST(%s AS vector)
            LIMIT %s
        """

        try:
            with psycopg.connect(
                host=self._config.host,
                port=self._config.port,
                dbname=self._config.dbname,
                user=self._config.user,
                password=self._config.password,
                connect_timeout=self._config.connect_timeout_seconds,
            ) as connection:
                with connection.cursor() as cursor:
                    rows = []
                    for ppm in ppm_windows:
                        for neutral_mass in neutral_mass_candidates:
                            delta = max(neutral_mass * (ppm / 1e6), self._config.min_mass_window_da)
                            lower_mass = neutral_mass - delta
                            upper_mass = neutral_mass + delta

                            cursor.execute(
                                mass_filtered_query,
                                (
                                    lower_mass,
                                    upper_mass,
                                    vector_literal,
                                    vector_literal,
                                    effective_top_k,
                                ),
                            )
                            rows = cursor.fetchall()
                            if rows:
                                break
                        if rows:
                            break

                    if not rows and self._config.allow_vector_only_fallback:
                        cursor.execute(
                            vector_only_query,
                            (
                                vector_literal,
                                vector_literal,
                                effective_top_k,
                            ),
                        )
                        rows = cursor.fetchall()
        except Exception as error:
            raise DatabaseSearchError(f"Molecular DB search failed: {error}") from error

        candidates: list[MoleculeCandidate] = []
        for smiles, monoisotopic_mass, cosine_distance in rows:
            if cosine_distance is None:
                continue
            score = max(0.0, min(1.0, 1.0 - float(cosine_distance)))
            candidates.append(
                MoleculeCandidate(
                    smiles=smiles,
                    mass=float(monoisotopic_mass),
                    similarity_score= round(score * 100, 4),
                )
            )

        return candidates

def _to_pgvector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"

def _get_ppm_windows(base_ppm: float) -> list[float]:
    return [float(window) for window in sorted({base_ppm, base_ppm * 5, base_ppm * 20})]

@lru_cache(maxsize=1)
def get_db_search_client() -> DbSearchClient:
    return DbSearchClient(DatabaseSearchConfig())
