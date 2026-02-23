from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache

import psycopg

from app.models import MoleculeCandidate


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
    ppm_tolerance: float = float(os.getenv("SEARCH_PPM_TOLERANCE", "1000"))
    top_k: int = int(os.getenv("SEARCH_TOP_K", "10"))
    connect_timeout_seconds: int = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5"))


class DbSearchClient:
    def __init__(self, config: DatabaseSearchConfig):
        self._config = config

    def search_candidates(
        self,
        *,
        precursor_mz: float,
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

        lower_mass = precursor_mz * (1 - effective_ppm * 1e-6)
        upper_mass = precursor_mz * (1 + effective_ppm * 1e-6)
        vector_literal = _to_pgvector_literal(embedding)

        query = f"""
            WITH mass_filtered AS (
                SELECT
                    smiles,
                    monoisotopic_mass,
                    mol_embedding
                FROM {self._config.table_name}
                WHERE monoisotopic_mass BETWEEN %s AND %s
            )
            SELECT
                smiles,
                monoisotopic_mass,
                (mol_embedding <=> CAST(%s AS vector)) AS cosine_distance
            FROM mass_filtered
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
                    cursor.execute(
                        query,
                        (
                            lower_mass,
                            upper_mass,
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

@lru_cache(maxsize=1)
def get_db_search_client() -> DbSearchClient:
    return DbSearchClient(DatabaseSearchConfig())
