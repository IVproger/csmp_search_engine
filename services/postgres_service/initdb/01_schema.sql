CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS molecular_search (
    formula TEXT,
    smiles TEXT NOT NULL,
    inchikey TEXT PRIMARY KEY,
    monoisotopic_mass DOUBLE PRECISION NOT NULL,
    mol_embedding VECTOR(256) NOT NULL
);

CREATE TABLE IF NOT EXISTS molecular_search_staging (
    formula TEXT,
    smiles TEXT,
    inchikey TEXT,
    monoisotopic_mass DOUBLE PRECISION,
    mol_embedding TEXT
);
