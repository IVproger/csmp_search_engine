TRUNCATE TABLE molecular_search_staging;

COPY molecular_search_staging (
    formula,
    smiles,
    inchikey,
    monoisotopic_mass,
    mol_embedding
)
FROM '/seed/molecules_with_embeddings.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    QUOTE '"',
    ESCAPE '"'
);

INSERT INTO molecular_search (
    formula,
    smiles,
    inchikey,
    monoisotopic_mass,
    mol_embedding
)
SELECT
    formula,
    smiles,
    inchikey,
    monoisotopic_mass,
    (
        '[' || regexp_replace(trim(both FROM trim(both '[]' FROM mol_embedding)), '[[:space:]]+', ',', 'g') || ']'
    )::vector(256)
FROM molecular_search_staging
ON CONFLICT (inchikey) DO UPDATE
SET
    formula = EXCLUDED.formula,
    smiles = EXCLUDED.smiles,
    monoisotopic_mass = EXCLUDED.monoisotopic_mass,
    mol_embedding = EXCLUDED.mol_embedding;

TRUNCATE TABLE molecular_search_staging;

REINDEX INDEX idx_molecular_search_monoisotopic_mass;
REINDEX INDEX idx_molecular_search_mol_embedding_hnsw;
