CREATE INDEX IF NOT EXISTS idx_molecular_search_monoisotopic_mass
    ON molecular_search USING btree (monoisotopic_mass);

CREATE INDEX IF NOT EXISTS idx_molecular_search_mol_embedding_hnsw
    ON molecular_search USING hnsw (mol_embedding vector_cosine_ops);
