FROM pgvector/pgvector:pg16

COPY initdb/ /docker-entrypoint-initdb.d/
COPY scripts/ /opt/db-scripts/

RUN chmod -R 755 /docker-entrypoint-initdb.d /opt/db-scripts
