import os

import psycopg2
from psycopg2.extensions import connection


def get_connection() -> connection:
    """Get PostgreSQL connection, preferring environment variables over defaults."""
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "vrag"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "postgres"),
    )
