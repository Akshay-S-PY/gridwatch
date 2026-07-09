"""
SQL safety layer for the NL query engine — the main risk surface of text-to-SQL.

Defence in depth:
  1. Static validation: strip fences, require exactly one statement, require it to
     be a SELECT/WITH, reject any write/DDL keyword.
  2. Execution in a READ ONLY transaction with a statement timeout and row cap, so
     even if validation is somehow bypassed the database itself refuses writes.
"""
import re

from sqlalchemy import text

from db.config import engine

MAX_ROWS = 500
STATEMENT_TIMEOUT_MS = 10_000

# Write / DDL / admin keywords that must never appear as a statement start or token.
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|"
    r"COPY|MERGE|CALL|DO|VACUUM|ANALYZE|REINDEX|REFRESH|COMMENT|SET|RESET|"
    r"LOCK|PREPARE|EXECUTE)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    pass


def clean_sql(raw: str) -> str:
    """Strip markdown fences / stray prose and trailing semicolons."""
    s = raw.strip()
    # pull the contents of a ```sql ... ``` block if present
    fence = re.search(r"```(?:sql)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    return s.rstrip(";").strip()


def validate(sql: str) -> str:
    """Return the cleaned SQL if it's a safe single read-only SELECT, else raise."""
    s = clean_sql(sql)
    if not s:
        raise UnsafeSQLError("Empty query.")

    # Single statement only — no stacked queries.
    if ";" in s:
        raise UnsafeSQLError("Multiple statements are not allowed.")

    lowered = s.lstrip("(").lstrip()
    if not re.match(r"(?is)^(select|with)\b", lowered):
        raise UnsafeSQLError("Only SELECT (or WITH ... SELECT) queries are allowed.")

    if _FORBIDDEN.search(s):
        raise UnsafeSQLError("Query contains a forbidden (write/DDL) keyword.")

    return s


def run(sql: str) -> list[dict]:
    """
    Validate then execute in a READ ONLY transaction with a statement timeout.
    Returns up to MAX_ROWS rows as dicts.
    """
    safe = validate(sql)
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # READ ONLY must be set before any query in the transaction.
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}"))
            result = conn.execute(text(safe))
            rows = [dict(r) for r in result.mappings().fetchmany(MAX_ROWS)]
        finally:
            trans.rollback()  # never commit — read-only by construction
    return rows
