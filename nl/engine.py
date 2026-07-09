"""
The NL query engine — the DAZN-equivalent conversational layer.

Flow:  question
        -> retrieve similar past anomalies from Qdrant (grounding context)
        -> LLM generates one read-only SQL query
        -> validate + execute against a read-only connection (one repair retry on error)
        -> LLM explains the result in plain English
        -> return {answer, sql, rows, similar}
"""
import json
import logging

from llm import client as llm
from nl import rag
from nl.schema_context import SYSTEM_PROMPT
from nl.sql_guard import UnsafeSQLError, run

logger = logging.getLogger(__name__)

MAX_ROWS_TO_LLM = 50  # rows shown to the model when it writes the explanation


def _generate_sql(question: str, similar: list[str], error: str | None = None) -> str:
    context = ""
    if similar:
        context = "Similar past anomalies (for context):\n- " + "\n- ".join(similar) + "\n\n"
    repair = ""
    if error:
        repair = (
            f"\n\nYour previous query failed with: {error}\n"
            "Return a corrected query."
        )
    user = (
        f"{context}Operator question: {question}\n\n"
        "Return ONLY the SQL query — no prose, no markdown fences." + repair
    )
    return llm.chat(SYSTEM_PROMPT, user, max_tokens=500, temperature=0.0)


def _explain(question: str, sql: str, rows: list[dict]) -> str:
    preview = rows[:MAX_ROWS_TO_LLM]
    system = (
        "You are GridWatch's operations analyst. Answer the operator's question in "
        "plain English using ONLY the query result. Be concrete and concise (2-4 "
        "sentences). Quote key numbers with units (gCO2/kWh, %, m/s). If the result "
        "is empty, say so plainly and suggest why."
    )
    user = (
        f"Question: {question}\n\n"
        f"SQL run:\n{sql}\n\n"
        f"Result rows (JSON, up to {MAX_ROWS_TO_LLM}):\n{json.dumps(preview, default=str)}"
    )
    return llm.chat(system, user, max_tokens=400)


def answer(question: str) -> dict:
    """Answer an operator question end-to-end. Never raises — returns an error field."""
    if not llm.available():
        return {"question": question, "error": "LLM not configured (set OPENAI_API_KEY)."}

    similar = rag.search(question, k=5)

    sql = ""
    rows: list[dict] = []
    last_error = None
    for attempt in range(2):  # one repair retry
        try:
            raw_sql = _generate_sql(question, similar, error=last_error)
            rows = run(raw_sql)          # validates + executes read-only
            from nl.sql_guard import clean_sql
            sql = clean_sql(raw_sql)
            last_error = None
            break
        except UnsafeSQLError as e:
            last_error = f"unsafe query rejected ({e})"
            logger.warning(f"nl: {last_error}")
        except Exception as e:  # noqa: BLE001 — DB/SQL error, feed back for repair
            last_error = str(e).split("\n")[0][:300]
            logger.warning(f"nl: query failed (attempt {attempt+1}): {last_error}")

    if last_error is not None:
        return {"question": question, "sql": sql, "rows": [],
                "similar": similar, "error": f"Could not run a valid query: {last_error}"}

    explanation = _explain(question, sql, rows)
    return {
        "question": question,
        "answer": explanation,
        "sql": sql,
        "rows": rows,
        "row_count": len(rows),
        "similar": similar,
    }
