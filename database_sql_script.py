"""Conservative, session-local SQL-script importer for the DuckDB demo.

Supports only CREATE SCHEMA, CREATE TABLE and INSERT ... VALUES. Uploads are
never executed on the currently active database; a fresh DB is swapped in only
when the entire script succeeds. This is a demo guard, not a hardened sandbox.
"""
from __future__ import annotations

import re
from pathlib import Path
import sqlglot
from sqlglot import exp

MAX_SCRIPT_BYTES = 1_000_000
MAX_STATEMENTS = 100
MAX_TABLES = 30
MAX_ROWS = 10_000
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
_RESERVED = {"information_schema", "pg_catalog", "pg_internal", "main", "temp", "system"}


def _identifier(value):
    if not value or not _IDENT.fullmatch(str(value)):
        raise ValueError(f"Unsupported SQL identifier: {value!r}")
    if str(value).lower() in _RESERVED:
        raise ValueError(f"Reserved schema/name: {value!r}")


def _table(node):
    if not isinstance(node, exp.Table):
        raise ValueError("Only ordinary database tables are supported")
    if node.catalog:
        raise ValueError("Cross-database references are not supported")
    schema = node.db or "public"
    _identifier(schema)
    _identifier(node.name)
    if schema.lower() != "public":
        raise ValueError("Uploaded tables must use the public schema")
    return node.name.lower()


def validate_upload_script(raw: bytes) -> list[str]:
    if not raw or len(raw) > MAX_SCRIPT_BYTES:
        raise ValueError("SQL script must be nonempty and at most 1 MB")
    try:
        script = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("Upload a UTF-8 encoded SQL script") from exc
    try:
        statements = sqlglot.parse(script, read="duckdb")
    except Exception as exc:
        raise ValueError(f"SQL parsing failed: {exc}") from exc
    statements = [s for s in statements if s is not None]
    if not statements or len(statements) > MAX_STATEMENTS:
        raise ValueError("Script must contain 1–100 supported SQL statements")

    created = set()
    inserted_rows = 0
    approved = []
    for stmt in statements:
        # Disallow hidden queries, file/URL access, extension commands, etc.
        forbidden = (exp.Select, exp.Subquery, exp.Command, exp.Copy,
                     exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Merge)
        if any(isinstance(n, forbidden) for n in stmt.walk()):
            raise ValueError("SQL script contains an unsupported operation")
        if isinstance(stmt, exp.Create):
            kind = str(stmt.args.get("kind") or "").upper()
            if kind == "SCHEMA":
                name = stmt.this.sql(dialect="duckdb").replace('"', '').strip()
                if name.lower() != "public":
                    raise ValueError("Only CREATE SCHEMA public is supported")
            elif kind == "TABLE":
                schema = stmt.this
                if not isinstance(schema, exp.Schema):
                    raise ValueError("CREATE TABLE must specify columns explicitly")
                table = _table(schema.this)
                if table in created:
                    raise ValueError("Duplicate CREATE TABLE in uploaded script")
                created.add(table)
                if len(created) > MAX_TABLES:
                    raise ValueError("Maximum 30 tables per script")
                # Restrict DDL to simple column definitions, PK/FK, UNIQUE,
                # NOT NULL and CHECK; no computed/default function calls.
                if stmt.args.get("expression") is not None or stmt.args.get("properties") is not None:
                    raise ValueError("CREATE TABLE AS and table properties are not supported")
                for n in stmt.walk():
                    if isinstance(n, (exp.Anonymous, exp.Func, exp.Dot, exp.CTE)):
                        raise ValueError("Functions and computed expressions are not supported in DDL")
            else:
                raise ValueError("Only CREATE SCHEMA and CREATE TABLE are supported")
        elif isinstance(stmt, exp.Insert):
            target = stmt.this
            if isinstance(target, exp.Schema):
                target = target.this
            table = _table(target)
            if table not in created:
                raise ValueError("INSERT target must be created earlier in the same script")
            values = stmt.expression
            if not isinstance(values, exp.Values):
                raise ValueError("Only INSERT ... VALUES is supported")
            inserted_rows += len(values.expressions)
            if inserted_rows > MAX_ROWS:
                raise ValueError("Maximum 10,000 inserted rows per script")
            for row in values.expressions:
                if not isinstance(row, exp.Tuple):
                    raise ValueError("INSERT rows must contain literal values")
                for value in row.expressions:
                    # Unary negatives and simple casts of literals are allowed.
                    node = value
                    while isinstance(node, (exp.Neg, exp.Cast)):
                        node = node.this
                    if not isinstance(node, (exp.Literal, exp.Null, exp.Boolean)):
                        raise ValueError("INSERT values must be constants (no functions or queries)")
        else:
            raise ValueError("Only CREATE SCHEMA, CREATE TABLE, INSERT ... VALUES are allowed")
        approved.append(stmt.sql(dialect="duckdb"))
    if not created:
        raise ValueError("SQL script must create at least one table")
    return approved


def load_sql_script(raw: bytes):
    """Build a NEW isolated database; caller swaps it in only on success."""
    from database import new_database
    statements = validate_upload_script(raw)
    con = new_database()
    try:
        con.execute("BEGIN TRANSACTION")
        for statement in statements:
            con.execute(statement)
        con.execute("COMMIT")
        count = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'").fetchone()[0]
        if not count:
            raise ValueError("No user tables were created")
        return con
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        con.close()
        raise


def load_builtin_medical():
    """Trusted packaged medical demo seed, not user input."""
    from database import new_database
    con = new_database()
    try:
        script = Path(__file__).with_name("medical_schema.sql").read_text(encoding="utf-8")
        con.execute(script)
        return con
    except Exception:
        con.close()
        raise


def switch_database(st, con, source: str):
    """Reset cross-database SQL/CSV state and refresh live metadata."""
    from uuid import uuid4
    import json
    from database import bind_database, get_database_schema, get_database_tables
    old = st.session_state.get("demo_db")
    st.session_state.demo_db = con
    bind_database(con)
    st.session_state.database_source = source
    st.session_state.live_tables = json.loads(get_database_tables.invoke({}))
    st.session_state.live_columns = json.loads(get_database_schema.invoke({}))
    st.session_state.aws_connected = True
    st.session_state.aws_status = ("Medical demo database connected" if source == "medical" else "Uploaded SQL database connected")
    st.session_state.redshift_thread = uuid4().hex
    st.session_state.redshift_approval = None
    st.session_state.redshift_messages = []
    st.session_state.csv_messages = []
    st.session_state.csv_upload_thread = None
    st.session_state.csv_upload_pending = False
    st.session_state.csv_upload_result = None
    st.session_state.csv_save_message = False
    st.session_state.csv_review_notice = False
    st.session_state.csv_editor_version = st.session_state.get("csv_editor_version", 0) + 1
    st.session_state.pop("csv_workspace", None)
    st.session_state.pop("csv_file_token", None)
    st.session_state.pop("demo_graph", None)
    st.session_state.pop("csv_graph", None)
    st.session_state.pop("csv_target", None)
    st.session_state.pop("selected_explorer_table", None)
    st.session_state.demo_archive = {}
    from demo_archive import bind_archive
    bind_archive(st.session_state.demo_archive)
    if old is not None and old is not con:
        try:
            old.close()
        except Exception:
            pass
