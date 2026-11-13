"""Session-isolated DuckDB backend; Redshift-compatible tool interface."""
import hashlib
import json
import re
from contextvars import ContextVar
from uuid import uuid4
import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp
from langchain_core.tools import tool
from aws_config import MAX_RESULT_ROWS

_ACTIVE_DB = ContextVar("active_demo_db", default=None)

def bind_database(con):
    _ACTIVE_DB.set(con)

def connection():
    con = _ACTIVE_DB.get()
    if con is None:
        raise RuntimeError("Connect to the demo environment first.")
    return con

def new_database():
    con = duckdb.connect(":memory:", config={"enable_external_access": "false", "allow_unsigned_extensions": "false", "memory_limit": "256MB", "threads": "2"})
    con.execute("CREATE SCHEMA IF NOT EXISTS public")
    return con

def import_csv(con, csv_bytes, table_name, schema_name="public"):
    """Explicit import only; existing tables cannot be replaced."""
    from io import BytesIO
    if len(csv_bytes) > 5_000_000:
        raise ValueError("CSV must be 5 MB or smaller")
    df = pd.read_csv(BytesIO(csv_bytes))
    if df.empty or not df.columns.is_unique or len(df.columns) > 100:
        raise ValueError("CSV must contain rows and unique columns (maximum 100)")
    for name in (schema_name, table_name, *df.columns):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise ValueError(f"Unsupported identifier: {name!r}")
    def q(s): return '"' + s.replace('"','""') + '"'
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {q(schema_name)}")
    # Registering an in-process DataFrame is permitted; no filesystem access.
    temp = 'csv_' + uuid4().hex
    con.register(temp, df)
    try:
        con.execute(f"CREATE TABLE {q(schema_name)}.{q(table_name)} AS SELECT * FROM {q(temp)}")
    finally:
        con.unregister(temp)
    return len(df)

def validate_sql(sql, operation):
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must not be empty")
    statements = sqlglot.parse(sql, read="redshift")
    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Exactly one SQL statement is required")
    stmt = statements[0]
    allowed = (exp.Select, exp.Union, exp.Intersect, exp.Except) if operation == "read" else (exp.Insert, exp.Update, exp.Delete) if operation == "write" else ()
    if not isinstance(stmt, allowed):
        raise ValueError("Unsupported SQL operation")
    forbidden = (exp.Create, exp.Drop, exp.Alter, exp.Merge, exp.Command, exp.Copy)
    if any(isinstance(node, forbidden) for node in stmt.walk()):
        raise ValueError("Unsafe SQL construct")
    # Only user-managed public schema tables may be changed.
    if operation == "write":
        for t in stmt.find_all(exp.Table):
            if t.db and t.db.lower() != "public":
                raise ValueError("Writes restricted to public schema")
    for fn in stmt.find_all(exp.Func):
        name = fn.sql_name().lower() if hasattr(fn, "sql_name") else ""
        if name in {"read_csv", "read_csv_auto", "read_parquet", "read_json", "sqlite_scan", "postgres_scan", "httpfs", "query", "query_table", "install", "load", "read_text", "write_file"}:
            raise ValueError("External access functions are disabled")
    return sql

def _run(sql, limit=None):
    validate_sql(sql, "read")
    stmt = sqlglot.parse_one(sql, read="redshift")
    translated = stmt.sql(dialect="duckdb")
    # SQL result cap applied outside SQL to avoid modifying generated statement.
    con = connection()
    result = con.execute(translated)
    df = result.fetch_df()
    if limit is not None: df = df.head(limit)
    return df.to_json(orient="records", date_format="iso")

@tool
def execute_read_query(sql: str) -> str:
    """Run one validated SELECT query on the isolated demo database and return JSON rows."""
    return _run(sql, MAX_RESULT_ROWS)

def sql_fingerprint(sql):
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()

@tool
def prepare_write_query(sql: str) -> str:
    """Validate SQL and prepare a fingerprint for human approval without executing it."""
    validate_sql(sql, "write")
    return json.dumps({"sql": sql, "fingerprint": sql_fingerprint(sql), "status": "pending_approval"})

def execute_write_query(sql, approved_fingerprint):
    validate_sql(sql, "write")
    if not approved_fingerprint or sql_fingerprint(sql) != approved_fingerprint:
        raise PermissionError("SQL approval fingerprint mismatch")
    con = connection()
    translated = sqlglot.parse_one(sql, read="redshift").sql(dialect="duckdb")
    con.execute("BEGIN TRANSACTION")
    try:
        cursor = con.execute(translated)
        affected = cursor.fetchone()[0] if cursor.description and cursor.description[0][0] == 'Count' else 0
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return {"status": "success", "statement_id": uuid4().hex, "affected_rows": affected}

def get_updated_records(sql, id_column, ids):
    stmt = sqlglot.parse_one(sql, read="redshift")
    if not isinstance(stmt, exp.Update) or not isinstance(stmt.this, exp.Table) or not ids: return []
    values = [exp.Literal.number(str(x)) if isinstance(x, int) else exp.Literal.string(str(x)) for x in ids]
    query = exp.select("*").from_(stmt.this.copy()).where(exp.In(this=exp.column(id_column), expressions=values))
    return json.loads(execute_read_query.invoke({"sql": query.sql(dialect="redshift")}))

@tool
def get_database_schema() -> str:
    """Discover current demo table columns and data types dynamically."""
    return _run("SELECT table_schema, table_name, column_name, data_type, ordinal_position FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_schema, table_name, ordinal_position")

@tool
def get_database_tables() -> str:
    """Discover available user tables dynamically."""
    return _run("SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_schema, table_name")

def get_table_details(schema, table):
    if not any(r['table_schema']==schema and r['table_name']==table for r in json.loads(get_database_tables.invoke({}))):
        raise ValueError("Destination table is not available")
    sql = """SELECT column_name, data_type, is_nullable, column_default, character_maximum_length,
    numeric_precision, numeric_scale, ordinal_position FROM information_schema.columns
    WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position"""
    df = connection().execute(sql, [schema, table]).fetch_df()
    return json.loads(df.to_json(orient="records"))

def test_connection():
    return connection().execute("SELECT current_database()").fetchone()

database_tools = [execute_read_query, prepare_write_query, get_database_schema, get_database_tables]
