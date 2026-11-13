"""Create a persistent medical DuckDB file for local inspection."""
from pathlib import Path
import duckdb
root = Path(__file__).resolve().parent
path = root / "medical.duckdb"
if path.exists():
    raise SystemExit("medical.duckdb already exists; move or delete it before recreating.")
con = duckdb.connect(str(path))
try:
    con.execute((root / "medical_schema.sql").read_text(encoding="utf-8"))
    print("Created:", path)
    print(con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name").fetchall())
finally:
    con.close()
