import json
import uuid
import duckdb
import pandas as pd
import sqlglot
import re
from dataclasses import dataclass, field
from typing import Optional

from sqlglot import exp
from pydantic import BaseModel, Field

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from aws_config import (
    AWS_REGION,
    AWS_PROFILE,
    BEDROCK_MODEL_ID
)


# ==========================================================
# 1. INITIALIZE BEDROCK
# ==========================================================

llm_config = {
    "model": BEDROCK_MODEL_ID,
    "region_name": AWS_REGION,
    "temperature": 0
}

if AWS_PROFILE:
    llm_config["credentials_profile_name"] = AWS_PROFILE

llm = ChatBedrockConverse(**llm_config)


# ==========================================================
# 2. STRUCTURED SQL OUTPUT
# ==========================================================

class SQLResponse(BaseModel):

    sql: str = Field(
        description="A single valid DuckDB SQL statement."
    )


# ==========================================================
# 3. READ CSV
# ==========================================================

def read_csv(source):

    df = pd.read_csv(source)

    if df.empty:
        raise ValueError("CSV file contains no data.")

    if not df.columns.is_unique:
        raise ValueError("CSV contains duplicate column names.")

    return df


# ==========================================================
# 4. INSPECT DATASET
# ==========================================================

def inspect_csv(df):

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict()
    }


# ==========================================================
# 5. GENERATE SQL USING BEDROCK
# ==========================================================

def generate_csv_sql(question, df):

    parser = PydanticOutputParser(
        pydantic_object=SQLResponse
    )

    schema = "\n".join(
        f"{column}: {dtype}"
        for column, dtype in df.dtypes.items()
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert DuckDB SQL engineer.

        Convert the user's data transformation request
        into one valid DuckDB SQL statement.

        The editable dataset is stored in a table
        named working.

        STRICT RULES:

        1. Generate only UPDATE, DELETE, or INSERT.
        2. Use only the table named working.
        3. Use only existing columns.
        4. Never generate DROP, CREATE, ALTER, TRUNCATE,
           MERGE, COPY, ATTACH, or other commands.
        5. Generate exactly one SQL statement.
        6. Do not access external files or databases.
        7. Do not generate subqueries or joins.
        8. UPDATE and DELETE must have a WHERE clause.
        9. Use DuckDB-compatible SQL.
        10. Return only the required structured output.
        11. When the user explicitly requests a change to every row,
            use WHERE TRUE for UPDATE or DELETE.
        12. For percentage decreases, multiply the numeric column
            by (1 - percentage / 100).
        13. For percentage increases, multiply the numeric column
            by (1 + percentage / 100).

        Dataset schema:
        {schema}

        {format_instructions}
        """),
        ("human", "{question}")
    ])

    chain = prompt | llm | parser

    result = chain.invoke({
        "question": question,
        "schema": schema,
        "format_instructions": parser.get_format_instructions()
    })

    return result.sql


# ==========================================================
# 6. VALIDATE GENERATED SQL
# ==========================================================

def validate_csv_sql(sql):

    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL cannot be empty.")

    statements = sqlglot.parse(
        sql,
        read="duckdb"
    )

    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed.")

    statement = statements[0]

    if not isinstance(
        statement,
        (exp.Update, exp.Delete, exp.Insert)
    ):
        raise ValueError(
            "Only UPDATE, DELETE, and INSERT are allowed."
        )

    target = statement.this

    if not isinstance(target, exp.Table):
        raise ValueError("Invalid target table.")

    if (
        target.name.lower() != "working"
        or target.db
        or target.catalog
    ):
        raise ValueError(
            "SQL can modify only the working dataset."
        )

    if isinstance(statement, (exp.Update, exp.Delete)):

        if statement.args.get("where") is None:
            raise ValueError(
                "UPDATE and DELETE require a WHERE clause."
            )

    # Restrict the SQL to simple local data transformations.
    # No subqueries, joins, external access, or SQL functions.

    forbidden = (
        exp.Select,
        exp.Subquery,
        exp.Join,
        exp.From,
        exp.With,
        exp.Func,
        exp.Command,
        exp.Returning
    )

    for node in statement.walk():

        if isinstance(node, forbidden):
            raise ValueError(
                "Unsupported SQL expression detected."
            )

        if isinstance(node, exp.Table) and node is not target:
            raise ValueError(
                "References to other tables are not allowed."
            )

    return sql


# ==========================================================
# 7. EXECUTE SQL LOCALLY USING DUCKDB
# ==========================================================

def execute_csv_sql(df, sql):

    validated_sql = validate_csv_sql(sql)

    # DuckDB runs entirely in memory.
    # External file/database access is disabled.

    con = duckdb.connect(
        database=":memory:",
        config={
            "enable_external_access": "false"
        }
    )

    try:

        con.register("source_df", df)

        con.execute(
            'CREATE TABLE working AS SELECT * FROM source_df'
        )

        con.unregister("source_df")

        before_count = con.execute(
            "SELECT COUNT(*) FROM working"
        ).fetchone()[0]

        con.execute(validated_sql)

        updated_df = con.execute(
            "SELECT * FROM working"
        ).df()

        after_count = len(updated_df)

        return {
            "dataframe": updated_df,
            "rows_before": before_count,
            "rows_after": after_count,
            "sql": validated_sql
        }

    finally:

        con.close()


# ==========================================================
# 8. CSV WORKSPACE CLASS
# ==========================================================

@dataclass
class CSVWorkspace:

    original_df: pd.DataFrame

    working_df: pd.DataFrame = field(init=False)

    saved_df: Optional[pd.DataFrame] = field(
        default=None,
        init=False
    )

    previous_df: Optional[pd.DataFrame] = field(
        default=None,
        init=False
    )

    pending_sql: Optional[str] = field(
        default=None,
        init=False
    )

    pending_change_id: Optional[str] = field(
        default=None,
        init=False
    )

    status: str = field(
        default="Workspace initialized.",
        init=False
    )

    def __post_init__(self):

        self.original_df = self.original_df.copy(deep=True)

        self.working_df = self.original_df.copy(deep=True)

    # ------------------------------------------------------
    # CHECK PENDING CHANGE
    # ------------------------------------------------------

    @property
    def has_pending_change(self):

        return self.previous_df is not None

    # ------------------------------------------------------
    # APPLY BULK CHANGE
    # ------------------------------------------------------

    def apply_bulk_change(self, question):

        if self.has_pending_change:
            raise ValueError(
                "Accept or reject the pending change first."
            )

        instruction = question.strip()

        # --------------------------------------------------
        # BLOCK COLUMN MODIFICATIONS THROUGH BULK CHAT
        # --------------------------------------------------

        column_request = re.search(
            r"\b(?:add|create|remove|delete|drop|rename)\b"
            r".*\bcolumns?\b"
            r"|\bcolumns?\b.*"
            r"\b(?:add|create|remove|delete|drop|rename)\b",
            instruction,
            flags=re.IGNORECASE
        )

        if column_request:
            raise ValueError(
                "To add, rename, or delete columns, please use "
                "'⚙️ Manage Working Dataset Columns' above the "
                "Working Dataset instead of bulk transformation."
            )

        # --------------------------------------------------
        # DELETE FIRST / LAST N ROWS
        # --------------------------------------------------
        
        row_match = re.fullmatch(
            r"(?:delete|remove)\s+(?:the\s+)?"
            r"(first|last)\s+"
            r"(?:(\d+)\s+)?rows?"
            r"(?:\s+(?:of|from)\s+(?:the\s+)?"
            r"(?:working\s+)?(?:dataset|table))?",
            instruction,
            flags=re.IGNORECASE
        )

        if row_match:
            position = row_match.group(1).lower()
            count = int(row_match.group(2) or 1)

            total_rows = len(self.working_df)

            if count <= 0 or count > total_rows:
                raise ValueError(
                    f"Invalid row count. Dataset contains {total_rows} rows."
                )

            if position == "first":
                sql = f"DELETE FROM working WHERE rowid < {count}"
            else:
                sql = f"DELETE FROM working WHERE rowid >= {total_rows - count}"

            return self.apply_sql_change(sql)

        # --------------------------------------------------
        # EXISTING BEDROCK BULK TRANSFORMATION
        # --------------------------------------------------

        sql = generate_csv_sql(
            instruction,
            self.working_df
        )

        return self.apply_sql_change(sql)

    # ------------------------------------------------------
    # APPLY GENERATED SQL
    # ------------------------------------------------------

    def apply_sql_change(self, sql):

        if self.has_pending_change:
            raise ValueError(
                "Accept or reject the pending change first."
            )

        previous = self.working_df.copy(deep=True)

        result = execute_csv_sql(
            previous,
            sql
        )

        self.previous_df = previous

        self.working_df = result["dataframe"]

        self.pending_sql = result["sql"]

        self.pending_change_id = uuid.uuid4().hex

        self.saved_df = None

        self.status = (
            "Bulk change applied. "
            "Review the working dataset and accept or reject."
        )

        return {
            "status": self.status,
            "sql": self.pending_sql,
            "rows_before": result["rows_before"],
            "rows_after": result["rows_after"]
        }

    # ------------------------------------------------------
    # ACCEPT BULK CHANGE
    # ------------------------------------------------------

    def accept_bulk_change(self):

        if not self.has_pending_change:
            raise ValueError("No pending change.")

        self.previous_df = None

        self.pending_sql = None

        self.pending_change_id = None

        self.status = "Bulk change accepted."

        return self.status

    # ------------------------------------------------------
    # REJECT BULK CHANGE
    # ------------------------------------------------------

    def reject_bulk_change(self):

        if not self.has_pending_change:
            raise ValueError("No pending change.")

        self.working_df = self.previous_df.copy(deep=True)

        self.previous_df = None

        self.pending_sql = None

        self.pending_change_id = None

        self.status = (
            "Bulk change rejected. "
            "Previous working dataset restored."
        )

        return self.status

    # ------------------------------------------------------
    # MANUAL DATASET EDITING
    # ------------------------------------------------------

    def update_working_dataset(self, edited_df):

        if self.has_pending_change:
            raise ValueError(
                "Resolve the pending bulk change first."
            )

        if not isinstance(edited_df, pd.DataFrame):
            raise TypeError(
                "Edited dataset must be a Pandas DataFrame."
            )

        if not edited_df.columns.is_unique:
            raise ValueError(
                "Duplicate column names are not allowed."
            )

        self.working_df = edited_df.copy(deep=True)

        self.saved_df = None

        self.status = "Manual changes updated."

        return self.status

    # ------------------------------------------------------
    # RESET ALL CHANGES
    # ------------------------------------------------------

    def reset_all_changes(self):

        self.working_df = self.original_df.copy(deep=True)

        self.previous_df = None

        self.pending_sql = None

        self.pending_change_id = None

        self.saved_df = None

        self.status = "All changes reversed."

        return self.status

    # ------------------------------------------------------
    # SAVE DATASET
    # ------------------------------------------------------

    def save(self, use_original=False):

        if self.has_pending_change:
            raise ValueError(
                "Accept or reject the pending change first."
            )

        source = (
            self.original_df
            if use_original
            else self.working_df
        )

        self.saved_df = source.copy(deep=True)

        self.status = (
            f"Dataset saved successfully. "
            f"{len(source)} rows, "
            f"{len(source.columns)} columns."
        )

        return self.status

    # ------------------------------------------------------
    # GET DATASET INFORMATION
    # ------------------------------------------------------

    def get_info(self):

        return {
            "original": inspect_csv(self.original_df),
            "working": inspect_csv(self.working_df),
            "saved": self.saved_df is not None,
            "pending_change": self.has_pending_change,
            "status": self.status
        }


# ==========================================================
# 9. LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    file_path = input(
        "Enter CSV file path: "
    ).strip()

    df = read_csv(file_path)

    workspace = CSVWorkspace(df)

    print("\nOriginal Dataset:")
    print(workspace.original_df.head())

    print("\nWorking Dataset:")
    print(workspace.working_df.head())

    while True:

        question = input(
            "\nEnter bulk change (or exit): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:

            result = workspace.apply_bulk_change(question)

            print("\nGenerated SQL:")
            print(result["sql"])

            print("\nUpdated Working Dataset:")
            print(workspace.working_df.head(20))

            decision = input(
                "\nAccept change? (yes/no): "
            ).strip().lower()

            if decision == "yes":

                print(
                    workspace.accept_bulk_change()
                )

            else:

                print(
                    workspace.reject_bulk_change()
                )

            print("\nCurrent Working Dataset:")
            print(workspace.working_df.head(20))

        except Exception as error:

            print("\nError:", error)

    print("\nWorkspace session ended.")