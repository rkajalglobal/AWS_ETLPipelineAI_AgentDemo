import json
import hashlib
from datetime import datetime, timezone
from botocore.exceptions import ClientError
import re
import pandas as pd

from typing import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from aws_config import AWS_REGION, AWS_PROFILE, BEDROCK_MODEL_ID
from database import execute_read_query, get_table_details
from demo_archive import archive_csv


# ==========================================================
# 1. BEDROCK
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
# 2. STRUCTURED OUTPUT
# ==========================================================

class SchemaMatchResponse(BaseModel):
    compatible: bool
    suggestions: list[str] = Field(default_factory=list)


parser = PydanticOutputParser(
    pydantic_object=SchemaMatchResponse
)


# ==========================================================
# 3. LANGGRAPH STATE
# ==========================================================

class CSVAgentState(TypedDict, total=False):

    csv_schema: dict

    target_schema: str
    target_table: str

    redshift_schema: list[dict]

    approved: bool
    compatible: bool

    suggestions: list[str]
    load_columns: list[str]

    status: str
    csv_content: str
    s3_key: str
    upload_status: str


# ==========================================================
# 4. DISCOVER REDSHIFT TABLES
# ==========================================================

def get_available_tables():

    sql = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema NOT IN (
              'information_schema',
              'pg_catalog'
          )
        ORDER BY table_schema, table_name
    """

    result = execute_read_query.invoke({"sql": sql})

    return json.loads(result)


# ==========================================================
# 5. GET CSV SCHEMA
# ==========================================================

def get_csv_schema(df: pd.DataFrame):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    if not df.columns.is_unique:
        raise ValueError("Duplicate CSV column names.")

    return {
        "row_count": len(df),
        "columns": [
            {
                "name": str(name),
                "type": str(dtype),
                "has_nulls": bool(df[name].isna().any())
            }
            for name, dtype in df.dtypes.items()
        ]
    }


# ==========================================================
# 6. APPROVAL NODE
# ==========================================================

def approval_node(state: CSVAgentState):

    decision = interrupt({
        "message": "Approve uploading the saved CSV?",
        "target_schema": state["target_schema"],
        "target_table": state["target_table"]
    })

    approved = decision is True

    return {
        "approved": approved,
        "status": (
            "Upload approved."
            if approved
            else "Upload rejected."
        )
    }


def route_approval(state: CSVAgentState):

    return "get_schema" if state["approved"] else "end"


# ==========================================================
# 7. GET ANY REDSHIFT TABLE SCHEMA
# ==========================================================

def get_schema_node(state: CSVAgentState):

    schema_name = state["target_schema"]
    table_name = state["target_table"]

    available = get_available_tables()

    if not any(
        row["table_schema"] == schema_name
        and row["table_name"] == table_name
        for row in available
    ):
        raise ValueError(
            "Selected Redshift table does not exist "
            "or is not accessible."
        )

    # Identifiers are resolved against discovered tables.
    # Escape SQL string literals for metadata filtering.

    schema_literal = schema_name.replace("'", "''")
    table_literal = table_name.replace("'", "''")

    columns = get_table_details(schema_name, table_name)


    if not columns:
        raise ValueError(
            "No column metadata found for selected table."
        )

    return {
        "redshift_schema": columns,
        "status": "Redshift schema retrieved."
    }


# ==========================================================
# 8. IDENTIFY GENERATED / DEFAULT COLUMNS
# ==========================================================

def is_generated_column(column: dict):

    default = str(
        column.get("column_default") or ""
    ).lower()

    return (
        "identity" in default
        or "nextval(" in default
    )


def has_default(column: dict):

    return column.get("column_default") is not None


# ==========================================================
# 9. INDEPENDENT SCHEMA VALIDATION
# ==========================================================

def check_schema_compatibility(
    csv_schema: dict,
    redshift_schema: list[dict]
):

    csv_columns = {
        col["name"].lower(): col
        for col in csv_schema["columns"]
    }

    db_columns = {
        col["column_name"].lower(): col
        for col in redshift_schema
    }

    errors = []
    load_columns = []

    # Check every CSV column.

    for name, csv_col in csv_columns.items():

        if name not in db_columns:

            errors.append(
                f"CSV column '{name}' does not exist "
                "in the destination table."
            )

            continue

        db_col = db_columns[name]

        if is_generated_column(db_col):

            errors.append(
                f"Column '{name}' is auto-generated. "
                "Remove it from the CSV if Redshift "
                "should generate its values."
            )

            continue

        load_columns.append(db_col["column_name"])

        # Check nullability.

        if (
            db_col["is_nullable"] == "NO"
            and csv_col["has_nulls"]
        ):

            errors.append(
                f"Column '{name}' contains NULL values "
                "but the destination requires NOT NULL."
            )

        # Check basic type compatibility.

        csv_type = csv_col["type"].lower()
        db_type = db_col["data_type"].lower()
        # DuckDB type names differ from Redshift names; normalize by type family.
        db_type = {
            "int8": "bigint", "int4": "integer", "int2": "smallint",
            "int64": "bigint", "int32": "integer", "int16": "smallint",
            "hugeint": "bigint", "ubigint": "bigint", "uinteger": "integer",
            "double": "double precision", "float": "real", "float8": "double precision",
            "float4": "real", "bool": "boolean", "string": "varchar"
        }.get(db_type.split("(")[0], db_type.split("(")[0])
        if db_type.startswith("decimal"):
            db_type = "decimal"

        if csv_type.startswith("int"):

            allowed = {
                "smallint",
                "integer",
                "bigint",
                "numeric",
                "decimal",
                "real",
                "double precision"
            }

        elif csv_type.startswith("float"):

            allowed = {
                "real",
                "double precision",
                "numeric",
                "decimal"
            }

        elif csv_type.startswith("bool"):

            allowed = {"boolean"}

        elif csv_type.startswith("datetime"):

            allowed = {
                "date",
                "timestamp",
                "timestamp without time zone",
                "timestamp with time zone"
            }

        else:
            allowed = {
               "character varying",
               "character",
               "varchar",
               "char",
               "text",
               "date",
               "timestamp",
               "timestamp without time zone",
               "timestamp with time zone"
            }

        if db_type not in allowed:

            errors.append(
                f"Column '{name}': CSV type '{csv_type}' "
                f"does not match Redshift type '{db_type}'."
            )

    # Check destination columns missing from CSV.

    for name, db_col in db_columns.items():

        if name in csv_columns:
            continue

        if is_generated_column(db_col):
            continue

        if has_default(db_col):
            continue

        if db_col["is_nullable"] == "YES":
            continue

        errors.append(
            f"Required destination column '{name}' "
            "is missing from the CSV."
        )

    return {
        "compatible": not errors,
        "errors": errors,
        "load_columns": load_columns
    }


# ==========================================================
# 10. BEDROCK COLUMN MATCHING
# ==========================================================

def match_schema_node(state: CSVAgentState):

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an Amazon Redshift data engineer.

        Compare a CSV schema against a destination
        Redshift table schema.

        Requirements:

        - Work with any table and column names.
        - Identify mismatched column names.
        - Identify incompatible data types.
        - Identify missing required columns.
        - Recognize auto-generated identity columns.
        - Recognize columns with defaults.
        - Suggest corrections.
        - Auto-generated identity columns may be omitted from CSV files.
        - Pandas int64 is compatible with Redshift BIGINT.
        - Pandas integer types may be compatible with Redshift NUMERIC.
        - Pandas string columns containing valid ISO dates may be compatible with Redshift DATE.
        - Report only genuine incompatibilities requiring user action.
        - Do not report compatible columns as problems.
        

        Do not modify data.
        Do not generate SQL.
        Do not upload anything.

        CSV schema:
        {csv_schema}

        Redshift schema:
        {redshift_schema}

        {format_instructions}
        """),
        ("human", "Analyze the schema compatibility.")
    ])

    chain = prompt | llm | parser

    response = chain.invoke({
        "csv_schema": json.dumps(state["csv_schema"]),
        "redshift_schema": json.dumps(
            state["redshift_schema"]
        ),
        "format_instructions": parser.get_format_instructions()
    })

    return {
        "compatible": response.compatible,
        "suggestions": response.suggestions,
        "status": "Bedrock schema matching completed."
    }


# ==========================================================
# 11. VALIDATE BEDROCK RESULT
# ==========================================================

def validate_schema_node(state: CSVAgentState):

    validation = check_schema_compatibility(
        state["csv_schema"],
        state["redshift_schema"]
    )

    suggestions = list(dict.fromkeys(
        state.get("suggestions", [])
        + validation["errors"]
    ))

    return {
        "compatible": validation["compatible"],
        "suggestions": suggestions,
        "load_columns": validation["load_columns"],
        "status": (
            "Schema validation passed. Review Bedrock suggestions."
            if validation["compatible"]
            else "Schema validation failed. Review required corrections."
        )
    }


# ==========================================================
# 12. ROUTING
# ==========================================================

def route_schema_match(state: CSVAgentState):

    return (
        "ready"
        if state["compatible"]
        else "mismatch"
    )


def ready_node(state: CSVAgentState):

    return {
        "status": (
            "Schema matched. Ready for detailed data "
            "validation before upload."
        )
    }


def mismatch_node(state: CSVAgentState):

    return {
        "status": (
            "Upload stopped. Review suggestions, "
            "edit the working dataset, save, "
            "and approve upload again."
        )
    }


def upload_csv_to_s3(csv_content: str, schema_name: str, table_name: str):
    """Simulate S3 archival within the current visitor session."""
    return archive_csv(csv_content, schema_name, table_name)


def upload_s3_node(state: CSVAgentState):
    return upload_csv_to_s3(state["csv_content"], state["target_schema"], state["target_table"])


# ==========================================================
# 13. BUILD LANGGRAPH
# ==========================================================

workflow = StateGraph(CSVAgentState)

workflow.add_node("approval", approval_node)
workflow.add_node("get_schema", get_schema_node)
workflow.add_node("match_schema", match_schema_node)
workflow.add_node("validate_schema", validate_schema_node)
workflow.add_node("ready", ready_node)
workflow.add_node("mismatch", mismatch_node)
workflow.add_node("upload_s3", upload_s3_node)

workflow.add_edge(START, "approval")

workflow.add_conditional_edges(
    "approval",
    route_approval,
    {
        "get_schema": "get_schema",
        "end": END
    }
)

workflow.add_edge("get_schema", "match_schema")
workflow.add_edge("match_schema", "validate_schema")

workflow.add_conditional_edges(
    "validate_schema",
    route_schema_match,
    {
        "ready": "ready",
        "mismatch": "mismatch"
    }
)

workflow.add_edge("ready", "upload_s3")
workflow.add_edge("upload_s3", END)
workflow.add_edge("mismatch", END)

agent = workflow.compile(
    checkpointer=MemorySaver()
)


# ==========================================================
# 14. START AGENT
# ==========================================================

def start_csv_agent(
    df: pd.DataFrame,
    target_schema: str,
    target_table: str,
    thread_id: str
):

    state = {
        "csv_schema": get_csv_schema(df),
        "csv_content": df.to_csv(index=False),
        "target_schema": target_schema,
        "target_table": target_table,
        "approved": False,
        "compatible": False,
        "suggestions": [],
        "load_columns": [],
        "status": "Waiting for upload approval."
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    return agent.invoke(state, config=config)


# ==========================================================
# 15. RESUME AGENT
# ==========================================================

def resume_csv_agent(
    approved: bool,
    thread_id: str
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    return agent.invoke(
        Command(resume=approved),
        config=config
    )


# ==========================================================
# 16. LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    print("\nAvailable Redshift tables:")

    tables = get_available_tables()

    for table in tables:

        print(
            f"{table['table_schema']}."
            f"{table['table_name']}"
        )

    schema_name = input(
        "\nEnter schema name: "
    ).strip()

    table_name = input(
        "Enter destination table: "
    ).strip()

    csv_path = input(
        "Enter CSV file path: "
    ).strip()

    df = pd.read_csv(csv_path)

    thread_id = "csv-upload-test-001"

    result = start_csv_agent(
        df=df,
        target_schema=schema_name,
        target_table=table_name,
        thread_id=thread_id
    )

    if "__interrupt__" in result:

        decision = input(
            "\nApprove upload validation? (yes/no): "
        ).strip().lower()

        result = resume_csv_agent(
            approved=(decision == "yes"),
            thread_id=thread_id
        )

    print("\nStatus:", result.get("status"))

    print("Compatible:", result.get("compatible"))

    print("\nSuggestions:")

    for suggestion in result.get("suggestions", []):
        print("-", suggestion)

    print("\nColumns selected for loading:")
    print(result.get("load_columns", []))