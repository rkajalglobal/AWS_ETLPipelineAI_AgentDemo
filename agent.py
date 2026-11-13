import json
import re
from typing import TypedDict

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.simplify import simplify
from sqlglot.optimizer.qualify import qualify
from sqlglot.schema import MappingSchema

from langchain_aws import ChatBedrockConverse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from aws_config import AWS_REGION, AWS_PROFILE, BEDROCK_MODEL_ID

from database import (
    execute_read_query,
    execute_write_query,
    get_database_schema,
    prepare_write_query,
    validate_sql,
    get_updated_records,
)


# ==========================================================
# 1. BEDROCK MODEL
# ==========================================================

model_config = {
    "model": BEDROCK_MODEL_ID,
    "region_name": AWS_REGION,
    "temperature": 0,
}

if AWS_PROFILE:
    model_config["credentials_profile_name"] = AWS_PROFILE

llm = ChatBedrockConverse(**model_config)


# ==========================================================
# 2. STRING OUTPUT PARSER
# ==========================================================

parser = StrOutputParser()


# ==========================================================
# 4. LANGGRAPH STATE
# ==========================================================

class AgentState(TypedDict, total=False):
    question: str

    schema: str
    sql: str
    operation: str
    route: str

    result: str
    answer: str
    error: str

    selection: dict
    selection_applied: bool

    approval: bool
    approved_fingerprint: str
    entire_table: bool


# ==========================================================
# 5. SQL GENERATION PROMPT
# ==========================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert Amazon Redshift SQL generator.

Generate SQL using only the current user request, supplied database
schema, and explicitly provided selection metadata.

Rules:
- Return exactly one SQL statement without JSON, Markdown, or explanations.
- Generate only SELECT, INSERT, UPDATE, or DELETE.
- Use only tables and columns that exist in the supplied schema.
- Use fully qualified schema.table names.
- Determine table relationships dynamically from the supplied schema.
- Never invent database values, columns, relationships, or record IDs.
- Never reuse previous SQL queries or filters.
- Treat every new SELECT request independently.
- Generate SELECT queries regardless of whether matching records exist.

SCHEMA VALIDATION:
- Before generating SQL, verify that every referenced table exists
  in the supplied schema.
- Verify that every referenced column belongs to the correct table.
- Never reference a column from a table that does not contain it.
- When a requested column belongs to another table, use a JOIN
  based on relationships established by the supplied schema.
- Use table aliases to qualify columns in queries involving JOINs.
- Do not assume columns with similar names belong to the same table.
- Do not invent JOIN relationships.
- If the required table, column, or relationship cannot be established
  from the supplied schema, return CLARIFY: followed by a question.

TEXT MATCHING:
- For descriptive text searches, use ILIKE with leading and trailing
  % wildcards, including when matching multiple values.
- Use exact matching only for identifiers or explicitly exact values.
- Do not restrict results to one matching record or identifier unless
  explicitly requested.
- When searching across related tables, apply text filters to the
  table containing the descriptive column.
  
SUBQUERIES AND RELATIONSHIPS:
- Prefer explicit JOINs when retrieving data across related tables.
- Use JOIN conditions based only on relationships defined in
  the supplied database schema.
- Never use a scalar subquery with = unless the schema or
  query guarantees exactly one returned row.
- When a subquery may return multiple rows, use IN or EXISTS.
- Never assume that a patient, employee, customer, or other
  entity has only one related record.
- For one-to-many and many-to-many relationships, use JOINs
  rather than scalar subqueries.
- Use DISTINCT when appropriate to prevent duplicate results.
- Never invent relationships between tables.

SELECT:
- Generate a fresh SELECT statement for every new SELECT request.
- Include record IDs in row-level SELECT queries when available.
- Return all matching records unless the user requests a limit.
- Do not reuse previous selection filters for a new SELECT request.
- Use JOINs when requested information spans multiple tables.

UPDATE/DELETE:
- When the user refers to previously selected records, use
  WHERE __previous_selection__ as the identity restriction.
- Never invent or generate primary-key IDs for previously selected records.
- Do not add an additional primary-key condition unless the user
  explicitly specifies an ID.
- Preserve additional user-requested conditions involving dates,
  prices, statuses, names, or other non-primary-key columns.
- When the user refers to all previously selected records,
  do not add unnecessary filters.
- Never use previously selected IDs for an unrelated request.
- Allow unrestricted UPDATE/DELETE only when explicitly requested.
- If target records are ambiguous, return CLARIFY: followed by a question.
- Do not execute modifications; generate SQL for validation and approval.

MEMORY:
- Do not use conversation history.
- Use only the current question, database schema, and selection metadata.
- Selection metadata may be used only when the current request explicitly
  refers to previously selected records.
- Never infer current filters from previous questions or SQL statements.

OUTPUT:
- Return only the SQL statement.
- If clarification is required, return:
  CLARIFY: <question>

Database schema:
{schema}

Previous selection metadata:
{selection}
"""
    ),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()


# ==========================================================
# 6. GET DATABASE SCHEMA
# ==========================================================

def schema_node(state: AgentState):

    return {
        "schema": get_database_schema.invoke({})
    }


# ==========================================================
# 7. GENERATE SQL
# ==========================================================
def generate_node(state: AgentState):

    selection = state.get("selection") or {}

    selection_info = {
        "schema": selection.get("schema"),
        "table": selection.get("table"),
        "id_column": selection.get("id_column"),
        "count": selection.get("count"),
    } if selection else {}

    sql = chain.invoke({
        "schema": state["schema"],
        "selection": json.dumps(selection_info),
        "question": state["question"],
    }).strip()

    print("\n========== BEDROCK GENERATED SQL ==========")
    print(sql)
    print("===========================================\n")

    # Remove Markdown code fences.
    if sql.startswith("```"):
        sql = sql[3:].strip()

        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()

        if sql.endswith("```"):
            sql = sql[:-3].strip()

    if sql.upper().startswith("CLARIFY:"):
        return {
            "sql": "",
            "operation": "blocked",
            "answer": sql.partition(":")[2].strip(),
        }

    return {
        "sql": sql,
        "operation": "",
        "route": "",
        "selection_applied": False,
        "approval": False,
        "approved_fingerprint": "",
        "result": "",
        "answer": "",
        "error": "",
    }
# ==========================================================
# 8. CAPTURE SELECTED RECORD IDS
# ==========================================================

def capture_selection(sql, result):
    """Save IDs from a simple SELECT, including an unambiguous JOIN."""
    statement = sqlglot.parse_one(sql, read="redshift")
    if not isinstance(statement, exp.Select) or any(
        statement.find(kind) for kind in
        (exp.Subquery, exp.Group, exp.Distinct, exp.Union, exp.Having)
    ):
        return {}

    rows = json.loads(result)
    if not isinstance(rows, list) or not rows or len(rows) > 1000 or not isinstance(rows[0], dict):
        return {}

    # Track the primary (FROM) table only. The ID must be explicitly
    # projected from that table, not inferred from a joined table.
    from_clause = statement.args.get("from_")
    table = from_clause.this if from_clause else None
    if not isinstance(table, exp.Table) or not table.db:
        return {}

    table_alias = table.alias_or_name.lower()
    id_names = (f"{table.name.lower().removesuffix('s')}_id", "id")
    matching = []
    projections = statement.expressions
    has_join = bool(statement.args.get("joins"))

    for projection in projections:
        column = projection.this if isinstance(projection, exp.Alias) else projection
        if not isinstance(column, exp.Column):
            continue
        if column.name.lower() not in id_names:
            continue
        if column.table and column.table.lower() != table_alias:
            continue
        # Unqualified IDs in JOINs may refer to a different table.
        if has_join and not column.table:
            continue
        output_name = projection.alias_or_name
        if output_name in rows[0]:
            matching.append((column.name, output_name))

    # SELECT * returns ID fields even though the SQL AST has no explicit
    # ID projection. Only infer these fields for a single-table query.
    # JOIN wildcards are deliberately excluded because their source is ambiguous.
    if not matching and not has_join and any(
        isinstance(projection, exp.Star)
        or (isinstance(projection, exp.Column)
            and isinstance(projection.this, exp.Star)
            and (not projection.table or projection.table.lower() == table_alias))
        for projection in projections
    ):
        for name in id_names:
            candidates = [key for key in rows[0] if key.lower() == name]
            if len(candidates) == 1:
                matching.append((candidates[0], candidates[0]))
                break

    if len(matching) != 1:
        return {}

    id_column, output_name = matching[0]
    ids = [row.get(output_name) for row in rows]
    if any(value is None or isinstance(value, bool) or not isinstance(value, (int, str)) for value in ids):
        return {}
    if len(set(ids)) != len(ids):
        return {}

    return {
        "schema": table.db,
        "table": table.name,
        "id_column": id_column,
        "ids": ids,
        "count": len(ids),
    }


# ==========================================================
# 9. APPLY SAVED SELECTION USING SQLGLOT
# ==========================================================
def apply_selection(sql, selection):

    if not selection or not selection.get("ids"):
        raise ValueError(
            "No saved record selection is available. "
            "Run a SELECT query including the unique record ID first."
        )

    statement = sqlglot.parse_one(sql, read="redshift")

    if not isinstance(statement, (exp.Update, exp.Delete)):
        raise ValueError(
            "Saved selections support UPDATE and DELETE only."
        )

    table = statement.this

    if (
        not isinstance(table, exp.Table)
        or table.name.lower() != selection["table"].lower()
        or (table.db or "").lower() != selection["schema"].lower()
    ):
        raise ValueError(
            "The modification target does not match the selected table."
        )

    id_column = selection["id_column"]

    # The saved IDs are the authoritative record identities.
    values = [
        exp.Literal.number(str(value))
        if isinstance(value, int)
        else exp.Literal.string(value)
        for value in selection["ids"]
    ]

    selection_condition = exp.In(
        this=exp.column(id_column),
        expressions=values,
    )

    # Preserve legitimate additional filters.
    existing_where = statement.args.get("where")

    if existing_where is not None:

        original_condition = existing_where.this.copy()

        # Remove model-generated primary-key predicates.
        # Only remove standalone predicates joined by AND.
        # Never remove predicates inside OR/NOT expressions.
        def remove_id_conditions(node):

            if isinstance(node, exp.And):
                left = remove_id_conditions(node.this)
                right = remove_id_conditions(node.expression)

                if left is None:
                    return right

                if right is None:
                    return left

                return exp.and_(left, right)

            if isinstance(node, (exp.EQ, exp.In)):
                column = node.this

                if (
                    isinstance(column, exp.Column)
                    and column.name.lower() == id_column.lower()
                    and (
                        not column.table
                        or column.table.lower() == table.alias_or_name.lower()
                    )
                ):
                    return None

            return node.copy()

        filtered_condition = remove_id_conditions(original_condition)

        if filtered_condition is not None:
            selection_condition = exp.and_(
                selection_condition,
                filtered_condition
            )

    statement.set(
        "where",
        exp.Where(this=selection_condition)
    )

    return statement.sql(dialect="redshift")

def normalize_text_filters(sql, question, schema):
    """
    Convert general text comparisons into partial,
    case-insensitive searches using SQLGlot.
    """

    # Respect explicitly requested exact matching.
    exact_terms = (
        "exactly", "exact match", "equal to",
        "equals", "identical to"
    )

    if any(term in question.lower() for term in exact_terms):
        return sql

    tree = sqlglot.parse_one(sql, read="redshift")

    def make_ilike(column, value):
        return exp.ILike(
            this=column.copy(),
            expression=exp.Literal.string(
                f"%{value.this}%"
            )
        )

    def transform(node):

        # Convert column = 'text'
        if isinstance(node, exp.EQ):
            if (
                isinstance(node.this, exp.Column)
                and isinstance(node.expression, exp.Literal)
                and node.expression.is_string
            ):
                return make_ilike(node.this, node.expression)

        # Convert column IN ('text1', 'text2', ...)
        if isinstance(node, exp.In):
            if (
                isinstance(node.this, exp.Column)
                and node.expressions
                and all(
                    isinstance(value, exp.Literal)
                    and value.is_string
                    for value in node.expressions
                )
            ):
                conditions = [
                    make_ilike(node.this, value)
                    for value in node.expressions
                ]

                combined = conditions[0]

                for condition in conditions[1:]:
                    combined = exp.or_(combined, condition)

                return combined

        # Convert ILIKE 'text' into ILIKE '%text%'
        if isinstance(node, exp.ILike):
            if (
                isinstance(node.expression, exp.Literal)
                and node.expression.is_string
            ):
                value = node.expression.this

                if not value.startswith("%"):
                    value = "%" + value

                if not value.endswith("%"):
                    value += "%"

                return exp.ILike(
                    this=node.this.copy(),
                    expression=exp.Literal.string(value)
                )

        return node

    return tree.transform(transform).sql(dialect="redshift")

def router_node(state: AgentState):
    """Route on generated SQL only; never guess intent from English keywords."""
    if not state.get("sql"):
        return {"route": "blocked"}
    try:
        statements = sqlglot.parse(state["sql"], read="redshift")
        if len(statements) != 1 or statements[0] is None:
            raise ValueError("Expected one SQL statement")
        statement = statements[0]
    except Exception:
        return {"route": "blocked", "sql": "", "answer":
                "The SQL generator did not return valid SQL. Please rephrase your request."}
    if isinstance(statement, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return {"route": "read"}
    if isinstance(statement, (exp.Insert, exp.Update, exp.Delete)):
        return {"route": "write"}
    return {"route": "blocked", "sql": "", "answer":
            "Only SELECT, INSERT, UPDATE, and DELETE are supported."}


def validate_read_node(state: AgentState):
    try:
        sql = normalize_text_filters(state["sql"], state["question"], state["schema"])
        sql = validate_sql(sql, "read")
        return {"sql": sql, "operation": "read"}
    except Exception as exc:
        return {"sql": "", "operation": "blocked", "answer": f"SELECT validation failed: {exc}"}


def validate_write_node(state: AgentState):
    """Apply confirmed target IDs; do not resolve destination values yet."""
    try:
        sql = state["sql"]
        statement = sqlglot.parse_one(sql, read="redshift")
        selection_applied = False
        entire_table = False
        if isinstance(statement, (exp.Update, exp.Delete)):
            where = statement.args.get("where")
            condition = simplify(where.this.copy()) if where is not None else None
            unrestricted = where is None or (
                isinstance(condition, exp.Boolean) and condition.this is True
            )
            whole_table_requested = bool(re.search(
                r"\b(?:entire|whole)\s+table\b|"
                r"\b(?:all|every)\s+(?:rows?|records?|entries)\b|"
                r"\b(?:all|every)\s+\w+\s+(?:rows?|records?)\b",
                state["question"], re.IGNORECASE
            ))
            if unrestricted and whole_table_requested:
                entire_table = True
            else:
                selection = state.get("selection") or {}
                target = statement.this
                matches = (
                    selection.get("ids") and isinstance(target, exp.Table)
                    and target.name.lower() == selection.get("table", "").lower()
                    and (target.db or "").lower() == selection.get("schema", "").lower()
                )
                if not matches:
                    return {"sql": "", "operation": "blocked", "answer":
                            "Please select the data first on which you want to operate."}
                if whole_table_requested:
                    return {"sql": "", "operation": "blocked", "answer":
                            "For an entire-table operation, explicitly request all records without a filter."}
                # For a date-change request containing both an old and a new
                # ISO date, ensure the original date is present in the WHERE.
                # Otherwise the generated SQL could silently affect every ID.
                dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", state["question"])
                if isinstance(statement, exp.Update) and len(set(dates)) >= 2:
                    where_sql = where.this.sql(dialect="redshift") if where else ""
                    if not any(date in where_sql for date in dates):
                        return {"sql": "", "operation": "blocked", "answer":
                                "The generated UPDATE did not filter by the original date. "
                                "No changes made. Please specify the appointment ID or retry."}
                sql = apply_selection(sql, selection)
                selection_applied = True
        return {"sql": sql, "operation": "write", "selection_applied": selection_applied,
                "entire_table": entire_table}
    except Exception as exc:
        return {"sql": "", "operation": "blocked", "answer": f"WRITE validation failed: {exc}"}


def resolve_lookup_node(state: AgentState):
    """Run read-only SELECT subqueries, then replace verified scalar values."""
    try:
        statement = sqlglot.parse_one(state["sql"], read="redshift")
        if isinstance(statement, exp.Update):
            for assignment in statement.expressions:
                if not isinstance(assignment, exp.EQ):
                    continue
                rhs = assignment.expression
                if not isinstance(rhs, exp.Subquery):
                    continue
                lookup = rhs.this
                if not isinstance(lookup, exp.Select) or len(lookup.expressions) != 1:
                    raise ValueError("Destination lookup must return one column.")
                # Use the SQL SELECT already generated within the UPDATE.
                # First attempt case-insensitive exact matching, keeping underscores literal.
                exact_lookup = lookup.copy()
                def exact_text(node):
                    if (isinstance(node, exp.EQ) and isinstance(node.this, exp.Column)
                            and isinstance(node.expression, exp.Literal)
                            and node.expression.is_string):
                        return exp.EQ(
                            this=exp.Lower(this=node.this.copy()),
                            expression=exp.Lower(this=node.expression.copy()))
                    if (isinstance(node, exp.ILike) and isinstance(node.this, exp.Column)
                            and isinstance(node.expression, exp.Literal)
                            and node.expression.is_string):
                        value = node.expression.this
                        if re.search(r"[_\d]", value):
                            return exp.EQ(
                                this=exp.Lower(this=node.this.copy()),
                                expression=exp.Lower(this=exp.Literal.string(value.strip("%"))))
                    return node
                exact_lookup = exact_lookup.transform(exact_text)
                exact_lookup.set("limit", exp.Limit(expression=exp.Literal.number(2)))
                sql = validate_sql(exact_lookup.sql(dialect="redshift"), "read")
                rows = json.loads(execute_read_query.invoke({"sql": sql}))
                if not isinstance(rows, list):
                    raise ValueError("Destination lookup returned an invalid result.")
                # For natural-language labels, permit a broader search ONLY when exact
                # lookup found nothing. Never broaden an identifier with a numeric suffix.
                literals = [n.this for n in lookup.find_all(exp.Literal) if n.is_string]
                explicit_identifier = any(re.search(r"[_\d]", v) for v in literals)
                if not rows and not explicit_identifier:
                    broad = normalize_text_filters(lookup.sql(dialect="redshift"), "", state["schema"])
                    broad_tree = sqlglot.parse_one(broad, read="redshift")
                    broad_tree.set("limit", exp.Limit(expression=exp.Literal.number(2)))
                    sql = validate_sql(broad_tree.sql(dialect="redshift"), "read")
                    rows = json.loads(execute_read_query.invoke({"sql": sql}))
                if not rows:
                    raise ValueError("No matching destination record found. No changes made.")
                if len(rows) != 1:
                    raise ValueError("Multiple destination records match. Select an exact destination first. No changes made.")
                values = list(rows[0].values())
                if len(values) != 1 or values[0] is None or isinstance(values[0], bool):
                    raise ValueError("Destination lookup did not return one usable value.")
                value = values[0]
                if isinstance(value, (int, float)):
                    assignment.set("expression", exp.Literal.number(str(value)))
                elif isinstance(value, str):
                    assignment.set("expression", exp.Literal.string(value))
                else:
                    raise ValueError("Unsupported destination value type.")
            sql = statement.sql(dialect="redshift")
        else:
            sql = state["sql"]
        # Validate the final SQL that will be fingerprinted and approved.
        sql = validate_sql(sql, "write")
        return {"sql": sql, "operation": "write"}
    except Exception as exc:
        return {"sql": "", "operation": "blocked", "answer": str(exc)}


# ==========================================================
# SCHEMA VALIDATION (SQLGLOT + LIVE REDSHIFT METADATA)
# ==========================================================

def schema_validation_node(state: AgentState):
    """Reject SQL with unknown or incorrectly qualified tables/columns.

    Fetches live column metadata rather than parsing the human-readable
    schema prompt. This node does not execute the proposed SQL or modify it.
    """
    try:
        metadata_sql = """
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name, ordinal_position
        """
        metadata_raw = execute_read_query.invoke({"sql": metadata_sql})
        metadata_rows = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        if not isinstance(metadata_rows, list) or not metadata_rows:
            raise ValueError("Database column metadata is unavailable; SQL was not executed.")

        mapping = {}
        for row in metadata_rows:
            namespace = str(row["table_schema"]).lower()
            table = str(row["table_name"]).lower()
            column = str(row["column_name"]).lower()
            dtype = str(row.get("data_type") or "UNKNOWN")
            mapping.setdefault(namespace, {}).setdefault(table, {})[column] = dtype

        live_schema = MappingSchema(mapping, dialect="redshift", normalize=False)
        statement = sqlglot.parse_one(state["sql"], read="redshift")

        # Explicitly check table references, including nested SELECTs.
        # Do not confuse CTE names with physical tables.
        cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
        for table_ref in statement.find_all(exp.Table):
            if not table_ref.db and table_ref.name.lower() in cte_names:
                continue
            namespace = (table_ref.db or "public").lower()
            table_name = table_ref.name.lower()
            if table_name not in mapping.get(namespace, {}):
                raise ValueError(f"Table {namespace}.{table_name} does not exist or is not visible.")

        # SQLGlot checks column ownership, ambiguous unqualified columns,
        # and scoped references (JOINs, CTEs and subqueries).
        # Work on a copy: the original approved SQL is never rewritten here.
        qualify(
            statement.copy(),
            dialect="redshift",
            schema=live_schema,
            validate_qualify_columns=True,
            identify=False,
        )
        return {"operation": state["operation"]}
    except Exception as exc:
        return {
            "sql": "",
            "operation": "blocked",
            "answer": f"Schema validation failed: {exc}",
        }


# ==========================================================
# 11. EXECUTE SELECT
# ==========================================================

def read_node(state: AgentState):

    sql = state["sql"]

    try:
        result = execute_read_query.invoke({"sql": sql})

    except Exception as error:
        answer = f"Query execution failed: {error}"

        return {
            "sql": sql,
            "result": "",
            "answer": answer,
            "error": str(error),
        }

    rows = json.loads(result)

    selection = capture_selection(sql, result)

    count = len(rows) if isinstance(rows, list) else 0

    answer = f"Query completed. Returned {count} rows."

    if selection:
        answer += (
            f" Tracked {selection['count']} selected records "
            "for follow-up questions."
        )

    return {
        "sql": sql,
        "result": result,
        "answer": answer,
        "selection": selection,
        "error": "",
    }

# ==========================================================
# 12. HUMAN APPROVAL
# ==========================================================

def approval_node(state: AgentState):

    prepared = json.loads(
        prepare_write_query.invoke({
            "sql": state["sql"]
        })
    )

    if state.get("entire_table"):

        message = (
            "⚠️ WARNING: This SQL query has no effective WHERE "
            "restriction and will modify the ENTIRE table.\n\n"
            "If you want to modify only specific records, "
            "run a SELECT query again to select the desired data, "
            "then request the modification.\n\n"
            "Do you want to proceed with modifying the entire table?"
        )

    else:
        message = "Approve database modification?"

    decision = interrupt({
        "message": message,
        "sql": prepared["sql"],
        "fingerprint": prepared["fingerprint"],
        "entire_table": state.get("entire_table", False),
    })

    approved = (
        isinstance(decision, dict)
        and decision.get("approved") is True
    )

    return {
        "approval": approved,
        "approved_fingerprint": (
            prepared["fingerprint"]
            if approved
            else ""
        ),
    }


# ==========================================================
# 13. EXECUTE APPROVED WRITE
# ==========================================================

def write_node(state: AgentState):
    if not state.get("approval"):
        raise PermissionError("Human approval required.")

    sql = state["sql"]
    selection = state.get("selection") or {}
    try:
        result = execute_write_query(
            sql=sql, approved_fingerprint=state["approved_fingerprint"]
        )
    except Exception as exc:
        # Do not automatically retry a WRITE whose execution status may be uncertain.
        answer = f"Database modification failed or could not be confirmed: {exc}. Do not retry without checking the data."
        return {
            "result": "", "answer": answer, "error": str(exc),
            "selection": {}, "approval": False, "approved_fingerprint": "",
        }

    affected_rows = result.get("affected_rows", 0)
    statement = sqlglot.parse_one(sql, read="redshift")
    updated_records = []
    preview_error = ""
    if isinstance(statement, exp.Update):
        try:
            if state.get("selection_applied") and selection:
                updated_records = get_updated_records(
                    sql, selection["id_column"], selection["ids"]
                )
            elif state.get("entire_table") and isinstance(statement.this, exp.Table):
                preview_sql = f"SELECT * FROM {statement.this.sql(dialect='redshift')} LIMIT 200"
                updated_records = json.loads(execute_read_query.invoke({"sql": preview_sql}))
        except Exception as exc:
            preview_error = str(exc)

    if updated_records:
        answer = f"Database updated successfully. {affected_rows} row(s) affected. Showing updated records."
        output = json.dumps(updated_records, default=str)
    elif isinstance(statement, exp.Delete):
        answer = f"Database deletion completed. {affected_rows} row(s) affected."
        output = ""
    else:
        answer = f"Database modification completed. {affected_rows} row(s) affected."
        if isinstance(statement, exp.Update):
            answer += " Updated records could not be displayed."
        if preview_error:
            answer += f" Preview error: {preview_error}"
        output = ""

    return {
        "result": output, "answer": answer, "error": "",
        "selection": {}, "approval": False, "approved_fingerprint": "",
    }


def blocked_node(state: AgentState):
    answer = state["answer"]
    return {
        "result": "",
        "error": "",
    }


# ==========================================================
# 14. REJECT WRITE
# ==========================================================

def reject_node(state: AgentState):

    answer = (
        "Modification rejected. No changes made."
    )

    return {
        "result": answer,
        "answer": answer,
    }


# ==========================================================
# 15. BUILD LANGGRAPH
# ==========================================================

def build_graph(checkpointer=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("schema", schema_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("router", router_node)
    workflow.add_node("validate_read", validate_read_node)
    workflow.add_node("validate_write", validate_write_node)
    workflow.add_node("resolve_lookup", resolve_lookup_node)
    workflow.add_node("schema_validation", schema_validation_node)
    workflow.add_node("read", read_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("write", write_node)
    workflow.add_node("reject", reject_node)
    workflow.add_node("blocked", blocked_node)

    workflow.add_edge(START, "schema")
    workflow.add_edge("schema", "generate")
    workflow.add_edge("generate", "router")
    workflow.add_conditional_edges("router", lambda s: s["route"],
        {"read": "validate_read", "write": "validate_write", "blocked": "blocked"})
    workflow.add_conditional_edges("validate_read", lambda s: s["operation"],
        {"read": "schema_validation", "blocked": "blocked"})
    workflow.add_conditional_edges("validate_write", lambda s: s["operation"],
        {"write": "resolve_lookup", "blocked": "blocked"})
    workflow.add_conditional_edges("resolve_lookup", lambda s: s["operation"],
        {"write": "schema_validation", "blocked": "blocked"})
    workflow.add_conditional_edges("schema_validation", lambda s: s["operation"],
        {"read": "read", "write": "approval", "blocked": "blocked"})
    workflow.add_conditional_edges("approval", lambda s: "write" if s["approval"] else "reject",
        {"write": "write", "reject": "reject"})
    for node in ("read", "write", "reject", "blocked"):
        workflow.add_edge(node, END)
    return workflow.compile(checkpointer=checkpointer or InMemorySaver())


# ==========================================================
# 16. RUN AGENT
# ==========================================================

def run_agent(graph, question, thread_id):

    return graph.invoke(
        {
            "question": question
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        },
    )


# ==========================================================
# 17. RESUME AFTER HUMAN APPROVAL
# ==========================================================

def resume_agent(graph, approved, thread_id):

    return graph.invoke(
        Command(
            resume={
                "approved": approved
            }
        ),
        config={
            "configurable": {
                "thread_id": thread_id
            }
        },
    )


# ==========================================================
# 18. LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    graph = build_graph()

    thread_id = "local-development"

    while True:

        question = input(
            "\nAsk a question (or exit): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:

            result = run_agent(
                graph,
                question,
                thread_id
            )

            if "__interrupt__" in result:

                for item in result["__interrupt__"]:

                    print("\nPending SQL:")
                    print(item.value["sql"])

                decision = input(
                    "\nApprove? (yes/no): "
                ).strip().lower()

                result = resume_agent(
                    graph,
                    decision == "yes",
                    thread_id,
                )

            print(
                "\nSQL:",
                result.get("sql", "")
            )

            print(
                "\nAnswer:",
                result.get("answer", "")
            )

            print(
                "\nResult:",
                result.get("result", "")
            )

        except Exception as error:

            print(
                "\nError:",
                error
            )