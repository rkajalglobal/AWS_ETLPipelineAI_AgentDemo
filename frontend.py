"""AWS Intelligent Data Engineering Agent. Run: streamlit run frontend.py"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AWS Intelligent Data Engineering Agent", page_icon="☁️", layout="wide")
css = Path(__file__).with_name('style.css')
if css.exists():
    st.markdown(f'<style>{css.read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)
st.markdown("""
<style>
.aws-visual, .aws-bot {
    position: relative;
    top: 12px;
}
</style>
""", unsafe_allow_html=True)
st.markdown('''<div class="aws-hero">
<div class="aws-visual" aria-hidden="true">
  <div class="aws-cloud-icon"><svg viewBox="0 0 100 75" xmlns="http://www.w3.org/2000/svg"><path d="M25 60h52a18 18 0 0 0 0-36h-3A26 26 0 0 0 24 29a16 16 0 0 0 1 31Z" fill="#ffb3a7" stroke="#c85660" stroke-width="4" stroke-linejoin="round"/><path d="M30 66q21 11 43-1" fill="none" stroke="#d46a3b" stroke-width="5" stroke-linecap="round"/></svg></div>
  <span class="aws-orbit aws-orbit-one"></span><span class="aws-orbit aws-orbit-two"></span>
</div>
<div class="aws-heading"><div class="aws-hero-title">MedQuery AI — Medical Data Engineering Agent</div><div class="aws-hero-sub">Medical DuckDB (live) · Bedrock (live) · S3 archive (simulated) · LangGraph</div></div>
<div class="aws-bot" aria-label="Animated AI agent"><div class="aws-bot-antenna"></div><div class="aws-bot-face"><span class="aws-bot-eye"></span><span class="aws-bot-eye"></span><div class="aws-bot-mouth"></div></div><div class="aws-bot-label">AI AGENT</div></div>
</div>''', unsafe_allow_html=True)

def redshift_graph():
    from agent import build_graph
    from langgraph.checkpoint.memory import MemorySaver
    if 'demo_graph' not in st.session_state:
        st.session_state.demo_graph = build_graph(MemorySaver())
    return st.session_state.demo_graph

def database_tables():
    from database import get_database_tables
    return json.loads(get_database_tables.invoke({}))

def database_schema():
    from database import get_database_schema
    return json.loads(get_database_schema.invoke({}))

def add_message(key, role, text, sql=None, data=None):
    st.session_state[key].append(dict(role=role, text=str(text), sql=sql, data=data))

def render_chat(key):
    for msg in st.session_state[key]:
        with st.chat_message(msg['role']):
            st.markdown(msg['text'])
            if msg.get('sql'):
                with st.expander('Show SQL Query'):
                    st.code(msg['sql'], language='sql')
            if msg.get('data') is not None:
                with st.expander('Show Results', expanded=False):
                    if isinstance(msg['data'], pd.DataFrame):
                        data = msg['data']
                        st.caption(f'{len(data):,} rows returned. Displaying up to 200 rows.')
                        st.dataframe(data.head(200), use_container_width=True, height=min(38 * (min(len(data), 200) + 1), 600), hide_index=True)
                    elif isinstance(msg['data'], (dict, list)):
                        st.json(msg['data'], expanded=False)
                    else:
                        st.text(str(msg['data'])[:2000])

def display_redshift_result(result):

    if result.get('__interrupt__'):
        pending = result['__interrupt__'][0].value
        st.session_state.redshift_approval = pending

        add_message(
            'redshift_messages',
            'assistant',
            'Database WRITE requires your approval. Review the SQL below.',
            pending.get('sql')
        )
        return

    st.session_state.redshift_approval = None

    raw = result.get('result', '')

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)

            if isinstance(parsed, list):
                data = pd.DataFrame(parsed)
                message = (
                    f'Query completed. Returned {len(data):,} rows '
                    '(subject to result limit). '
                    'Open Show Results to view up to 200 rows.'
                )

            elif isinstance(parsed, dict):
                data = parsed
                message = (
                    'Database operation completed.'
                    if parsed.get('status') == 'success'
                    else str(parsed)
                )

            else:
                data, message = None, str(parsed)

        except (ValueError, TypeError):
            data, message = None, raw or 'Operation completed.'

    else:
        data, message = raw, 'Operation completed.'

    # Display the SQL for SELECT operations only.
    # WRITE SQL is already displayed in the approval message.
    answer = result.get('answer') or message
    sql = result.get('sql')

    if result.get('operation') == 'write':
        sql = None

    add_message(
        'redshift_messages',
        'assistant',
        answer,
        sql,
        data
    )
for key in ('redshift_messages', 'csv_messages'):
    st.session_state.setdefault(key, [])
st.session_state.setdefault('redshift_thread', uuid4().hex)
st.session_state.setdefault('redshift_approval', None)
st.session_state.setdefault('csv_editor_version', 0)
st.session_state.setdefault('csv_upload_thread', None)
st.session_state.setdefault('csv_upload_pending', False)
st.session_state.setdefault('csv_upload_result', None)
st.session_state.setdefault('csv_save_message', False)
st.session_state.setdefault('csv_review_notice', False)

from database import bind_database
from database_sql_script import load_builtin_medical, load_sql_script, switch_database
from demo_archive import bind_archive
if 'demo_db' not in st.session_state:
    st.session_state.demo_db = load_builtin_medical()
    st.session_state.database_source = 'medical'
if 'demo_archive' not in st.session_state:
    st.session_state.demo_archive = {}
bind_database(st.session_state.demo_db)
bind_archive(st.session_state.demo_archive)

# Automatically connect for first-time visitors: no manual setup required.
if 'aws_connected' not in st.session_state:
    try:
        from database import test_connection
        test_connection()
        st.session_state.live_tables = database_tables()
        st.session_state.live_columns = database_schema()
        st.session_state.aws_connected = True
        st.session_state.aws_status = 'Medical demo database connected'
    except Exception as exc:
        st.session_state.aws_connected = False
        st.session_state.aws_status = f'Connection failed: {exc}'

with st.sidebar:
    st.markdown('### ☁️ Demo Redshift Environment')
    if st.button('Connect to Demo Redshift Environment', use_container_width=True):
        try:
            from database import test_connection
            test_connection()
            # Load schema ONLY after a real, successful connection.
            live_tables = database_tables()
            live_columns = database_schema()
            st.session_state.aws_connected = True
            st.session_state.aws_status = 'Demo database connected'
            st.session_state.live_tables = live_tables
            st.session_state.live_columns = live_columns
        except Exception as exc:
            st.session_state.aws_connected = False
            st.session_state.aws_status = f'Connection failed: {exc}'
            st.session_state.pop('live_tables', None)
            st.session_state.pop('live_columns', None)
    connection_state = st.session_state.get('aws_connected')
    dot_class = 'aws-dot-green' if connection_state is True else 'aws-dot-red' if connection_state is False else 'aws-dot-gray'
    status_text = st.session_state.get('aws_status', 'Not checked')
    import html
    st.markdown(f'<div class="aws-connection"><span class="aws-dot {dot_class}"></span><span>{html.escape(status_text)}</span></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('### 🗄️ Database SQL Script')
    source = st.radio(
        'Database Source',
        ['Built-in Medical Database', 'Upload SQL Script'],
        index=0 if st.session_state.get('database_source', 'medical') == 'medical' else 1,
        key='database_source_choice',
    )
    if source == 'Upload SQL Script':
        st.caption('Upload a UTF-8 DuckDB .sql script containing CREATE TABLE and INSERT INTO ... VALUES. Maximum 1 MB.')
        script_file = st.file_uploader('Choose SQL Script', type=['sql'], key='database_sql_upload')
        if script_file is not None and st.button('Load Database', use_container_width=True, type='primary'):
            try:
                with st.spinner('Validating SQL and building an isolated database…'):
                    new_con = load_sql_script(script_file.getvalue())
                    switch_database(st, new_con, 'uploaded')
                st.success('Uploaded database is active. Schema and tables have refreshed.')
                st.rerun()
            except Exception as exc:
                st.error(f'Database upload rejected: {exc}')
        if st.session_state.get('database_source') == 'uploaded':
            st.success('Uploaded SQL database is active')
    elif st.session_state.get('database_source') != 'medical':
        try:
            new_con = load_builtin_medical()
            switch_database(st, new_con, 'medical')
            st.rerun()
        except Exception as exc:
            st.error(f'Could not restore medical database: {exc}')
    st.divider()
    st.markdown('### Database Explorer')
    choices = []
    if connection_state is True:
        if st.button('Refresh database schema', use_container_width=True):
            try:
                st.session_state.live_tables = database_tables()
                st.session_state.live_columns = database_schema()
            except Exception as exc:
                st.session_state.aws_connected = False
                st.session_state.aws_status = f'Connection failed: {exc}'
                st.session_state.pop('live_tables', None)
                st.session_state.pop('live_columns', None)
                st.error(f'Schema refresh failed: {exc}')
        all_tables = st.session_state.get('live_tables', [])
        tables = [t for t in all_tables if t.get('table_schema', '').lower() not in ('pg_internal', 'pg_catalog', 'information_schema')]
        choices = [f"{x['table_schema']}.{x['table_name']}" for x in tables]
        if choices:
            selected = st.selectbox('Table', choices, index=None, placeholder='Choose a table', key=f"selected_explorer_table_{id(st.session_state.demo_db)}")
            if selected:
                schema_name, table_name = selected.split('.', 1)
                st.markdown(f'**Schema:** `{schema_name}` &nbsp; **Table:** `{table_name}`')
                columns = [x for x in st.session_state.get('live_columns', [])
                           if x.get('table_schema') == schema_name and x.get('table_name') == table_name]
                if columns:
                    schema_df = pd.DataFrame(columns)
                    schema_df = schema_df[[c for c in ('column_name', 'data_type') if c in schema_df.columns]]
                    schema_df = schema_df.rename(columns={'column_name': 'Column', 'data_type': 'Data type'})
                    st.dataframe(schema_df, use_container_width=True, hide_index=True, height=min(480, 38 * (len(schema_df) + 1)))
                else:
                    st.caption('No column metadata returned for this table.')
        else:
            st.caption('No tables found in the current database.')
    else:
        st.info('Connecting to the medical demo database...')
    st.caption('Demo: DuckDB + simulated S3 · real Amazon Bedrock')

# ==================================================
# AWS CONNECTION GATE
# ==================================================

if st.session_state.get("aws_connected") is not True:
    st.caption("🔒 Connect to the demo environment first.")
    st.stop()

redshift_tab, csv_tab = st.tabs(['🗄️ Medical SQL AI Agent', '📄 CSV AI Workspace'])
with redshift_tab:
    st.markdown('<div class="aws-section">SQL database conversation</div>', unsafe_allow_html=True)
     # Short reminder (always visible)
    st.caption(
        "💡 Tip: Run SELECT before UPDATE/DELETE. "
        "For exact matching, use 'exact' and provide the complete column value."
    )

    # Example questions apply to the built-in medical demo schema only.
    if st.session_state.get('database_source', 'medical') == 'medical':
        with st.expander("🧪 Sample SQL questions — Medical demo database", expanded=False):
            st.warning(
                "These sample questions are designed for the current built-in medical database only. "
                "They may not work with an uploaded database because its tables, columns, and records can differ."
            )
            st.markdown("""
            **Read / SELECT**
            - Show all appointments for Ella Kim, including appointment IDs, dates, and times.
            - Show all patients with appointments with Dr. James Anderson.
            - List medications with their categories, departments, and prices.
            - Show medications costing more than $10.

            **UPDATE — run the corresponding SELECT first**
            - First: Show medication ID, name, and price for Metformin.  
              Then: Change the price of the selected Metformin medication to $9.50.
            - First: Show Ella Kim's appointments with appointment IDs, dates, and times.  
              Then: Change the selected appointment dated 2026-05-31 to 2026-04-12.
            - First: Show the medication ID, name, and category for Cetirizine.  
              Then: Change the category of the selected medication to Allergy.

            **DELETE — run the corresponding SELECT first and it will reject to delete rows which are connected to another table through foreign key.**
            - First: Show appointments with status Cancelled, including appointment IDs.  
              Then: Delete only the previously selected cancelled appointments from the appointments table that have no dependent records in other tables. Do not delete or modify any records in prescriptions, diagnoses, or other tables. Show the eligible appointment IDs and generated SQL, and request my approval before executing.

            ⚠️ **Writes require approval.** Review the generated SQL and affected records
            carefully before approving. Deletes may fail when related diagnosis or
            prescription records exist; do not bypass foreign-key protection.
            """)

    render_chat('redshift_messages')
    if st.session_state.redshift_approval:
        st.warning('Pending SQL modification — approve or reject before sending another question.')
        yes, no = st.columns(2)
        for col, approved, label in ((yes, True, 'Approve SQL'), (no, False, 'Reject SQL')):
            with col:
                if st.button(label, use_container_width=True):
                    try:
                        from agent import resume_agent
                        with st.spinner('Processing approval…'):
                            result = resume_agent(redshift_graph(), approved, st.session_state.redshift_thread)
                        display_redshift_result(result)
                    except Exception as exc:
                        # Do not leave a failed approval visible or replay its checkpoint.
                        st.session_state.redshift_approval = None
                        st.session_state.redshift_thread = uuid4().hex
                        add_message('redshift_messages', 'assistant',
                                    f'Operation failed or its outcome is uncertain: {exc}. '
                                    'Check the database before trying again. Start with a new SELECT.')
                    st.rerun()
    question = st.chat_input('Ask your database a question…', key='redshift_chat', disabled=bool(st.session_state.redshift_approval))
    if question:
        add_message('redshift_messages', 'user', question)
        try:
            from agent import run_agent
            with st.spinner('Querying Redshift AI agent…'):
                result = run_agent(redshift_graph(), question, st.session_state.redshift_thread)
            display_redshift_result(result)
        except Exception as exc:
            add_message('redshift_messages', 'assistant', f'Error: {exc}')
        st.rerun()

with csv_tab:
    if st.session_state.pop("csv_accept_notice", False):
       st.toast(
           "✅ SQL changes accepted! Please check the Working Dataset editor "
           "to review the updated values.",
            icon="✅",
            duration=8
        )
    if st.session_state.pop('csv_review_notice', False):
        st.toast('Changes applied! Scroll down to review the SQL and Accept or Reject.', icon='👇', duration=5)

    file = st.file_uploader('Upload local CSV to workspace', type=['csv'], key='csv_file')
        # CSV Workspace guidance
    st.info(
        "📘 **CSV Workspace Guide:** Upload a CSV file and prepare it for an existing "
        "Redshift table using the editable Working Dataset. Modify values, column names, "
        "or rows manually or through AI-powered bulk transformations.\n\n"
        "Review and save your changes before uploading. The dataset must match "
        "the destination table's schema to proceed."
    )
    if file is not None:
        raw = file.getvalue()
        token = hashlib.sha256(raw).hexdigest()
        if token != st.session_state.get('csv_file_token'):
            try:
                from csv_workspace import read_csv, CSVWorkspace
                from io import BytesIO
                st.session_state.csv_workspace = CSVWorkspace(read_csv(BytesIO(raw)))
                st.session_state.csv_file_token = token
                st.session_state.csv_editor_version += 1
                st.session_state.csv_messages = []
                st.session_state.csv_save_message = False
                st.session_state.csv_review_notice = False
                st.session_state.csv_upload_pending = False
                st.session_state.csv_upload_result = None
            except Exception as exc:
                st.error(f'CSV error: {exc}')
    ws = st.session_state.get('csv_workspace')
    if ws is None:
        st.info('Choose a CSV file to open the Original and Working datasets.')
    else:
        st.markdown('<div class="aws-section">Original dataset · read-only</div>', unsafe_allow_html=True)
        st.dataframe(ws.original_df, use_container_width=True, height=220)
        st.markdown('<div class="aws-section">Working dataset · editable</div>', unsafe_allow_html=True)
        
        # Row deletion guidance and editable working dataset.
        st.caption(
            "💡 To delete rows, select them using the boxes on the left "
            "and press Delete. Click Apply manual edits to keep the changes."
        )
        edited = st.data_editor(
            ws.working_df, use_container_width=True, height=270,
            num_rows='dynamic', disabled=ws.has_pending_change,
            key=f'working_editor_{st.session_state.csv_editor_version}'
        )

        # Compare the editor's CURRENT contents with the committed workspace.
        # This detects row additions/deletions and cell changes without refreshing.
        manual_edits_pending = not edited.equals(ws.working_df)
        if manual_edits_pending:
            st.warning(
                "⚠️ Unapplied manual edits detected. Click **Apply manual edits** "
                "before adding, renaming, or deleting columns. Column buttons "
                "are disabled until your edits are applied."
            )
        else:
            st.caption(
                "💡 After editing cells or adding/deleting rows, click "
                "**Apply manual edits** before using the column buttons."
            )

        # ==================================================
        # COMPACT COLUMN MANAGEMENT TOOLBAR
        # ==================================================

        @st.dialog("➕ Add Column")
        def add_column_dialog():
            new_name = st.text_input("New column name")

            if st.button("Add Column", type="primary"):
                new_name = new_name.strip()

                if not new_name:
                    st.error("Please enter a column name.")
                elif new_name in ws.working_df.columns:
                    st.error("Column already exists.")
                else:
                    ws.working_df[new_name] = None
                    ws.saved_df = None
                    ws.status = f"Column '{new_name}' added successfully."
                    st.session_state.csv_editor_version += 1
                    st.session_state.csv_upload_result = None
                    st.rerun()

        @st.dialog("✏️ Rename Column")
        def rename_column_dialog():
            old_name = st.selectbox(
                "Select column",
                ws.working_df.columns.tolist()
            )
            new_name = st.text_input("New column name")

            if st.button("Rename Column", type="primary"):
                new_name = new_name.strip()

                if not new_name:
                    st.error("Please enter a new column name.")
                elif new_name in ws.working_df.columns and new_name != old_name:
                    st.error("Column already exists.")
                else:
                    ws.working_df.rename(
                        columns={old_name: new_name},
                        inplace=True
                    )
                    ws.saved_df = None
                    ws.status = f"Column '{old_name}' renamed to '{new_name}'."
                    st.session_state.csv_editor_version += 1
                    st.session_state.csv_upload_result = None
                    st.rerun()

        @st.dialog("🗑️ Delete Column")
        def delete_column_dialog():
            column_name = st.selectbox(
                "Select column to delete",
                ws.working_df.columns.tolist()
            )

            if st.button("Delete Column", type="primary"):
                if len(ws.working_df.columns) <= 1:
                    st.error("Cannot delete the last remaining column.")
                else:
                    ws.working_df.drop(
                        columns=[column_name],
                        inplace=True
                    )
                    ws.saved_df = None
                    ws.status = f"Column '{column_name}' deleted successfully."
                    st.session_state.csv_editor_version += 1
                    st.session_state.csv_upload_result = None
                    st.rerun()

        # Toolbar buttons
        add_col, rename_col, delete_col = st.columns(3)

        with add_col:
            if st.button(
                "➕ Add Column",
                disabled=ws.has_pending_change or manual_edits_pending,
                use_container_width=True
            ):
                add_column_dialog()

        with rename_col:
            if st.button(
                "✏️ Rename Column",
                disabled=ws.has_pending_change or manual_edits_pending,
                use_container_width=True
            ):
                rename_column_dialog()

        with delete_col:
            if st.button(
                "🗑️ Delete Column",
                disabled=ws.has_pending_change or manual_edits_pending,
                use_container_width=True
            ):
                delete_column_dialog()

        # Main workspace actions are separate from bulk-change approval.
        cols = st.columns(3)
        with cols[0]:
            if st.button('Apply manual edits', disabled=ws.has_pending_change, use_container_width=True):
                ws.update_working_dataset(edited)
                st.session_state.csv_editor_version += 1
                st.session_state.csv_upload_result = None
                st.session_state.csv_save_message = False
                st.rerun()
        with cols[1]:
            if st.button('Reset all', use_container_width=True):
                ws.reset_all_changes()
                st.session_state.csv_editor_version += 1
                st.session_state.csv_upload_result = None
                st.session_state.csv_save_message = False
                st.rerun()
        with cols[2]:
            if st.button('Save after all edits', disabled=ws.has_pending_change, use_container_width=True):
                ws.update_working_dataset(edited)
                ws.save(use_original=False)
                st.session_state.csv_editor_version += 1
                st.session_state.csv_upload_result = None
                st.session_state.csv_save_message = True
                st.rerun()

        st.caption(ws.status)
        if st.session_state.csv_save_message:
            st.success('Excellent! Your edits have been saved. Now validate against a demo table and simulate S3 archival.Select the table and make sure your csv matched the table schema before uploading.')

        st.info(
              "💡 **Before uploading:** Click **Save after all edits** "
              "to save your finalized Working Dataset and enable the "
              "**Request S3 upload approval** button."
            )
        st.markdown('<div class="aws-section">Archive approved CSV in simulated S3</div>', unsafe_allow_html=True)
        try:
            available = choices  # Live metadata from successful Redshift connection only.
        except Exception:
            available = choices
        target = st.selectbox('Target active database table for schema matching', available, index=None, placeholder='Select destination schema.table', key=f"csv_target_{id(st.session_state.demo_db)}")
        if st.button('Request S3 upload approval', disabled=ws.saved_df is None or not target or st.session_state.csv_upload_pending):
            try:
                from csv_agent import start_csv_agent
                schema, table = target.split('.', 1)
                thread = uuid4().hex
                result = start_csv_agent(ws.saved_df, schema, table, thread)
                st.session_state.csv_upload_thread = thread
                st.session_state.csv_upload_pending = '__interrupt__' in result
                st.session_state.csv_upload_result = result if not st.session_state.csv_upload_pending else None
            except Exception as exc:
                st.error(str(exc))
            st.rerun()
        if st.session_state.csv_upload_pending:
            st.warning('Confirm upload of the saved CSV snapshot to S3. This is simulated and does not upload to AWS.')
            left, right = st.columns(2)
            for col, approved, label in ((left, True, 'Approve S3 upload'), (right, False, 'Cancel upload')):
                with col:
                    if st.button(label, use_container_width=True):
                        try:
                            from csv_agent import resume_csv_agent
                            with st.spinner('Checking schema and simulating S3 archive…'):
                                st.session_state.csv_upload_result = resume_csv_agent(approved, st.session_state.csv_upload_thread)
                        except Exception as exc:
                            error_text = str(exc)
                            st.session_state.csv_upload_result = {'status': f'Upload error: {error_text}'}
                        st.session_state.csv_upload_pending = False
                        st.rerun()
        if st.session_state.csv_upload_result:
            result = st.session_state.csv_upload_result
            st.info(result.get('status', 'Completed'))
            if result.get('suggestions'):
                with st.expander('Bedrock suggestions'):
                    for suggestion in result['suggestions']:
                        st.write('•', suggestion)
            if result.get('s3_key'):
                st.code(result['s3_key'])
            if result.get('upload_status') == 'duplicate':
                st.warning('Identical CSV already exists in simulated S3. Upload skipped.')
        st.markdown('<div class="aws-section">CSV transformation conversation</div>', unsafe_allow_html=True)
        render_chat('csv_messages')

        # Bulk-change decisions appear directly below the generated SQL in chat.
        if ws.has_pending_change:
            st.warning('👇 Review the SQL above and the updated Working dataset, then Accept or Reject below.')
            accept_col, reject_col = st.columns(2)
            with accept_col:
                if st.button('✓ Accept Changes', type='primary', use_container_width=True):
                    ws.accept_bulk_change()
                    st.session_state.csv_accept_notice = True
                    st.session_state.csv_editor_version += 1
                    st.rerun()
            with reject_col:
                if st.button('✕ Reject Changes', use_container_width=True):
                    ws.reject_bulk_change()
                    st.session_state.csv_editor_version += 1
                    st.rerun()
        instruction = st.chat_input('Describe a CSV transformation…', key='csv_chat', disabled=ws.has_pending_change)
        if instruction:
            add_message('csv_messages', 'user', instruction)
            try:
                with st.spinner('Applying transformation…'):
                    result = ws.apply_bulk_change(instruction)
                st.session_state.csv_editor_version += 1
                st.session_state.csv_upload_result = None
                st.session_state.csv_save_message = False
                st.session_state.csv_review_notice = True
                add_message('csv_messages', 'assistant', f"Applied transformation to Working ({result['rows_before']} → {result['rows_after']} rows). Review above and Accept or Reject.", result['sql'])
            # except Exception as exc:
            #     add_message('csv_messages', 'assistant', f'Transformation failed: {exc}')
            except Exception as exc:
                error_message = str(exc)

                if "Manage Working Dataset Columns" in error_message:
                    add_message(
                        'csv_messages',
                        'assistant',
                        f"💡 {error_message}"
                    )
                else:
                    add_message(
                        'csv_messages',
                        'assistant',
                        f"Transformation failed: {error_message}"
                    )
            st.rerun()