# MedQuery AWS AI — Medical Data Engineering Agent

**An AI-powered data engineering application built with AWS Bedrock, LangGraph, LangChain, Streamlit, and DuckDB.**

MedQuery AI allows users to interact with structured databases using natural language, execute SQL queries, manage CSV datasets, and approve database modifications through a human-in-the-loop workflow.

## Key Features

- **Natural Language to SQL:** Converts user questions into SQL using Amazon Bedrock (Nova Micro).
- **Intelligent SQL Agent:** Executes SELECT queries automatically and requires human approval for UPDATE and DELETE operations.
- **Conversation Context:** Remembers previously selected records for follow-up database modifications.
- **CSV Workspace:** Supports CSV editing, column management, bulk transformations, and schema validation.
- **Database Explorer:** Dynamically displays tables and column information.
- **SQL Script Upload:** Allows users to load and query custom database schemas.
- **Simulated S3 Archival:** Demonstrates CSV archival with duplicate detection.

## Technology Stack

| Component---> Technology |
------------------------------------------------------
| Frontend ---> Streamlit                            |                           
| AI Model ---> Amazon Bedrock-claude-sonnet-5.5     |
| AI Orchestration ---> LangGraph, LangChain         |
| Database ---> DuckDB (simulated Amazon Redshift)   |
| Data Processing ---> Pandas, SQLGlot               |
| Cloud Storage ---> Simulated Amazon S3             |
| AWS Integration ---> Boto3                         |
------------------------------------------------------
## Architecture


                  Streamlit UI
                       |
              LangGraph AI Agent
                       |
              Amazon Bedrock LLM
                       |
                SQL Generation
                       |
               SQL Validation
                       |
              +--------+--------+
              |                 |
           SELECT          UPDATE/DELETE
              |                 |
        Auto Execute       Human Approval
              |                 |
              +--------+--------+
                       |
                     DuckDB
                       
CSV Upload --→ Edit/Transform --→ Schema Validation
           --→ Human Approval --→ Simulated S3 Archive


## Getting Started

1. Clone the repository.
2. Install dependencies:

   pip install -r requirements.txt

3. Configure AWS credentials with access to Amazon Bedrock in `us-east-1`.
4. Launch the application:

   streamlit run frontend.py
   

## Demo Environment

- Uses a sample medical database containing patients, appointments, prescriptions, and medications.
- Supports uploading custom SQL database scripts.
- Database changes are temporary and reset when a new demo session starts.
- Amazon Bedrock is live; Amazon Redshift and S3 are simulated.
- CSV archival does not insert records into DuckDB or upload files to AWS S3.

## Project Objective

Demonstrate an end-to-end AI-assisted data engineering workflow combining natural-language SQL generation, dynamic schema handling, data transformation, database operations, and human-in-the-loop approval.

