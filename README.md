# ✈️ AIR-treides — Enterprise Cargo Intelligence Platform

An autonomous, stateful **ReAct (Reasoning + Acting) Agent** system and analytics dashboard for real-time air cargo operational intelligence. Powered by **LangGraph**, **Groq (Qwen3-32b)**, **Supabase PostgreSQL**, and **Streamlit**.

---

## 📖 Table of Contents
1. [Core Features](#-core-features)
2. [System Architecture](#-system-architecture)
3. [Technology Stack](#-technology-stack)
4. [Project Directory Layout](#-project-directory-layout)
5. [Getting Started](#-getting-started)
6. [Testing Suite](#-testing-suite)
7. [Security & Guardrails](#-security--guardrails)

---

## 🌟 Core Features

### 📊 Real-Time Operations Cockpit
A tabbed, premium interface built with Emirates Airline-inspired burgundy and gold aesthetics:
*   **💬 AI Assistant**: Natural language conversation interface supporting database Text-to-SQL querying and web-news search synthesis.
*   **📊 Live Dashboard**: Interactive Plotly visualizations showing Month-over-Month tonnage trends, carrier market shares, import/export breakdowns, live database records, and files upload transaction audit tables.

### 🔒 Human-in-the-Loop (HITL) SQL Review
*   **SQL Interception Card**: When enabled, the workflow pauses execution before entering the tools node, prompting the operator to review, edit, or reject proposed SQL statements, preventing accidental queries or data errors.

### 📁 Hardened Data Ingestion
*   **Synonym Normalization**: Auto-aligns varied headers (e.g. `carrier` -> `airline`, `QTY` -> `weight_tons`) into a standardized database schema.
*   **Upsert Ingestion Mode**: Employs a PostgreSQL unique index and `INSERT ... ON CONFLICT` statement to update existing cargo weights rather than duplicate records on overlapping uploads.
*   **Upload Audit Log**: Automatically logs all ingestion events (filenames, row counts, timestamps) into the `upload_audit` table.

---

## 📐 System Architecture

The core runtime uses a cyclical **ReAct Agent** loop orchestrated by a compiled LangGraph workflow:

```mermaid
graph TD
    A[Upload CSV/Excel] -->|Normalizes headers| B[Sync to PostgreSQL via SQLAlchemy]
    C[Natural Language Query] -->|Input| D[ReAct Agent: StateGraph]
    D -->|SQL Query Tool| E[(Supabase DB)]
    D -->|Web News Tool| F[Tavily Search API]
    D -->|Wait for Approval| G{Human-in-the-Loop Toggle}
    G -->|Approve| H[Execute SQL]
    G -->|Reject| I[Cancel execution & inform agent]
    E & F -->|Yield Context| J[Markdown Report Writer]
    J -->|Real-time token stream| K[Streamlit UI Chat Tab]
```

---

## 🛠️ Technology Stack

*   **Frontend User Interface**: Streamlit (customized with CSS styling blocks)
*   **Orchestration Engine**: LangGraph (`StateGraph` with volatile `MemorySaver` checkpointing)
*   **Inference Model**: Groq Cloud API (`qwen/qwen3-32b`)
*   **Database Integration**: Supabase (PostgreSQL hosting) and SQLAlchemy Core (Connection management & bulk insert dialect statement builders)
*   **Search Engine**: Tavily Search API
*   **Data Analysis & Plots**: Pandas, OpenPyXL, and Plotly Express

---

## 📂 Project Directory Layout

```
air-cargo-intelligence/
│
├── app.py                     # Entrypoint & Tab Layout UI (Chat + Dashboard)
├── requirements.txt           # Python dependency specifications
├── .env                       # Environment variables (Credentials)
├── PROJECT_STUDY_GUIDE.md     # Technical Manual & Placement Interview Q&A
│
├── backend/
│   ├── __init__.py
│   ├── graph.py               # Workflow structure compilation & checkpoints
│   ├── ingestion.py           # Ingestion pipeline, Normalizer, and SQL Upsert
│   ├── nodes.py               # ReAct tools, System prompts, & state transitions
│   └── state.py               # LangGraph AgentState configurations
│
└── tests/
    ├── conftest.py            # Pytest workspace configurations
    ├── test_ingestion.py      # Normalization and header synonoym validations
    ├── test_router.py         # Graph execution route unit checks
    └── test_sql_safety.py     # SQL validation and regex injection tests
```

---

## 🚀 Getting Started

### 1. Clone & Set Up the Workspace
Navigate to the root directory:
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
SUPABASE_URL=postgresql://username:password@hostname:5432/postgres
```

### 3. Initialize Unique Deduplication Index (Optional)
To support **Upsert Mode** during data uploads, execute the following query inside the Supabase SQL editor to create a unique constraint key:
```sql
CREATE UNIQUE INDEX IF NOT EXISTS air_cargo_data_dedupe_idx
  ON air_cargo_data (month, airport_name, airline, commodity_type, export_import);
```

### 4. Run the Application
Start the Streamlit interface locally:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser!

---

## 🧪 Testing Suite

Automated unit tests ensure components remain robust against regressions:
```bash
pytest tests/ -v
```
Tests cover column synonym standardizations, agent workflow routing rules, and Text-to-SQL safety validations.

---

## 🛡️ Security & Guardrails

To prevent security threats (e.g. data tampering) from LLM-generated queries:
1. **SELECT Verification**: The input query must start case-insensitively with `SELECT`.
2. **Forbidden Keywords Validation**: Queries are scanned against a compile regex keyword pattern blocking modifying terms (e.g., `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`). Any violations are immediately rejected.
3. **Read-Only Database Roles**: Connections should ideally bind to a PostgreSQL role with read-only tables access permissions.
