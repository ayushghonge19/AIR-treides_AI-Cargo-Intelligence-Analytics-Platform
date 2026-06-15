import uuid
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus, unquote
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.graph import app
from backend.ingestion import process_and_upload_file
from backend.nodes import stream_report_writer, get_safe_db_url

# ---------------------------------------------------------------------------
# Status labels
# ---------------------------------------------------------------------------

NODE_LABELS = {
    "agent": "🧠 Analyzing your question...",
    "tools": "🔧 Executing tools...",
}

TOOL_LABELS = {
    "execute_sql_query": "📊 Queried the cargo database",
    "search_web_news": "🌐 Searched web for market context",
}

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None
    if "awaiting_approval" not in st.session_state:
        st.session_state.awaiting_approval = False
    if "pending_tool_call" not in st.session_state:
        st.session_state.pending_tool_call = None
    if "hitl_enabled" not in st.session_state:
        st.session_state.hitl_enabled = True
    if "upsert_mode" not in st.session_state:
        st.session_state.upsert_mode = False

def _graph_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}

# Helper to execute query to dataframe or value
def run_query_val(query: str):
    engine = create_engine(get_safe_db_url(os.getenv("SUPABASE_URL")))
    with engine.connect() as conn:
        res = conn.execute(text(query)).fetchone()
        return res[0] if res else 0

def run_query_df(query: str):
    engine = create_engine(get_safe_db_url(os.getenv("SUPABASE_URL")))
    return pd.read_sql(query, engine)

# ---------------------------------------------------------------------------
# Graph Runner Logic
# ---------------------------------------------------------------------------

def _run_graph():
    config = _graph_config()
    
    while True:
        snapshot = app.get_state(config)
        next_nodes = snapshot.next
        
        if not next_nodes:
            break
            
        if "tools" in next_nodes:
            messages = snapshot.values.get("messages", [])
            if not messages:
                break
            last_msg = messages[-1]
            tool_calls = getattr(last_msg, "tool_calls", [])
            
            sql_tool_call = None
            for tc in tool_calls:
                if tc.get("name") == "execute_sql_query":
                    sql_tool_call = tc
                    break
            
            # If HITL SQL Approval is enabled, pause and display approval UI
            if sql_tool_call and st.session_state.hitl_enabled:
                st.session_state.pending_tool_call = sql_tool_call
                st.session_state.awaiting_approval = True
                return
                
            # If HITL is not enabled, or it's a search tool, run automatically
            for event in app.stream(None, config, stream_mode="updates"):
                pass
            continue
            
        if "report_writer" in next_nodes:
            # Phase 2 — Stream/write final response
            messages = snapshot.values.get("messages", [])
            used_tools = any(isinstance(m, ToolMessage) for m in messages)
            
            if used_tools:
                final_answer = st.write_stream(stream_report_writer(snapshot.values))
                final_answer = final_answer or ""
            else:
                last_ai_content = ""
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                        last_ai_content = msg.content
                        break
                final_answer = last_ai_content
                st.markdown(final_answer)
                
            try:
                app.update_state(
                    config,
                    {
                        "final_answer": final_answer,
                        "messages": [AIMessage(content=final_answer)],
                    },
                )
                for _ in app.stream(None, config, stream_mode="updates"):
                    pass
            except Exception:
                pass
                
            if final_answer:
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            break

def _start_new_run(user_input: str):
    config = _graph_config()
    try:
        # Start graph execution and let it run until first interrupt
        for event in app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="updates",
        ):
            pass
        _run_graph()
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------

_init_session()

st.set_page_config(
    page_title="AIR-treides Cargo Intelligence",
    page_icon="✈️",
    layout="wide",
)

# Custom Premium Styling (Emirates Brand Aesthetics)
st.markdown(
    """
    <style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Apply Fonts and Backgrounds */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #FDFBF7 !important; /* Elegant Cream Background */
        color: #1E1E1E !important;
    }

    /* Force Dark Text Color globally on labels, paragraphs, spans, lists, table contents, and chat text */
    p, span, label, li, ul, ol, th, td,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stWidgetLabel"],
    .stWidgetLabel p,
    label p,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #1E1E1E !important;
    }

    /* Headings should be Burgundy */
    h1, h2, h3, h4, h5, h6,
    h1 span, h2 span, h3 span, h4 span, h5 span, h6 span,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #8D1B3D !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F5EFEB !important; /* Soft sand background */
        border-right: 1px solid #D1A153 !important; /* Gold border */
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #8D1B3D !important;
    }

    /* Sidebar divider */
    hr {
        border-color: #E2D7CC !important;
    }

    /* Header styling */
    [data-testid="stHeader"] {
        background-color: rgba(253, 251, 247, 0.9) !important;
        border-bottom: 2px solid #8D1B3D !important; /* Sleek Burgundy line */
    }

    /* Buttons - force white text */
    .stButton > button,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        background-color: #8D1B3D !important; /* Emirates Burgundy */
        color: #FFFFFF !important;
        border: 1px solid #8D1B3D !important;
        border-radius: 4px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(141, 27, 61, 0.1) !important;
    }

    .stButton > button:hover,
    .stButton > button:hover p,
    .stButton > button:hover span,
    .stButton > button:hover div {
        background-color: #FFFFFF !important;
        color: #8D1B3D !important;
        border: 1px solid #8D1B3D !important;
        box-shadow: 0 4px 8px rgba(141, 27, 61, 0.2) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Tab bar styling overrides */
    [data-baseweb="tab"] p, 
    [data-baseweb="tab"] span, 
    [data-baseweb="tab"] div {
        color: #666666 !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        transition: color 0.3s ease !important;
    }
    
    [data-baseweb="tab"][aria-selected="true"] p, 
    [data-baseweb="tab"][aria-selected="true"] span, 
    [data-baseweb="tab"][aria-selected="true"] div {
        color: #8D1B3D !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #C49A45 !important; /* Gold dashed border */
        border-radius: 6px !important;
        background-color: #FFFFFF !important;
        padding: 10px !important;
    }

    /* Status indicator widget */
    div[data-testid="stStatusWidget"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2D7CC !important;
        border-left: 4px solid #C49A45 !important; /* Gold left strip */
        border-radius: 4px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }

    /* Chat Message Styling */
    [data-testid="stChatMessage"] {
        border-radius: 8px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
        font-size: 15px !important;
    }

    /* Assistant Messages (Cream-gold tint) */
    [data-testid="stChatMessage"][data-testid="stChatMessageContent-assistant"], 
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #F8F5EE !important; 
        border-left: 4px solid #C49A45 !important; /* Gold Accent */
    }

    /* User Messages (Slight gray-sand tint) */
    [data-testid="stChatMessage"][data-testid="stChatMessageContent-user"],
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #FFFFFF !important;
        border-left: 4px solid #8D1B3D !important; /* Burgundy Accent */
    }

    /* Chat input bar styling */
    [data-testid="stChatInput"] {
        border-radius: 8px !important;
        border: 1px solid #D1A153 !important; /* Gold border */
        box-shadow: 0 4px 12px rgba(196, 154, 69, 0.1) !important;
        background-color: #FFFFFF !important;
    }
    
    textarea[data-testid="stChatInputTextArea"] {
        color: #1E1E1E !important;
    }

    /* Markdown tables / reports styling */
    table {
        border-collapse: collapse !important;
        width: 100% !important;
        margin: 16px 0 !important;
        font-size: 14px !important;
    }

    th {
        background-color: #8D1B3D !important; /* Burgundy header */
        color: #FFFFFF !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
        text-align: left !important;
        border: 1px solid #E2D7CC !important;
    }

    td {
        padding: 10px 16px !important;
        border: 1px solid #E2D7CC !important;
        background-color: #FFFFFF !important;
    }

    tr:nth-child(even) td {
        background-color: #FDFBF7 !important;
    }

    /* Success, info, warning messages */
    .stAlert {
        border-radius: 6px !important;
        border: 1px solid rgba(196, 154, 69, 0.2) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div style="display: flex; align-items: center; margin-bottom: 25px; border-bottom: 2px solid #8D1B3D; padding-bottom: 20px;">
        <div style="background-color: #8D1B3D; color: white; padding: 12px 18px; border-radius: 4px; font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 26px; margin-right: 18px; letter-spacing: 1px; box-shadow: 0 4px 8px rgba(141, 27, 61, 0.2);">
            AIR
        </div>
        <div>
            <h1 style="margin: 0; color: #8D1B3D; font-family: 'Outfit', sans-serif; font-size: 34px; font-weight: 800; letter-spacing: 0.5px; line-height: 1.2;">
                AIR-treides
            </h1>
            <p style="margin: 0; font-family: 'Inter', sans-serif; font-size: 14px; color: #666; font-weight: 500;">
                Enterprise Cargo Intelligence Engine • Powered by LangGraph & Groq
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Session Settings")
    
    # HITL and Upsert Controls
    st.session_state.hitl_enabled = st.toggle(
        "Enable SQL Review (HITL)",
        value=st.session_state.hitl_enabled,
        help="Pause the agent execution before running any SQL query to review/approve it."
    )
    
    st.session_state.upsert_mode = st.toggle(
        "Upsert Ingestion Mode",
        value=st.session_state.upsert_mode,
        help="Update existing rows based on key columns instead of duplicating."
    )

    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.awaiting_approval = False
        st.session_state.pending_tool_call = None
        st.rerun()

    st.divider()
    st.header("📁 Data Ingestion Pipeline")
    uploaded_file = st.file_uploader(
        "Upload air cargo CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help=(
            "Expected columns: month, airport_name, airline, "
            "commodity_type, export_import, weight_tons"
        ),
    )

    if uploaded_file is not None:
        upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.last_uploaded != upload_key:
            try:
                with st.status("Processing upload...", expanded=True) as upl:
                    upl.write("📋 Validating schema...")
                    upl.write("💾 Syncing to database...")
                    result = process_and_upload_file(uploaded_file, upsert=st.session_state.upsert_mode)

                st.session_state.last_uploaded = upload_key

                if result["success"]:
                    st.success(
                        f"✅ Inserted/Updated {result['rows_inserted']} rows "
                        f"into air_cargo_data."
                    )
                    st.toast(
                        f"Ingested {result['rows_inserted']} rows!",
                        icon="✅",
                    )
                else:
                    st.error(result["error"])
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()
    st.header("About")
    st.markdown(
        """
    **AIR-treides Engine**
    - 🧠 Autonomous ReAct agent
    - 📊 SQL analytics on cargo data
    - 🌐 Real-time web news search
    - 💬 Multi-turn conversation memory
    """
    )
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")

# ---------------------------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------------------------

tab_chat, tab_dashboard = st.tabs(["💬 AI Assistant", "📊 Real-Time Analytics Dashboard"])

# ---------------------------------------------------------------------------
# Tab 1: AI Assistant
# ---------------------------------------------------------------------------

with tab_chat:
    # Render chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Render pending SQL approval if active
    if st.session_state.awaiting_approval and st.session_state.pending_tool_call:
        with st.chat_message("assistant"):
            st.warning("⚠️ **SQL Query Approval Required**")
            st.markdown("The agent has proposed the following database query:")
            st.code(st.session_state.pending_tool_call["args"]["query"], language="sql")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve & Run", key="btn_approve", use_container_width=True):
                    sql_call = st.session_state.pending_tool_call
                    st.session_state.awaiting_approval = False
                    st.session_state.pending_tool_call = None
                    
                    config = _graph_config()
                    with st.status("🔧 Executing approved database query...", expanded=True) as status:
                        for event in app.stream(None, config, stream_mode="updates"):
                            pass
                        status.update(label="✅ Query executed successfully", state="complete")
                    
                    # Continue workflow
                    _run_graph()
                    st.rerun()
                    
            with col2:
                if st.button("Reject", key="btn_reject", use_container_width=True):
                    sql_call = st.session_state.pending_tool_call
                    st.session_state.awaiting_approval = False
                    st.session_state.pending_tool_call = None
                    
                    config = _graph_config()
                    app.update_state(
                        config,
                        {
                            "messages": [
                                ToolMessage(
                                    content="Error: The query was rejected by the user.",
                                    name=sql_call["name"],
                                    tool_call_id=sql_call["id"],
                                )
                            ]
                        },
                        as_node="tools",
                    )
                    
                    # Continue workflow
                    _run_graph()
                    st.rerun()

    # Chat Input
    if user_input := st.chat_input("Ask about air cargo data, trends, or news..."):
        if st.session_state.awaiting_approval:
            st.error("Please approve or reject the pending SQL query before sending a new message.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()

# Run the pipeline if a new message was added and not yet answered
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_input = st.session_state.messages[-1]["content"]
    with tab_chat:
        with st.chat_message("assistant"):
            _start_new_run(user_input)
            st.rerun()

# ---------------------------------------------------------------------------
# Tab 2: Live Operations Dashboard
# ---------------------------------------------------------------------------

with tab_dashboard:
    st.subheader("📊 Live Operations Cockpit")
    
    # Check if there is data in air_cargo_data
    try:
        total_rows = run_query_val("SELECT COUNT(*) FROM air_cargo_data")
    except Exception as exc:
        st.error(f"Failed to connect to database: {exc}")
        total_rows = 0
        
    if total_rows == 0:
        st.warning("⚠️ **No Data Ingested Yet**")
        st.markdown("Use the Data Ingestion Pipeline in the sidebar to upload cargo logs (CSV or Excel) and populate this dashboard.")
    else:
        # Fetch metrics
        total_weight = run_query_val("SELECT SUM(weight_tons) FROM air_cargo_data")
        total_airlines = run_query_val("SELECT COUNT(DISTINCT airline) FROM air_cargo_data")
        total_airports = run_query_val("SELECT COUNT(DISTINCT airport_name) FROM air_cargo_data")
        
        # KPI Cards Row
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(
                f"""
                <div style="background-color: white; border: 1px solid #E2D7CC; border-left: 5px solid #8D1B3D; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                    <p style="margin: 0; font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase;">Total Weight</p>
                    <h3 style="margin: 5px 0 0 0; color: #8D1B3D; font-size: 26px; font-weight: 800;">{total_weight:,.2f} <span style="font-size: 14px; font-weight: 500; color: #666;">Tons</span></h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi_col2:
            st.markdown(
                f"""
                <div style="background-color: white; border: 1px solid #E2D7CC; border-left: 5px solid #C49A45; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                    <p style="margin: 0; font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase;">Total Records</p>
                    <h3 style="margin: 5px 0 0 0; color: #C49A45; font-size: 26px; font-weight: 800;">{total_rows:,} <span style="font-size: 14px; font-weight: 500; color: #666;">Rows</span></h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi_col3:
            st.markdown(
                f"""
                <div style="background-color: white; border: 1px solid #E2D7CC; border-left: 5px solid #8D1B3D; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                    <p style="margin: 0; font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase;">Airlines</p>
                    <h3 style="margin: 5px 0 0 0; color: #8D1B3D; font-size: 26px; font-weight: 800;">{total_airlines:,} <span style="font-size: 14px; font-weight: 500; color: #666;">Carriers</span></h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi_col4:
            st.markdown(
                f"""
                <div style="background-color: white; border: 1px solid #E2D7CC; border-left: 5px solid #C49A45; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                    <p style="margin: 0; font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase;">Airports</p>
                    <h3 style="margin: 5px 0 0 0; color: #C49A45; font-size: 26px; font-weight: 800;">{total_airports:,} <span style="font-size: 14px; font-weight: 500; color: #666;">Hubs</span></h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.write("")
        st.write("")
        
        # Charts Row
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            df_mom = run_query_df("SELECT month, SUM(weight_tons) as weight FROM air_cargo_data GROUP BY month ORDER BY month")
            if not df_mom.empty:
                df_mom["month"] = df_mom["month"].astype(str)
                fig_line = px.line(
                    df_mom, x="month", y="weight", 
                    labels={"month": "Month", "weight": "Weight (Tons)"},
                    title="📈 MoM Cargo Tonnage Trends"
                )
                fig_line.update_traces(line_color="#8D1B3D", line_width=3)
                fig_line.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Outfit",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_line, use_container_width=True)
                
        with chart_col2:
            df_flow = run_query_df("SELECT export_import, SUM(weight_tons) as weight FROM air_cargo_data GROUP BY export_import")
            if not df_flow.empty:
                fig_pie = px.pie(
                    df_flow, values="weight", names="export_import",
                    hole=0.4,
                    title="🔄 Flow Distribution (Export vs Import)",
                    color_discrete_sequence=["#8D1B3D", "#C49A45"]
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_family="Outfit",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
        # Row 2 charts
        chart_col3, chart_col4 = st.columns([2, 1])
        
        with chart_col3:
            df_airlines = run_query_df("SELECT airline, SUM(weight_tons) as weight FROM air_cargo_data GROUP BY airline ORDER BY weight DESC LIMIT 10")
            if not df_airlines.empty:
                fig_bar = px.bar(
                    df_airlines, x="weight", y="airline",
                    orientation="h",
                    labels={"weight": "Weight (Tons)", "airline": "Airline"},
                    title="✈️ Top 10 Airlines by Cargo Tonnage"
                )
                fig_bar.update_traces(marker_color="#C49A45")
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Outfit",
                    yaxis={'categoryorder':'total ascending'},
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
        with chart_col4:
            st.markdown("<p style='font-weight: 700; color: #8D1B3D; margin-top: 15px;'>📋 Ingestion Audit Log</p>", unsafe_allow_html=True)
            try:
                df_audit = run_query_df("SELECT filename, rows_inserted, upload_timestamp FROM upload_audit ORDER BY upload_timestamp DESC LIMIT 5")
                if not df_audit.empty:
                    df_audit.columns = ["File Name", "Rows", "Timestamp"]
                    st.dataframe(df_audit, use_container_width=True, hide_index=True)
                else:
                    st.info("No files uploaded yet.")
            except Exception:
                st.info("Audit log is empty.")
                
        # Data preview table
        st.markdown("<p style='font-weight: 700; color: #8D1B3D; margin-top: 20px;'>🔍 Live Data Preview (Latest 10 records)</p>", unsafe_allow_html=True)
        df_preview = run_query_df("SELECT month, airport_name, airline, commodity_type, export_import, weight_tons FROM air_cargo_data ORDER BY id DESC LIMIT 10")
        if not df_preview.empty:
            df_preview.columns = ["Month", "Airport Name", "Airline", "Commodity", "Flow", "Weight (Tons)"]
            st.dataframe(df_preview, use_container_width=True, hide_index=True)
