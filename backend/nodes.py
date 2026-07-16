import json
import os
import re
import time

from groq import RateLimitError

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from urllib.parse import unquote, quote_plus

from backend.state import AgentState

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------------------------------------
# Database URL helper
# ---------------------------------------------------------------------------

def get_safe_db_url(url: str) -> str:
    """URL-encode the database password if it has special characters to make it safe for SQLAlchemy."""
    if not url:
        return url
    pattern = r"^([^:]+://)([^:]+):(.*)@([^@]+)$"
    match = re.match(pattern, url)
    if match:
        scheme, username, password, rest = match.groups()
        safe_password = quote_plus(unquote(password))
        return f"{scheme}{username}:{safe_password}@{rest}"
    return url

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = (
    "You are an expert air cargo intelligence analyst at AIR-treides.\n\n"
    "You have access to tools to query an air cargo database and search the web. "
    "Use them when the user asks about cargo data, airlines, airports, weights, "
    "trends, or market news.\n\n"
    "For general greetings or questions unrelated to air cargo data, respond "
    "directly without using any tools.\n\n"
    "Database schema:\n"
    "Table: air_cargo_data\n"
    "Columns: month (DATE), airport_name (TEXT), airline (TEXT), "
    "commodity_type (TEXT), export_import (TEXT), weight_tons (NUMERIC)\n\n"
    "When writing SQL, use only SELECT statements. Never modify data."
)

REPORT_SYSTEM_PROMPT = (
    "Based on the conversation below, write a helpful response.\n"
    "If the conversation involves data analysis with tool results, write a "
    "detailed analytical report in markdown with headings and bullet points.\n"
    "If it is a simple greeting or general question, just respond naturally "
    "and briefly. Do not over-elaborate simple answers."
)

# ---------------------------------------------------------------------------
# SQL safety
# ---------------------------------------------------------------------------

_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def is_safe_select(query: str) -> bool:
    """Validate that the query is a safe, read-only SELECT statement."""
    normalized = query.strip()
    if not normalized.upper().startswith("SELECT"):
        return False
    if _FORBIDDEN_SQL.search(normalized):
        return False
    return True


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="qwen/qwen3-32b",
        groq_api_key=GROQ_API_KEY,
        temperature=0,
    )


def _get_db() -> SQLDatabase:
    if not SUPABASE_URL:
        raise ValueError("SUPABASE_URL environment variable is not set.")
    safe_url = get_safe_db_url(SUPABASE_URL)
    return SQLDatabase.from_uri(safe_url, include_tables=["air_cargo_data"])


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def execute_sql_query(query: str) -> str:
    """Execute a read-only SQL SELECT query on the air_cargo_data table to retrieve cargo metrics, trends, rankings, and aggregations.

    Args:
        query: A PostgreSQL SELECT statement for the air_cargo_data table.
    """
    try:
        if not is_safe_select(query):
            return "Error: Only read-only SELECT queries are permitted."

        db = _get_db()
        results = db.run(query)
        return f"Query results:\n{results}"
    except Exception as exc:
        return f"SQL error: {exc}"


@tool
def search_web_news(query: str) -> str:
    """Search the web for the latest air cargo industry news, disruptions, and market context.

    Args:
        query: The search query about air cargo trends or news.
    """
    try:
        tavily = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
        results = tavily.invoke({"query": f"air cargo {query}"})

        if isinstance(results, list):
            summaries = []
            for r in results[:3]:
                title = r.get("title", "No title")
                content = r.get("content", "")[:300]
                url = r.get("url", "")
                summaries.append(f"- **{title}**: {content}... [Source]({url})")
            return "\n".join(summaries) if summaries else "No results found."
        return str(results)[:1000]
    except Exception as exc:
        return f"Web search error: {exc}"


# ---------------------------------------------------------------------------
# Message trimming (stay under Groq free-tier TPM limits)
# ---------------------------------------------------------------------------

MAX_HISTORY = 6


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


# Few-shot examples to guide the model on OpenAI-compatible JSON tool call formatting
FEW_SHOT_EXAMPLES = [
    HumanMessage(content="What is the total weight for Air India?"),
    AIMessage(
        content="",
        tool_calls=[{
            "name": "execute_sql_query",
            "args": {"query": "SELECT SUM(weight_tons) FROM air_cargo_data WHERE airline = 'Air India';"},
            "id": "fs_call_1",
            "type": "tool_call"
        }]
    ),
    ToolMessage(
        content="Query results:\n[(450.50,)]",
        name="execute_sql_query",
        tool_call_id="fs_call_1"
    ),
    AIMessage(content="The total cargo weight handled by Air India is 450.50 tons."),
]


def agent(state: AgentState) -> dict:
    """Main reasoning node — decides whether to use tools or respond directly."""
    messages = list(state.get("messages", []))

    # Trim old messages to stay under token limits
    if len(messages) > MAX_HISTORY:
        messages = messages[-MAX_HISTORY:]

    system = SystemMessage(content=AGENT_SYSTEM_PROMPT)
    llm = _get_llm().bind_tools(TOOLS)

    # Combine system prompt, few-shot examples, and actual message history
    payload = [system] + FEW_SHOT_EXAMPLES + list(messages)

    # Invoke with retry logic for tool_use_failed and rate limit errors
    for attempt in range(5):
        try:
            response = llm.invoke(payload)
            return {"messages": [response]}
        except RateLimitError as exc:
            wait = min(30, 2 ** attempt * 5)
            time.sleep(wait)
            continue
        except Exception as exc:
            err_str = str(exc)
            if "tool_use_failed" in err_str or "Failed to call a function" in err_str:
                time.sleep(0.5)
                continue
            if "rate_limit" in err_str.lower() or "429" in err_str:
                wait = min(30, 2 ** attempt * 5)
                time.sleep(wait)
                continue
            raise exc

    # Final fallback attempt
    response = llm.invoke(payload)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route: if the agent made tool calls → 'tools', otherwise → 'report_writer'."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "report_writer"


def run_tools(state: AgentState) -> dict:
    """Execute the tool calls requested by the agent."""
    return tool_node.invoke(state)


def report_writer(state: AgentState) -> dict:
    """Compile the final response from conversation context."""
    existing = state.get("final_answer", "")
    if existing:
        return {"final_answer": existing, "messages": [AIMessage(content=existing)]}

    # Build condensed conversation
    lines = []
    for msg in list(state.get("messages", []))[-MAX_HISTORY:]:
        role = getattr(msg, "type", "unknown")
        content = msg.content or ""
        if content:
            lines.append(f"[{role}] {content[:500]}")
    conversation = "\n".join(lines)

    llm = _get_llm()
    for attempt in range(5):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=REPORT_SYSTEM_PROMPT),
                    HumanMessage(content=f"Conversation:\n{conversation}"),
                ]
            )
            answer = response.content or ""
            return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
        except RateLimitError:
            wait = min(30, 2 ** attempt * 5)
            time.sleep(wait)
            continue
        except Exception as exc:
            err_str = str(exc)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                wait = min(30, 2 ** attempt * 5)
                time.sleep(wait)
                continue
            answer = f"Report generation error: {exc}"
            return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    answer = "Rate limit exceeded. Please wait a moment and try again."
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


def stream_report_writer(state_values: dict):
    """Generator that yields report chunks for real-time streaming in the UI."""
    lines = []
    for msg in list(state_values.get("messages", []))[-MAX_HISTORY:]:
        role = getattr(msg, "type", "unknown")
        content = msg.content or ""
        if content:
            lines.append(f"[{role}] {content[:500]}")
    conversation = "\n".join(lines)

    llm = _get_llm()
    for attempt in range(5):
        try:
            for chunk in llm.stream(
                [
                    SystemMessage(content=REPORT_SYSTEM_PROMPT),
                    HumanMessage(content=f"Conversation:\n{conversation}"),
                ]
            ):
                if chunk.content:
                    yield chunk.content
            return  # Successfully streamed
        except RateLimitError:
            wait = min(30, 2 ** attempt * 5)
            time.sleep(wait)
            continue
        except Exception as exc:
            err_str = str(exc)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                wait = min(30, 2 ** attempt * 5)
                time.sleep(wait)
                continue
            yield f"\n\nReport generation error: {exc}"
            return
    yield "\n\nRate limit exceeded. Please wait a moment and try again."


TOOLS = [execute_sql_query, search_web_news]
tool_node = ToolNode(TOOLS)
