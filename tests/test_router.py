from langchain_core.messages import AIMessage, HumanMessage

from backend.nodes import should_continue


def test_should_continue_to_tools_when_tool_calls_present():
    state = {
        "messages": [
            HumanMessage(content="Top airports by weight"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_sql_query",
                        "args": {"query": "SELECT 1"},
                        "id": "call_1",
                    }
                ],
            ),
        ]
    }
    assert should_continue(state) == "tools"


def test_should_continue_to_report_when_no_tool_calls():
    state = {
        "messages": [
            HumanMessage(content="Summarize findings"),
            AIMessage(content="Here is the analysis based on available data."),
        ]
    }
    assert should_continue(state) == "report_writer"


def test_should_continue_to_report_when_empty_tool_calls():
    state = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Ready to report.", tool_calls=[]),
        ]
    }
    assert should_continue(state) == "report_writer"
