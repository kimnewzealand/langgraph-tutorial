from typing import Any, Dict, List, Optional, Annotated, TypedDict
import operator

from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from src.agent.utils import setup_llm, validate_startup_requirements

DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_THREAD_ID = "1"


class ContextSchema(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    """
    placeholder: Optional[str] = None


class AgentState(TypedDict):
    """State for the policy compliance agent."""

    messages: Annotated[list[AnyMessage], add_messages]

    documents_loaded: bool
    last_query: Optional[str]

    tools_used: Annotated[List[str], operator.add]
    tool_results: Dict[str, Any]

    current_step: Optional[str]          # Track current workflow step
    error_count: int                     # Track errors for recovery

    response_time: Optional[float]


llm = setup_llm(DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL)

if not validate_startup_requirements(DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL):
    raise ValueError("Failed to validate startup requirements")

# Node
def tool_calling_llm(state: AgentState) -> Dict[str, Any]:
    try:
        response=[llm.invoke(state["messages"])]
        return {"messages": response}
    except Exception as e:
        print(f"Failed to invoke model: {e}")
        return {}

graph = (
    StateGraph(AgentState, context_schema=ContextSchema)
    .add_edge(START, "tool_calling_llm")
    .add_node("tool_calling_llm", tool_calling_llm)
    .add_edge("tool_calling_llm", END)
    .compile(name="New Agent")
)
