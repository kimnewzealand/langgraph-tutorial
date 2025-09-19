from typing import Any, Dict, List, Optional, Annotated, TypedDict
import hashlib
import time

from langchain_core.messages import AnyMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.utils import setup_llm, validate_startup_requirements, load_documents, get_vectorstore

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
    ollama_validated: bool = False
    documents_loaded: bool = False
    tool_calls: List[str] = []

# Simple response cache for repeated queries
_response_cache = {}
_cache_ttl = 300  # 5 minutes

def get_cache_key(query: str) -> str:
    """Generate cache key for query."""
    return hashlib.md5(query.encode()).hexdigest()

def get_cached_response(query: str) -> Optional[str]:
    """Get cached response if available and not expired."""
    cache_key = get_cache_key(query)
    if cache_key in _response_cache:
        response, timestamp = _response_cache[cache_key]
        if time.time() - timestamp < _cache_ttl:
            return response
        else:
            del _response_cache[cache_key]
    return None

def cache_response(query: str, response: str):
    """Cache response with timestamp."""
    cache_key = get_cache_key(query)
    _response_cache[cache_key] = (response, time.time())

# Tools
def load_documents_tool(state: AgentState) -> Dict[str, Any]:
    """Load and embed documents into documents for semantic search.

    Tool interface for loading policy documents and creating vector embeddings.
    Checks if documents are already loaded to avoid redundant processing.

    Returns:
        Dictionary containing load status and any relevant metadata
    """
    load_documents()
    return {
        "messages": [AIMessage(content="I've loaded the policy documents and created embeddings for semantic search. You can now query the documents for specific information.")],
        "documents_loaded": True
    }

def query_documents_tool(query: str) -> Dict[str, Any]:
    """Query the documents for semantically relevant content from policy documents.

    Performs semantic search on loaded policy documents using vector similarity.
    Returns ranked results with relevance indicators and content truncation to
    focus on the most pertinent information. Designed for extracting specific
    information like classification levels, deadlines, and policy requirements.

    Args:
        query: Natural language query to search for in the documents

    Returns:
        Dictionary containing query results, relevance scores and metadata
    """
    vectorstore = get_vectorstore()
    if not vectorstore:
        return {"messages": [AIMessage(content="Vector store not initialized. Please load documents first.")]}

    # Use similarity_search_with_score for better performance insights
    try:
        results = vectorstore.similarity_search_with_score(query, k=3)
        if not results:
            return {"messages": [AIMessage(content="No relevant results found.")]}

        # Filter by relevance threshold for better quality
        filtered_results = [(doc, score) for doc, score in results if score < 0.8]  # Lower score = higher similarity

        if not filtered_results:
            return {"messages": [AIMessage(content="No sufficiently relevant results found.")]}

        response = "Here are the relevant policy sections:\n\n"
        for i, (doc, score) in enumerate(filtered_results, 1):
            # Truncate very long content for performance
            content = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
            response += f"{i}. {content}\n\n"

        return {"messages": [AIMessage(content=response)]}
    except Exception as e:
        # Fallback to basic search if similarity_search_with_score fails
        results = vectorstore.similarity_search(query, k=3)
        if not results:
            return {"messages": [AIMessage(content="No relevant results found.")]}

        response = "Here are the relevant policy sections:\n\n"
        for i, doc in enumerate(results, 1):
            content = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
            response += f"{i}. {content}\n\n"

        return {"messages": [AIMessage(content=response)]}

tools = [load_documents_tool, query_documents_tool]

llm = setup_llm(DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL)
if llm:
    llm = llm.bind_tools(tools)

# Cache system prompt to avoid repeated file I/O
_system_prompt_cache = None

def get_system_message() -> SystemMessage:
    """Get cached system message to avoid repeated file reads."""
    global _system_prompt_cache
    if _system_prompt_cache is None:
        with open("src/agent/system_prompt.txt", "r") as f:
            _system_prompt_cache = SystemMessage(content=f.read())
    return _system_prompt_cache

system_message = get_system_message()

# Node
def assistant(state: AgentState) -> Dict[str, Any]:
    """Main assistant node that processes messages and generates responses."""
    # Validate Ollama setup if not already done
    if not state.get("ollama_validated"):
        if not validate_startup_requirements(DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL):
            raise ValueError("Failed to validate startup requirements")
        state["ollama_validated"] = True

    messages = state["messages"]

    if not state.get("documents_loaded"):
        return load_documents_tool(state)

    # Check for cached response for simple queries
    if messages and len(messages) > 0:
        last_message = messages[-1]
        if hasattr(last_message, 'content') and isinstance(last_message.content, str):
            cached_response = get_cached_response(last_message.content)
            if cached_response:
                return {"messages": [AIMessage(content=cached_response)]}

    try:
        # Use streaming for better perceived performance (optional)
        response = llm.invoke([system_message] + messages)

        if isinstance(response, AIMessage):
            has_text = bool(getattr(response, "content", "") and str(response.content).strip())
            tool_calls = getattr(response, "tool_calls", None)
            if not has_text and tool_calls:
                info_msg = AIMessage(content="Working on that… running tools now.")
                return {"messages": [info_msg, response]}

        if not response or (isinstance(response, AIMessage) and not (getattr(response, "content", None) and str(response.content).strip())):
            return {"messages": [AIMessage(content="I'm ready to help you with your policy compliance questions.")]}

        # Cache successful responses
        if isinstance(response, AIMessage) and response.content and messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content') and isinstance(last_message.content, str):
                cache_response(last_message.content, str(response.content))

        return {"messages": [response]}
    except Exception as e:
        print(f"Failed to invoke model: {e}")
        return {"messages": [AIMessage(content=f"Error: {e}")]}

graph = (
    StateGraph(AgentState)
    .add_edge(START, "assistant")
    .add_node("assistant", assistant)
    .add_node("tools", ToolNode(tools))
    .add_conditional_edges(
        "assistant",
        tools_condition
    )
    .add_edge("tools", "assistant")
    .add_edge("assistant", END)
    .compile()
)
