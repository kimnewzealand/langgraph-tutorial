from typing import Any, Dict, List, Optional, Annotated, TypedDict

from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.utils import setup_llm, validate_startup_requirements, load_documents, get_vectorstore

DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
# DEFAULT_OLLAMA_MODEL = "llama3.2:1b" # Using a smaller model doesn't pass the evaluation
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = "http://localhost:11434"


class ContextSchema(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    """
    system_prompt: str

    @staticmethod
    def get_default_system_prompt() -> str:
        """Load system prompt from file."""
        with open("src/agent/system_prompt.txt", "r") as f:
            return f.read()


class AgentState(TypedDict):
    """State for the policy compliance agent."""

    messages: Annotated[list[AnyMessage], add_messages]
    ollama_validated: bool = False
    documents_loaded: bool = False
    system_initialised: bool = False
    tool_calls: List[str] = []

def create_initial_state() -> AgentState:
    """Create a fresh initial state for the agent."""
    return AgentState(
        messages=[],
        ollama_validated=False,
        documents_loaded=False,
        system_initialised=False,
        tool_calls=[]
    )

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

    try:
        results = vectorstore.similarity_search_with_score(query, k=3)
        if not results:
            return {"messages": [AIMessage(content="No relevant results found.")]}

        filtered_results = [(doc, score) for doc, score in results if score < 0.8]  # Lower score = higher similarity

        if not filtered_results:
            return {"messages": [AIMessage(content="No sufficiently relevant results found.")]}

        response = "Here are the relevant policy sections:\n\n"
        for i, (doc, score) in enumerate(filtered_results, 1):
            content = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
            response += f"{i}. {content}\n\n"

        return {"messages": [AIMessage(content=response)]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error querying documents: {str(e)}")]}

def create_plan_tool(query: str) -> Dict[str, Any]:
    """Create a plan based on the query and the documents.

    Generates a prioritized action plan by comparing policy documents with updates document.
    Uses LLM to analyze differences and create structured recommendations.

    Args:
        query: Natural language query to search for differences in the policy and update documents

    Returns:
        Dictionary containing the plan with priority, target dates and action steps
    """
    try:
        # Get vectorstore from global state (should be loaded by initialise node)
        vectorstore = get_vectorstore()
        if not vectorstore:
            # Try to load documents if not already loaded
            if not load_documents():
                return {"messages": [AIMessage(content="Failed to load documents. Please check document directory and try again.")]}
            vectorstore = get_vectorstore()
            if not vectorstore:
                return {"messages": [AIMessage(content="Documents not available. Please ensure documents are properly loaded.")]}

        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=5)
        if not docs:
            return {"messages": [AIMessage(content="No relevant policy documents found for your query.")]}

        # Create context from retrieved documents
        context = "\n\n".join([f"Document: {doc.page_content}" for doc in docs])

        # Create a simple, direct prompt
        plan_prompt = f"""Based on the following policy documents, create a detailed action plan for: "{query}"

POLICY DOCUMENTS:
{context}

Create a practical action plan with the following format:

**Action Plan for: {query}**

1. **Priority: [High/Medium/Low]**
   - Target Date: [Specific date within next 30 days]
   - Description: [Clear, actionable steps]
   - Dependencies: [Prerequisites or blockers]
   - Resources: [Required tools, approvals, or stakeholders]

2. **Priority: [High/Medium/Low]**
   - Target Date: [Specific date within next 30 days]
   - Description: [Clear, actionable steps]
   - Dependencies: [Prerequisites or blockers]
   - Resources: [Required tools, approvals, or stakeholders]

3. **Priority: [High/Medium/Low]**
   - Target Date: [Specific date within next 30 days]
   - Description: [Clear, actionable steps]
   - Dependencies: [Prerequisites or blockers]
   - Resources: [Required tools, approvals, or stakeholders]

Focus on specific, actionable steps based on the actual policy content provided above."""

        # Invoke LLM directly with the constructed prompt
        response = llm.invoke([HumanMessage(content=plan_prompt)])

        if not response or not response.content:
            return {"messages": [AIMessage(content="Failed to generate action plan.")]}

        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error creating plan: {str(e)}")]}

tools = [load_documents_tool, query_documents_tool, create_plan_tool]

llm = setup_llm(DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL)
if llm:
    llm = llm.bind_tools(tools)

# Nodes
def should_initialise(state: AgentState) -> str:
    """Conditional node to determine if initialisation is needed or if we can go directly to assistant.

    Returns:
        "initialise" if system needs initialisation
        "assistant" if system is already initialised
    """
    if state.get("system_initialised"):
        return "assistant"

    if state.get("documents_loaded") and state.get("ollama_validated"):
        state["system_initialised"] = True
        return "assistant"
    else:
        return "initialise"

def initialise(state: AgentState) -> Dict[str, Any]:
    """Node to initialise the agent by setting up all required components.

    This node runs automatically at startup to:
    - Validate Ollama startup requirements
    - Setup and configure the LLM
    - Load and embed policy documents
    - Set all initialisation flags
    - Prepare the system for user queries

    Only runs once per application lifecycle using state initialisation flag.

    Returns:
        Dictionary with initialisation status and updated state
    """
    if state.get("system_initialised"):
        return {
            "messages": [AIMessage(content="System is already initialised and ready.")],
            "documents_loaded": True,
            "ollama_validated": True,
            "system_initialised": True
        }

    if not validate_startup_requirements(DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL):
        raise ValueError("Failed to validate startup requirements - ensure Ollama is running and models are available")

    if llm is None:
        raise ValueError("Failed to setup LLM - check Ollama configuration")

    if not load_documents():
        raise ValueError("Failed to load documents - check documents directory and embedding model")

    return {
        "messages": [AIMessage(content="I've loaded the internal documents. You can now query the documents for specific information.")],
        "documents_loaded": True,
        "ollama_validated": True,
        "system_initialised": True
    }

def assistant(state: AgentState) -> Dict[str, Any]:
    """Main assistant node that processes messages and generates responses.

    This node is responsible for:
    - Processing user messages
    - Generating responses using the LLM
    - Handling tool calls and responses
    - Maintaining conversation history

    Returns:
        Dictionary with generated response and updated state
    """
    messages = state["messages"]

    system_prompt = ContextSchema.get_default_system_prompt()
    system_message = SystemMessage(content=system_prompt)

    try:
        response = llm.invoke([system_message] + messages)

        if isinstance(response, AIMessage):
            has_text = bool(getattr(response, "content", "") and str(response.content).strip())
            tool_calls = getattr(response, "tool_calls", None)
            if not has_text and tool_calls:
                info_msg = AIMessage(content="Working on that… running tools now.")
                return {"messages": [info_msg, response]}

        if not response or (isinstance(response, AIMessage) and not (getattr(response, "content", None) and str(response.content).strip())):
            return {"messages": [AIMessage(content="I'm ready to help you with your policy compliance questions.")]}

        return {"messages": [response]}
    except Exception as e:
        print(f"Failed to invoke model: {e}")
        return {"messages": [AIMessage(content=f"Error: {e}")]}

graph_builder = (
    StateGraph(AgentState)
    .add_conditional_edges(
        START,
        should_initialise,
        {
            "initialise": "initialise",
            "assistant": "assistant"
        }
    )
    .add_node("initialise", initialise)
    .add_edge("initialise", "assistant")
    .add_node("assistant", assistant)
    .add_node("tools", ToolNode(tools))
    .add_conditional_edges(
        "assistant",
        tools_condition
    )
    .add_edge("tools", "assistant")
    .add_edge("assistant", END)
)

graph = graph_builder.compile()
