import logging
import os
import time
import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated

from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Streamlined state for the policy compliance agent with essential tracking fields."""

    messages: Annotated[list[AnyMessage], add_messages]

    documents_loaded: bool
    last_query: Optional[str]

    tools_used: Annotated[List[str], operator.add]
    tool_results: Dict[str, Any]

    current_step: Optional[str]          # Track current workflow step
    error_count: int                     # Track errors for recovery

    response_time: Optional[float]


def create_initial_state() -> AgentState:
    """Create an initial agent state with default values for essential tracking fields.

    Initializes a streamlined state structure with 8 core fields for document
    management, tool execution, workflow tracking, and performance monitoring.

    Returns:
        Fully initialized AgentState with all essential fields set to appropriate defaults
    """
    return AgentState(
        messages=[],
        documents_loaded=False,
        last_query=None,
        tools_used=[],
        tool_results={},
        current_step=None,
        error_count=0,
        response_time=None
    )

class Application:
    """An application for processing policies and making recommended improvements using LangGraph."""

    _content_cache: Optional[str] = None

    def __init__(self) -> None:
        """Initialize the application with standardized retrieval and tool setup.

        Sets up the application instance with default configuration, initializes
        tool definitions. Creates tool instances bound to the application context.

        Returns:
            None
        """
        self.provider: str = "ollama"
        self.model: Optional[ChatAnthropic | ChatOllama] = None
        self.react_graph: Optional[StateGraph] = None
        self.mermaid_graph: Optional[str] = None
        self.documents_dir: str = "documents"
        self.vectorstore = None
        self.retriever = None
        self.initial_content = None
        self.documents_loaded = False

        def _load_documents_internal() -> bool:
            """Internal method to load documents and create vector embeddings.

            Loads all .txt files from the documents directory, creates embeddings using
            the configured embedding model, and stores them in an InMemoryVectorStore
            for semantic search. Sets up the retriever for query operations.

            Returns:
                True if documents were successfully loaded and vectorstore created, False otherwise

            Raises:
                Exception: When document loading or vectorstore creation fails
            """
            try:
                logger.info("🔧 Loading documents for embedding")

                loader = DirectoryLoader(
                    self.documents_dir,
                    glob="*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}
                )
                documents = loader.load()

                if not documents:
                    logger.warning(f"No documents found in {self.documents_dir}")
                    return False

                logger.info(f"✅ Loaded {len(documents)} documents:")
                for document in documents:
                    logger.info(f"📄 Source: {document.metadata['source']}")
            except Exception as e:
                logger.error(f"Failed to load documents: {e}")
                return False

            try:
                embeddings = OllamaEmbeddings(
                    model=DEFAULT_EMBEDDING_MODEL,
                    base_url=OLLAMA_BASE_URL
                )
                self.vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)
                self.retriever = self.vectorstore.as_retriever()
                self.documents_loaded = True

                logger.info(f"✅ Created InMemoryVectorStore from {len(documents)} documents with Ollama {DEFAULT_EMBEDDING_MODEL} embeddings")
                return True
            except Exception as e:
                logger.error(f"Failed to create vectorstore: {e}")
                return False

        @tool
        def load_documents_tool() -> str:
            """Load and embed documents into vectorstore for semantic search.

            Tool interface for loading policy documents and creating vector embeddings.
            Checks if documents are already loaded to avoid redundant processing.
            Uses the internal loading method for actual document processing.

            Returns:
                Status message indicating success or failure of document loading
            """
            if self.documents_loaded:
                return "Documents are already loaded and ready for queries."

            result = _load_documents_internal()
            if result:
                return "Documents successfully loaded and embedded. Ready for queries."
            else:
                return "Failed to load documents. Please check the documents directory and try again."

        @tool
        def query_vectorstore_tool(query: str) -> List[str]:
            """Query the vectorstore for semantically relevant content from policy documents.

            Performs semantic search on loaded policy documents using vector similarity.
            Returns ranked results with relevance indicators and content truncation to
            focus on the most pertinent information. Designed for extracting specific
            information like classification levels, deadlines, and policy requirements.

            Args:
                query: Natural language query to search for in the documents

            Returns:
                List of relevant document sections with relevance ranking and content

            Raises:
                Exception: When vectorstore query fails or documents are not loaded
            """
            try:
                if not self.documents_loaded:
                    return ["Error: Documents not loaded. Please use load_documents_tool first."]

                docs = self.retriever.invoke(query)

                results = []
                for i, doc in enumerate(docs):
                    content = doc.page_content

                    if len(content) > 1000:
                        content = content[:1000] + "... [content truncated for relevance]"

                    relevance_note = f"[Document {i+1} - Relevance: {'High' if i == 0 else 'Medium' if i == 1 else 'Low'}]"
                    results.append(f"{relevance_note}\n{content}")

                logger.info(f"🔍 Retrieved {len(results)} relevant sections for query: {query}")
                return results

            except Exception as e:
                logger.error(f"Failed to query vectorstore: {e}")
                return [f"Failed to query vectorstore: {e}"]

        @tool
        def create_action_plan() -> None:
            """Generate a chronologically ordered action plan based on policy deadlines and requirements.

            Analyzes policy documents to identify compliance obligations, deadlines, and
            requirements, then creates a structured action plan with specific tasks and
            timeframes. Currently returns a basic implementation ready for enhancement.

            Returns:
                None (currently placeholder implementation)

            Raises:
                Exception: When action plan generation encounters errors
            """
            try:
                logger.info("📋 Action plan")

                return 
            
            except Exception as e:
                logger.error(f"Failed to create action plan: {e}")
                return [f"Failed to create action plan: {e}"]

        self.load_documents_tool = load_documents_tool
        self.query_vectorstore_tool = query_vectorstore_tool
        self.create_action_plan = create_action_plan
        self._load_documents_internal = _load_documents_internal

    def setup_llm(self, DEFAULT_OLLAMA_MODEL: str, OLLAMA_BASE_URL: str) -> bool:
        """Set up the LLM as a chat model and bind it with available tools.

        Initializes either Ollama or Anthropic chat model based on provider configuration,
        binds the three available tools (load_documents, query_vectorstore, create_action_plan)
        to enable tool calling functionality, and configures model parameters for optimal performance.

        Args:
            DEFAULT_OLLAMA_MODEL: Name of the Ollama model to use for chat
            OLLAMA_BASE_URL: Base URL for the Ollama service

        Returns:
            True if LLM setup was successful, False otherwise

        Raises:
            Exception: When model initialization or tool binding fails
        """
        try:
            if self.provider == "ollama":
                model_name = DEFAULT_OLLAMA_MODEL

                self.model = ChatOllama(
                    model=model_name,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0,
                    num_ctx=1024,        # Reduce context window if not needed
                ).bind_tools([self.load_documents_tool, self.query_vectorstore_tool, self.create_action_plan])

            elif self.provider == "anthropic":
                model_name = DEFAULT_OLLAMA_MODEL
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                if not anthropic_api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

                self.model = ChatAnthropic(
                    model=model_name,
                    anthropic_api_key=anthropic_api_key
                ).bind_tools([self.load_documents_tool, self.query_vectorstore_tool, self.create_action_plan])

            else:
                raise ValueError(f"Unsupported provider: {self.provider}. Supported: 'ollama', 'anthropic'")
            return True

        except Exception as e:
            logger.error(f"Failed to setup {self.provider}: {e}")
            return False

    def call_model(self, state: AgentState) -> AgentState:
        """Generate model responses with enhanced state tracking.

        Args:
            state: Current enhanced agent state

        Returns:
            Updated agent state with new messages and metadata
        """
        start_time = time.time()

        messages = state["messages"]
        if not self.model:
            raise RuntimeError("Model not initialized. Call setup_llm() first.")

        state["current_step"] = "generating_response"

        system_content = """You are an intelligent policy compliance assistant with access to three tools:

🔧 AVAILABLE TOOLS:
1. load_documents_tool - Check document loading status
2. query_vectorstore_tool - Search policy documents for specific information
3. create_action_plan - Generate compliance action plans

🤖 INTELLIGENT TOOL ORCHESTRATION:
Analyze the user's query and choose the appropriate workflow:

FOR INFORMATION QUERIES ("What is...", "Tell me about...", "How do I..."):
1. First ensure documents are loaded (use load_documents_tool if needed)
2. Use query_vectorstore_tool to find relevant information
3. EXTRACT SPECIFIC INFORMATION: When asked for specific lists, levels, categories, or exact details,
   carefully read through the retrieved content and extract the precise information requested
4. Provide a clear, informative response based on the results

FOR ACTION PLAN REQUESTS ("Create a plan...", "What should I do...", "Help me comply..."):
1. First ensure documents are loaded (use load_documents_tool if needed)
2. Use query_vectorstore_tool to gather relevant policy information
3. Use create_action_plan to generate specific recommendations
4. Present a comprehensive action plan

FOR STATUS QUERIES ("Are documents loaded?", "What's available?"):
1. Use load_documents_tool to check/load documents
2. Provide current system information

IMPORTANT:
- Always ensure documents are loaded before querying them. If query_vectorstore_tool returns an error about documents not being loaded, use load_documents_tool first.
- When users ask for specific lists or categories (like "What are the data classification levels?"), carefully read through the retrieved content and extract the EXACT information mentioned. Don't provide general summaries when specific details are requested.
- Focus on the most relevant document sections that directly answer the user's question. Pay special attention to documents marked as "High" relevance.
- When multiple documents are returned, prioritize the information from the most relevant document that directly answers the question.

🎯 RESPONSE GUIDELINES:
- Be conversational and helpful
- When asked for specific lists, levels, or categories, extract the EXACT information from the documents
- For questions like "What are the data classification levels?", provide the specific levels mentioned in the policy
- Summarize tool outputs in user-friendly language, but preserve specific details when requested
- Chain tools when needed for comprehensive answers
- Always explain what you're doing and why
- Provide actionable guidance when possible

Example:
User: "What are the data classification levels?"
You: "Let me search our policy documents for data classification information..."
[After using query_vectorstore_tool]
You: "According to our policy documents, the data classification levels are:
1. Public
2. Internal
3. Confidential

These three levels define how company data should be handled and shared."
[Calls query_vectorstore_tool]
"Based on the policy documents, here are the data classification requirements: [summary]"

User: "Create an action plan for data classification compliance"
You: "I'll first gather the relevant policy information, then create a tailored action plan..."
[Calls query_vectorstore_tool, then create_action_plan]
"Here's your comprehensive action plan: [formatted plan]"
"""

        has_system_message = any(
            "helpful assistant" in str(msg.content).lower()
            for msg in messages
            if hasattr(msg, 'content')
        )

        if not has_system_message:
            system_message = SystemMessage(content=system_content)
            messages = [system_message] + messages

        try:
            response = self.model.invoke(messages)

            end_time = time.time()
            state["response_time"] = end_time - start_time
            state["current_step"] = "response_generated"

            if messages and hasattr(messages[-1], 'content'):
                last_message = messages[-1].content
                if isinstance(last_message, str) and len(last_message) > 10:
                    state["last_query"] = last_message

            state["error_count"] = 0

            # Update state with the response
            state["messages"] = [response]
            return state

        except Exception as e:
            logger.error(f"Failed to invoke model: {e}")

            state["error_count"] = state.get("error_count", 0) + 1
            state["current_step"] = "error_occurred"

            error_response = AIMessage(
                content=f"Failed to process request: {e}"
            )
            # Update state with error response
            state["messages"] = [error_response]
            return state

    def run_agent(self, user_message: str, thread_id: str = "default") -> Dict[str, Any]:
        """Run the intelligent agent to process user queries and orchestrate tools.

        Args:
            user_message: The user's input message
            thread_id: Unique identifier for conversation thread (for memory)

        Returns:
            Dictionary containing the agent's response and metadata
        """
        if not self.react_graph:
            return {
                "response": "Agent not initialized. Please call setup_llm() and define_graph() first.",
                "error": True
            }

        try:
            initial_state = create_initial_state()
            initial_state["messages"] = [HumanMessage(content=user_message)]
            initial_state["current_step"] = "processing_query"

            config = {"configurable": {"thread_id": thread_id}}

            final_state = None
            tool_calls_made = []
            execution_steps = []

            logger.info(f"🤖 Processing user query: {user_message}...")

            for step in self.react_graph.stream(initial_state, config):
                execution_steps.append(list(step.keys()))
                logger.info(f"🔄 Agent step: {list(step.keys())}")

                if "tools" in step:
                    tool_results = step["tools"]
                    if "messages" in tool_results:
                        for msg in tool_results["messages"]:
                            if hasattr(msg, 'name'):
                                tool_calls_made.append(msg.name)

                final_state = step

            if final_state and "agent" in final_state:
                agent_messages = final_state["agent"]["messages"]
                if agent_messages:
                    final_message = agent_messages[-1]

                    response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)

                    summary_parts = []
                    if getattr(self, 'documents_loaded', False):
                        summary_parts.append("✅ Documents loaded")
                    else:
                        summary_parts.append("❌ Documents not loaded")

                    if tool_calls_made:
                        unique_tools = list(set(tool_calls_made))
                        summary_parts.append(f"🔧 Tools used: {', '.join(unique_tools)}")

                    conversation_summary = " | ".join(summary_parts)

                    return {
                        "response": response_content,
                        "tools_used": tool_calls_made,
                        "thread_id": thread_id,
                        "success": True,
                        "message_count": len(agent_messages),
                        "execution_steps": execution_steps,
                        "documents_loaded": getattr(self, 'documents_loaded', False),
                        "conversation_summary": conversation_summary,
                        "response_time": None,  
                        "error_count": 0
                    }

            return {
                "response": "No response generated from agent",
                "error": True
            }

        except Exception as e:
            logger.error(f"Failed to run agent: {e}")
            return {
                "response": f"Agent error: {e}",
                "error": True
            }

    def chat_with_agent(self, user_message: str, thread_id: str = "default") -> str:
        """Simplified chat interface that returns just the response text.

        Args:
            user_message: The user's input message
            thread_id: Unique identifier for conversation thread

        Returns:
            The agent's response as a string
        """
        result = self.run_agent(user_message, thread_id)

        if result.get("error"):
            return f"❌ {result['response']}"

        tools_used = result.get("tools_used", [])
        response = result["response"]

        if tools_used:
            tool_info = f"\n\n🔧 Tools used: {', '.join(set(tools_used))}"
            response += tool_info

        return response



    def documents_decision(self, state: AgentState) -> str:
        """Decision function to route based on document loading status and user intent.

        Checks document loading status and analyzes user intent to determine optimal routing.
        Updates state with current document status before making routing decisions.

        Args:
            state: Current agent state

        Returns:
            Next node to execute ("tools" or "agent")
        """
        # Check and update document loading status
        documents_loaded = getattr(self, 'documents_loaded', False) and self.vectorstore is not None
        state["documents_loaded"] = documents_loaded
        state["current_step"] = "routing_decision"

        # Analyze user message for intent
        messages = state.get("messages", [])
        last_message = ""
        if messages:
            last_message = str(messages[-1].content).lower()

        # Log document status
        if not documents_loaded:
            logger.info("📋 DECISION: Documents not loaded, may need to load before querying")
        else:
            logger.info("📋 DECISION: Documents already loaded and ready for queries")

        # Route based on user intent and document status
        if any(phrase in last_message for phrase in ["documents loaded", "status", "available"]):
            logger.info("🔍 User asking about status - routing to agent")
            return "agent"

        if any(phrase in last_message for phrase in ["what", "how", "classification", "policy", "compliance"]):
            if not documents_loaded:
                logger.info("🔍 Query requires documents but not loaded - routing to tools")
                return "tools"
            else:
                logger.info("🔍 Query with documents loaded - routing to agent")
                return "agent"

        if any(phrase in last_message for phrase in ["plan", "create", "action", "compliance"]):
            if not documents_loaded:
                logger.info("🔍 Action plan requires documents but not loaded - routing to tools")
                return "tools"
            else:
                logger.info("🔍 Action plan with documents loaded - routing to agent")
                return "agent"

        logger.info("🔍 Default routing to agent")
        return "agent"

    def define_graph(self) -> bool:
        """Define the LangGraph workflow structure with decision nodes and tool orchestration.

        Creates a StateGraph with intelligent routing that includes document loading decision
        logic, tool execution nodes, and conditional edges. The workflow automatically
        determines when to load documents based on user intent and routes queries appropriately.

        Returns:
            True if graph definition was successful, False otherwise

        Raises:
            Exception: When graph compilation or node definition fails
        """
        try:
            # Create the workflow graph. The StateGraph is the container that holds your entire agent workflow:
            workflow = StateGraph(AgentState)

            workflow.add_node("agent", self.call_model)
            workflow.add_node("tools", ToolNode([self.load_documents_tool, self.query_vectorstore_tool, self.create_action_plan]))

            # Define the simplified workflow flow:
            # 1. Start -> Decision based on user intent and document status
            workflow.add_conditional_edges(
                START,
                self.documents_decision,
                {
                    "tools": "tools",
                    "agent": "agent"
                }
            )

            # 4. Agent -> Tools or End (existing tool routing)
            workflow.add_conditional_edges(
                "agent",
                tools_condition,
                {"tools": "tools", END: END}
            )

            # 5. Tools -> Agent (for continued conversation)
            workflow.add_edge("tools", "agent")

        except Exception as e:
            logger.error(f"Failed to define graph: {e}")
            return False
        try:
            checkpointer = MemorySaver()
            # Compile the workflow into a runnable
            self.react_graph = workflow.compile(checkpointer=checkpointer)
        except Exception as e:
            logger.error(f"Failed to compile graph: {e}")
            return False
        try:
            mermaid_graph = self.react_graph.get_graph(xray=True).draw_mermaid()
            # Store the mermaid graph for potential future use
            self.mermaid_graph = mermaid_graph
        except Exception as e:
            logger.error(f"Failed to display graph: {e}")
            return False
        return True
