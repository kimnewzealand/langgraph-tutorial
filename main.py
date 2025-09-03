import logging
import os
import sys
from typing import TypedDict, List, Dict, Any, Optional, Annotated

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
load_dotenv()

# Constants
DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"
DEFAULT_OLLAMA_MODEL = "qwen2:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_THREAD_ID = "1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    # The document provided
    graph_state: Optional[str]  # Contains the graph state
    input_file: Optional[str]  # Contains file path (PDF/PNG)

    # The name of the document the policy is from
    document: str
    # The policy/standard being processed eg LLM Usage Policy
    policy: Dict[str, Any]  
    # Category of the policy
    policy_category: Optional[str]
    # Status of the policy ie draft, final
    policy_status: Optional[str]
    # Policy version number
    policy_version: Optional[str]
    # Track conversation with LLM for analysis
    messages: Annotated[list[AnyMessage], add_messages]

class Application:
    """An application for processing policies using LangGraph."""

    def __init__(self) -> None:
        """Initialize the application."""
        self.model: Optional[ChatAnthropic | ChatOllama] = None
        self.mermaid_graph: Optional[str] = None
        self.react_graph: Optional[StateGraph] = None
        self.workflow: Optional[StateGraph] = None


    @tool
    def retrieve_content(query: str) -> List[str]:
        """Function to retrieve usable documents for AI assistant"""
        try:
            file_path = "documents/sample_data.txt"
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return [content]
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []
  
    def setup_llm(self, provider: str = "ollama", model_name: Optional[str] = None) -> bool:
        """Setup the Language Model with multiple provider options.

        Args:
            provider: LLM provider ("ollama" or "anthropic")
            model_name: Optional specific model name

        Returns:
            True if successful, False otherwise
        """
        try:
            if provider == "ollama":
                model_name = model_name or DEFAULT_OLLAMA_MODEL
                self.model = ChatOllama(
                    model=model_name,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0
                ).bind_tools([self.retrieve_content])

            elif provider == "anthropic":
                model_name = model_name or DEFAULT_ANTHROPIC_MODEL
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                if not anthropic_api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

                self.model = ChatAnthropic(
                    model=model_name,
                    anthropic_api_key=anthropic_api_key
                ).bind_tools([self.retrieve_content])

            else:
                raise ValueError(f"Unsupported provider: {provider}. Supported: 'ollama', 'anthropic'")

            logger.info(f"✅ {provider.title()} setup successful with model: {model_name}")
            return True

        except ImportError as e:
            logger.error(f"Missing package for {provider}: {e}")
            logger.info(f"Install with: pip install langchain-{provider}")
            return False
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting up {provider}: {e}")
            return False

    def call_model(self, state: MessagesState) -> Dict[str, List[Any]]:
        """Generate model responses.

        Args:
            state: Current state containing messages

        Returns:
            Dictionary with updated messages
        """
        messages = state["messages"]
        if not self.model:
            raise RuntimeError("Model not initialized. Call setup_llm() first.")

        # Add system message for better behavior if not already present
        system_content = """You are a helpful assistant. When using tools to retrieve information:
1. Only extract the specific information requested
2. Provide concise, direct answers (1-2 sentences maximum)
3. Do not repeat or include the entire document content in your response
4. Focus on answering the exact question asked

Example: If asked "What are the levels of data classification?" respond with "There are three levels of data classification: Public, Internal, and Confidential." Do not include the full document text."""

        # Check if system message already exists
        has_system_message = any(
            "helpful assistant" in str(msg.content).lower()
            for msg in messages
            if hasattr(msg, 'content')
        )

        # Insert system message at the beginning if not present
        if not has_system_message:
            system_message = SystemMessage(content=system_content)
            messages = [system_message] + messages

        response = self.model.invoke(messages)
        return {"messages": [response]}
    
    def define_graph(self):
        # Define the workflow graph
        try:
            workflow = StateGraph(AgentState)
            # Define nodes: these do the work
            workflow.add_node("tools", ToolNode([self.retrieve_content]))
            workflow.add_node("agent", self.call_model)

            # Define edges: these determine how the control flow moves
            workflow.add_edge(START, "agent")

            workflow.add_conditional_edges(
            "agent",
            # If the latest message requires a tool, route to tools
            # Otherwise, provide a direct response
            tools_condition,
            {"tools": "tools", END: END}
            )
            workflow.add_edge("tools", "agent")
        except Exception as e:
            logger.error(f"Failed to define graph: {e}")
            return False
        # Initialize memory
        try:
            checkpointer = MemorySaver()
            # Compile the workflow into a runnable
            self.react_graph = workflow.compile(checkpointer=checkpointer)
        except Exception as e:
            logger.error(f"Failed to compile graph: {e}")
            return False
        # Show the workflow visualization
        try:
            mermaid_graph = self.react_graph.get_graph(xray=True).draw_mermaid()
            print("📊 Workflow Diagram Code:")
            print("-" * 40)
            print(mermaid_graph)
            print("-" * 40)
            print("✅ Copy the above code to a Mermaid viewer like:")
            print("   • https://mermaid.live/")
            print("="*80 + "\n")
            # Store the mermaid graph for potential future use
            self.mermaid_graph = mermaid_graph
        except Exception as e:
            logger.error(f"Failed to display graph: {e}")
            return False
        return True

def main(verbose: bool = True) -> None:
    """Main function to run the LangGraph application.

    Args:
        verbose: If True, shows detailed step-by-step execution
    """
    # Initialize graph application
    try:
        graph = Application()
        logger.info("Application initialized")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        return
    if graph.setup_llm():
        logger.info("LLM setup successfully")
    else:
        logger.error("Failed to setup LLM. Exiting.")
        return
    if graph.define_graph():
        logger.info("Graph defined successfully")
    else:
        logger.error("Failed to define graph. Exiting.")
        return
    
    try:
        messages = [HumanMessage(content="What is the levels of data classification?")]
        config = {"configurable": {"thread_id": DEFAULT_THREAD_ID}}

        if verbose:
            print("\n🔄 Starting LangGraph execution with verbose output...")
            print("=" * 60)

            # Use stream for verbose step-by-step output
            print("📝 Streaming execution steps:")
            for chunk in graph.react_graph.stream({"messages": messages}, config):
                print(f"🔹 Step: {list(chunk.keys())}")
                for node_name, node_output in chunk.items():
                    print(f"   Node '{node_name}' output:")
                    if 'messages' in node_output:
                        for msg in node_output['messages']:
                            print(f"      {type(msg).__name__}: {msg.content[:100]}...")
                    else:
                        print(f"      {node_output}")
                print("-" * 40)
        else:
            # Simple non-verbose execution
            print("\n🔄 Running LangGraph...")
            graph.react_graph.invoke({"messages": messages}, config)

        print("\n✅ Final execution result:")
        print("=" * 60)

        # Get the final state (works for both verbose and non-verbose)
        final_state = graph.react_graph.get_state(config)

        if verbose:
            print(f"📊 Final state keys: {list(final_state.values.keys())}")

        # Show the final messages
        if 'messages' in final_state.values:
            print("\n💬 Final conversation:")
            for i, m in enumerate(final_state.values['messages']):
                print(f"Message {i+1}:")
                m.pretty_print()
                print()
        else:
            print("⚠️ No messages found in final state")

    except Exception as e:
        logger.error(f"Failed to run graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()