import logging
import os
import re
import time
from datetime import datetime, timedelta
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
from ollama_utils import check_ollama_status,warm_up_model

# Load environment variables from .env file
load_dotenv()

# LLM parmameters
DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"
DEFAULT_OLLAMA_MODEL = "qwen2:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_THREAD_ID = "1"
# Set environment variables for better performance
OLLAMA_NUM_PARALLEL=4
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_FLASH_ATTENTION=1

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
    # Adds the latest message rather than overwriting it with the latest state.
    messages: Annotated[list[AnyMessage], add_messages]

class Application:
    """An application for processing policies using LangGraph."""

    def __init__(self) -> None:
        """Initialize the application."""
        self.model: Optional[ChatAnthropic | ChatOllama] = None
        self.mermaid_graph: Optional[str] = None
        self.react_graph: Optional[StateGraph] = None
        self.workflow: Optional[StateGraph] = None
        self.provider: Optional[str] = "ollama"


    @tool
    def retrieve_content(query: str) -> List[str]:
        """Fast content retrieval"""
        try:
            file_path = "documents/sample_data.txt"
        
            with open(file_path, "r", encoding="utf-8") as f:
                content= f.read()
        
            return content
        
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ["Error retrieving content."]
        
    @tool
    def create_action_plan(query: str) -> List[str]:
        """Generate a chronologically ordered action plan based on policy deadlines and requirements.

        Args:
            query: The query to analyze for action items and deadlines

        Returns:
            List of formatted action items with calculated due dates, sorted chronologically
        """
        try:
            logger.info("Creating action plan based on policy deadlines")

            # First, retrieve the content to analyze
            file_path = "documents/sample_data.txt"
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Define deadline patterns to search for
            deadline_patterns = {
                r'within (\d+) hours?': lambda m: datetime.now() + timedelta(hours=int(m.group(1))),
                r'within (\d+) business days?': lambda m: datetime.now() + timedelta(days=int(m.group(1))),
                r'within (\d+) days?': lambda m: datetime.now() + timedelta(days=int(m.group(1))),
                r'quarterly by the (\d+)(?:st|nd|rd|th)': lambda m: self._get_next_quarterly_date(int(m.group(1))),
                r'by December (\d+)(?:st|nd|rd|th)': lambda m: datetime(datetime.now().year, 12, int(m.group(1))),
                r'by the (\d+)(?:st|nd|rd|th) of each quarter': lambda m: self._get_next_quarterly_date(int(m.group(1))),
                r'annually.*by December (\d+)(?:st|nd|rd|th)': lambda m: datetime(datetime.now().year, 12, int(m.group(1)))
            }

            # Action keywords to identify actionable items
            action_keywords = [
                'approval', 'approved', 'training', 'review', 'audit', 'conduct',
                'monitor', 'report', 'revoke', 'grant', 'address', 'required'
            ]

            action_items = []
            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                # Check if line contains action keywords
                has_action = any(keyword in line.lower() for keyword in action_keywords)

                if has_action:
                    # Look for deadline patterns in this line
                    for pattern, date_func in deadline_patterns.items():
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            try:
                                due_date = date_func(match)

                                # Extract policy section (look for numbered sections)
                                section_match = re.search(r'(\d+(?:\.\d+)*)', line)
                                section = section_match.group(1) if section_match else f"Line {line_num}"

                                # Clean up the action description
                                action_desc = re.sub(r'\s+', ' ', line).strip()
                                if len(action_desc) > 100:
                                    action_desc = action_desc[:97] + "..."

                                action_items.append({
                                    'due_date': due_date,
                                    'action': action_desc,
                                    'section': section,
                                    'days_until': (due_date - datetime.now()).days
                                })
                            except Exception as e:
                                logger.warning(f"Error parsing date from line {line_num}: {e}")

            # Sort by due date (earliest first)
            action_items.sort(key=lambda x: x['due_date'])

            # Format the results
            if not action_items:
                return ["No time-sensitive action items found in the policy documents."]

            formatted_items = []
            for item in action_items:
                due_date_str = item['due_date'].strftime("%Y-%m-%d")
                days_until = item['days_until']

                if days_until < 0:
                    urgency = "⚠️ OVERDUE"
                elif days_until <= 7:
                    urgency = "🔴 URGENT"
                elif days_until <= 30:
                    urgency = "🟡 SOON"
                else:
                    urgency = "🟢 PLANNED"

                formatted_item = (
                    f"{urgency} | Due: {due_date_str} ({days_until:+d} days) | "
                    f"Section {item['section']}: {item['action']}"
                )
                formatted_items.append(formatted_item)

            return formatted_items

        except Exception as e:
            logger.error(f"Error creating action plan: {e}")
            return [f"Error creating action plan: {str(e)}"]

    def policy_analysis_node(self, state: MessagesState) -> Dict[str, List[Any]]:
        """Custom node that automatically calls both tools in sequence"""
        try:
            logger.info("🔍 Running comprehensive policy analysis...")

            # Get the user's query from the last message
            last_message = state["messages"][-1]
            query = last_message.content

            # Step 1: Retrieve policy content
            logger.info("📄 Step 1: Retrieving policy content...")
            policy_content = self.retrieve_content(query)

            # Step 2: Create action plan
            logger.info("📋 Step 2: Creating action plan...")
            action_plan = self.create_action_plan(query)

            # Combine results into a comprehensive response
            combined_response = []
            combined_response.extend(policy_content)
            combined_response.append("\n--- COMPLIANCE ACTION PLAN ---")
            combined_response.extend(action_plan)

            # Create a response message
            from langchain_core.messages import AIMessage
            response_content = "\n".join(combined_response)
            response = AIMessage(content=response_content)

            return {"messages": [response]}

        except Exception as e:
            logger.error(f"Error in policy analysis: {e}")
            error_response = AIMessage(content=f"Error during policy analysis: {str(e)}")
            return {"messages": [error_response]}

    def _get_next_quarterly_date(self, day: int) -> datetime:
        """Calculate the next quarterly deadline date.

        Args:
            day: Day of the month for the quarterly deadline

        Returns:
            Next quarterly date (March, June, September, or December)
        """
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month

        # Quarterly months: March (3), June (6), September (9), December (12)
        quarterly_months = [3, 6, 9, 12]

        # Find the next quarterly month
        next_quarter_month = None
        for month in quarterly_months:
            quarter_date = datetime(current_year, month, day)
            if quarter_date > current_date:
                next_quarter_month = month
                break

        # If no quarter found in current year, use March of next year
        if next_quarter_month is None:
            return datetime(current_year + 1, 3, day)

        return datetime(current_year, next_quarter_month, day)
  
    def setup_llm(self, model_name: Optional[str] = None) -> bool:
        """Setup the Language Model with multiple provider options.

        Args:
            model_name: Optional specific model name

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.provider == "ollama":
                model_name = model_name or DEFAULT_OLLAMA_MODEL
                self.model = ChatOllama(
                    model=model_name,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0,
                    num_ctx=1024,        # Reduce context window if not needed
                ).bind_tools([self.retrieve_content, self.create_action_plan])

            elif self.provider == "anthropic":
                model_name = model_name or DEFAULT_ANTHROPIC_MODEL
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                if not anthropic_api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

                self.model = ChatAnthropic(
                    model=model_name,
                    anthropic_api_key=anthropic_api_key
                ).bind_tools([self.retrieve_content, self.create_action_plan])

            else:
                raise ValueError(f"Unsupported provider: {self.provider}. Supported: 'ollama', 'anthropic'")
            return True

        except ImportError as e:
            logger.error(f"Missing package for {self.provider}: {e}")
            logger.info(f"Install with: pip install langchain-{self.provider}")
            return False
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting up {self.provider}: {e}")
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
        system_content = """You are a helpful policy compliance assistant. When users ask about policies:

1. ALWAYS use retrieve_content first to get policy information
2. AFTER retrieving policy content, ALWAYS use create_action_plan to identify deadlines and compliance requirements
3. Provide both the direct answer AND actionable compliance items
4. Keep answers concise but comprehensive

Example workflow:
- User asks: "What are the data classification levels?"
- You should:
  a) Call retrieve_content to get policy details
  b) Call create_action_plan to find related deadlines/actions
  c) Answer: "There are three levels: Public, Internal, Confidential. Here are related compliance actions: [action plan items]"

Always be proactive in helping users understand both the policy content AND their compliance obligations."""

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
            workflow.add_node("tools", ToolNode([self.retrieve_content,self.create_action_plan]))
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
        if graph.provider == "ollama":
            logger.info(f"  Check if Ollama service is running with model {DEFAULT_OLLAMA_MODEL}")
            if check_ollama_status(DEFAULT_OLLAMA_MODEL):
                if warm_up_model(graph,DEFAULT_OLLAMA_MODEL):
                    logger.info("✅ LLM setup successful")
            else:
                logger.error("Failed to setup LLM. Exiting.")
                return
    else:
        logger.error("Failed to setup LLM. Exiting.")
        return
    if graph.define_graph():
        logger.info("Graph defined successfully")
    else:
        logger.error("Failed to define graph. Exiting.")
        return
    
    try:
        messages = [HumanMessage(content="What are the levels of data classification? Also create an action plan with deadlines for any compliance requirements related to data classification.")]
        config = {"configurable": {"thread_id": DEFAULT_THREAD_ID}}
        start_time = time.time()
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
                            print(f"{type(msg).__name__}: {msg.content[:100]}...")
                    else:
                        print(f"{node_output}")
                print("-" * 40)
        else:
            # Simple non-verbose execution
            print("\n🔄 Running LangGraph...")
            graph.react_graph.invoke({"messages": messages}, config)
        end_time = time.time()
        logger.info(f"Model response time: {end_time - start_time:.2f} seconds")
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