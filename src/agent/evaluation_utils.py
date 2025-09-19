import logging
import sys
import os

from agent.graph import graph, AgentState, DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL, OLLAMA_BASE_URL
from agent.utils import validate_startup_requirements
from langchain_core.messages import HumanMessage
import json
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def write_evaluation_log(graph_instance, result: str, details: str, execution_time: float = None, model_name: str = None) -> None:
    """Write comprehensive evaluation results to a timestamped JSON log file.

    Creates a detailed log entry containing evaluation results, model configuration,
    system information, and performance metrics. Automatically creates logs directory
    if it doesn't exist and generates timestamped filenames for organization.

    Args:
        graph_instance: The compiled LangGraph instance
        result: Evaluation result status (PASSED, FAILED, ERROR)
        details: Detailed description of evaluation findings and metrics
        execution_time: Time taken to execute the evaluation query (in seconds)
        model_name: Name of the model used for evaluation

    Returns:
        None

    Raises:
        Exception: When log file creation or writing fails
    """
    try:
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            logger.info(f"📁 Created logs directory: {logs_dir}")

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(logs_dir, f"evaluation_{timestamp_str}.json")

        graph_info = {
            # Model Configuration
            "provider": "ollama",
            "chat_model_name": model_name or DEFAULT_OLLAMA_MODEL,
            "temperature": 0,
            "embedding_model_name": DEFAULT_EMBEDDING_MODEL,
            "base_url": OLLAMA_BASE_URL,
            "num_ctx": 1024,

            # Graph and Workflow Status
            "graph_compiled": graph_instance is not None,
            "graph_type": "StateGraph",

            # Tools Information
            "tools_available": [
                "load_documents_tool",
                "query_documents_tool"
            ],

            # Workflow Architecture
            "workflow_type": "langgraph_stateful",
            "state_fields": [
                "messages", "ollama_validated", "documents_loaded", "tool_calls"
            ],
            "graph_nodes": ["assistant", "tools"],
            "conditional_edges": ["assistant->tools_condition"],
            "memory_enabled": False
        }

        system_info = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.name,
        }

        log_entry = {
            "timestamp": timestamp.isoformat(),
            "evaluation": {
                "result": result,
                "details": details,
                "execution_time_seconds": execution_time
            },
            "graph_info": graph_info,
            "system": system_info,
        }

        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

        logger.info(f"📝 Evaluation results written to {log_filename}")

    except Exception as e:
        logger.error(f"Failed to write evaluation log: {e}")

def run_evaluation() -> None:
    """Run comprehensive evaluation of the agent's data classification extraction capabilities.

    Tests the agent's ability to correctly identify and extract the three data classification
    levels (Public, Internal, Confidential) from policy documents using the new graph structure.
    Logs results to JSON file with detailed metrics and system information.

    Returns:
        None

    Raises:
        Exception: When evaluation encounters unexpected errors during execution
    """
    try:
        logger.info("🧪 EVALUATION: Testing new graph structure with data classification query")
        logger.info("🎯 Expected result: Agent should identify 3 levels of data classification")

        # Validate startup requirements first
        if not validate_startup_requirements(DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL):
            logger.error("❌ EVALUATION FAILED: Startup requirements not met")
            write_evaluation_log(graph, "FAILED", "Startup requirements validation failed", None, DEFAULT_OLLAMA_MODEL)
            return

        start_time = time.time()

        # Create initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content="How many levels of data classification are there? Please list them.")],
            "ollama_validated": False,
            "documents_loaded": False,
            "tool_calls": []
        }

        # Run the graph
        logger.info("🚀 Running graph with data classification query...")
        final_state = None

        for state in graph.stream(initial_state):
            final_state = state
            logger.info(f"🔄 Graph step completed: {list(state.keys())}")

        end_time = time.time()
        execution_time = end_time - start_time

        # Extract response from final state
        if final_state and "assistant" in final_state:
            messages = final_state["assistant"].get("messages", [])
            if messages:
                response_content = str(messages[-1].content).lower()
                logger.info(f"📝 Agent response: {response_content[:200]}...")

                # Check for data classification levels
                classifications_found = set()
                if "public" in response_content:
                    classifications_found.add("public")
                if "internal" in response_content:
                    classifications_found.add("internal")
                if "confidential" in response_content:
                    classifications_found.add("confidential")

                classification_count = len(classifications_found)
                logger.info(f"🔍 Classifications found: {sorted(list(classifications_found))}")

                if classification_count == 3:
                    logger.info(f"✅ EVALUATION PASSED: Found {classification_count} classification levels (expected: 3)")
                    evaluation_result = "PASSED"
                    evaluation_details = f"Found {classification_count} classification levels: {sorted(list(classifications_found))}"
                else:
                    logger.error(f"❌ EVALUATION FAILED: Found {classification_count} classification levels (expected: 3)")
                    evaluation_result = "FAILED"
                    evaluation_details = f"Found {classification_count} classification levels: {sorted(list(classifications_found))}"

                write_evaluation_log(graph, evaluation_result, evaluation_details, execution_time, DEFAULT_OLLAMA_MODEL)
                logger.info("=" * 60)
            else:
                logger.error("❌ EVALUATION FAILED: No response messages found")
                write_evaluation_log(graph, "FAILED", "No response messages in final state", execution_time, DEFAULT_OLLAMA_MODEL)
        else:
            logger.error("❌ EVALUATION FAILED: No final state or assistant node found")
            write_evaluation_log(graph, "FAILED", "Graph execution did not produce expected final state", execution_time, DEFAULT_OLLAMA_MODEL)

    except Exception as e:
        logger.error(f"❌ EVALUATION ERROR: {e}")
        evaluation_result = "ERROR"
        evaluation_details = f"Exception during evaluation: {e}"
        write_evaluation_log(graph, evaluation_result, evaluation_details, None, DEFAULT_OLLAMA_MODEL)

