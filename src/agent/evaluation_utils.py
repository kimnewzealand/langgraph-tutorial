import logging
import sys
import os
import threading
from collections import defaultdict

# Optional psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("psutil not available - memory monitoring disabled")

# Add the project root directory to the path when running as standalone script
if __name__ == "__main__":
    # Get the directory containing this script (src/agent/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the src directory (parent of agent directory)
    src_dir = os.path.dirname(script_dir)
    # Get the project root directory (parent of src directory)
    root_dir = os.path.dirname(src_dir)
    # Add root to Python path so imports work correctly
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

from src.agent.graph import graph, AgentState, DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL, OLLAMA_BASE_URL
from src.agent.utils import validate_startup_requirements
from langchain_core.messages import HumanMessage
import json
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Comprehensive metrics collection for evaluation runs."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics to initial state."""
        # Performance & Efficiency Metrics
        self.timing = {
            "total_execution_time": 0.0,
            "initialization_time": 0.0,
            "llm_response_time": 0.0,
            "tool_execution_time": 0.0,
            "document_loading_time": 0.0
        }

        self.response_quality = {
            "response_length_chars": 0,
            "response_length_words": 0,
            "expected_elements_found": 0,
            "expected_elements_total": 0,
            "completeness_score": 0.0
        }

        # Technical & System Metrics
        self.resource_usage = {
            "memory_peak_mb": 0.0,
            "memory_start_mb": 0.0,
            "memory_end_mb": 0.0,
            "api_call_count": 0,
            "document_count": 0,
            "embedding_calls": 0,
            "llm_calls": 0
        }

        self.error_metrics = {
            "error_count": 0,
            "timeout_count": 0,
            "recovery_attempts": 0,
            "validation_failures": 0
        }

        # Workflow & State Metrics
        self.workflow = {
            "nodes_executed": [],
            "node_execution_count": defaultdict(int),
            "tools_used": [],
            "tool_usage_count": defaultdict(int),
            "state_transitions": 0,
            "edge_traversals": []
        }

        # Internal tracking
        self._start_time = None
        self._step_start_times = {}
        self._memory_monitor = None
        self._peak_memory = 0.0

    def start_timing(self, step_name: str = "total"):
        """Start timing for a specific step."""
        current_time = time.time()
        if step_name == "total":
            self._start_time = current_time
            # Start memory monitoring if psutil is available
            if PSUTIL_AVAILABLE:
                self.resource_usage["memory_start_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
                self._start_memory_monitoring()
            else:
                self.resource_usage["memory_start_mb"] = 0.0
        self._step_start_times[step_name] = current_time

    def end_timing(self, step_name: str = "total"):
        """End timing for a specific step."""
        if step_name not in self._step_start_times:
            logger.warning(f"No start time recorded for step: {step_name}")
            return

        duration = time.time() - self._step_start_times[step_name]

        if step_name == "total":
            self.timing["total_execution_time"] = duration
            if PSUTIL_AVAILABLE:
                self.resource_usage["memory_end_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
                self._stop_memory_monitoring()
            else:
                self.resource_usage["memory_end_mb"] = 0.0
        elif step_name in self.timing:
            self.timing[step_name] = duration

        logger.debug(f"⏱️ {step_name}: {duration:.3f}s")

    def _start_memory_monitoring(self):
        """Start monitoring peak memory usage in background thread."""
        if not PSUTIL_AVAILABLE:
            return

        self._memory_monitor_active = True
        self._peak_memory = psutil.Process().memory_info().rss / 1024 / 1024

        def monitor_memory():
            while self._memory_monitor_active:
                try:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    self._peak_memory = max(self._peak_memory, current_memory)
                    time.sleep(0.1)  # Check every 100ms
                except:
                    break

        self._memory_monitor = threading.Thread(target=monitor_memory, daemon=True)
        self._memory_monitor.start()

    def _stop_memory_monitoring(self):
        """Stop memory monitoring and record peak usage."""
        if not PSUTIL_AVAILABLE:
            self.resource_usage["memory_peak_mb"] = 0.0
            return

        self._memory_monitor_active = False
        if self._memory_monitor:
            self._memory_monitor.join(timeout=1.0)
        self.resource_usage["memory_peak_mb"] = self._peak_memory

    def record_api_call(self, call_type: str = "general"):
        """Record an API call."""
        self.resource_usage["api_call_count"] += 1
        if call_type == "embedding":
            self.resource_usage["embedding_calls"] += 1
        elif call_type == "llm":
            self.resource_usage["llm_calls"] += 1

    def record_node_execution(self, node_name: str):
        """Record execution of a graph node."""
        self.workflow["nodes_executed"].append(node_name)
        self.workflow["node_execution_count"][node_name] += 1
        self.workflow["state_transitions"] += 1
        logger.debug(f"🔄 Node executed: {node_name}")

    def record_tool_usage(self, tool_name: str):
        """Record usage of a tool."""
        self.workflow["tools_used"].append(tool_name)
        self.workflow["tool_usage_count"][tool_name] += 1
        logger.debug(f"🔧 Tool used: {tool_name}")

    def record_edge_traversal(self, edge_name: str):
        """Record traversal of a graph edge."""
        self.workflow["edge_traversals"].append(edge_name)

    def record_response_quality(self, response_text: str, expected_elements: list, found_elements: list):
        """Record response quality metrics."""
        self.response_quality["response_length_chars"] = len(response_text)
        self.response_quality["response_length_words"] = len(response_text.split())
        self.response_quality["expected_elements_found"] = len(found_elements)
        self.response_quality["expected_elements_total"] = len(expected_elements)

        if len(expected_elements) > 0:
            self.response_quality["completeness_score"] = len(found_elements) / len(expected_elements)
        else:
            self.response_quality["completeness_score"] = 1.0

    def record_error(self, error_type: str = "general"):
        """Record an error occurrence."""
        self.error_metrics["error_count"] += 1
        if error_type == "timeout":
            self.error_metrics["timeout_count"] += 1
        elif error_type == "validation":
            self.error_metrics["validation_failures"] += 1

    def record_documents_loaded(self, count: int):
        """Record number of documents loaded."""
        self.resource_usage["document_count"] = count

    def get_metrics_dict(self) -> dict:
        """Get all metrics as a dictionary."""
        return {
            "performance": {
                "timing": self.timing.copy(),
                "response_quality": self.response_quality.copy()
            },
            "technical": {
                "resource_usage": self.resource_usage.copy(),
                "error_metrics": self.error_metrics.copy()
            },
            "workflow": {
                "execution_summary": {
                    "nodes_executed": self.workflow["nodes_executed"].copy(),
                    "node_execution_count": dict(self.workflow["node_execution_count"]),
                    "tools_used": self.workflow["tools_used"].copy(),
                    "tool_usage_count": dict(self.workflow["tool_usage_count"]),
                    "state_transitions": self.workflow["state_transitions"],
                    "edge_traversals": self.workflow["edge_traversals"].copy()
                }
            }
        }

def write_evaluation_log(graph_instance, result: str, details: str, execution_time: float = None,
                        graph_params: dict = None, metrics: EvaluationMetrics = None) -> None:
    """Write comprehensive evaluation results to a timestamped JSON log file.

    Creates a detailed log entry containing evaluation results, model configuration,
    system information, and performance metrics. Automatically creates logs directory
    if it doesn't exist and generates timestamped filenames for organization.

    Args:
        graph_instance: The compiled LangGraph instance
        result: Evaluation result status (PASSED, FAILED, ERROR)
        details: Detailed description of evaluation findings and metrics
        execution_time: Time taken to execute the evaluation query (in seconds)
        graph_params: Dictionary containing graph configuration parameters
        metrics: EvaluationMetrics instance with comprehensive metrics data

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

        # Use provided graph parameters or defaults
        if graph_params is None:
            graph_params = {
                "provider": "ollama",
                "chat_model_name": DEFAULT_OLLAMA_MODEL,
                "temperature": 0,
                "embedding_model_name": DEFAULT_EMBEDDING_MODEL,
                "base_url": OLLAMA_BASE_URL,
                "num_ctx": 1024,
                "tools_available": ["load_documents_tool", "query_documents_tool"],
                "workflow_type": "langgraph_stateful",
                "state_fields": ["messages", "ollama_validated", "documents_loaded", "system_initialised", "tool_calls"],
                "graph_nodes": ["should_initialise", "initialise", "assistant", "tools"],
                "conditional_edges": ["should_initialise", "assistant->tools_condition"],
                "memory_enabled": False
            }

        graph_info = {
            # Model Configuration
            "provider": graph_params.get("provider", "ollama"),
            "chat_model_name": graph_params.get("chat_model_name", DEFAULT_OLLAMA_MODEL),
            "temperature": graph_params.get("temperature", 0),
            "embedding_model_name": graph_params.get("embedding_model_name", DEFAULT_EMBEDDING_MODEL),
            "base_url": graph_params.get("base_url", OLLAMA_BASE_URL),
            "num_ctx": graph_params.get("num_ctx", 1024),

            # Graph and Workflow Status
            "graph_compiled": graph_instance is not None,
            "graph_type": "StateGraph",

            # Tools Information
            "tools_available": graph_params.get("tools_available", []),

            # Workflow Architecture
            "workflow_type": graph_params.get("workflow_type", "langgraph_stateful"),
            "state_fields": graph_params.get("state_fields", []),
            "graph_nodes": graph_params.get("graph_nodes", []),
            "conditional_edges": graph_params.get("conditional_edges", []),
            "memory_enabled": graph_params.get("memory_enabled", False)
        }

        system_info = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.name,
        }

        # Build log entry with comprehensive metrics
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

        # Add comprehensive metrics if available
        if metrics:
            metrics_data = metrics.get_metrics_dict()
            log_entry.update(metrics_data)
        else:
            # Fallback to basic metrics structure
            log_entry.update({
                "performance": {
                    "timing": {"total_execution_time": execution_time or 0.0},
                    "response_quality": {}
                },
                "technical": {
                    "resource_usage": {},
                    "error_metrics": {}
                },
                "workflow": {
                    "execution_summary": {}
                }
            })

        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

        logger.info(f"📝 Evaluation results written to {log_filename}")

    except Exception as e:
        logger.error(f"Failed to write evaluation log: {e}")

def run_evaluation(custom_chat_model: str = None, custom_embedding_model: str = None) -> None:
    """Run comprehensive evaluation of the agent's data classification extraction capabilities.

    Tests the agent's ability to correctly identify and extract the three data classification
    levels (Public, Internal, Confidential) from policy documents using the new graph structure.
    Logs results to JSON file with detailed metrics and system information.

    Args:
        custom_chat_model: Optional custom chat model name to override default
        custom_embedding_model: Optional custom embedding model name to override default

    Returns:
        None

    Raises:
        Exception: When evaluation encounters unexpected errors during execution
    """
    # Initialize comprehensive metrics collection
    metrics = EvaluationMetrics()

    try:
        logger.info("🧪 EVALUATION: Testing new graph structure with policy-specific query")
        logger.info("🎯 Expected result: Agent should return three levels, public, internal and confidential")

        # Start total timing
        metrics.start_timing("total")

        # Use custom models if provided, otherwise use defaults
        chat_model = custom_chat_model or DEFAULT_OLLAMA_MODEL
        embedding_model = custom_embedding_model or DEFAULT_EMBEDDING_MODEL

        # Create graph parameters for logging
        graph_params = {
            "provider": "ollama",
            "chat_model_name": chat_model,
            "temperature": 0,
            "embedding_model_name": embedding_model,
            "base_url": OLLAMA_BASE_URL,
            "num_ctx": 1024,
            "tools_available": ["load_documents_tool", "query_documents_tool"],
            "workflow_type": "langgraph_stateful",
            "state_fields": ["messages", "ollama_validated", "documents_loaded", "system_initialised", "tool_calls"],
            "graph_nodes": ["should_initialise", "initialise", "assistant", "tools"],
            "conditional_edges": ["should_initialise", "assistant->tools_condition"],
            "memory_enabled": False
        }

        # Validate startup requirements first
        metrics.start_timing("initialization_time")
        if not validate_startup_requirements(chat_model, embedding_model):
            logger.error("❌ EVALUATION FAILED: Startup requirements not met")
            metrics.record_error("validation")
            metrics.end_timing("total")
            write_evaluation_log(graph, "FAILED", "Startup requirements validation failed", None, graph_params, metrics)
            return
        metrics.end_timing("initialization_time")

        start_time = time.time()

        # Create initial state with a query that should force tool usage
        # This query asks for specific company policy details that require document lookup
        initial_state: AgentState = {
            "messages": [HumanMessage(content="List the data classification levels?")],
            "ollama_validated": False,
            "documents_loaded": False,
            "tool_calls": []
        }

        # Run the graph with metrics tracking
        logger.info("🚀 Running graph with data classification query...")
        final_state = None
        step_count = 0
        current_step_start = time.time()

        for state in graph.stream(initial_state):
            step_count += 1
            final_state = state
            step_nodes = list(state.keys())
            step_end = time.time()
            step_duration = step_end - current_step_start

            logger.info(f"🔄 Graph step completed: {step_nodes}")

            # Record node executions and track specific operations with timing
            for node_name in step_nodes:
                metrics.record_node_execution(node_name)

                # Track specific node types with their timing
                if "tool" in node_name.lower():
                    metrics.start_timing("tool_execution_time")
                    metrics.record_tool_usage(node_name)
                    metrics.timing["tool_execution_time"] += step_duration
                    # Record API call for tool usage (tools may involve LLM or embedding calls)
                    metrics.record_api_call("general")
                    logger.debug(f"🔧 Tool execution time for {node_name}: {step_duration:.3f}s")

                elif node_name == "initialise":
                    metrics.start_timing("document_loading_time")
                    metrics.timing["document_loading_time"] = step_duration
                    metrics.record_documents_loaded(2)
                    # Record embedding API call (document loading involves embedding creation)
                    metrics.record_api_call("embedding")
                    logger.debug(f"📄 Document loading time: {step_duration:.3f}s")

                elif node_name == "assistant":
                    metrics.start_timing("llm_response_time")
                    metrics.timing["llm_response_time"] = step_duration
                    # Record LLM API call (assistant node involves LLM inference)
                    metrics.record_api_call("llm")
                    logger.debug(f"🤖 LLM response time: {step_duration:.3f}s")

            # Prepare for next step timing
            current_step_start = time.time()
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"⏱️ Total execution time: {execution_time:.2f} seconds")

        # Extract response from final state
        if final_state and "assistant" in final_state:
            messages = final_state["assistant"].get("messages", [])
            if messages:
                response_content = str(messages[-1].content).lower()
                full_response = str(messages[-1].content)
                logger.info(f"📝 Agent response: {response_content[:200]}...")

                # Check for specific policy timeframes that require document lookup
                expected_timeframes = ["48 hours", "72 hours"]
                timeframes_found = set()
                if "internal" in response_content:
                    timeframes_found.add("internal")
                if "public" in response_content:
                    timeframes_found.add("public")
                if "confidential" in response_content:
                    timeframes_found.add("confidential")


                timeframe_count = len(timeframes_found)
                logger.info(f"🔍 Data Classification levels found: {sorted(list(timeframes_found))}")

                # Record response quality metrics
                metrics.record_response_quality(
                    response_text=full_response,
                    expected_elements=expected_timeframes,
                    found_elements=list(timeframes_found)
                )

                if timeframe_count >= 1:  # At least one specific timeframe found
                    logger.info(f"✅ EVALUATION PASSED: Found {timeframe_count} specific policy timeframes")
                    evaluation_result = "PASSED"
                    evaluation_details = f"Found {timeframe_count} policy timeframes: {sorted(list(timeframes_found))}"
                else:
                    logger.error(f"❌ EVALUATION FAILED: Found {timeframe_count} specific policy timeframes (expected: at least 1)")
                    evaluation_result = "FAILED"
                    evaluation_details = f"Found {timeframe_count} policy timeframes: {sorted(list(timeframes_found))}"

                # End total timing and write comprehensive log
                metrics.end_timing("total")
                write_evaluation_log(graph, evaluation_result, evaluation_details, execution_time, graph_params, metrics)
                logger.info("=" * 60)
            else:
                logger.error("❌ EVALUATION FAILED: No response messages found")
                metrics.record_error("validation")
                metrics.end_timing("total")
                write_evaluation_log(graph, "FAILED", "No response messages in final state", execution_time, graph_params, metrics)
        else:
            logger.error("❌ EVALUATION FAILED: No final state or assistant node found")
            metrics.record_error("validation")
            metrics.end_timing("total")
            write_evaluation_log(graph, "FAILED", "Graph execution did not produce expected final state", execution_time, graph_params, metrics)

    except Exception as e:
        logger.error(f"❌ EVALUATION ERROR: {e}")
        metrics.record_error("general")
        metrics.end_timing("total")
        evaluation_result = "ERROR"
        evaluation_details = f"Exception during evaluation: {e}"
        write_evaluation_log(graph, evaluation_result, evaluation_details, None, graph_params, metrics)


if __name__ == "__main__":
    """Run evaluation as a standalone script."""
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run LangGraph Agent Evaluation")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging output")
    parser.add_argument("--model", "-m", type=str, default=None,
                       help="Override default Ollama model")
    parser.add_argument("--embedding-model", "-e", type=str, default=None,
                       help="Override default embedding model")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 Verbose logging enabled")

    # Override models if specified
    if args.model:
        logger.info(f"🔧 Using custom chat model: {args.model}")
    if args.embedding_model:
        logger.info(f"🔧 Using custom embedding model: {args.embedding_model}")

    # Print startup banner
    print("=" * 60)
    print("🧪 LangGraph Agent Evaluation Suite")
    print("=" * 60)
    print(f"📋 Chat Model: {args.model or DEFAULT_OLLAMA_MODEL}")
    print(f"🔤 Embedding Model: {args.embedding_model or DEFAULT_EMBEDDING_MODEL}")
    print(f"🌐 Base URL: {OLLAMA_BASE_URL}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print("=" * 60)

    try:
        # Run the evaluation with custom models if specified
        run_evaluation(custom_chat_model=args.model, custom_embedding_model=args.embedding_model)
        print("\n✅ Evaluation completed successfully!")
        print("📝 Check the logs/ directory for detailed results")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        logger.warning("Evaluation interrupted by user (Ctrl+C)")

    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        exit(1)

