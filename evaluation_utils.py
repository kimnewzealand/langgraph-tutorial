import logging
from agent_utils import Application
import json
from datetime import datetime
import time
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def write_evaluation_log(graph: 'Application', result: str, details: str, execution_time: float = None, DEFAULT_EMBEDDING_MODEL: str = None) -> None:
    """Write comprehensive evaluation results to a timestamped JSON log file.

    Creates a detailed log entry containing evaluation results, model configuration,
    system information, and performance metrics. Automatically creates logs directory
    if it doesn't exist and generates timestamped filenames for organization.

    Args:
        graph: The Application instance containing model and configuration information
        result: Evaluation result status (PASSED, FAILED, ERROR)
        details: Detailed description of evaluation findings and metrics
        execution_time: Time taken to execute the evaluation query (in seconds)
        DEFAULT_EMBEDDING_MODEL: Name of the embedding model used for vectorstore

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
            "provider": getattr(graph, 'provider', 'unknown'),
            "chat_model_name": getattr(graph.model, 'model', 'unknown') if hasattr(graph, 'model') else 'unknown',
            "temperature": getattr(graph.model, 'temperature', 'unknown') if hasattr(graph, 'model') else 'unknown',
            "embedding_model_name": DEFAULT_EMBEDDING_MODEL,
            "base_url": getattr(graph.model, 'base_url', 'unknown') if hasattr(graph, 'model') else 'unknown',
        }

        system_info = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.name,
            "cache_enabled": hasattr(graph, '_content_cache')
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
            "workflow_features": {
                "description": "Policy compliance analysis system",
            }
        }

        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

        logger.info(f"📝 Evaluation results written to {log_filename}")

    except Exception as e:
        logger.error(f"Failed to write evaluation log: {e}")

def run_evaluation(graph: 'Application', DEFAULT_EMBEDDING_MODEL: str) -> None:
    """Run comprehensive evaluation of the agent's data classification extraction capabilities.

    Tests the agent's ability to correctly identify and extract the three data classification
    levels (Public, Internal, Confidential) from policy documents. Logs results to JSON file
    with detailed metrics and system information.

    Args:
        graph: The Application instance containing the agent and tools
        DEFAULT_EMBEDDING_MODEL: Name of the embedding model being used

    Returns:
        None

    Raises:
        Exception: When evaluation encounters unexpected errors during execution
    """
    try:
        logger.info("🧪 EVALUATION: Testing query_vectorstore_tool tool only")
        logger.info("🎯 Expected result: 3 levels of data classification")
        start_time = time.time()
        if not graph.documents_loaded:
            graph.load_documents_tool.invoke({})
        content_result = graph.query_vectorstore_tool.invoke({"query": "How many levels of data classification are there?"})
        end_time = time.time()
        execution_time = end_time - start_time

        content_text = " ".join(content_result).lower()

        classifications_found = set()
        if "public" in content_text:
            classifications_found.add("public")
            if "internal" in content_text:
                classifications_found.add("internal")
            if "confidential" in content_text:
                classifications_found.add("confidential")

            classification_count = len(classifications_found)
            logger.info(f"🔍 Classifications found: {sorted(list(classifications_found))}")

            if classification_count == 3:
                logger.info(f"✅ EVALUATION PASSED: Found {classification_count} classification levels (expected: 3)")
                evaluation_result = "PASSED"
                evaluation_details = f"Found {classification_count} classification levels (expected: 3)"

                write_evaluation_log(graph, evaluation_result, evaluation_details, execution_time)

                logger.info("=" * 60)

            else:
                logger.error(f"❌ EVALUATION FAILED: Found {classification_count} classification levels (expected: 3)")
                evaluation_result = "FAILED"
                evaluation_details = f"Found {classification_count} classification levels (expected: 3)"
                write_evaluation_log(graph, evaluation_result, evaluation_details, execution_time)
                logger.info("🚫 Skipping policy workflow due to evaluation failure")
        else:
            logger.error(f"❌ EVALUATION FAILED: {content_text}")
            evaluation_result = "FAILED"
            evaluation_details = f"{content_text}"
            write_evaluation_log(graph, evaluation_result, evaluation_details, execution_time)
    except Exception as e:
            logger.error(f"Failed to test query tool: {e}")
            evaluation_result = "ERROR"
            evaluation_details = f"Exception during evaluation: {e}"
            write_evaluation_log(graph, evaluation_result, evaluation_details, None)

