import logging
from dotenv import load_dotenv
from agent_utils import Application
from ollama_utils import warm_up_model, validate_startup_requirements
from evaluation_utils import run_evaluation

load_dotenv()

DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_THREAD_ID = "1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Main entry point for the LangGraph policy compliance assistant application.

    Orchestrates the complete application lifecycle including startup validation,
    model initialization, graph definition, agent demonstration, and evaluation.
    Validates Ollama service availability, sets up the LLM with tool binding,
    demonstrates intelligent query processing, and runs comprehensive evaluation.

    Returns:
        None

    Raises:
        Exception: When application initialization, model setup, or graph definition fails
    """
    if not validate_startup_requirements(DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL):
        logger.error("❌ Startup validation failed. Please fix the issues above.")
        return
    
    try:
        graph = Application()
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        return
 
    if graph.setup_llm(DEFAULT_OLLAMA_MODEL,OLLAMA_BASE_URL):
        logger.info("✅ Set up LLM")
    else:
        logger.error("Failed to setup LLM. Exiting.")
        return

    if graph.provider == "ollama":
        if warm_up_model(graph, DEFAULT_OLLAMA_MODEL):
            logger.info("✅ LLM warmup successful")
        else:
            logger.error("Failed to warm up model. Exiting.")
            return

    if not graph.define_graph():
        logger.error("Failed to define graph. Exiting.")
        return

    logger.info("Graph defined successfully")

    logger.info("🤖 DEMONSTRATING AGENT")
    logger.info("=" * 60)

    test_queries = [
        "Are the documents loaded properly?",
        "What are the data classification levels?",
        "Create an action plan for data classification compliance",
    ]

    for i, query in enumerate(test_queries, 1):
        logger.info(f"🔍 Test {i}: {query}")
        try:
            response = graph.chat_with_agent(query, thread_id=f"demo_{i}")
            logger.info(f"🤖 Agent Response: {response}..")
            logger.info("-" * 40)
        except Exception as e:
            logger.error(f"Failed to test agent query '{query}': {e}")

    logger.info("✅ Agent demonstration completed")
    logger.info("=" * 60)
        
    run_evaluation(graph,DEFAULT_EMBEDDING_MODEL)

if __name__ == "__main__":
    main()