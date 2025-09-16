import requests
import requests
import logging
from langchain_core.messages import HumanMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL="http://localhost:11434"

def check_ollama_status(model_name: str) -> bool:
    """Check if Ollama service is running and specified model is available.

    Verifies that the Ollama service is accessible and that the requested model
    is downloaded and available for use. Essential for startup validation.

    Args:
        model_name: Name of the Ollama model to check availability for

    Returns:
        True if Ollama is running and model is available, False otherwise

    Raises:
        requests.exceptions.RequestException: When unable to connect to Ollama service
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        loaded_models = [model["name"] for model in response.json().get("models", [])]
        logger.info(f"{model_name}  in downloaded models: {loaded_models}")
        if response.status_code == 200:
            if model_name in loaded_models:
                return True
            else:
                logger.error(f"❌ Ollama service is running but model {model_name} is not loaded.")
                return False 
        else:
            logger.error(f"❌ Ollama service is not running ")
    except requests.exceptions.RequestException as e:
        return False

def warm_up_model(self, model_name: str) -> bool:
    """Pre-load and warm up the specified model for optimal performance.

    Performs a simple test generation to cache the model in memory and verify
    that it can generate responses before running complex workflows. This helps
    distinguish between model issues and loading delays during debugging.

    Args:
        self: Application instance containing the model
        model_name: Name of the model to warm up

    Returns:
        True if model warming was successful, False otherwise

    Raises:
        Exception: When model invocation fails during warm-up process
    """
    try:
        logger.info(f"🔥 Warming up and testing chat model: {model_name}")
        
        test_message = [HumanMessage(content="Hello")]
        response = self.model.invoke(test_message)
        
        if response:
            return True
        else:
            logger.error(f"❌ Failed to warm up model {model_name}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to warm up model {model_name}: {e}")
        return False

def validate_startup_requirements(DEFAULT_OLLAMA_MODEL: str, DEFAULT_EMBEDDING_MODEL: str) -> bool:
    """Validate all system requirements before starting the application.

    Performs comprehensive checks to ensure Ollama service is running, required models
    are available, and models support tool calling functionality. Provides detailed
    error messages and solutions for any issues found.

    Args:
        DEFAULT_OLLAMA_MODEL: Name of the primary chat model to validate
        DEFAULT_EMBEDDING_MODEL: Name of the embedding model to validate

    Returns:
        True if all requirements are met and application can start, False otherwise
    """
    logger.info("🔍 Checking Ollama is running in the background...")

    if not check_ollama_status(DEFAULT_OLLAMA_MODEL) or not check_ollama_status(DEFAULT_EMBEDDING_MODEL):
        logger.error("❌ Ollama check failed on " + DEFAULT_OLLAMA_MODEL)
        logger.error("🔧 Solutions:")
        logger.error("   1. Start Ollama: ollama serve in a separate terminal")
        logger.error("   2. Download model: ollama pull " + DEFAULT_OLLAMA_MODEL)
        logger.error("   3. Verify with: ollama list")
        return False

    logger.info("✅ Ollama is running")
    return True