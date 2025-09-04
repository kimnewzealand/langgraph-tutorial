import sys
from typing import Any, Dict
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

def check_ollama_status(model_name) -> Dict[str, Any]:
    """Check if Ollama is running and model is loaded"""
    try:
        # Check if Ollama service is running with model
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        loaded_models = [model["name"] for model in response.json().get("models", [])]
        logger.info(f"  Downloaded models: {loaded_models}")
        if response.status_code == 200:
            if model_name in loaded_models:
                logger.info(f"  Ollama service is running with model: {model_name}")   
                return True
            else:
                logger.error(f"❌ Ollama service is running but model {model_name} is not loaded.")
                return False 
        else:
            logger.error(f"❌ Ollama service is not running ")
    except requests.exceptions.RequestException as e:
        return False

def warm_up_model(self,model_name: str) -> bool:
    """Pre-load and warm up the model"""
    try:
        logger.info(f"  🔥 Warming up and testing model: {model_name}")
        
        # Simple test call to load model into memory
        test_message = [HumanMessage(content="Hello")]
        response = self.model.invoke(test_message)
        
        if response.status_code == 200:
            logger.info(f"  Model {model_name} warmed up successfully")
            return True
        else:
            logger.error(f"❌ Failed to warm up model {model_name}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to warm up model {model_name}: {e}")
        return False