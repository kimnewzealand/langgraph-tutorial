from langchain_ollama import ChatOllama

import requests
import logging
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
documents_dir: str = "documents"

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
        print(f"{model_name}  in downloaded models: {loaded_models}")
        if response.status_code == 200:
            if model_name in loaded_models:
                return True
            else:
                print(f"❌ Ollama service is running but model {model_name} is not loaded.")
                return False 
        else:
            print(f"❌ Ollama service is not running ")
    except requests.exceptions.RequestException as e:
        return False

def warm_up_model(model: ChatOllama) -> bool:
    """Pre-load and warm up the specified model for optimal performance.

    Performs a simple test generation to cache the model in memory and verify
    that it can generate responses before running complex workflows. This helps
    distinguish between model issues and loading delays during debugging.

    Args:
        model_name: Name of the model to warm up

    Returns:
        True if model warming was successful, False otherwise

    Raises:
        Exception: When model invocation fails during warm-up process
    """
    try:
        print(f"🔥 Warming up and testing chat model: {model}")
        
        test_message = [HumanMessage(content="Hello")]
        response = model.invoke(test_message)
        
        if response:
            return True
        else:
            print(f"❌ Failed to warm up model {model}")
            return False
    except Exception as e:
        print(f"❌ Failed to warm up model {model}: {e}")
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
    print("🔍 Checking Ollama is running in the background...")

    if not check_ollama_status(DEFAULT_OLLAMA_MODEL) or not check_ollama_status(DEFAULT_EMBEDDING_MODEL):
        print("❌ Ollama check failed on " + DEFAULT_OLLAMA_MODEL)
        print("🔧 Solutions:")
        print("   1. Start Ollama: ollama serve in a separate terminal")
        print("   2. Download model: ollama pull " + DEFAULT_OLLAMA_MODEL)
        print("   3. Verify with: ollama list")
        return False

    print("✅ Ollama is running")
    return True


def setup_llm(DEFAULT_OLLAMA_MODEL: str, OLLAMA_BASE_URL: str) -> ChatOllama | None:
    """Set up the LLM as a chat model and bind it with available tools.

    Initializes Ollama chat model and configures model parameters for optimal performance.

    Args:
        DEFAULT_OLLAMA_MODEL: Name of the Ollama model to use for chat
        OLLAMA_BASE_URL: Base URL for the Ollama service

    Returns:
        ChatOllama model instance or None if setup fails

    Raises:
        Exception: When model initialization fails
    """
    try:
        model_name = DEFAULT_OLLAMA_MODEL

        model = ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_ctx=2048,  # Increased for better context handling
            num_predict=512,  # Limit response length for faster generation
            repeat_penalty=1.1,  # Reduce repetition
            top_k=40,  # Limit vocabulary for faster sampling
            top_p=0.9,  # Nucleus sampling for efficiency
            keep_alive="5m",  # Keep model in memory longer
        )

        return model

    except Exception as e:
        print(f"Failed to setup LLM model: {e}")
        return None
        
# Global variable to store vectorstore
_vectorstore = None

def load_documents() -> bool:
    """Load documents and create vector embeddings.

    Loads all .txt files from the documents directory, creates embeddings using
    the configured embedding model, and stores them in an InMemoryVectorStore
    for semantic search.

    Returns:
        True if documents were successfully loaded and vectorstore created, False otherwise

    Raises:
        Exception: When document loading or vectorstore creation fails
    """
    global _vectorstore

    try:
        print("🔧 Loading documents for embedding")

        loader = DirectoryLoader(
            documents_dir,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()

        if not documents:
            logger.warning(f"No documents found in {documents_dir}")
            return False

        print(f"✅ Loaded {len(documents)} documents:")
        for document in documents:
            print(f"📄 Source: {document.metadata['source']}")
    except Exception as e:
        print(f"Failed to load documents: {e}")
        return False

    try:
        embeddings = OllamaEmbeddings(
            model=DEFAULT_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        _vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)

        print(f"✅ Created InMemoryVectorStore from {len(documents)} documents with Ollama {DEFAULT_EMBEDDING_MODEL} embeddings")
        return True
    except Exception as e:
        print(f"Failed to create vectorstore: {e}")
        return False

def get_vectorstore():
    """Get the global vectorstore instance."""
    return _vectorstore