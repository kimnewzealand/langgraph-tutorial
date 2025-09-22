# LangGraph Tutorial Project Plan

## 🎯 Project Goal
This repository demonstrates a modern LangGraph-powered intelligent policy compliance assistant that:
1. **Modular Architecture**: Clean separation of concerns with `src/agent/` structure
2. **Stateful Workflows**: LangGraph StateGraph with proper state management
3. **Tool Integration**: Seamless tool calling with validation and error handling
4. **Local AI**: Ollama-based LLM and embeddings for privacy and cost efficiency
5. **Document Intelligence**: Semantic search and information extraction from policy documents

## 🏗️ New Architecture Principles
- **Modular Design**: Organized codebase with clear separation (`graph.py`, `utils.py`, `evaluation_utils.py`)
- **State-First**: TypedDict-based state management with proper type hints
- **Tool-Centric**: Function-based tools with clear interfaces and validation
- **Local & Secure**: All processing happens locally with Ollama models
- **Testable**: Comprehensive evaluation framework for quality assurance

## ✅ Current Implementation Status

### 🔧 **New Modular Architecture - COMPLETED**
- ✅ **`src/agent/graph.py`**: Main LangGraph StateGraph definition with nodes and edges
- ✅ **`src/agent/utils.py`**: Utility functions for LLM setup, validation, and document loading
- ✅ **`src/agent/evaluation_utils.py`**: Comprehensive evaluation framework with detailed metrics
- ✅ **`src/agent/system_prompt.txt`**: Generic system prompt with tool orchestration guidelines
- ✅ **ContextSchema**: System prompt loading from file with static method
- ✅ **AgentState TypedDict**: Clean state management with proper type hints
- ✅ **State-Based Initialization**: One-time system setup with conditional routing
- ✅ **Startup Validation**: Ollama service and model availability checking
- ✅ **Error Handling**: Graceful failure handling throughout the workflow
- ✅ **LangGraph Studio Compatible**: No custom checkpointer for studio compatibility

### 📊 **Evaluation & Monitoring - COMPLETED**
- ✅ **Comprehensive Metrics**: Performance timing, response quality, resource usage
- ✅ **Standalone Evaluation**: Command-line evaluation suite with detailed logging
- ✅ **Performance Analysis**: Accurate timing breakdown (initialization, LLM, document loading)
- ✅ **API Call Tracking**: Embedding and LLM API call monitoring
- ✅ **Quality Metrics**: Response completeness and accuracy scoring
- ✅ **Workflow Analysis**: Node execution tracking and state transitions
- ✅ **JSON Logging**: Structured evaluation results for analysis

### 🛠️ **Tool Implementation - COMPLETED**

#### 1. **`load_documents_tool(state: AgentState)`** ✅
- **Type**: State-modifying function tool
- **Purpose**: Load documents and update state with loading status
- **Implementation**: Calls `load_documents()` from utils and updates `documents_loaded` state
- **Returns**: Dictionary with AIMessage confirming document loading

#### 2. **`query_documents_tool(query: str)`** ✅
- **Type**: Query function tool
- **Purpose**: Semantic search through loaded policy documents
- **Implementation**: Full vector similarity search with relevance scoring
- **Returns**: Dictionary with ranked query results and metadata

#### 3. **`create_plan_tool(query: str)`** ✅
- **Type**: Planning function tool
- **Purpose**: Generate structured action plans based on policy documents
- **Implementation**: RAG-based plan generation with document context
- **Returns**: Dictionary with detailed action plan including priorities and timelines

#### 4. **System Integration Tools** ✅
- **`validate_startup_requirements()`**: Checks Ollama service and model availability
- **`setup_llm()`**: Initializes ChatOllama with proper configuration
- **`load_documents()`**: Creates embeddings and vector store from document directory
- **`get_vectorstore()`**: Global vectorstore access for tool integration

### 📊 **Current LangGraph Workflow**

#### **StateGraph Architecture with Initialization**
```
START → should_initialise → initialise (if needed) → assistant
          ↓                      ↓                      ↓
    Check system state    Load documents &         System Prompt
    (documents_loaded,    validate Ollama         + User Message
     ollama_validated,                                  ↓
     system_initialised)                         LLM Processing
          ↓                                            ↓
    If already initialised                    Response/Tool Calls
          ↓                                            ↓
      assistant ←─────────────────────────────── tools_condition
                                                       ↓
                                                Tool Execution
                                                       ↓
                                                Back to assistant
                                                       ↓
                                                     END
```

#### **AgentState Management**
- **`messages`**: Annotated list of conversation messages with `add_messages`
- **`ollama_validated`**: Boolean flag for Ollama service validation status
- **`documents_loaded`**: Boolean flag for document loading status
- **`system_initialised`**: Boolean flag for one-time system initialization
- **`tool_calls`**: List of executed tool names for tracking

#### **ContextSchema Integration**
- **`system_prompt`**: TypedDict field for system prompt configuration
- **`get_default_system_prompt()`**: Static method to load from `system_prompt.txt`

#### **Key Architectural Benefits**
- **One-Time Initialization**: System setup runs only once per session
- **State-Based Routing**: Conditional edges based on initialization status
- **Type Safety**: TypedDict state with proper type annotations
- **Modularity**: Separated concerns across multiple files
- **Studio Compatible**: No custom checkpointer for LangGraph Studio compatibility
- **System Prompt**: Loaded from file via ContextSchema static method
- **Testability**: Isolated functions enable comprehensive testing
- **Maintainability**: Clear structure makes debugging and enhancement easier

## 🎯 **Next Development Priorities**

### 🧪 **Testing & Validation**
- [ ] **Unit Tests**: Individual function testing for utils and tools
- [ ] **Integration Tests**: End-to-end workflow testing
- [ ] **Performance Tests**: Benchmark document loading and query times with different models
- [ ] **Tool Usage Tests**: Verify actual tool calling vs. simulation

### 📊 **Feature Enhancements**
- [ ] **Timeline Management**: Deadline extraction and chronological sorting
- [ ] **Output Formatting**: Structured response formats (JSON, Markdown)
- [ ] **Model Optimization**: Test with larger models for better tool calling

### 🔧 **System Improvements**
- [ ] **State Management**: Use Pydantic for better state validation and management on runtime
- [ ] **Logging Enhancement**: Structured logging with different levels
- [ ] **Memory Optimization**: Efficient document and embedding storage
- [ ] **Tool Call Reliability**: Improve LLM tool calling with better prompts or larger models

## 🏁 **Success Criteria**
- ✅ **Modular Architecture**: Clean separation with `src/agent/` structure
- ✅ **Type Safety**: Proper TypedDict state management
- ✅ **State-Based Initialization**: One-time setup with conditional routing
- ✅ **ContextSchema Integration**: System prompt loading from file
- ✅ **Tool Integration**: Function-based tools with state modification
- ✅ **Startup Validation**: Service and model availability checking
- ✅ **LangGraph Studio Compatible**: No custom checkpointer conflicts
- ✅ **Document Processing**: Complete vector store and querying implementation
- ✅ **Evaluation Framework**: Updated testing compatible with new structure
- ✅ **Performance Monitoring**: Comprehensive metrics and timing analysis
- ✅ **Bug Resolution**: Critical issues identified and resolved
- [ ] **Comprehensive Testing**: Unit and integration test coverage

## 🔧 **Technical Stack**
- **Framework**: LangGraph 0.6.6 with StateGraph
- **Architecture**: Modular design with `src/agent/` structure
- **Initialization**: State-based one-time setup with conditional routing
- **System Prompt**: ContextSchema with file-based loading
- **LLM**: Local Ollama llama3.2:3b model
- **Embeddings**: nomic-embed-text:latest (768 dimensions)
- **Vector Store**: LangChain InMemoryVectorStore
- **Document Processing**: LangChain DirectoryLoader + TextLoader
- **Language**: Python 3.13.5 with proper type hints
- **State Management**: TypedDict with Annotated fields
- **Studio Integration**: LangGraph Studio compatible (no custom checkpointer)
- **Development**: Local-only operation with Ollama backend

