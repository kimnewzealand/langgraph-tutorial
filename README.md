# Langgraph Tutorial

This repository demonstrates a LangGraph-powered AI agent that processes policy documents and answers user queries about data classification and company policies.

The agent uses a reactive workflow pattern with **dual-tool integration**, allowing it to both retrieve relevant information from documents AND generate chronologically ordered action plans with compliance deadlines. It supports multiple LLM providers (Ollama, Anthropic) with verbose execution tracking that shows step-by-step agent decision-making and tool usage.

The system implements best practices for agent architecture including proper state management, error handling, configurable execution modes, and intelligent tool orchestration for comprehensive policy compliance management.




## LangGraph vs Traditional Agentic RAG

This implementation differs from traditional agentic RAG systems in several key ways:

| Aspect | This LangGraph Repo | Traditional Agentic RAG |
|--------|-------------------|------------------------|
| **Retrieval** | File-based, returns full document | Vector similarity, returns relevant chunks |
| **Flow Control** | Explicit graph with conditional edges | Linear retrieve → generate pipeline |
| **State Management** | Persistent conversation state | Stateless or simple context |
| **Tool Integration** | Dual-tool orchestration (retrieve + action planning) | Primarily retrieval-focused |
| **Decision Making** | Agent decides when/how to use tools | Automatic retrieval for every query |
| **Workflow Visibility** | Full execution tracing with verbose mode | Black-box retrieval + generation |

### Key Advantages of This Approach:

- ✅ **Agent autonomy** - decides when to retrieve information
- ✅ **Dual-tool capability** - retrieves policy content AND generates action plans
- ✅ **Workflow transparency** - you can see every decision step
- ✅ **State persistence** - maintains conversation context
- ✅ **Conditional logic** - different paths for different query types
- ✅ **Compliance management** - automatically extracts deadlines and creates prioritized task lists

This makes it more of a **"workflow-driven agent with tools"** rather than a pure RAG system, providing greater flexibility and control over the information retrieval and response generation process.

## 🛠️ **Key Features**

### **Dual-Tool Architecture**
The agent employs two specialized tools that work together:

#### **1. `retrieve_content` Tool**
- **Purpose**: Retrieves policy information from documents
- **Input**: User query about policies
- **Output**: Relevant policy sections and content
- **Usage**: Provides foundational policy knowledge

#### **2. `create_action_plan` Tool** ⭐ **NEW**
- **Purpose**: Generates chronologically ordered compliance action plans
- **Input**: Policy-related query
- **Output**: Prioritized list of actionable items with deadlines
- **Features**:
  - 📅 **Smart deadline detection** (e.g., "within 48 hours", "quarterly by 30th")
  - 🗓️ **Calendar calculations** from today's date
  - 🚨 **Urgency classification** (URGENT, SOON, PLANNED, OVERDUE)
  - 📋 **Formatted action items** with due dates and policy sections
  - ⏰ **Real-time compliance tracking**

### **Enhanced Query Processing**
Ask comprehensive questions to trigger both tools:
```
"What are the data classification levels? Also create an action plan with deadlines for compliance requirements."
```

### **Intelligent Tool Orchestration**
- **Automatic tool selection** based on query content
- **Sequential tool execution** for comprehensive responses
- **Context-aware processing** with conversation state management

## Benefits of Langgraph

>LangGraph is a framework that allows you to build production-ready applications by giving you control tools over the flow of your agent.


>LangGraph is particularly valuable when you need Control over your applications. It gives you the tools to build an application that follows a predictable process while still leveraging the power of LLMs.

>At its core, LangGraph uses a directed graph structure to define the flow of your application:

- Nodes represent individual processing steps (like calling an LLM, using a tool, or making a decision).
- Edges define the possible transitions between steps.
- State is user defined and maintained and passed between nodes during execution. When deciding which node to target next, this is the current state that we look at.

See tutorial https://huggingface.co/learn/agents-course/en/unit2/langgraph/introduction

## 🚀 **Performance & Architecture**

### **Ollama Integration**
- **Local LLM execution** with `qwen2:7b` model (7 billion parameters)
- **Optimized for speed** with reduced context windows and response limits
- **Model warm-up** and health checking for reliable performance
- **Automatic status monitoring** and diagnostics

### **Smart System Prompts**
- **Proactive tool usage** - encourages comprehensive policy analysis
- **Concise responses** - prevents document content regurgitation
- **Compliance-focused** - emphasizes actionable deadlines and requirements

### **Development Tools**
- **Augment Code** as the AI coding copilot
- **Verbose execution mode** for debugging and workflow visualization
- **Mermaid diagram generation** for workflow understanding
- **Comprehensive error handling** and logging

## 📊 **Sample Output**

### **Enhanced Query with Action Plan**
Query: *"What are the levels of data classification? Also create an action plan with deadlines for compliance requirements."*

**Policy Information (from `retrieve_content`):**
```
There are three levels of data classification: Public, Internal, and Confidential.

- **Public Data**: Can be shared externally
- **Internal Data**: Restricted to company use only
- **Confidential Data**: Requires special handling including encryption
```

**Compliance Action Plan (from `create_action_plan`):**
```
🔴 URGENT | Due: 2025-09-09 (+1 days) | Section 2.1: LLM usage approval required within 48 hours
🔴 URGENT | Due: 2025-09-10 (+2 days) | Section 3.3: AI tool approval needed within 72 hours
🔴 URGENT | Due: 2025-09-10 (+2 days) | Section 4.4: Access requests must be approved within 3 business days
🟡 SOON | Due: 2025-10-07 (+29 days) | Section 5.2: Address non-compliance issues within 30 days
🟢 PLANNED | Due: 2025-12-31 (+114 days) | Section 5.3: Complete annual security training by Dec 31st
```

### **Verbose Execution Tracking**
```
🔹 Step: ['agent'] - Agent analyzing query and deciding on tools
🔹 Step: ['tools'] - Executing retrieve_content and create_action_plan
🔹 Step: ['agent'] - Synthesizing results into comprehensive response
```


## Setup

Follow these steps to set up the environment:

Use Python 3.13

1. **Create a virtual environment**:
    ```bash
    py -m venv .venv
    ```

2. **Activate the virtual environment**:
    - On Windows:
      ```bash
      .venv\Scripts\Activate
      ```
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt    
    ```

4. **Create a `.env` file**: Create a file named `.env` in the root directory of your project. This file will contain your HuggingFace API key. Add the following line to the file:
    
    ```bash
    ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    ```
    
    **To get a Anthropic API key:**
    1. Go to [Anthropic](https://anthropic.co/)
    2. Create an account or sign in
    3. Go to your profile settings
    4. Navigate to "Access Tokens"
    5. Create a new token with "read" permissions
    6. Copy the token and paste it in your `.env` file

5. Download Ollama
   https://python.langchain.com/docs/integrations/chat/ollama/

6. **Run the agent**:
    ```bash
    py main.py
    ```

## 💡 **Usage Tips**

### **Getting Comprehensive Results**
To trigger both tools and get complete policy analysis with action plans, use queries like:
- *"What are the data classification levels and what compliance actions are needed?"*
- *"Explain the LLM usage policy and create an action plan for deadlines"*
- *"Show me access control requirements and generate a compliance timeline"*

### **Tool-Specific Queries**
- **Policy information only**: *"What are the data classification levels?"*
- **Action plan only**: *"Create an action plan for policy compliance deadlines"*

### **Verbose Mode**
The application runs in verbose mode by default, showing:
- Step-by-step tool execution
- Real-time decision making
- Tool output and agent responses
- Performance metrics and timing
  
## License

This project is for educational purposes only.
