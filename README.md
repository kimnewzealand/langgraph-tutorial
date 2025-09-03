# Langgraph Tutorial

>LangGraph is a framework that allows you to build production-ready applications by giving you control tools over the flow of your agent.

## Benefits of Langgraph

>LangGraph is particularly valuable when you need Control over your applications. It gives you the tools to build an application that follows a predictable process while still leveraging the power of LLMs.

>At its core, LangGraph uses a directed graph structure to define the flow of your application:

- Nodes represent individual processing steps (like calling an LLM, using a tool, or making a decision).
- Edges define the possible transitions between steps.
- State is user defined and maintained and passed between nodes during execution. When deciding which node to target next, this is the current state that we look at.

See tutorial https://huggingface.co/learn/agents-course/en/unit2/langgraph/introduction

Notes on repo:
- Used Augment Code as the code copilot.
- Use Ollama for local model execution.

Sample output:

```
================================== Ai Message ==================================

There are three main levels of data classification: Public, Internal, and Confidential.

- **Public Data**: Can be shared externally.
- **Internal Data**: Restricted to company use only.
- **Confidential Data**: Requires special handling including encryption.

These classifications guide how data should be handled based on sensitivity and access rights.
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

6. Run the agent:
    ```bash
    py main.py
    ```
  
## License

This project is for educational purposes only.
