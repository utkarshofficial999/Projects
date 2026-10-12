# Tutorial 1: Getting Started

This guide walks you through installing AgentEngine and running your first autonomous agent.

## 1. Installation

### Prerequisites
- Python 3.10+
- `pip` or `poetry`

### Install from PyPI
```bash
pip install agent-engine
```

### Install from Source
```bash
git clone https://github.com/your-repo/agent-engine.git
cd agent-engine
pip install -e .
```

## 2. Configuration

AgentEngine uses environment variables for configuration. Create a `.env` file in your project root:

```env
OPENAI_API_KEY=sk-your-key-here
AGENT_ENGINE_LOG_LEVEL=INFO
AGENT_ENGINE_MEMORY_DB_PATH=./state/memory.db
```

## 3. Running the CLI

The CLI provides an interactive REPL for testing agents.

```bash
agent-engine cli
```

Example session:
```
> What is the capital of France?
[Thought] I need to search for the capital of France.
[Action] search_tool(query="capital of France")
[Observation] Paris is the capital of France.
[Answer] The capital of France is Paris.
```

## 4. Starting the API Server

For programmatic access, start the FastAPI server:

```bash
agent-engine serve --host 0.0.0.0 --port 8000
```

### API Endpoints

- `POST /chat`: Send a message to the agent.
- `GET /health`: Check server health.
- `POST /tools/register`: Register a new tool at runtime.

Example request:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, AgentEngine!"}'
```

## 5. Next Steps

- [Custom Tool Development](02_custom_tools.md)
- [Multi-Agent Orchestration](03_multi_agent.md)
