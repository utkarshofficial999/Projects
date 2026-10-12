# AgentEngine Documentation

Welcome to the official documentation for **AgentEngine**, a modular, production-ready Multi-Agent AI System.

## 📚 Table of Contents

- [Getting Started](tutorials/01_getting_started.md)
- [System Architecture](architecture.md)
- [Custom Tool Development](tutorials/02_custom_tools.md)
- [Multi-Agent Orchestration](tutorials/03_multi_agent.md)

## 🚀 Quick Start

```bash
# Install
pip install agent-engine

# Run CLI
agent-engine cli

# Start API Server
agent-engine serve --port 8000
```

## 🏗️ Core Concepts

1. **ReAct Engine**: The reasoning loop that drives Thought → Action → Observation cycles.
2. **Tool Registry**: A dynamic, type-safe registry for registering and executing tools.
3. **Memory Subsystem**: Dual-tier memory (Short-term sliding window + Long-term SQLite/Vector).
4. **Self-Reflection**: Critic and Healer agents that evaluate and correct agent outputs.
5. **Message Bus**: Asynchronous communication layer for multi-agent collaboration.
