# Tutorial 3: Multi-Agent Orchestration

Learn how to set up and orchestrate multiple agents for complex tasks.

## 1. Supervisor Pattern

The Supervisor agent routes tasks to specialized worker agents. It acts as a central coordinator.

```python
from agent_engine.agents.supervisor import SupervisorAgent
from agent_engine.agents.base_agent import BaseAgent

# Define specialized agents
researcher = BaseAgent(
    name="Researcher",
    system_prompt="You are a research expert. Gather facts and data."
)

writer = BaseAgent(
    name="Writer",
    system_prompt="You are a professional writer. Create engaging content."
)

# Create Supervisor
supervisor = SupervisorAgent(
    name="Supervisor",
    agents=[researcher, writer],
    system_prompt="You coordinate tasks between Researcher and Writer."
)
```

## 2. Message Bus

Agents communicate via an asynchronous message bus. The Supervisor publishes tasks, and workers subscribe to relevant channels.

```python
from agent_engine.agents.message_bus import MessageBus

bus = MessageBus()
await bus.subscribe("research", researcher.handle_message)
await bus.subscribe("writing", writer.handle_message)
```

## 3. Self-Reflection Loop

The Critic agent evaluates outputs from worker agents. If errors are detected, the Healer agent suggests corrections.

```python
from agent_engine.reflection.critic import CriticAgent
from agent_engine.reflection.healer import HealerAgent

critic = CriticAgent()
healer = HealerAgent()

# Evaluate output
feedback = await critic.evaluate(output=writer_output)
if not feedback.is_valid:
    corrected = await healer.correct(output=writer_output, feedback=feedback)
```

## 4. Running the Multi-Agent System

```python
import asyncio

async def main():
    result = await supervisor.execute("Write a blog post about AI trends")
    print(result)

asyncio.run(main())
```

## 5. Monitoring & Logging

Enable structured logging to monitor agent interactions:

```env
AGENT_ENGINE_LOG_LEVEL=DEBUG
AGENT_ENGINE_LOG_FORMAT=json
```

Logs are written to `logs/agent_engine.log` and can be parsed with `jq` or other tools.
