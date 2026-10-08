"""
agent_engine.agents
===================

Agent implementations and orchestration logic.

This module contains:
- Base agent class with common functionality
- Specialized agent types (Researcher, Coder, Reviewer)
- SupervisorAgent for task delegation and orchestration
- MessageBus for asynchronous inter-agent communication
- Handoff logic for dynamic control transfer
"""

from agent_engine.agents.base_agent import BaseAgent, AgentState, AgentStatus
from agent_engine.agents.message_bus import MessageBus, Message, MessageType
from agent_engine.agents.supervisor import SupervisorAgent, TaskAssignment, HandoffRequest
from agent_engine.agents.specialized.researcher import ResearcherAgent
from agent_engine.agents.specialized.coder import CoderAgent
from agent_engine.agents.specialized.reviewer import ReviewerAgent

__all__ = [
    "BaseAgent",
    "AgentState",
    "AgentStatus",
    "MessageBus",
    "Message",
    "MessageType",
    "SupervisorAgent",
    "TaskAssignment",
    "HandoffRequest",
    "ResearcherAgent",
    "CoderAgent",
    "ReviewerAgent",
]
