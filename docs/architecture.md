# System Architecture

This document details the internal architecture of AgentEngine, including data flow, component interactions, and design patterns.

## 🏛️ High-Level Diagram

```mermaid
graph TD
    User([User / API Request]) --> Supervisor[Multi-Agent Supervisor / Router]
    
    subgraph AgentEngine Core Runtime
        Supervisor --> AgentA[Task Worker Agent]
        Supervisor --> AgentB[Research & Analysis Agent]
        Supervisor --> AgentC[Critic / Evaluator Agent]
        
        AgentA <--> ReActLoop[ReAct Reasoning Loop]
        AgentB <--> ReActLoop
        
        ReActLoop <--> ToolRegistry[Dynamic Tool Registry & Sandbox]
        ReActLoop <--> MemoryManager[Memory Subsystem]
        
        MemoryManager --> ShortTerm[Short-Term Context Buffer]
        MemoryManager --> LongTerm[Long-Term SQLite / Vector Store]
        
        AgentC --> SelfReflection[Self-Reflection & Healing Loop]
        SelfReflection -.->|Refined Plan / Feedback| Supervisor
    end
    
    subgraph Delivery & Exposure
        Supervisor --> CLI[Interactive Rich CLI]
        Supervisor --> API[FastAPI Async REST Endpoints]
    end
```

## 🔄 ReAct Execution Loop

The core of AgentEngine is the **ReAct (Reasoning + Acting)** loop. Each agent instance runs this loop until a final answer is produced or a termination condition is met.

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant R as ReAct Engine
    participant T as Tool Registry
    participant M as Memory

    U->>A: Task / Query
    A->>M: Retrieve Context
    M-->>A: Contextual Data
    A->>R: Initialize Loop
    
    loop ReAct Cycle
        R->>A: Generate Thought
        A->>R: Thought
        R->>A: Select Action & Tool
        A->>T: Execute Tool
        T-->>A: Observation
        A->>M: Update Memory
        R->>R: Check Termination
    end
    
    A-->>U: Final Answer
```

## 💾 Memory Flow

AgentEngine uses a dual-tier memory system:

1. **Short-Term Memory**: A sliding window buffer that maintains the most recent `N` interactions. This is used for immediate context awareness.
2. **Long-Term Memory**: A persistent SQLite or Vector Store that retains semantic embeddings and historical state across sessions.

```mermaid
graph LR
    A[Agent] --> STM[Short-Term Memory]
    A --> LTM[Long-Term Memory]
    STM -->|Flush| LTM
    LTM -->|Retrieve| A
```

## 👥 Multi-Agent Communication

Agents communicate via an asynchronous **Message Bus**. The Supervisor routes tasks to specialized agents and aggregates results.

```mermaid
sequenceDiagram
    participant S as Supervisor
    participant Bus as Message Bus
    participant W1 as Worker Agent 1
    participant W2 as Worker Agent 2
    participant C as Critic Agent

    S->>Bus: Publish Task
    Bus->>W1: Route Task
    Bus->>W2: Route Task
    W1->>Bus: Publish Result
    W2->>Bus: Publish Result
    Bus->>S: Aggregate Results
    S->>C: Request Evaluation
    C-->>S: Feedback / Approval
```

## 🛠️ Tool Execution & Sandboxing

Tools are executed in a sandboxed environment to prevent side effects. The Tool Registry validates inputs using Pydantic schemas before execution.

```mermaid
graph TD
    A[Agent] --> TR[Tool Registry]
    TR --> V[Pydantic Validation]
    V --> SB[Sandbox Executor]
    SB --> T[Tool Implementation]
    T --> SB
    SB --> A
```
