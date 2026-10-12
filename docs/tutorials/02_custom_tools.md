# Tutorial 2: Custom Tool Development

Learn how to create and register custom tools for your agents.

## 1. Defining a Tool

Tools are Python classes that inherit from `BaseTool`. Each tool must define:
- `name`: Unique identifier.
- `description`: Human-readable description for the LLM.
- `parameters`: Pydantic model defining input schema.
- `execute`: Async method that performs the tool's logic.

### Example: Weather Tool

```python
from pydantic import BaseModel, Field
from agent_engine.tools.base_tool import BaseTool

class WeatherParams(BaseModel):
    city: str = Field(..., description="City name")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather for a city"
    parameters: type[BaseModel] = WeatherParams

    async def execute(self, params: WeatherParams) -> str:
        # Simulate API call
        return f"Sunny, 25°C in {params.city}"
```

## 2. Registering the Tool

Use the `ToolRegistry` to register your tool:

```python
from agent_engine.tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register(WeatherTool())
```

## 3. Using the Tool in an Agent

Agents automatically discover registered tools. No additional configuration is needed.

```python
from agent_engine.agents.base_agent import BaseAgent

agent = BaseAgent(
    name="WeatherAgent",
    system_prompt="You are a helpful weather assistant."
)
```

## 4. Sandboxing

Tools are executed in a sandboxed environment. Ensure your tool:
- Does not modify global state.
- Handles exceptions gracefully.
- Returns a string or JSON-serializable object.

## 5. Testing Your Tool

Write unit tests for your tool:

```python
import pytest
from agent_engine.tools.builtin.weather_tool import WeatherTool

@pytest.mark.asyncio
async def test_weather_tool():
    tool = WeatherTool()
    result = await tool.execute(city="Paris")
    assert "Sunny" in result
```
