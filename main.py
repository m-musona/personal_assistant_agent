from tools.tool_registry import ToolRegistry
from agent.agent import Agent

agent = Agent(ToolRegistry())
print(agent.chat("Say hello in one sentence."))
