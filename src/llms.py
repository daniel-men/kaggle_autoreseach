import os
from typing import Optional

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from src.custom_dcode_agent import create_cli_agent
from src.schemas.CodeResultModel import CodeResultModel
from src.utils import load_dotenv


def get_llm(provider: str, model: str, temperature: float = 0.):
    if provider == "ollama":
        llm = ChatOllama(
            model=model,
            temperature=0,
        )
    elif provider == "openai":
        llm =   ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="http://localhost:11434/v1",
            api_key="not-needed"
        )
    else:
        raise NotImplementedError()
    return llm

def get_claude_llm(temperature: float = 0., model: str | None = None):
    load_dotenv()
    return ChatAnthropic(
        model=model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        temperature=temperature,
    )

def print_stream_event(event: dict) -> None:
    """
    Pretty-print LangGraph/Deep Agent stream events.

    Works with stream_mode='updates', where each event is usually shaped like:
        {"node_name": {"messages": [...]}}
    """

    for node_name, node_update in event.items():
        if not isinstance(node_update, dict):
            continue

        messages = node_update.get("messages", [])

        for message in messages:
            if isinstance(message, AIMessage):
                if message.tool_calls:
                    print(f"\n[{node_name}] tool calls:")
                    for call in message.tool_calls:
                        print(f"  - {call['name']}({call.get('args', {})})")

                if message.content:
                    print(f"\n[{node_name}] assistant:")
                    print(message.content)

            elif isinstance(message, ToolMessage):
                print(f"\n[{node_name}] tool result: {message.name}")
                content = str(message.content)

                # Avoid dumping huge file contents.
                if len(content) > 2000:
                    content = content[:2000] + "\n... [truncated]"

                print(content)


def call_llm(prompt: str, context: str, system_prompt: Optional[str] = None, provider: str = "openai", model: str = "qwen3-coder:30b", temperature: float = 0.0, stream: bool = False):
    llm = get_llm(provider=provider, model=model, temperature=temperature)

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt + "\n" + context))

    if stream:
        chunks = None
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
            chunks = chunk if chunks is None else chunks + chunk
        print()
        return chunks

    return llm.invoke(messages)



def call_dcode(prompt: str, context: str, system_prompt: Optional[str] = None, cwd: Optional[str] = None, provider: str = "openai", model: str = "qwen3-coder:30b", temperature: float = 0.0, stream: bool = False):
    llm = get_llm(provider=provider, model=model, temperature=temperature)
    agent, backend = create_cli_agent(
        model=llm,
        assistant_id="dcode",
        system_prompt=system_prompt,
        cwd=cwd,
        interactive=False,
        # Keep this False for safety.
        # The agent will ask for approval before risky actions depending on setup.
        auto_approve=True,
        enable_shell=True,
        enable_memory=True,
        enable_skills=True,
        # Optional but recommended.
        shell_allow_list=[
            "python",
            "python3",
            "pytest",
            "pip",
        ],
        tools=[PythonREPLTool],
        response_format=CodeResultModel

    )

    """agent = create_deep_agent(
        model=llm,
        system_prompt=system_prompt,
        response_format=CodeResultModel,
    )"""

    input_state = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt + "\n" + context,
                }
            ]
        }

    if stream:
        results = []
        for event in agent.stream(input_state, stream_mode="updates"):
            print_stream_event(event)
            results.append(event)
        return results
    
    result = agent.invoke(input_state)

    if "structured_response" in result:
        return result["structured_response"]

    # The agent ended its turn without calling the structured-output tool
    # (small local models often just stop instead of calling it). Fall back
    # to Ollama's native JSON-schema constrained decoding over the
    # conversation so far, which is more reliable than tool-calling here.
    structured_model = llm.with_structured_output(
        CodeResultModel, method="json_schema"
    )
    return structured_model.invoke(
        result["messages"]
        + [
            {
                "role": "user",
                "content": (
                    "Based on the output above, produce the final "
                    "python code now."
                ),
            }
        ]
    )
