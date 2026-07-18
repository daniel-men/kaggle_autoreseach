import os
from typing import Optional

from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

from src.utils import load_dotenv
from deepagents_code.agent import create_cli_agent


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

def call_llm(prompt: str, context: str, provider: str = "openai", model: str = "qwen3-coder:30b", temperature: float = 0.0):
    llm = get_llm(provider=provider, model=model, temperature=temperature)

    return llm.invoke(prompt, context=context)

def call_dcode(prompt: str, context: str, provider: str = "openai", model: str = "qwen3-coder:30b", temperature: float = 0.0):
    llm = get_llm(provider=provider, model=model, temperature=temperature)
    agent, backend = create_cli_agent(
        model=llm,
        assistant_id="dcode",
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
        
    )
    input_state = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "context": context,
                }
            ]
        }
    return agent.invoke(input_state)
