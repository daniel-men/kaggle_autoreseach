from langchain_community.llms.vllm import VLLM
import multiprocessing
import os

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp
from langchain_anthropic import ChatAnthropic

from src.utils import load_dotenv

def get_vllm(temperature: float = 0.):
    llm = VLLM(
        model="/Users/Mensing/Downloads/Qwen3.5-9B-Q4_K_M.gguf",
        trust_remote_code=True,  # mandatory for hf models
        
        temperature=temperature,
       
    )
    return llm

def get_llm_llama(temperature: float = 0.):
    return ChatLlamaCpp(
        temperature=temperature,
        model_path="/Users/Mensing/Downloads/Qwen3.5-9B-Q4_K_M.gguf",
        n_gpu_layers=999,
        n_threads=multiprocessing.cpu_count() - 1,
        repeat_penalty=1.5,
        top_p=0.5
    )

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
            base_url="http://localhost:8080/v1",
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