try:
    from langchain_core.messages import AIMessage, ToolMessage
except ImportError:  # pragma: no cover - optional dependency
    AIMessage = ToolMessage = None

from src.llms import call_dcode
from src.prompts import (
    EXPERIMENT_IMPLEMENTATION_PROMPT,
    IMPLEMENT_METRIC_PROMPT,
    IMPLEMENT_PREPROCESSING_CODE,
    REPAIR_CODE_PROMPT,
)
from src.utils import get_file_content, write_python_code_to_file


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

# TODO move into metric_graph
def implement_metric(slug: str, metric: str):

    code_result = call_dcode(
        prompt=IMPLEMENT_METRIC_PROMPT(metric=metric),
        context="",
    )
    content = getattr(code_result, "content", code_result)
    if isinstance(content, dict):
        content = content.get("content") or content.get("text") or str(content)
    else:
        content = str(content)

    write_python_code_to_file(
        content=content,
        filename="inferred_metrics.py",
        slug=slug,
        append=True,
    )
    return code_result


def ask_for_code(slug: str, context: str, stream: bool = False):
    return call_dcode(
        prompt=EXPERIMENT_IMPLEMENTATION_PROMPT(slug=slug),
        context=context,
    )


def repair_code(
    slug: str, file_path: str, traceback: str, context: str = "", stream: bool = False
):
    current_code = get_file_content(path=file_path)

    return call_dcode(
        prompt=REPAIR_CODE_PROMPT(
            file_path=file_path, current_code=current_code, traceback=traceback
        ),
        context=context,
    )


def implement_preprocessing(slug: str, context: str, stream: bool = False):
    return call_dcode(
        prompt=IMPLEMENT_PREPROCESSING_CODE, context=context
    )
