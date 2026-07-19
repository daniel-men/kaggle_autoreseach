from src.llms import call_dcode, call_llm
from src.prompts import (
    EXPERIMENT_IMPLEMENTATION_PROMPT,
    IMPLEMENT_METRIC_PROMPT,
    IMPLEMENT_PREPROCESSING_CODE,
    REPAIR_CODE_PROMPT,
)
from src.utils import get_file_content, write_python_code_to_file


# TODO move into metric_graph
def implement_metric(slug: str, metric: str):
    system_prompt, instruction = IMPLEMENT_METRIC_PROMPT(metric=metric)

    code_result = call_llm(
        prompt=instruction,
        system_prompt=system_prompt,
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
    system_prompt, instruction = EXPERIMENT_IMPLEMENTATION_PROMPT(slug=slug)
    return call_llm(
        prompt=instruction,
        system_prompt=system_prompt,
        context=context,
        stream=stream,
    )


def repair_code(
    slug: str, file_path: str, traceback: str, context: str = "", stream: bool = False
):
    current_code = get_file_content(path=file_path)
    system_prompt, instruction = REPAIR_CODE_PROMPT(
        file_path=file_path, current_code=current_code, traceback=traceback
    )

    return call_llm(
        prompt=instruction,
        system_prompt=system_prompt,
        context=context,
        stream=stream,
    )


def implement_preprocessing(slug: str, context: str, stream: bool = False):
    system_prompt, instruction = IMPLEMENT_PREPROCESSING_CODE
    return call_llm(
        prompt=instruction,
        system_prompt=system_prompt,
        context=context,
        stream=stream,
    )
