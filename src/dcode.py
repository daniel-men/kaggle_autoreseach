import os

try:
    from deepagents_code.agent import create_cli_agent
except ImportError:  # pragma: no cover - optional dependency
    create_cli_agent = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None

try:
    from langchain_experimental.tools.python.tool import PythonREPLTool
except ImportError:  # pragma: no cover - optional dependency
    PythonREPLTool = None

try:
    from langchain_core.messages import AIMessage, ToolMessage
except ImportError:  # pragma: no cover - optional dependency
    AIMessage = ToolMessage = None

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




def call_dcode(slug: str, prompt: str, context: str, stream: bool = False):
    model = ChatOpenAI(
        #model="deepseek-coder-v2",  # or 32b if your machine can run it
        model="qwen3-coder:30b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        #temperature=0.,
        #extra_body={"options": {"num_ctx": 32768}},
    )

    model_output = model.invoke(
        prompt + "\n" + context
    )
    return model_output
    """model = ChatOpenAI(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        base_url="http://localhost:8080/v1",
        api_key="token",
        temperature=0,
    )"""

    agent, backend = create_cli_agent(
        model=model,
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
                    "content": prompt + "\n" + context,
                }
            ]
        }
    
    if stream:
        for event in agent.stream(input_state, stream_mode="updates"):
            print_stream_event(event)

        result = event
    else:
        result = agent.invoke(input_state)

    return result

def implement_metric(slug: str, metric: str):
    prompt = (
        f"Implement the metric {metric} in python. "
        "It should take two arguments, y_true and y_pred, and return a single numeric value. "
        "It is okay to import the metric from a library if it exists. "
        "Return the code only, wrapped in a python code block, and label the file as inferred_metrics.py."
    )
    code_result = call_dcode(slug=slug, prompt=prompt, context="", stream=False)
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
        slug=slug,
        prompt=f"""
        You are a skillful senior machine learning engineer.

        Read the attached experiment plan (JSON) and implement it as a Python script.

        Data:
        - Load {os.getcwd()}/runs/{slug}/data/preprocessed_data.csv. It has already been cleaned, imputed, and
          encoded (categorical variables are numeric). Do not re-impute, re-encode, scale,
          or otherwise re-preprocess it.
        - The target column name is given by `likely_target_column` in the context. Split
          it from the features before training.

        Implementation:
        - Implement the experiment's `method` from the context exactly.
        - Split the dataframe into training and test sets.
        - Set random seeds wherever applicable so results are reproducible.
        - Install any missing packages with pip if needed.
        - Do not write placeholder, dummy, or scaffold code. Implement the full working
          machine learning pipeline.

        Output:
        - Only change or create the file solution.py in the solution directory.
        - Do not import or compute any metrics on your own.
        - Implement a predict() function with no arguments that runs the full pipeline:
            load data, split into train/test, train the model, and generate test-set predictions.
        - The predict() function must return exactly two objects: y_true and y_pred.
        - y_pred must be the model predictions on the test set and y_true must be the corresponding
          ground truth values from the test set.
        - Do not return additional values, dictionaries, or printed output.
        """,
        context=context,
        stream=stream
    )

def repair_code(slug: str, file_path: str, traceback: str, context: str = "", stream: bool = False):
    current_code = get_file_content(path=file_path)
    
    return call_dcode(
        slug=slug,
        prompt=f"""
        You are a skillful senior machine learning engineer acting as a debugger.

        Running {file_path} raised the error shown below (traceback). Read the
        file, understand the root cause, and fix it.

        Code:
        {current_code}

        Traceback:
        {traceback}

        Requirements:
        - Make sure the indentation of the file is correct.
        - Make the minimal change needed to fix the error. Preserve the existing
          approach, function signatures, and return contract (e.g. predict()
          must still return a dict of metrics) unless they are the cause of the
          bug.
        - Do not rewrite the solution from scratch.
        - Do not implement placeholder or dummy functions
        """,
        context=context,
        stream=stream,
    )


def implement_preprocessing(slug: str, context: str, stream: bool = False):
    return call_dcode(
        slug=slug,
        prompt="""
        You are a skillful data engineer.

        Read the attached context and write a preprocessing pipeline.

        Requirements:
        - Identify the target column from the context and keep it unchanged in the output
          (do not encode, scale, or impute the target).
        - Impute missing values using sensible defaults per dtype (e.g. median for numeric
          columns, most frequent value or a dedicated "missing" category for categorical
          columns).
        - Encode categorical variables numerically (e.g. one-hot or ordinal encoding, as appropriate
          for the number of categories).
        - Make sure all columns can be used for downstream machine learning tasks.
        - Drop columns that are pure identifiers, constant, or otherwise not useful for
          modeling, but keep the target column.
        - Do not perform feature engineering beyond cleaning, imputation, and encoding.
        - Implement a preprocess() function that performs these steps. It takes as only input the path to csv and returns the preprocessed dataframe.
        - Implement the actual code and not a scaffold.
        - Briefly report which columns were dropped, imputed, or encoded, and confirm
          that data/preprocessed_data.csv was written successfully.
        """,
        context=context,
        stream=stream      
        )