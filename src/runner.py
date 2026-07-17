from pathlib import Path
import importlib.util
import os
import traceback


def load_function(slug: str, filename: str, fn_name: str):
    path = Path.cwd() / "runs" / slug / "solution" / filename
    if not path.exists():
        raise FileNotFoundError(f"Solution file not found: {path}")

    spec = importlib.util.spec_from_file_location("module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, fn_name):
        raise AttributeError(f"Module {path} does not define a f'{fn_name}' function")

    return getattr(module, fn_name)


def run_function(slug: str, filename: str, fn_name: str, args) -> dict:
    run_dir = Path.cwd() / "runs" / slug
    prev_cwd = os.getcwd()
    try:
        predict = load_function(slug=slug, filename=filename, fn_name=fn_name)
        os.chdir(run_dir)
        try:
            if args is None:
                result = predict()
            else:
                result = predict(*args)
        finally:
            os.chdir(prev_cwd)
    except Exception:
        return {"success": False, "traceback": traceback.format_exc()}

    return {"success": True, "result": result}




