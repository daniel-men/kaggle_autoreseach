from src.runner import load_function, run_function


def test_load_function_returns_named_function(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    solution_dir = tmp_path / "runs" / "demo" / "solution"
    solution_dir.mkdir(parents=True)
    (solution_dir / "solution.py").write_text(
        "def predict():\n    return 'ok'\n",
        encoding="utf-8",
    )

    predict = load_function("demo", "solution.py", "predict")

    assert predict() == "ok"


def test_run_function_executes_from_run_directory_and_restores_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    solution_dir = tmp_path / "runs" / "demo" / "solution"
    solution_dir.mkdir(parents=True)
    (solution_dir / "solution.py").write_text(
        "from pathlib import Path\n\n"
        "def where_am_i(suffix):\n"
        "    return Path.cwd().name + suffix\n",
        encoding="utf-8",
    )

    result = run_function("demo", "solution.py", "where_am_i", args=("!",))

    assert result == {"success": True, "result": "demo!"}
    assert tmp_path == tmp_path.cwd()


def test_run_function_returns_traceback_on_exception(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    solution_dir = tmp_path / "runs" / "demo" / "solution"
    solution_dir.mkdir(parents=True)
    (solution_dir / "solution.py").write_text(
        "def explode():\n    raise ValueError('boom')\n",
        encoding="utf-8",
    )

    result = run_function("demo", "solution.py", "explode", args=None)

    assert result["success"] is False
    assert "ValueError: boom" in result["traceback"]
