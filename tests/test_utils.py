from pathlib import Path

from src.dcode import implement_metric
from src.utils import write_python_code_to_file


def test_write_python_code_to_file_supports_multiple_named_blocks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    content = """
```python
# FILE: solution.py
print('solution')
```

```python
# FILE: inferred_metrics.py

def accuracy(y_true, y_pred):
    return 1.0
```
"""

    write_python_code_to_file(content=content, filename="solution.py", slug="demo")

    solution_path = tmp_path / "runs" / "demo" / "solution" / "solution.py"
    metrics_path = tmp_path / "runs" / "demo" / "solution" / "inferred_metrics.py"

    assert solution_path.exists()
    assert metrics_path.exists()
    assert "print('solution')" in solution_path.read_text(encoding="utf-8")
    assert "def accuracy" in metrics_path.read_text(encoding="utf-8")


def test_implement_metric_writes_metrics_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_call_dcode(slug, prompt, context, stream=False):
        return """
```python
# FILE: inferred_metrics.py

def accuracy(y_true, y_pred):
    return 1.0
```
"""

    monkeypatch.setattr("src.dcode.call_dcode", fake_call_dcode)

    implement_metric(slug="demo", metric="accuracy")

    metrics_path = tmp_path / "runs" / "demo" / "solution" / "inferred_metrics.py"
    assert metrics_path.exists()
    assert "def accuracy" in metrics_path.read_text(encoding="utf-8")


def test_implement_metric_appends_to_existing_metrics_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    responses = iter([
        """
```python
# FILE: inferred_metrics.py

def accuracy(y_true, y_pred):
    return 1.0
```
""",
        """
```python
# FILE: inferred_metrics.py

def f1_score(y_true, y_pred):
    return 0.5
```
""",
    ])

    def fake_call_dcode(slug, prompt, context, stream=False):
        return next(responses)

    monkeypatch.setattr("src.dcode.call_dcode", fake_call_dcode)

    implement_metric(slug="demo", metric="accuracy")
    implement_metric(slug="demo", metric="f1_score")

    metrics_path = tmp_path / "runs" / "demo" / "solution" / "inferred_metrics.py"
    content = metrics_path.read_text(encoding="utf-8")
    assert "def accuracy" in content
    assert "def f1_score" in content
