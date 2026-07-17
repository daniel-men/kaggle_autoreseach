from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Protocol

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field

from src.utils import slug_from_url, _workspace_for_challenge
from deepagents import create_deep_agent

RENDER_ATTEMPTS = [15000, 30000, 60000]
BAD_RENDER_MARKERS = (
    "Something went wrong and this page crashed",
    "Loading CSS chunk",
    "TypeError: Failed to fetch",
    "Unable to load",
)
EXPECTED_RENDER_MARKERS = {
    "description": ("The Challenge", "How Kaggle", "Submission File Format"),
    "evaluation": ("Goal", "Metric", "Submission File Format"),
    "data": ("Dataset Description", "Data Dictionary", "Variable Notes"),
}
COMBINED_DOCS_CLEANUP_SYSTEM = (
    "You clean Kaggle competition documentation for an automated modeling agent. "
    "Keep only factual information needed to understand the prediction task, the evaluation/submission "
    "requirements, and the dataset/data dictionary. Remove navigation text, duplicated content, "
    "rendering artifacts, metadata panels, ads, empty sections, and UI chrome. Do not invent facts. "
    "Return only clean Markdown."
)


class RenderAttempt(BaseModel):
    virtual_time_budget_ms: int
    returncode: int | None = None
    stdout_chars: int
    stderr_chars: int
    content_score: int
    has_bad_marker: bool


class RenderedPage(BaseModel):
    name: str
    url: str
    renderer: str | None = None
    error: str | None = None
    warning: str | None = None
    html_path: str | None = None
    text_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    returncode: int | None = None
    stderr_chars: int | None = None
    render_attempts: list[RenderAttempt] = Field(default_factory=list)
    content_score: int | None = None
    text_chars: int | None = None
    text_preserved_from_cache: str | None = None
    text_repaired_from: str | None = None


class ChallengeDocsArtifact(BaseModel):
    competition_slug: str
    url: str
    rendered_pages: list[RenderedPage] = Field(default_factory=list)
    combined_text_path: str
    combined_text_raw_path: str | None = None
    combined_text_cleaned_by_llm: bool | None = None
    combined_text_chars: int | None = None
    combined_text_cleanup_error: str | None = None
    combined_text_cleanup_response_path: str | None = None


class ChallengeDocsState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace: Path
    challenge: str
    docs_dir: Path | None = None
    competition_slug: str | None = None
    url: str | None = None
    rendered_pages: list[RenderedPage] = Field(default_factory=list)
    artifact: ChallengeDocsArtifact | None = None


def write_challenge_docs(
    workspace: Path,
    challenge: str,
) -> dict[str, Any]:
    if (workspace / "reports" / "challenge_docs" / "combined.md").exists():
        return
    graph = _build_challenge_docs_graph()
    state = graph.invoke(
        ChallengeDocsState(workspace=workspace, challenge=challenge)
    )
    final_state = _coerce_docs_state(state)
    if final_state.artifact is None:
        raise RuntimeError(
            "Challenge docs graph completed without producing an artifact."
        )
    return final_state.artifact.model_dump(mode="json", exclude_none=True)


def _build_challenge_docs_graph() -> Any:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is required for kaggle_docs_scrape. Install it with `pip install langgraph`."
        ) from exc

    graph = StateGraph(ChallengeDocsState)
    graph.add_node("prepare", _prepare_docs_state)
    graph.add_node("render_pages", _render_pages_node)
    graph.add_node("build_artifact", _build_artifact_node)
    graph.add_node("write_combined", _write_combined_node)
    graph.add_node("cleanup_with_llm", _cleanup_with_llm_node)
    graph.add_node("write_metadata", _write_metadata_node)

    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "render_pages")
    graph.add_edge("render_pages", "build_artifact")
    graph.add_edge("build_artifact", "write_combined")
    graph.add_edge("write_combined", "cleanup_with_llm")
    graph.add_edge("cleanup_with_llm", "write_metadata")
    graph.add_edge("write_metadata", END)
    return graph.compile()


def _coerce_docs_state(
    state: ChallengeDocsState | dict[str, Any],
) -> ChallengeDocsState:
    if isinstance(state, ChallengeDocsState):
        return state
    return ChallengeDocsState.model_validate(state)


def _prepare_docs_state(state: ChallengeDocsState) -> dict[str, Any]:
    load_dotenv()
    docs_dir = state.workspace / "reports" / "challenge_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    slug = slug_from_url(state.challenge)
    url = (
        state.challenge
        if "kaggle.com" in state.challenge
        else f"https://www.kaggle.com/competitions/{slug}"
    )
    return {"docs_dir": docs_dir, "competition_slug": slug, "url": url}


def _render_pages_node(state: ChallengeDocsState) -> dict[str, Any]:
    if state.docs_dir is None or state.url is None:
        raise RuntimeError("Challenge docs state was not prepared before rendering.")
    records = _render_kaggle_pages(state.url, state.docs_dir)
    return {
        "rendered_pages": [RenderedPage.model_validate(record) for record in records]
    }


def _build_artifact_node(state: ChallengeDocsState) -> dict[str, Any]:
    if state.docs_dir is None or state.competition_slug is None or state.url is None:
        raise RuntimeError(
            "Challenge docs state was not prepared before artifact creation."
        )
    artifact = ChallengeDocsArtifact(
        competition_slug=state.competition_slug,
        url=state.url,
        rendered_pages=state.rendered_pages,
        combined_text_path=str(state.docs_dir / "combined.md"),
    )
    return {"artifact": artifact}


def _write_combined_node(state: ChallengeDocsState) -> dict[str, Any]:
    if state.docs_dir is None or state.artifact is None:
        raise RuntimeError(
            "Challenge docs artifact was not ready before combined docs writing."
        )
    _write_combined_docs(
        state.docs_dir, state.artifact.model_dump(mode="json", exclude_none=True)
    )
    return {}


def _cleanup_with_llm_node(state: ChallengeDocsState) -> dict[str, Any]:
    if state.docs_dir is None or state.artifact is None:
        raise RuntimeError("Challenge docs artifact was not ready before LLM cleanup.")
    
    artifact = state.artifact.model_dump(mode="json", exclude_none=True)
    _postprocess_combined_docs_with_llm(state.docs_dir, artifact)
    return {"artifact": ChallengeDocsArtifact.model_validate(artifact)}


def _write_metadata_node(state: ChallengeDocsState) -> dict[str, Any]:
    if state.docs_dir is None or state.artifact is None:
        raise RuntimeError(
            "Challenge docs artifact was not ready before metadata writing."
        )
    artifact = state.artifact.model_dump(mode="json", exclude_none=True)
    (state.docs_dir / "challenge_docs.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    return {}


def _render_kaggle_pages(challenge_url: str, docs_dir: Path) -> list[dict[str, Any]]:
    chrome = _find_chrome()
    base = challenge_url.rstrip("/")
    pages = [
        ("description", f"{base}/overview/description"),
        ("evaluation", f"{base}/overview/evaluation"),
        ("data", f"{base}/data"),
    ]
    render_dir = docs_dir / "rendered_pages"
    render_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, url in pages:
        record: dict[str, Any] = {"name": name, "url": url, "renderer": chrome}
        if not chrome:
            record["error"] = "Chrome executable not found; rendered scrape skipped."
            out.append(record)
            continue
        try:
            html, render_info = _render_kaggle_dom(chrome, url, name)
            record.update(render_info)
            record["html_path"] = str(render_dir / f"{name}.html")
            (render_dir / f"{name}.html").write_text(
                html, encoding="utf-8", errors="replace"
            )
            record["metadata"] = _html_metadata(html)
            text = _extract_rendered_page_text(name, html)
            record["text_path"] = str(render_dir / f"{name}.txt")
            if _has_expected_content(name, text):
                record["text_chars"] = len(text)
                (render_dir / f"{name}.txt").write_text(text, encoding="utf-8")
            else:
                existing = render_dir / f"{name}.txt"
                record["warning"] = (
                    f"Rendered page did not contain expected content for {name}."
                )
                cached_text = (
                    existing.read_text(encoding="utf-8", errors="replace")
                    if existing.exists()
                    else ""
                )
                if _has_expected_content(name, cached_text):
                    record["text_chars"] = len(cached_text)
                    record["text_preserved_from_cache"] = str(existing)
                else:
                    record["text_chars"] = len(text)
                    (render_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        except Exception as exc:
            record["error"] = str(exc)
        out.append(record)
    _repair_rendered_description(render_dir, out)
    _repair_rendered_evaluation(render_dir, out)
    return out


def _render_kaggle_dom(
    chrome: str, url: str, page_name: str
) -> tuple[str, dict[str, Any]]:
    attempts = []
    best_html = ""
    best_score = -1
    best_result: subprocess.CompletedProcess[str] | None = None
    for budget in RENDER_ATTEMPTS:
        with tempfile.TemporaryDirectory(prefix="kaggle-render-") as profile:
            result = subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-background-networking",
                    "--disable-extensions",
                    "--disable-popup-blocking",
                    "--ignore-certificate-errors",
                    f"--user-data-dir={profile}",
                    f"--virtual-time-budget={budget}",
                    "--dump-dom",
                    url,
                ],
                text=True,
                capture_output=True,
                timeout=max(45, int(budget / 1000) + 30),
                check=False,
            )
        html = result.stdout or ""
        text = _html_to_text(html)
        score = _content_score(page_name, text)
        attempts.append(
            {
                "virtual_time_budget_ms": budget,
                "returncode": result.returncode,
                "stdout_chars": len(html),
                "stderr_chars": len(result.stderr or ""),
                "content_score": score,
                "has_bad_marker": any(marker in text for marker in BAD_RENDER_MARKERS),
            }
        )
        if score > best_score:
            best_score = score
            best_html = html
            best_result = result
        if _has_expected_content(
            page_name, _extract_rendered_page_text(page_name, html)
        ):
            break

    return best_html, {
        "returncode": best_result.returncode if best_result else None,
        "stderr_chars": len(best_result.stderr or "") if best_result else 0,
        "render_attempts": attempts,
        "content_score": best_score,
    }


def _find_chrome() -> str | None:
    for candidate in [
        shutil.which("chromium"),
        shutil.which("google-chrome"),
        shutil.which("chrome"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _html_metadata(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    meta: dict[str, Any] = {}
    title = soup.find("title")
    if title:
        meta["title"] = title.get_text(strip=True)
    for tag in soup.find_all("meta"):
        key = tag.get("name") or tag.get("property")
        content = tag.get("content")
        if key and content:
            meta[str(key)] = str(content)
    links = []
    for tag in soup.find_all("link"):
        rel = " ".join(tag.get("rel") or [])
        href = tag.get("href")
        if rel and href:
            links.append({"rel": rel, "href": href})
    if links:
        meta["links"] = links
    return meta


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dataset_description_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    heading = soup.find(
        string=lambda value: bool(value and value.strip() == "Dataset Description")
    )
    if not heading:
        return _html_to_text(html)

    candidates = []
    for parent in heading.parents:
        if parent.name != "div":
            continue
        text = parent.get_text("\n", strip=True)
        if "Data Dictionary" in text and "Metadata" not in text:
            candidates.append((len(text), parent))
    if not candidates:
        return _html_to_text(html)

    _, container = min(candidates, key=lambda item: item[0])
    text = container.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_rendered_page_text(page_name: str, html: str) -> str:
    if page_name == "description":
        return _section_text(html, "description", "Description") or _html_to_text(html)
    if page_name == "evaluation":
        return _section_text(html, "evaluation", "Evaluation") or _html_to_text(html)
    if page_name == "data":
        return _dataset_description_text(html)
    return _html_to_text(html)


def _content_score(page_name: str, text: str) -> int:
    score = sum(
        1 for marker in EXPECTED_RENDER_MARKERS.get(page_name, ()) if marker in text
    )
    score -= sum(2 for marker in BAD_RENDER_MARKERS if marker in text)
    return score


def _has_expected_content(page_name: str, text: str) -> bool:
    markers = EXPECTED_RENDER_MARKERS.get(page_name, ())
    return bool(
        text
        and all(marker in text for marker in markers)
        and not any(marker in text for marker in BAD_RENDER_MARKERS)
    )


def _section_text(html: str, section_id: str, fallback_heading: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    container = soup.find(id=section_id)
    if container is None:
        heading = soup.find(
            string=lambda value: bool(value and value.strip() == fallback_heading)
        )
        container = (
            next(
                (
                    parent
                    for parent in heading.parents
                    if parent.name == "div" and parent.get("id") == section_id
                ),
                None,
            )
            if heading
            else None
        )
    if container is None:
        return None
    text = container.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _repair_rendered_description(
    render_dir: Path, records: list[dict[str, Any]]
) -> None:
    description = next(
        (record for record in records if record.get("name") == "description"), None
    )
    if not description:
        return
    text_path = Path(description.get("text_path") or render_dir / "description.txt")
    current = (
        text_path.read_text(encoding="utf-8", errors="replace")
        if text_path.exists()
        else ""
    )
    if "The Challenge" in current and "Something went wrong" not in current:
        return
    for fallback_name in ["evaluation", "description"]:
        html_path = render_dir / f"{fallback_name}.html"
        if not html_path.exists():
            continue
        text = _section_text(
            html_path.read_text(encoding="utf-8", errors="replace"),
            "description",
            "Description",
        )
        if text and "The Challenge" in text:
            text_path.write_text(text, encoding="utf-8")
            description["text_chars"] = len(text)
            description["text_repaired_from"] = str(html_path)
            return


def _repair_rendered_evaluation(
    render_dir: Path, records: list[dict[str, Any]]
) -> None:
    evaluation = next(
        (record for record in records if record.get("name") == "evaluation"), None
    )
    if not evaluation:
        return
    text_path = Path(evaluation.get("text_path") or render_dir / "evaluation.txt")
    current = (
        text_path.read_text(encoding="utf-8", errors="replace")
        if text_path.exists()
        else ""
    )
    if _has_expected_content("evaluation", current):
        return
    for fallback_name in ["evaluation", "description"]:
        html_path = render_dir / f"{fallback_name}.html"
        if not html_path.exists():
            continue
        text = _section_text(
            html_path.read_text(encoding="utf-8", errors="replace"),
            "evaluation",
            "Evaluation",
        )
        if text and _has_expected_content("evaluation", text):
            text_path.write_text(text, encoding="utf-8")
            evaluation["text_chars"] = len(text)
            evaluation["text_repaired_from"] = str(html_path)
            return


def _write_combined_docs(docs_dir: Path, artifact: dict[str, Any]) -> None:
    chunks = []

    for page in artifact.get("rendered_pages", []):
        chunks.append(f"## Rendered Kaggle Page: {page['name']}\n")
        chunks.append(f"Source: {page['url']}\n")
        if page.get("error"):
            chunks.append(f"Render error: {page['error']}\n")
        text_path = page.get("text_path")
        if text_path and Path(text_path).exists():
            chunks.append(Path(text_path).read_text(encoding="utf-8", errors="replace"))
            chunks.append("\n")

    (docs_dir / "combined.md").write_text("\n".join(chunks), encoding="utf-8")


def _postprocess_combined_docs_with_llm(
    docs_dir: Path,
    artifact: dict[str, Any],
) -> None:
    combined_path = docs_dir / "combined.md"
    raw_path = docs_dir / "combined.raw.md"
    response_path = docs_dir / "combined.llm_response.md"
    raw_text = (
        combined_path.read_text(encoding="utf-8", errors="replace")
        if combined_path.exists()
        else ""
    )

    artifact["combined_text_raw_path"] = str(raw_path)
    artifact["combined_text_cleaned_by_llm"] = False
    artifact["combined_text_chars"] = len(raw_text)
    raw_path.write_text(raw_text, encoding="utf-8")

    if not raw_text.strip():
        artifact["combined_text_cleanup_error"] = (
            "combined.md was empty before LLM cleanup."
        )
        return

    prompt = (
        "Clean the following rendered Kaggle documentation into concise Markdown for this competition.\n\n"
        "Keep:\n"
        "- The task/challenge objective.\n"
        "- The target and what must be predicted.\n"
        "- Evaluation metric and submission file format when present.\n"
        "- Dataset description, file descriptions, data dictionary, and variable notes.\n\n"
        "Remove anything unrelated to the task or data description, including website navigation, "
        "duplicate text, render errors, sharing/sidebar/footer text, and generic Kaggle UI content.\n\n"
        f"Competition slug: {artifact.get('competition_slug')}\n"
        f"Competition URL: {artifact.get('url')}\n\n"
        "Rendered documentation:\n"
        f"{raw_text}"
    )

    agent = create_deep_agent(
        model=ChatOllama(
            model="qwen2.5:7b",
        ),
        system_prompt=COMBINED_DOCS_CLEANUP_SYSTEM,
    )

    try:
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )
        response = response["messages"][1].content
    except Exception as exc:
        artifact["combined_text_cleanup_error"] = (
            "LLM cleanup failed; leaving combined.md unmodified. "
            f"{type(exc).__name__}: {exc}"
        )
        return

    response_path.write_text(response, encoding="utf-8")
    cleaned = _strip_markdown_fence(response)
    if not cleaned.strip():
        artifact["combined_text_cleanup_error"] = (
            "LLM cleanup returned empty content; leaving combined.md unmodified."
        )
        artifact["combined_text_cleanup_response_path"] = str(response_path)
        return

    combined_path.write_text(cleaned.strip() + "\n", encoding="utf-8")
    artifact["combined_text_cleaned_by_llm"] = True
    artifact["combined_text_chars"] = len(cleaned.strip())
    artifact["combined_text_cleanup_response_path"] = str(response_path)


def _strip_markdown_fence(text: str) -> str:
    match = re.fullmatch(
        r"\s*```(?:markdown|md)?\s*\n(.*?)\n```\s*", text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return text.strip()


if __name__ == "__main__":
    p = _workspace_for_challenge("https://www.kaggle.com/competitions/titanic")
    p.mkdir(parents=True, exist_ok=True)
    write_challenge_docs(p, "https://www.kaggle.com/competitions/titanic")
