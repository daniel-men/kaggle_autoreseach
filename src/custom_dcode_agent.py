"""Agent management and creation."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import MemoryMiddleware, SkillsMiddleware

# Backwards-compat flag: SDKs before 0.5.4 accept only `list[str]` for
# `SkillsMiddleware.sources`; newer SDKs expose the `SkillSource` alias
# that permits `(path, label)` tuples. The `skills` module is already
# loaded by the `SkillsMiddleware` import above, so the extra lookup
# here adds no startup cost.
try:
    from deepagents.middleware.skills import SkillSource as _SkillSource  # noqa: F401
except ImportError:
    _SUPPORTS_SKILL_SOURCE_TUPLES = False
else:
    _SUPPORTS_SKILL_SOURCE_TUPLES = True

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from deepagents.backends.sandbox import SandboxBackendProtocol
    from deepagents.middleware.async_subagents import AsyncSubAgent
    from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
    from langchain.agents.middleware import InterruptOnConfig
    from langchain.agents.middleware.types import AgentState
    from langchain.messages import ToolCall
    from langchain.tools import BaseTool
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import ToolMessage
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.pregel import Pregel
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from deepagents_code.mcp_tools import MCPServerInfo
    from deepagents_code.output import OutputFormat

from deepagents_code.agent import ShellAllowListMiddleware, _resolve_ptc_option, get_system_prompt
from langchain.agents.middleware.types import AgentMiddleware

from deepagents_code import theme
from deepagents_code._constants import DEFAULT_AGENT_NAME
from deepagents_code.config import (
    _ShellAllowAll,
    config,
    console,
    get_default_coding_instructions,
    get_glyphs,
    settings,
)
from deepagents_code.configurable_model import ConfigurableModelMiddleware
from deepagents_code.filesystem_empty_result import _FilesystemEmptyResultMiddleware
from deepagents_code.integrations.sandbox_factory import get_default_working_dir
from deepagents_code.local_context import (
    LocalContextMiddleware,
    _AsyncExecutableBackend,
    _ExecutableBackend,
)
from deepagents_code.project_utils import ProjectContext, get_server_project_context
from deepagents_code.subagents import list_subagents
from deepagents_code.unicode_security import (
    check_url_safety,
    detect_dangerous_unicode,
    format_warning_detail,
    render_with_unicode_markers,
    strip_dangerous_unicode,
    summarize_issues,
)

logger = logging.getLogger(__name__)


def create_cli_agent(
    model: str | BaseChatModel,
    assistant_id: str,
    *,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    sandbox: SandboxBackendProtocol | None = None,
    sandbox_type: str | None = None,
    system_prompt: str | None = None,
    interactive: bool = True,
    auto_approve: bool = False,
    interrupt_shell_only: bool = False,
    shell_allow_list: list[str] | None = None,
    enable_ask_user: bool = True,
    enable_memory: bool = True,
    enable_skills: bool = True,
    enable_shell: bool = True,
    enable_interpreter: bool = False,
    checkpointer: BaseCheckpointSaver | None = None,
    mcp_server_info: list[MCPServerInfo] | None = None,
    cwd: str | Path | None = None,
    project_context: ProjectContext | None = None,
    async_subagents: list[AsyncSubAgent] | None = None,
    response_format = None
) -> tuple[Pregel, CompositeBackend]:
    """Create a CLI-configured agent with flexible options.

    This is the main entry point for creating a Deep Agents Code agent, usable
    both internally and from external code (e.g., benchmarking frameworks).

    Args:
        model: LLM model to use (e.g., `'provider:model'`)
        assistant_id: Agent identifier for memory/state storage
        tools: Additional tools to provide to agent
        sandbox: Optional sandbox backend for remote execution
            (e.g., `ModalSandbox`).

            If `None`, uses local filesystem + shell.
        sandbox_type: Type of sandbox provider
            (`'agentcore'`, `'daytona'`, `'langsmith'`, `'modal'`, `'runloop'`).
            Used for system prompt generation.
        system_prompt: Override the default system prompt.

            If `None`, generates one based on `sandbox_type`, `assistant_id`,
            and `interactive`.
        interactive: When `False`, the auto-generated system prompt is
            tailored for headless non-interactive execution. Ignored when
            `system_prompt` is provided explicitly.
        auto_approve: If `True`, no tools trigger human-in-the-loop
            interrupts — all calls (shell execution, file writes/edits,
            web search, URL fetch) run automatically.

            If `False`, tools pause for user confirmation via the approval menu.
            See `_add_interrupt_on` for the full list of gated tools.
        interrupt_shell_only: If `True`, all HITL interrupts are disabled;
            shell commands are validated inline by `ShellAllowListMiddleware`
            against the configured allow-list instead.

            Used in non-interactive mode with a restrictive shell allow-list
            to avoid splitting traces into multiple LangSmith runs.

            Has no effect when `auto_approve` is `True` (interrupts are already
            disabled) or when `shell_allow_list` is `SHELL_ALLOW_ALL`.
        shell_allow_list: Explicit restrictive shell allow-list forwarded from
            the CLI process. When provided (and `interrupt_shell_only` is
            `True`), used directly instead of reading `settings.shell_allow_list`
            (which may not be set in the server subprocess environment).
        enable_ask_user: Enable `AskUserMiddleware` so the agent can ask
            clarifying questions.

            Disabled in non-interactive mode.
        enable_memory: Enable `MemoryMiddleware` for persistent memory
        enable_skills: Enable `SkillsMiddleware` for custom agent skills
        enable_shell: Enable shell execution via `LocalShellBackend`
            (only in local mode). When enabled, the `execute` tool is available.
        enable_interpreter: Wire `CodeInterpreterMiddleware` from
            `langchain-quickjs` into the main agent.

            Local-mode only — passing a non-`None` `sandbox` while
            `enable_interpreter=True` raises `ValueError`. Subagents do not
            receive the interpreter in v1.

            PTC (`tools.*` host bridge) calls bypass `interrupt_on`/HITL
            approval, so `settings.interpreter_ptc` is the only effective
            control over which host tools can be invoked from inside the
            REPL. `js_eval` itself is intentionally not gated by HITL —
            per-call approval would be unusably noisy and would not block
            PTC fan-out anyway. The `"safe"` preset is therefore restricted
            to tools that are already non-HITL outside the REPL (read-only
            file inspection); exposing HITL-gated tools — network fetch,
            subagent dispatch, shell, file writes — requires an explicit
            list or `interpreter_ptc="all"` with
            `interpreter_ptc_acknowledge_unsafe=True`.

            Requires the `quickjs` optional extra
            (`langchain-quickjs>=0.1.2,<0.2.0`).
        checkpointer: Optional checkpointer for session persistence.
            When `None`, the graph is compiled without a checkpointer.
        mcp_server_info: MCP server metadata to surface in the system prompt.
        cwd: Override the working directory for the agent's filesystem backend
            and system prompt.
        project_context: Explicit project path context for project-sensitive
            behavior such as project `AGENTS.md` files, skills, subagents, and
            MCP trust.
        async_subagents: Remote LangGraph deployments to expose as async subagent tools.

            Loaded from `[async_subagents]` in `config.toml` or passed directly.

    Returns:
        2-tuple of `(agent_graph, backend)`

            - `agent_graph`: Configured LangGraph Pregel instance ready
                for execution
            - `composite_backend`: `CompositeBackend` for file operations

    Raises:
        ValueError: When `enable_interpreter=True` is paired with a
            non-`None` `sandbox`, when `settings.interpreter_ptc` contains
            unknown tool names, or when `interpreter_ptc="all"` is used
            without `auto_approve` or `interpreter_ptc_acknowledge_unsafe`.
    """
    tools = tools or []
    effective_cwd = (
        Path(cwd)
        if cwd is not None
        else (project_context.user_cwd if project_context is not None else None)
    )

    # Setup agent directory for persistent memory (if enabled)
    if enable_memory or enable_skills:
        agent_dir = settings.ensure_agent_dir(assistant_id)
        agent_md = agent_dir / "AGENTS.md"
        if not agent_md.exists():
            # Create empty file for user customizations
            # Base instructions are loaded fresh from get_system_prompt()
            agent_md.touch()

    # Skills directories (if enabled)
    skills_dir = None
    user_agent_skills_dir = None
    project_skills_dir = None
    project_agent_skills_dir = None
    if enable_skills:
        skills_dir = settings.ensure_user_skills_dir(assistant_id)
        user_agent_skills_dir = settings.get_user_agent_skills_dir()
        project_skills_dir = (
            project_context.project_skills_dir()
            if project_context is not None
            else settings.get_project_skills_dir()
        )
        project_agent_skills_dir = (
            project_context.project_agent_skills_dir()
            if project_context is not None
            else settings.get_project_agent_skills_dir()
        )

    # Load custom subagents from filesystem
    custom_subagents: list[SubAgent | CompiledSubAgent] = []
    restrictive_shell_allow_list: list[str] | None = None
    if interrupt_shell_only and not auto_approve:
        # Prefer the explicitly forwarded allow-list (set by the CLI process
        # and passed through ServerConfig).  Fall back to settings only for
        # direct callers (e.g. benchmarking frameworks) that don't go through
        # the server subprocess path.
        if shell_allow_list:
            restrictive_shell_allow_list = list(shell_allow_list)
        elif settings.shell_allow_list and not isinstance(
            settings.shell_allow_list, _ShellAllowAll
        ):
            restrictive_shell_allow_list = list(settings.shell_allow_list)
        else:
            logger.warning(
                "interrupt_shell_only=True but no restrictive shell allow-list "
                "available; falling back to standard HITL interrupts"
            )

    user_agents_dir = settings.get_user_agents_dir(assistant_id)
    project_agents_dir = (
        project_context.project_agents_dir()
        if project_context is not None
        else settings.get_project_agents_dir()
    )

    def _subagent_cli_middleware(*, has_explicit_model: bool) -> list[AgentMiddleware]:
        middleware: list[AgentMiddleware] = []
        if not has_explicit_model:
            middleware.append(ConfigurableModelMiddleware())
        if restrictive_shell_allow_list is not None:
            middleware.append(ShellAllowListMiddleware(restrictive_shell_allow_list))
        return middleware

    for subagent_meta in list_subagents(
        user_agents_dir=user_agents_dir,
        project_agents_dir=project_agents_dir,
    ):
        # Treat a falsy spec (`None` or `""`) as "no explicit model" so an empty
        # `model:` in subagent frontmatter inherits the runtime model rather than
        # being forwarded verbatim to `resolve_model("")`.
        model_spec = subagent_meta["model"]
        has_explicit_model = bool(model_spec)
        subagent: SubAgent = {
            "name": subagent_meta["name"],
            "description": subagent_meta["description"],
            "system_prompt": subagent_meta["system_prompt"],
        }
        if model_spec:
            subagent["model"] = model_spec
        subagent_middleware = _subagent_cli_middleware(
            has_explicit_model=has_explicit_model
        )
        if subagent_middleware:
            subagent["middleware"] = subagent_middleware
        custom_subagents.append(subagent)

    from deepagents.middleware.subagents import (
        GENERAL_PURPOSE_SUBAGENT,
        SubAgent as RuntimeSubAgent,
    )

    if not any(
        subagent["name"] == GENERAL_PURPOSE_SUBAGENT["name"]
        for subagent in custom_subagents
    ):
        general_purpose_subagent: RuntimeSubAgent = {
            "name": GENERAL_PURPOSE_SUBAGENT["name"],
            "description": GENERAL_PURPOSE_SUBAGENT["description"],
            "system_prompt": GENERAL_PURPOSE_SUBAGENT["system_prompt"],
            "middleware": _subagent_cli_middleware(has_explicit_model=False),
        }
        custom_subagents.append(general_purpose_subagent)

    # Build middleware stack based on enabled features
    agent_middleware = [
        ConfigurableModelMiddleware(),
        _FilesystemEmptyResultMiddleware(),
    ]

    # Resume state: declares the `_context_tokens` and `_model_spec` channels
    # and writes them from `after_model` (token count from the latest
    # `AIMessage.usage_metadata`, model spec from `context["effective_model"]`).
    # The CLI reads them back from `state_values` on thread resume.
    from deepagents_code.resume_state import ResumeStateMiddleware

    agent_middleware.append(ResumeStateMiddleware())

    # Add ask_user middleware (must be early so its tool is available)
    if enable_ask_user:
        from deepagents_code.ask_user import AskUserMiddleware

        agent_middleware.append(AskUserMiddleware())

    # Add memory middleware
    if enable_memory:
        memory_sources = [str(settings.get_user_agent_md_path(assistant_id))]
        project_agent_md_paths = (
            project_context.project_agent_md_paths()
            if project_context is not None
            else settings.get_project_agent_md_path()
        )
        memory_sources.extend(str(p) for p in project_agent_md_paths)

        agent_middleware.append(
            MemoryMiddleware(
                backend=FilesystemBackend(virtual_mode=False),
                sources=memory_sources,
            )
        )

    # Add skills middleware
    if enable_skills:
        # Lowest to highest precedence:
        # built-in -> user .deepagents -> user .agents
        # -> project .deepagents -> project .agents
        # -> user .claude (experimental) -> project .claude (experimental)
        # Labels disambiguate user- vs project-scoped sources that share a
        # `.../skills` leaf; the middleware would otherwise derive identical
        # labels from the parent directory name.
        sources: list[tuple[str, str]] = [
            (str(settings.get_built_in_skills_dir()), "Built-in"),
            (str(skills_dir), "User Deepagents"),
            (str(user_agent_skills_dir), "User Agents"),
        ]
        if project_skills_dir:
            sources.append((str(project_skills_dir), "Project Deepagents"))
        if project_agent_skills_dir:
            sources.append((str(project_agent_skills_dir), "Project Agents"))

        # Experimental: Claude Code skill directories
        user_claude_skills_dir = settings.get_user_claude_skills_dir()
        if user_claude_skills_dir.exists():
            sources.append((str(user_claude_skills_dir), "User Claude"))
        project_claude_skills_dir = settings.get_project_claude_skills_dir()
        if project_claude_skills_dir:
            sources.append((str(project_claude_skills_dir), "Project Claude"))

        # Backwards-compat: strip labels when the installed SDK is too old
        # to accept `(path, label)` tuples. Label-based disambiguation
        # regresses to the pre-alias behavior (user- and project-scoped
        # `.claude/skills` collapse to the same label), but functionality
        # is preserved.
        middleware_sources: Sequence[str | tuple[str, str]] = (
            sources if _SUPPORTS_SKILL_SOURCE_TUPLES else [path for path, _ in sources]
        )

        agent_middleware.append(
            SkillsMiddleware(
                backend=FilesystemBackend(virtual_mode=False),
                sources=middleware_sources,
            )
        )

    # CONDITIONAL SETUP: Local vs Remote Sandbox
    if sandbox is None:
        # ========== LOCAL MODE ==========
        root_dir = effective_cwd if effective_cwd is not None else Path.cwd()
        if enable_shell:
            # Create environment for shell commands
            # Restore user's original LANGSMITH_PROJECT so their code traces separately
            shell_env = os.environ.copy()
            if settings.user_langchain_project:
                shell_env["LANGSMITH_PROJECT"] = settings.user_langchain_project

            # Use LocalShellBackend for filesystem + shell execution.
            # The SDK's FilesystemMiddleware exposes per-command timeout
            # on the execute tool natively.
            backend = LocalShellBackend(
                root_dir=root_dir,
                inherit_env=True,
                env=shell_env,
            )
        else:
            # No shell access - use plain FilesystemBackend
            backend = FilesystemBackend(root_dir=root_dir, virtual_mode=False)
    else:
        # ========== REMOTE SANDBOX MODE ==========
        backend = sandbox  # Remote sandbox (ModalSandbox, etc.)
        # Note: Shell middleware not used in sandbox mode
        # File operations and execute tool are provided by the sandbox backend

    if enable_interpreter:
        if sandbox is not None:
            msg = (
                "enable_interpreter=True is not supported with a remote "
                "sandbox in this release. Disable the sandbox or unset "
                "enable_interpreter."
            )
            raise ValueError(msg)
        # Lazy import keeps `dcode -v` fast — see AGENTS.md startup-perf rule.
        from langchain_quickjs import CodeInterpreterMiddleware, PTCOption

        ptc_names = _resolve_ptc_option(
            settings.interpreter_ptc,
            tools=tools,
            acknowledge_unsafe=settings.interpreter_ptc_acknowledge_unsafe,
            auto_approve=auto_approve,
        )
        ptc_option: PTCOption | None = (
            cast("PTCOption", list(ptc_names)) if ptc_names is not None else None
        )
        agent_middleware.append(
            CodeInterpreterMiddleware(
                tool_name="js_eval",
                timeout=settings.interpreter_timeout_seconds,
                memory_limit=settings.interpreter_memory_limit_mb * 1024 * 1024,
                max_ptc_calls=settings.interpreter_max_ptc_calls,
                max_result_chars=settings.interpreter_max_result_chars,
                ptc=ptc_option,
            )
        )

    # Local context middleware (git info, directory tree, etc.).
    if isinstance(backend, (_ExecutableBackend, _AsyncExecutableBackend)):
        agent_middleware.append(
            LocalContextMiddleware(backend=backend, mcp_server_info=mcp_server_info)
        )

    # Add shell allow-list middleware when interrupt_shell_only is active.
    shell_middleware_added = False
    if restrictive_shell_allow_list is not None:
        agent_middleware.append(ShellAllowListMiddleware(restrictive_shell_allow_list))
        shell_middleware_added = True

    # Get or use custom system prompt
    if system_prompt is None:
        system_prompt = get_system_prompt(
            assistant_id=assistant_id,
            sandbox_type=sandbox_type,
            interactive=interactive,
            cwd=effective_cwd,
        )

    # Configure interrupt_on based on auto_approve / shell_middleware_added
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None
    if auto_approve or shell_middleware_added:  # noqa: SIM108  # if-else clearer than ternary for dual-path config
        # No HITL interrupts — tools run automatically.
        # When shell_middleware_added is True, shell validation is handled by
        # ShellAllowListMiddleware (added above) which rejects disallowed
        # commands inline as error ToolMessages, keeping the entire run in
        # a single LangSmith trace.
        interrupt_on = {}
    else:
        # Full HITL for destructive operations
        interrupt_on = _add_interrupt_on()  # type: ignore[assignment]  # InterruptOnConfig is compatible at runtime

    # Set up composite backend with routing
    # For local FilesystemBackend, route large tool results to /tmp to avoid polluting
    # the working directory. For sandbox backends, no special routing is needed.
    if sandbox is None:
        # Local mode: Route large results to a unique temp directory
        large_results_backend = FilesystemBackend(
            root_dir=tempfile.mkdtemp(prefix="deepagents_large_results_"),
            virtual_mode=True,
        )
        conversation_history_backend = FilesystemBackend(
            root_dir=tempfile.mkdtemp(prefix="deepagents_conversation_history_"),
            virtual_mode=True,
        )
        composite_backend = CompositeBackend(
            default=backend,
            routes={
                "/large_tool_results/": large_results_backend,
                "/conversation_history/": conversation_history_backend,
            },
        )
    else:
        # Sandbox mode: No special routing needed
        composite_backend = CompositeBackend(
            default=backend,
            routes={},
        )

    from deepagents.middleware.summarization import create_summarization_tool_middleware

    agent_middleware.append(
        create_summarization_tool_middleware(model, composite_backend)
    )

    # Create the agent
    all_subagents: list[SubAgent | CompiledSubAgent | AsyncSubAgent] = [
        *custom_subagents,
        *(async_subagents or []),
    ]
    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        backend=composite_backend,
        middleware=agent_middleware,
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
        subagents=all_subagents or None,
        response_format=response_format
    ).with_config(config)
    return agent, composite_backend
