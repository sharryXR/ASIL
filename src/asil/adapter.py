"""Abstract base class for ASIL adapters."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from asil.protocol import Observation, Action, Meta, AppState, Element, Environment, Navigation
from asil.rendering import RenderArtifact


@dataclass(frozen=True)
class GUISessionSpec:
    """Minimal runtime contract for launching a real GUI surface.

    This is intentionally additive: adapters can expose it for real GUI-agent
    runs without affecting the existing structured observation/action protocol.
    """

    surface_type: str  # "desktop" | "browser" | "multi_window"
    window_title_pattern: str
    window_class_pattern: str | None = None
    launch_command: tuple[str, ...] = ()
    browser_url: str = ""
    browser_navigation_mode: str = "fresh_page"
    browser_post_navigation_settle_ms: int = 0
    startup_timeout_s: float = 45.0
    post_launch_delay_s: float = 3.0
    run_as_user: str | None = None
    min_width: int = 800
    min_height: int = 600
    persist_shortcuts: tuple[str, ...] = field(default_factory=tuple)
    extra_env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    post_launch_callback: Callable[[], None] | None = None
    backend_ready_probe: Callable[[], None] | None = None
    ui_ready_probe: Callable[..., None] | None = None
    close_callback: Callable[[], None] | None = None
    child_specs: dict[str, "GUISessionSpec"] = field(default_factory=dict)
    primary_child: str = ""
    capture_active_window: bool = False


class ASILAdapter(ABC):
    """Base class all ASIL software adapters must implement."""

    app_name: str = "Unknown"
    app_version: str = ""
    supported_action_types: list[str] = []

    @abstractmethod
    def observe(self) -> Observation:
        """Extract the complete current state of the software."""
        ...

    @abstractmethod
    def execute(self, action: Action) -> Observation:
        """Execute an action and return the resulting observation."""
        ...

    @abstractmethod
    def validate_action(self, action: Action) -> bool:
        """Check whether an action is valid for this adapter."""
        ...

    @property
    def source_path(self) -> Path | None:
        """Return the primary file path managed by this adapter, if any."""
        return None

    def clone(self, new_path: Path) -> "ASILAdapter":
        """Create an independent copy of this adapter backed by a new file.

        Only supported for file-based adapters. Service-based adapters
        (OBS, Gitea) raise NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support clone()")

    def get_context(self) -> dict[str, str]:
        """Return template context for placeholder resolution in task actions."""
        return {}

    def describe_rendering(self) -> RenderArtifact:
        """Describe the type of page artifact this adapter can render."""
        return RenderArtifact(
            filename="",
            kind="state_render",
            backend=type(self).__name__,
            actual_page=False,
            description="Adapter-specific state visualization",
            capture_complete=True,
        )

    def get_gui_session_spec(self) -> GUISessionSpec | None:
        """Return a real-GUI session contract when the software supports it.

        Returning None keeps the current adapter behavior unchanged and signals
        that the benchmark must fall back to a software-level registry or
        refuse GUI-agent execution for that adapter.
        """
        return None

    def gui_eval_mode(self) -> Literal[
        "api_live",
        "persist_then_observe",
        "live_shadow_required",
        "custom_sync_existing",
    ]:
        """Describe how GUI-visible state should be synchronized for scoring."""
        return "api_live"

    def _set_gui_shadow_state(self, state: dict[str, Any] | None) -> None:
        self.__dict__["_gui_shadow_state"] = state

    def _get_gui_shadow_state(self) -> dict[str, Any] | None:
        shadow = self.__dict__.get("_gui_shadow_state")
        return shadow if isinstance(shadow, dict) else None

    def clear_gui_shadow_state(self) -> None:
        self.__dict__.pop("_gui_shadow_state", None)

    def sync_from_gui(self, session: Any | None = None) -> None:
        """Synchronize canonical observation state from a real GUI session.

        GUI-agent runs execute real window interactions but still score through
        the adapter's canonical `observe()` output. Adapters whose canonical
        observation source is not automatically mutated by GUI actions can
        override this hook to read the live application state and write it back
        into the adapter's canonical source before `observe()` is called.

        The default implementation is a no-op for adapters where GUI edits
        already persist to the same file/service/API that `observe()` reads.
        """
        del session
        return None

    def _build_observation(
        self,
        source: str,
        elements: list[dict[str, Any] | Element],
        app_state: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
        navigation: dict[str, Any] | None = None,
        data_summary: str = "",
    ) -> Observation:
        """Convenience builder — normalises dicts into typed models."""
        typed_elements = [
            e if isinstance(e, Element) else Element(**e) for e in elements
        ]
        return Observation(
            meta=Meta(
                app_name=self.app_name,
                app_version=self.app_version,
                observation_source=source,
            ),
            app_state=AppState(**(app_state or {})),
            interactive_elements=typed_elements,
            environment=Environment(**(environment or {})),
            navigation=Navigation(**(navigation or {})),
            data_summary=data_summary,
        )
