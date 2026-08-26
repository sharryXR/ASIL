"""CLI Baseline adapter — simulates CLI-Anything-style command wrappers."""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Any

from asil.adapter import ASILAdapter
from asil.protocol import Action, Element, Observation, Meta, AppState


class CLIBaseline(ASILAdapter):
    """Wraps an ASIL adapter with CLI-style command translation.

    Simulates the information loss of CLI wrappers:
    - observe() returns text-like representations instead of typed values
    - execute() shells out to CLI commands instead of direct data manipulation
    """

    app_name = "CLI-Wrapper"
    supported_action_types = ["modify_file", "set_value", "invoke_function", "api_call"]

    def __init__(self, delegate: ASILAdapter) -> None:
        self._delegate = delegate
        self.app_name = f"CLI-{delegate.app_name}"

    @property
    def source_path(self) -> Path | None:
        return self._delegate.source_path

    def clone(self, new_path: Path) -> "CLIBaseline":
        return CLIBaseline(self._delegate.clone(new_path))

    def get_context(self) -> dict[str, str]:
        return self._delegate.get_context()

    def observe(self) -> Observation:
        # Get full ASIL observation, then degrade it to CLI-like text output
        full_obs = self._delegate.observe()

        # Degrade elements: lose type information, constraints, structured actions
        degraded_elements = []
        for e in full_obs.interactive_elements:
            # CLI output is text-based — convert value to string representation
            if isinstance(e.value, dict):
                text_value = " ".join(f"{k}={v}" for k, v in e.value.items() if v is not None)
            else:
                text_value = str(e.value) if e.value is not None else ""

            degraded_elements.append(Element(
                id=e.id,
                type=e.type,  # keep type but lose structured metadata
                label=e.label,
                value=text_value,  # flatten to text
                editable=e.editable,
                data_type="string",  # everything is string in CLI
                constraints={},  # no constraints in CLI output
                actions=[],  # no action discovery in CLI
                metadata={},  # no metadata in CLI
            ))

        return Observation(
            meta=Meta(
                app_name=self.app_name,
                app_version=full_obs.meta.app_version,
                observation_source="cli_output",
            ),
            app_state=full_obs.app_state,
            interactive_elements=degraded_elements,
            environment=full_obs.environment,
            navigation=full_obs.navigation,
            data_summary=full_obs.data_summary,
        )

    def execute(self, action: Action) -> Observation:
        # CLI baseline shells out to commands instead of direct manipulation
        # For testing, we delegate to the underlying adapter
        # In production, this would call: inkscape --actions=..., libreoffice --headless, etc.
        self._delegate.execute(action)
        # Return CLI-degraded observation (not raw delegate output)
        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types
