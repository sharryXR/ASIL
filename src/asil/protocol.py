"""ASIL Protocol — Core types for Agent-Software Interaction Layer."""

from __future__ import annotations
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Meta(BaseModel):
    """Metadata about the observation source."""
    app_name: str
    app_version: str = ""
    asil_version: str = "1.0"
    observation_source: str  # file_parse | script_api | rest_api | dom_snapshot
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class Element(BaseModel):
    """A single interactive element in the software."""
    id: str
    type: str
    label: str = ""
    value: Any = None
    editable: bool = True
    data_type: str = "string"
    constraints: dict[str, Any] = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AppState(BaseModel):
    """Current application state."""
    current_view: str = ""
    active_document: str = ""
    document_path: str = ""


class Process(BaseModel):
    name: str
    status: str = "running"  # running | completed | failed
    progress: float | None = None


class Environment(BaseModel):
    """Background environment values not directly visible in UI."""
    undo_stack_depth: int = 0
    clipboard_content_type: str = ""
    active_processes: list[Process] = Field(default_factory=list)
    unsaved_changes: bool = False
    recent_errors: list[dict[str, str]] = Field(default_factory=list)
    system: dict[str, float] = Field(default_factory=dict)


class ViewInfo(BaseModel):
    id: str
    label: str = ""
    description: str = ""


class Navigation(BaseModel):
    """Navigation topology."""
    available_views: list[ViewInfo] = Field(default_factory=list)
    current_path: str = ""
    breadcrumb: list[str] = Field(default_factory=list)
    reachable_from_here: list[str] = Field(default_factory=list)


class Observation(BaseModel):
    """Complete structured state of a software application."""
    meta: Meta
    app_state: AppState = Field(default_factory=AppState)
    interactive_elements: list[Element] = Field(default_factory=list)
    environment: Environment = Field(default_factory=Environment)
    navigation: Navigation = Field(default_factory=Navigation)
    data_summary: str = ""


class Action(BaseModel):
    """A structured action to execute on the software."""
    action_type: str  # set_value | invoke_function | modify_file | api_call | navigate | batch
    target: str
    params: dict[str, Any] = Field(default_factory=dict)
    expect_observation: bool = True
