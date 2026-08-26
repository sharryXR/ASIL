"""Helpers for GUI evaluation synchronization policy."""

from __future__ import annotations

import inspect
from typing import Any, Literal

from asil.adapter import ASILAdapter


GUIEvalMode = Literal[
    "api_live",
    "persist_then_observe",
    "live_shadow_required",
    "custom_sync_existing",
]


_SOFTWARE_TO_MODE: dict[str, GUIEvalMode] = {
    "gitea": "api_live",
    "obs": "api_live",
    "audacity": "custom_sync_existing",
    "nautilus": "custom_sync_existing",
    "thunderbird": "custom_sync_existing",
    "jupyterlab": "live_shadow_required",
    "code_server": "live_shadow_required",
    "drawio": "live_shadow_required",
    "inkscape": "persist_then_observe",
    "libreoffice": "persist_then_observe",
    "libreoffice_writer": "persist_then_observe",
    "libreoffice_impress": "persist_then_observe",
    "gimp": "persist_then_observe",
    "kdenlive": "persist_then_observe",
    "blender": "persist_then_observe",
    "multi_apps": "custom_sync_existing",
}

_APP_NAME_TO_SOFTWARE = {
    "jupyterlab": "jupyterlab",
    "code-server": "code_server",
    "draw.io": "drawio",
    "gitea": "gitea",
    "obs": "obs",
    "audacity": "audacity",
    "nautilus": "nautilus",
    "thunderbird": "thunderbird",
    "inkscape": "inkscape",
    "libreoffice": "libreoffice",
    "libreoffice writer": "libreoffice_writer",
    "libreoffice impress": "libreoffice_impress",
    "gimp": "gimp",
    "kdenlive": "kdenlive",
    "blender": "blender",
    "multi apps": "multi_apps",
}


def adapter_software_key(adapter: Any) -> str:
    explicit = getattr(adapter, "software_name", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    app_name = str(getattr(adapter, "app_name", "") or "").strip().lower()
    if app_name in _APP_NAME_TO_SOFTWARE:
        return _APP_NAME_TO_SOFTWARE[app_name]
    class_name = type(adapter).__name__
    lowered = class_name.removesuffix("Adapter").lower()
    return lowered.replace("libreofficewriter", "libreoffice_writer").replace(
        "libreofficeimpress", "libreoffice_impress"
    )


def gui_eval_mode_for_adapter(adapter: Any) -> GUIEvalMode:
    mode_implementation = getattr(type(adapter), "gui_eval_mode", None)
    if callable(mode_implementation) and mode_implementation is not ASILAdapter.gui_eval_mode:
        mode = adapter.gui_eval_mode()
        if isinstance(mode, str):
            return mode  # type: ignore[return-value]
    software = adapter_software_key(adapter)
    return _SOFTWARE_TO_MODE.get(software, "api_live")


def gui_eval_mode_by_software(software: str) -> GUIEvalMode:
    return _SOFTWARE_TO_MODE.get(software, "api_live")


def requires_gui_persist(adapter: Any) -> bool:
    return gui_eval_mode_for_adapter(adapter) == "persist_then_observe"


def requires_live_shadow(adapter: Any) -> bool:
    return gui_eval_mode_for_adapter(adapter) == "live_shadow_required"


def sync_adapter_from_gui(adapter: Any, session: Any | None = None) -> None:
    sync_method = getattr(adapter, "sync_from_gui", None)
    if not callable(sync_method):
        return
    try:
        signature = inspect.signature(sync_method)
    except (TypeError, ValueError):
        sync_method(session)
        return
    positional_params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    accepts_varargs = any(
        parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values()
    )
    if accepts_varargs or positional_params:
        sync_method(session)
        return
    sync_method()
