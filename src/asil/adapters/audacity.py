"""ASIL adapter for Audacity — deterministic project state with honest rendering."""

from __future__ import annotations

import html
import errno
import json
import math
import os
import pwd
import re
import shutil
import stat
import subprocess
import tempfile
import time
import wave
import uuid
from copy import deepcopy
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.protocol import Action, Element, Observation
from asil.rendering import (
    RenderArtifact,
    assert_png_not_blank,
    ensure_audio_backend,
    ensure_user_access,
    ensure_virtual_display,
    launch_gui_process,
    terminate_process,
)


def _default_state() -> dict[str, Any]:
    return {
        "project_name": "podcast_mix",
        "sample_rate_hz": 48000,
        "transport": {"playback": "stopped", "record_armed": False},
        "selection": {"start": 2.5, "end": 6.0, "focused_track_id": "track_voice"},
        "export": {"format": "wav", "filename": "podcast_mix.wav", "directory": "/tmp/exports"},
        "tracks": [
            {
                "id": "track_voice",
                "name": "Voiceover",
                "kind": "audio",
                "mute": False,
                "solo": False,
                "gain_db": -1.5,
                "pan": 0.0,
                "height_px": 148,
                "clips": [
                    {"id": "clip_voice_1", "name": "Intro Take", "start": 0.0, "end": 8.0, "color": "blue"}
                ],
            },
            {
                "id": "track_music",
                "name": "Music Bed",
                "kind": "audio",
                "mute": True,
                "solo": False,
                "gain_db": -8.0,
                "pan": -0.25,
                "height_px": 124,
                "clips": [
                    {"id": "clip_music_1", "name": "Theme Loop", "start": 0.0, "end": 15.0, "color": "green"}
                ],
            },
        ],
        "labels": [
            {"id": "label_intro", "text": "Intro", "start": 0.0, "end": 2.0},
            {"id": "label_edit", "text": "Tighten pause", "start": 4.0, "end": 4.8},
        ],
        "history": ["Open project", "Trim intro", "Rename tracks"],
    }


class AudacityAdapter(ASILAdapter):
    app_name = "Audacity"
    supported_action_types = ["modify_file", "set_value"]

    def __init__(self, project_path: str | Path) -> None:
        self.project_path = Path(project_path)
        self._last_capture_complete = False
        if not self.project_path.exists():
            self.setup_state("default")

    @classmethod
    def from_evaluation_context(cls, tmp: Path, sandbox=None, mock: bool = False) -> "AudacityAdapter":
        del sandbox, mock
        return cls(tmp / "audacity_project.json")

    @property
    def source_path(self) -> Path:
        return self.project_path

    def clone(self, new_path: Path) -> "AudacityAdapter":
        shutil.copy2(self.project_path, new_path)
        return AudacityAdapter(new_path)

    def get_context(self) -> dict[str, str]:
        return {"project_path": str(self.project_path)}

    def reset_state(self) -> None:
        self.setup_state("default")

    def setup_state(self, initial_state: str) -> None:
        state = _default_state()
        if initial_state == "blank":
            state["tracks"] = []
            state["labels"] = []
            state["selection"] = {"start": 0.0, "end": 0.0, "focused_track_id": ""}
            state["history"] = ["New project"]
        elif initial_state == "playback_active":
            state["transport"]["playback"] = "playing"
        elif initial_state == "music_unmuted":
            self._find_track(state, "track_music")["mute"] = False
        self._write_state(state)

    def validate_action(self, action: Action) -> bool:
        if action.action_type not in self.supported_action_types:
            return False
        return action.target in {"audacity_project", str(self.project_path)}

    def observe(self) -> Observation:
        state = self._read_state()
        elements: list[Element] = []

        for track in state["tracks"]:
            elements.append(
                Element(
                    id=f"track:{track['id']}",
                    type="track",
                    label=track["name"],
                    value={
                        "name": track["name"],
                        "kind": track["kind"],
                        "mute": track["mute"],
                        "solo": track["solo"],
                        "gain_db": track["gain_db"],
                        "pan": track["pan"],
                        "height_px": track["height_px"],
                        "clip_count": len(track["clips"]),
                    },
                    editable=True,
                    actions=["rename_track", "toggle_mute", "toggle_solo", "set_gain", "set_pan"],
                    children=[f"clip:{clip['id']}" for clip in track["clips"]],
                )
            )
            for clip in track["clips"]:
                elements.append(
                    Element(
                        id=f"clip:{clip['id']}",
                        type="clip",
                        label=clip["name"],
                        value={
                            "name": clip["name"],
                            "track_id": track["id"],
                            "start": clip["start"],
                            "end": clip["end"],
                            "duration": round(float(clip["end"]) - float(clip["start"]), 3),
                            "color": clip.get("color", ""),
                        },
                        editable=True,
                        actions=["rename_clip", "move_clip", "trim_clip"],
                    )
                )

        for label in state["labels"]:
            elements.append(
                Element(
                    id=f"label:{label['id']}",
                    type="label",
                    label=label["text"],
                    value={"text": label["text"], "start": label["start"], "end": label["end"]},
                    editable=True,
                    actions=["edit_label", "move_label"],
                )
            )

        elements.append(
            Element(
                id="selection",
                type="selection",
                label="Selection",
                value=dict(state["selection"]),
                editable=True,
                actions=["set_selection"],
            )
        )
        elements.append(
            Element(
                id="transport",
                type="transport",
                label="Transport",
                value=dict(state["transport"]),
                editable=True,
                actions=["set_transport"],
            )
        )
        elements.append(
            Element(
                id="export_settings",
                type="export",
                label="Export Settings",
                value=dict(state["export"]),
                editable=True,
                actions=["set_export"],
            )
        )

        return self._build_observation(
            source="file_parse",
            elements=elements,
            app_state={
                "current_view": "multitrack_timeline",
                "active_document": state["project_name"],
                "document_path": str(self.project_path),
            },
            environment={
                "system": {"sample_rate_hz": float(state["sample_rate_hz"])},
                "unsaved_changes": True,
            },
            data_summary=(
                f"Audacity project {state['project_name']} with {len(state['tracks'])} tracks, "
                f"{len(state['labels'])} labels, selection {state['selection']['start']}-{state['selection']['end']}s"
            ),
        )

    def execute(self, action: Action) -> Observation:
        if not self.validate_action(action):
            raise ValueError(f"Unsupported Audacity action: {action}")

        state = self._read_state()
        for operation in action.params.get("operations", []):
            op = operation.get("action")
            if op == "set_track_value":
                track = self._find_track(state, operation["track_id"])
                track[operation["field"]] = operation["value"]
            elif op == "add_track":
                new_track = {
                    "id": str(operation["track_id"]),
                    "name": str(operation["name"]),
                    "kind": str(operation.get("kind", "audio")),
                    "mute": bool(operation.get("mute", False)),
                    "solo": bool(operation.get("solo", False)),
                    "gain_db": float(operation.get("gain_db", 0.0)),
                    "pan": float(operation.get("pan", 0.0)),
                    "height_px": int(operation.get("height_px", 132)),
                    "clips": list(operation.get("clips", [])),
                }
                state["tracks"].append(new_track)
            elif op == "delete_track":
                track_id = str(operation["track_id"])
                state["tracks"] = [track for track in state["tracks"] if track["id"] != track_id]
            elif op == "move_track":
                track_id = str(operation["track_id"])
                new_index = max(0, int(operation.get("index", 0)))
                track = self._find_track(state, track_id)
                state["tracks"] = [existing for existing in state["tracks"] if existing["id"] != track_id]
                if new_index >= len(state["tracks"]):
                    state["tracks"].append(track)
                else:
                    state["tracks"].insert(new_index, track)
            elif op == "set_clip_value":
                clip = self._find_clip(state, operation["clip_id"])
                clip[operation["field"]] = operation["value"]
            elif op == "set_selection":
                state["selection"] = {
                    "start": float(operation["start"]),
                    "end": float(operation["end"]),
                    "focused_track_id": operation.get("focused_track_id", state["selection"].get("focused_track_id", "")),
                }
            elif op == "set_label":
                label = self._find_label(state, operation["label_id"])
                label.update(
                    {
                        "text": operation["text"],
                        "start": float(operation["start"]),
                        "end": float(operation["end"]),
                    }
                )
            elif op == "add_label":
                state["labels"].append(
                    {
                        "id": operation["label_id"],
                        "text": operation["text"],
                        "start": float(operation["start"]),
                        "end": float(operation["end"]),
                    }
                )
            elif op == "set_export":
                state["export"][operation["field"]] = operation["value"]
            elif op == "set_transport":
                state["transport"][operation["field"]] = operation["value"]
            else:
                raise ValueError(f"Unsupported Audacity operation: {op}")

        state["history"].append(f"Applied {len(action.params.get('operations', []))} operation(s)")
        self._write_state(state)
        return self.observe()

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="app_window",
            backend="x11-window-capture",
            actual_page=True,
            description="Screenshot of the real Audacity window showing the multitrack editor timeline",
        )

    def get_gui_session_spec(self) -> GUISessionSpec:
        audacity_bin = shutil.which("audacity")
        if audacity_bin is None:
            raise RuntimeError("audacity is not installed.")

        ensure_audio_backend(run_as_user="asilgui")

        home_path = self.project_path.parent / f"_audacity_gui_home_{uuid.uuid4().hex[:8]}"
        extra_env = self._prepare_gui_home(home_path)
        self._cleanup_script_pipe_files()
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=(audacity_bin,),
            window_title_pattern=r".*",
            window_class_pattern=r"Audacity",
            run_as_user="asilgui",
            startup_timeout_s=60.0,
            post_launch_delay_s=5.0,
            post_launch_callback=self._prime_editor_window,
            min_width=700,
            min_height=350,
            persist_shortcuts=("ctrl+s",),
            extra_env=extra_env,
        )

    def sync_from_gui(self, session=None) -> None:
        del session
        try:
            self._dismiss_welcome_dialog(timeout=3.0)
        except Exception:
            pass
        snapshot = self._read_gui_track_snapshot()
        if not snapshot:
            return

        state = self._read_state()
        existing_tracks = {track["id"]: deepcopy(track) for track in state["tracks"]}
        existing_by_name = {
            self._normalize_track_name(track["name"]): track["id"]
            for track in state["tracks"]
        }

        synced_tracks: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        if all(str(track_info.get("id") or "").strip() for track_info in snapshot):
            for index, track_info in enumerate(snapshot, start=1):
                track_name = str(track_info.get("name", "")).strip()
                if not track_name:
                    continue
                track_id = str(track_info.get("id") or "").strip()
                if not track_id:
                    track_id = existing_by_name.get(
                        self._normalize_track_name(track_name),
                        self._track_id_from_name(track_name, index=index),
                    )

                template = existing_tracks.get(track_id)
                if template is None:
                    template = self._new_track_template(track_id, track_name)
                template["id"] = track_id
                template["name"] = track_name
                synced_tracks.append(template)
                used_ids.add(track_id)
        else:
            synced_tracks = self._sync_tracks_from_visible_names(state["tracks"], snapshot)
            used_ids = {track["id"] for track in synced_tracks}

        if synced_tracks:
            state["tracks"] = synced_tracks
            focused_track_id = state.get("selection", {}).get("focused_track_id", "")
            if focused_track_id and focused_track_id not in used_ids:
                state["selection"]["focused_track_id"] = synced_tracks[0]["id"]
            self._write_state(state)

    @staticmethod
    def _new_track_template(track_id: str, track_name: str) -> dict[str, Any]:
        return {
            "id": track_id,
            "name": track_name,
            "kind": "audio",
            "mute": False,
            "solo": False,
            "gain_db": 0.0,
            "pan": 0.0,
            "height_px": 132,
            "clips": [],
        }

    @classmethod
    def _track_name_tokens(cls, name: str) -> set[str]:
        return {part for part in cls._normalize_track_name(name).split(" ") if part}

    @classmethod
    def _rename_cost(
        cls,
        existing_name: str,
        visible_name: str,
        *,
        existing_index: int,
        visible_index: int,
    ) -> float:
        existing_normalized = cls._normalize_track_name(existing_name)
        visible_normalized = cls._normalize_track_name(visible_name)
        if existing_normalized == visible_normalized:
            return 0.0
        if cls._track_name_tokens(existing_name) & cls._track_name_tokens(visible_name):
            return 0.25
        # When OCR only leaves one leading visible track after deletes/renames,
        # preserve the first canonical track id unless a later exact match wins.
        if existing_index == 0 and visible_index == 0:
            return 0.9
        return 2.1

    def _sync_tracks_from_visible_names(
        self,
        existing_tracks: list[dict[str, Any]],
        snapshot: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        visible_tracks = [track_info for track_info in snapshot if str(track_info.get("name", "")).strip()]
        visible_names = [str(track_info.get("name", "")).strip() for track_info in visible_tracks]

        @lru_cache(maxsize=None)
        def best_alignment(existing_index: int, visible_index: int) -> tuple[float, tuple[tuple[str, int, int], ...]]:
            if existing_index >= len(existing_tracks) and visible_index >= len(visible_names):
                return 0.0, ()

            candidates: list[tuple[float, tuple[tuple[str, int, int], ...], tuple[int, int, int]]] = []
            if existing_index < len(existing_tracks) and visible_index < len(visible_names):
                rename_cost = self._rename_cost(
                    existing_tracks[existing_index]["name"],
                    visible_names[visible_index],
                    existing_index=existing_index,
                    visible_index=visible_index,
                )
                next_cost, next_actions = best_alignment(existing_index + 1, visible_index + 1)
                candidates.append(
                    (
                        rename_cost + next_cost,
                        (("match", existing_index, visible_index),) + next_actions,
                        (0, existing_index, visible_index),
                    )
                )
            if existing_index < len(existing_tracks):
                next_cost, next_actions = best_alignment(existing_index + 1, visible_index)
                candidates.append(
                    (
                        1.0 + next_cost,
                        (("delete", existing_index, -1),) + next_actions,
                        (1, existing_index, visible_index),
                    )
                )
            if visible_index < len(visible_names):
                next_cost, next_actions = best_alignment(existing_index, visible_index + 1)
                candidates.append(
                    (
                        1.0 + next_cost,
                        (("add", -1, visible_index),) + next_actions,
                        (2, existing_index, visible_index),
                    )
                )
            best_cost, best_actions, _ = min(candidates, key=lambda item: (item[0], item[2]))
            return best_cost, best_actions

        _cost, actions = best_alignment(0, 0)
        synced_tracks: list[dict[str, Any]] = []
        for action, existing_index, visible_index in actions:
            if action == "delete":
                continue
            visible_track = visible_tracks[visible_index]
            visible_name = visible_names[visible_index]
            if action == "match":
                template = deepcopy(existing_tracks[existing_index])
                template["name"] = visible_name
                self._apply_visible_track_fields(template, visible_track)
                synced_tracks.append(template)
                continue
            track_id = self._track_id_from_name(visible_name, index=visible_index + 1)
            template = self._new_track_template(track_id, visible_name)
            self._apply_visible_track_fields(template, visible_track)
            synced_tracks.append(template)
        return synced_tracks

    @staticmethod
    def _apply_visible_track_fields(track: dict[str, Any], visible_track: dict[str, Any]) -> None:
        if "mute" in visible_track:
            track["mute"] = bool(visible_track["mute"])
        if "solo" in visible_track:
            track["solo"] = bool(visible_track["solo"])
        if "gain_db" in visible_track:
            track["gain_db"] = float(visible_track["gain_db"])
        if "pan" in visible_track:
            track["pan"] = float(visible_track["pan"])

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        output = Path(output_path) if output_path else self.project_path.with_suffix(".png")
        audacity_bin = shutil.which("audacity")
        if audacity_bin is None:
            raise RuntimeError("audacity is not installed.")

        ensure_audio_backend(run_as_user="asilgui")
        with tempfile.TemporaryDirectory(prefix="asil_audacity_home_") as home_root:
            home_path = Path(home_root)
            extra_env = self._prepare_gui_home(home_path)
            self._cleanup_script_pipe_files()
            proc = launch_gui_process(
                [audacity_bin],
                extra_env=extra_env,
                run_as_user="asilgui",
            )
            try:
                self._prime_editor_window()
                preferred_titles = self._preferred_window_titles()
                self._capture_editor_window(output, preferred_titles=preferred_titles)
            finally:
                terminate_process(proc)
        return output

    def _read_gui_track_snapshot(self) -> list[dict[str, Any]]:
        pipe_snapshot = self._read_gui_track_snapshot_via_script_pipe()
        if pipe_snapshot:
            return pipe_snapshot
        preferred_titles = self._preferred_window_titles()
        with tempfile.TemporaryDirectory(prefix="asil_audacity_sync_") as tmpdir:
            capture_path = Path(tmpdir) / "audacity-sync.png"
            try:
                self._dismiss_welcome_dialog(timeout=3.0)
            except Exception:
                pass
            self._capture_editor_window(capture_path, preferred_titles=preferred_titles)
            names = self._extract_track_names_from_image(capture_path)
        return [{"name": name} for name in names]

    def _read_gui_track_snapshot_via_script_pipe(self) -> list[dict[str, Any]]:
        try:
            pipe_to, pipe_from = self._wait_for_script_pipe(timeout=5.0)
            write_handle, read_handle = self._open_script_pipe_pair(pipe_to, pipe_from)
            with write_handle, read_handle:
                response = self._run_script_pipe_command(
                    write_handle,
                    read_handle,
                    "GetInfo: Type=Tracks Format=JSON",
                    timeout=5.0,
                )
        except Exception:
            return []
        return self._parse_track_snapshot_from_pipe_response(response)

    @staticmethod
    def _pipe_snapshot_lookup(item: dict[str, Any], *keys: str) -> Any:
        lowered = {str(key).lower(): value for key, value in item.items()}
        for key in keys:
            if key.lower() in lowered:
                return lowered[key.lower()]
        return None

    @classmethod
    def _parse_track_snapshot_from_pipe_response(cls, response: str) -> list[dict[str, Any]]:
        payload_lines: list[str] = []
        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("OK:"):
                continue
            if "BatchCommand finished" in line:
                break
            payload_lines.append(raw_line)
        payload = "\n".join(payload_lines).strip()
        if not payload:
            return []

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, dict):
            raw_tracks = parsed.get("tracks") or parsed.get("Tracks") or []
        elif isinstance(parsed, list):
            raw_tracks = parsed
        else:
            return []

        snapshot: list[dict[str, Any]] = []
        for item in raw_tracks:
            if not isinstance(item, dict):
                continue
            name = cls._pipe_snapshot_lookup(item, "name", "Name")
            if not name:
                continue
            snapshot_item: dict[str, Any] = {"name": str(name).strip()}
            mute = cls._pipe_snapshot_lookup(item, "mute", "Mute")
            solo = cls._pipe_snapshot_lookup(item, "solo", "Solo")
            gain = cls._pipe_snapshot_lookup(item, "gain_db", "gain", "Gain")
            pan = cls._pipe_snapshot_lookup(item, "pan", "Pan")
            if mute is not None:
                snapshot_item["mute"] = bool(mute)
            if solo is not None:
                snapshot_item["solo"] = bool(solo)
            if gain is not None:
                try:
                    snapshot_item["gain_db"] = float(gain)
                except (TypeError, ValueError):
                    pass
            if pan is not None:
                try:
                    snapshot_item["pan"] = float(pan)
                except (TypeError, ValueError):
                    pass
            snapshot.append(snapshot_item)
        return snapshot

    def _prime_editor_window(self) -> None:
        self._dismiss_welcome_dialog()
        window_id = self._wait_for_editor_window_id(timeout=60.0)
        self._focus_editor_window(window_id)
        pipe_to, pipe_from = self._wait_for_script_pipe(timeout=20.0)
        self._apply_state_via_script_pipe(pipe_to, pipe_from)
        self._dismiss_welcome_dialog(timeout=2.0)

    def _focus_editor_window(self, window_id: str) -> None:
        xdotool = shutil.which("xdotool")
        if xdotool is None:
            return
        subprocess.run(
            [xdotool, "windowactivate", "--sync", window_id],
            check=False,
            capture_output=True,
            env=self._gui_env(),
            text=True,
        )
        time.sleep(0.5)

    def _extract_track_names_from_image(self, image_path: Path) -> list[str]:
        tesseract_bin = shutil.which("tesseract")
        if tesseract_bin is None:
            return []

        with Image.open(image_path) as image:
            width, height = image.size
            crop = image.crop((0, 150, min(width, 150), max(150, height - 80)))
            enlarged = crop.convert("L").resize((crop.width * 3, crop.height * 3))
            processed = enlarged.point(lambda value: 0 if value < 170 else 255, mode="1")

            with tempfile.TemporaryDirectory(prefix="asil_audacity_ocr_") as tmpdir:
                processed_path = Path(tmpdir) / "audacity-track-headers.png"
                processed.save(processed_path)
                result = subprocess.run(
                    [tesseract_bin, str(processed_path), "stdout", "--psm", "6"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
        if result.returncode != 0:
            return []

        names: list[str] = []
        seen: set[str] = set()
        for raw_line in result.stdout.splitlines():
            candidate = self._candidate_track_name_from_ocr_line(raw_line)
            if not candidate:
                continue
            normalized = self._normalize_track_name(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            names.append(candidate)
        return names

    @staticmethod
    def _normalize_track_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()

    @classmethod
    def _canonicalize_track_name(cls, name: str) -> str:
        words = [part for part in cls._normalize_track_name(name).split(" ") if part]
        return " ".join(word.upper() if len(word) <= 3 else word.capitalize() for word in words)

    @classmethod
    def _track_id_from_name(cls, name: str, *, index: int) -> str:
        normalized = cls._normalize_track_name(name).replace(" ", "_")
        normalized = normalized or f"track_{index}"
        return f"track_{normalized}"

    @classmethod
    def _candidate_track_name_from_ocr_line(cls, raw_line: str) -> str:
        tokens = re.findall(r"[A-Za-z]+", raw_line)
        if not tokens:
            return ""

        stopwords = {
            "mute",
            "solo",
            "select",
            "mono",
            "float",
            "hz",
            "bit",
            "i",
            "wi",
            "ww",
            "mae",
            "or",
            "pr",
            "lr",
            "l",
            "r",
            "w",
            "s",
        }
        lexicon = (
            "voiceover",
            "music",
            "bed",
            "narration",
            "main",
            "ambience",
            "intro",
            "theme",
            "loop",
        )

        cleaned: list[str] = []
        for token in tokens:
            normalized = cls._normalize_track_name(token)
            if not normalized or normalized in stopwords:
                continue
            match = get_close_matches(normalized, lexicon, n=1, cutoff=0.6)
            if match:
                normalized = match[0]
            cleaned.append(normalized)
            if len(cleaned) >= 2:
                break

        if not cleaned:
            return ""
        return " ".join(word.capitalize() for word in cleaned)

    def _dismiss_welcome_dialog(self, *, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        xdotool = shutil.which("xdotool")
        wmctrl = shutil.which("wmctrl")
        if xdotool is None:
            raise RuntimeError("xdotool is required for Audacity GUI rendering but is not installed.")

        while time.time() < deadline:
            blocking_ids = []
            for pattern in self._blocking_window_patterns():
                blocking_ids.extend(self._search_window_ids(pattern))
            if not blocking_ids:
                for window_id in self._search_window_ids(".*"):
                    title = self._window_title(window_id)
                    if not self._is_editor_title(title):
                        continue
                    try:
                        width, height = self._window_size(window_id)
                    except Exception:
                        continue
                    if width >= 500 and height >= 150:
                        return
                time.sleep(0.5)
                continue
            for window_id in blocking_ids:
                try:
                    width, height = self._window_size(window_id)
                except Exception:
                    continue
                subprocess.run(
                    [xdotool, "windowraise", window_id],
                    check=False,
                    capture_output=True,
                    env=self._gui_env(),
                    text=True,
                )
                subprocess.run(
                    [xdotool, "windowactivate", "--sync", window_id],
                    check=False,
                    capture_output=True,
                    env=self._gui_env(),
                    text=True,
                )
                subprocess.run(
                    [xdotool, "key", "--window", window_id, "Escape"],
                    check=False,
                    capture_output=True,
                    env=self._gui_env(),
                    text=True,
                )
                subprocess.run(
                    [xdotool, "key", "--window", window_id, "Return"],
                    check=False,
                    capture_output=True,
                    env=self._gui_env(),
                    text=True,
                )
                subprocess.run(
                    [
                        xdotool,
                        "mousemove",
                        "--window",
                        window_id,
                        str(max(40, width - 44)),
                        str(max(24, height - 24)),
                        "click",
                        "1",
                    ],
                    check=False,
                    capture_output=True,
                    env=self._gui_env(),
                    text=True,
                )
                subprocess.run(
                    [
                        xdotool,
                        "mousemove",
                        "--window",
                        window_id,
                        str(max(140, width - 160)),
                        str(max(24, height - 36)),
                        "click",
                        "1",
                    ],
                    check=False,
                    capture_output=True,
                    env=self._gui_env(),
                    text=True,
                )
                subprocess.run(
                    [xdotool, "key", "--window", window_id, "alt+F4"],
                    check=False,
                    capture_output=True,
                    env=self._gui_env(),
                    text=True,
                )
                subprocess.run(
                    [xdotool, "windowclose", window_id],
                    check=False,
                    capture_output=True,
                    env=self._gui_env(),
                    text=True,
                )
                if wmctrl is not None:
                    subprocess.run(
                        [wmctrl, "-ic", window_id],
                        check=False,
                        capture_output=True,
                        env=self._gui_env(),
                        text=True,
                    )
                subprocess.run(
                    [xdotool, "windowkill", window_id],
                    check=False,
                    capture_output=True,
                    env=self._gui_env(),
                    text=True,
                )
            time.sleep(0.75)
            if not any(self._search_window_ids(pattern) for pattern in self._blocking_window_patterns()):
                return

    def _wait_for_editor_window_id(self, *, timeout: float = 60.0) -> str:
        deadline = time.time() + timeout
        preferred_titles = self._preferred_window_titles()

        while time.time() < deadline:
            candidates: list[tuple[int, str]] = []
            for window_id in self._search_window_ids(".*"):
                title = self._window_title(window_id)
                if not self._is_editor_title(title):
                    continue
                width, height = self._window_size(window_id)
                if width < 500 or height < 150:
                    continue
                score = self._editor_window_score(title, preferred_titles, width, height)
                if score > 0:
                    candidates.append((score, window_id))
            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][1]
            self._dismiss_welcome_dialog(timeout=1.0)
            time.sleep(0.5)

        preferred = ", ".join(sorted(preferred_titles))
        raise RuntimeError(f"Timed out waiting for an Audacity editor window. Preferred titles: {preferred}")

    def _capture_window_by_id(
        self,
        window_id: str,
        output_path: str | Path,
        *,
        preferred_titles: set[str] | None = None,
        settle_delay: float = 2.0,
    ) -> Path:
        tool = shutil.which("import")
        if tool is None:
            raise RuntimeError("ImageMagick 'import' is required for Audacity GUI rendering but is not installed.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if settle_delay > 0:
            time.sleep(settle_delay)

        try:
            left, top, width, height = self._window_geometry(window_id)
        except Exception:
            recovered_window_id = self._recover_editor_window_id(preferred_titles or set())
            left, top, width, height = self._window_geometry(recovered_window_id)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            root_capture = Path(handle.name)
        try:
            subprocess.run(
                [tool, "-display", self._gui_env()["DISPLAY"], "-window", "root", str(root_capture)],
                check=True,
                capture_output=True,
                env=self._gui_env(),
                text=True,
            )
            with Image.open(root_capture) as image:
                x0 = max(left, 0)
                y0 = max(top, 0)
                x1 = min(left + width, image.width)
                y1 = min(top + height, image.height)
                image.crop((x0, y0, x1, y1)).save(output)
                self._last_capture_complete = (
                    left >= 0
                    and top >= 0
                    and (left + width) <= image.width
                    and (top + height) <= image.height
                )
        finally:
            root_capture.unlink(missing_ok=True)
        assert_png_not_blank(output)
        return output

    def _capture_editor_window(
        self,
        output_path: str | Path,
        *,
        preferred_titles: set[str],
        timeout: float = 10.0,
    ) -> Path:
        deadline = time.time() + timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                live_window_id = self._recover_editor_window_id(preferred_titles)
                return self._capture_window_by_id(
                    live_window_id,
                    output_path,
                    preferred_titles=preferred_titles,
                    settle_delay=0.5,
                )
            except Exception as exc:
                last_error = exc
                time.sleep(0.75)
        try:
            time.sleep(0.75)
            return self._capture_root_display(output_path)
        except Exception as exc:
            last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unable to capture a live Audacity editor window.")

    def _capture_root_display(self, output_path: str | Path) -> Path:
        tool = shutil.which("import")
        if tool is None:
            raise RuntimeError("ImageMagick 'import' is required for Audacity GUI rendering but is not installed.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [tool, "-display", self._gui_env()["DISPLAY"], "-window", "root", str(output)],
            check=True,
            capture_output=True,
            env=self._gui_env(),
            text=True,
        )
        self._last_capture_complete = False
        assert_png_not_blank(output)
        return output

    def _recover_editor_window_id(self, preferred_titles: set[str]) -> str:
        candidates: list[tuple[int, str]] = []
        for window_id in self._search_window_ids(".*"):
            title = self._window_title(window_id)
            if not self._is_editor_title(title):
                continue
            try:
                width, height = self._window_size(window_id)
            except Exception:
                continue
            if width < 500 or height < 150:
                continue
            score = self._editor_window_score(title, preferred_titles, width, height)
            if score > 0:
                candidates.append((score, window_id))
        if not candidates:
            raise RuntimeError("Unable to recover a live Audacity editor window for screenshot capture.")
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _preferred_window_titles(self) -> set[str]:
        state = self._read_state()
        titles = {self.project_path.stem.lower(), state["project_name"].lower()}
        for track in state["tracks"]:
            stem = str(track["name"]).replace("_", " ").strip().lower()
            if stem:
                titles.add(stem)
        return {title for title in titles if title}

    def _preview_audio_plan(self) -> tuple[Path | None, list[Path]]:
        preview_audio = self._ensure_preview_audio_inputs()
        if not preview_audio:
            return None, []
        return preview_audio[0], preview_audio[1:]

    def _import_audio_file(self, editor_window_id: str, audio_path: Path) -> None:
        xdotool = shutil.which("xdotool")
        if xdotool is None:
            raise RuntimeError("xdotool is required for Audacity GUI rendering but is not installed.")

        env = self._gui_env()
        subprocess.run(
            [xdotool, "windowactivate", "--sync", editor_window_id],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )
        subprocess.run(
            [xdotool, "key", "--window", editor_window_id, "ctrl+shift+i"],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )
        time.sleep(1.0)
        subprocess.run(
            [xdotool, "key", "ctrl+l"],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
        time.sleep(0.2)
        subprocess.run(
            [xdotool, "type", "--delay", "0", str(audio_path)],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )
        subprocess.run(
            [xdotool, "key", "Return"],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )
        time.sleep(0.5)
        subprocess.run(
            [xdotool, "key", "Return"],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
        time.sleep(2.0)

    def _prepare_gui_home(self, home_root: Path) -> dict[str, str]:
        for subdir in ("home", "config", "data", "cache"):
            (home_root / subdir).mkdir(parents=True, exist_ok=True)
        ensure_user_access(home_root, run_as_user="asilgui")

        env = {
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "HOME": str(home_root / "home"),
            "XDG_CONFIG_HOME": str(home_root / "config"),
            "XDG_DATA_HOME": str(home_root / "data"),
            "XDG_CACHE_HOME": str(home_root / "cache"),
        }
        seed_cfg = self._ensure_seed_profile()
        target_cfg_dir = home_root / "home" / ".audacity-data"
        target_cfg_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(seed_cfg.parent, target_cfg_dir, dirs_exist_ok=True)
        ensure_user_access(home_root, run_as_user="asilgui")

        module_path = Path("/usr/lib/audacity/modules/mod-script-pipe.so")
        module_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(module_path.stat().st_mtime))
        cfg_path = target_cfg_dir / "audacity.cfg"
        lines = cfg_path.read_text(encoding="utf-8").splitlines() if cfg_path.exists() else []
        lines = self._upsert_pref(lines, "Directories", "TempDir", "/var/tmp/audacity-root")
        lines = self._upsert_pref(lines, "Module", "mod-script-pipe", "1")
        lines = self._upsert_pref(lines, "ModulePath", "mod-script-pipe", str(module_path))
        lines = self._upsert_pref(lines, "ModuleDateTime", "mod-script-pipe", module_timestamp)
        cfg_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

        return env

    def _ensure_seed_profile(self) -> Path:
        seed_dir = Path("/home/asilgui/.audacity-data")
        cfg_path = seed_dir / "audacity.cfg"
        if cfg_path.exists():
            return cfg_path
        audacity_bin = shutil.which("audacity")
        if audacity_bin is None:
            raise RuntimeError("audacity is not installed.")
        ensure_audio_backend(run_as_user="asilgui")
        env = self._gui_env()
        proc = launch_gui_process(
            [audacity_bin],
            run_as_user="asilgui",
        )
        try:
            deadline = time.time() + 8.0
            while time.time() < deadline:
                if cfg_path.exists():
                    return cfg_path
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
            self._request_audacity_shutdown(env)
        finally:
            terminate_process(proc)
        post_exit_deadline = time.time() + 10.0
        while time.time() < post_exit_deadline:
            if cfg_path.exists():
                return cfg_path
            time.sleep(0.5)
        raise RuntimeError("Audacity did not generate a seed profile.")

    @staticmethod
    def _find_audacity_cfg(home_root: Path) -> Path | None:
        matches = sorted(home_root.rglob("audacity.cfg"))
        return matches[0] if matches else None

    def _request_audacity_shutdown(self, env: dict[str, str]) -> None:
        wmctrl = shutil.which("wmctrl")
        xdotool = shutil.which("xdotool")
        if wmctrl is not None:
            for title in ("Welcome to Audacity!", "Audacity is starting up...", "Audacity"):
                subprocess.run(
                    [wmctrl, "-c", title],
                    check=False,
                    capture_output=True,
                    env=env,
                    text=True,
                )
        if xdotool is not None:
            result = subprocess.run(
                [xdotool, "search", "--class", "Audacity"],
                check=False,
                capture_output=True,
                env=env,
                text=True,
            )
            for window_id in result.stdout.splitlines():
                window_id = window_id.strip()
                if not window_id:
                    continue
                subprocess.run(
                    [xdotool, "windowactivate", "--sync", window_id],
                    check=False,
                    capture_output=True,
                    env=env,
                    text=True,
                )
                subprocess.run(
                    [xdotool, "key", "--window", window_id, "ctrl+q"],
                    check=False,
                    capture_output=True,
                    env=env,
                    text=True,
                )
                subprocess.run(
                    [xdotool, "key", "--window", window_id, "alt+F4"],
                    check=False,
                    capture_output=True,
                    env=env,
                    text=True,
                )

    @staticmethod
    def _upsert_pref(lines: list[str], section: str, key: str, value: str) -> list[str]:
        section_header = f"[{section}]"
        out = list(lines)
        section_start = None
        section_end = len(out)
        for index, line in enumerate(out):
            if line.strip() == section_header:
                section_start = index
                break
        if section_start is not None:
            for index in range(section_start + 1, len(out)):
                if out[index].startswith("[") and out[index].endswith("]"):
                    section_end = index
                    break
            for index in range(section_start + 1, section_end):
                if out[index].startswith(f"{key}="):
                    out[index] = f"{key}={value}"
                    return out
            out.insert(section_end, f"{key}={value}")
            return out
        if out and out[-1] != "":
            out.append("")
        out.extend([section_header, f"{key}={value}"])
        return out

    @staticmethod
    def _script_pipe_paths(username: str = "asilgui") -> tuple[Path, Path]:
        uid = pwd.getpwnam(username).pw_uid
        return (
            Path(f"/tmp/audacity_script_pipe.to.{uid}"),
            Path(f"/tmp/audacity_script_pipe.from.{uid}"),
        )

    def _cleanup_script_pipe_files(self) -> None:
        pipe_to, pipe_from = self._script_pipe_paths()
        for pipe_path in (pipe_to, pipe_from):
            try:
                pipe_path.unlink(missing_ok=True)
            except PermissionError:
                if os.geteuid() != 0:
                    raise
                pipe_path.chmod(0o666)
                pipe_path.unlink(missing_ok=True)

    def _make_script_pipe_accessible(self, pipe_to: Path, pipe_from: Path) -> None:
        if os.geteuid() != 0:
            return
        for pipe_path in (pipe_to, pipe_from):
            if pipe_path.exists():
                try:
                    pipe_path.chmod(0o666)
                except OSError:
                    pass

    def _script_pipe_ready_error(self, pipe_to: Path, pipe_from: Path) -> str | None:
        for pipe_path in (pipe_to, pipe_from):
            try:
                mode = pipe_path.stat().st_mode
            except OSError as exc:
                return str(exc)
            if not stat.S_ISFIFO(mode):
                return f"{pipe_path} exists but is not a FIFO"

        try:
            fd = os.open(pipe_to, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno == errno.ENXIO:
                return "Audacity command pipe exists but has no reader yet"
            if exc.errno in {errno.EACCES, errno.EPERM}:
                return f"Permission denied opening {pipe_to}"
            return str(exc)
        else:
            os.close(fd)

        try:
            fd = os.open(pipe_from, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EPERM}:
                return f"Permission denied opening {pipe_from}"
            return str(exc)
        else:
            os.close(fd)
        return None

    def _wait_for_script_pipe(self, *, timeout: float = 20.0) -> tuple[Path, Path]:
        deadline = time.time() + timeout
        pipe_to, pipe_from = self._script_pipe_paths()
        last_error = ""
        while time.time() < deadline:
            if pipe_to.exists() and pipe_from.exists():
                ready_error = self._script_pipe_ready_error(pipe_to, pipe_from)
                if ready_error is None:
                    return pipe_to, pipe_from
                last_error = ready_error
                if "Permission denied" in ready_error:
                    self._make_script_pipe_accessible(pipe_to, pipe_from)
            time.sleep(0.25)
        detail = f" Last error: {last_error}" if last_error else ""
        raise RuntimeError(f"Timed out waiting for Audacity mod-script-pipe.{detail}")

    def _apply_state_via_script_pipe(self, pipe_to: str | Path, pipe_from: str | Path) -> None:
        state = self._read_state()
        write_handle, read_handle = self._open_script_pipe_pair(pipe_to, pipe_from)
        with write_handle, read_handle:
            for track in state["tracks"]:
                self._run_script_pipe_command(write_handle, read_handle, "NewMonoTrack:")
                self._run_script_pipe_command(
                    write_handle,
                    read_handle,
                    f'SetTrackStatus: Name="{self._escape_pipe_value(track["name"])}"',
                )

    @staticmethod
    def _open_script_pipe_pair(pipe_to: str | Path, pipe_from: str | Path):
        # The readiness probe and these opens are separated by a race: Audacity
        # can close its pipe endpoint after the probe. Never block the benchmark
        # worker while reopening the FIFOs; restore blocking mode only after
        # both descriptors have been acquired successfully.
        write_fd = os.open(pipe_to, os.O_WRONLY | os.O_NONBLOCK)
        try:
            read_fd = os.open(pipe_from, os.O_RDONLY | os.O_NONBLOCK)
        except Exception:
            os.close(write_fd)
            raise
        try:
            os.set_blocking(write_fd, True)
            os.set_blocking(read_fd, True)
        except Exception:
            os.close(read_fd)
            os.close(write_fd)
            raise
        return (
            os.fdopen(write_fd, "w", encoding="utf-8", buffering=1),
            os.fdopen(read_fd, "r", encoding="utf-8", errors="replace", buffering=1),
        )

    @staticmethod
    def _escape_pipe_value(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    def _run_script_pipe_command(
        self,
        write_handle,
        read_handle,
        command: str,
        *,
        timeout: float = 5.0,
    ) -> str:
        write_handle.write(command + "\n")
        write_handle.flush()
        deadline = time.time() + timeout
        lines: list[str] = []
        while time.time() < deadline:
            line = read_handle.readline()
            if line == "":
                time.sleep(0.1)
                continue
            stripped = line.rstrip("\r\n")
            if not stripped:
                if lines:
                    return "\n".join(lines)
                continue
            lines.append(stripped)
            if "BatchCommand finished" in stripped:
                continue
        raise RuntimeError(f"Audacity script-pipe command timed out: {command}")

    def _editor_window_score(self, title: str, preferred_titles: set[str], width: int, height: int) -> int:
        normalized_title = " ".join(title.replace("_", " ").split()).lower()
        area = width * height
        if self._is_blocking_dialog_title(normalized_title):
            return 0
        if normalized_title == "audacity":
            return 2_000_000 + area
        if normalized_title in preferred_titles:
            return 1_000_000 + area
        if any(preferred in normalized_title or normalized_title in preferred for preferred in preferred_titles):
            return 900_000 + area
        return 100_000 + area

    def _is_editor_title(self, title: str) -> bool:
        normalized = " ".join(title.replace("_", " ").split()).lower()
        if not normalized:
            return False
        return not self._is_blocking_dialog_title(normalized)

    @staticmethod
    def _blocking_window_patterns() -> tuple[str, ...]:
        return (
            "^Welcome to Audacity!?$",
            ".*How to get help.*",
            ".*Automatic Crash Recovery.*",
            ".*Project Recovery.*",
            ".*Recovery.*",
            ".*Select Audio Host.*",
            ".*No devices found.*",
            ".*Missing Files.*",
        )

    @staticmethod
    def _is_blocking_dialog_title(normalized_title: str) -> bool:
        blocked_fragments = (
            "welcome to audacity",
            "audacity is starting up",
            "missing files",
            "how to get help",
            "automatic crash recovery",
            "project recovery",
            "recovery",
            "select audio host",
            "no devices found",
        )
        return any(fragment in normalized_title for fragment in blocked_fragments)

    def _search_window_ids(self, name_pattern: str) -> list[str]:
        xdotool = shutil.which("xdotool")
        if xdotool is None:
            raise RuntimeError("xdotool is required for Audacity GUI rendering but is not installed.")

        result = subprocess.run(
            [xdotool, "search", "--name", name_pattern],
            check=False,
            capture_output=True,
            env=self._gui_env(),
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []

        window_ids: list[str] = []
        for raw_id in result.stdout.splitlines():
            raw_id = raw_id.strip()
            if not raw_id:
                continue
            try:
                window_ids.append(hex(int(raw_id)))
            except ValueError:
                window_ids.append(raw_id)
        return window_ids

    def _window_title(self, window_id: str) -> str:
        xdotool = shutil.which("xdotool")
        if xdotool is None:
            raise RuntimeError("xdotool is required for Audacity GUI rendering but is not installed.")

        result = subprocess.run(
            [xdotool, "getwindowname", str(window_id)],
            check=False,
            capture_output=True,
            env=self._gui_env(),
            text=True,
        )
        return result.stdout.strip()

    def _window_size(self, window_id: str) -> tuple[int, int]:
        _left, _top, width, height = self._window_geometry(window_id)
        return width, height

    def _window_geometry(self, window_id: str) -> tuple[int, int, int, int]:
        xwininfo = shutil.which("xwininfo")
        if xwininfo is None:
            raise RuntimeError("xwininfo is required for Audacity GUI rendering but is not installed.")

        result = subprocess.run(
            [xwininfo, "-id", str(window_id)],
            check=True,
            capture_output=True,
            env=self._gui_env(),
            text=True,
        )
        left_match = re.search(r"Absolute upper-left X:\s+(-?\d+)", result.stdout)
        top_match = re.search(r"Absolute upper-left Y:\s+(-?\d+)", result.stdout)
        width_match = re.search(r"Width:\s+(\d+)", result.stdout)
        height_match = re.search(r"Height:\s+(\d+)", result.stdout)
        if left_match is None or top_match is None or width_match is None or height_match is None:
            raise RuntimeError(f"Unable to read Audacity window size for {window_id}.")
        return (
            int(left_match.group(1)),
            int(top_match.group(1)),
            int(width_match.group(1)),
            int(height_match.group(1)),
        )

    def _gui_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(ensure_virtual_display(run_as_user="asilgui"))
        return env

    def _timeline_html(self) -> str:
        state = self._read_state()
        track_rows = []
        no_clips_badge = "<span class='badge'>No clips</span>"
        for track in state["tracks"]:
            clip_badges = "".join(
                (
                    f"<span class='badge' style='margin-right:8px;'>"
                    f"{html.escape(clip['name'])} {clip['start']:.1f}s-{clip['end']:.1f}s"
                    "</span>"
                )
                for clip in track["clips"]
            )
            track_rows.append(
                "<div class='panel' style='padding:16px; margin-bottom:14px;'>"
                "<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<strong>{html.escape(track['name'])}</strong>"
                f"<span class='badge'>{'Muted' if track['mute'] else 'Live'} | "
                f"{'Solo' if track['solo'] else 'Blend'} | Gain {track['gain_db']} dB | Pan {track['pan']}</span>"
                "</div>"
                f"<div style='margin-top:10px;'>{clip_badges or no_clips_badge}</div>"
                "</div>"
            )

        label_rows = "".join(
            f"<tr><td>{html.escape(label['text'])}</td><td>{label['start']:.1f}s</td><td>{label['end']:.1f}s</td></tr>"
            for label in state["labels"]
        )
        body = (
            "<h1>Audacity Timeline</h1>"
            f"<p>Project <strong>{html.escape(state['project_name'])}</strong> at {state['sample_rate_hz']} Hz. "
            f"Selection {state['selection']['start']:.1f}s-{state['selection']['end']:.1f}s on "
            f"{html.escape(state['selection'].get('focused_track_id', '') or 'timeline')}.</p>"
            "<div style='display:grid; grid-template-columns: 2fr 1fr; gap: 18px;'>"
            "<div>"
            + "".join(track_rows)
            + "</div>"
            "<div>"
            "<div class='panel' style='padding:16px; margin-bottom:14px;'>"
            "<h2 style='margin-top:0;'>Transport</h2>"
            f"<p>Status: <strong>{html.escape(state['transport']['playback'])}</strong></p>"
            f"<p>Record armed: <strong>{'yes' if state['transport']['record_armed'] else 'no'}</strong></p>"
            "</div>"
            "<div class='panel' style='padding:16px; margin-bottom:14px;'>"
            "<h2 style='margin-top:0;'>Export</h2>"
            f"<p>{html.escape(state['export']['filename'])}</p>"
            f"<p>{html.escape(state['export']['format']).upper()} to {html.escape(state['export']['directory'])}</p>"
            "</div>"
            "<div class='panel' style='padding:16px;'>"
            "<h2 style='margin-top:0;'>Labels</h2>"
            "<table><thead><tr><th>Text</th><th>Start</th><th>End</th></tr></thead><tbody>"
            f"{label_rows}</tbody></table>"
            "</div>"
            "</div>"
            "</div>"
        )
        return html_page("Audacity Timeline", body)

    def _read_state(self) -> dict[str, Any]:
        return json.loads(self.project_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict[str, Any]) -> None:
        self.project_path.parent.mkdir(parents=True, exist_ok=True)
        self.project_path.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _ensure_preview_audio_inputs(self) -> list[Path]:
        state = self._read_state()
        audio_dir = self.project_path.parent / "_audacity_preview_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        sample_rate = int(state["sample_rate_hz"])
        duration_seconds = 2
        frame_count = sample_rate * duration_seconds
        amplitude = 10_000
        frequency = 330.0
        paths: list[Path] = []
        for index, track in enumerate(state["tracks"], start=1):
            safe_name = "".join(ch if ch.isalnum() else "_" for ch in track["name"]).strip("_") or f"track_{index}"
            audio_path = audio_dir / f"{safe_name}.wav"
            if not audio_path.exists():
                with wave.open(str(audio_path), "w") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    for frame in range(frame_count):
                        sample = int(amplitude * math.sin(2.0 * math.pi * frequency * frame / sample_rate))
                        wav_file.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True))
            paths.append(audio_path)

        if not paths:
            fallback = audio_dir / f"{state['project_name']}.wav"
            if not fallback.exists():
                with wave.open(str(fallback), "w") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    for frame in range(frame_count):
                        sample = int(amplitude * math.sin(2.0 * math.pi * frequency * frame / sample_rate))
                        wav_file.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True))
            paths.append(fallback)
        return paths

    @staticmethod
    def _find_track(state: dict[str, Any], track_id: str) -> dict[str, Any]:
        for track in state["tracks"]:
            if track["id"] == track_id:
                return track
        raise KeyError(track_id)

    @staticmethod
    def _find_clip(state: dict[str, Any], clip_id: str) -> dict[str, Any]:
        for track in state["tracks"]:
            for clip in track["clips"]:
                if clip["id"] == clip_id:
                    return clip
        raise KeyError(clip_id)

    @staticmethod
    def _find_label(state: dict[str, Any], label_id: str) -> dict[str, Any]:
        for label in state["labels"]:
            if label["id"] == label_id:
                return label
        raise KeyError(label_id)
