"""ASIL adapter for VLC Media Player — deterministic local playback-state mock."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

from PIL import ImageFont

from asil.adapter import ASILAdapter
from asil.protocol import Action, Element, Observation
from asil.rendering import (
    RenderArtifact,
    capture_window_to_png,
    ensure_audio_backend,
    ensure_user_access,
    launch_gui_process,
    send_keys_to_window,
    terminate_process,
)

_VLC_WINDOW_PATTERN = r"VLC media player|.* - VLC media player|.* - VLC"
_VLC_PRIVACY_DIALOG_PATTERN = "Privacy and Network Access Policy"


def _default_playlist() -> list[dict[str, Any]]:
    return [
        {
            "id": "media_01",
            "title": "City Lights Intro",
            "duration_seconds": 180,
            "audio_tracks": ["Stereo"],
            "subtitle_tracks": ["Off", "English CC"],
        },
        {
            "id": "media_02",
            "title": "Ocean Walk",
            "duration_seconds": 420,
            "audio_tracks": ["Stereo"],
            "subtitle_tracks": ["Off", "English CC", "Spanish"],
        },
        {
            "id": "media_03",
            "title": "Night Train Live",
            "duration_seconds": 510,
            "audio_tracks": ["Stereo", "Commentary"],
            "subtitle_tracks": ["Off", "English CC"],
        },
        {
            "id": "media_04",
            "title": "Studio Session",
            "duration_seconds": 600,
            "audio_tracks": ["Stereo", "Live Mix"],
            "subtitle_tracks": ["Off", "English CC"],
        },
    ]


class VLCAdapter(ASILAdapter):
    app_name = "VLC Media Player"
    supported_action_types = ["invoke_function"]

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)
        if not self.state_path.exists():
            self.setup_state("default")

    @classmethod
    def from_evaluation_context(
        cls,
        tmp: str | Path,
        sandbox=None,
        mock: bool = False,
    ) -> "VLCAdapter":
        del sandbox, mock
        return cls(Path(tmp) / "vlc_state.json")

    @property
    def source_path(self) -> Path:
        return self.state_path

    def clone(self, new_path: Path) -> "VLCAdapter":
        shutil.copy2(self.state_path, new_path)
        return VLCAdapter(new_path)

    def get_context(self) -> dict[str, str]:
        path = str(self.state_path)
        return {"state_path": path, "document_path": path}

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="app_window",
            backend="x11-window-capture",
            actual_page=True,
            description="Screenshot of the real VLC window showing the current media session",
        )

    def reset_state(self) -> None:
        self.setup_state("default")

    def setup_state(self, initial_state: str) -> None:
        self._write_state(self._initial_state(initial_state or "default"))

    def validate_action(self, action: Action) -> bool:
        return (
            action.action_type in self.supported_action_types
            and action.target == "vlc"
            and isinstance(action.params.get("operations"), list)
        )

    def observe(self) -> Observation:
        state = self._load_state()
        current_item = self._current_item(state)
        playlist = state["playlist"]
        progress = round(
            current_item["duration_seconds"] and state["position_seconds"] / current_item["duration_seconds"],
            3,
        )
        elements = [
            Element(
                id="player",
                type="player",
                label="Playback State",
                value={
                    "status": state["status"],
                    "current_title": current_item["title"],
                    "position_seconds": state["position_seconds"],
                    "duration_seconds": current_item["duration_seconds"],
                    "progress": progress,
                    "volume": state["volume"],
                    "muted": state["muted"],
                    "rate": state["rate"],
                    "repeat_mode": state["repeat_mode"],
                    "shuffle": state["shuffle"],
                    "subtitle_track": state["subtitle_track"],
                    "audio_track": state["audio_track"],
                    "playlist_index": state["current_index"] + 1,
                    "playlist_count": len(playlist),
                },
                editable=True,
                actions=[
                    "play",
                    "pause",
                    "stop",
                    "seek_to",
                    "set_volume",
                    "toggle_mute",
                    "set_playback_rate",
                    "set_repeat_mode",
                    "set_shuffle",
                    "set_subtitle_track",
                    "set_audio_track",
                    "set_current_media",
                ],
            ),
            Element(
                id="timeline",
                type="slider",
                label="Timeline",
                value={
                    "position_seconds": state["position_seconds"],
                    "duration_seconds": current_item["duration_seconds"],
                },
                editable=True,
                actions=["seek_to"],
            ),
            Element(
                id="volume_control",
                type="slider",
                label="Volume",
                value={"volume": state["volume"], "muted": state["muted"]},
                editable=True,
                actions=["set_volume", "toggle_mute"],
            ),
            Element(
                id="playback_rate",
                type="control",
                label="Playback Rate",
                value={"rate": state["rate"]},
                editable=True,
                actions=["set_playback_rate"],
            ),
            Element(
                id="playback_modes",
                type="control",
                label="Playback Modes",
                value={"repeat_mode": state["repeat_mode"], "shuffle": state["shuffle"]},
                editable=True,
                actions=["set_repeat_mode", "set_shuffle"],
            ),
            Element(
                id="track_selection",
                type="control",
                label="Track Selection",
                value={
                    "audio_track": state["audio_track"],
                    "subtitle_track": state["subtitle_track"],
                },
                editable=True,
                actions=["set_audio_track", "set_subtitle_track"],
            ),
        ]

        for index, item in enumerate(playlist, start=1):
            is_current = index - 1 == state["current_index"]
            elements.append(
                Element(
                    id=f"playlist:{index:02d}",
                    type="media_item",
                    label=item["title"],
                    value={
                        "title": item["title"],
                        "duration_seconds": item["duration_seconds"],
                        "is_current": is_current,
                        "status": state["status"] if is_current else "queued",
                    },
                    editable=True,
                    actions=["set_current_media"],
                    metadata={"playlist_index": index},
                )
            )

        return self._build_observation(
            source="state_mock",
            elements=elements,
            app_state={
                "current_view": "player",
                "active_document": current_item["title"],
                "document_path": str(self.state_path),
            },
            environment={
                "system": {
                    "playlist_size": float(len(playlist)),
                    "current_duration_seconds": float(current_item["duration_seconds"]),
                }
            },
            navigation={
                "current_path": "vlc/player",
                "breadcrumb": ["VLC", "Player"],
                "reachable_from_here": ["vlc/player", "vlc/playlist", "vlc/audio"],
            },
            data_summary=(
                f"{state['status'].title()} '{current_item['title']}' at {state['position_seconds']}s / "
                f"{current_item['duration_seconds']}s, volume {state['volume']}%, "
                f"shuffle={state['shuffle']}, repeat={state['repeat_mode']}"
            ),
        )

    def execute(self, action: Action) -> Observation:
        if not self.validate_action(action):
            raise ValueError(f"Unsupported VLC action: {action}")

        state = self._load_state()
        for operation in action.params.get("operations", []):
            op_name = operation.get("action")
            if op_name == "set_current_media":
                playlist_index = int(operation["playlist_index"]) - 1
                playlist_index = max(0, min(playlist_index, len(state["playlist"]) - 1))
                state["current_index"] = playlist_index
                state["position_seconds"] = int(operation.get("position_seconds", 0))
                self._sync_tracks(state)
                if operation.get("autoplay"):
                    state["status"] = "playing"
            elif op_name == "play":
                state["status"] = "playing"
            elif op_name == "pause":
                state["status"] = "paused"
            elif op_name == "stop":
                state["status"] = "stopped"
                state["position_seconds"] = 0
            elif op_name == "seek_to":
                state["position_seconds"] = self._clamp_position(state, int(operation["position_seconds"]))
            elif op_name == "set_volume":
                state["volume"] = max(0, min(int(operation["volume"]), 125))
            elif op_name == "toggle_mute":
                state["muted"] = bool(operation["muted"]) if "muted" in operation else not state["muted"]
            elif op_name == "set_playback_rate":
                state["rate"] = max(0.25, min(float(operation["rate"]), 4.0))
            elif op_name == "set_repeat_mode":
                repeat_mode = str(operation["repeat_mode"])
                if repeat_mode not in {"off", "one", "all"}:
                    raise ValueError(f"Unsupported repeat mode: {repeat_mode}")
                state["repeat_mode"] = repeat_mode
            elif op_name == "set_shuffle":
                state["shuffle"] = bool(operation["enabled"])
            elif op_name == "set_subtitle_track":
                track = str(operation["track"])
                if track not in self._current_item(state)["subtitle_tracks"]:
                    raise ValueError(f"Unsupported subtitle track for current media: {track}")
                state["subtitle_track"] = track
            elif op_name == "set_audio_track":
                track = str(operation["track"])
                if track not in self._current_item(state)["audio_tracks"]:
                    raise ValueError(f"Unsupported audio track for current media: {track}")
                state["audio_track"] = track
            else:
                raise ValueError(f"Unknown VLC operation: {op_name}")

        state["position_seconds"] = self._clamp_position(state, state["position_seconds"])
        self._write_state(state)
        return self.observe()

    def render_to_png(self, output_path: str | Path) -> Path:
        state = self._load_state()
        current_item = self._current_item(state)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        vlc_bin = shutil.which("vlc")
        if vlc_bin is None:
            raise RuntimeError("vlc is not installed.")

        media_path = self._ensure_preview_media(current_item)
        ensure_user_access(self.state_path.parent, run_as_user="asilgui")
        ensure_audio_backend(run_as_user="asilgui")
        proc = launch_gui_process(
            [vlc_bin, "--no-video-title-show", str(media_path)],
            run_as_user="asilgui",
        )
        try:
            self._dismiss_first_run_dialogs()
            send_keys_to_window(
                _VLC_WINDOW_PATTERN,
                ["Escape"],
                timeout=20.0,
                min_width=480,
                min_height=320,
            )
            capture_window_to_png(
                out,
                title_pattern=_VLC_WINDOW_PATTERN,
                timeout=45.0,
                margin=12,
                settle_delay=4.0,
                min_width=480,
                min_height=320,
            )
        finally:
            terminate_process(proc)
        return out

    def _dismiss_first_run_dialogs(self) -> None:
        try:
            send_keys_to_window(
                _VLC_PRIVACY_DIALOG_PATTERN,
                ["Return"],
                timeout=6.0,
                min_width=240,
                min_height=120,
            )
        except Exception:
            pass

    def _initial_state(self, initial_state: str) -> dict[str, Any]:
        state = {
            "playlist": _default_playlist(),
            "current_index": 0,
            "status": "stopped",
            "position_seconds": 0,
            "volume": 80,
            "muted": False,
            "rate": 1.0,
            "repeat_mode": "off",
            "shuffle": False,
            "subtitle_track": "Off",
            "audio_track": "Stereo",
        }

        if initial_state == "playing_default":
            state.update({"status": "playing", "position_seconds": 24})
        elif initial_state == "paused_on_second":
            state.update({"current_index": 1, "status": "paused", "position_seconds": 95, "volume": 60})
        elif initial_state == "playing_marathon":
            state.update(
                {
                    "current_index": 2,
                    "status": "playing",
                    "position_seconds": 210,
                    "volume": 35,
                    "shuffle": True,
                    "repeat_mode": "all",
                    "subtitle_track": "English CC",
                }
            )
        elif initial_state == "muted_showcase":
            state.update({"current_index": 3, "status": "paused", "position_seconds": 40, "volume": 50, "muted": True})
        elif initial_state == "subtitle_preview":
            state.update({"current_index": 1, "status": "paused", "position_seconds": 95, "subtitle_track": "English CC"})
        elif initial_state == "shuffled_queue":
            state.update({"current_index": 2, "status": "playing", "position_seconds": 180, "shuffle": True, "repeat_mode": "all"})
        elif initial_state == "fast_preview":
            state.update({"status": "paused", "position_seconds": 72, "rate": 1.5})

        self._sync_tracks(state)
        return state

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            self.setup_state("default")
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _current_item(self, state: dict[str, Any]) -> dict[str, Any]:
        return state["playlist"][int(state["current_index"])]

    def _sync_tracks(self, state: dict[str, Any]) -> None:
        current = self._current_item(state)
        if state["audio_track"] not in current["audio_tracks"]:
            state["audio_track"] = current["audio_tracks"][0]
        if state["subtitle_track"] not in current["subtitle_tracks"]:
            state["subtitle_track"] = current["subtitle_tracks"][0]

    def _clamp_position(self, state: dict[str, Any], position_seconds: int) -> int:
        duration = int(self._current_item(state)["duration_seconds"])
        return max(0, min(int(position_seconds), duration))

    @staticmethod
    def _font(size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", max(10, size))
        except OSError:
            return ImageFont.load_default()

    def _ensure_preview_media(self, item: dict[str, Any]) -> Path:
        media_dir = self.state_path.parent / "_vlc_preview_media"
        media_dir.mkdir(parents=True, exist_ok=True)
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", item["title"]).strip("_") or item["id"]
        media_path = media_dir / f"{filename}.mp4"
        if media_path.exists():
            return media_path

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg is required to generate preview video media for VLC rendering.")
        subprocess.run(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=960x540:rate=24",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100",
                "-shortest",
                "-t",
                "3",
                "-pix_fmt",
                "yuv420p",
                str(media_path),
            ],
            check=True,
            capture_output=True,
        )
        return media_path
