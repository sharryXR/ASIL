"""ASIL adapter for Kdenlive — deterministic project XML manipulation."""

from __future__ import annotations

import html
import math
import os
import re
import uuid
import subprocess
import shutil
import time
import wave
from pathlib import Path

from lxml import etree
from PIL import Image, ImageDraw

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.protocol import Action, Element, Observation
from asil.rendering import (
    RenderArtifact,
    capture_window_to_png,
    ensure_user_access,
    ensure_virtual_display,
    launch_gui_process,
    terminate_process,
)


_DEFAULT_PROJECT_XML = """<kdenliveProject profile="HD 1080p 30 fps" fps="30" width="1920" height="1080" proxy="0">
  <bin>
    <clip id="clip_intro" type="video" title="Intro Wide" resource="intro.mp4" duration="180" folder="Footage" />
    <clip id="clip_broll" type="video" title="City B-Roll" resource="city.mp4" duration="240" folder="Footage" />
    <clip id="clip_music" type="audio" title="Theme Bed" resource="theme.wav" duration="300" folder="Audio" />
    <clip id="clip_title" type="title" title="Opening Title" resource="opening_title.kdenlivetitle" title_text="Quarterly Update" duration="90" folder="Titles" />
  </bin>
  <timeline ruler_fps="30">
    <track id="video_main" kind="video" name="V1" muted="0" locked="0">
      <clipref id="tl_intro" clip_id="clip_intro" start="0" duration="120" in="0" out="120" />
      <clipref id="tl_title" clip_id="clip_title" start="120" duration="60" in="0" out="60" />
    </track>
    <track id="video_overlay" kind="video" name="V2" muted="0" locked="0">
      <clipref id="tl_broll" clip_id="clip_broll" start="150" duration="90" in="30" out="120" />
    </track>
    <track id="audio_main" kind="audio" name="A1" muted="0" locked="0">
      <clipref id="tl_music" clip_id="clip_music" start="0" duration="240" in="0" out="240" />
    </track>
  </timeline>
  <guides>
    <marker id="marker_intro" frame="120" comment="Title transition" color="#f59e0b" />
  </guides>
</kdenliveProject>
"""

_BASELINE_TRACK_IDS = {"video_main", "video_overlay", "audio_main"}


class KdenliveAdapter(ASILAdapter):
    app_name = "Kdenlive"
    supported_action_types = ["modify_file"]

    def __init__(self, project_path: str | Path) -> None:
        self.project_path = Path(project_path)
        self._tree: etree._ElementTree | None = None
        if not self.project_path.exists():
            self.setup_state("default")

    @classmethod
    def from_evaluation_context(
        cls,
        tmp: str | Path,
        sandbox=None,
        mock: bool = False,
    ) -> "KdenliveAdapter":
        del sandbox, mock
        return cls(Path(tmp) / "project.kdenlive")

    @property
    def source_path(self) -> Path:
        return self.project_path

    def clone(self, new_path: Path) -> "KdenliveAdapter":
        shutil.copy2(self.project_path, new_path)
        return KdenliveAdapter(new_path)

    def reset_state(self) -> None:
        self.setup_state("default")

    def setup_state(self, initial_state: str) -> None:
        del initial_state
        self.project_path.write_text(_DEFAULT_PROJECT_XML, encoding="utf-8")

    def get_context(self) -> dict[str, str]:
        return {
            "project_path": str(self.project_path),
            "kdenlive_path": str(self.project_path),
        }

    def _load(self) -> etree._Element:
        self._tree = etree.parse(str(self.project_path))
        return self._tree.getroot()

    def _save(self) -> None:
        assert self._tree is not None
        self._tree.write(
            str(self.project_path),
            xml_declaration=True,
            encoding="utf-8",
            pretty_print=True,
        )

    @staticmethod
    def _int(value: str | None, default: int = 0) -> int:
        if value is None or value == "":
            return default
        return int(value)

    @staticmethod
    def _bool(value: str | None) -> bool:
        return str(value or "0").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _element_id(prefix: str, raw_id: str | None) -> str:
        text = str(raw_id or "")
        return text if text.startswith(prefix) else f"{prefix}{text}"

    def observe(self) -> Observation:
        root = self._load()
        elements: list[Element] = []

        profile = root.get("profile", "")
        fps = self._int(root.get("fps"), 0)
        width = self._int(root.get("width"), 0)
        height = self._int(root.get("height"), 0)
        proxy_enabled = self._bool(root.get("proxy"))

        elements.append(
            Element(
                id="project_settings",
                type="project_settings",
                label="Project Settings",
                value={
                    "profile": profile,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "proxy_enabled": proxy_enabled,
                },
                editable=True,
                actions=["set_attribute"],
            )
        )

        bin_clips: dict[str, dict[str, object]] = {}
        for clip in root.xpath("./bin/clip"):
            clip_id = clip.get("id", "")
            clip_value = {
                "clip_type": clip.get("type", "video"),
                "title": clip.get("title", clip_id),
                "resource": clip.get("resource", ""),
                "duration": self._int(clip.get("duration"), 0),
                "folder": clip.get("folder", ""),
                "title_text": clip.get("title_text", ""),
            }
            bin_clips[clip_id] = clip_value
            elements.append(
                Element(
                    id=self._element_id("bin_clip:", clip_id),
                    type="bin_clip",
                    label=str(clip_value["title"]),
                    value=clip_value,
                    editable=True,
                    actions=["set_attribute", "delete"],
                )
            )

        tracks = root.xpath("./timeline/track")
        for index, track in enumerate(tracks):
            track_id = track.get("id", f"track_{index}")
            track_name = track.get("name", track_id)
            cliprefs = track.xpath("./clipref")
            elements.append(
                Element(
                    id=self._element_id("track:", track_id),
                    type="track",
                    label=track_name,
                    value={
                        "name": track_name,
                        "kind": track.get("kind", "video"),
                        "muted": self._bool(track.get("muted")),
                        "locked": self._bool(track.get("locked")),
                        "clip_count": len(cliprefs),
                    },
                    editable=True,
                    actions=["set_attribute", "add_element", "delete"],
                    metadata={"index": index},
                )
            )

            for clipref in cliprefs:
                timeline_id = clipref.get("id", "")
                clip_id = clipref.get("clip_id", "")
                clip_value = dict(bin_clips.get(clip_id, {}))
                clip_value.update(
                    {
                        "track": track_name,
                        "track_id": track_id,
                        "position": self._int(clipref.get("start"), 0),
                        "duration": self._int(clipref.get("duration"), 0),
                        "in": self._int(clipref.get("in"), 0),
                        "out": self._int(clipref.get("out"), 0),
                        "clip_id": clip_id,
                        "clip_title": clip_value.get("title", clip_id),
                    }
                )
                elements.append(
                    Element(
                        id=self._element_id("timeline_clip:", timeline_id),
                        type="timeline_clip",
                        label=str(clip_value["clip_title"]),
                        value=clip_value,
                        editable=True,
                        actions=["set_attribute", "delete"],
                    )
                )

        for marker in root.xpath("./guides/marker"):
            marker_id = marker.get("id", "")
            elements.append(
                Element(
                    id=self._element_id("marker:", marker_id),
                    type="marker",
                    label=marker.get("comment", marker_id),
                    value={
                        "frame": self._int(marker.get("frame"), 0),
                        "comment": marker.get("comment", ""),
                        "color": marker.get("color", ""),
                    },
                    editable=True,
                    actions=["set_attribute", "delete"],
                )
            )

        return self._build_observation(
            source="file_parse",
            elements=elements,
            app_state={
                "current_view": "timeline",
                "active_document": self.project_path.name,
                "document_path": str(self.project_path),
            },
            environment={
                "system": {
                    "fps": float(fps),
                    "width": float(width),
                    "height": float(height),
                },
                "unsaved_changes": False,
            },
            data_summary=(
                f"Kdenlive project with {len(bin_clips)} bin clips, "
                f"{len(tracks)} tracks, and {len(root.xpath('./guides/marker'))} markers"
            ),
        )

    def execute(self, action: Action) -> Observation:
        root = self._load()

        for operation in action.params.get("operations", []):
            op_action = operation.get("action", "set_attribute")

            if op_action == "set_attribute":
                xpath = operation.get("xpath", "")
                attribute = operation.get("attribute", "")
                value = operation.get("value", "")
                if not xpath or not attribute:
                    continue
                for target in root.xpath(xpath):
                    target.set(attribute, str(value))
                continue

            if op_action == "set_text":
                xpath = operation.get("xpath", "")
                value = operation.get("value", "")
                if not xpath:
                    continue
                for target in root.xpath(xpath):
                    target.text = str(value)
                continue

            if op_action == "add_element":
                parents = root.xpath(operation.get("parent_xpath", ""))
                if not parents:
                    continue
                attributes = {key: str(value) for key, value in operation.get("attributes", {}).items()}
                new_element = etree.SubElement(parents[0], operation.get("tag", "element"), attributes)
                if "text" in operation:
                    new_element.text = str(operation["text"])
                continue

            if op_action == "delete":
                xpath = operation.get("xpath", "")
                if not xpath:
                    continue
                for target in root.xpath(xpath):
                    parent = target.getparent()
                    if parent is not None:
                        parent.remove(target)
                continue

            raise ValueError(f"Unsupported Kdenlive operation: {op_action}")

        self._save()
        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="app_window",
            backend="x11-window-capture",
            actual_page=True,
            description="Screenshot of the real Kdenlive window showing the current project",
        )

    def get_gui_session_spec(self) -> GUISessionSpec:
        kdenlive_bin = shutil.which("kdenlive")
        if kdenlive_bin is None:
            raise RuntimeError("kdenlive is not installed.")
        preview_project = self._ensure_preview_project()
        ensure_user_access(preview_project.parent, run_as_user="asilgui")
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=(kdenlive_bin, str(preview_project)),
            window_title_pattern=r".*Kdenlive.*|kdenlive",
            window_class_pattern=r"kdenlive",
            run_as_user="asilgui",
            startup_timeout_s=60.0,
            post_launch_delay_s=6.0,
            post_launch_callback=self._dismiss_startup_dialogs,
            min_width=900,
            min_height=700,
            persist_shortcuts=("ctrl+s",),
            extra_env={"LIBGL_ALWAYS_SOFTWARE": "1"},
        )

    def sync_from_gui(self, session=None) -> None:
        """Import the state saved by Kdenlive into the canonical ASIL project.

        The GUI edits the native MLT preview, while evaluation reads the compact
        ``kdenliveProject`` document.  This method is deliberately a parser, not
        an action executor: it mirrors only state that Kdenlive actually saved.
        """

        del session
        preview_path = self.project_path.parent / "_kdenlive_preview_assets" / "preview_project.kdenlive"
        if not preview_path.exists():
            return

        preview_tree: etree._ElementTree | None = None
        parse_error: Exception | None = None
        for _attempt in range(3):
            try:
                preview_tree = etree.parse(str(preview_path))
                break
            except (OSError, etree.XMLSyntaxError) as exc:
                parse_error = exc
                time.sleep(0.05)
        if preview_tree is None:
            raise RuntimeError(f"Could not parse Kdenlive's saved project: {parse_error}")

        native_root = preview_tree.getroot()
        if native_root.tag != "mlt":
            raise RuntimeError(f"Unexpected Kdenlive preview root: {native_root.tag}")

        canonical_root = self._load()
        canonical_tracks = list(canonical_root.xpath("./timeline/track"))
        native_tracks = self._parse_native_timeline(native_root, canonical_root)
        if not native_tracks:
            raise RuntimeError("Kdenlive's saved project did not contain a native timeline graph.")

        native_profile = native_root.find("profile")
        if native_profile is not None:
            frame_rate_num = self._int(native_profile.get("frame_rate_num"), self._int(canonical_root.get("fps"), 25))
            frame_rate_den = max(1, self._int(native_profile.get("frame_rate_den"), 1))
            canonical_root.set("fps", str(max(1, round(frame_rate_num / frame_rate_den))))
            canonical_root.set("width", native_profile.get("width", canonical_root.get("width", "1920")))
            canonical_root.set("height", native_profile.get("height", canonical_root.get("height", "1080")))
            canonical_root.set("profile", native_profile.get("description", canonical_root.get("profile", "")))

        timeline_nodes = canonical_root.xpath("./timeline")
        if not timeline_nodes:
            timeline = etree.SubElement(canonical_root, "timeline")
        else:
            timeline = timeline_nodes[0]
        timeline.set("ruler_fps", canonical_root.get("fps", "25"))
        for track in list(timeline.xpath("./track")):
            timeline.remove(track)

        native_by_id = {str(track["id"]): track for track in native_tracks}
        ordered_ids = [track.get("id", "") for track in canonical_tracks if track.get("id", "") in native_by_id]
        ordered_ids.extend(str(track["id"]) for track in native_tracks if str(track["id"]) not in ordered_ids)
        for track_id in ordered_ids:
            state = native_by_id[track_id]
            track = etree.SubElement(
                timeline,
                "track",
                {
                    "id": track_id,
                    "kind": str(state["kind"]),
                    "name": str(state["name"]),
                    "muted": "1" if bool(state["muted"]) else "0",
                    "locked": "1" if bool(state["locked"]) else "0",
                },
            )
            for clip in state["clips"]:
                etree.SubElement(track, "clipref", {key: str(value) for key, value in clip.items()})

        self._save()

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        out = Path(output_path) if output_path else self.project_path.with_suffix(".png")
        kdenlive_bin = shutil.which("kdenlive")
        if kdenlive_bin is None:
            raise RuntimeError("kdenlive is not installed.")
        preview_project = self._ensure_preview_project()
        ensure_user_access(preview_project.parent, run_as_user="asilgui")

        proc = launch_gui_process(
            [kdenlive_bin, str(preview_project)],
            extra_env={"LIBGL_ALWAYS_SOFTWARE": "1"},
            run_as_user="asilgui",
        )
        try:
            self._dismiss_startup_dialogs()
            capture_metadata = {"capture_complete": True}
            capture_window_to_png(
                out,
                title_pattern="Kdenlive",
                window_class_pattern="kdenlive",
                timeout=60.0,
                margin=12,
                settle_delay=6.0,
                min_width=900,
                min_height=700,
                capture_metadata=capture_metadata,
            )
            self._last_capture_complete = bool(capture_metadata.get("capture_complete", True))
        finally:
            terminate_process(proc)
        return out

    def _dismiss_startup_dialogs(self, *, timeout: float = 12.0) -> None:
        xdotool = shutil.which("xdotool")
        if xdotool is None:
            raise RuntimeError("xdotool is required for Kdenlive GUI rendering but is not installed.")
        env = os.environ.copy()
        env.update(ensure_virtual_display(run_as_user="asilgui"))
        deadline = time.time() + timeout
        while time.time() < deadline:
            ids = subprocess.run(
                [xdotool, "search", "--class", "kdenlive"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.split()
            if ids:
                for wid in ids:
                    subprocess.run(
                        [xdotool, "key", "--window", wid, "Return"],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
            time.sleep(1.0)

    def _ensure_preview_project(self) -> Path:
        root = self._load()
        asset_dir = self.project_path.parent / "_kdenlive_preview_assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        fps = max(1, self._int(root.get("fps"), 25))
        width = max(1, self._int(root.get("width"), 1280))
        height = max(1, self._int(root.get("height"), 720))
        profile_description = self._profile_description(width, height, fps, self._bool(root.get("proxy")))

        title_file = asset_dir / "opening_title.kdenlivetitle"
        audio_file = asset_dir / "theme.wav"
        if not title_file.exists():
            title_file.write_text(
                "<kdenlivetitle width='1920' height='1080'><item type='QGraphicsTextItem' text='Quarterly Update' "
                "font='DejaVu Sans' font-size='72' color='255,255,255,255' x='420' y='420'/></kdenlivetitle>",
                encoding="utf-8",
            )
        if not audio_file.exists():
            sample_rate = 22_050
            audio_duration = max(
                [self._int(clip.get("duration"), fps * 2) for clip in root.xpath("./bin/clip[@type='audio']")]
                or [fps * 2]
            )
            frame_count = sample_rate * max(2, math.ceil(audio_duration / fps))
            amplitude = 8_000
            frequency = 392.0
            with wave.open(str(audio_file), "w") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                for frame in range(frame_count):
                    sample = int(amplitude * math.sin(2.0 * math.pi * frequency * frame / sample_rate))
                    wav_file.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True))

        mlt = etree.Element(
            "mlt",
            {
                "LC_NUMERIC": "C",
                "version": "7.4.0",
                "producer": "main_bin",
                "root": str(asset_dir),
            },
        )
        etree.SubElement(
            mlt,
            "profile",
            {
                "description": profile_description,
                "width": str(width),
                "height": str(height),
                "progressive": "1",
                "sample_aspect_num": "1",
                "sample_aspect_den": "1",
                "display_aspect_num": "16",
                "display_aspect_den": "9",
                "frame_rate_num": str(fps),
                "frame_rate_den": "1",
                "colorspace": "709",
            },
        )

        color_palette = ["#2f5fa7", "#487d5c", "#8b5cf6", "#f59e0b", "#ef4444", "#14b8a6"]
        producer_map: dict[str, str] = {}
        producer_numbers: dict[str, str] = {}
        bin_entries: list[tuple[str, int]] = []

        bin_clips = root.xpath("./bin/clip")
        for index, clip in enumerate(bin_clips, start=1):
            clip_id = clip.get("id", f"clip_{index}")
            producer_id = f"producer_{clip_id}"
            producer_map[clip_id] = producer_id
            producer_numbers[clip_id] = str(index)
            clip_type = clip.get("type", "video")
            title = clip.get("title", clip_id)
            title_text = clip.get("title_text", title)
            duration = max(1, self._int(clip.get("duration"), 75))
            color = color_palette[(index - 1) % len(color_palette)]
            control_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"kdenlive:{clip_id}:{title}:{clip_type}"))
            visual_asset = asset_dir / f"{clip_id}.png"
            if clip_type in {"video", "title"}:
                canvas = Image.new("RGB", (1280, 720), color)
                draw = ImageDraw.Draw(canvas)
                draw.rounded_rectangle((60, 60, 1220, 660), radius=32, outline="white", width=6)
                draw.text((96, 96), title, fill="white")
                if title_text and title_text != title:
                    draw.text((96, 156), title_text, fill="white")
                canvas.save(visual_asset, format="PNG")

            producer = etree.SubElement(
                mlt,
                "producer",
                {
                    "id": producer_id,
                    "in": self._frames_to_timecode(0, fps),
                    "out": self._frames_to_timecode(duration - 1, fps),
                },
            )
            if clip_type in {"video", "title"}:
                self._append_mlt_property(producer, "eof", "pause")
                self._append_mlt_property(producer, "resource", str(visual_asset))
                self._append_mlt_property(producer, "ttl", str(fps))
                self._append_mlt_property(producer, "aspect_ratio", "1")
                self._append_mlt_property(producer, "progressive", "1")
                self._append_mlt_property(producer, "seekable", "1")
                self._append_mlt_property(producer, "meta.media.width", "1280")
                self._append_mlt_property(producer, "meta.media.height", "720")
                self._append_mlt_property(producer, "mlt_service", "qimage")
            elif clip_type == "audio":
                self._append_mlt_property(producer, "resource", str(audio_file))
                self._append_mlt_property(producer, "mlt_service", "avformat-novalidate")
            else:
                self._append_mlt_property(producer, "resource", f"color:{color}")
                self._append_mlt_property(producer, "mlt_service", "color")
            self._append_mlt_property(producer, "length", self._frames_to_timecode(duration, fps))
            self._append_mlt_property(producer, "kdenlive:duration", self._frames_to_timecode(duration, fps))
            self._append_mlt_property(producer, "kdenlive:clipname", title)
            self._append_mlt_property(producer, "kdenlive:folderid", "-1")
            self._append_mlt_property(producer, "kdenlive:id", str(index))
            self._append_mlt_property(producer, "kdenlive:clip_type", clip_type)
            self._append_mlt_property(producer, "kdenlive:producer_type", "1")
            self._append_mlt_property(producer, "kdenlive:control_uuid", control_uuid)
            self._append_mlt_property(producer, "kdenlive:asil_clip_id", clip_id)
            self._append_mlt_property(producer, "_kdenlive_processed", "1")
            bin_entries.append((producer_id, duration))

        main_bin = etree.SubElement(mlt, "playlist", {"id": "main_bin"})
        document_properties = {
            "kdenlive:docproperties.activeTrack": "2",
            "kdenlive:docproperties.audioChannels": "2",
            "kdenlive:docproperties.audioTarget": "-1",
            "kdenlive:docproperties.disablepreview": "0",
            "kdenlive:docproperties.documentid": str(uuid.uuid5(uuid.NAMESPACE_URL, str(self.project_path)).int)[:13],
            "kdenlive:docproperties.enableTimelineZone": "0",
            "kdenlive:docproperties.enableproxy": "1" if self._bool(root.get("proxy")) else "0",
            "kdenlive:docproperties.generateproxy": "0",
            "kdenlive:docproperties.kdenliveversion": "21.12.3",
            "kdenlive:docproperties.position": "0",
            "kdenlive:docproperties.profile": self._kdenlive_profile_id(width, height, fps),
            "kdenlive:docproperties.scrollPos": "0",
            "kdenlive:docproperties.seekOffset": "30000",
            "kdenlive:docproperties.version": "1.04",
            "kdenlive:docproperties.verticalzoom": "1",
            "kdenlive:docproperties.videoTarget": "-1",
            "kdenlive:docproperties.zonein": "0",
            "kdenlive:docproperties.zoneout": str(max(fps * 3, 75)),
            "kdenlive:docproperties.zoom": "8",
            "kdenlive:docproperties.groups": "[\n]\n",
            "xml_retain": "1",
        }
        for name, value in document_properties.items():
            self._append_mlt_property(main_bin, name, value)
        for producer_id, duration in bin_entries:
            etree.SubElement(
                main_bin,
                "entry",
                {
                    "producer": producer_id,
                    "in": self._frames_to_timecode(0, fps),
                    "out": self._frames_to_timecode(duration - 1, fps),
                },
            )

        project_span = fps * 20 * 60
        black_track = etree.SubElement(
            mlt,
            "producer",
            {
                "id": "black_track",
                "in": self._frames_to_timecode(0, fps),
                "out": self._frames_to_timecode(project_span, fps),
            },
        )
        self._append_mlt_property(black_track, "length", "2147483647")
        self._append_mlt_property(black_track, "eof", "continue")
        self._append_mlt_property(black_track, "resource", "black")
        self._append_mlt_property(black_track, "aspect_ratio", "1")
        self._append_mlt_property(black_track, "mlt_service", "color")
        self._append_mlt_property(black_track, "mlt_image_format", "rgba")
        self._append_mlt_property(black_track, "set.test_audio", "0")

        timeline_tracks = root.xpath("./timeline/track")
        native_track_order = [track for track in reversed(timeline_tracks) if track.get("kind") == "audio"]
        native_track_order.extend(track for track in timeline_tracks if track.get("kind", "video") != "audio")
        logical_tractors: list[tuple[etree._Element, etree._Element]] = []
        timeline_end = 1
        for track_index, track in enumerate(native_track_order):
            track_id = track.get("id", f"track_{track_index}")
            track_name = track.get("name", track_id)
            kind = track.get("kind", "video")
            playlist_ids = [f"playlist_asil_{track_id}_0", f"playlist_asil_{track_id}_1"]
            playlists = [etree.SubElement(mlt, "playlist", {"id": playlist_id}) for playlist_id in playlist_ids]
            if kind == "audio":
                for playlist in playlists:
                    self._append_mlt_property(playlist, "kdenlive:audio_track", "1")

            position = 0
            primary_playlist = playlists[0]
            for clipref in sorted(track.xpath("./clipref"), key=lambda node: self._int(node.get("start"), 0)):
                start = self._int(clipref.get("start"), 0)
                duration = max(1, self._int(clipref.get("duration"), 1))
                clip_id = clipref.get("clip_id", "")
                producer_id = producer_map.get(clip_id)
                if producer_id is None:
                    continue
                if start > position:
                    etree.SubElement(
                        primary_playlist,
                        "blank",
                        {"length": self._frames_to_timecode(start - position, fps)},
                    )
                clip_in = self._int(clipref.get("in"), 0)
                entry = etree.SubElement(
                    primary_playlist,
                    "entry",
                    {
                        "producer": producer_id,
                        "in": self._frames_to_timecode(clip_in, fps),
                        "out": self._frames_to_timecode(clip_in + duration - 1, fps),
                    },
                )
                self._append_mlt_property(entry, "kdenlive:id", producer_numbers[clip_id])
                self._append_mlt_property(entry, "kdenlive:asil_clipref_id", clipref.get("id", clip_id))
                position = start + duration
                timeline_end = max(timeline_end, position)

            logical_tractor = etree.SubElement(
                mlt,
                "tractor",
                {
                    "id": f"tractor_asil_{track_id}",
                    "in": self._frames_to_timecode(0, fps),
                    "out": self._frames_to_timecode(max(0, position - 1), fps),
                },
            )
            if kind == "audio":
                self._append_mlt_property(logical_tractor, "kdenlive:audio_track", "1")
            self._append_mlt_property(logical_tractor, "kdenlive:trackheight", "74")
            self._append_mlt_property(logical_tractor, "kdenlive:timeline_active", "1")
            self._append_mlt_property(logical_tractor, "kdenlive:collapsed", "0")
            self._append_mlt_property(logical_tractor, "kdenlive:thumbs_format", "")
            self._append_mlt_property(logical_tractor, "kdenlive:audio_rec", "")
            self._append_mlt_property(logical_tractor, "kdenlive:track_name", track_name)
            self._append_mlt_property(logical_tractor, "kdenlive:locked_track", "1" if self._bool(track.get("locked")) else "0")
            self._append_mlt_property(logical_tractor, "kdenlive:asil_track_id", track_id)
            for playlist_id in playlist_ids:
                attrs = {"producer": playlist_id, "hide": "video" if kind == "audio" else "audio"}
                etree.SubElement(logical_tractor, "track", attrs)
            if kind == "audio":
                self._append_audio_filters(logical_tractor, prefix=f"track_{track_index}")
            logical_tractors.append((track, logical_tractor))

        master_tractor = etree.SubElement(
            mlt,
            "tractor",
            {
                "id": "tractor_main",
                "in": self._frames_to_timecode(0, fps),
                "out": self._frames_to_timecode(project_span + timeline_end, fps),
            },
        )
        etree.SubElement(master_tractor, "track", {"producer": "black_track"})
        for original_track, logical_tractor in logical_tractors:
            attrs = {"producer": logical_tractor.get("id", "")}
            if self._bool(original_track.get("muted")):
                attrs["hide"] = "both"
            etree.SubElement(master_tractor, "track", attrs)

        for index, (track, _logical_tractor) in enumerate(logical_tractors, start=1):
            transition = etree.SubElement(master_tractor, "transition", {"id": f"transition_asil_{index}"})
            self._append_mlt_property(transition, "a_track", "0")
            self._append_mlt_property(transition, "b_track", str(index))
            if track.get("kind") == "audio":
                self._append_mlt_property(transition, "mlt_service", "mix")
                self._append_mlt_property(transition, "kdenlive_id", "mix")
                self._append_mlt_property(transition, "accepts_blanks", "1")
                self._append_mlt_property(transition, "sum", "1")
            else:
                self._append_mlt_property(transition, "version", "0.9")
                self._append_mlt_property(transition, "mlt_service", "frei0r.cairoblend")
                self._append_mlt_property(transition, "kdenlive_id", "frei0r.cairoblend")
            self._append_mlt_property(transition, "internal_added", "237")
            self._append_mlt_property(transition, "always_active", "1")
        self._append_audio_filters(master_tractor, prefix="master")

        preview_path = asset_dir / "preview_project.kdenlive"
        preview_path.write_bytes(
            etree.tostring(
                mlt,
                xml_declaration=True,
                encoding="utf-8",
                pretty_print=True,
            )
        )
        return preview_path

    def _parse_native_timeline(
        self,
        native_root: etree._Element,
        canonical_root: etree._Element,
    ) -> list[dict[str, object]]:
        fps = self._native_fps(native_root, default=self._int(canonical_root.get("fps"), 25))
        tractors = {tractor.get("id", ""): tractor for tractor in native_root.xpath("./tractor")}
        master_candidates: list[tuple[int, etree._Element]] = []
        for tractor in tractors.values():
            track_refs = [track.get("producer", "") for track in tractor.xpath("./track")]
            logical_refs = [ref for ref in track_refs if ref in tractors]
            if "black_track" in track_refs and logical_refs:
                master_candidates.append((len(logical_refs), tractor))
        if not master_candidates:
            return []
        master = max(master_candidates, key=lambda item: item[0])[1]

        canonical_tracks = list(canonical_root.xpath("./timeline/track"))
        canonical_by_id = {track.get("id", ""): track for track in canonical_tracks}
        canonical_bin = {clip.get("id", ""): clip for clip in canonical_root.xpath("./bin/clip")}
        producer_clip_ids = self._native_producer_clip_ids(native_root, canonical_root)
        logical_pairs = [
            (master_track, tractors[producer_id])
            for master_track in master.xpath("./track")
            if (producer_id := master_track.get("producer", "")) in tractors
        ]
        native_track_ids = self._assign_native_track_ids(logical_pairs, canonical_tracks)

        logical_states: list[dict[str, object]] = []
        for logical_index, (master_track, tractor) in enumerate(logical_pairs):
            kind = "audio" if self._bool(self._mlt_property(tractor, "kdenlive:audio_track")) else "video"
            name = self._mlt_property(tractor, "kdenlive:track_name")
            track_id = native_track_ids[logical_index]
            if not name:
                existing = canonical_by_id.get(track_id)
                name = existing.get("name", track_id) if existing is not None else track_id

            existing_track = canonical_by_id.get(track_id)
            old_cliprefs = list(existing_track.xpath("./clipref")) if existing_track is not None else []
            unmatched_old = list(old_cliprefs)
            used_clipref_ids: set[str] = set()
            clips: list[dict[str, str]] = []
            playlist_ids = [track.get("producer", "") for track in tractor.xpath("./track")]
            for playlist_index, playlist_id in enumerate(playlist_ids):
                playlist_nodes = native_root.xpath(f"./playlist[@id={self._xpath_literal(playlist_id)}]")
                if not playlist_nodes:
                    continue
                position = 0
                for child in playlist_nodes[0]:
                    if child.tag == "blank":
                        position += self._timecode_to_frames(child.get("length"), fps)
                        continue
                    if child.tag != "entry":
                        continue
                    clip_in = self._timecode_to_frames(child.get("in"), fps)
                    clip_out = self._timecode_to_frames(child.get("out"), fps)
                    duration = max(1, clip_out - clip_in + 1)
                    clip_id = producer_clip_ids.get(child.get("producer", ""), "")
                    if not clip_id:
                        numeric_id = self._mlt_property(child, "kdenlive:id")
                        clip_id = producer_clip_ids.get(f"numeric:{numeric_id}", "")
                    if not clip_id or clip_id not in canonical_bin:
                        raise RuntimeError(
                            f"Could not map native Kdenlive producer {child.get('producer', '')!r} "
                            "to a canonical bin clip."
                        )

                    clipref_id = self._mlt_property(child, "kdenlive:asil_clipref_id")
                    if not clipref_id or clipref_id in used_clipref_ids:
                        clipref_id = self._match_existing_clipref_id(
                            unmatched_old,
                            clip_id=clip_id,
                            start=position,
                            duration=duration,
                            clip_in=clip_in,
                        )
                    if not clipref_id or clipref_id in used_clipref_ids:
                        clipref_id = self._derive_timeline_clip_id(
                            track_id=track_id,
                            track_name=name,
                            clip_id=clip_id,
                            start=position,
                            old_cliprefs=old_cliprefs,
                        )
                    base_id = clipref_id
                    suffix = 2
                    while clipref_id in used_clipref_ids:
                        clipref_id = f"{base_id}_{suffix}"
                        suffix += 1
                    used_clipref_ids.add(clipref_id)
                    clips.append(
                        {
                            "id": clipref_id,
                            "clip_id": clip_id,
                            "start": str(position),
                            "duration": str(duration),
                            "in": str(clip_in),
                            "out": str(clip_in + duration),
                            "_playlist_index": str(playlist_index),
                        }
                    )
                    position += duration

            clips.sort(key=lambda clip: (self._int(clip["start"]), self._int(clip["_playlist_index"])))
            for clip in clips:
                clip.pop("_playlist_index", None)
            master_hide = str(master_track.get("hide", "")).lower()
            muted = master_hide == "both" or (kind == "audio" and "audio" in master_hide) or (
                kind == "video" and "video" in master_hide
            )
            logical_states.append(
                {
                    "id": track_id,
                    "kind": kind,
                    "name": name,
                    "muted": muted,
                    "locked": self._bool(self._mlt_property(tractor, "kdenlive:locked_track")),
                    "clips": clips,
                }
            )
        return logical_states

    def _native_producer_clip_ids(
        self,
        native_root: etree._Element,
        canonical_root: etree._Element,
    ) -> dict[str, str]:
        canonical_clips = list(canonical_root.xpath("./bin/clip"))
        numeric_ids = {str(index): clip.get("id", "") for index, clip in enumerate(canonical_clips, start=1)}
        titles = {clip.get("title", ""): clip.get("id", "") for clip in canonical_clips}
        resources = {Path(clip.get("resource", "")).name: clip.get("id", "") for clip in canonical_clips}
        result = {f"numeric:{number}": clip_id for number, clip_id in numeric_ids.items()}
        for producer in native_root.xpath("./producer | ./chain"):
            producer_id = producer.get("id", "")
            clip_id = self._mlt_property(producer, "kdenlive:asil_clip_id")
            numeric_id = self._mlt_property(producer, "kdenlive:id")
            if not clip_id and numeric_id:
                clip_id = numeric_ids.get(numeric_id, "")
            if not clip_id:
                title = self._mlt_property(producer, "kdenlive:clipname")
                clip_id = titles.get(title, "")
            if not clip_id:
                resource_name = Path(self._mlt_property(producer, "resource")).name
                clip_id = resources.get(resource_name, "")
                if not clip_id and resource_name.startswith("clip_"):
                    candidate = Path(resource_name).stem
                    if canonical_root.xpath(f"./bin/clip[@id={self._xpath_literal(candidate)}]"):
                        clip_id = candidate
                if not clip_id and resource_name == "theme.wav" and "clip_music" in {
                    clip.get("id", "") for clip in canonical_clips
                }:
                    clip_id = "clip_music"
            if clip_id:
                result[producer_id] = clip_id
                if numeric_id:
                    result[f"numeric:{numeric_id}"] = clip_id
        return result

    def _assign_native_track_ids(
        self,
        logical_pairs: list[tuple[etree._Element, etree._Element]],
        canonical_tracks: list[etree._Element],
    ) -> list[str]:
        canonical_ids = {track.get("id", "") for track in canonical_tracks}
        descriptors = [
            {
                "kind": "audio" if self._bool(self._mlt_property(tractor, "kdenlive:audio_track")) else "video",
                "name": self._mlt_property(tractor, "kdenlive:track_name"),
                "explicit_id": self._mlt_property(tractor, "kdenlive:asil_track_id"),
            }
            for _master_track, tractor in logical_pairs
        ]
        assignments: list[str | None] = [None] * len(descriptors)
        assigned_ids: set[str] = set()

        for index, descriptor in enumerate(descriptors):
            explicit_id = str(descriptor["explicit_id"])
            if explicit_id in canonical_ids and explicit_id not in assigned_ids:
                assignments[index] = explicit_id
                assigned_ids.add(explicit_id)

        for index, descriptor in enumerate(descriptors):
            if assignments[index] is not None:
                continue
            exact_matches = [
                track.get("id", "")
                for track in canonical_tracks
                if track.get("kind", "video") == descriptor["kind"]
                and track.get("name", "") == descriptor["name"]
                and track.get("id", "") not in assigned_ids
            ]
            if len(exact_matches) == 1:
                assignments[index] = exact_matches[0]
                assigned_ids.add(exact_matches[0])

        for kind in ("audio", "video"):
            native_indexes = [
                index
                for index, descriptor in enumerate(descriptors)
                if descriptor["kind"] == kind and assignments[index] is None
            ]
            remaining_canonical = [
                track.get("id", "")
                for track in canonical_tracks
                if track.get("kind", "video") == kind
                and track.get("id", "") in _BASELINE_TRACK_IDS
                and track.get("id", "") not in assigned_ids
            ]
            if len(native_indexes) == len(remaining_canonical):
                for index, track_id in zip(native_indexes, remaining_canonical):
                    assignments[index] = track_id
                    assigned_ids.add(track_id)

        for index, descriptor in enumerate(descriptors):
            if assignments[index] is not None:
                continue
            kind = str(descriptor["kind"])
            name = str(descriptor["name"])
            track_slug = self._canonical_track_slug(name or f"track_{index}")
            derived = f"{'audio' if kind == 'audio' else 'video'}_{track_slug}"
            track_id = derived
            suffix = 2
            while track_id in assigned_ids:
                track_id = f"{derived}_{suffix}"
                suffix += 1
            assignments[index] = track_id
            assigned_ids.add(track_id)

        return [str(track_id) for track_id in assignments]

    def _match_existing_clipref_id(
        self,
        unmatched: list[etree._Element],
        *,
        clip_id: str,
        start: int,
        duration: int,
        clip_in: int,
    ) -> str:
        for clipref in list(unmatched):
            if (
                clipref.get("clip_id", "") == clip_id
                and self._int(clipref.get("start")) == start
                and self._int(clipref.get("duration")) == duration
                and self._int(clipref.get("in")) == clip_in
            ):
                unmatched.remove(clipref)
                return clipref.get("id", "")
        return ""

    def _derive_timeline_clip_id(
        self,
        *,
        track_id: str,
        track_name: str,
        clip_id: str,
        start: int,
        old_cliprefs: list[etree._Element],
    ) -> str:
        clip_token = self._slug(clip_id.removeprefix("clip_"))
        if track_id == "video_main":
            if clip_token == "intro" and any(ref.get("clip_id") == clip_id for ref in old_cliprefs):
                return "tl_intro_return"
            return f"tl_{clip_token}_main_{start}"
        if track_id == "video_overlay":
            return f"tl_{clip_token}_overlay_{start}"
        if track_id.startswith("video_"):
            track_token = track_id.removeprefix("video_")
            if track_token.endswith("_cue"):
                track_token = track_token.removesuffix("_cue")
            return f"tl_{track_token}_{clip_token}"
        return f"tl_{clip_token}_{self._slug(track_name)}_{start}"

    @staticmethod
    def _native_fps(native_root: etree._Element, *, default: int) -> int:
        profile = native_root.find("profile")
        if profile is None:
            return max(1, default)
        numerator = KdenliveAdapter._int(profile.get("frame_rate_num"), default)
        denominator = max(1, KdenliveAdapter._int(profile.get("frame_rate_den"), 1))
        return max(1, round(numerator / denominator))

    @staticmethod
    def _timecode_to_frames(value: str | None, fps: int) -> int:
        text = str(value or "0").strip()
        if not text:
            return 0
        if ":" not in text:
            try:
                return max(0, round(float(text) * fps))
            except ValueError:
                return 0
        try:
            hours, minutes, seconds = text.split(":", 2)
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except (TypeError, ValueError):
            return 0
        return max(0, round(total_seconds * fps))

    @staticmethod
    def _mlt_property(parent: etree._Element, name: str) -> str:
        for prop in parent.xpath("./property"):
            if prop.get("name") == name:
                return str(prop.text or "")
        return ""

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "track"

    @classmethod
    def _canonical_track_slug(cls, name: str) -> str:
        slug = cls._slug(name)
        return {
            "lower_third_cue": "lower_third",
            "review_layer": "review",
        }.get(slug, slug)

    @staticmethod
    def _xpath_literal(value: str) -> str:
        if "'" not in value:
            return f"'{value}'"
        if '"' not in value:
            return f'"{value}"'
        parts = value.split("'")
        return "concat(" + ", \"'\", ".join(f"'{part}'" for part in parts) + ")"

    @staticmethod
    def _append_mlt_property(parent: etree._Element, name: str, value: str) -> etree._Element:
        prop = etree.SubElement(parent, "property", {"name": name})
        prop.text = value
        return prop

    @classmethod
    def _append_audio_filters(cls, tractor: etree._Element, *, prefix: str) -> None:
        volume = etree.SubElement(tractor, "filter", {"id": f"filter_{prefix}_volume"})
        cls._append_mlt_property(volume, "window", "75")
        cls._append_mlt_property(volume, "max_gain", "20dB")
        cls._append_mlt_property(volume, "mlt_service", "volume")
        cls._append_mlt_property(volume, "internal_added", "237")
        cls._append_mlt_property(volume, "disable", "1")

        panner = etree.SubElement(tractor, "filter", {"id": f"filter_{prefix}_panner"})
        cls._append_mlt_property(panner, "channel", "-1")
        cls._append_mlt_property(panner, "mlt_service", "panner")
        cls._append_mlt_property(panner, "internal_added", "237")
        cls._append_mlt_property(panner, "start", "0.5")
        cls._append_mlt_property(panner, "disable", "1")

        audio_level = etree.SubElement(tractor, "filter", {"id": f"filter_{prefix}_audiolevel"})
        cls._append_mlt_property(audio_level, "iec_scale", "0")
        cls._append_mlt_property(audio_level, "mlt_service", "audiolevel")
        cls._append_mlt_property(audio_level, "disable", "1")

    @staticmethod
    def _kdenlive_profile_id(width: int, height: int, fps: int) -> str:
        if width == 1920 and height == 1080:
            return f"atsc_1080p_{fps}"
        if width == 1280 and height == 720:
            return f"atsc_720p_{fps}"
        return "atsc_1080p_25"

    @staticmethod
    def _frames_to_timecode(frames: int, fps: int) -> str:
        fps = max(1, fps)
        frames = max(0, int(frames))
        hours, remainder = divmod(frames, fps * 3600)
        minutes, remainder = divmod(remainder, fps * 60)
        seconds, frame = divmod(remainder, fps)
        milliseconds = round(frame * 1000 / fps)
        if milliseconds == 1000:
            milliseconds = 0
            seconds += 1
            if seconds == 60:
                seconds = 0
                minutes += 1
                if minutes == 60:
                    minutes = 0
                    hours += 1
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    @staticmethod
    def _profile_description(width: int, height: int, fps: int, proxy_enabled: bool) -> str:
        if width == 1920 and height == 1080:
            base = "HD 1080p"
        elif width == 1280 and height == 720:
            base = "HD 720p"
        else:
            base = f"{width}x{height}"
        description = f"{base} {fps} fps"
        if proxy_enabled:
            description += " Proxy"
        return description
