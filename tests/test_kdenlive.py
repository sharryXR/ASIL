import json
from pathlib import Path
from unittest.mock import patch

import pytest
from lxml import etree

from asil.eval.runner import TaskDefinition, run_task
from asil.eval.task_audit import audit_task_file
from asil.protocol import Action

from asil.adapters.kdenlive import KdenliveAdapter


def _write_project(path: Path) -> Path:
    path.write_text(
        """<kdenliveProject profile="HD 1080p 30 fps" fps="30" width="1920" height="1080" proxy="0">
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
</kdenliveProject>""",
        encoding="utf-8",
    )
    return path


def _kdenlive_task(task_id: str) -> dict:
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "kdenlive"
    return json.loads((root / f"{task_id}.json").read_text(encoding="utf-8"))


def _all_kdenlive_tasks() -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in _all_kdenlive_task_paths()]


def _all_kdenlive_task_paths() -> list[Path]:
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "kdenlive"
    return sorted(path for path in root.glob("kdenlive_*.json") if path.stem.removeprefix("kdenlive_").isdigit())


def _mlt_property(parent: etree._Element, name: str) -> str:
    return parent.xpath(f"string(./property[@name='{name}'])")


def _native_track(root: etree._Element, track_id: str) -> etree._Element:
    tracks = root.xpath(f"./tractor[property[@name='kdenlive:asil_track_id'] = '{track_id}']")
    assert len(tracks) == 1
    return tracks[0]


def test_observe_returns_project_bin_track_timeline_and_marker_elements(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))

    obs = adapter.observe()

    assert obs.meta.app_name == "Kdenlive"
    assert obs.meta.observation_source == "file_parse"
    assert obs.app_state.current_view == "timeline"
    assert len(obs.interactive_elements) == 13

    project = next(e for e in obs.interactive_elements if e.id == "project_settings")
    track = next(e for e in obs.interactive_elements if e.id == "track:video_main")
    clip = next(e for e in obs.interactive_elements if e.id == "timeline_clip:tl_title")
    marker = next(e for e in obs.interactive_elements if e.id == "marker:marker_intro")

    assert project.value["fps"] == 30
    assert project.value["proxy_enabled"] is False
    assert track.value["name"] == "V1"
    assert track.value["kind"] == "video"
    assert clip.value["clip_type"] == "title"
    assert clip.value["clip_title"] == "Opening Title"
    assert clip.value["title_text"] == "Quarterly Update"
    assert marker.value["frame"] == 120


def test_execute_updates_attributes_adds_elements_and_deletes_nodes(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))

    obs = adapter.execute(
        Action(
            action_type="modify_file",
            target=str(adapter.source_path),
            params={
                "operations": [
                    {
                        "xpath": "/kdenliveProject/timeline/track[@id='video_main']",
                        "attribute": "name",
                        "value": "Primary Video",
                    },
                    {
                        "action": "add_element",
                        "parent_xpath": "/kdenliveProject/guides",
                        "tag": "marker",
                        "attributes": {
                            "id": "marker_review",
                            "frame": "300",
                            "comment": "Review beat",
                            "color": "#22c55e",
                        },
                    },
                    {
                        "action": "delete",
                        "xpath": "/kdenliveProject/guides/marker[@id='marker_intro']",
                    },
                ]
            },
        )
    )

    track = next(e for e in obs.interactive_elements if e.id == "track:video_main")
    marker = next(e for e in obs.interactive_elements if e.id == "marker:marker_review")

    assert track.value["name"] == "Primary Video"
    assert marker.value["comment"] == "Review beat"
    assert "marker:marker_intro" not in {e.id for e in obs.interactive_elements}


def test_clone_and_context_copy_project_file(tmp_path: Path):
    source = _write_project(tmp_path / "project.kdenlive")
    adapter = KdenliveAdapter(source)
    clone_path = tmp_path / "cloned.kdenlive"

    cloned = adapter.clone(clone_path)

    assert clone_path.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert cloned.get_context()["project_path"] == str(clone_path)
    assert cloned.get_context()["kdenlive_path"] == str(clone_path)


@patch("asil.adapters.kdenlive.terminate_process")
@patch("asil.adapters.kdenlive.ensure_user_access")
@patch.object(KdenliveAdapter, "_dismiss_startup_dialogs")
@patch.object(KdenliveAdapter, "_ensure_preview_project")
@patch("asil.adapters.kdenlive.capture_window_to_png")
@patch("asil.adapters.kdenlive.launch_gui_process")
@patch("asil.adapters.kdenlive.shutil.which", return_value="/usr/bin/kdenlive")
def test_rendering_metadata_and_png_capture_use_real_window(
    mock_which,
    mock_launch,
    mock_capture,
    mock_ensure_preview_project,
    mock_dismiss_startup_dialogs,
    mock_ensure_user_access,
    mock_terminate,
    tmp_path: Path,
):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))
    preview_project = tmp_path / "preview_project.kdenlive"
    preview_project.write_text("<mlt/>", encoding="utf-8")
    mock_ensure_preview_project.return_value = preview_project
    mock_launch.return_value = object()

    artifact = adapter.describe_rendering()
    out = adapter.render_to_png(tmp_path / "project.png")

    assert artifact.actual_page is True
    assert "real" in artifact.description.lower()
    assert artifact.backend == "x11-window-capture"
    mock_which.assert_called_once_with("kdenlive")
    mock_launch.assert_called_once()
    launch_args, launch_kwargs = mock_launch.call_args
    assert launch_args[0][0] == "/usr/bin/kdenlive"
    assert launch_args[0] == ["/usr/bin/kdenlive", str(preview_project)]
    mock_ensure_user_access.assert_called_once_with(preview_project.parent, run_as_user="asilgui")
    mock_dismiss_startup_dialogs.assert_called_once_with()
    mock_capture.assert_called_once_with(
        tmp_path / "project.png",
        title_pattern="Kdenlive",
        window_class_pattern="kdenlive",
        timeout=60.0,
        margin=12,
        settle_delay=6.0,
        min_width=900,
        min_height=700,
        capture_metadata={"capture_complete": True},
    )
    mock_terminate.assert_called_once_with(mock_launch.return_value)
    assert out == tmp_path / "project.png"


def test_preview_project_reflects_current_track_names_and_bin_entries(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))
    adapter.execute(
        Action(
            action_type="modify_file",
            target=str(adapter.source_path),
            params={
                "operations": [
                    {
                        "xpath": "/kdenliveProject/timeline/track[@id='video_main']",
                        "attribute": "name",
                        "value": "Primary Video",
                    },
                    {
                        "xpath": "/kdenliveProject/bin/clip[@id='clip_broll']",
                        "attribute": "title",
                        "value": "Cutaway Skyline",
                    },
                ]
            },
        )
    )

    preview_path = adapter._ensure_preview_project()
    preview_text = preview_path.read_text(encoding="utf-8")

    assert '<playlist id="main_bin">' in preview_text
    assert "Primary Video" in preview_text
    assert "Cutaway Skyline" in preview_text
    assert "kdenlive:track_name" in preview_text


def test_preview_project_profile_description_reflects_current_settings(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))
    adapter.execute(
        Action(
            action_type="modify_file",
            target=str(adapter.source_path),
            params={
                "operations": [
                    {
                        "xpath": "/kdenliveProject",
                        "attribute": "fps",
                        "value": "24",
                    },
                    {
                        "xpath": "/kdenliveProject",
                        "attribute": "width",
                        "value": "1280",
                    },
                    {
                        "xpath": "/kdenliveProject",
                        "attribute": "height",
                        "value": "720",
                    },
                    {
                        "xpath": "/kdenliveProject",
                        "attribute": "proxy",
                        "value": "1",
                    },
                ]
            },
        )
    )

    preview_text = adapter._ensure_preview_project().read_text(encoding="utf-8")

    assert 'description="HD 720p 24 fps Proxy"' in preview_text


def test_preview_project_uses_image_assets_for_visual_clips(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))

    preview_path = adapter._ensure_preview_project()
    preview_text = preview_path.read_text(encoding="utf-8")

    assert "clip_intro.png" in preview_text
    assert "clip_broll.png" in preview_text
    assert "clip_title.png" in preview_text


def test_preview_project_uses_native_kdenlive_bin_and_logical_track_graph(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))

    preview_path = adapter._ensure_preview_project()
    root = etree.parse(str(preview_path)).getroot()

    assert root.get("producer") == "main_bin"
    main_bin = root.xpath("./playlist[@id='main_bin']")[0]
    assert main_bin.xpath("./property[@name='kdenlive:docproperties.kdenliveversion']/text()") == ["21.12.3"]
    assert len(main_bin.xpath("./entry")) == 4

    producer_ids = {producer.get("id") for producer in root.xpath("./producer | ./chain")}
    assert {entry.get("producer") for entry in main_bin.xpath("./entry")} <= producer_ids

    master_tracks = [
        tractor
        for tractor in root.xpath("./tractor")
        if tractor.xpath("./track[1][@producer='black_track']")
    ]
    assert len(master_tracks) == 1
    logical_tractor_ids = [
        track.get("producer")
        for track in master_tracks[0].xpath("./track[position() > 1]")
    ]
    logical_tractors = {
        tractor.get("id"): tractor
        for tractor in root.xpath("./tractor")
        if tractor.get("id") in logical_tractor_ids
    }
    assert set(logical_tractors) == set(logical_tractor_ids)

    track_ids = {
        tractor.xpath("string(./property[@name='kdenlive:asil_track_id'])")
        for tractor in logical_tractors.values()
    }
    assert track_ids == {"video_main", "video_overlay", "audio_main"}
    for tractor in logical_tractors.values():
        playlist_ids = [track.get("producer") for track in tractor.xpath("./track")]
        assert len(playlist_ids) == 2
        assert all(root.xpath(f"./playlist[@id='{playlist_id}']") for playlist_id in playlist_ids)


def test_sync_from_gui_imports_native_track_and_easy_timeline_edits(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))
    preview_path = adapter._ensure_preview_project()
    preview_tree = etree.parse(str(preview_path))
    native_root = preview_tree.getroot()

    video_main = _native_track(native_root, "video_main")
    name_property = video_main.xpath("./property[@name='kdenlive:track_name']")[0]
    name_property.text = "Primary Video"
    lock_property = video_main.xpath("./property[@name='kdenlive:locked_track']")[0]
    lock_property.text = "1"

    producers_by_clip_id = {
        _mlt_property(producer, "kdenlive:asil_clip_id"): producer.get("id")
        for producer in native_root.xpath("./producer | ./chain")
        if _mlt_property(producer, "kdenlive:asil_clip_id")
    }
    title_producer = producers_by_clip_id["clip_title"]
    main_playlist_id = video_main.xpath("./track[1]/@producer")[0]
    main_playlist = native_root.xpath(f"./playlist[@id='{main_playlist_id}']")[0]
    title_entry = main_playlist.xpath(f"./entry[@producer='{title_producer}']")[0]
    main_playlist.remove(title_entry)

    video_overlay = _native_track(native_root, "video_overlay")
    overlay_playlist_id = video_overlay.xpath("./track[1]/@producer")[0]
    overlay_playlist = native_root.xpath(f"./playlist[@id='{overlay_playlist_id}']")[0]
    existing_broll_entry = overlay_playlist.xpath("./entry")[0]
    for child in list(overlay_playlist):
        if child.tag in {"blank", "entry"}:
            overlay_playlist.remove(child)
    etree.SubElement(overlay_playlist, "blank", {"length": adapter._frames_to_timecode(24, 30)})
    title_entry = etree.SubElement(
        overlay_playlist,
        "entry",
        {
            "producer": title_producer,
            "in": adapter._frames_to_timecode(0, 30),
            "out": adapter._frames_to_timecode(59, 30),
        },
    )
    etree.SubElement(title_entry, "property", {"name": "kdenlive:id"}).text = "4"
    etree.SubElement(overlay_playlist, "blank", {"length": adapter._frames_to_timecode(66, 30)})
    overlay_playlist.append(existing_broll_entry)
    preview_tree.write(str(preview_path), xml_declaration=True, encoding="utf-8", pretty_print=True)

    adapter.sync_from_gui()

    canonical = etree.parse(str(adapter.source_path)).getroot()
    main_track = canonical.xpath("./timeline/track[@id='video_main']")[0]
    assert main_track.get("name") == "Primary Video"
    assert main_track.get("locked") == "1"
    assert not main_track.xpath("./clipref[@id='tl_title']")
    added_title = canonical.xpath("./timeline/track[@id='video_overlay']/clipref[@id='tl_title_overlay_24']")[0]
    assert added_title.attrib == {
        "id": "tl_title_overlay_24",
        "clip_id": "clip_title",
        "start": "24",
        "duration": "60",
        "in": "0",
        "out": "60",
    }


def test_sync_from_gui_matches_renamed_track_after_kdenlive_rewrites_native_ids(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))
    preview_path = adapter._ensure_preview_project()
    preview_tree = etree.parse(str(preview_path))
    native_root = preview_tree.getroot()

    video_main = _native_track(native_root, "video_main")
    video_main.xpath("./property[@name='kdenlive:track_name']")[0].text = "Primary Video"
    for prop in native_root.xpath(
        ".//property[@name='kdenlive:asil_track_id' or @name='kdenlive:asil_clipref_id']"
    ):
        prop.getparent().remove(prop)
    preview_tree.write(str(preview_path), xml_declaration=True, encoding="utf-8", pretty_print=True)

    adapter.sync_from_gui()

    canonical = etree.parse(str(adapter.source_path)).getroot()
    tracks = {track.get("id"): track for track in canonical.xpath("./timeline/track")}
    assert set(tracks) == {"video_main", "video_overlay", "audio_main"}
    assert tracks["video_main"].get("name") == "Primary Video"
    assert tracks["video_overlay"].get("name") == "V2"


def test_sync_from_gui_assigns_easy_broll_id_from_secondary_timeline_lane(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))
    preview_path = adapter._ensure_preview_project()
    preview_tree = etree.parse(str(preview_path))
    native_root = preview_tree.getroot()
    video_main = _native_track(native_root, "video_main")
    secondary_playlist_id = video_main.xpath("./track[2]/@producer")[0]
    secondary_playlist = native_root.xpath(f"./playlist[@id='{secondary_playlist_id}']")[0]
    broll_producer = native_root.xpath(
        "./producer[property[@name='kdenlive:asil_clip_id'] = 'clip_broll']/@id"
    )[0]
    etree.SubElement(secondary_playlist, "blank", {"length": adapter._frames_to_timecode(30, 30)})
    broll_entry = etree.SubElement(
        secondary_playlist,
        "entry",
        {
            "producer": broll_producer,
            "in": adapter._frames_to_timecode(0, 30),
            "out": adapter._frames_to_timecode(71, 30),
        },
    )
    etree.SubElement(broll_entry, "property", {"name": "kdenlive:id"}).text = "2"
    preview_tree.write(str(preview_path), xml_declaration=True, encoding="utf-8", pretty_print=True)

    adapter.sync_from_gui()

    canonical = etree.parse(str(adapter.source_path)).getroot()
    added_broll = canonical.xpath("./timeline/track[@id='video_main']/clipref[@id='tl_broll_main_30']")[0]
    assert added_broll.get("clip_id") == "clip_broll"
    assert added_broll.get("start") == "30"
    assert added_broll.get("duration") == "72"


@pytest.mark.parametrize(
    ("track_name", "expected_track_id", "clip_id", "start", "duration", "expected_clipref_id"),
    [
        ("Guest Cue", "video_guest_cue", "clip_title", 264, 72, "tl_guest_title"),
        ("Lower Third Cue", "video_lower_third", "clip_title", 96, 48, "tl_lower_third_title"),
        ("Interview Cue", "video_interview_cue", "clip_broll", 300, 60, "tl_interview_broll"),
        ("Cutaways", "video_cutaways", None, 0, 0, None),
        ("Review Layer", "video_review", None, 0, 0, None),
        ("CTA Cue", "video_cta_cue", "clip_title", 420, 48, "tl_cta_title"),
        ("Review Cue", "video_review_cue", "clip_broll", 180, 72, "tl_review_broll"),
        ("Outro Cue", "video_outro_cue", "clip_intro", 360, 90, "tl_outro_intro"),
    ],
)
def test_sync_from_gui_derives_stable_ids_for_new_native_track_and_clip(
    tmp_path: Path,
    track_name: str,
    expected_track_id: str,
    clip_id: str | None,
    start: int,
    duration: int,
    expected_clipref_id: str | None,
):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))
    preview_path = adapter._ensure_preview_project()
    preview_tree = etree.parse(str(preview_path))
    native_root = preview_tree.getroot()
    master = native_root.xpath("./tractor[track[1][@producer='black_track']]")[0]

    playlist_a = etree.Element("playlist", {"id": "playlist_gui_new_0"})
    if clip_id is not None:
        etree.SubElement(playlist_a, "blank", {"length": adapter._frames_to_timecode(start, 30)})
        producer = native_root.xpath(
            f"./producer[property[@name='kdenlive:asil_clip_id'] = '{clip_id}']/@id"
        )[0]
        entry = etree.SubElement(
            playlist_a,
            "entry",
            {
                "producer": producer,
                "in": adapter._frames_to_timecode(0, 30),
                "out": adapter._frames_to_timecode(duration - 1, 30),
            },
        )
        etree.SubElement(entry, "property", {"name": "kdenlive:id"}).text = "4"
    playlist_b = etree.Element("playlist", {"id": "playlist_gui_guest_1"})
    master_index = native_root.index(master)
    native_root.insert(master_index, playlist_a)
    native_root.insert(master_index + 1, playlist_b)
    guest_tractor = etree.Element("tractor", {"id": "tractor_gui_guest"})
    etree.SubElement(guest_tractor, "property", {"name": "kdenlive:track_name"}).text = track_name
    etree.SubElement(guest_tractor, "property", {"name": "kdenlive:locked_track"}).text = "1"
    etree.SubElement(guest_tractor, "track", {"producer": "playlist_gui_new_0", "hide": "audio"})
    etree.SubElement(guest_tractor, "track", {"producer": "playlist_gui_guest_1", "hide": "audio"})
    native_root.insert(master_index + 2, guest_tractor)
    etree.SubElement(master, "track", {"producer": "tractor_gui_guest"})
    preview_tree.write(str(preview_path), xml_declaration=True, encoding="utf-8", pretty_print=True)

    adapter.sync_from_gui()

    canonical = etree.parse(str(adapter.source_path)).getroot()
    guest = canonical.xpath(f"./timeline/track[@id='{expected_track_id}']")[0]
    assert guest.get("name") == track_name
    assert guest.get("locked") == "1"
    if clip_id is not None and expected_clipref_id is not None:
        clip = guest.xpath(f"./clipref[@id='{expected_clipref_id}']")[0]
        assert clip.get("clip_id") == clip_id
        assert clip.get("start") == str(start)
        assert clip.get("duration") == str(duration)


def test_sync_from_gui_rekeys_a_new_default_track_after_later_gui_rename(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))
    preview_path = adapter._ensure_preview_project()
    preview_tree = etree.parse(str(preview_path))
    native_root = preview_tree.getroot()
    master = native_root.xpath("./tractor[track[1][@producer='black_track']]")[0]
    master_index = native_root.index(master)
    native_root.insert(master_index, etree.Element("playlist", {"id": "playlist_gui_v3_0"}))
    native_root.insert(master_index + 1, etree.Element("playlist", {"id": "playlist_gui_v3_1"}))
    tractor = etree.Element("tractor", {"id": "tractor_gui_v3"})
    track_name = etree.SubElement(tractor, "property", {"name": "kdenlive:track_name"})
    track_name.text = "V3"
    etree.SubElement(tractor, "track", {"producer": "playlist_gui_v3_0", "hide": "audio"})
    etree.SubElement(tractor, "track", {"producer": "playlist_gui_v3_1", "hide": "audio"})
    native_root.insert(master_index + 2, tractor)
    etree.SubElement(master, "track", {"producer": "tractor_gui_v3"})
    preview_tree.write(str(preview_path), xml_declaration=True, encoding="utf-8", pretty_print=True)

    adapter.sync_from_gui()
    first_sync = etree.parse(str(adapter.source_path)).getroot()
    assert first_sync.xpath("./timeline/track[@id='video_v3']")

    track_name.text = "Guest Cue"
    preview_tree.write(str(preview_path), xml_declaration=True, encoding="utf-8", pretty_print=True)
    adapter.sync_from_gui()

    second_sync = etree.parse(str(adapter.source_path)).getroot()
    assert not second_sync.xpath("./timeline/track[@id='video_v3']")
    guest = second_sync.xpath("./timeline/track[@id='video_guest_cue']")[0]
    assert guest.get("name") == "Guest Cue"

def test_schema_describes_modify_file_operations():
    schema_path = Path(__file__).resolve().parent.parent / "src" / "asil" / "action_schemas" / "kdenlive.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["software"] == "Kdenlive"
    assert schema["supported_action_types"] == ["modify_file"]
    assert "stable visible surfaces" in schema["description"].lower()
    examples = schema["actions"][0]["examples"]
    assert any(
        example["action"]["params"]["operations"][0].get("action") == "add_element"
        and example["action"]["params"]["operations"][0].get("tag") == "clipref"
        for example in examples
    )
    assert any(example["action"]["params"]["operations"][0].get("action") == "delete" for example in examples)
    assert any("xpath" in example["action"]["params"]["operations"][0] for example in examples)
    assert not any("marker" in json.dumps(example).lower() for example in examples)
    assert not any("/kdenliveProject/bin" in json.dumps(example) for example in examples)


def test_kdenlive_tasks_keep_only_stable_visible_surfaces():
    allowed_visibility = {
        "visible_in_track_header",
        "visible_in_track_stack",
        "visible_in_project_profile_summary",
        "visible_in_timeline_clip_block",
    }
    allowed_success_surfaces = {"timeline", "project_header"}

    reports = [audit_task_file(path) for path in _all_kdenlive_task_paths()]
    assert all(report.ok for report in reports), [report.to_dict() for report in reports if not report.ok]

    for task in _all_kdenlive_tasks():
        gui_expectations = task["gui_expectations"]
        assert gui_expectations["success_surface"] in allowed_success_surfaces
        assert set(gui_expectations["checkpoint_visibility"].values()) <= allowed_visibility
        assert "audio_timeline" not in json.dumps(gui_expectations)
        assert "track_controls" not in json.dumps(gui_expectations)


def test_kdenlive_tasks_remove_move_trim_and_hidden_bin_clip_families():
    for task in _all_kdenlive_tasks():
        for action in task["_asil"]["actions"]:
            for operation in action["params"]["operations"]:
                op_action = operation.get("action", "set_attribute")
                xpath = operation.get("xpath", "")
                parent_xpath = operation.get("parent_xpath", "")

                assert not (
                    op_action == "set_attribute"
                    and "/clipref" in xpath
                    and operation.get("attribute") in {"start", "duration", "in", "out"}
                ), task["id"]
                assert parent_xpath != "/kdenliveProject/bin", task["id"]
                assert operation.get("tag") != "marker", task["id"]


def test_kdenlive_timeline_block_tasks_use_existing_visible_video_or_title_clips():
    timeline_task_ids = [
        "kdenlive_03",
        "kdenlive_04",
        "kdenlive_05",
        "kdenlive_09",
        "kdenlive_10",
        "kdenlive_11",
        "kdenlive_13",
        "kdenlive_14",
        "kdenlive_15",
        "kdenlive_18",
        "kdenlive_20",
    ]
    allowed_clip_ids = {"clip_intro", "clip_broll", "clip_title"}

    for task_id in timeline_task_ids:
        task = _kdenlive_task(task_id)
        clipref_ops = [
            operation
            for action in task["_asil"]["actions"]
            for operation in action["params"]["operations"]
            if operation.get("tag") == "clipref"
        ]
        delete_ops = [
            operation
            for action in task["_asil"]["actions"]
            for operation in action["params"]["operations"]
            if operation.get("action") == "delete"
        ]

        if clipref_ops:
            assert all(op["attributes"]["clip_id"] in allowed_clip_ids for op in clipref_ops), task_id
            assert all(op["attributes"]["duration"] in {"48", "60", "72", "90"} for op in clipref_ops), task_id
        else:
            assert delete_ops, task_id
            assert all("video" in op["xpath"] for op in delete_ops), task_id


def test_kdenlive_tasks_exist_and_pass_audit():
    task_root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "kdenlive"

    task_files = sorted(path for path in task_root.glob("kdenlive_*.json") if path.stem.removeprefix("kdenlive_").isdigit())
    task_ids = [path.stem for path in task_files]
    reports = [audit_task_file(path) for path in task_files]

    assert task_ids == [f"kdenlive_{index:02d}" for index in range(1, 21)]
    assert all(report.ok for report in reports), [report.to_dict() for report in reports if not report.ok]


def test_first_task_runs_deterministically_with_placeholder_resolution(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))
    task_path = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "kdenlive" / "kdenlive_01.json"
    task = TaskDefinition.from_json(task_path)[0]

    result = run_task(adapter, task)
    obs = adapter.observe()
    track = next(e for e in obs.interactive_elements if e.id == "track:video_main")

    assert result.success is True
    assert result.score == 1.0
    assert track.value["name"] == "Primary Video"


def test_validate_action_accepts_modify_file_only(tmp_path: Path):
    adapter = KdenliveAdapter(_write_project(tmp_path / "project.kdenlive"))

    assert adapter.validate_action(Action(action_type="modify_file", target="x", params={}))
    assert not adapter.validate_action(Action(action_type="api_call", target="x", params={}))
