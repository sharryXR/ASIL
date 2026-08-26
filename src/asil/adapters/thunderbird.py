"""ASIL adapter for Thunderbird — deterministic mailbox state with honest rendering."""

from __future__ import annotations

import json
import mailbox
import re
import shutil
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.protocol import Action, Element, Observation
from asil.rendering import (
    RenderArtifact,
    activate_window,
    capture_window_to_png,
    click_window_relative,
    ensure_user_access,
    launch_gui_process,
    read_clipboard_text,
    send_keys_to_window,
    wait_for_window,
    terminate_process,
    type_text_to_window,
)


def _message(
    message_id: str,
    folder: str,
    subject: str,
    sender: str,
    recipient: str,
    body: str,
    *,
    read: bool,
    starred: bool = False,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": message_id,
        "folder": folder,
        "subject": subject,
        "from": sender,
        "to": recipient,
        "body": body,
        "read": read,
        "starred": starred,
        "tags": list(tags or []),
    }


def _default_state() -> dict[str, Any]:
    return {
        "folders": ["Inbox", "Projects", "Finance", "Archive", "Sent", "Trash", "Drafts"],
        "selected_folder": "Inbox",
        "selected_message_id": "msg_client_followup",
        "draft": None,
        "messages": [
            _message(
                "msg_client_followup",
                "Inbox",
                "Client follow-up",
                "aria@northwind.test",
                "morgan@acme.test",
                "Can you send the revised rollout timeline by Wednesday afternoon?",
                read=False,
            ),
            _message(
                "msg_budget_review",
                "Inbox",
                "Budget review notes",
                "cfo@acme.test",
                "morgan@acme.test",
                "Please confirm whether Thursday's budget review still works for you.",
                read=False,
            ),
            _message(
                "msg_invoice_april",
                "Inbox",
                "April invoice",
                "billing@contoso.test",
                "finance@acme.test",
                "Attached is the April invoice for the retained support work.",
                read=False,
            ),
            _message(
                "msg_status_digest",
                "Inbox",
                "Status digest draft",
                "ops@acme.test",
                "morgan@acme.test",
                "Please review the status digest draft before the Friday sync.",
                read=True,
                tags=["Follow Up"],
            ),
            _message(
                "msg_launch_copy",
                "Inbox",
                "Launch copy review",
                "marketing@acme.test",
                "morgan@acme.test",
                "The launch copy is ready for a final polish before approval.",
                read=False,
                starred=True,
                tags=["Launch"],
            ),
            _message(
                "msg_vendor_statement",
                "Inbox",
                "Vendor statement received",
                "accounts@vendor.test",
                "finance@acme.test",
                "The monthly vendor statement is attached for reconciliation.",
                read=True,
            ),
            _message(
                "msg_roadmap_notes",
                "Inbox",
                "Roadmap notes",
                "product@acme.test",
                "morgan@acme.test",
                "Updated roadmap notes are ready for next week's planning review.",
                read=False,
            ),
            _message(
                "msg_payroll_checklist",
                "Inbox",
                "Payroll checklist",
                "hr@acme.test",
                "finance@acme.test",
                "Please verify the payroll checklist before close of business.",
                read=False,
                tags=["Finance"],
            ),
            _message(
                "msg_design_mockups",
                "Projects",
                "Design mockups ready",
                "design@acme.test",
                "morgan@acme.test",
                "The new landing page mockups are ready for review in the project folder.",
                read=False,
                starred=True,
                tags=["Design"],
            ),
            _message(
                "msg_contract_redline",
                "Projects",
                "Contract redline",
                "legal@partner.test",
                "morgan@acme.test",
                "Please review the latest contract redline before Friday.",
                read=True,
            ),
            _message(
                "msg_refund_request",
                "Finance",
                "Refund request",
                "support@shop.test",
                "finance@acme.test",
                "A customer refund request is waiting for approval in the finance queue.",
                read=False,
                tags=["Finance"],
            ),
            _message(
                "msg_ops_digest",
                "Archive",
                "Operations digest",
                "ops@acme.test",
                "morgan@acme.test",
                "Here is last week's operations digest for reference.",
                read=True,
            ),
            _message(
                "msg_old_flyer",
                "Trash",
                "Old campaign flyer",
                "marketing@acme.test",
                "morgan@acme.test",
                "Removing the outdated campaign flyer from active folders.",
                read=True,
            ),
        ],
    }


def _compose_focus_state() -> dict[str, Any]:
    state = _default_state()
    state["draft"] = {
        "to": "team@acme.test",
        "subject": "Status update",
        "body": "Draft the weekly status update for the team.",
        "source_id": None,
        "is_open": True,
    }
    state["selected_message_id"] = None
    return state


class ThunderbirdAdapter(ASILAdapter):
    app_name = "Thunderbird"
    supported_action_types = ["invoke_function"]
    _COMPOSE_TO_COORDS = (250, 120)
    _COMPOSE_SUBJECT_COORDS = (250, 160)
    _COMPOSE_BODY_COORDS = (240, 245)

    def __init__(self, mailbox_path: str | Path) -> None:
        self.mailbox_path = Path(mailbox_path)
        if not self.mailbox_path.exists():
            self.setup_state("default")

    @classmethod
    def from_evaluation_context(
        cls,
        tmp: str | Path,
        sandbox=None,
        mock: bool = False,
    ) -> "ThunderbirdAdapter":
        del sandbox, mock
        return cls(Path(tmp) / "thunderbird_mailbox.json")

    @property
    def source_path(self) -> Path:
        return self.mailbox_path

    def clone(self, new_path: Path) -> "ThunderbirdAdapter":
        shutil.copy2(self.mailbox_path, new_path)
        return ThunderbirdAdapter(new_path)

    def get_context(self) -> dict[str, str]:
        return {"mailbox_path": str(self.mailbox_path)}

    def reset_state(self) -> None:
        self.setup_state("default")

    def setup_state(self, initial_state: str) -> None:
        state_name = (initial_state or "default").strip()
        if state_name == "compose_focus":
            self._write_state(_compose_focus_state())
            return
        if state_name == "projects_focus":
            state = _default_state()
            state["selected_folder"] = "Projects"
            state["selected_message_id"] = "msg_design_mockups"
            self._write_state(state)
            return
        if state_name == "finance_focus":
            state = _default_state()
            state["selected_folder"] = "Finance"
            state["selected_message_id"] = "msg_refund_request"
            self._write_state(state)
            return
        if state_name == "trash_focus":
            state = _default_state()
            state["selected_folder"] = "Trash"
            state["selected_message_id"] = "msg_old_flyer"
            self._write_state(state)
            return
        self._write_state(_default_state())

    def validate_action(self, action: Action) -> bool:
        return (
            action.action_type in self.supported_action_types
            and action.target == "thunderbird"
            and isinstance(action.params.get("operations"), list)
        )

    def observe(self) -> Observation:
        state = self._normalized_mailbox_state(self._read_state())
        selected_folder = state["selected_folder"]
        selected_message = self._selected_message(state)
        visible_messages = [message for message in state["messages"] if message["folder"] == selected_folder]
        elements: list[Element] = []

        for folder in state["folders"]:
            folder_messages = [message for message in state["messages"] if message["folder"] == folder]
            elements.append(
                Element(
                    id=f"folder:{folder}",
                    type="folder",
                    label=folder,
                    value={
                        "name": folder,
                        "selected": folder == selected_folder,
                        "message_count": len(folder_messages),
                        "unread_count": sum(1 for message in folder_messages if not message["read"]),
                    },
                    editable=False,
                    actions=["switch_folder"],
                )
            )

        for message in visible_messages:
            elements.append(
                Element(
                    id=f"message:{message['id']}",
                    type="message",
                    label=message["subject"],
                    value={
                        "folder": message["folder"],
                        "from": message["from"],
                        "subject": message["subject"],
                        "read": message["read"],
                        "starred": message["starred"],
                        "tags": list(message["tags"]),
                        "selected": message["id"] == state.get("selected_message_id"),
                        "preview": self._preview(message["body"]),
                        "body": message["body"],
                    },
                    editable=True,
                    actions=[
                        "select_message",
                        "set_read",
                        "set_starred",
                        "add_tag",
                        "remove_tag",
                        "move_message",
                        "archive_message",
                        "delete_message",
                    ],
                )
            )

        if selected_message and selected_message["folder"] == selected_folder:
            elements.append(
                Element(
                    id=f"message_view:{selected_message['id']}",
                    type="message_view",
                    label=selected_message["subject"],
                    value={
                        "folder": selected_message["folder"],
                        "subject": selected_message["subject"],
                        "from": selected_message["from"],
                        "to": selected_message["to"],
                        "body": selected_message["body"],
                        "read": selected_message["read"],
                        "starred": selected_message["starred"],
                        "tags": list(selected_message["tags"]),
                    },
                    editable=False,
                    actions=["create_draft_reply"],
                )
            )

        if state.get("draft"):
            draft = dict(state["draft"])
            draft["is_open"] = True
            elements.append(
                Element(
                    id="compose:draft",
                    type="compose",
                    label=draft.get("subject", "Draft"),
                    value=draft,
                    editable=True,
                    actions=["update_draft", "send_draft"],
                )
            )

        current_view = "compose" if state.get("draft") else "mail_3pane"
        active_document = state["draft"]["subject"] if state.get("draft") else selected_folder
        breadcrumb = ["Compose"] if state.get("draft") else [selected_folder]

        return self._build_observation(
            source="json_mailbox",
            elements=elements,
            app_state={
                "current_view": current_view,
                "active_document": active_document,
                "document_path": str(self.mailbox_path),
            },
            navigation={
                "current_path": active_document,
                "breadcrumb": breadcrumb,
            },
            data_summary=(
                f"Thunderbird mailbox with {len(state['messages'])} messages; "
                f"selected folder is {selected_folder}"
            ),
        )

    def execute(self, action: Action) -> Observation:
        if not self.validate_action(action):
            raise ValueError(f"Unsupported Thunderbird action: {action}")

        state = self._read_state()
        for operation in action.params.get("operations", []):
            op_name = operation.get("action")
            if op_name == "switch_folder":
                self._switch_folder(state, str(operation["folder"]))
            elif op_name == "select_message":
                self._select_message(state, str(operation["id"]))
            elif op_name == "set_starred":
                self._find_message(state, str(operation["id"]))["starred"] = bool(operation.get("starred", True))
            elif op_name == "set_read":
                self._find_message(state, str(operation["id"]))["read"] = bool(operation.get("read", True))
            elif op_name == "move_message":
                message = self._find_message(state, str(operation["id"]))
                destination = str(operation["destination"])
                self._ensure_folder(state, destination)
                message["folder"] = destination
                state["selected_folder"] = destination
                state["selected_message_id"] = message["id"]
            elif op_name == "archive_message":
                message = self._find_message(state, str(operation["id"]))
                self._ensure_folder(state, "Archive")
                message["folder"] = "Archive"
                message["read"] = True
                state["selected_folder"] = "Archive"
                state["selected_message_id"] = message["id"]
            elif op_name == "delete_message":
                message = self._find_message(state, str(operation["id"]))
                self._ensure_folder(state, "Trash")
                message["folder"] = "Trash"
                state["selected_folder"] = "Trash"
                state["selected_message_id"] = message["id"]
            elif op_name == "restore_message":
                message = self._find_message(state, str(operation["id"]))
                destination = str(operation.get("destination", "Inbox"))
                self._ensure_folder(state, destination)
                message["folder"] = destination
                state["selected_folder"] = destination
                state["selected_message_id"] = message["id"]
            elif op_name == "add_tag":
                message = self._find_message(state, str(operation["id"]))
                tag = str(operation["tag"])
                if tag not in message["tags"]:
                    message["tags"].append(tag)
            elif op_name == "remove_tag":
                message = self._find_message(state, str(operation["id"]))
                tag = str(operation["tag"])
                message["tags"] = [existing for existing in message["tags"] if existing != tag]
            elif op_name == "create_draft_reply":
                source = self._find_message(state, str(operation.get("source_id") or state.get("selected_message_id")))
                state["draft"] = {
                    "to": source["from"],
                    "subject": source["subject"] if source["subject"].startswith("Re: ") else f"Re: {source['subject']}",
                    "body": operation.get(
                        "body",
                        f"Hi,\n\n{source['body']}\n",
                    ),
                    "source_id": source["id"],
                    "is_open": True,
                }
                state["selected_folder"] = source["folder"]
                state["selected_message_id"] = source["id"]
            elif op_name == "create_draft":
                state["draft"] = {
                    "to": str(operation.get("to", "team@acme.test")),
                    "subject": str(operation.get("subject", "")),
                    "body": str(operation.get("body", "")),
                    "source_id": None,
                    "is_open": True,
                }
                state["selected_message_id"] = None
            elif op_name == "update_draft":
                draft = self._require_draft(state)
                changes = operation.get("changes", {})
                if not isinstance(changes, dict):
                    raise ValueError("`update_draft` requires a mapping of changes.")
                draft.update(changes)
                draft["is_open"] = True
            elif op_name == "send_draft":
                draft = self._require_draft(state)
                sent_id = self._next_sent_id(state)
                self._ensure_folder(state, "Sent")
                state["messages"].append(
                    _message(
                        sent_id,
                        "Sent",
                        str(draft["subject"]),
                        "morgan@acme.test",
                        str(draft["to"]),
                        str(draft["body"]),
                        read=True,
                    )
                )
                state["selected_folder"] = "Sent"
                state["selected_message_id"] = sent_id
                state["draft"] = None
            else:
                raise ValueError(f"Unsupported Thunderbird operation: {op_name}")

        self._write_state(state)
        return self.observe()

    def sync_from_gui(self, session=None) -> None:
        del session
        state = self._read_state()
        synced = self._read_saved_draft_state()
        draft = state.get("draft")
        if draft is None:
            draft = {
                "to": "",
                "subject": "",
                "body": "",
                "is_open": True,
            }
            state["draft"] = draft
        for key in ("to", "subject", "body"):
            synced_value = str(synced.get(key, ""))
            if synced_value.strip():
                draft[key] = synced_value
        draft["is_open"] = bool(synced["is_open"])
        self._write_state(state)

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="app_window",
            backend="x11-window-capture",
            actual_page=True,
            description="Screenshot of the real Thunderbird window showing the main 3-pane mailbox or compose view",
        )

    def get_gui_session_spec(self) -> GUISessionSpec:
        thunderbird_bin = shutil.which("thunderbird")
        if thunderbird_bin is None:
            raise RuntimeError("thunderbird is not installed.")
        profile_dir = self._ensure_profile()
        profile_home = profile_dir.parent
        profile_home.mkdir(parents=True, exist_ok=True)
        ensure_user_access(profile_home, run_as_user=None)
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=(thunderbird_bin, "-profile", str(profile_dir), "--new-instance"),
            window_title_pattern=r"Write:.*|.*\(no subject\).*Thunderbird",
            window_class_pattern=r"Thunderbird|thunderbird",
            startup_timeout_s=60.0,
            post_launch_delay_s=4.0,
            post_launch_callback=self._prime_compose_window,
            min_width=500,
            min_height=400,
            extra_env={
                "HOME": str(profile_home),
                "XDG_CONFIG_HOME": str(profile_home / ".config"),
                "XDG_CACHE_HOME": str(profile_home / ".cache"),
                "XDG_DATA_HOME": str(profile_home / ".local" / "share"),
            },
        )

    def _prime_compose_window(self) -> None:
        state = self._read_state()
        draft = dict(state.get("draft") or {})
        main_title_pattern = r".*Thunderbird|Local Folders - Mozilla Thunderbird"
        compose_title_pattern = r"Write:.*|.*\(no subject\).*Thunderbird"

        wait_for_window(
            main_title_pattern,
            timeout=45.0,
            min_width=700,
            min_height=500,
            window_class_pattern=r"Thunderbird|thunderbird",
        )
        send_keys_to_window(
            main_title_pattern,
            ["ctrl+n"],
            timeout=30.0,
            min_width=700,
            min_height=500,
            window_class_pattern=r"Thunderbird|thunderbird",
        )
        time.sleep(3.0)
        wait_for_window(
            compose_title_pattern,
            timeout=30.0,
            min_width=500,
            min_height=400,
            window_class_pattern=r"Thunderbird|thunderbird",
        )
        self._populate_compose_fields(compose_title_pattern, draft, timeout=20.0)

    def render_to_png(self, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        thunderbird_bin = shutil.which("thunderbird")
        if thunderbird_bin is None:
            raise RuntimeError("thunderbird is not installed.")

        profile_dir = self._ensure_profile()
        ensure_user_access(profile_dir, run_as_user="asilgui")
        state = self._read_state()

        def _launch(command: list[str]):
            return launch_gui_process(command)

        proc = None
        try:
            if state.get("draft"):
                draft = state["draft"]
                compose_title_pattern = r"Write:.*|.*\(no subject\).*Thunderbird"
                proc = _launch([thunderbird_bin, "-profile", str(profile_dir), "--new-instance"])
                wait_for_window(
                    ".*Mozilla Thunderbird|.* - Thunderbird",
                    timeout=45.0,
                    min_width=700,
                    min_height=500,
                )
                time.sleep(2.0)
                send_keys_to_window(
                    ".*Mozilla Thunderbird|.* - Thunderbird",
                    ["ctrl+n"],
                    timeout=30.0,
                    min_width=600,
                    min_height=500,
                )
                time.sleep(3.0)
                self._populate_compose_fields(compose_title_pattern, draft, timeout=30.0)
                capture_window_to_png(
                    out,
                    title_pattern=compose_title_pattern,
                    timeout=60.0,
                    margin=12,
                    settle_delay=6.0,
                    min_width=500,
                    min_height=400,
                )
                return out

            proc = _launch([thunderbird_bin, "-profile", str(profile_dir), "--new-instance"])
            self._focus_mail_3pane()
            capture_window_to_png(
                out,
                title_pattern=".*Mozilla Thunderbird|.* - Thunderbird",
                timeout=60.0,
                margin=12,
                settle_delay=6.0,
                min_width=900,
                min_height=600,
            )
            return out
        finally:
            if proc is not None:
                terminate_process(proc)
        return out

    def _focus_mail_3pane(self) -> None:
        title_pattern = ".*Mozilla Thunderbird|.* - Thunderbird"
        try:
            click_window_relative(
                title_pattern,
                x_offset=64,
                y_offset=92,
                timeout=30.0,
                min_width=900,
                min_height=600,
            )
            time.sleep(0.5)
            send_keys_to_window(
                title_pattern,
                ["Right"],
                timeout=30.0,
                min_width=900,
                min_height=600,
            )
            time.sleep(0.5)
            click_window_relative(
                title_pattern,
                x_offset=86,
                y_offset=121,
                timeout=30.0,
                min_width=900,
                min_height=600,
            )
            time.sleep(1.0)
        except Exception:
            pass

    def _compose_spec(self) -> str:
        state = self._read_state()
        if state.get("draft"):
            draft = state["draft"]
            return ",".join(
                [
                    f"to={draft.get('to', '')}",
                    f"subject={draft.get('subject', '')}",
                    f"body={draft.get('body', '')}",
                ]
            )

        selected = self._selected_message(state)
        if selected is not None:
            return ",".join(
                [
                    f"to={selected.get('to', '')}",
                    f"subject={selected.get('subject', '')}",
                    f"body={selected.get('body', '')}",
                ]
            )

        return "to=team@acme.test,subject=Thunderbird Preview,body=Open Thunderbird preview"

    def _read_compose_window_state(self) -> dict[str, Any]:
        compose_pattern = r"Write:.*|.*\(no subject\).*Thunderbird"
        self._copy_compose_field(
            compose_pattern,
            x_offset=self._COMPOSE_TO_COORDS[0],
            y_offset=self._COMPOSE_TO_COORDS[1],
        )
        to_value = read_clipboard_text().strip()

        self._copy_compose_field(
            compose_pattern,
            x_offset=self._COMPOSE_SUBJECT_COORDS[0],
            y_offset=self._COMPOSE_SUBJECT_COORDS[1],
        )
        subject_value = read_clipboard_text().strip()

        self._copy_compose_field(
            compose_pattern,
            x_offset=self._COMPOSE_BODY_COORDS[0],
            y_offset=self._COMPOSE_BODY_COORDS[1],
        )
        body_value = self._normalize_compose_body(read_clipboard_text())

        return {
            "to": to_value,
            "subject": subject_value,
            "body": body_value,
            "is_open": True,
        }

    @staticmethod
    def _normalize_compose_body(text: str) -> str:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        normalized_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                normalized_lines.append(stripped)
        return "\n".join(normalized_lines).strip()

    def _populate_compose_fields(
        self,
        compose_title_pattern: str,
        draft: dict[str, Any],
        *,
        timeout: float,
    ) -> None:
        for key, x_offset, y_offset in (
            ("to", *self._COMPOSE_TO_COORDS),
            ("subject", *self._COMPOSE_SUBJECT_COORDS),
            ("body", *self._COMPOSE_BODY_COORDS),
        ):
            value = str(draft.get(key, ""))
            if not value:
                continue
            click_window_relative(
                compose_title_pattern,
                x_offset=x_offset,
                y_offset=y_offset,
                timeout=timeout,
                min_width=500,
                min_height=400,
                window_class_pattern=r"Thunderbird|thunderbird",
            )
            type_text_to_window(
                compose_title_pattern,
                value,
                timeout=timeout,
                min_width=500,
                min_height=400,
                window_class_pattern=r"Thunderbird|thunderbird",
            )
            if key == "to":
                send_keys_to_window(
                    compose_title_pattern,
                    ["Return"],
                    timeout=timeout,
                    min_width=500,
                    min_height=400,
                    window_class_pattern=r"Thunderbird|thunderbird",
                )

    def _read_saved_draft_state(self) -> dict[str, Any]:
        compose_pattern = r"Write:.*|.*\(no subject\).*Thunderbird"
        try:
            send_keys_to_window(
                compose_pattern,
                ["ctrl+s"],
                timeout=20.0,
                min_width=500,
                min_height=400,
                window_class_pattern=r"Thunderbird|thunderbird",
            )
            time.sleep(1.0)
            draft_message = self._latest_saved_draft_message()
            if draft_message is not None:
                return {
                    "to": draft_message.get("to", ""),
                    "subject": draft_message.get("subject", ""),
                    "body": self._normalize_compose_body(str(draft_message.get("body", ""))),
                    "is_open": True,
                }
        except Exception:
            pass
        return self._read_compose_window_state()

    def _latest_saved_draft_message(self) -> dict[str, str] | None:
        drafts_path = self._ensure_profile() / "Mail" / "Local Folders" / "Drafts"
        if not drafts_path.exists() or drafts_path.stat().st_size == 0:
            return None

        mbox = mailbox.mbox(drafts_path)
        try:
            messages = list(mbox)
        finally:
            mbox.close()
        if not messages:
            return None

        message = messages[-1]
        return {
            "to": self._normalize_header(message.get("To")),
            "subject": self._normalize_header(message.get("Subject")),
            "body": self._message_body_text(message),
        }

    @staticmethod
    def _normalize_header(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split())

    @classmethod
    def _message_body_text(cls, message: Any) -> str:
        if message.is_multipart():
            preferred = message.get_body(preferencelist=("plain", "html"))
            if preferred is not None:
                payload = preferred.get_payload(decode=True) or b""
                charset = preferred.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
                if preferred.get_content_type() == "text/html":
                    return cls._html_to_text(text)
                return text.strip()
        payload = message.get_payload(decode=True)
        if payload is None:
            raw = message.get_payload()
            text = raw if isinstance(raw, str) else ""
        else:
            charset = message.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
        if message.get_content_type() == "text/html":
            return cls._html_to_text(text)
        return text.strip()

    @staticmethod
    def _html_to_text(html: str) -> str:
        class _HTMLTextExtractor(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.parts: list[str] = []

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag.lower() in {"br", "p", "div", "li"}:
                    self.parts.append("\n")

            def handle_endtag(self, tag: str) -> None:
                if tag.lower() in {"p", "div", "li"}:
                    self.parts.append("\n")

            def handle_data(self, data: str) -> None:
                self.parts.append(data)

        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        text = "".join(extractor.parts).replace("\r\n", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _copy_compose_field(
        self,
        title_pattern: str,
        *,
        x_offset: int,
        y_offset: int,
    ) -> None:
        click_window_relative(
            title_pattern,
            x_offset=x_offset,
            y_offset=y_offset,
            timeout=20.0,
            min_width=500,
            min_height=400,
            window_class_pattern=r"Thunderbird",
        )
        send_keys_to_window(
            title_pattern,
            ["ctrl+a", "ctrl+c"],
            timeout=20.0,
            min_width=500,
            min_height=400,
            window_class_pattern=r"Thunderbird",
        )
        time.sleep(0.2)

    def _ensure_profile(self) -> Path:
        profile_dir = self.mailbox_path.parent / "_thunderbird_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        local_folders = profile_dir / "Mail" / "Local Folders"
        local_folders.mkdir(parents=True, exist_ok=True)
        profiles_ini = profile_dir.parent / "profiles.ini"
        profiles_ini.write_text(
            "\n".join(
                [
                    "[General]",
                    "StartWithLastProfile=1",
                    "Version=2",
                    "",
                    "[Profile0]",
                    "Name=default",
                    "IsRelative=1",
                    f"Path={profile_dir.name}",
                    "Default=1",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        state = self._read_state()
        folder_messages: dict[str, list[dict[str, Any]]] = {}
        for message in state["messages"]:
            folder_messages.setdefault(message["folder"], []).append(message)
        for folder, messages in folder_messages.items():
            if folder in {"Inbox", "Sent", "Trash", "Drafts"}:
                filename = folder
            else:
                filename = f"{folder}.sbd/{folder}"
            mailbox_file = local_folders / filename
            mailbox_file.parent.mkdir(parents=True, exist_ok=True)
            mailbox_file.write_text(
                "\n".join(
                    f"From - Thu Jan 01 00:00:00 1970\nSubject: {msg['subject']}\nFrom: {msg['from']}\nTo: {msg['to']}\n\n{msg['body']}\n"
                    for msg in messages
                ),
                encoding="utf-8",
            )

        prefs = profile_dir / "prefs.js"
        prefs.write_text(
            '\n'.join(
                [
                    'user_pref("mail.accountmanager.accounts", "account1");',
                    'user_pref("mail.accountmanager.defaultaccount", "account1");',
                    'user_pref("mail.account.account1.server", "server1");',
                    'user_pref("mail.server.server1.directory-rel", "[ProfD]Mail/Local Folders");',
                    'user_pref("mail.server.server1.hostname", "Local Folders");',
                    'user_pref("mail.server.server1.name", "Local Folders");',
                    'user_pref("mail.server.server1.type", "none");',
                    'user_pref("mail.server.server1.userName", "local");',
                    'user_pref("mail.identity.id1.fullName", "Morgan Acme");',
                    'user_pref("mail.identity.id1.useremail", "morgan@acme.test");',
                    'user_pref("mail.identity.id1.valid", true);',
                    'user_pref("mail.account.account1.identities", "id1");',
                    'user_pref("mail.root.none-rel", "[ProfD]Mail");',
                    'user_pref("mail.shell.checkDefaultClient", false);',
                    'user_pref("mail.provider.suppress_dialog_on_startup", true);',
                    'user_pref("mailnews.start_page.enabled", false);',
                    'user_pref("mailnews.start_page.url", "about:blank");',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return profile_dir

    def _write_message_file(self, profile_dir: Path, message: dict[str, Any]) -> Path:
        message_dir = profile_dir / "Messages"
        message_dir.mkdir(parents=True, exist_ok=True)
        path = message_dir / f"{message['id']}.eml"
        path.write_text(
            "\n".join(
                [
                    f"Subject: {message['subject']}",
                    f"From: {message['from']}",
                    f"To: {message['to']}",
                    "",
                    str(message["body"]),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def _render_html(self) -> str:
        state = self._read_state()
        selected_folder = state["selected_folder"]
        visible_messages = [message for message in state["messages"] if message["folder"] == selected_folder]
        selected = self._selected_message(state)

        folder_items = "".join(
            (
                "<div style='display:flex;justify-content:space-between;padding:10px 12px;"
                f"border-radius:10px;background:{'#e7f0ff' if folder == selected_folder else 'transparent'};'>"
                f"<span>{folder}</span><span>{sum(1 for m in state['messages'] if m['folder'] == folder and not m['read'])}</span>"
                "</div>"
            )
            for folder in state["folders"]
        )

        message_rows = "".join(
            (
                "<tr>"
                f"<td>{'★' if message['starred'] else ''}</td>"
                f"<td>{message['subject']}</td>"
                f"<td>{message['from']}</td>"
                f"<td>{'Unread' if not message['read'] else 'Read'}</td>"
                f"<td>{', '.join(message['tags'])}</td>"
                "</tr>"
            )
            for message in visible_messages
        )

        preview = ""
        if state.get("draft"):
            draft = state["draft"]
            preview = (
                "<div class='panel' style='padding:18px;'>"
                "<h2 style='margin-top:0;'>Compose</h2>"
                f"<p><strong>To:</strong> {draft['to']}</p>"
                f"<p><strong>Subject:</strong> {draft['subject']}</p>"
                f"<pre style='white-space:pre-wrap;'>{draft['body']}</pre>"
                "</div>"
            )
        elif selected and selected["folder"] == selected_folder:
            preview = (
                "<div class='panel' style='padding:18px;'>"
                f"<h2 style='margin-top:0;'>{selected['subject']}</h2>"
                f"<p><strong>From:</strong> {selected['from']}</p>"
                f"<p><strong>To:</strong> {selected['to']}</p>"
                f"<p><strong>Tags:</strong> {', '.join(selected['tags']) or 'None'}</p>"
                f"<pre style='white-space:pre-wrap;'>{selected['body']}</pre>"
                "</div>"
            )

        body = (
            "<div style='display:grid;grid-template-columns:260px 1fr 1fr;gap:20px;'>"
            "<div class='panel' style='padding:16px;'><h2 style='margin-top:0;'>Folders</h2>"
            f"{folder_items}</div>"
            "<div class='panel' style='padding:16px;'><h2 style='margin-top:0;'>Message List</h2>"
            "<table><thead><tr><th></th><th>Subject</th><th>From</th><th>Status</th><th>Tags</th></tr></thead><tbody>"
            f"{message_rows}</tbody></table></div>"
            f"{preview}</div>"
        )
        return html_page("Thunderbird Mailbox", body)

    def _read_state(self) -> dict[str, Any]:
        return json.loads(self.mailbox_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict[str, Any]) -> None:
        self.mailbox_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
        self.mailbox_path.write_text(payload, encoding="utf-8")

    @staticmethod
    def _preview(body: str) -> str:
        single_line = " ".join(body.split())
        return single_line if len(single_line) <= 80 else single_line[:77] + "..."

    @staticmethod
    def _ensure_folder(state: dict[str, Any], folder: str) -> None:
        if folder not in state["folders"]:
            state["folders"].append(folder)

    @staticmethod
    def _find_message(state: dict[str, Any], message_id: str) -> dict[str, Any]:
        for message in state["messages"]:
            if message["id"] == message_id:
                return message
        raise KeyError(message_id)

    @staticmethod
    def _selected_message(state: dict[str, Any]) -> dict[str, Any] | None:
        selected_id = state.get("selected_message_id")
        if not selected_id:
            return None
        for message in state["messages"]:
            if message["id"] == selected_id:
                return message
        return None

    @classmethod
    def _normalized_mailbox_state(cls, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("draft"):
            return state

        selected_folder = state.get("selected_folder")
        visible_messages = [message for message in state["messages"] if message["folder"] == selected_folder]
        selected_message = cls._selected_message(state)
        if visible_messages and selected_message and selected_message["folder"] == selected_folder:
            return state

        if visible_messages:
            state["selected_message_id"] = visible_messages[0]["id"]
            return state

        fallback_message = next(iter(state["messages"]), None)
        if fallback_message is not None:
            state["selected_folder"] = fallback_message["folder"]
            state["selected_message_id"] = fallback_message["id"]
        else:
            state["selected_message_id"] = None
        return state

    def _switch_folder(self, state: dict[str, Any], folder: str) -> None:
        self._ensure_folder(state, folder)
        state["selected_folder"] = folder
        first_message = next((message for message in state["messages"] if message["folder"] == folder), None)
        state["selected_message_id"] = None if first_message is None else first_message["id"]

    def _select_message(self, state: dict[str, Any], message_id: str) -> None:
        message = self._find_message(state, message_id)
        state["selected_folder"] = message["folder"]
        state["selected_message_id"] = message["id"]

    @staticmethod
    def _require_draft(state: dict[str, Any]) -> dict[str, Any]:
        draft = state.get("draft")
        if not isinstance(draft, dict):
            raise ValueError("No open draft is available.")
        return draft

    @staticmethod
    def _next_sent_id(state: dict[str, Any]) -> str:
        sent_ids = [
            int(message["id"].split("_")[1])
            for message in state["messages"]
            if message["id"].startswith("sent_") and message["id"].split("_")[1].isdigit()
        ]
        return f"sent_{max(sent_ids, default=0) + 1:03d}"
