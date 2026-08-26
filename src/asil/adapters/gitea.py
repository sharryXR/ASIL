"""ASIL adapter for Gitea — Pattern C (REST API).

observe() returns structured Elements so that the generic _check_single()
validation rules (element_exists, element_value, element_contains) work
without any Gitea-specific code in runner.py.

Element ID conventions:
  repo:{owner}/{name}          — repository
  issue:{number}               — issue in current repo
  pr:{number}                  — pull request in current repo
  milestone:{id}               — milestone in current repo
  label:{id}                   — label in current repo
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.protocol import Action, Element, Observation
from asil.rendering import RenderArtifact, capture_url_to_png, send_keys_to_window, type_text_to_window


class GiteaAdapter(ASILAdapter):
    app_name = "Gitea"
    supported_action_types = ["api_call"]

    def __init__(self, base_url: str, token: str, owner: str = "asil_admin", repo: str = "test-repo") -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.owner = owner
        self.repo = repo
        self._current_ui_path = f"/{self.owner}/{self.repo}"
        self._basic_auth = (
            os.environ.get("GITEA_ADMIN", "asil_admin"),
            os.environ.get("GITEA_PASSWORD", "asil_password"),
        )

    def get_context(self) -> dict[str, str]:
        return {"owner": self.owner, "repo": self.repo, "token": self.token}

    def get_gui_session_spec(self) -> GUISessionSpec:
        return GUISessionSpec(
            surface_type="browser",
            browser_url="about:blank",
            browser_navigation_mode="current_page",
            browser_post_navigation_settle_ms=1_000,
            window_title_pattern=r".*",
            window_class_pattern=r"chromium|Chromium|chrome|Google-chrome",
            min_width=1000,
            min_height=700,
            startup_timeout_s=60.0,
            post_launch_delay_s=5.0,
            post_launch_callback=self._prime_browser_session,
            backend_ready_probe=self._probe_backend_ready,
            ui_ready_probe=self._probe_ui_ready,
        )

    def _probe_backend_ready(self) -> None:
        from asil.gui_agent.session import GUISessionStartupError

        login_url = f"{self.base_url}/user/login"
        try:
            response = requests.get(login_url, timeout=10, proxies=self._NO_PROXY)
        except requests.RequestException as exc:
            raise GUISessionStartupError("backend_unready", f"Gitea backend is not reachable at {login_url}.") from exc
        if response.status_code >= 500:
            raise GUISessionStartupError(
                "backend_unready",
                f"Gitea backend returned HTTP {response.status_code} for {login_url}.",
            )
        target_url = f"{self.base_url}{self._current_ui_path}"
        try:
            target_response = requests.get(target_url, timeout=10, proxies=self._NO_PROXY)
        except requests.RequestException as exc:
            raise GUISessionStartupError("backend_unready", f"Gitea target page is not reachable at {target_url}.") from exc
        target_body = target_response.text.lower()
        if target_response.status_code == 404 or "page not found" in target_body:
            raise GUISessionStartupError(
                "backend_unready",
                f"Gitea target page {self._current_ui_path} is not ready yet.",
            )
        if target_response.status_code >= 500:
            raise GUISessionStartupError(
                "backend_unready",
                f"Gitea target page returned HTTP {target_response.status_code} for {target_url}.",
            )
        target_repo_prefix = f"/{self.owner}/{self.repo}"
        if self._current_ui_path.startswith(target_repo_prefix):
            api_response = self._request("GET", f"/api/v1/repos/{self.owner}/{self.repo}")
            if api_response.status_code == 404:
                raise GUISessionStartupError(
                    "backend_unready",
                    f"Gitea repository {self.owner}/{self.repo} is not ready yet.",
                )
            api_response.raise_for_status()

    def _prime_browser_session(self, session=None) -> None:
        from asil.gui_agent.session import (
            GUISessionStartupError,
            _assert_browser_page_ready,
            navigate_browser_target,
        )

        if session is not None and getattr(session, "browser_page", None) is not None:
            page = session.browser_page
            login_url = self._login_url()
            target_url = f"{self.base_url}{self._current_ui_path}"
            body_ready_timeout_ms = 30_000
            network_idle_timeout_ms = 20_000
            current_url = str(getattr(page, "url", "") or "")
            # Public repositories do not redirect anonymous visitors to the
            # login page. Always visit the login endpoint first so mutating GUI
            # tasks run as the evaluator's configured owner.
            if not current_url.startswith(login_url):
                navigate_browser_target(session, login_url, timeout_ms=45_000)
                page = session.browser_page
                self._settle_browser_page(page)
            _assert_browser_page_ready(
                session,
                required_selectors=("body",),
                app_name="Gitea",
                timeout_ms=body_ready_timeout_ms,
            )
            try:
                page.wait_for_load_state("domcontentloaded", timeout=body_ready_timeout_ms)
            except Exception:
                pass
            _assert_browser_page_ready(
                session,
                required_selectors=("body",),
                app_name="Gitea",
                timeout_ms=body_ready_timeout_ms,
            )
            if (
                str(getattr(page, "url", "") or "").startswith(login_url)
                or page.locator('input[name="user_name"]').count() > 0
            ):
                username, password = self._basic_auth
                username_field = page.locator('input[name="user_name"]').first
                password_field = page.locator('input[name="password"]').first
                username_field.fill(username)
                password_field.fill(password)
                password_field.press("Enter")
                try:
                    page.wait_for_load_state("networkidle", timeout=network_idle_timeout_ms)
                except Exception:
                    pass
                _assert_browser_page_ready(
                    session,
                    required_selectors=("body",),
                    app_name="Gitea",
                    timeout_ms=body_ready_timeout_ms,
                )
                if (
                    str(getattr(page, "url", "") or "").startswith(login_url)
                    or page.locator('input[name="user_name"]').count() > 0
                ):
                    raise GUISessionStartupError(
                        "window_timeout",
                        "Gitea login did not complete before opening the task page.",
                    )
            if not str(page.url).startswith(target_url):
                navigate_browser_target(session, target_url, timeout_ms=45_000)
                page = session.browser_page
                self._settle_browser_page(page)
                _assert_browser_page_ready(
                    session,
                    required_selectors=("body",),
                    app_name="Gitea",
                    timeout_ms=body_ready_timeout_ms,
                )
                try:
                    page.wait_for_load_state("networkidle", timeout=network_idle_timeout_ms)
                except Exception:
                    pass
                _assert_browser_page_ready(
                    session,
                    required_selectors=("body",),
                    app_name="Gitea",
                    timeout_ms=body_ready_timeout_ms,
                )
            return

        title_pattern = r".*"
        class_pattern = r"chromium|Chromium|chrome|Google-chrome"
        username, password = self._basic_auth
        type_text_to_window(
            title_pattern,
            username,
            window_class_pattern=class_pattern,
            min_width=1000,
            min_height=700,
        )
        send_keys_to_window(
            title_pattern,
            ["Tab"],
            window_class_pattern=class_pattern,
            min_width=1000,
            min_height=700,
        )
        type_text_to_window(
            title_pattern,
            password,
            window_class_pattern=class_pattern,
            min_width=1000,
            min_height=700,
        )
        send_keys_to_window(
            title_pattern,
            ["Return"],
            window_class_pattern=class_pattern,
            min_width=1000,
            min_height=700,
        )
        time.sleep(2.0)

    def _settle_browser_page(self, page) -> None:
        wait_for_timeout = getattr(page, "wait_for_timeout", None)
        if callable(wait_for_timeout):
            try:
                wait_for_timeout(1_000)
            except Exception:
                pass

    def _login_url(self) -> str:
        return f"{self.base_url}/user/login"

    def sync_from_gui(self, session=None) -> None:
        page = getattr(session, "browser_page", None) if session is not None else None
        live_url = str(getattr(page, "url", "") or "")
        if not live_url:
            return
        try:
            base = urlsplit(self.base_url)
            live = urlsplit(live_url)
            default_ports = {"http": 80, "https": 443}
            base_port = base.port or default_ports.get(base.scheme.casefold())
            live_port = live.port or default_ports.get(live.scheme.casefold())
        except ValueError:
            return
        if (
            live.scheme.casefold() != base.scheme.casefold()
            or (live.hostname or "").casefold() != (base.hostname or "").casefold()
            or live_port != base_port
        ):
            return
        # app_view must describe the page the GUI actually shows. Query and
        # fragment state do not identify a Gitea surface for evaluator rules.
        self._current_ui_path = live.path or "/"

    def _probe_ui_ready(self, session) -> None:
        from asil.gui_agent.session import (
            GUISessionStartupError,
            _assert_browser_page_ready,
            _browser_page_failure_category,
            navigate_browser_target,
        )

        _assert_browser_page_ready(session, required_selectors=("body",), app_name="Gitea")
        page = session.browser_page
        target_explore = self._current_ui_path == "/explore/repos"
        repo_selectors = (
            "#repo-files-table",
            ".repository",
            ".repo-header",
            ".repo-button-row",
            ".repository.view.issue",
            f'a[href="/{self.owner}/{self.repo}/issues"]',
            f'a[href="/{self.owner}/{self.repo}/pulls"]',
        )
        explore_selectors = (
            ".ui.repository.list",
            ".repo-search",
            ".explore.repository",
            ".user.profile.repositories",
            'a[href="/explore/repos"]',
        )
        repo_ready_script = f"""
            () => {{
                const repoSelectors = [
                    '#repo-files-table',
                    '.repository',
                    '.repo-header',
                    '.repo-button-row',
                    '.repository.view.issue',
                    'a[href="/{self.owner}/{self.repo}/issues"]',
                    'a[href="/{self.owner}/{self.repo}/pulls"]',
                ];
                const exploreSelectors = [
                    '.ui.repository.list',
                    '.repo-search',
                    '.explore.repository',
                    '.user.profile.repositories',
                    'a[href="/explore/repos"]',
                ];
                if ({str(target_explore).lower()} &&
                    (exploreSelectors.some((selector) => document.querySelector(selector)) ||
                     (location.pathname.startsWith('/explore/repos') &&
                      document.querySelectorAll('a[href^="/{self.owner}/"]').length > 0))) {{
                    return 'explore';
                }}
                if (repoSelectors.some((selector) => document.querySelector(selector))) {{
                    return 'repo';
                }}
                if (document.querySelector('input[name="user_name"]')) {{
                    return 'login';
                }}
                const bodyText = (document.body?.innerText || '').toLowerCase();
                if (bodyText.includes('page not found')) {{
                    return 'not_found';
                }}
                return false;
            }}
        """
        try:
            page.wait_for_function(repo_ready_script, timeout=45_000)
        except Exception:
            failure_category = _browser_page_failure_category(page)
            if failure_category == "browser_crashed":
                raise GUISessionStartupError(
                    "browser_crashed",
                    "Gitea page failed before it became ready.",
                )
            if failure_category == "blank_shell":
                navigate_browser_target(session, f"{self.base_url}{self._current_ui_path}", timeout_ms=30_000)
                page = session.browser_page
                try:
                    page.wait_for_function(repo_ready_script, timeout=15_000)
                except Exception:
                    pass
        if target_explore and (
            any(page.locator(selector).count() > 0 for selector in explore_selectors)
            or (
                str(getattr(page, "url", "") or "").startswith(f"{self.base_url}/explore/repos")
                and page.locator(f'a[href^="/{self.owner}/"]').count() > 0
            )
        ):
            return
        if any(page.locator(selector).count() > 0 for selector in repo_selectors):
            return
        body_text = ""
        try:
            body_text = str(page.locator("body").inner_text(timeout=1_000) or "")
        except Exception:
            body_text = ""
        if "page not found" in body_text.lower():
            raise GUISessionStartupError("backend_unready", "Gitea target page returned Page Not Found.")
        if page.locator('input[name="user_name"]').count() > 0:
            raise GUISessionStartupError("window_timeout", "Gitea login form remained visible after session priming.")
        if target_explore:
            raise GUISessionStartupError("window_timeout", "Gitea repository explorer UI did not become ready.")
        raise GUISessionStartupError("window_timeout", "Gitea repository UI did not become ready.")

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"token {self.token}", "Content-Type": "application/json"}

    _NO_PROXY = {"http": None, "https": None}
    _AUX_REPOS = ("old-project", "project-showcase")

    def _request(self, method: str, endpoint: str, *, body: dict | None = None):
        http_fn = getattr(requests, method.lower())
        url = f"{self.base_url}{endpoint}"
        kwargs: dict[str, Any] = {
            "timeout": 10,
            "proxies": self._NO_PROXY,
        }
        if body is not None and method in {"POST", "PUT", "PATCH"}:
            kwargs["json"] = body

        if self.token:
            resp = http_fn(url, headers=self._headers, **kwargs)
            if resp.status_code != 401:
                return resp

        return http_fn(
            url,
            headers={"Content-Type": "application/json"},
            auth=self._basic_auth,
            **kwargs,
        )

    def _get(self, endpoint: str) -> Any:
        resp = self._request("GET", endpoint)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def _patch(self, endpoint: str, body: dict) -> None:
        self._request("PATCH", endpoint, body=body).raise_for_status()

    def _delete(self, endpoint: str) -> None:
        resp = self._request("DELETE", endpoint)
        if resp.status_code not in (200, 204, 404):
            resp.raise_for_status()

    def reset_state(self) -> None:
        """Reset repo to a clean state by deleting and recreating it.

        This resets all issue/PR/label/milestone counters to 1.
        Recreates the repo with main branch and feature/login-fix branch.
        Called before each task to ensure task isolation.
        """
        for repo_name in self._AUX_REPOS:
            self._delete(f"/api/v1/repos/{self.owner}/{repo_name}")

        # Delete the repo
        self._delete(f"/api/v1/repos/{self.owner}/{self.repo}")

        # Recreate with auto_init so main branch exists
        self._request(
            "POST",
            "/api/v1/user/repos",
            body={"name": self.repo, "auto_init": True, "default_branch": "main"},
        ).raise_for_status()

        # Add a commit to feature/login-fix so PRs can be created
        # Retry a few times since the repo may not be fully initialized yet
        import time
        readme = None
        for _ in range(5):
            readme = self._get(f"/api/v1/repos/{self.owner}/{self.repo}/contents/README.md?ref=main")
            if readme:
                break
            time.sleep(0.5)
        if readme:
            import base64
            self._request(
                "POST",
                f"/api/v1/repos/{self.owner}/{self.repo}/branches",
                body={"new_branch_name": "feature/login-fix", "old_branch_name": "main"},
            ).raise_for_status()
            self._request(
                "PUT",
                f"/api/v1/repos/{self.owner}/{self.repo}/contents/README.md",
                body={
                    "message": "fix: resolve mobile login issue",
                    "content": base64.b64encode(b"Fix mobile login bug\n").decode(),
                    "branch": "feature/login-fix",
                    "sha": readme["sha"],
                },
            ).raise_for_status()
        self._current_ui_path = f"/{self.owner}/{self.repo}"

    def setup_state(self, initial_state: str) -> None:
        """Populate repo with the data required by a given initial_state key.

        Called after reset_state() so each task starts from a known baseline.

        States:
          "default"           — empty repo (no issues/labels/milestones/PRs)
          "with_issues"       — one open issue: "Initial issue for testing"
          "with_closed_issue" — one closed issue: "Initial issue for testing"
          "with_extra_repo"   — create an additional repo owned by the user
        """
        if initial_state in ("with_issues", "with_closed_issue"):
            resp = self._request(
                "POST",
                f"/api/v1/repos/{self.owner}/{self.repo}/issues",
                body={"title": "Initial issue for testing", "body": "Used by evaluation tasks."},
            )
            resp.raise_for_status()
            if initial_state == "with_closed_issue":
                issue_num = resp.json()["number"]
                self._request(
                    "PATCH",
                    f"/api/v1/repos/{self.owner}/{self.repo}/issues/{issue_num}",
                    body={"state": "closed"},
                ).raise_for_status()
        elif initial_state == "with_extra_repo":
            self._request(
                "POST",
                "/api/v1/user/repos",
                body={"name": "old-project", "auto_init": True, "default_branch": "main"},
            ).raise_for_status()
            self._current_ui_path = "/explore/repos"
            return
        self._current_ui_path = f"/{self.owner}/{self.repo}/issues" if "issue" in initial_state else f"/{self.owner}/{self.repo}"

    def observe(self) -> Observation:
        """Return a full snapshot of the current Gitea state.

        Fetches repos, issues, PRs, milestones, and labels so that all
        validation rules can be evaluated against the returned elements.
        """
        elements: list[Element] = []

        # --- Repositories ---
        try:
            data = self._get("/api/v1/repos/search?limit=50")
            repos = data.get("data", []) if isinstance(data, dict) else (data or [])
            for repo in repos:
                elements.append(Element(
                    id=f"repo:{repo['full_name']}",
                    type="repository",
                    label=repo["full_name"],
                    value={
                        "name": repo.get("name", ""),
                        "description": repo.get("description", ""),
                        "private": repo.get("private", False),
                        "has_issues": repo.get("has_issues", True),
                        "has_wiki": repo.get("has_wiki", True),
                        "has_projects": repo.get("has_projects", True),
                        "stars": repo.get("stars_count", 0),
                        "forks": repo.get("forks_count", 0),
                        "language": repo.get("language", ""),
                    },
                    editable=True,
                    actions=["edit_settings", "create_issue", "list_issues", "delete"],
                ))
        except Exception:
            pass

        # --- Issues ---
        try:
            issues = self._get(f"/api/v1/repos/{self.owner}/{self.repo}/issues?type=issues&limit=50&state=open") or []
            closed = self._get(f"/api/v1/repos/{self.owner}/{self.repo}/issues?type=issues&limit=50&state=closed") or []
            for issue in (issues + closed):
                elements.append(Element(
                    id=f"issue:{issue['number']}",
                    type="issue",
                    label=issue["title"],
                    value={
                        "number": issue["number"],
                        "title": issue["title"],
                        "body": issue.get("body", ""),
                        "state": issue["state"],
                        "labels": [lbl["name"] for lbl in (issue.get("labels") or [])],
                        "assignees": [a["login"] for a in (issue.get("assignees") or [])],
                        "milestone": (issue.get("milestone") or {}).get("title"),
                    },
                    editable=True,
                    actions=["edit", "close", "reopen", "add_label", "assign", "comment"],
                ))
        except Exception:
            pass

        # --- Pull Requests ---
        try:
            prs = self._get(f"/api/v1/repos/{self.owner}/{self.repo}/pulls?state=all&limit=50") or []
            for pr in prs:
                elements.append(Element(
                    id=f"pr:{pr['number']}",
                    type="pull_request",
                    label=pr["title"],
                    value={
                        "number": pr["number"],
                        "title": pr["title"],
                        "body": pr.get("body", ""),
                        "state": pr["state"],
                        "head": pr.get("head", {}).get("label", ""),
                        "base": pr.get("base", {}).get("label", ""),
                        "merged": pr.get("merged", pr.get("has_merged", False)),
                    },
                    editable=True,
                    actions=["merge", "close", "review"],
                ))
        except Exception:
            pass

        # --- Milestones ---
        try:
            milestones = self._get(f"/api/v1/repos/{self.owner}/{self.repo}/milestones?limit=50") or []
            for ms in milestones:
                elements.append(Element(
                    id=f"milestone:{ms['id']}",
                    type="milestone",
                    label=ms["title"],
                    value={
                        "id": ms["id"],
                        "title": ms["title"],
                        "description": ms.get("description", ""),
                        "due_on": ms.get("due_on", ""),
                        "open_issues": ms.get("open_issues", 0),
                        "closed_issues": ms.get("closed_issues", 0),
                        "state": ms.get("state", "open"),
                    },
                    editable=True,
                    actions=["edit", "close"],
                ))
        except Exception:
            pass

        # --- Labels ---
        try:
            labels = self._get(f"/api/v1/repos/{self.owner}/{self.repo}/labels?limit=50") or []
            for lbl in labels:
                elements.append(Element(
                    id=f"label:{lbl['id']}",
                    type="label",
                    label=lbl["name"],
                    value={
                        "id": lbl["id"],
                        "name": lbl["name"],
                        "color": "#" + lbl.get("color", "").lstrip("#"),
                        "description": lbl.get("description", ""),
                    },
                    editable=True,
                    actions=["edit", "delete"],
                ))
        except Exception:
            pass

        # --- Issue Comments (for comment-related tasks) ---
        try:
            comments = self._get(f"/api/v1/repos/{self.owner}/{self.repo}/issues/comments?limit=50") or []
            for c in comments:
                elements.append(Element(
                    id=f"comment:{c['id']}",
                    type="comment",
                    label=c.get("body", "")[:80],
                    value={
                        "id": c["id"],
                        "issue_url": c.get("issue_url", ""),
                        "body": c.get("body", ""),
                        "user": (c.get("user") or {}).get("login", ""),
                    },
                    editable=True,
                    actions=["edit", "delete"],
                ))
        except Exception:
            pass

        # --- Releases ---
        try:
            releases = self._get(f"/api/v1/repos/{self.owner}/{self.repo}/releases?limit=50") or []
            for rel in releases:
                elements.append(Element(
                    id=f"release:{rel['id']}",
                    type="release",
                    label=rel.get("tag_name", ""),
                    value={
                        "id": rel["id"],
                        "tag_name": rel.get("tag_name", ""),
                        "name": rel.get("name", ""),
                        "body": rel.get("body", ""),
                        "draft": rel.get("draft", False),
                        "prerelease": rel.get("prerelease", False),
                    },
                    editable=True,
                    actions=["edit", "delete"],
                ))
        except Exception:
            pass

        return self._build_observation(
            source="rest_api",
            elements=elements,
            app_state={"current_view": self._current_ui_path.lstrip("/")},
            data_summary=f"Gitea: {len(elements)} items in {self.owner}/{self.repo}",
        )

    def _sync_current_ui_path(self, endpoint: str) -> None:
        repo_root = f"/{self.owner}/{self.repo}"
        if "/repos/search" in endpoint:
            self._current_ui_path = "/explore/repos"
        elif "/pulls" in endpoint:
            self._current_ui_path = f"{repo_root}/pulls"
        elif "/milestones" in endpoint:
            self._current_ui_path = f"{repo_root}/milestones"
        elif "/labels" in endpoint:
            self._current_ui_path = f"{repo_root}/labels"
        elif "/releases" in endpoint:
            self._current_ui_path = f"{repo_root}/releases"
        elif "/issues" in endpoint or "/comments" in endpoint:
            self._current_ui_path = f"{repo_root}/issues"
        else:
            self._current_ui_path = repo_root

    def _resolve_label_ids(self, label_names_or_ids: list) -> list[int]:
        """Convert a list of label names or IDs to a list of integer IDs."""
        if not label_names_or_ids:
            return []
        if all(isinstance(x, int) for x in label_names_or_ids):
            return label_names_or_ids
        labels = self._get(f"/api/v1/repos/{self.owner}/{self.repo}/labels?limit=50") or []
        name_to_id = {lbl["name"]: lbl["id"] for lbl in labels}
        result = []
        for item in label_names_or_ids:
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, str) and item in name_to_id:
                result.append(name_to_id[item])
        return result

    def _resolve_milestone_id(self, milestone_name_or_id) -> int | None:
        """Convert a milestone name or positional index (1-based) to an actual Gitea milestone ID.

        In ground-truth tasks, milestone IDs are written as 1, 2, etc. (positional).
        Gitea assigns global IDs that don't reset between repos, so we resolve by position.
        String values are resolved by title match.
        """
        milestones = self._get(f"/api/v1/repos/{self.owner}/{self.repo}/milestones?limit=50") or []
        if isinstance(milestone_name_or_id, str):
            for ms in milestones:
                if ms["title"] == milestone_name_or_id:
                    return ms["id"]
            return None
        # Integer: treat as 1-based positional index into the milestone list
        idx = int(milestone_name_or_id) - 1
        if 0 <= idx < len(milestones):
            return milestones[idx]["id"]
        return None

    def execute(self, action: Action) -> Observation:
        method = action.params.get("method", "GET").upper()
        endpoint = action.params.get("endpoint", "")
        body = dict(action.params.get("body", {}))

        # Resolve label names to IDs for label assignment endpoints
        if "labels" in body and "/labels" in endpoint:
            body["labels"] = self._resolve_label_ids(body["labels"])

        # Resolve milestone name to ID when creating/editing issues
        if "milestone" in body and isinstance(body["milestone"], (str, int)):
            resolved = self._resolve_milestone_id(body["milestone"])
            if resolved is not None:
                body["milestone"] = resolved

        endpoint = endpoint.replace("{{owner}}", self.owner).replace("{{repo}}", self.repo)
        self._sync_current_ui_path(endpoint)
        # Merge endpoints return 405 while Gitea computes the merge index; retry with backoff
        is_merge = "/merge" in endpoint and method == "POST"
        if is_merge:
            import time
            for attempt in range(5):
                resp = self._request(method, endpoint, body=body)
                if resp.status_code != 405:
                    break
                time.sleep(1.0)
        else:
            self._request(method, endpoint, body=body)

        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="web_ui_screenshot",
            backend="playwright+chromium",
            actual_page=True,
            description="Screenshot of the current Gitea web UI page",
        )

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        """Capture the current Gitea page as a PNG."""
        out = Path(output_path) if output_path else Path(f"gitea_{self.repo}.png")
        url = f"{self.base_url}{self._current_ui_path}"
        return capture_url_to_png(url, out)


class MockGiteaAdapter(ASILAdapter):
    app_name = "Gitea"
    supported_action_types = ["api_call"]

    def __init__(self, owner: str = "asil_admin", repo: str = "test-repo") -> None:
        self.owner = owner
        self.repo = repo
        self._current_ui_path = f"/{self.owner}/{self.repo}"
        self.reset_state()

    def get_context(self) -> dict[str, str]:
        return {"owner": self.owner, "repo": self.repo, "token": "mock-token"}

    def reset_state(self) -> None:
        self.repos: dict[str, dict[str, Any]] = {}
        self.issues: list[dict[str, Any]] = []
        self.labels: list[dict[str, Any]] = []
        self.milestones: list[dict[str, Any]] = []
        self.pulls: list[dict[str, Any]] = []
        self.comments: list[dict[str, Any]] = []
        self.releases: list[dict[str, Any]] = []
        self._next_label_id = 1
        self._next_milestone_id = 1
        self._next_comment_id = 1
        self._next_release_id = 1
        self._create_repo(self.owner, self.repo, {"name": self.repo, "auto_init": True, "default_branch": "main"})
        self._current_ui_path = f"/{self.owner}/{self.repo}"

    def prepare_task(self, task: Any) -> None:
        self.reset_state()
        self.setup_state(getattr(task, "initial_state", "default") or "default")

    def setup_state(self, initial_state: str) -> None:
        if initial_state in ("with_issues", "with_closed_issue"):
            issue = self._create_issue({"title": "Initial issue for testing", "body": "Used by evaluation tasks."})
            if initial_state == "with_closed_issue":
                issue["state"] = "closed"
        elif initial_state == "with_extra_repo":
            self._create_repo(self.owner, "old-project", {"name": "old-project", "auto_init": True, "default_branch": "main"})
            self._current_ui_path = "/explore/repos"
            return
        self._current_ui_path = f"/{self.owner}/{self.repo}/issues" if "issue" in initial_state else f"/{self.owner}/{self.repo}"

    def _create_repo(self, owner: str, name: str, body: dict[str, Any]) -> dict[str, Any]:
        repo = {
            "owner": owner,
            "name": name,
            "full_name": f"{owner}/{name}",
            "description": body.get("description", ""),
            "private": bool(body.get("private", False)),
            "has_issues": bool(body.get("has_issues", True)),
            "has_wiki": bool(body.get("has_wiki", True)),
            "has_projects": bool(body.get("has_projects", True)),
            "stars": 0,
            "forks": 0,
            "language": "",
        }
        self.repos[repo["full_name"]] = repo
        return repo

    def _create_issue(self, body: dict[str, Any]) -> dict[str, Any]:
        issue = {
            "number": int(body.get("_number") or len(self.issues) + 1),
            "title": body.get("title", ""),
            "body": body.get("body", ""),
            "state": body.get("state", "open"),
            "labels": list(body.get("labels", [])),
            "assignees": list(body.get("assignees", [])),
            "milestone": body.get("milestone"),
        }
        self.issues.append(issue)
        return issue

    def _issue(self, number: int = 1) -> dict[str, Any]:
        for issue in self.issues:
            if int(issue["number"]) == int(number):
                return issue
        return self._create_issue({"_number": number, "title": "Initial issue for testing", "body": "Used by evaluation tasks."})

    def _recompute_milestone_counts(self) -> None:
        for milestone in self.milestones:
            milestone_id = milestone.get("id")
            open_count = 0
            closed_count = 0
            for issue in self.issues:
                if issue.get("milestone") != milestone_id:
                    continue
                if issue.get("state") == "closed":
                    closed_count += 1
                else:
                    open_count += 1
            milestone["open_issues"] = open_count
            milestone["closed_issues"] = closed_count

    def execute(self, action: Action) -> Observation:
        method = str(action.params.get("method", "GET")).upper()
        endpoint = str(action.params.get("endpoint", ""))
        body = dict(action.params.get("body", {}) or {})
        endpoint = endpoint.replace("{{owner}}", self.owner).replace("{{repo}}", self.repo)
        self._sync_current_ui_path(endpoint)

        if method == "POST" and endpoint == "/api/v1/user/repos":
            owner = str(body.get("_owner") or self.owner)
            name = str(body.get("name") or "new-repo")
            self._create_repo(owner, name, body)
        elif method in {"PATCH", "PUT"} and endpoint == f"/api/v1/repos/{self.owner}/{self.repo}":
            repo = self.repos[f"{self.owner}/{self.repo}"]
            for key in ("description", "private", "has_issues", "has_wiki", "has_projects"):
                if key in body:
                    repo[key] = body[key]
        elif method == "POST" and endpoint.endswith("/labels") and "/issues/" in endpoint:
            issue = self._issue(1)
            issue["labels"] = list(body.get("labels", []))
        elif method == "POST" and endpoint.endswith("/labels"):
            self.labels.append(
                {
                    "id": int(body.get("_id") or self._next_label_id),
                    "name": body.get("name", ""),
                    "color": str(body.get("color", "")).lstrip("#"),
                    "description": body.get("description", ""),
                }
            )
            self._next_label_id = max(self._next_label_id + 1, int(body.get("_id") or 0) + 1)
        elif method == "POST" and endpoint.endswith("/milestones"):
            self.milestones.append(
                {
                    "id": int(body.get("_id") or self._next_milestone_id),
                    "title": body.get("title", ""),
                    "description": body.get("description", ""),
                    "due_on": body.get("due_on", ""),
                    "open_issues": int(body.get("open_issues", 0) or 0),
                    "closed_issues": int(body.get("closed_issues", 0) or 0),
                    "state": body.get("state", "open"),
                }
            )
            self._next_milestone_id = max(self._next_milestone_id + 1, int(body.get("_id") or 0) + 1)
        elif method == "POST" and endpoint.endswith("/issues"):
            self._create_issue(body)
        elif method in {"PATCH", "PUT"} and "/issues/" in endpoint:
            issue = self._issue(int(body.get("_number") or endpoint.rstrip("/").split("/")[-1]))
            for key in ("title", "body", "state", "labels", "assignees", "milestone"):
                if key in body:
                    issue[key] = body[key]
        elif method == "POST" and endpoint.endswith("/comments"):
            self.comments.append(
                {
                    "id": int(body.get("_id") or self._next_comment_id),
                    "body": body.get("body", ""),
                    "issue_url": f"{self.owner}/{self.repo}/issues/1",
                    "user": self.owner,
                }
            )
            self._next_comment_id = max(self._next_comment_id + 1, int(body.get("_id") or 0) + 1)
        elif method == "POST" and endpoint.endswith("/pulls"):
            self.pulls.append(
                {
                    "number": int(body.get("_number") or len(self.pulls) + 1),
                    "title": body.get("title", ""),
                    "body": body.get("body", ""),
                    "state": body.get("state", "open"),
                    "head": body.get("head", ""),
                    "base": body.get("base", "main"),
                    "merged": bool(body.get("merged", False)),
                }
            )
        elif method == "POST" and endpoint.endswith("/merge"):
            if not self.pulls:
                self.pulls.append({"number": 1, "title": "", "body": "", "state": "open", "head": "", "base": "main", "merged": False})
            self.pulls[0]["merged"] = True
            self.pulls[0]["state"] = "closed"
        elif method == "POST" and endpoint.endswith("/releases"):
            self.releases.append(
                {
                    "id": int(body.get("_id") or self._next_release_id),
                    "tag_name": body.get("tag_name", ""),
                    "name": body.get("name", ""),
                    "body": body.get("body", ""),
                    "draft": bool(body.get("draft", False)),
                    "prerelease": bool(body.get("prerelease", False)),
                }
            )
            self._next_release_id = max(self._next_release_id + 1, int(body.get("_id") or 0) + 1)
        self._recompute_milestone_counts()
        return self.observe()

    def _sync_current_ui_path(self, endpoint: str) -> None:
        repo_root = f"/{self.owner}/{self.repo}"
        if "/repos/search" in endpoint:
            self._current_ui_path = "/explore/repos"
        elif "/pulls" in endpoint:
            self._current_ui_path = f"{repo_root}/pulls"
        elif "/milestones" in endpoint:
            self._current_ui_path = f"{repo_root}/milestones"
        elif "/labels" in endpoint:
            self._current_ui_path = f"{repo_root}/labels"
        elif "/releases" in endpoint:
            self._current_ui_path = f"{repo_root}/releases"
        elif "/issues" in endpoint or "/comments" in endpoint:
            self._current_ui_path = f"{repo_root}/issues"
        else:
            self._current_ui_path = repo_root

    def observe(self) -> Observation:
        elements: list[Element] = []
        for repo in self.repos.values():
            elements.append(
                Element(
                    id=f"repo:{repo['full_name']}",
                    type="repository",
                    label=repo["full_name"],
                    value={k: repo.get(k, "") for k in ("name", "description", "private", "has_issues", "has_wiki", "has_projects", "stars", "forks", "language")},
                    editable=True,
                    actions=["edit_settings", "create_issue", "list_issues", "delete"],
                )
            )
        for issue in self.issues:
            elements.append(
                Element(
                    id=f"issue:{issue['number']}",
                    type="issue",
                    label=issue["title"],
                    value={
                        "number": issue["number"],
                        "title": issue["title"],
                        "body": issue.get("body", ""),
                        "state": issue.get("state", "open"),
                        "labels": issue.get("labels", []),
                        "assignees": issue.get("assignees", []),
                        "milestone": issue.get("milestone"),
                    },
                    editable=True,
                    actions=["edit", "close", "reopen", "add_label", "assign", "comment"],
                )
            )
        for pr in self.pulls:
            elements.append(
                Element(
                    id=f"pr:{pr['number']}",
                    type="pull_request",
                    label=pr["title"],
                    value=dict(pr),
                    editable=True,
                    actions=["merge", "close", "review"],
                )
            )
        for ms in self.milestones:
            elements.append(Element(id=f"milestone:{ms['id']}", type="milestone", label=ms["title"], value=dict(ms), editable=True, actions=["edit", "close"]))
        for lbl in self.labels:
            value = dict(lbl)
            value["color"] = "#" + str(value.get("color", "")).lstrip("#")
            elements.append(Element(id=f"label:{lbl['id']}", type="label", label=lbl["name"], value=value, editable=True, actions=["edit", "delete"]))
        for c in self.comments:
            elements.append(Element(id=f"comment:{c['id']}", type="comment", label=str(c.get("body", ""))[:80], value=dict(c), editable=True, actions=["edit", "delete"]))
        for rel in self.releases:
            elements.append(Element(id=f"release:{rel['id']}", type="release", label=rel.get("tag_name", ""), value=dict(rel), editable=True, actions=["edit", "delete"]))
        return self._build_observation(
            source="mock_rest_api",
            elements=elements,
            app_state={"current_view": self._current_ui_path.lstrip("/")},
            data_summary=f"Mock Gitea: {len(elements)} items in {self.owner}/{self.repo}",
        )

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types
