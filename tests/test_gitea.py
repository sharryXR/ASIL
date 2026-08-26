"""Tests for Gitea adapter — uses mocked HTTP, no Gitea server required."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from asil.adapters.gitea import GiteaAdapter
from asil.gui_agent.session import GUISessionStartupError
from asil.protocol import Action

MOCK_REPOS = {
    "data": [
        {
            "id": 1,
            "full_name": "user/project-a",
            "name": "project-a",
            "description": "A test project",
            "stars_count": 5,
            "forks_count": 1,
            "private": False,
            "language": "Python",
        },
        {
            "id": 2,
            "full_name": "user/project-b",
            "name": "project-b",
            "description": "",
            "stars_count": 0,
            "forks_count": 0,
            "private": True,
            "language": "Go",
        },
    ]
}

MOCK_ISSUES = [
    {
        "number": 1,
        "title": "Fix login bug",
        "body": "Users cannot log in on mobile Safari.",
        "state": "open",
        "labels": [{"name": "bug"}],
        "assignees": [{"login": "dev1"}],
        "milestone": {"title": "v1.0"},
    },
    {
        "number": 2,
        "title": "Add dark mode",
        "state": "open",
        "labels": [{"name": "enhancement"}],
        "assignees": [],
        "milestone": None,
    },
]


def _make_response(json_data, status_code=200):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


def _mock_get_for_observe(url, **kwargs):
    """Return appropriate mock data based on the URL being requested."""
    if "/repos/search" in url:
        return _make_response(MOCK_REPOS)
    if "/issues" in url and "state=open" in url:
        return _make_response(MOCK_ISSUES)
    if "/issues" in url and "state=closed" in url:
        return _make_response([])
    if "/pulls" in url:
        return _make_response([])
    if "/milestones" in url:
        return _make_response([])
    if "/labels" in url:
        return _make_response([])
    # Default: 404
    return _make_response(None, 404)


@patch("asil.adapters.gitea.requests")
def test_observe_repos(mock_requests):
    mock_requests.get.side_effect = _mock_get_for_observe
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    obs = adapter.observe()

    assert obs.meta.app_name == "Gitea"
    repo_elements = [e for e in obs.interactive_elements if e.type == "repository"]
    assert len(repo_elements) == 2
    ids = {e.id for e in repo_elements}
    assert "repo:user/project-a" in ids


@patch("asil.adapters.gitea.requests")
def test_observe_issues(mock_requests):
    mock_requests.get.side_effect = _mock_get_for_observe
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    obs = adapter.observe()

    issue_elements = [e for e in obs.interactive_elements if e.type == "issue"]
    assert len(issue_elements) == 2
    bug = next(e for e in issue_elements if e.id == "issue:1")
    assert bug.value["state"] == "open"
    assert "bug" in bug.value["labels"]
    assert "mobile Safari" in bug.value["body"]


@patch("asil.adapters.gitea.requests")
def test_observe_milestones_include_due_on(mock_requests):
    def _mock_get(url, **kwargs):
        if "/repos/search" in url:
            return _make_response(MOCK_REPOS)
        if "/issues" in url and "state=open" in url:
            return _make_response([])
        if "/issues" in url and "state=closed" in url:
            return _make_response([])
        if "/pulls" in url:
            return _make_response([])
        if "/milestones" in url:
            return _make_response([
                {
                    "id": 1,
                    "title": "v1.0",
                    "description": "Initial milestone",
                    "due_on": "2026-06-30T00:00:00Z",
                    "open_issues": 2,
                    "closed_issues": 1,
                    "state": "open",
                }
            ])
        if "/labels" in url:
            return _make_response([])
        return _make_response(None, 404)

    mock_requests.get.side_effect = _mock_get
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    obs = adapter.observe()

    milestone = next(e for e in obs.interactive_elements if e.type == "milestone")
    assert milestone.value["due_on"] == "2026-06-30T00:00:00Z"


@patch("asil.adapters.gitea.requests")
def test_execute_create_issue(mock_requests):
    mock_requests.post.return_value = _make_response({"number": 3, "title": "New issue"}, 201)
    mock_requests.get.side_effect = _mock_get_for_observe

    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    action = Action(
        action_type="api_call",
        target="gitea_rest",
        params={
            "method": "POST",
            "endpoint": "/api/v1/repos/user/project-a/issues",
            "body": {"title": "New issue", "body": "Description"},
        },
    )
    adapter.execute(action)
    mock_requests.post.assert_called_once()
    call_args = mock_requests.post.call_args
    assert "/issues" in call_args[0][0]


@patch("asil.adapters.gitea.requests")
def test_execute_patch_repo(mock_requests):
    mock_requests.patch.return_value = _make_response({"full_name": "user/project-a"})
    mock_requests.get.side_effect = _mock_get_for_observe

    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    action = Action(
        action_type="api_call",
        target="gitea_rest",
        params={
            "method": "PATCH",
            "endpoint": "/api/v1/repos/user/project-a",
            "body": {"description": "Updated"},
        },
    )
    adapter.execute(action)
    mock_requests.patch.assert_called_once()


@patch("asil.adapters.gitea.requests")
def test_observe_includes_closed_merged_pull_requests(mock_requests):
    def _mock_get(url, **kwargs):
        if "/repos/search" in url:
            return _make_response(MOCK_REPOS)
        if "/issues" in url and "state=open" in url:
            return _make_response([])
        if "/issues" in url and "state=closed" in url:
            return _make_response([])
        if "/pulls" in url:
            return _make_response(
                [
                    {
                        "number": 1,
                        "title": "Merge login fix",
                        "body": "Fix mobile login bug.",
                        "state": "closed",
                        "merged": True,
                        "head": {"label": "asil_admin:feature/login-fix"},
                        "base": {"label": "asil_admin:main"},
                    }
                ]
            )
        if "/milestones" in url:
            return _make_response([])
        if "/labels" in url:
            return _make_response([])
        return _make_response(None, 404)

    mock_requests.get.side_effect = _mock_get
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    obs = adapter.observe()

    pr = next(e for e in obs.interactive_elements if e.type == "pull_request")
    assert pr.value["state"] == "closed"
    assert pr.value["merged"] is True


@patch("asil.adapters.gitea.requests.get")
def test_probe_backend_ready_waits_for_current_repo_to_exist(mock_get):
    login_response = _make_response({}, 200)
    target_missing = _make_response({}, 404)
    mock_get.side_effect = [login_response, target_missing]

    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    adapter._request = MagicMock()

    try:
        adapter._probe_backend_ready()
        assert False, "Expected backend_unready when target page still returns 404"
    except GUISessionStartupError as exc:
        assert exc.category == "backend_unready"
        assert "target page" in str(exc)


def test_gui_session_spec_primes_browser_login():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")

    with patch("asil.adapters.gitea.type_text_to_window") as mock_type, patch(
        "asil.adapters.gitea.send_keys_to_window"
    ) as mock_keys, patch("asil.adapters.gitea.time.sleep") as mock_sleep:
        spec = adapter.get_gui_session_spec()
        assert spec.post_launch_callback is not None
        assert spec.browser_url == "about:blank"
        assert spec.browser_navigation_mode == "current_page"
        assert spec.browser_post_navigation_settle_ms == 1_000
        spec.post_launch_callback()

    assert mock_type.call_count == 2
    assert mock_keys.call_count == 2
    typed_values = [call.args[1] for call in mock_type.call_args_list]
    assert typed_values == ["asil_admin", "asil_password"]
    sent_keys = [call.args[1] for call in mock_keys.call_args_list]
    assert sent_keys == [["Tab"], ["Return"]]
    mock_sleep.assert_called_once_with(2.0)
    assert spec.backend_ready_probe is not None
    assert spec.ui_ready_probe is not None


def test_prime_browser_session_submits_login_with_enter_and_navigates():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")

    events: list[tuple[str, str]] = []

    class FakeField:
        def __init__(self, name: str, page):
            self.name = name
            self.page = page

        @property
        def first(self):
            return self

        def fill(self, value: str):
            events.append((f"{self.name}:fill", value))

        def press(self, key: str):
            events.append((f"{self.name}:press", key))
            if self.name == "password" and key == "Enter":
                self.page.url = "http://localhost:3000/"

        def count(self) -> int:
            return int(self.page.url.endswith("/user/login"))

    class FakePage:
        def __init__(self):
            self.url = "http://localhost:3000/user/login"
            self.username = FakeField("username", self)
            self.password = FakeField("password", self)

        def content(self) -> str:
            return "<html><body>Sign In</body></html>"

        def wait_for_load_state(self, state: str, timeout: int = 0):
            events.append(("load_state", state))

        def wait_for_selector(self, selector: str, timeout: int = 0):
            assert selector == "body"

        def locator(self, selector: str):
            if selector == "body":
                class BodyField:
                    def count(self) -> int:
                        return 1

                    def inner_text(self, timeout: int = 0) -> str:
                        return "Sign In"

                return BodyField()
            if selector == 'input[name="user_name"]':
                return self.username
            if selector == 'input[name="password"]':
                return self.password
            raise AssertionError(f"unexpected selector: {selector}")

    class FakeSession:
        def __init__(self):
            self.browser_page = FakePage()

    session = FakeSession()

    with patch("asil.gui_agent.session.navigate_browser_target") as mock_navigate:
        adapter._prime_browser_session(session)

    assert ("username:fill", "asil_admin") in events
    assert ("password:fill", "asil_password") in events
    assert ("password:press", "Enter") in events
    mock_navigate.assert_called_once_with(session, "http://localhost:3000/asil_admin/test-repo", timeout_ms=45_000)


def test_prime_browser_session_authenticates_before_opening_public_repo():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    navigations: list[str] = []
    events: list[tuple[str, str]] = []

    class Locator:
        def __init__(self, page, selector: str):
            self.page = page
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self) -> int:
            if self.selector == "body":
                return 1
            return int(self.page.url.endswith("/user/login") and self.selector in {
                'input[name="user_name"]',
                'input[name="password"]',
            })

        def fill(self, value: str):
            events.append((f"{self.selector}:fill", value))

        def press(self, key: str):
            events.append((f"{self.selector}:press", key))
            if self.selector == 'input[name="password"]' and key == "Enter":
                self.page.url = "http://localhost:3000/"

        def inner_text(self, timeout: int = 0) -> str:
            return "Sign In" if self.page.url.endswith("/user/login") else "Repository ready"

    class FakePage:
        def __init__(self):
            self.url = "http://localhost:3000/asil_admin/test-repo"

        def content(self) -> str:
            return "<html><body>ready</body></html>"

        def wait_for_load_state(self, state: str, timeout: int = 0):
            return None

        def wait_for_selector(self, selector: str, timeout: int = 0):
            assert selector == "body"

        def wait_for_timeout(self, timeout_ms: int):
            return None

        def locator(self, selector: str):
            return Locator(self, selector)

    class FakeSession:
        def __init__(self):
            self.browser_page = FakePage()

    session = FakeSession()

    def fake_navigate(target_session, url, timeout_ms=0):
        assert target_session is session
        assert timeout_ms == 45_000
        navigations.append(url)
        session.browser_page.url = url

    with patch("asil.gui_agent.session.navigate_browser_target", side_effect=fake_navigate):
        adapter._prime_browser_session(session)

    assert navigations == [
        "http://localhost:3000/user/login",
        "http://localhost:3000/asil_admin/test-repo",
    ]
    assert ('input[name="user_name"]:fill', "asil_admin") in events
    assert ('input[name="password"]:fill', "asil_password") in events
    assert ('input[name="password"]:press', "Enter") in events


def test_prime_browser_session_rejects_failed_login_before_opening_public_repo():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")

    class Locator:
        def __init__(self, selector: str):
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self) -> int:
            return 1 if self.selector in {
                "body",
                'input[name="user_name"]',
                'input[name="password"]',
            } else 0

        def fill(self, value: str):
            return None

        def press(self, key: str):
            return None

        def inner_text(self, timeout: int = 0) -> str:
            return "Sign In"

    class FailedLoginPage:
        url = "http://localhost:3000/user/login"

        def content(self) -> str:
            return "<html><body>Sign In</body></html>"

        def wait_for_load_state(self, state: str, timeout: int = 0):
            return None

        def wait_for_selector(self, selector: str, timeout: int = 0):
            return None

        def locator(self, selector: str):
            return Locator(selector)

    session = type("Session", (), {"browser_page": FailedLoginPage()})()

    with patch("asil.gui_agent.session.navigate_browser_target") as mock_navigate:
        try:
            adapter._prime_browser_session(session)
            assert False, "Expected failed Gitea authentication to abort startup"
        except GUISessionStartupError as exc:
            assert exc.category == "window_timeout"
            assert "login" in str(exc).lower()

    mock_navigate.assert_not_called()


def test_prime_browser_session_visits_login_then_repo_from_blank_page():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")

    class FakePage:
        def __init__(self):
            self.url = "about:blank"

        def content(self) -> str:
            return "<html><body>Repository ready</body></html>"

        def wait_for_load_state(self, state: str, timeout: int = 0):
            return None

        def wait_for_selector(self, selector: str, timeout: int = 0):
            assert selector == "body"
            return None

        def locator(self, selector: str):
            page = self

            class Locator:
                def count(self) -> int:
                    if selector == "body":
                        return 1
                    return int(
                        page.url.endswith("/user/login")
                        and selector in {
                            'input[name="user_name"]',
                            'input[name="password"]',
                        }
                    )

                @property
                def first(self):
                    return self

                def fill(self, value: str):
                    return None

                def press(self, key: str):
                    if selector == 'input[name="password"]' and key == "Enter":
                        page.url = "http://localhost:3000/"

                def inner_text(self, timeout: int = 0) -> str:
                    return "Repository ready"

            return Locator()

    class FakeSession:
        def __init__(self):
            self.browser_page = FakePage()

    session = FakeSession()

    navigations: list[str] = []

    def fake_navigate(target_session, url, timeout_ms=0):
        assert target_session is session
        assert timeout_ms == 45_000
        navigations.append(url)
        session.browser_page.url = url

    with patch("asil.gui_agent.session.navigate_browser_target", side_effect=fake_navigate) as mock_navigate:
        adapter._prime_browser_session(session)

    assert mock_navigate.call_count == 2
    assert navigations == [
        "http://localhost:3000/user/login",
        "http://localhost:3000/asil_admin/test-repo",
    ]


def test_easy_gitea_task_fixtures_remain_frozen_for_gui50_comparability():
    examples = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "gitea"
    create_repo = json.loads((examples / "gitea_01.json").read_text(encoding="utf-8"))
    create_issue = json.loads((examples / "gitea_02.json").read_text(encoding="utf-8"))
    delete_repo = json.loads((examples / "gitea_04.json").read_text(encoding="utf-8"))

    assert create_repo["_asil"]["validation"]["element_exists"] == "repo:asil_admin/test-repo"
    assert create_repo["gui_expectations"]["visible_change_summary"] == (
        "Create a new public repository named 'test-repo'"
    )
    assert create_issue["instruction"] == "Create an issue titled 'Bug: login fails on mobile'"
    assert create_issue["_asil"]["validation"]["conditions"] == [
        {
            "element_contains": {
                "id": "issue:1",
                "key": "title",
                "expected": "Bug: login fails on mobile",
            }
        }
    ]
    assert delete_repo["_asil"]["initial_state"] == "default"


def test_prime_browser_session_raises_on_browser_crash_page():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")

    class CrashLocator:
        @property
        def first(self):
            return self

        def count(self) -> int:
            raise AssertionError("login fields should not be queried on a crash page")

    class CrashPage:
        url = "http://localhost:3000/user/login"

        def content(self) -> str:
            return "<html><body>Aw, Snap!</body></html>"

        def wait_for_load_state(self, state: str, timeout: int = 0):
            return None

        def locator(self, selector: str):
            if selector == "body":
                class BodyLocator:
                    def count(self) -> int:
                        return 1

                    def inner_text(self, timeout: int = 0) -> str:
                        return "Aw, Snap!"

                return BodyLocator()
            return CrashLocator()

    class FakeSession:
        def __init__(self):
            self.browser_page = CrashPage()

    try:
        adapter._prime_browser_session(FakeSession())
        assert False, "Expected browser crash to abort Gitea startup"
    except GUISessionStartupError as exc:
        assert exc.category == "browser_crashed"


def test_probe_ui_ready_waits_for_repository_surface_to_appear():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")

    class FakeLocator:
        def __init__(self, page, selector: str):
            self._page = page
            self._selector = selector

        def count(self) -> int:
            if self._selector == "body":
                return 1
            if self._selector == 'input[name="user_name"]':
                return 0
            repo_selectors = {
                "#repo-files-table",
                ".repository",
                ".repo-header",
                ".repo-button-row",
                f'a[href="/{adapter.owner}/{adapter.repo}/issues"]',
                f'a[href="/{adapter.owner}/{adapter.repo}/pulls"]',
            }
            return int(self._page.repo_ready and self._selector in repo_selectors)

        def inner_text(self, timeout: int = 0) -> str:
            return "repository ready" if self._page.repo_ready else "loading"

    class FakePage:
        def __init__(self) -> None:
            self.url = "http://localhost:3000/asil_admin/test-repo"
            self.repo_ready = False
            self.waited_for_function = False

        def content(self) -> str:
            return "<html><body>loading</body></html>"

        def locator(self, selector: str):
            return FakeLocator(self, selector)

        def wait_for_selector(self, selector: str, timeout: int = 0):
            return None

        def wait_for_function(self, script: str, timeout: int = 0):
            self.waited_for_function = True
            self.repo_ready = True
            return None

    session = MagicMock()
    session.browser_page = FakePage()

    adapter._probe_ui_ready(session)

    assert session.browser_page.waited_for_function is True


def test_probe_ui_ready_accepts_explore_repositories_surface():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    adapter._current_ui_path = "/explore/repos"

    class FakeLocator:
        def __init__(self, page, selector: str):
            self._page = page
            self._selector = selector

        def count(self) -> int:
            if self._selector == "body":
                return 1
            if self._selector == 'input[name="user_name"]':
                return 0
            explore_selectors = {
                ".ui.repository.list",
                ".repo-search",
                ".explore.repository",
                ".user.profile.repositories",
                'a[href="/explore/repos"]',
                f'a[href^="/{adapter.owner}/"]',
            }
            return int(self._page.explore_ready and self._selector in explore_selectors)

        def inner_text(self, timeout: int = 0) -> str:
            return "Repositories" if self._page.explore_ready else "Explore"

    class FakePage:
        def __init__(self) -> None:
            self.url = "http://localhost:3000/explore/repos"
            self.explore_ready = False
            self.waited_for_function = False

        def content(self) -> str:
            return "<html><body>Explore</body></html>"

        def locator(self, selector: str):
            return FakeLocator(self, selector)

        def wait_for_selector(self, selector: str, timeout: int = 0):
            return None

        def wait_for_function(self, script: str, timeout: int = 0):
            self.waited_for_function = True
            self.explore_ready = True
            return None

    session = MagicMock()
    session.browser_page = FakePage()

    adapter._probe_ui_ready(session)

    assert session.browser_page.waited_for_function is True


def test_sync_current_ui_path_maps_repo_search_to_explore():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")

    adapter._sync_current_ui_path("/api/v1/repos/search?limit=50")

    assert adapter._current_ui_path == "/explore/repos"


def test_sync_from_gui_uses_actual_browser_path_for_app_view():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    adapter._current_ui_path = "/explore/repos"
    session = MagicMock()
    session.browser_page.url = "http://localhost:3000/user/login?redirect_to=%2Fexplore%2Frepos"

    adapter.sync_from_gui(session)

    assert adapter._current_ui_path == "/user/login"


def test_sync_from_gui_ignores_unrelated_browser_origin():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    adapter._current_ui_path = "/explore/repos"
    session = MagicMock()
    session.browser_page.url = "https://example.com/not-gitea"

    adapter.sync_from_gui(session)

    assert adapter._current_ui_path == "/explore/repos"


def test_validate_action():
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="t")
    good = Action(action_type="api_call", target="gitea", params={})
    bad = Action(action_type="invoke_function", target="bpy", params={})
    assert adapter.validate_action(good)
    assert not adapter.validate_action(bad)


@patch("asil.adapters.gitea.capture_url_to_png")
def test_render_to_png_uses_current_ui_path(mock_capture):
    adapter = GiteaAdapter(base_url="http://localhost:3000", token="test-token")
    adapter._current_ui_path = "/asil_admin/test-repo/issues"
    adapter.render_to_png("/tmp/gitea.png")

    mock_capture.assert_called_once_with(
        "http://localhost:3000/asil_admin/test-repo/issues",
        Path("/tmp/gitea.png"),
    )


@patch("asil.adapters.gitea.requests")
def test_get_falls_back_to_basic_auth_after_unauthorized_token(mock_requests):
    unauthorized = _make_response({"message": "unauthorized"}, 401)
    authorized = _make_response({"login": "asil_admin"}, 200)
    mock_requests.get.side_effect = [unauthorized, authorized]

    adapter = GiteaAdapter(base_url="http://localhost:3000", token="bad-token")

    payload = adapter._get("/api/v1/user")

    assert payload == {"login": "asil_admin"}
    first_call = mock_requests.get.call_args_list[0]
    second_call = mock_requests.get.call_args_list[1]
    assert first_call.kwargs["headers"]["Authorization"] == "token bad-token"
    assert "auth" not in first_call.kwargs
    assert second_call.kwargs["auth"] == ("asil_admin", "asil_password")
