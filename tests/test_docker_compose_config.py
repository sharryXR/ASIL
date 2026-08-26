from pathlib import Path
import re


def test_gitea_initializer_can_write_shared_token_volume():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    service = re.search(r"(?ms)^  gitea-init:\n(.*?)(?=^  [a-zA-Z0-9_-]+:\n)", compose)

    assert service is not None
    assert 'user: "0:0"' in service.group(1)
    assert "gitea-token:/shared" in service.group(1)
    init_script = (Path(__file__).resolve().parent.parent / "docker" / "gitea-init.sh").read_text(
        encoding="utf-8"
    )
    assert "test -s /shared/gitea_token.txt" in init_script


def test_eval_service_uses_container_scoped_obs_env_defaults():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert "OBS_WS_HOST=${EVAL_OBS_WS_HOST:-obs-mock}" in compose
    assert "OBS_WS_PORT=${EVAL_OBS_WS_PORT:-4444}" in compose
    assert "OBS_WS_PROTOCOL=${EVAL_OBS_WS_PROTOCOL:-auto}" in compose
    assert "OBS_REAL_GUI=${EVAL_OBS_REAL_GUI:-true}" in compose


def test_eval_service_passes_gui_timeout_env_overrides():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert "ASIL_GUI_LLM_TIMEOUT_S=${ASIL_GUI_LLM_TIMEOUT_S:-}" in compose
    assert "ASIL_GUI_LLM_CALL_TIMEOUT_S=${ASIL_GUI_LLM_CALL_TIMEOUT_S:-}" in compose
    assert "ASIL_GUI_INIT_WATCHDOG_S=${ASIL_GUI_INIT_WATCHDOG_S:-}" in compose


def test_eval_service_passes_gui_backend_env_overrides():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert "ASIL_GUI_AGENT_BACKEND=${ASIL_GUI_AGENT_BACKEND:-}" in compose
    assert "ASIL_GUI_REASONING_EFFORT=${ASIL_GUI_REASONING_EFFORT:-}" in compose


def test_eval_service_includes_code_server_and_jupyterlab_sidecars():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert "code-server:" in compose
    assert "jupyterlab:" in compose
    assert "CODE_SERVER_URL=http://code-server:8080" in compose
    assert "JUPYTERLAB_URL=http://jupyterlab:8888" in compose
    assert "CODE_SERVER_WORKSPACE_ROOT=/shared-workspaces" in compose
    assert "JUPYTERLAB_WORKSPACE_ROOT=/shared-workspaces" in compose


def test_eval_service_includes_drawio_sidecar():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert "drawio:" in compose
    assert "DRAWIO_URL=http://drawio:8080" in compose


def test_multi_apps_reuses_existing_sidecars_without_new_service():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert "multi_apps:" not in compose
    assert "code-server:" in compose
    assert "jupyterlab:" in compose
    assert "drawio:" in compose


def test_browser_sidecars_use_pinned_image_digests():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert "codercom/code-server@sha256:" in compose
    assert "jupyter/minimal-notebook@sha256:" in compose
    assert "jgraph/drawio@sha256:" in compose
    assert "codercom/code-server:latest" not in compose
    assert "jgraph/drawio:latest" not in compose


def test_code_server_runs_with_stable_startup_flags():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert "--disable-workspace-trust" in compose
    assert "--disable-telemetry" in compose
    assert "--disable-update-check" in compose
    assert "--user-data-dir" in compose
    assert "/shared-workspaces/code-server-user-data" in compose


def test_eval_service_enables_init_to_reap_browser_zombies():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert "eval:" in compose
    assert "init: true" in compose


def test_compose_uses_shared_local_images_for_built_services():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert "image: asil-obs-mock:local" in compose
    assert "image: asil-eval:local" in compose


def test_eval_dockerfile_includes_writer_and_impress_runtime():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    assert "libreoffice-calc" in dockerfile
    assert "libreoffice-writer" in dockerfile
    assert "libreoffice-impress" in dockerfile


def test_eval_dockerfile_installs_playwright_browser_runtime():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    assert "playwright" in dockerfile
    assert "python -m playwright install chromium" in dockerfile


def test_eval_dockerfile_includes_kdenlive_runtime():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    assert "kdenlive" in dockerfile
    assert "mediainfo" in dockerfile
    assert "frei0r-plugins" in dockerfile


def test_eval_dockerfile_includes_gimp_runtime():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    assert "gimp" in dockerfile


def test_eval_dockerfile_includes_vlc_runtime():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    assert "vlc" in dockerfile


def test_eval_dockerfile_includes_celluloid_runtime():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    assert "celluloid" in dockerfile


def test_eval_dockerfile_includes_audacity_runtime():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    assert "audacity" in dockerfile
    assert "pulseaudio" in dockerfile


def test_eval_dockerfile_includes_thunderbird_runtime():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    assert "thunderbird" in dockerfile


def test_eval_dockerfile_includes_nautilus_runtime():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    assert "nautilus" in dockerfile


def test_eval_dockerfile_creates_gui_runtime_user():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    assert "useradd -m -s /bin/bash asilgui" in dockerfile


def test_eval_dependencies_include_socks_proxy_support():
    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")

    assert "httpx[socks]" in pyproject or "socksio" in pyproject


def test_eval_dockerfile_uses_pinned_constraints_and_checks_playwright_version():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    assert "COPY constraints-host.txt" in dockerfile
    assert "-c constraints-host.txt" in dockerfile
    assert 'playwright==1.58.0' in (
        Path(__file__).resolve().parent.parent / "constraints-host.txt"
    ).read_text(encoding="utf-8")
    assert 'version("playwright")' in dockerfile
    assert '1.58.0' in dockerfile


def test_eval_dockerfile_uses_official_playwright_download_then_fallback():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    install = "python -m playwright install chromium"
    assert dockerfile.count(install) >= 2
    assert "PLAYWRIGHT_DOWNLOAD_CONNECTION_TIMEOUT" in dockerfile
    assert "PLAYWRIGHT_FALLBACK_HOST" in dockerfile
    assert "timeout \"$PLAYWRIGHT_DOWNLOAD_TOTAL_TIMEOUT\"" in dockerfile
    assert "https://npmmirror.com/mirrors/chrome-for-testing" in dockerfile
    first_install = dockerfile.index(install)
    fallback = dockerfile.index("PLAYWRIGHT_CHROMIUM_DOWNLOAD_HOST")
    assert first_install < fallback


def test_eval_dockerfile_labels_checkout_provenance_and_launches_chromium():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.eval").read_text(
        encoding="utf-8"
    )

    for label in (
        "org.asil.git-commit",
        "org.asil.constraints-sha256",
        "org.asil.source-sha256",
    ):
        assert label in dockerfile
    assert "async_playwright" in dockerfile
    assert "browser.close" in dockerfile
    assert dockerfile.index("LABEL org.asil.git-commit") > dockerfile.index("async_playwright")


def test_obs_mock_dockerfile_pins_base_digest_and_websockets():
    dockerfile = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.obs-mock").read_text(
        encoding="utf-8"
    )

    assert re.search(r"^FROM python:3\.11-slim@sha256:[0-9a-f]{64}$", dockerfile, re.MULTILINE)
    assert "websockets==16.0" in dockerfile


def test_all_third_party_compose_services_use_immutable_digests():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert re.search(r"image: gitea/gitea@sha256:[0-9a-f]{64}", compose)
    assert re.search(r"image: curlimages/curl@sha256:[0-9a-f]{64}", compose)
    assert ":latest" not in compose
    assert "image: asil-eval:local" in compose
    assert "image: asil-obs-mock:local" in compose


def test_compose_passes_provenance_build_arguments_to_local_images():
    compose = (Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    for variable in (
        "ASIL_GIT_COMMIT",
        "ASIL_CONSTRAINTS_SHA256",
        "ASIL_SOURCE_SHA256",
    ):
        assert f"{variable}: ${{{variable}:-unknown}}" in compose
