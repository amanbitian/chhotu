"""The P2 desktop features, the disclosure label, the launcher command, and config parsing. No hardware needed."""

import io
import json
import wave

import httpx
import pytest
import soundfile as sf

from vox_assistant.config import Settings
from vox_assistant.disclosure import label_text, label_wav, labelled_wav, read_label
from vox_assistant.hotkey import parse
from vox_assistant.llm_server import build_command
from vox_assistant.permissions import Risk, Tool, ToolRegistry, ok
from vox_assistant.tools import build_registry
from vox_assistant.tools import models as models_mod
from vox_assistant.tools.apps import _is_public_web_url, parse_search_results, summarize_page
from vox_assistant.tools.knowledge import search_text
from vox_assistant.tools.models import match_lane, match_model
from vox_assistant.tools.projects import agent_command, launch_line, match_project
from vox_assistant.tools.windows import match_window
from vox_assistant.tools.workflows import make_tools as workflow_tools, validate
from vox_assistant.tts_service import pick_device


def test_hotkey_parsing():
    assert parse("ctrl+alt+j") == (0x2 | 0x1, ord("J"))
    assert parse("Pause") == (0, 0x13) and parse("f13") == (0, 0x7C)
    for bad in ("", "j", "ctrl+", "hyper+j", "ctrl+alt+spacebar"):
        with pytest.raises(ValueError):
            parse(bad)


def test_web_results_and_prices_are_extracted_without_inventing_values():
    search = '''<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.amazon.in%2Fdp%2FABC">Sony headphones</a>
                <a class="result__snippet">Amazon listing for the Sony model.</a>'''
    assert parse_search_results(search) == [{"title": "Sony headphones", "url": "https://www.amazon.in/dp/ABC",
                                              "snippet": "Amazon listing for the Sony model."}]
    page = summarize_page("<title>Sony</title><p>Deal price ₹35,989.00</p><script>₹1</script>")
    assert page["title"] == "Sony" and page["currency_amounts"] == ["₹35,989.00"]
    assert not _is_public_web_url("http://127.0.0.1:8765/api/status")


def test_window_matching():
    ws = [{"title": "README.md - VoxVortex - Visual Studio Code", "app": "code"},
          {"title": "Untitled - Notepad", "app": "notepad"},
          {"title": "Inbox - Mozilla Firefox", "app": "firefox"},
          {"title": "Docs - Mozilla Firefox", "app": "firefox"}]
    assert match_window("notepad", ws)["app"] == "notepad"
    assert match_window("the VS Code window", ws, {"vs code": "code"})["app"] == "code"
    assert match_window("visual studio code", ws)["app"] == "code"
    assert match_window("firefox inbox", ws)["title"].startswith("Inbox")
    assert isinstance(match_window("firefox", ws), list)  # two windows: ask which
    assert match_window("excel", ws) is None


def test_project_matching_and_agent_command(tmp_path):
    for name in ("VoxVortex_pc_assistant", "VoxVortex", "Audio Clone", "Cutline"):
        (tmp_path / name).mkdir()
    projects = list(tmp_path.iterdir())
    assert match_project("the Cutline project", projects).name == "Cutline"
    assert match_project("Cutline repo", projects).name == "Cutline"
    assert match_project("vox vortex pc assistant", projects).name == "VoxVortex_pc_assistant"
    assert match_project("voxvortex", projects).name == "VoxVortex"  # exact wins over the longer name
    assert match_project("audio-clone", projects).name == "Audio Clone"
    assert match_project("nothing", projects) is None
    cmd = agent_command(["opencode", "--prompt", "{task}"], 'fix "it" & del *; rm -rf | evil > x')
    assert cmd[:2] == ["opencode", "--prompt"] and not any(c in cmd[2] for c in '"&;|>')
    line = launch_line(["opencode", "--prompt", "{task}"], tmp_path / "Audio Clone", r"C:\npm\opencode.CMD")
    assert line.startswith(f'cmd /c start "" /D "{tmp_path / "Audio Clone"}" cmd /v:on /k ""C:\\npm\\opencode.CMD"')
    assert line.endswith('"--prompt" "!VOXA_TASK!""') and "fix" not in line  # the task travels in an env var


def test_lane_and_model_matching():
    lanes = [{"key": "r9700", "display": "AMD Radeon AI PRO R9700", "port": 8081},
             {"key": "9070xt", "display": "AMD Radeon RX 9070 XT", "port": 8082}]
    assert match_lane("the R9700", lanes)["key"] == "r9700"
    assert match_lane("RX 9070 XT", lanes)["key"] == "9070xt" and match_lane("9070", lanes)["key"] == "9070xt"
    assert match_lane("the gpu", lanes) is None
    cat = [{"id": "a", "name": "Qwen3.5-4B-Q4_K_M", "exists": True}, {"id": "b", "name": "Qwen3.5-9B-Q4_K_M", "exists": True},
           {"id": "c", "name": "gemma-4-12B-it-Q4_K_M", "exists": True}]
    assert match_model("gemma 12b", cat)["id"] == "c"
    assert len(match_model("qwen3.5", cat)) == 2 and match_model("llama", cat) is None


def _dual_gpu(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(models_mod.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


CATALOG = {"catalog": {"lanes": [{"key": "r9700", "display": "R9700", "port": 8081}, {"key": "9070xt", "display": "RX 9070 XT", "port": 8082}],
                       "models": [{"id": "lm:gemma", "name": "gemma-4-12B-it-Q4_K_M", "exists": True, "size_gb": 6.9}]}}


async def test_load_and_unload_go_through_dual_gpu_and_protect_the_assistants_own_lane(settings, monkeypatch):
    sent = []

    def handler(req: httpx.Request):
        if req.url.path == "/api/bootstrap":
            return httpx.Response(200, json=CATALOG)
        sent.append((req.url.path, json.loads(req.content)))
        if req.url.path == "/api/clients/deploy_lane":
            return httpx.Response(200, json={"servers": [{"lane_keys": ["9070xt"], "alive": True}]})
        return httpx.Response(200, json={"released": True})

    _dual_gpu(monkeypatch, handler)
    tools = {t.name: t for t in models_mod.make_tools(settings)}
    assert tools["load_model"].risk is Risk.DESTRUCTIVE and tools["unload_model"].risk is Risk.DESTRUCTIVE
    r = await tools["load_model"].handler({"model": "gemma 12b", "gpu": "9070"})
    assert r["ok"] and r["verified_running"] and sent[-1] == ("/api/clients/deploy_lane", {"model": "lm:gemma", "lane": "9070xt"})
    r = await tools["unload_model"].handler({"gpu": "the 9070 XT"})
    assert r["released"] and sent[-1] == ("/api/unload", {"lane": "9070xt"})  # never the bare form that stops everything

    settings.llm_base_url = "http://127.0.0.1:8081/v1"  # the assistant itself on the R9700 lane
    n = len(sent)
    assert not (await tools["unload_model"].handler({"gpu": "r9700"}))["ok"]
    assert not (await tools["load_model"].handler({"model": "gemma", "gpu": "r9700"}))["ok"]
    assert len(sent) == n


def test_notes_search_ranks_by_words_and_skips_secrets(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "gpu.md").write_text("The R9700 thermal limit is 90 C. Thermal paste was replaced in May.", encoding="utf-8")
    (tmp_path / "notes" / "other.md").write_text("Shopping list: milk, thermal socks.", encoding="utf-8")
    (tmp_path / "notes" / "secret_tokens.md").write_text("thermal limit password hunter2", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.md").write_text("thermal limit", encoding="utf-8")
    hits = search_text([tmp_path], "R9700 thermal limit")
    assert hits[0]["path"].endswith("gpu.md") and "90 C" in hits[0]["snippet"]
    assert not any("secret" in h["path"] or "node_modules" in h["path"] for h in hits)


async def test_workflows_run_every_step_with_one_approval_and_refuse_destructive_steps(settings):
    ran = []

    async def step(args):
        ran.append(args)
        return ok(n=len(ran))

    reg = ToolRegistry([Tool("peek", "", {}, Risk.READ, step), Tool("poke", "", {}, Risk.ACT, step, describe=lambda a: "poke it"),
                        Tool("nuke", "", {}, Risk.DESTRUCTIVE, step)])
    settings.workflows = {"morning": {"description": "x", "steps": [{"tool": "peek"}, {"tool": "poke", "args": {"v": 1}}]},
                          "bad": {"steps": [{"tool": "nuke"}]}, "typo": {"steps": [{"tool": "peak"}]}}
    (wf,) = workflow_tools(settings, reg)
    assert wf.risk is Risk.ACT and "then poke it" in wf.intent({"name": "morning"})
    r = await wf.handler({"name": "Morning"})
    assert r["all_ok"] and ran == [{}, {"v": 1}]
    assert not (await wf.handler({"name": "bad"}))["ok"] and not (await wf.handler({"name": "typo"}))["ok"]
    assert len(ran) == 2
    assert len(validate(settings.workflows, reg)) == 2


def test_default_registry_has_every_feature_and_valid_workflows(settings):
    reg, store = build_registry(settings)
    try:
        for name in ("gpu_status", "list_models", "available_models", "load_model", "unload_model", "system_info",
                     "find_files", "read_text_file", "search_notes", "open_app", "open_path", "open_url", "web_search",
                     "read_search_result",
                     "list_projects", "open_project", "delegate_coding_task", "list_windows", "window_action",
                     "close_window", "look_at_screen", "read_clipboard", "set_clipboard", "set_reminder",
                     "list_reminders", "cancel_reminder", "run_workflow"):
            assert reg.get(name) is not None, name
        assert validate(settings.workflows, reg) == []
        assert reg.get("close_window").risk is Risk.DESTRUCTIVE and reg.get("look_at_screen").risk is Risk.READ
    finally:
        store.close()


def test_wav_disclosure_label_round_trips_and_stays_playable():
    pcm = b"\x10\x00" * 4800
    wav = labelled_wav(pcm, 24000, "me_v1")
    assert read_label(wav) == label_text("me_v1") and "AI-generated" in read_label(wav)
    with wave.open(io.BytesIO(wav)) as w:  # stdlib reader still works
        assert w.getframerate() == 24000 and w.readframes(10_000) == pcm
    audio, rate = sf.read(io.BytesIO(wav), dtype="int16")  # and libsndfile
    assert rate == 24000 and len(audio) == 4800
    with pytest.raises(ValueError):
        label_wav(b"nope", "x")


def test_llm_server_command(settings):
    cmd = build_command(settings)
    assert cmd[cmd.index("--port") + 1] == "8083" and "--mmproj" in cmd and cmd[cmd.index("--reasoning") + 1] == "off"
    settings.llm_mmproj = None
    assert "--mmproj" not in build_command(settings) and "--cache-reuse" in build_command(settings)
    settings.llm_base_url = "http://192.168.1.5:8083/v1"
    with pytest.raises(ValueError):
        build_command(settings)


def test_tts_device_by_name():
    names = ["AMD Radeon AI PRO R9700", "AMD Radeon RX 9070 XT", "AMD Radeon(TM) Graphics"]
    assert pick_device("9070", names) == "cuda:1" and pick_device("r9700", names) == "cuda:0"
    assert pick_device("cuda:0", []) == "cuda:0" and pick_device("cpu", []) == "cpu"
    with pytest.raises(RuntimeError):
        pick_device("4090", names)


def test_config_parses_booleans_and_optional_paths_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXA_GUEST_MODE", "false")
    monkeypatch.setenv("VOXA_LLM_MMPROJ", "")
    s = Settings.load(tmp_path)
    assert s.guest_mode is False and s.llm_mmproj is None
    monkeypatch.setenv("VOXA_GUEST_MODE", "1")
    assert Settings.load(tmp_path).guest_mode is True


async def test_coding_task_approval_names_auto_approved_changes(settings, tmp_path):
    from vox_assistant.tools.projects import make_tools as project_tools

    (tmp_path / "proj" / "Cutline").mkdir(parents=True)
    settings.search_roots = [str(tmp_path / "proj")]
    started = []

    class Tasks:
        def start(self, folder, task, *, allow_changes):
            started.append(allow_changes)
            return {"id": "t1", "status": "running"}

    tool = {t.name: t for t in project_tools(settings, Tasks())}["delegate_coding_task"]
    plain = {"project": "cutline", "task": "fix the tests"}
    assert "without asking" not in tool.intent(plain)
    assert "change files without asking you" in tool.intent({**plain, "allow_changes": True})
    for value, expect in ((True, True), ("true", True), ("false", False), ("no", False), (1, False), (None, False)):
        assert (await tool.handler({**plain, "allow_changes": value}))["ok"]
        assert started[-1] is expect, value
        assert ("without asking" in tool.intent({**plain, "allow_changes": value})) is expect
