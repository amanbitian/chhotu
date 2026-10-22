"""Settings: defaults, overridden by `config.toml` in the data directory, then by VOXA_* environment variables."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    data_dir: Path = PROJECT_ROOT / "data"
    # LLM: the assistant's own small model (`vox llm-server`), next to whatever Dual_gpu_setup runs on the same GPU.
    # A 27B lane model also works (http://127.0.0.1:8081/v1) but is ~5x slower to answer; see scripts/agent_eval.py.
    llm_base_url: str = "http://127.0.0.1:8083/v1"
    llm_model: str = "default"
    llama_server: Path = Path("F:/llama.cpp/b11699-rocm/llama-server.exe")
    llm_gguf: Path = Path("G:/LLM models/lmstudio-community/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf")
    llm_mmproj: Path | None = Path("G:/LLM models/lmstudio-community/Qwen3.5-4B-GGUF/mmproj-Qwen3.5-4B-BF16.gguf")  # vision: look_at_screen
    llm_gpu: str = "ROCm0"  # llama.cpp device name; ROCm0 is the R9700
    llm_ctx: int = 32768
    dual_gpu_url: str = "http://127.0.0.1:8090"
    # TTS: tts_service.py, run inside the Audio Clone virtualenv (it owns the ROCm PyTorch + VoxCPM2 install).
    tts_url: str = "http://127.0.0.1:8870"
    audio_clone_root: Path = Path("F:/Projects/Audio Clone")
    tts_device: str = "9070"  # "auto", "cpu", "cuda:N", or a fragment of the GPU name; the brief puts speech on the 9070 XT
    tts_preset: str = "Draft"
    # "driver": use the display driver's amdhip64_7.dll (the wheel's copy crashes with current drivers); "bundled": the wheel's.
    tts_hip_runtime: str = "driver"
    # ASR: faster-whisper on the CPU by default, so the GPUs stay free for the LLM and the voice.
    whisper_model: str = "small.en"
    whisper_device: str = "cpu"
    whisper_compute: str = "int8"
    # Measured on this 32-thread CPU, small.en per request: default threads 998 ms, 8 threads 818 ms, 16 threads 859 ms.
    # (base.en is 3x faster but its word error rate was 13.5% against 4.7%: too many wrong commands.)
    whisper_threads: int = 8
    silero_model: Path = Path("F:/Projects/VoxVortex/models/silero_vad.onnx")
    input_device: str | int | None = None
    output_device: str | int | None = None
    gpu_temp_limit_c: int = 90
    wake_phrase: str = "hey vox"
    # How Whisper tends to spell the wake phrase; matched exactly like wake_phrase.
    wake_aliases: list[str] = field(default_factory=lambda: ["hey box", "hey vaux", "hey vocs", "hey vogs"])
    # A small Whisper run on the partial utterance while you are still speaking, so the wake chime comes quickly.
    wake_spotter_model: str = "tiny.en"  # "" disables spotting (the wake phrase is then found after the utterance ends)
    # Push-to-talk: wakes the assistant like the wake phrase; "" disables. Ctrl+Alt+Space is taken on this PC,
    # Ctrl+Shift+Space is VS Code's and Ctrl+Alt+V is Office's Paste Special.
    hotkey: str = "ctrl+alt+j"
    # While listening, touch the LLM and the voice model this often (seconds; 0 = only when woken). Measured: after
    # >= 30 s idle the GPUs go cold and the first token / first audio cost 1.5-2.6 s more; within 10 s they do not.
    keep_warm_s: int = 15
    # Said (pre-synthesized, so instantly) when a read-only tool starts before any words: the answer itself needs the
    # tool result first. [] disables.
    ack_phrases: list[str] = field(default_factory=lambda: ["Let me check.", "One moment.", "Checking."])
    assistant_name: str = "Vox"
    web_port: int = 8765
    # Guest mode: when other people can hear, speak in the neutral voice and say that this is an AI assistant.
    guest_mode: bool = False
    # Desktop tools
    search_roots: list[str] = field(default_factory=lambda: ["F:/Projects"])
    knowledge_roots: list[str] = field(default_factory=lambda: ["F:/Projects"])  # search_notes reads text files here
    apps: dict[str, str] = field(default_factory=lambda: {
        "vs code": "code", "firefox": "firefox", "chrome": "chrome", "notepad": "notepad",
        "calculator": "calc", "file explorer": "explorer", "terminal": "wt", "task manager": "taskmgr",
    })
    # delegate_coding_task runs this in a new terminal in the project folder; {task} is replaced by the request.
    # OpenCode's interactive UI in a visible window (not --auto, which approves everything without asking).
    coding_agent: list[str] = field(default_factory=lambda: ["opencode", "--prompt", "{task}"])
    web_search_url: str = "https://duckduckgo.com/?q={query}"
    # Named sequences of tool calls, e.g. [workflows.start_work] steps = [{tool = "open_app", args = {name = "vs code"}}]
    workflows: dict[str, dict] = field(default_factory=lambda: {
        "status_check": {"description": "GPU temperatures and models, system load, and pending reminders",
                         "steps": [{"tool": "gpu_status"}, {"tool": "system_info"}, {"tool": "list_reminders"}]},
    })

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def audit_db(self) -> Path:
        return self.data_dir / "audit.sqlite3"

    @classmethod
    def load(cls, data_dir: Path | None = None) -> "Settings":
        s = cls()
        if data_dir is not None:
            s.data_dir = Path(data_dir)
        elif os.getenv("VOXA_DATA_DIR"):
            s.data_dir = Path(os.environ["VOXA_DATA_DIR"])
        cfg = s.data_dir / "config.toml"
        if cfg.is_file():
            s._apply(tomllib.loads(cfg.read_text(encoding="utf-8")))
        s._apply({f.name: os.environ[f"VOXA_{f.name.upper()}"] for f in fields(cls) if f"VOXA_{f.name.upper()}" in os.environ})
        s.data_dir.mkdir(parents=True, exist_ok=True)
        return s

    def _apply(self, values: dict) -> None:
        known = {f.name: f for f in fields(self)}
        for key, value in values.items():
            if key not in known:
                raise ValueError(f"Unknown setting {key!r}")
            current = getattr(self, key)
            if isinstance(current, Path) or (current is None and "Path" in str(known[key].type)):
                value = Path(value) if value else None
            elif isinstance(current, bool) and isinstance(value, str):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int) and not isinstance(current, bool):
                value = int(value)
            elif isinstance(current, list) and isinstance(value, str):
                value = [v for v in value.split(";") if v]
            setattr(self, key, value)
