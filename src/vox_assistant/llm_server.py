"""The assistant's own llama-server: a small model on one GPU, beside whatever Dual_gpu_setup has loaded there.

Why a separate server: the voice path needs short answers fast (Qwen3.5-4B: ~0.4 s to the first words, 24/24 on
scripts/agent_eval.py) while the Dual_gpu lanes hold 27B models for other work (~1.4 s, and they are paged out of VRAM
when idle). It is not a Dual_gpu lane, so that service neither manages nor evicts it.
"""

from __future__ import annotations

import subprocess
from urllib.parse import urlparse

from .config import Settings


def build_command(settings: Settings) -> list[str]:
    url = urlparse(settings.llm_base_url)
    if url.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError(f"llm_base_url {settings.llm_base_url} is not on this PC; nothing to start")
    cmd = [str(settings.llama_server), "--model", str(settings.llm_gguf), "--alias", "voice",
           "--device", settings.llm_gpu, "--gpu-layers", "999", "--ctx-size", str(settings.llm_ctx), "--parallel", "1",
           "--host", "127.0.0.1", "--port", str(url.port or 80), "--flash-attn", "auto", "--jinja",
           # Thinking off: a spoken reply cannot wait for a hidden monologue.
           "--reasoning-format", "deepseek", "--reasoning", "off", "--reasoning-budget", "0"]
    if settings.llm_mmproj:
        cmd += ["--mmproj", str(settings.llm_mmproj)]
    else:
        cmd += ["--cache-reuse", "256"]  # llama.cpp disables cache reuse for multimodal models anyway
    return cmd


def run(settings: Settings) -> int:
    for path in (settings.llama_server, settings.llm_gguf, settings.llm_mmproj):
        if path is not None and not path.is_file():
            print(f"Not found: {path}  (set llama_server / llm_gguf / llm_mmproj in data/config.toml)")
            return 1
    cmd = build_command(settings)
    print("Starting:", subprocess.list2cmdline(cmd), flush=True)
    return subprocess.call(cmd)
