"""Read-only machine state: GPUs and loaded models (via Dual_gpu_setup), CPU/RAM/disk (psutil)."""

from __future__ import annotations

import httpx
import psutil

from ..config import Settings
from ..permissions import Risk, Tool, fail, ok

GB = 1024**3


def summarize_gpus(hardware: dict, status: dict, temp_limit_c: float) -> list[dict]:
    """Join telemetry (per card) with the model each lane is serving. Pure, so it is unit-testable."""
    models = {}
    for s in status.get("servers") or []:
        for lane in s.get("lane_keys") or [s.get("target")]:
            models[lane] = {"model": s.get("model"), "alive": s.get("alive"), "context_window": s.get("context_window")}
    out = []
    for gpu in (hardware.get("hardware") or {}).get("gpus") or []:
        if not gpu.get("lane"):
            continue  # e.g. the CPU's integrated GPU: not one of the managed LLM lanes
        now = gpu.get("latest") or {}
        junction = now.get("temp_junction_c")
        used, total = now.get("vram_used_bytes"), gpu.get("vram_total_bytes") or now.get("vram_total_bytes")
        served = models.get(gpu.get("lane"), {})
        out.append({
            "gpu": gpu.get("name"), "lane": gpu.get("lane"),
            "model": served.get("model"), "model_alive": served.get("alive"),
            "busy_pct": now.get("gpu_busy_pct"), "power_w": now.get("power_w"),
            "temp_junction_c": junction, "temp_edge_c": now.get("temp_edge_c"), "temp_memory_c": now.get("temp_memory_c"),
            "vram_used_gb": round(used / GB, 1) if used is not None else None,
            "vram_total_gb": round(total / GB, 1) if total else None,
            "within_temp_limit": None if junction is None else junction <= temp_limit_c,
            "temp_limit_c": temp_limit_c,
        })
    return out


def make_tools(settings: Settings) -> list[Tool]:
    async def gpu_status(args: dict) -> dict:
        try:
            async with httpx.AsyncClient(base_url=settings.dual_gpu_url, timeout=5.0) as c:
                hw = (await c.get("/api/hardware")).json()
                st = (await c.get("/api/status")).json()
        except httpx.HTTPError as exc:
            return fail(f"The Dual GPU service at {settings.dual_gpu_url} is not reachable ({type(exc).__name__}).")
        gpus = summarize_gpus(hw, st, settings.gpu_temp_limit_c)
        if not gpus:
            reason = (hw.get("hardware") or {}).get("unavailable_reason") or "no GPU telemetry returned"
            return fail(f"No GPU telemetry available: {reason}")
        return ok(gpus=gpus, service_state=st.get("state"))

    async def list_models(args: dict) -> dict:
        try:
            async with httpx.AsyncClient(base_url=settings.dual_gpu_url, timeout=5.0) as c:
                st = (await c.get("/api/status")).json()
        except httpx.HTTPError as exc:
            return fail(f"The Dual GPU service is not reachable ({type(exc).__name__}).")
        servers = [{"lane": s.get("target"), "model": s.get("model"), "running": s.get("alive"),
                    "context_window": s.get("context_window")} for s in st.get("servers") or []]
        return ok(state=st.get("state"), loaded=servers)

    async def system_info(args: dict) -> dict:
        mem, disks = psutil.virtual_memory(), {}
        for part in psutil.disk_partitions(all=False):
            try:
                u = psutil.disk_usage(part.mountpoint)
            except OSError:
                continue
            disks[part.mountpoint] = {"free_gb": round(u.free / GB, 1), "total_gb": round(u.total / GB, 1)}
        return ok(cpu_percent=psutil.cpu_percent(interval=0.3), ram_used_gb=round(mem.used / GB, 1),
                  ram_total_gb=round(mem.total / GB, 1), disks=disks)

    empty = {"type": "object", "properties": {}}
    return [
        Tool("gpu_status", "Live status of both GPUs: which model each is serving, temperature, power draw, VRAM use, "
             "and whether temperature is within the configured limit. Use this instead of guessing.", empty, Risk.READ, gpu_status),
        Tool("list_models", "Which language models are currently loaded and on which GPU lane.", empty, Risk.READ, list_models),
        Tool("system_info", "CPU load, RAM use and free disk space.", empty, Risk.READ, system_info),
    ]
