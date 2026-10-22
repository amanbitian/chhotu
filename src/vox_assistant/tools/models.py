"""LLM workload control through Dual_gpu_setup: what can be loaded, load a model on a GPU, unload a GPU.

Loading or unloading stops whatever that GPU was running (perhaps another application's model), so both are
`destructive`: always asked, never pre-approved. Dual_gpu_setup's own admission checks (VRAM fit, leases, running
requests drain for up to 30 s) still apply. The assistant refuses to unload the lane its own LLM runs on.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

from ..config import Settings
from ..permissions import Risk, Tool, fail, ok


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


_FILLER = {"the", "gpu", "card", "graphics", "radeon", "rx", "amd", "ai", "pro", "lane", "one", "my"}


def match_lane(query: str, lanes: list[dict]) -> dict | None:
    """'the R9700', 'AMD Radeon AI PRO R9700', '9070', 'RX 9070 XT' -> the lane."""
    q = "".join(w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _FILLER)
    hits = [l for l in lanes if q and (q in _norm(l.get("key", "")) or q in _norm(l.get("display", "")))]
    return hits[0] if len(hits) == 1 else None


def match_model(query: str, models: list[dict]) -> dict | list[dict] | None:
    words = [w for w in re.findall(r"[a-z0-9.]+", query.lower()) if w not in ("model", "the", "gguf")]
    if not words:
        return None
    hits = [m for m in models if m.get("exists", True) and
            all(w in (str(m.get("name", "")) + " " + str(m.get("display_name", ""))).lower() for w in words)]
    if len(hits) == 1:
        return hits[0]
    return hits[:8] or None


def make_tools(settings: Settings) -> list[Tool]:
    base = settings.dual_gpu_url

    async def catalog() -> dict:
        async with httpx.AsyncClient(base_url=base, timeout=30.0) as c:  # bootstrap also returns run history: can be slow
            return (await c.get("/api/bootstrap")).json()["catalog"]

    def own_lane(lanes: list[dict]) -> str | None:
        port = urlparse(settings.llm_base_url).port
        return next((l["key"] for l in lanes if l.get("port") == port), None)

    async def available_models(args: dict) -> dict:
        try:
            cat = await catalog()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return fail(f"The Dual GPU service is not reachable ({type(exc).__name__}).")
        query = str(args.get("query", "")).strip()
        models = [m for m in cat.get("models", []) if m.get("exists", True)]
        found = match_model(query, models) if query else models
        found = [found] if isinstance(found, dict) else (found or [])
        return ok(gpus=[{"lane": l["key"], "name": l.get("display"), "vram_gb": l.get("vram_gb")} for l in cat.get("lanes", [])],
                  models=[{"name": m.get("display_name") or m.get("name"), "size_gb": m.get("size_gb"), "pinned_to": m.get("pin_lane")}
                          for m in found[:12]], total_models=len(models))

    async def load_model(args: dict) -> dict:
        try:
            cat = await catalog()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return fail(f"The Dual GPU service is not reachable ({type(exc).__name__}).")
        lane = match_lane(str(args.get("gpu", "")), cat.get("lanes", []))
        if lane is None:
            return fail("Which GPU?", gpus=[l["key"] for l in cat.get("lanes", [])])
        model = match_model(str(args.get("model", "")), cat.get("models", []))
        if model is None:
            return fail("No model matches that name. Ask me which models are available.")
        if isinstance(model, list):
            return fail("Several models match; which one?", candidates=[m.get("display_name") or m.get("name") for m in model])
        if lane["key"] == own_lane(cat.get("lanes", [])):
            return fail("My own language model runs on that GPU's server, so I will not replace it.")
        try:
            async with httpx.AsyncClient(base_url=base, timeout=300.0) as c:
                r = await c.post("/api/clients/deploy_lane", json={"model": model["id"], "lane": lane["key"]})
        except httpx.HTTPError as exc:
            return fail(f"Loading did not finish ({type(exc).__name__}).")
        if r.status_code != 200:
            return fail(f"Dual GPU refused: {r.json().get('detail') or r.json().get('error') or r.status_code}")
        servers = [s for s in r.json().get("servers", []) if lane["key"] in (s.get("lane_keys") or [s.get("target")])]
        alive = any(s.get("alive") for s in servers)
        return ok(gpu=lane["key"], model=model.get("display_name") or model.get("name"), verified_running=alive)

    async def unload_model(args: dict) -> dict:
        try:
            cat = await catalog()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return fail(f"The Dual GPU service is not reachable ({type(exc).__name__}).")
        lane = match_lane(str(args.get("gpu", "")), cat.get("lanes", []))
        if lane is None:
            return fail("Which GPU?", gpus=[l["key"] for l in cat.get("lanes", [])])
        if lane["key"] == own_lane(cat.get("lanes", [])):
            return fail("My own language model runs on that GPU's server, so I will not unload it.")
        try:
            async with httpx.AsyncClient(base_url=base, timeout=90.0) as c:
                r = await c.post("/api/unload", json={"lane": lane["key"]})
        except httpx.HTTPError as exc:
            return fail(f"Unloading did not finish ({type(exc).__name__}).")
        if r.status_code != 200:
            return fail(f"Dual GPU refused: {r.json().get('detail') or r.json().get('error') or r.status_code}")
        return ok(gpu=lane["key"], released=bool(r.json().get("released")),
                  note=None if r.json().get("released") else "Nothing was loaded there.")

    gpu = {"type": "string", "description": "which GPU, e.g. r9700 or 9070"}
    return [
        Tool("available_models", "Which language models can be loaded (optionally filtered by name), and the GPUs.",
             {"type": "object", "properties": {"query": {"type": "string"}}}, Risk.READ, available_models),
        Tool("load_model", "Load a language model on a GPU through Dual GPU, replacing what that GPU runs.",
             {"type": "object", "properties": {"model": {"type": "string"}, "gpu": gpu}, "required": ["model", "gpu"]},
             Risk.DESTRUCTIVE, load_model, timeout_s=330.0,
             describe=lambda a: f"load {a.get('model')} on the {a.get('gpu')}, which stops whatever that GPU is running now"),
        Tool("unload_model", "Unload the language model from a GPU through Dual GPU, freeing its memory.",
             {"type": "object", "properties": {"gpu": gpu}, "required": ["gpu"]}, Risk.DESTRUCTIVE, unload_model,
             timeout_s=120.0, describe=lambda a: f"unload the model on the {a.get('gpu')}, stopping anything using it"),
    ]
