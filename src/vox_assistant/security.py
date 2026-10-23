"""Request guard shared by the web app and the TTS service (both listen on 127.0.0.1 and can touch the user's data).

* Host must be a loopback name (blocks DNS rebinding: a hostile page resolving its own name to 127.0.0.1).
* Origin, when present, must be loopback (blocks other web pages from posting to us).
* POST must be application/json, including body-less posts: a browser can send a form-style POST cross-site without
  a preflight (a "simple request"), so it is rejected unless a JSON content type is declared. PATCH is held to the same
  rule for consistency.
* PUT and DELETE are never simple requests: a cross-site page must pass a CORS preflight, which this server never
  grants, so they need no content type (a DELETE has no body).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

LOOPBACK = {"127.0.0.1", "localhost", "[::1]"}


def _host_only(value: str) -> str:
    value = value.split("://", 1)[-1].split("/", 1)[0]
    return value.rsplit(":", 1)[0] if not value.endswith("]") and ":" in value else value


def install_local_only(app: FastAPI) -> None:
    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if _host_only(request.headers.get("host") or "") not in LOOPBACK:
            return JSONResponse({"detail": "Unexpected Host header."}, status_code=403)
        origin = request.headers.get("origin")
        if origin and _host_only(origin) not in LOOPBACK:
            return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)
        if request.method in ("POST", "PATCH") and not request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse({"detail": "State-changing requests require application/json."}, status_code=415)
        return await call_next(request)
