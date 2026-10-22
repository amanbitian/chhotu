# VoxVortex Desktop

A local voice assistant for this PC. It listens, thinks with a local Qwen, acts on the desktop **only with your permission**, and answers in a clone of your own voice. Nothing leaves the machine.

| Document | What it is |
| --- | --- |
| **README.md** (this file) | What it does, how to run and check it, what is and is not verified |
| [architecture.md](architecture.md) | Detailed design: processes and GPU placement, turn pipeline, tools and permissions, voice loop, TTS service, security, performance |
| [sdd.md](sdd.md) | Software design document: requirements and status, design decisions, verification, traceability, defects found, risks, coverage of the original brief |

It is a thin layer over projects that already exist here, not a second speech stack:

| Need | Provided by |
| --- | --- |
| VAD, sentence segmenter, LLM streaming client, ASR/TTS interfaces | `F:\Projects\VoxVortex` (library, not modified) |
| The assistant's brain | Qwen3.5-4B (+ vision) from `G:\LLM models`, served by `F:\llama.cpp` on the R9700 (`vox llm-server`) |
| Voice cloning | VoxCPM2 in the `F:\Projects\Audio Clone` environment, on the RX 9070 XT, wrapped by `tts_service.py` |
| GPU telemetry and model management | `F:\Projects\Dual_gpu_setup` (your 27B lanes keep running untouched) |

## Status

Everything works end to end on this machine, **including the cloned voice** (fixed: see [what was wrong](#what-was-wrong-with-the-cloned-voice)). 116 automated tests pass; the real-component smoke test reports **29 PASS, 0 FAIL, 1 WARN**; the clone check **12/12**; the live desktop check **17/17**; the agent eval **47/48** with no false claims.

What only you can do: **enroll your own voice** (`vox enroll`) and try `vox listen` with a headset. Until you enroll, the active voice is `demo_v1`, Audio Clone's synthetic demo voice.

## Quick start

```powershell
cd F:\Projects\VoxVortex_pc_assistant
.\scripts\start_all.ps1            # voice LLM + voice service in their own windows, then the web app (http://127.0.0.1:8765)
.\scripts\start_all.ps1 listen     # same, then hands-free mode
.\.venv\Scripts\vox.exe doctor     # what is reachable, and environment problems
```

1. **Enroll your voice** (microphone, quiet room, headset): `vox enroll --name me`. Read 4–8 prompted lines; each take is checked for clipping, noise and length; the good ones become a 20–36 s reference with an exact transcript. Result: `me_v1` (enrolling again makes `me_v2`; `vox voices use me_v1` reverts).
2. **Score it**: `vox voices check me_v1` clones a test sentence and compares it with your recording (speaker-verification score; 0.86 or more = same speaker). Then listen: `vox say "Testing my cloned voice."`.
3. **Talk to it**: `vox listen`, then say "hey vox, …" (a chime means it heard you, often before you finish), or press **Ctrl+Alt+J** and just talk. `--always` skips the wake phrase.
4. **Type to it**: the web app, or `vox ask "..."`.

`vox` is `.\.venv\Scripts\vox.exe`, or `python -m vox_assistant.cli`.

## What it can do

| Tool | Risk | Notes |
| --- | --- | --- |
| `gpu_status`, `list_models`, `system_info` | read | Real telemetry; junction temperature vs `gpu_temp_limit_c` (90 C) |
| `available_models` | read | The Dual_gpu catalog (89 models here) |
| `load_model`, `unload_model` | **destructive** | Through Dual_gpu, one GPU at a time; always asks; refuses the GPU its own brain uses |
| `find_files`, `read_text_file` | read | Inside `search_roots`; secret-looking names hidden |
| `search_notes` | read | Full-text search of your .md/.txt notes under `knowledge_roots` (0.1–0.2 s once indexed) |
| `look_at_screen` | read | Screenshot described by the local vision model; never stored |
| `list_windows` | read | Open windows |
| `window_action` | act | Focus, minimize, maximize, restore; checked afterwards |
| `close_window` | **destructive** | Like clicking ×; never forced |
| `open_app`, `open_path` | act | Allowlisted apps; files/folders but never programs or scripts |
| `open_url`, `web_search` | act | Opens your browser |
| `list_projects`, `open_project` | read / act | "Switch to the Cutline project" opens it in VS Code |
| `delegate_coding_task` | act | Starts OpenCode in a new terminal in that project, with your request |
| `read_clipboard`, `set_clipboard` | read / act | Written values are read back |
| `set_reminder`, `cancel_reminder`, `list_reminders` | act / read | Spoken when due |
| `run_workflow` | act | Your named sequences from `config.toml`; one approval that names every step (default: `status_check`) |

**Permissions.** `read` runs freely. `act` asks first ("I'm about to open Notepad. Say yes or no.", or an Approve button in the web page). `destructive` always asks and can never be pre-approved. A "no" anywhere in the answer wins; silence is a refusal. The sound of a voice is never treated as authorization. Every request, decision and result is in `data/audit.sqlite3`.

**It will not claim what it did not do.** A reply that announces an action no tool performed is discarded before it is spoken, and the model is made to call the tool ([architecture §6.4](architecture.md#64-false-claim-guard)).

**Privacy and disclosure, handled in code (the LLM never sees these):** "mute" / "stop listening"; "hey vox unmute"; "use a neutral voice" / "use my voice"; **"guest mode"** (neutral voice, and it says "I'm an AI assistant, and this is a computer voice") / "guest mode off". Every WAV the project writes is labelled as AI-generated speech in its metadata. `vox voices revoke <id>` deletes a recording and clears the service's cached voice data.

**Local web security.** Both HTTP services bind to 127.0.0.1 and reject foreign `Host`/`Origin` headers and non-JSON POSTs, so a web page you visit cannot drive them.

## How fast

Measured on this PC, from the moment you stop talking, for a question that needs two tools ("which models are running and are the temperatures safe?"), answered in the cloned voice:

| | |
| --- | --- |
| First sound ("Let me check", in your voice) | **1.3–1.6 s** |
| First audio of the actual answer | **2.9–3.6 s** (the first build took 6.7–18 s) |
| A question without tools ("what's the capital of France?") | not timed end to end; the model's first words come 0.1–0.15 s after the transcript (agent eval), then one short clause is synthesized |
| Wake chime | while you are still speaking; hotkey < 1 ms |

The brief's 1–2 s target is met by the acknowledgement, not yet by the answer itself: VoxCPM2 must finish a whole clause before it can play it. [architecture §16](architecture.md#16-performance) lists each optimisation and its measured effect.

## Configuration

Defaults live in `src/vox_assistant/config.py`; override in `data/config.toml` or with `VOXA_<NAME>` environment variables:

```toml
input_device = "Headset"          # name fragment or index; see `vox devices`
hotkey = "ctrl+alt+j"             # "" to disable
guest_mode = false                # true: neutral voice and an AI notice from the first reply
keep_warm_s = 15                  # 0: let the GPUs idle down (the first reply after a pause is ~2 s slower)
search_roots = ["F:/Projects"]
knowledge_roots = ["F:/Projects"]
coding_agent = ["opencode", "--prompt", "{task}"]

[workflows.start_work]
description = "VS Code and the Dual GPU dashboard"
steps = [{tool = "open_app", args = {name = "vs code"}}, {tool = "open_url", args = {url = "http://127.0.0.1:8090"}}]
```

The voice LLM: `llm_gguf`, `llm_mmproj`, `llm_gpu` (`ROCm0` = R9700), `llm_base_url` (`:8083`). To use a 27B lane instead, set `llm_base_url = "http://127.0.0.1:8081/v1"` (slower; see `scripts/agent_eval.py`). The voice model's GPU: `tts_device = "9070"` (a name fragment, `cuda:N`, or `cpu`).

## Checks

| Command | What it proves |
| --- | --- |
| `.venv\Scripts\python.exe -m pytest` | 116 unit tests, no hardware needed |
| `.venv\Scripts\python.exe scripts\smoke_test.py` | Real LLM, telemetry, mic and speaker, VAD, Whisper, the voice loop in real time in the cloned voice (with a latency timeline), permission by voice, mute, web app, clone round trip. Silent: a Windows voice plays the person and speaker output is captured. `--quick` for the voice-loop part only |
| `scripts\clone_check.py` | Enroll (synthetic speaker through the real recording path) → clone → intelligibility → similarity → revoke |
| `scripts\desktop_check.py [--with-agent]` | Every desktop tool live. Visible: opens and closes Notepad, opens one browser tab, brings VS Code forward; restores your clipboard |
| `scripts\agent_eval.py [--url …]` | Tool choice, permission and false-claim behaviour of a model, and its speed |

## Verified, and not

**Verified on this machine (2026-10-22):** all of the above. Highlights: clone of a synthetic speaker scores 0.978 against its own reference and 0.474 against a different one; first-time synthesis of a new voice 3.0 s (RTF 0.52); Notepad opened, minimized, maximized, restored, focused and closed; the coding agent started in the right folder; screen descriptions from the local vision model; web page checked in headless Edge at desktop and 375 px phone width.

**Not verified:**

- **Your voice**: not enrolled yet; similarity and a blind listening test need you.
- **A person at the microphone**: `vox enroll` and `vox listen` ran only with synthetic speech fed through the real code paths. Echo and barge-in on real speakers are untested (use a headset).
- **Live `load_model` / `unload_model`**: tested against a mock Dual_gpu and the live catalog, not run against your lanes (it would replace their configuration).
- Hindi/Hinglish; Android companion (not built); clicking inside web pages (not built).

### What was wrong with the cloned voice

Not the iGPU. The ROCm PyTorch wheel in Audio Clone bundles HIP runtime 10.0.3581, which crashes with your Adrenalin 32.0.31036 driver (runtime 10.0.3679). The voice service now loads the driver's runtime first and turns TorchScript off (its kernel compiler had the same mismatch). Audio Clone and the driver are unchanged. Two further fixes made it fast: `MIOPEN_FIND_MODE=FAST` (each new sentence length used to cost 10–25 s the first time) and running on the 9070 XT (the R9700 cannot hold your 27B, the 4B and VoxCPM2 together; Windows was paging models out).

### Things on this PC you may want to fix

- **Your PATH has a stray `"`** at the end of `C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit"`. cmd.exe cannot see any PATH entry after it (Node.js, npm, so `opencode`, `npx`, `node` fail from cmd). The assistant works around it; `vox doctor` reports it.
- **Ctrl+Alt+Space is taken by another program**, which is why the hotkey is Ctrl+Alt+J.
- The 9070 XT is shared between the voice model and your Dual_gpu 27B Q3 lane; when both are busy Windows pages one of them.

## Layout

```
src/vox_assistant/  config voices enroll tts_service voice_similarity disclosure tts_client tts_neutral asr_local
                    audio_io hotkey llm_client llm_server permissions audit agent voice webapp security cli
                    tools/{system,models,files,knowledge,apps,projects,windows,screen,clipboard,reminders,workflows}
web/index.html      local web UI
scripts/            start.ps1 start_all.ps1 start_llm.ps1 start_tts.ps1
                    smoke_test.py clone_check.py desktop_check.py agent_eval.py
tests/              116 tests, no hardware      data/   (git-ignored) voices, audit log, reminders, config.toml
architecture.md  sdd.md
```

## Next

1. Enroll your voice, run `vox voices check`, and use `vox listen` for a day; note what feels wrong.
2. Compare Qwen3-TTS and Chatterbox behind the same `/audio/speech` contract: a TTS that streams within a sentence is what would bring the answer itself under 2 s.
3. In-page browser automation, an Android companion, Ubuntu support.
