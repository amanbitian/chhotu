# VoxVortex Desktop: Architecture

How the system is put together, why, and where each behaviour lives. For requirements and their verification see [sdd.md](sdd.md); for running it see [README.md](README.md).

Status date: 2026-10-22 (second build). Windows 11, Python 3.12, AMD R9700 (32 GB) + RX 9070 XT (16 GB) + Ryzen 9 9950X3D (32 threads).

## Contents

1. [Goals and boundaries](#1-goals-and-boundaries)
2. [System context](#2-system-context)
3. [Processes and GPU placement](#3-processes-and-gpu-placement)
4. [Module map](#4-module-map)
5. [The turn pipeline](#5-the-turn-pipeline)
6. [Agent, tools and permissions](#6-agent-tools-and-permissions)
7. [Voice loop internals](#7-voice-loop-internals)
8. [Voice profiles and the TTS service](#8-voice-profiles-and-the-tts-service)
9. [Web application](#9-web-application)
10. [Data and storage](#10-data-and-storage)
11. [Interfaces](#11-interfaces)
12. [Configuration](#12-configuration)
13. [Security and privacy model](#13-security-and-privacy-model)
14. [Failure handling](#14-failure-handling)
15. [Concurrency model](#15-concurrency-model)
16. [Performance](#16-performance)
17. [Extending the system](#17-extending-the-system)
18. [Known limits](#18-known-limits)

---

## 1. Goals and boundaries

A local desktop agent that **listens, reasons, acts on the PC with permission, and answers in the user's cloned voice**, with no cloud AI calls.

Design rules, in priority order:

1. **Never claim what did not happen.** Consequential facts come from tools, results are verified from real state, and a reply that announces an action no tool performed is discarded (section 6.4).
2. **Nothing changes the machine without a decision.** Every tool has a risk level; the gate, not the model, decides what needs approval.
3. **Reuse, do not rebuild.** VoxVortex supplies VAD, segmentation and the LLM client; Dual_gpu_setup supplies telemetry and model management; Audio Clone supplies the voice model; llama.cpp serves the voice LLM. This project is the layer that joins them.
4. **Degrade, do not stop.** If the cloned voice is unavailable the neutral Windows voice speaks; if the LLM errors the user hears an apology, not silence.
5. **Local and inspectable.** Voice data are plain files, actions are in an SQLite audit log, services bind to loopback.
6. **Measure, then optimise.** Every latency choice in section 16 comes from a measurement on this machine.

Out of scope for this release: Ubuntu/Wayland control, clicking and typing inside web pages (browser automation stops at opening pages and searches), Android companion, fine-tuned voices.

## 2. System context

```
                      ┌──────────────────────────── this project (vox_assistant) ─────────────────────────────┐
 microphone ─▶ Microphone ─▶ VoiceLoop ──▶ Agent ──▶ ToolRegistry ──▶ OS: apps, windows, files, clipboard,      │
                (VAD, spotter,   │  ▲        │  ▲        │              VS Code, coding agent, browser          │
                 hotkey)         │  │        │  │        │ PermissionGate ─▶ user (voice yes/no | button)       │
 speakers ◀── Speaker ◀──────────┘  │        ▼  │        └▶ AuditLog (SQLite)                                   │
                                    │   ToolChoiceLLM ─────▶ llama-server :8083  Qwen3.5-4B + vision (R9700)      │
                     ASR: faster-whisper small.en (CPU) ;    look_at_screen ─▶ same server, image input            │
                     spotter: tiny.en (CPU)                  gpu_status / load_model ─▶ Dual_gpu_setup :8090       │
                     ClonedVoiceTTS ──HTTP──▶ tts_service :8870 ──▶ VoxCPM2 on ROCm, RX 9070 XT (Audio Clone venv) │
                     NeutralVoiceTTS ──▶ Windows speech (System.Speech via PowerShell)                             │
 browser ◀──HTTP──▶ webapp :8765 ──▶ same Agent, VoiceProfiles, AuditLog                                          │
                      └────────────────────────────────────────────────────────────────────────────────────────┘
```

| Neighbour | Contract used | Owner |
| --- | --- | --- |
| `F:\Projects\VoxVortex` | Python import: `voxvortex.audio` (SileroVAD, Reframer), `voxvortex.tts` (SentenceSegmenter, normalize_for_speech), `voxvortex.llm` (OpenAICompatLLM, TextDelta, ToolCall), `voxvortex.asr` (is_hallucination) | editable install, not modified |
| `F:\Projects\Dual_gpu_setup` | `GET /api/hardware`, `/api/status`, `/api/bootstrap` (catalog); `POST /api/clients/deploy_lane`, `POST /api/unload {lane}` | not modified |
| `F:\Projects\Audio Clone` | its `.venv` interpreter, `voice_studio.runtime.configure_runtime`, `voice_studio.engine.VoxBackend` (`build_prompt`, `generate`) | not modified |
| `F:\llama.cpp\b11699-rocm` | `llama-server.exe` for the voice LLM | not modified |

## 3. Processes and GPU placement

| Process | Interpreter | Started by | Port | GPU | Owns |
| --- | --- | --- | --- | --- | --- |
| Voice LLM (llama-server) | llama.cpp ROCm build | `vox llm-server` / `scripts\start_llm.ps1` | 8083 | R9700 (`ROCm0`), ~4.8 GB | Qwen3.5-4B Q4_K_M + its vision projector |
| TTS service | Audio Clone `.venv` | `scripts\start_tts.ps1` | 8870 | RX 9070 XT (`tts_device="9070"`), ~5.2 GB | VoxCPM2, prompt cache, speaker-verification model (CPU) |
| Assistant (web) | project `.venv` | `vox serve` | 8765 | – | Agent, tools, voice profiles, audit, reminders |
| Assistant (voice) | project `.venv` | `vox listen` | – | – | the same, plus microphone, speaker, Whisper, spotter, Silero, hotkey |
| Dual_gpu_setup lanes | – | the user | 8081/8082 | R9700 / 9070 XT | the user's 27B models, unchanged |

`scripts\start_all.ps1` starts the voice LLM and TTS in their own windows, waits until both answer, then runs `serve` or `listen`.

**Why a separate small voice model** (decision D16 in the SDD). The brief's hardware plan puts "Qwen 4B/8B" on the R9700 for the assistant. Measured with `scripts/agent_eval.py` (same requests, real tools, actions declined): Qwen3.5-4B 24/24 correct and 0.37 s median to first words; Qwen3.8-27B on the Dual_gpu lane 23/24 and 1.37 s, and it announced "Opening YouTube now" before being asked. With all 26 tools the 4B scores 47/48 at 0.39 s median; the one miss listed the windows before trying to minimize one, which the eval now accepts. The 4B also carries a vision projector, which gives screen understanding without another model.

**Why the voice model is on the 9070 XT.** The brief puts speech there, and the R9700 cannot hold it: the 27B lane uses 25.8 GB, the voice LLM 4.8 GB, VoxCPM2 another 5–6 GB. When a card is over-committed, Windows silently pages an idle model's VRAM out to system RAM (observed: the 27B's 25.8 GB vanished from the R9700 when VoxCPM2 loaded beside it). Residual contention: if the Dual_gpu 9070 XT lane (27B Q3, ~14 GB) is active at the same time as the voice, the 16 GB card is over-committed and Windows pages one of them; nothing crashes, but the first request after a page-out is slow.

Why the TTS service is a separate process: VoxCPM2 needs Python 3.12, ROCm PyTorch 2.9.1 and `voxcpm==2.0.3`, which Audio Clone maintains; HTTP keeps those pins out of this project and isolates GPU-driver crashes. `vox_assistant` is on that interpreter's `PYTHONPATH` so the service reuses `voices.py`, `config.py`, `security.py` and `disclosure.py`.

`vox serve` and `vox listen` are independent processes with separate conversation histories; they share the SQLite files.

## 4. Module map

```
src/vox_assistant/
  config.py          Settings dataclass; defaults < data/config.toml < VOXA_* env (bools, lists, optional paths parsed)
  voices.py          VoiceProfiles (versioned, active, revoke), quality analysis, reference conversion   [no torch]
  enroll.py          guided script, take assessment, assembly, mic recording loop
  tts_service.py     FastAPI TTS service: HIP-runtime fix, MIOpen fast mode, GPU by name, /audio/speech, /warm,
                     /voices/{id}/similarity, labelled WAVs
  voice_similarity.py speaker-verification embeddings (WavLM x-vector), cosine score   [TTS venv only]
  disclosure.py      "this is AI-generated" WAV label (RIFF INFO) and the guest-mode notice
  tts_client.py      ClonedVoiceTTS: VoxVortex TTS protocol over HTTP, warm()
  tts_neutral.py     NeutralVoiceTTS: Windows System.Speech, text passed via environment variable
  asr_local.py       LocalWhisperASR: faster-whisper behind VoxVortex's ASR protocol (threads configurable)
  audio_io.py        Microphone (32 ms frames), Speaker (stoppable queue), device resolution
  hotkey.py          GlobalHotkey: Win32 RegisterHotKey on a thread (push-to-talk)
  llm_client.py      ToolChoiceLLM: OpenAICompatLLM + tool_choice + parallel_tool_calls + "</think>" stop
  llm_server.py      builds and runs the voice LLM's llama-server command line
  permissions.py     Risk, Tool (with per-tool timeout), ToolRegistry, PermissionGate, ok()/fail()
  audit.py           AuditLog (SQLite, append-only)
  agent.py           Agent: LLM <-> tools loop, false-claim guard, history, system prompt, prewarm()
  tools/             system, models, files, knowledge, apps, projects, windows, screen, clipboard, reminders, workflows
  voice.py           VoiceLoop: VAD turn-taking, wake spotting, hotkey wake, speculative ASR, pre-warm and keep-warm,
                     acknowledgements, eager first clause, barge-in, spoken confirm, mute / neutral / guest mode, reminders
  webapp.py          FastAPI app: speak, chat (NDJSON), confirm, voices, audit, reminders
  security.py        loopback-only request guard shared by webapp and tts_service
  cli.py             vox devices|enroll|voices [list|use|revoke|check]|say|ask|serve|listen|llm-server|doctor
web/index.html       single-page UI (no build step)
scripts/             start.ps1 start_all.ps1 start_llm.ps1 start_tts.ps1
                     smoke_test.py (real components) agent_eval.py (model accuracy and speed)
                     clone_check.py (enroll -> clone -> score -> revoke) desktop_check.py (live desktop tools)
tests/               116 tests (see sdd.md section 8)
```

Dependency direction: `cli, webapp, voice` → `agent` → `permissions, audit, llm_client`; `tools/*` → `permissions, config`; `tts_service` → `voices, config, security, disclosure`; `voices`, `config`, `disclosure` import nothing from the project. No cycles.

## 5. The turn pipeline

### 5.1 Hands-free turn (`vox listen`)

```
mic 16 kHz PCM16 ─▶ 32 ms frames ─▶ Silero VAD prob
   │
   ├─ 3 voiced frames ─▶ START utterance (pre-roll prepended)
   │      ├─ awake/always mode: pre-warm LLM + voice model now
   │      └─ asleep: every 250 ms between 0.6 and 3 s, tiny.en reads the utterance so far;
   │                 wake phrase heard ─▶ chime + pre-warm, while the user is still talking
   ├─ 200 ms of silence ─▶ start Whisper small.en on the utterance so far (speculative)
   ├─ speech resumes ─▶ discard that transcript
   ├─ 450 ms of silence ─▶ END utterance (use the speculative transcript) ─▶ drop if < 250 ms of speech
   ▼
hallucination filter ─▶ confirmation pending? ─yes─▶ deliver to the waiting _confirm()
   │no
   ▼
wake gate: phrase/alias in the first 3 words, or awake window (30 s), or hotkey pressed
   ▼
local commands (mute, neutral voice, my voice, guest mode on/off) ── the LLM never sees them
   ▼
Agent.run(request) ─▶ Say deltas ─▶ EagerSegmenter ─▶ queue ─▶ talker ─▶ TTS.synth ─▶ Speaker.play
                   └▶ ToolStarted(read) and nothing said yet ─▶ queue a pre-synthesized "Let me check."
```

The talker task is fed by a queue, so synthesis of chunk *n* overlaps generation of chunk *n+1*, and playback of chunk *n* overlaps synthesis of *n+1*. The **first** chunk ends at the first clause break after at least three words (`, ; : –`), or the first sentence end, whichever comes first; VoxCPM2 does not stream within a chunk, so a long first sentence delayed the first audio by about 7 s before this change.

### 5.2 Typed turn (web app)

`POST /api/chat` starts `Agent.run` in a task and streams newline-delimited JSON events (`say`, `tool`, `tool_result`, `confirm_request`, `error`, `done`). A confirmation request is just another event; the browser answers with `POST /api/confirm`. Closing the stream cancels the turn and any open confirmation.

### 5.3 Speak-only

`POST /api/speak {text, voice_mode, preset}` → `tts_service /audio/speech (wav)` → labelled WAV with `X-RTF`, `X-Audio-Seconds`, `X-Voice`, `X-AI-Generated` headers; or Windows speech if `voice_mode=neutral` (also labelled).

## 6. Agent, tools and permissions

### 6.1 Agent loop

```
append user message, stamped "[HH:MM] text"
repeat up to 6 rounds:
    for attempt in (0, 1):                       # attempt 1 only after a discarded false claim
        ONE leading system message (+ CORRECTION on attempt 1) and history
        stream LLM (parallel tool calls allowed; tool_choice="required" on attempt 1)
        gate the first sentence (6.4); yield Say fragments
    append assistant message (with tool_calls, if any)
    if no tool calls: return
    for each call: _execute(call) ─▶ ToolStarted, gate, handler (per-tool timeout), audit, ToolFinished
else: apologise
```

**Prompt-cache layout.** The system prompt carries only the date; the time is stamped into each user message when it is added, and history is never rewritten. The system text and the 26 tool schemas are therefore a byte-identical prefix that llama-server reuses on every request; with the minute in the system prompt the whole prefix (tools and history) was re-processed every new minute.

**`prewarm()`** sends the current prompt plus a placeholder user turn for one token. It fills the prefix cache and pages the model back into VRAM; failures are ignored.

History is trimmed to the last 24 messages at a `user` boundary. On cancellation the text actually produced is stored with an `[interrupted]` marker.

### 6.2 Tool contract and catalogue

```python
Tool(name, description, parameters(JSON schema), risk: READ|ACT|DESTRUCTIVE, handler(args)->dict,
     describe(args)->str, timeout_s=None)   # None = 30 s
```

Handlers return `ok(**data)` or `fail(error, **data)` and report *measured* outcomes. Text that originates outside the user's voice carries `untrusted_content: true`.

| Tool | Risk | Source of truth / verification | Confinement |
| --- | --- | --- | --- |
| `gpu_status`, `list_models` | read | Dual_gpu `/api/hardware` + `/api/status` | managed lanes only |
| `available_models` | read | Dual_gpu `/api/bootstrap` catalog | – |
| `load_model` | **destructive** | `POST /api/clients/deploy_lane`; reply's server for the lane must be alive | refuses the lane whose port the assistant's own LLM uses; 330 s timeout |
| `unload_model` | **destructive** | `POST /api/unload {"lane": …}` (never the bare form that stops every model) | same refusal; 120 s |
| `system_info` | read | psutil | – |
| `find_files`, `read_text_file` | read | directory walk / file | `search_roots`; secret-like names hidden |
| `search_notes` | read | full text of .md/.txt/.rst/.org/.adoc, cached by mtime | `knowledge_roots`; secret names and build dirs skipped; snippets untrusted |
| `read_clipboard` | read | PowerShell `Get-Clipboard` | 6000 chars returned |
| `set_clipboard` | act | value via env var, read back | – |
| `open_app` | act | `cmd /c start`, then the process must appear | `apps` allowlist only |
| `open_path` | act | `os.startfile` | inside roots; executables and scripts refused |
| `open_url`, `web_search` | act | default browser (`web_search_url`) | http/https only; result cannot be read back |
| `list_projects` | read | folders directly in `search_roots` | – |
| `open_project` | act | `code <folder>`, then a VS Code window titled with the folder must appear | project folders only |
| `delegate_coding_task` | act | new console in the project folder runs `coding_agent` (OpenCode); its process must appear | task text sanitised and passed by environment variable |
| `list_windows` | read | Win32 EnumWindows (visible, uncloaked, titled) | titles are untrusted |
| `window_action` | act | focus / minimize / maximize / restore, then window state is read back | – |
| `close_window` | **destructive** | WM_CLOSE, then the window must be gone; never forced | – |
| `look_at_screen` | read | screenshot (main display, ≤1280 px JPEG) described by the local vision model | description untrusted; 90 s |
| `set_reminder`, `cancel_reminder` | act | SQLite, read back | 1 min – 14 days |
| `list_reminders` | read | SQLite | – |
| `run_workflow` | act | runs the configured steps, reports each | destructive steps refused; one approval names every step |

### 6.3 Permission gate

| Risk | Decision |
| --- | --- |
| READ | run |
| ACT | ask, unless the tool name is in `auto_approve` (none by default) |
| DESTRUCTIVE | always ask; `auto_approve` is ignored |
| any, with no confirmer wired | refuse (`denied-no-confirmer`) |

The voice confirmer speaks "I'm about to *intent*. Say yes to go ahead, or no to cancel." and waits for the next utterance (15 s, one re-ask, **"no" anywhere wins**); the web app emits `confirm_request` and waits for the button (90 s). The intent comes from the tool's `describe()`, not the model. For a workflow the intent lists every step. The decision is about the request, never about who is speaking.

### 6.4 False-claim guard

Observed with the real 27B model: for some phrasings ("please open notepad") it answers "Notepad is open." without calling the tool. The agent enforces truthfulness in code:

1. Until a tool has run in the turn, the first sentence is held back (to `.!?`+space, newline, 100 chars or end of stream).
2. If it matches the action-claim pattern it is **discarded**, an audit row `guard/false-claim` is written, and the model is asked again with a correction in the single system message and `tool_choice: "required"`.
3. If the same reply also contained tool calls, the claim text is dropped and the calls proceed through the gate.
4. After a tool has run, confirmations are spoken normally; a second unsupported claim is accepted (bounded).

Encoded model constraints: Qwen's chat template **rejects any system message that is not first** (HTTP 500); with reasoning off the model sometimes emits `answer </think> answer`, so `</think>` is a stop sequence; llama-server allows one tool call per reply unless `parallel_tool_calls` is set, which the client now sends.

## 7. Voice loop internals

`VoiceLoop.run(frames)` consumes 32 ms frames; all collaborators (`asr`, `spotter`, `tts`, `neutral_tts`, `speaker`, `vad`, `agent`) are injected, so tests drive it with scripted frames.

| Parameter | Default | Meaning |
| --- | --- | --- |
| speech start | 3 voiced frames at p > 0.5 | |
| speculative ASR | after 200 ms of silence | discarded if speech resumes; saves ~0.25 s per turn |
| end of utterance | 450 ms of silence | |
| minimum speech | 250 ms | shorter bursts are noise |
| wake spotting | tiny.en on 0.6–3 s of audio, every 250 ms, **no prompt** | priming with the phrase caused 1/10 false wakes; unprompted 0/10 with 4/4 detected |
| hotkey | `ctrl+alt+j` | `RegisterHotKey`; wakes like the phrase |
| pre-warm | on wake, and at utterance start when awake; at most every 15 s | LLM one-token request + TTS `/warm` |
| keep-warm | every `keep_warm_s` = 15 s while idle and not muted | GPUs went cold after ≥ 30 s idle (+1.5–2.6 s on the next request) |
| acknowledgement | `ack_phrases`, synthesized once per voice at start | played when a read tool starts before any words; never synthesized on demand |
| barge-in | 320 ms of speech at p > 0.7 while busy | cancel turn, clear speaker |
| awake window | 30 s | follow-ups need no wake phrase |
| confirm timeout | 15 s | silence = refusal |

**Wake reaction.** While asleep the full transcript is not needed to wake: the spotter reads the start of the utterance as it is spoken and fires mid-sentence (smoke test: the chime came 3.5 s before the user finished a 4.8 s request). The full transcript still decides what the request is, and still matches the phrase or aliases. If the wake phrase is said alone, the chime is the acknowledgement (no "Yes?").

**Local commands** (regex on the whole utterance, before the LLM): `mute | stop listening | go to sleep`; `use/switch to a neutral|default|generic voice`; `use/switch to my voice`; `guest mode (on) | turn on guest mode | we have guests|visitors|company`; `guest mode off | turn off guest mode | I'm alone`. While muted only "wake phrase + unmute/wake up/listen" is acted on.

**Guest mode (AI disclosure).** When on, every reply uses the neutral voice and turning it on says "Just so everyone knows, I'm an AI assistant, and this is a computer voice." With `guest_mode = true` in the config, the first reply of the session starts with that sentence. "Use my voice" in guest mode is acknowledged but the neutral voice stays until guest mode is turned off.

**TTS fallback.** `_speak` tries the selected engine, then the neutral one; the first fallback emits `tts_fallback` once.

**Reminders.** A 2 s poll; a due item waits until the assistant is quiet, then is spoken.

## 8. Voice profiles and the TTS service

### 8.1 Profiles (`voices.py`)

```
data/voices/
  state.json                         {"active": "me_v2"}
  me_v1/ reference.wav  profile.json  # 16 kHz mono PCM16; id, name, version, transcript, quality, source, uuid
```

`create()` validates consent, name and transcript, converts to the reference format, runs the quality analysis (3–40 s; clipping measured on the raw input; SNR ≥ 12 dB), writes `<name>_v<N>`, and activates it unless `activate=False`. `revoke()` deletes the folder and falls back to the newest remaining profile. Ids are validated before any filesystem access.

### 8.2 Enrollment (`enroll.py`)

Eight scripted lines; `record_take` waits for speech, records until 1.1 s of silence; `assess_take` trims and analyses; takes are joined (350 ms gaps) once 20 s is reached (cap 36 s); the script text is the exact transcript. `scripts/clone_check.py` drives this exact path with a Windows voice reading the script into `record_take` through real Silero VAD.

### 8.3 TTS service (`tts_service.py`)

**Making ROCm PyTorch work on this machine** (both found while closing the cloned-voice gap):

1. The `rocm-sdk` 7.2.1 wheel bundles `amdhip64_7.dll` 10.0.3581, which crashes (`0xC0000005`) in `hipGetDeviceCount` with the installed Adrenalin driver 32.0.31036, whose own runtime is 10.0.3679 (the wheel's `hipInfo.exe` crashes too; a copy run against the driver's DLL lists all three GPUs). `use_driver_hip_runtime()` loads `%SystemRoot%\System32\amdhip64_7.dll` before `import torch`; Windows then binds torch, rocBLAS and MIOpen to that copy. Setting: `tts_hip_runtime = "driver" | "bundled"`.
2. TorchScript's GPU fuser compiles kernels with the wheel's `hiprtc0702.dll`, which does not match the driver runtime and crashed in VoxCPM2's `snake` activation; the driver ships no hipRTC. `PYTORCH_JIT=0` runs `snake` as ordinary PyTorch ops.

Then, in order: Audio Clone's `configure_runtime` (caches, ROCm target family; avoids the SDK's `offload-arch` launcher, which breaks on the space in "Audio Clone"), `MIOPEN_FIND_MODE=FAST`, GPU chosen by name (`tts_device="9070"` → `cuda:1`), `VoxBackend`.

**`MIOPEN_FIND_MODE=FAST`.** MIOpen's default benchmarks kernels for every new tensor shape, and every new sentence length is a new shape: measured 10–25 s (RTF 4–5) for the first synthesis of each length, every time a new length appeared. FAST picks a kernel immediately: first runs cost the same as repeats (RTF 0.5–0.65) for about 20 % slower warm synthesis.

**Endpoints.**

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | model loaded, device, load time, active voice |
| `POST /audio/speech` | `{input, voice?, response_format: pcm|wav, preset?}`; pcm streams sentence chunks and stops on client disconnect; wav carries `X-RTF`, `X-Audio-Seconds`, `X-Chunks`, `X-AI-Generated` and a RIFF INFO label |
| `POST /warm` | tiny synthesis to page weights back in; skipped if not loaded or busy (0.38 s warm, 2.6 s after ≥ 30 s idle) |
| `POST /voices/{id}/similarity` | clones a fixed sentence and scores it against the reference with `microsoft/wavlm-base-plus-sv` (CPU); optional `compare_with` scores against another speaker. Threshold 0.86 (model card) |
| `GET /voices`, `POST /voices/{id}/activate`, `DELETE /voices/{id}` | profile management (DELETE needs no body or content type, see 13) |
| `POST /load`, `/forget`, `/unload` | model and prompt-cache control |

`TTSState` holds the backend, an LRU of 8 acoustic prompts keyed by `(sha256(reference.wav), transcript)` and one lock (one GPU request at a time). `--preload` loads the model and synthesizes a warm-up sentence before serving.

## 9. Web application

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | the page |
| `/api/status` | GET | LLM (name of the served model) / TTS reachability, active voice, voices, tools, pending reminders |
| `/api/speak` | POST | text → labelled WAV (cloned via TTS service, or neutral) |
| `/api/chat` | POST | agent turn, NDJSON event stream |
| `/api/confirm` | POST | answer a `confirm_request` |
| `/api/reset` | POST | clear conversation |
| `/api/voices` | GET / POST | list / add from a base64 WAV (+ transcript + consent) |
| `/api/voices/{id}/activate` · `/revoke` | POST | switch / delete |
| `/api/reminders/due` | GET | page polls every 5 s |
| `/api/audit` | GET | latest audit rows |

The page is dependency-free HTML/JS with light/dark themes. Checked in headless Edge at 1400 px and inside a 375 px frame: no horizontal scroll (the narrow layout uses `minmax(0,1fr)`, because a plain `1fr` grid column cannot shrink below its widest content). All dynamic text is escaped.

## 10. Data and storage

| Path | Format | Contents |
| --- | --- | --- |
| `data/config.toml` | TOML | setting overrides, workflows |
| `data/voices/` | files | profiles (8.1) |
| `data/audit.sqlite3` | SQLite `events(id, ts, kind, tool, risk, args, decision, ok, summary)` | append-only |
| `data/reminders.sqlite3` | SQLite `reminders(id, due, text, state)` | `pending → fired | cancelled` |
| memory only | – | notes text cache (`search_notes`), acoustic prompts, acknowledgement audio |

## 11. Interfaces

**LLM (outbound).** `POST {llm_base_url}/chat/completions`, streamed, with `tools`, `max_tokens=400`, `temperature=0.3`, injected `stop:["</think>"]`, `parallel_tool_calls:true` when tools are sent, and `tool_choice:"required"` on a guard retry. `look_at_screen` sends one extra request with an `image_url` content part to the same server.

**Voice LLM server (`vox llm-server`).** `llama-server --model <llm_gguf> --mmproj <llm_mmproj> --alias voice --device ROCm0 --gpu-layers 999 --ctx-size 32768 --parallel 1 --port 8083 --flash-attn auto --jinja --reasoning off --reasoning-budget 0` (cache reuse is unavailable with a multimodal projector; the prompt prefix is still cached per slot).

**Dual_gpu_setup.** Telemetry as before; `available_models` reads `/api/bootstrap`; `load_model`/`unload_model` follow the service's client rules (lane-scoped, never the bare unload).

**TTS (inbound).** Section 8.3. **Web (inbound).** Section 9. **CLI.** `vox devices | enroll [--name] | voices [list|use|revoke|check <id>] | say [--neutral] [--preset] [--out] <text> | ask <text> | serve | listen [--always] | llm-server | doctor`.

## 12. Configuration

Precedence: defaults in `config.py` < `data/config.toml` < `VOXA_<NAME>` environment variables. Unknown keys are an error. Lists in env vars are `;`-separated; booleans accept 1/true/yes/on; an empty value clears an optional path.

| Group | Settings (defaults) |
| --- | --- |
| LLM | `llm_base_url` (`http://127.0.0.1:8083/v1`), `llm_model`, `llama_server`, `llm_gguf` (Qwen3.5-4B Q4_K_M), `llm_mmproj`, `llm_gpu` (`ROCm0`), `llm_ctx` (32768), `dual_gpu_url` |
| Voice | `tts_url`, `audio_clone_root`, `tts_device` (`9070`), `tts_preset` (`Draft`), `tts_hip_runtime` (`driver`) |
| Listening | `whisper_model` (`small.en`), `whisper_threads` (8), `whisper_device`/`whisper_compute`, `wake_phrase`, `wake_aliases`, `wake_spotter_model` (`tiny.en`), `hotkey` (`ctrl+alt+j`), `keep_warm_s` (15), `ack_phrases`, `input_device`/`output_device`, `silero_model` |
| Behaviour | `assistant_name`, `guest_mode` (false), `gpu_temp_limit_c` (90) |
| Tools | `search_roots`, `knowledge_roots`, `apps`, `coding_agent` (`["opencode","--prompt","{task}"]`), `web_search_url`, `workflows` (default `status_check`) |
| Web | `web_port` (8765) |

## 13. Security and privacy model

| Threat | Control |
| --- | --- |
| A web page the user visits drives the local services | loopback bind; `Host` must be loopback (defeats DNS rebinding); `Origin` must be loopback; POST/PATCH must be `application/json` (a body-less cross-site POST is a "simple request"). PUT/DELETE need no content type: a cross-site page must pass a CORS preflight, which these servers never grant. Same guard on the TTS service |
| Model hallucinates or is manipulated into acting | risk levels + gate outside the model; allowlisted apps; executables refused; guard against unperformed-action claims; destructive tools always ask and are excluded from workflows |
| Prompt injection via files, notes, clipboard, window titles, screen content | `untrusted_content` flag; system prompt; acting still needs a gated tool call; the vision prompt says to ignore instructions shown on screen |
| Injection through the coding-agent task text | the task travels in an environment variable expanded after cmd parsing (`!VOXA_TASK!`) and cmd metacharacters are removed |
| Path traversal / exfiltration | roots confinement; secret-like names hidden; the only network actions open the user's browser after approval |
| Someone else speaks to the assistant | the voice is not authentication; acts still need an explicit yes; mute and guest mode are instant local commands |
| Voice clone misuse | consent flag; revoke deletes recording and clears prompt caches; **guest mode** (neutral voice + spoken AI notice); **every WAV written is labelled AI-generated** (RIFF INFO comment naming the profile, `X-AI-Generated` header) |
| Screen capture | stays in memory, sent only to the local LLM, never stored |
| Workload control | load/unload are destructive, lane-scoped, and refuse the assistant's own lane |

Not defended: a malicious process running as the same user, audio played into the microphone (mitigated only by the gate), physical access. A coding agent's own permissions (what it may change without asking) are the agent's configuration.

## 14. Failure handling

| Failure | Behaviour |
| --- | --- |
| Cloned TTS service down / errors | neutral voice for that sentence; `tts_fallback`; web returns 503 with how to start it |
| ROCm runtime mismatch (fixed, 8.3) | if a future driver/wheel combination crashes again, the TTS process dies alone; set `tts_hip_runtime="bundled"` to compare; the smoke test's torch probe reports FAIL |
| GPU paged out / cold after idle | keep-warm while listening; pre-warm at wake; the first request after a page-out is slow, not wrong |
| LLM unreachable / error | spoken apology, `error` event |
| Tool exception or timeout | `fail(...)` for the model to report |
| ASR exception | `error` event; utterance dropped |
| Barge-in | turn cancelled, speaker emptied; a started tool is shielded and recorded |
| Hotkey taken by another program | warning at start; wake phrase still works; `vox doctor` reports it |
| Coding agent not on PATH | tool says so; `vox doctor` reports it |
| Broken PATH entry (stray `"`) | agent launched by the full path Python resolves; `vox doctor` warns |

## 15. Concurrency model

* One asyncio loop per process. Whisper, the spotter, notes search and TTS generation use `asyncio.to_thread`; PowerShell helpers use subprocesses.
* The hotkey runs its own Win32 message loop on a daemon thread and hands presses to the event loop with `call_soon_threadsafe`.
* Microphone and speaker use sounddevice callbacks; the mic queue drops the oldest frame rather than fall behind.
* TTS service: a `threading.Lock` serialises model use; `/warm` only takes it if free.
* The notes cache is guarded by a lock; it is filled by a background thread at start-up.
* Tasks spawned by the voice loop (spotter, speculative ASR, warm-ups, acknowledgement preparation, reminders) are tracked and cancelled on shutdown.

## 16. Performance

All figures measured on this machine on 2026-10-22.

**End to end (smoke test §5, real time, cloned voice, question needing two tools):**

| Build | First sound | First audio of the answer |
| --- | --- | --- |
| First build (27B, neutral voice) | – | 6.7–18 s |
| This build, before the voice-path fixes (4B, cloned voice) | – | 9.3–9.6 s |
| This build, final | **1.26–1.55 s** ("Let me check" in the cloned voice) | **2.9–3.6 s** |

Final timeline after the end of speech (one run): transcript +0.95 s, both tools started +1.26 s and finished +1.63 s (one LLM round), first answer text +1.76 s, first answer audio +3.59 s. The rest is synthesis of the first clause: VoxCPM2 generates a whole chunk before it can play.

| Change | Measured effect |
| --- | --- |
| Voice LLM 27B → Qwen3.5-4B (own server) | first words 1.37 → 0.37 s median; decode 26 → 125 tokens/s; accuracy 23/24 → 24/24 |
| Eager first clause + speakable model names + shorter replies | first answer audio 9.3 → 4.7 s; spoken answer 24 → 10 s |
| Speculative ASR at 200 ms of silence | transcript ~0.25 s sooner |
| `parallel_tool_calls` | two tools in one LLM round (~0.3 s) |
| Whisper 8 threads (32-thread CPU) | 998 → 818 ms per request, same accuracy (base.en: 3× faster but 13.5 % vs 4.7 % word errors) |
| `MIOPEN_FIND_MODE=FAST` | first synthesis of each new sentence length 10–25 s → 2–4 s |
| Keep-warm every 15 s while listening | avoids +1.5–2.6 s after ≥ 30 s idle (LLM and voice model) |
| System prompt without the minute | prompt prefix (system + 26 tools + history) cached across minutes |
| Notes text cache | `search_notes` 7–17 s → 0.09–0.2 s |

**Components.** VoxCPM2 on the 9070 XT: RTF 0.52 (Draft, warm), first audio of a short streamed sentence 0.82 s. Whisper small.en: 0.74–0.85 s for 4.8 s of speech. Wake spotter (tiny.en): ~140 ms per check. Hotkey: 0.6 ms from keypress to callback. `look_at_screen`: 2.9–4.7 s. Speaker similarity of a clone to its reference: 0.978, to a different speaker 0.474.

## 17. Extending the system

**Add a tool**: a `Tool` in `tools/<name>.py` `make_tools(settings)`, lowest honest risk level, `describe` for ACT/DESTRUCTIVE, verified results, `untrusted_content` for outside text, register in `tools/__init__.py`, add a unit test and a case in `scripts/agent_eval.py`, and re-run the eval: tool lists change how small models choose.

**Swap the voice LLM**: set `llm_gguf`/`llm_mmproj` (or point `llm_base_url` elsewhere) and run `scripts/agent_eval.py` before adopting it.

**Swap the TTS engine**: anything with `sample_rate` and `async synth(text)` fits `VoiceLoop`; a Qwen3-TTS or Chatterbox service can implement the same `/audio/speech` contract and be scored with `/voices/{id}/similarity` and `scripts/clone_check.py`.

**Ubuntu**: replace `tts_neutral`, `apps`, `windows`, `clipboard`, `hotkey` and the PowerShell helpers behind the same tool names.

## 18. Known limits

* The first audio of a *researched* answer is 2.9–3.6 s, above the brief's 1–2 s; the 1–2 s target is met only by the acknowledgement. Closing the rest needs a TTS that streams within a sentence.
* No acoustic echo cancellation: use a headset, or barge-in will mis-trigger.
* The 9070 XT is shared with the Dual_gpu 27B Q3 lane; when both are active, Windows pages one of them.
* The cloned voice's similarity to *you* is unmeasured until you enroll; the measurement path is ready (`vox voices check`).
* The false-claim detector is a first-sentence regex.
* English only by default.
* Browser automation stops at opening pages and searches; no clicking or form filling.
* `vox listen` and `vox enroll` have been exercised end to end with synthetic speech fed through the real code paths, but not with a person at a microphone.
* Windows only. No Android companion.
