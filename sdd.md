# VoxVortex Desktop: Software Design Document

| | |
| --- | --- |
| Product | VoxVortex Desktop, a local voice assistant that speaks in the owner's cloned voice and acts on the PC with permission |
| Version | 0.2.0 (second build: cloned voice working, P2 features, latency work) |
| Date | 2026-10-22 |
| Platform | Windows 11 Pro, Python 3.12, Ryzen 9 9950X3D, Radeon AI PRO R9700 32 GB + RX 9070 XT 16 GB, Adrenalin 32.0.31036 |
| Companion documents | [architecture.md](architecture.md) (how it works), [README.md](README.md) (how to run it) |

**Status words.** *Verified* = exercised by an automated test and/or a real-component script (smoke test, clone check, desktop check, agent eval). *Implemented* = code and unit tests with stand-ins, real dependency not exercised. *Needs a person* = can only be completed by the owner (their voice, their ears). *Not built* = out of this release.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Context, stakeholders, assumptions](#2-context-stakeholders-assumptions)
3. [Requirements](#3-requirements)
4. [Design decisions](#4-design-decisions)
5. [Design overview](#5-design-overview)
6. [Interface and data specifications](#6-interface-and-data-specifications)
7. [Quality attributes](#7-quality-attributes)
8. [Verification](#8-verification)
9. [Traceability](#9-traceability)
10. [Review findings (defects found and fixed)](#10-review-findings-defects-found-and-fixed)
11. [Risks and open issues](#11-risks-and-open-issues)
12. [Coverage of the original brief](#12-coverage-of-the-original-brief)
13. [Operations](#13-operations)
14. [Glossary](#14-glossary)

---

## 1. Purpose and scope

**Purpose.** Give the owner a private, hands-free desktop agent: speak to it, it understands, checks or does things on the PC through controlled tools, and answers aloud in a synthetic copy of the owner's own voice, with all inference local.

**In scope (this release).** Voice enrollment, versioned profiles, cloned speech on the GPU and a similarity check; hands-free voice loop with wake phrase, wake spotting, push-to-talk hotkey, interruption and spoken permission; typed chat and voice studio in a local web app; permissioned tools for GPU/host status, LLM workload control, files, notes search, apps, projects, coding-agent delegation, windows, screen understanding, web pages and search, clipboard, reminders and workflows; audit log; privacy controls (mute, neutral voice, guest mode with AI disclosure, revoke, labelled audio files).

**Out of scope.** Ubuntu/Wayland control; clicking or typing inside web pages; Android companion; fine-tuned voices; multi-user hosting.

## 2. Context, stakeholders, assumptions

**Stakeholders.** Owner (sole user and administrator). Other people in the room are *bystanders*: they must not be able to make the assistant act without the owner's approval, and they must be told when they hear an AI voice (guest mode).

**Existing assets** (none modified): VoxVortex (speech building blocks), Dual_gpu_setup (27B lanes on :8081/:8082, telemetry and model management on :8090), Audio Clone (VoxCPM2 environment), llama.cpp b11699 ROCm build, the model library on `G:\LLM models` (Qwen3.5-4B and its vision projector were already there; no LLM download was needed). Downloaded this round: Whisper `tiny.en` (wake spotter) and `microsoft/wavlm-base-plus-sv` (similarity check).

**Assumptions.** (A1) A headset in hands-free mode. (A2) Single-user machine, not exposed to a network. (A3) First-time model downloads need internet once.

**Constraints.** (C1) VoxCPM2 needs the Audio Clone environment (Python 3.12, ROCm PyTorch 2.9.1, `voxcpm==2.0.3`). (C2) The GPUs are shared with the owner's Dual_gpu lanes. (C3) The Dual_gpu `/v1` gateway serves only coding-agent-mode models. (C4) Windows 11.

## 3. Requirements

### 3.1 Functional

| ID | Requirement | Pri | Status |
| --- | --- | --- | --- |
| FR-01 | Create a voice profile from a guided recording; reject silent, short, clipped, noisy takes; store reference + exact transcript | P1 | Verified: rules (unit) and the full recording path (`record_take` + real Silero VAD, synthetic speaker, clone check). **Needs a person** for the owner's own voice |
| FR-02 | Versioned profiles; one active; switch, revert; revoke deletes the recording and cached voice data | P1 | Verified (unit; live revoke through the service in the clone check) |
| FR-03 | Explicit ownership/permission confirmation to create a profile | P1 | Verified |
| FR-04 | Type text, press Generate, hear it in the cloned voice | P1 | **Verified** live (web `/api/speak` cloned path, CLI, service) |
| FR-05 | Speak every reply in the active cloned voice | P1 | **Verified** live (smoke §5 in the cloned voice) |
| FR-06 | Neutral voice on request and as automatic fallback | P1 | Verified |
| FR-07 | Hands-free listening with VAD turn-taking, endpointing, noise rejection | P1 | Verified (real Silero + Whisper, real-time feed) |
| FR-08 | Wake phrase with a follow-up window, and push-to-talk | P1 | Verified: spotter mid-utterance (live), hotkey (live keypress), aliases |
| FR-09 | Barge-in stops speech and discards unsent audio | P1 | Verified with scripted audio; not with a live speaker/mic loop |
| FR-10 | Short, speakable conversational answers from the local LLM | P1 | Verified live (Qwen3.5-4B) |
| FR-11 | GPU status from real telemetry | P1 | Verified live |
| FR-12 | Open allowlisted applications and verify they run | P1 | Verified live (Notepad opened, verified, closed) |
| FR-13 | Find files by name within allowed folders | P1 | Verified live |
| FR-14 | Read clipboard; set clipboard with read-back | P2 | Verified live (and the owner's clipboard restored in full) |
| FR-15 | Local reminders spoken when due | P2 | Verified (unit, voice loop, HTTP) |
| FR-16 | Open a file/folder (never an executable), a URL, or a web search | P2 | Verified (unit; web search live) |
| FR-17 | Read / act / destructive risk levels; approval by voice or button; destructive never pre-approved | P1 | Verified |
| FR-18 | Never report an action as done unless a tool performed it | P1 | Verified (unit, 27B live; 4B eval: 0 false claims in 48) |
| FR-19 | Audit every tool request, decision and result | P1 | Verified |
| FR-20 | Mute, unmute, voice-mode switching by voice, without the LLM | P1 | Verified |
| FR-21 | Local web app: voice studio, chat with approval buttons, voice management, audit view | P1 | Verified (API on a real socket; rendering checked in headless Edge at 1400 px and 375 px) |
| FR-22 | CLI for enrollment, voices, speaking, asks, serving, listening, LLM server, diagnostics | P1 | Verified (`doctor`, `ask`, `voices check` path, `llm-server`); `enroll`/`listen` **need a person** at a microphone |
| FR-23 | LLM workload control by voice: list loadable models, load a model on a GPU, unload a GPU | P2 | Verified against a mock Dual_gpu (unit) and the live catalog (read); live load/unload **not run** (would replace the owner's lane configuration) |
| FR-24a | Project switching ("switch to the Cutline project" → VS Code) | P2 | Verified live (window title confirmed) |
| FR-24b | Coding-agent delegation (OpenCode in a new terminal in the project) | P2 | Verified live (agent process started in the right folder, then stopped) |
| FR-24c | Screen understanding (local vision model) | P2 | Verified live |
| FR-24d | Application/window control: list, focus, minimize, maximize, restore, close | P2 | Verified live on a Notepad window it opened |
| FR-24e | Personal knowledge retrieval (full-text search of notes) | P3 | Verified live |
| FR-24f | Voice-controlled workflows (named tool sequences, one approval) | P3 | Verified (unit + live `status_check`) |
| FR-25 | Disclose that speech is AI-generated: guest mode (neutral voice + spoken notice) and a label in every WAV written | spec §8 | Verified (unit; label read back from live service output) |
| FR-26 | Measure clone similarity to the reference speaker | NFR-05 support | Verified live (`/voices/{id}/similarity`, `vox voices check`) |
| FR-27 | Android companion | P3 | Not built |

### 3.2 Non-functional

| ID | Requirement | Target | Status |
| --- | --- | --- | --- |
| NFR-01 | No cloud AI requests during use | zero | Met by design (loopback clients; no cloud SDK). Not confirmed with a network monitor. `web_search` opens the user's own browser after approval |
| NFR-02 | First audible response after end of speech | 1–2 s | **Partly met**: first sound 1.26–1.55 s (acknowledgement in the cloned voice); first audio of the researched answer 2.9–3.6 s; a no-tool answer starts after ASR + ~0.15 s + one clause of synthesis |
| NFR-03 | Wake reaction | < 300 ms | **Met in a different form**: the spotter (~140 ms per check, every 250 ms) acknowledges the wake phrase *while the user is still speaking* (3.5 s before the end of a 4.8 s request); the hotkey reacts in < 1 ms |
| NFR-04 | Interruption stops audio promptly | ~64 ms after detection; detection 320 ms | Implemented; verified in scripted tests |
| NFR-05 | Voice similarity | blind-listening test | **Measured on a synthetic speaker**: 0.978 to its own reference vs 0.474 to another speaker (threshold 0.86). **Needs a person** for the owner's voice and the listening test |
| NFR-06 | Desktop operations confirmed from real state | always | Met where checkable; `open_url`/`web_search`/`open_path` say they cannot confirm |
| NFR-07 | Local services not drivable by a web page | loopback + header checks | Verified |
| NFR-08 | Voice data under owner control | plain files, deletable | Verified |
| NFR-09 | Degrade gracefully | fallback, apology | Verified |
| NFR-10 | Maintainable, hardware-free tests | | 116 tests, no GPU/mic needed |

## 4. Design decisions

D1–D15 from the first build stand (library reuse of VoxVortex, Windows first, VoxCPM2, TTS in its own process, Whisper on CPU, gate outside the model, false-claim guard, `</think>` stop, PowerShell/env-var OS integration, NDJSON chat, SQLite + plain files, loopback guard, script text as transcript). D5 (LLM via the 27B lane's llama-server) and D7 (transcript-only wake) are superseded:

| # | Decision | Alternatives | Rationale and consequence |
| --- | --- | --- | --- |
| D16 | **Own small voice LLM**: Qwen3.5-4B Q4_K_M + vision projector, own llama-server on the R9700 (:8083) | Keep the 27B lane (D5); load the 4B through Dual_gpu (would replace the 27B lane) | Measured 0.37 vs 1.37 s to first words, 24/24 vs 23/24 correct, 125 vs 26 tokens/s. The brief specified "Qwen 4B/8B" for this role. Fits beside the 27B (4.8 GB). The vision projector gives screen understanding. Not managed by Dual_gpu, so it is never evicted by it |
| D17 | **Driver HIP runtime + TorchScript off** in the TTS process | Reinstall/downgrade ROCm wheels or the driver; disable the iGPU | Root-caused (finding 15): the wheel's runtime crashes with the installed driver; the driver's works. A pre-import DLL load fixes it without touching Audio Clone or the driver. The JIT's hipRTC also mismatched; disabling it costs a small speed-up in one activation |
| D18 | **Voice model on the 9070 XT**, chosen by name | R9700 (the default choice of Audio Clone) | The R9700 cannot hold 27B + 4B + VoxCPM2; over-commit makes Windows page the idle model out (observed). Matches the brief's allocation. Contention remains with the 9070 XT Dual_gpu lane (R2) |
| D19 | **`MIOPEN_FIND_MODE=FAST`** | Default find mode; pre-warm many lengths | Default: 10–25 s for each new sentence length; FAST: 2–4 s first time, ~20 % slower warm |
| D20 | **Wake spotter + hotkey** in addition to the transcript wake gate | openWakeWord (no "hey vox" model; training a custom one is a project of its own); push-to-talk only | tiny.en on the partial utterance acknowledges mid-sentence; unprompted it had 0/10 false wakes and 4/4 detections. The hotkey is the brief's push-to-talk option |
| D21 | **Latency stack**: pre-warm on wake, keep-warm every 15 s, speculative ASR, parallel tool calls, eager first clause, cached acknowledgement, prompt-cache-stable system prompt, Whisper 8 threads | Bigger hardware changes; a different TTS | Each measured (architecture §16). Keep-warm trades ~2–3 % GPU time for avoiding a 1.5–2.6 s cold penalty; `keep_warm_s=0` turns it off |
| D22 | **Acknowledgement only for read tools**, only from pre-synthesized audio | Always; synthesize on demand | An action is followed by a spoken confirmation question anyway; on-demand synthesis would delay the answer it is meant to cover |
| D23 | **Guest mode + WAV labels** for AI disclosure | Always announce; watermark audio | Announcing to the owner alone every time is noise; a spoken notice when others are present, plus a metadata label that travels with exported files, covers the brief's §8 without degrading the audio |
| D24 | **Workload control through Dual_gpu's client API**, destructive, lane-scoped, own lane protected | Start/stop llama-server directly | Dual_gpu owns the lanes and their admission checks; its documented client rules (never the bare unload) are followed |
| D25 | **Coding agent in a visible console**, task via environment variable | Run headless (`opencode run`); `wt` | The owner watches and can stop it. `wt` drops argument quotes (finding 27); the env-var route also removes injection through task text |
| D26 | **Speaker-verification score** (WavLM x-vector, CPU) as the automatic similarity check | ECAPA (speechbrain not installed); only listening | transformers is already in the TTS environment; documented 0.86 threshold. Supports, does not replace, the blind-listening test |

## 5. Design overview

```
mic ─▶ VAD ─▶ spotter/hotkey ─▶ utterance ─▶ ASR (speculative) ─▶ wake gate ─▶ local commands ─▶ Agent ─▶ tools ─▶ OS / Dual_gpu
                                       pre-warm ─▶ LLM + voice model      confirm? ◀── PermissionGate ◀──┘   │
speaker ◀─ TTS (cloned | neutral) ◀─ acknowledgement / eager first clause / sentence stream ◀──────────────┘    AuditLog
```

Detail: [architecture.md](architecture.md).

## 6. Interface and data specifications

In architecture.md: processes and ports (§3), tool contract and catalogue (§6.2), gate (§6.3), TTS service API (§8.3), web API (§9), storage (§10), outbound requests (§11), configuration (§12).

Invariants:

* A `Tool` handler returns a dict with a boolean `ok`; failures carry `error`; tool failures never raise into a turn.
* The TTS service resolves the voice on every request.
* Exactly one system message, first, in every LLM request; it contains no clock time (prefix caching).
* `READ` tools have no side effects; `ACT`/`DESTRUCTIVE` tools never run without a recorded decision; workflows never contain destructive steps.
* Every WAV produced by this project carries an AI-generated label.
* Profile ids match `^[a-z0-9][a-z0-9_-]*_v\d+$` before filesystem access.

## 7. Quality attributes

**Performance.** Architecture §16 has every measurement. Headline (smoke test, real time, cloned voice, a question needing two tools): first sound 1.26–1.55 s, first audio of the answer 2.9–3.6 s (first build: 6.7–18 s with the neutral voice).

**Reliability.** Every dependency has a defined failure behaviour (architecture §14). The ROCm problem is now fixed at its cause; the process isolation (D4) remains.

**Security.** Threat table in architecture §13. New surfaces this release (screen capture, coding agent, workload control, notes) are read-only, gated, or both.

**Privacy.** Voice recordings exist only in `data/voices/`; prompt caches and the notes cache only in process memory; screenshots are never stored. Exported audio is labelled.

**Maintainability.** About 3,400 lines of Python source; no import cycles; every hardware dependency of the voice loop is injected; four real-component scripts complement the unit tests.

## 8. Verification

### 8.1 Automated tests (116, `pytest`, no GPU/microphone required)

| File | Tests | Covers |
| --- | ---: | --- |
| `test_voices.py` | 13 | versions, revoke, consent, id traversal, resampling, quality rules, clipping on raw input |
| `test_tts_service.py` | 7 | splitting, voice resolution and revocation (plain DELETE), wav headers, streaming, prompt cache, validation, `/warm`, WAV label |
| `test_agent.py` | 16 | gating, denial, pre-approval, errors, round limit, cancellation, history, cache-stable system prompt, time-stamped messages, prewarm |
| `test_guard.py` | 14 | claim detector and retry behaviour |
| `test_llm_client.py` | 4 | `stop`, `tool_choice`, `parallel_tool_calls`, two tool calls parsed from one stream |
| `test_tools.py` | 10 | first-build tools |
| `test_desktop_tools.py` | 12 | hotkey parsing, window/project/lane/model matching, coding-agent command and launch line, load/unload through a mock Dual_gpu incl. own-lane refusal, notes search, workflows, full registry, WAV label, LLM server command, TTS device by name, config parsing |
| `test_voice.py` | 23 | first-build loop behaviour plus wake spotting, false-wake rejection, hotkey, chime instead of "Yes?", guest mode (voice and notice), keep-warm, acknowledgements, speculative ASR, eager first clause |
| `test_webapp.py` | 8 | page, guard, speak proxy, voices, chat, confirmation over a real server |
| `test_gaps.py` | 9 | enrollment helpers, TTS guard, executables refused, HTTP reminders, wake spellings |

### 8.2 Real-component scripts

| Script | What it exercises | Latest result |
| --- | --- | --- |
| `scripts/smoke_test.py` | services; torch on the GPU via the driver runtime; mic and speaker; VAD, Whisper, wake phrase; real-time voice loop in the cloned voice with spotter, tools, acknowledgement and latency timeline; spoken permission refusal; mute; web app on a real socket; cloned speech round trip | **29 PASS, 0 FAIL, 1 WARN** (WARN: answer audio 3.59 s vs 1–2 s target) |
| `scripts/clone_check.py` | enrollment through `record_take` with real VAD (synthetic speaker), cloning, labelled WAV, Whisper round trip, streamed first audio, similarity vs own and other speaker, `/warm`, revoke | **12 PASS, 0 FAIL** (similarity 0.978 vs 0.474; first new-voice synthesis 3.0 s, RTF 0.52; first streamed audio 0.82 s) |
| `scripts/desktop_check.py --with-agent` | Notepad open/list/minimize/maximize/restore/focus/close, clipboard set/read/restore in full, projects, VS Code, web search, notes, screen, workflow, model catalog, coding agent | **17 PASS, 0 FAIL** |
| `scripts/agent_eval.py` | 24 requests × 2 through the real agent and tools, actions declined | Qwen3.5-4B: 47/48, 0 false claims, 0.39 s median to first words. Qwen3.8-27B (first 12 requests): 23/24, 1.37 s |

### 8.3 Not verified (explicit)

The owner's own voice (enrollment, similarity, blind listening); a person using `vox listen` with a microphone and speakers (echo, barge-in on real audio); live `load_model`/`unload_model` against the owner's running lanes; Hindi/Hinglish; behaviour under simultaneous load on the 9070 XT from the Dual_gpu lane and the voice model.

## 9. Traceability

| Requirement | Evidence |
| --- | --- |
| FR-01 | `test_voices`, `test_gaps`, clone check §1 |
| FR-02, FR-03 | `test_voices`, `test_tts_service`, `test_webapp`, clone check §4 |
| FR-04, FR-05 | smoke §5, §8 (cloned path), §9; clone check §2 |
| FR-06 | `test_voice` (neutral, fallback), smoke §8 |
| FR-07, FR-08 | `test_voice` (gating, spotter, hotkey), smoke §4–§5, live hotkey test |
| FR-09 | `test_voice::barge_in` |
| FR-10, FR-11 | smoke §5, agent eval |
| FR-12–FR-16 | `test_tools`, `test_gaps`, desktop check |
| FR-17, FR-18 | `test_agent`, `test_guard`, `test_voice`, `test_webapp`, smoke §6, agent eval |
| FR-19 | `test_agent`, `test_guard` |
| FR-20 | `test_voice`, smoke §7 |
| FR-21 | `test_webapp`, smoke §8, headless-Edge screenshots |
| FR-22 | `vox doctor`, `test_desktop_tools` (LLM server command) |
| FR-23 | `test_desktop_tools` (mock Dual_gpu), desktop check (catalog), agent eval |
| FR-24a–f | `test_desktop_tools`, desktop check, agent eval |
| FR-25 | `test_voice` (guest mode), `test_desktop_tools` / `test_tts_service` (label), clone check (label on live output) |
| FR-26, NFR-05 | clone check §3 |
| NFR-02, NFR-03 | smoke §5 timeline, spotter and hotkey measurements |
| NFR-07 | `test_webapp`, `test_gaps`, `test_tts_service` (DELETE/POST rules), smoke §8 |

## 10. Review findings (defects found and fixed)

Findings 1–13 are from the first build (false "Notepad is open" claim, second system message rejected, clipping masked by the limiter, executables via `open_path`, unguarded TTS service, web reminders never firing, wake spellings, duplicated `</think>` answers, console encoding, gateway rejecting reasoning-off models, iGPU in status, double denial rows, smoke leftovers). This round:

| # | Finding | Severity | How found | Fix |
| --- | --- | --- | --- | --- |
| 14 | (first build) ROCm access violation in `hipGetDeviceCount` | Blocking | – | closed by 15–16 |
| 15 | **Root cause of 14**: the rocm-sdk 7.2.1 wheel's `amdhip64_7.dll` (10.0.3581) crashes with Adrenalin 32.0.31036 (runtime 10.0.3679). The wheel's own `hipInfo.exe` crashes; a copy run against the driver's DLL lists all three GPUs. The iGPU was not the cause | Blocking | reproducing outside PyTorch | load the driver's DLL before `import torch` (D17) |
| 16 | TorchScript's fuser compiles with the wheel's hipRTC, crashing in VoxCPM2's `snake` once 15 was fixed | Blocking | faulthandler traceback | `PYTORCH_JIT=0` |
| 17 | `start_tts.ps1` killed the service on uvicorn's first stderr line whenever output was redirected (`$ErrorActionPreference='Stop'` in PowerShell 5.1) | High | background start died silently | `Continue` before launching, exit code passed through |
| 18 | VoxCPM2 next to the 27B on the R9700 over-commits VRAM; Windows pages the idle 27B out | High | VRAM counters | voice model on the 9070 XT (D18) |
| 19 | First synthesis of every new sentence length took 10–25 s (MIOpen exhaustive find) | High (latency) | clone check 35.8 s | `MIOPEN_FIND_MODE=FAST` (D19) |
| 20 | Both GPUs go cold after ≥ 30 s idle: +1.5–2.6 s on the next LLM token / synthesis; a cold-loaded 27B took 18.4 s to first token | High (latency) | idle-time measurements | pre-warm on wake, keep-warm while listening |
| 21 | The minute in the system prompt invalidated the prompt cache every minute | Medium (latency) | design review | date only; time stamped into user messages |
| 22 | A long first sentence (model file names) delayed first audio ~7 s | High (latency) | smoke timeline | eager first clause; speakable names and shorter replies in the prompt |
| 23 | llama-server allowed one tool call per reply, forcing two LLM rounds | Medium | smoke timeline | `parallel_tool_calls: true` |
| 24 | The TTS service refused body-less DELETE (415), so revoke through the service never worked; the old test hid it by sending a content type | High | clone check left profiles on disk | guard checks POST/PATCH only; test uses a plain DELETE |
| 25 | Priming the wake spotter with "Hey vox." made it hear the phrase in 1 of 10 unrelated sentences | Medium | false-wake test | spotter runs unprompted (0/10) |
| 26 | Ctrl+Alt+Space (first default) is taken by another program on this PC | Low | registration test | default `ctrl+alt+j`; `vox doctor` checks |
| 27 | Windows Terminal (`wt`) drops argument quotes, so the coding agent received a broken prompt | Medium | process probe | launch with `start` and an env-var task |
| 28 | A stray `"` in this PC's PATH hides every later entry (nodejs, npm) from cmd.exe, so `opencode` was "not recognized" | Medium (environment) | cmd probes | launch by the full path Python resolves; `vox doctor` warns. **The PATH itself is left for the owner to fix** |
| 29 | `search_notes` took 7–17 s: rescanned and reread every file, budget checked only per folder, included data files | Medium | agent eval | mtime-keyed text cache, budget inside the loop, notes-only suffixes, background warm-up (0.09–0.2 s) |
| 30 | `/api/bootstrap` sometimes exceeded a 10 s timeout | Low | desktop check | 30 s |
| 31 | **Test-script incident**: the first desktop check "restored" the owner's clipboard from `read_clipboard`, which returns at most 6000 characters, truncating a 23,804-character document | Medium (data, caused by verification) | noticed in the result line | restored in full from the identical text in the session transcript (exact prefix match); the script now saves and restores the raw clipboard (verified 23,804 characters) |
| 32 | Phone layout: a plain `1fr` grid column cannot shrink below its content | Low | headless Edge | `minmax(0,1fr)`, shrinkable rows; checked in a 375 px frame. (A first "overflow" screenshot was an artefact: headless Edge's minimum window is 504 px) |
| 33 | Web status showed the configured model name "default"; it was inserted into the page unescaped | Low | screenshot | served model name, escaped |
| 34 | Clone-check intelligibility test expected "seventy"; Whisper writes "70" | Low (test bug) | clone check | accept digits |

## 11. Risks and open issues

| ID | Risk / issue | Impact | Mitigation / next step |
| --- | --- | --- | --- |
| R1 | Driver/wheel drift: a future Adrenalin or ROCm wheel may change which HIP runtime works | Cloned voice stops | `tts_hip_runtime` switch; smoke §2 probes exactly the service's loading path |
| R2 | 9070 XT shared between the voice model (~5–6 GB) and the Dual_gpu 27B Q3 lane (~14 GB) | Paging when both active: slow first requests | Keep that lane unloaded while talking, or use a smaller model there; the assistant can unload it by voice (destructive, asks) |
| R3 | Answer audio 2.9–3.6 s vs 1–2 s | Conversational feel | Needs a TTS that streams within a sentence; compare Qwen3-TTS / Chatterbox behind the same `/audio/speech` contract (M1 comparison still open) |
| R4 | No echo cancellation | Self-hearing on speakers | Headset (A1) |
| R5 | Claim guard is a first-sentence regex | Unusual phrasing slips through | Verifier pass or code-generated confirmations |
| R6 | Small model on 26 tools | Occasional odd tool order | `scripts/agent_eval.py` before any model or tool change |
| R7 | Keep-warm keeps the GPUs from idling down | A few watts while listening | `keep_warm_s = 0` |
| R8 | Owner's voice not yet enrolled; listen/enroll not used by a person | Unknown UX defects | Owner steps in README |
| R9 | Reminder marked fired before it is spoken | Lost on a crash at that instant | Mark after speech |
| R10 | `listen` and `serve` keep separate histories | Confusing context | One supervisor process later |
| R11 | Coding agent's own permissions decide what it changes unasked | Unwanted edits | Configure OpenCode's permissions; it runs visibly and can be stopped |

## 12. Coverage of the original brief

| Feature | Pri | Status |
| --- | --- | --- |
| Voice activation | P1 | Done: wake phrase (spotted mid-utterance), aliases, push-to-talk hotkey |
| Conversational assistant | P1 | Done (Qwen3.5-4B, own server) |
| Reply in your cloned voice | P1 | **Done** on the GPU; *your* voice needs your enrollment |
| Open applications | P1 | Done, verified live |
| GPU monitoring | P1 | Done, live |
| File search | P1 | Done, live |
| Application control ("switch to the Cutline project") | P2 | Done: `open_project`, window list/focus/minimize/maximize/restore/close |
| Browser automation | P2 | Partial: open pages and web searches; no in-page clicking |
| Desktop screen understanding | P2 | Done (local vision model) |
| Coding-agent delegation | P2 | Done (OpenCode, visible terminal) |
| LLM workload control | P2 | Done (list, load, unload via Dual_gpu; destructive) — live load/unload not exercised |
| Local reminders, clipboard commands | P2 | Done |
| Personal knowledge | P3 | Done (notes full-text search) |
| Voice-controlled automation | P3 | Done (workflows) |
| Android companion | P3 | Not built |

Performance objectives (brief §7): wake reaction met as mid-utterance acknowledgement / hotkey; first audible response 1.3–1.6 s (acknowledgement), answer 2.9–3.6 s; voice similarity 0.978 on a synthetic speaker (owner pending); interruption implemented; desktop operations confirmed from state where checkable; local processing met.

Security requirements (brief §8): consent, revoke, mute, neutral voice, no voice-as-authorization, **AI disclosure (guest mode, WAV labels)**: done.

Milestones: M1 "prove your voice clone": the clone works and is scored; the Qwen3-TTS vs Chatterbox comparison is still open (R3). M2 integration: done. M3 desktop actions: done. M4 autonomous workflows: workflows and coding-agent delegation done; multi-step autonomous planning beyond configured workflows not built.

## 13. Operations

**Run.** `scripts\start_all.ps1 [serve|listen]`, or separately `scripts\start_llm.ps1`, `scripts\start_tts.ps1`, `scripts\start.ps1 serve|listen`. `vox doctor` for reachability and environment problems.

**Release check.** `pytest` (116 pass) → `smoke_test.py` (0 FAIL) → `clone_check.py` (0 FAIL) → `agent_eval.py` after any prompt, tool or model change → `desktop_check.py` after tool changes (visible side effects, all undone).

**Recovery.** Bad voice version: `vox voices use <older id>`. Voice leak concern: `vox voices revoke <id>`. Runaway assistant: say "mute", or close the process. Cloned voice broken after a driver update: `VOXA_TTS_HIP_RUNTIME=bundled` to compare, then smoke §2.

**Change control.** New tools state their risk level and get a unit test and an eval case; prompt or guard changes are re-run against the live model.

## 14. Glossary

| Term | Meaning |
| --- | --- |
| Acknowledgement | Pre-synthesized "Let me check." played while a read tool runs |
| Barge-in | Speaking over the assistant to interrupt it |
| Clone / profile | Reference recording + transcript the TTS model conditions on; `name_vN` |
| Gate / guard | Approval decision for tool calls / false-claim check |
| Guest mode | Neutral voice plus a spoken AI notice, for when others can hear |
| Keep-warm / pre-warm | Tiny requests that keep the LLM and voice model resident and hot |
| Lane | A GPU slot managed by Dual_gpu_setup (`r9700`, `9070xt`) |
| RTF | Real-time factor: generation seconds ÷ audio seconds |
| Speculative ASR | Transcribing at 200 ms of silence, used if the utterance then ends |
| Spotter | Small Whisper reading the start of an utterance for the wake phrase |
