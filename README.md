# VoxVortex Desktop

A local voice assistant for this PC, in **English, हिन्दी and Hinglish**. It listens, thinks with a local Qwen, acts on the desktop **only with your permission**, and answers in a clone of your own voice, in whichever language you choose. Nothing leaves the machine.

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
| Voice cloning and speech recognition | VoxCPM2 and Whisper large-v3-turbo in the `F:\Projects\Audio Clone` environment, on the RX 9070 XT, wrapped by `tts_service.py` (the "voice service") |
| GPU telemetry and model management | `F:\Projects\Dual_gpu_setup` (your 27B lanes keep running untouched) |

## Status

Everything works end to end on this machine in English, Hindi and Hinglish, **including the cloned voice**, which streams (fixed: see [what was wrong](#what-was-wrong-with-the-cloned-voice)). 157 automated tests pass; the real-component smoke test reports **40 PASS, 0 FAIL, 1 WARN**; the clone check **13/13**; the live desktop check **17/17**; the agent eval **68/68** with no false claims and no reply in the wrong language.

What only you can do: **enroll your own voice** (`vox enroll`, and optionally `vox enroll --language hi`) and try `vox listen` with a headset, in Hindi too. Until you enroll, the active voice is `demo_v1`, Audio Clone's synthetic demo voice. All Hindi measurements so far used synthetic speakers with English accents; your own Hindi is the real test.

## Quick start

```powershell
cd F:\Projects\VoxVortex_pc_assistant
.\scripts\start_all.ps1            # voice LLM + voice service in their own windows, then the web app (http://127.0.0.1:8765)
.\scripts\start_all.ps1 listen     # same, then hands-free mode
.\.venv\Scripts\vox.exe doctor     # what is reachable, and environment problems
```

1. **Enroll your voice** (microphone, quiet room, headset): `vox enroll`. Read 4–8 prompted lines; each take is checked for clipping, noise and length; the good ones become a 20–36 s reference with an exact transcript. Result: `me_v1` (enrolling again makes `me_v2`; `vox voices use me_v1` reverts). Optional: `vox enroll --language hi` records `me-hi_v1` from eight Hindi lines; it is then used for Hindi speech, your English one for the rest. Not required: a clone from an English recording already speaks Hindi as the same person (measured 0.972 similarity).
2. **Score it**: `vox voices check me_v1` clones a test sentence and compares it with your recording (speaker-verification score; 0.86 or more = same speaker). Then listen: `vox say "Testing my cloned voice."` and `vox say "आज मौसम बहुत अच्छा है।"`.
3. **Talk to it**: `vox listen`, then say "hey vox, …" in any of the three languages (a chime means it heard you, often before you finish), or press **Ctrl+Alt+J** and just talk. `--always` skips the wake phrase; `--language hi` starts in Hindi.
4. **Type to it**: the web app (with a language selector), or `vox ask "..."` / `vox ask --language hinglish "..."`.

## English, Hindi, Hinglish

Speak or type in any of the three, in any mode; the mode decides how it **answers**:

| Mode | Answers in | Switch to it by saying |
| --- | --- | --- |
| Auto (default) | the language you just used | "match my language" |
| English | English | "speak English", "English mein baat karo" |
| Hindi | हिन्दी (technical words like GPU, RAM, Notepad stay in English letters) | "Hindi mein baat karo", "हिंदी में बोलो" |
| Hinglish | a natural mix | "Hinglish mein baat karo" |

The switch is instant and handled without the LLM. Confirmations work in Hindi too: "हाँ", "जी", "haan", "ठीक है" approve; "नहीं", "ना", "मत करो", "nahi", "rehne do" refuse, and a "no" anywhere in the answer wins. Mute ("चुप हो जाओ", "mute karo"), guest mode ("मेहमान आए हैं") and the assistant's own phrases ("जी?", "एक पल रुकिए।") exist in all three languages. Numbers are spoken the way each language says them (52 → बावन in Hindi). In Hindi the assistant avoids gendered verbs for itself; set `hindi_self_gender = "male"` or `"female"` in `data/config.toml` to match your voice.

How it hears both languages: every utterance is transcribed on the GPU twice, as Hindi and as English. If it was clearly English, the assistant sees the English text; otherwise it sees both and works out what you meant. Measured with synthetic speakers: English 18/18 requests understood, Hindi 15/18, Hinglish 14/18 (the misses were accent-garbled words), in 0.4–0.5 s. Details: [architecture §19](architecture.md#19-english-hindi-and-hinglish).

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

Measured on this PC, from the moment you stop talking until the first audible sample of the answer, in the cloned voice, with your 9070 XT lane idle:

| | |
| --- | --- |
| A plain question ("what's the capital of France?") | **0.93–1.04 s** |
| A Hindi question that needs a tool ("मेरे ग्राफ़िक्स कार्ड का तापमान क्या है?"), answered in Hindi | **1.62–1.67 s** |
| An English question that needs two tools ("which models are running and are the temperatures safe?") | **1.82–2.14 s**; "Let me check" at ~1.1 s (the first build took 6.7–18 s) |
| Wake chime | while you are still speaking; hotkey < 1 ms |

All within (or, for two tools, at the edge of) the brief's 1–2 s. The voice streams (first audio ~0.1 s after the text), and speech recognition runs on the GPU (transcript ~0.6 s after you stop, including the 0.45 s wait to be sure you have finished). [architecture §16](architecture.md#16-performance) lists every optimisation and its measured effect.

### Sharing the GPUs

The voice model (~5.2 GB) and speech recognition (~1.7 GB) run on the 9070 XT, which also hosts your Dual_gpu 27B Q3 lane. Measured while that lane was generating: answers start **1–2 s later** (2.4–3.3 s plain, 3.3–4.3 s with tools), and when the 27B paged back in next to the voice model it was left with **~8 GB in system RAM, running at ~6 tokens/s**, and stayed that way after the voice model stopped, until the lane is reloaded. `vox doctor` warns about this. A GPU-priority setting for the voice was tried and made no difference. Options, your call:

1. **Leave as is**: fine when you are not using the 9070 XT lane at the same time; reload that lane from the Dual_gpu dashboard if it has become slow.
2. **Unload the 9070 XT lane while talking**: "hey vox, unload the model on the 9070" (it asks first), or from the dashboard.
3. **Give the R9700 lane a smaller context** (it is set to 151,552 tokens × 8 slots) so the voice LLM and the voice model (~11.6 GB together) both fit on the R9700, and set `tts_device = "r9700"`. The 9070 XT is then yours alone.

## Configuration

Defaults live in `src/vox_assistant/config.py`; override in `data/config.toml` or with `VOXA_<NAME>` environment variables:

```toml
language = "auto"                 # auto | en | hi | hinglish: how it answers (it understands all three in every mode)
hindi_self_gender = "neutral"     # neutral | male | female: Hindi verb forms the assistant uses for itself
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

The voice LLM: `llm_gguf`, `llm_mmproj`, `llm_gpu` (`ROCm0` = R9700), `llm_base_url` (`:8083`). To use a 27B lane instead, set `llm_base_url = "http://127.0.0.1:8081/v1"` (slower; see `scripts/agent_eval.py`). The voice model's GPU: `tts_device = "9070"` (a name fragment, `cuda:N`, or `cpu`); speech recognition follows it unless `asr_device` is set; `asr_backend = "cpu"` uses the English-only CPU recogniser instead. The neutral voice speaks Hindi through the voice service with the synthetic `demo_v1` reference; if you install the Windows Hindi speech pack, set `neutral_hindi_voice = "Microsoft Kalpana Desktop"` (or Hemant) to use it.

## Checks

| Command | What it proves |
| --- | --- |
| `.venv\Scripts\python.exe -m pytest` | 157 unit tests, no hardware needed |
| `.venv\Scripts\python.exe scripts\smoke_test.py` | Real LLM, telemetry, mic and speaker, VAD, GPU Whisper (and the CPU fallback), the voice loop in real time in the cloned voice (latency timelines), a spoken Hindi request answered in Hindi, switching language by voice, permission by voice, mute, web app incl. streamed page speech, clone round trip. Silent: a Windows voice (and, for Hindi, the voice service) plays the person and speaker output is captured. `--quick` for the voice-loop parts only |
| `scripts\clone_check.py` | Enroll (synthetic speaker through the real recording path) → clone → intelligibility → similarity in English and in Hindi → revoke |
| `scripts\language_eval.py` | Spoken English / Hindi / Hinglish requests → recognised → wake phrase → understood by the agent |
| `scripts\desktop_check.py [--with-agent]` | Every desktop tool live. Visible: opens and closes Notepad, opens one browser tab, brings VS Code forward; restores your clipboard |
| `scripts\agent_eval.py [--url …]` | Tool choice, permission, false-claim and reply-language behaviour of a model (English, Hindi, Hinglish), and its speed |

## Verified, and not

**Verified on this machine (2026-10-22):** all of the above. Highlights: clone of a synthetic speaker scores 0.977 against its own reference and 0.467 against a different one, and **0.972 when speaking Hindi** from an English recording; Hindi spoken requests understood 15/18 and answered in Hindi; Notepad opened, minimized, maximized, restored, focused and closed; the coding agent started in the right folder; screen descriptions from the local vision model; web page checked in headless Edge at desktop and 375 px phone width.

**Not verified:**

- **Your voice**: not enrolled yet; similarity and a blind listening test need you.
- **Native Hindi and Hinglish speech**: every Hindi test used synthetic speakers (VoxCPM2 clones of English voices, with an English accent); your own Hindi should be recognised better, but that is unmeasured.
- **A person at the microphone**: `vox enroll` and `vox listen` ran only with synthetic speech fed through the real code paths. Echo and barge-in on real speakers are untested (use a headset).
- **Live `load_model` / `unload_model`**: tested against a mock Dual_gpu and the live catalog, not run against your lanes (it would replace their configuration).
- **Streamed speech in the web page, by ear**: the endpoint is verified live (first bytes 0.08 s) and the page script is syntax-checked, but nobody has listened to it in a browser.
- Android companion (not built); clicking inside web pages (not built).

### What was wrong with the cloned voice

Not the iGPU. The ROCm PyTorch wheel in Audio Clone bundles HIP runtime 10.0.3581, which crashes with your Adrenalin 32.0.31036 driver (runtime 10.0.3679). The voice service now loads the driver's runtime first and turns TorchScript off (its kernel compiler had the same mismatch). Audio Clone and the driver are unchanged. Further fixes made it fast: AMD's MIOpen convolution library is switched off (it crashed on streaming and spent 10–25 s tuning every new sentence length; PyTorch's own convolutions are as fast), which let the service use VoxCPM2's built-in streaming (first audio 0.12 s instead of ~0.8–2.6 s), and it runs on the 9070 XT (the R9700 cannot hold your 27B, the 4B and VoxCPM2 together; Windows was paging models out).

### Things on this PC you may want to fix

- **Your PATH has a stray `"`** at the end of `C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit"`. cmd.exe cannot see any PATH entry after it (Node.js, npm, so `opencode`, `npx`, `node` fail from cmd). The assistant works around it; `vox doctor` reports it.
- **Ctrl+Alt+Space is taken by another program**, which is why the hotkey is Ctrl+Alt+J.
- The 9070 XT is shared between the voice model and your Dual_gpu 27B Q3 lane: see [Sharing the GPUs](#sharing-the-gpus). **Right now that lane has ~8 GB in system RAM and runs at ~6 tokens/s** (it paged back in during my contention test while the voice model held VRAM); reloading it from the Dual_gpu dashboard restores it.

## Layout

```
src/vox_assistant/  config voices enroll tts_service voice_similarity disclosure tts_client tts_neutral speech_text
                    asr_gpu asr_remote asr_local language audio_io hotkey llm_client llm_server permissions audit agent
                    voice webapp security cli
                    tools/{system,models,files,knowledge,apps,projects,windows,screen,clipboard,reminders,workflows}
web/index.html      local web UI
scripts/            start.ps1 start_all.ps1 start_llm.ps1 start_tts.ps1
                    smoke_test.py clone_check.py desktop_check.py agent_eval.py language_eval.py
tests/              157 tests, no hardware      data/   (git-ignored) voices, audit log, reminders, config.toml
architecture.md  sdd.md
```

## Next

1. Enroll your voice (and optionally a Hindi reference), run `vox voices check`, and use `vox listen` for a day in English and Hindi; note what feels wrong, especially Hindi words it mishears.
2. Decide how the voice should share the GPUs with your lanes ([Sharing the GPUs](#sharing-the-gpus)).
3. Optional: compare Qwen3-TTS and Chatterbox on clone quality (latency no longer needs it).
4. In-page browser automation, an Android companion, Ubuntu support.
