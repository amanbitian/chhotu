"""`vox` command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from .config import Settings
from .voices import VoiceError, VoiceProfiles


def _llm(settings: Settings):
    from .llm_client import ToolChoiceLLM

    return ToolChoiceLLM(settings.llm_base_url, settings.llm_model, temperature=0.3, timeout=90.0, first_token_timeout=60.0)


def _agent(settings: Settings, confirm):
    from .agent import Agent
    from .audit import AuditLog
    from .permissions import PermissionGate
    from .tasks import PanelTasks
    from .tools import build_registry

    audit = AuditLog(settings.audit_db)
    # Started by the control panel: coding tasks go to its Tasks panel. On its own: a visible terminal per task.
    panel = os.getenv("VOXA_PANEL_URL")
    registry, store = build_registry(settings, task_manager=PanelTasks(panel) if panel else None, audit=audit)
    return Agent(_llm(settings), registry, PermissionGate(confirm, audit, set(settings.auto_approve)), audit,
                 name=settings.assistant_name, gender=settings.hindi_self_gender), store


def _neutral_tts(settings: Settings):
    """The Windows voice, plus a way to speak Hindi neutrally (see NeutralVoiceTTS)."""
    from .tts_client import ClonedVoiceTTS
    from .tts_neutral import NeutralVoiceTTS

    hindi = None
    if settings.neutral_voice_profile and (settings.voices_dir / settings.neutral_voice_profile).is_dir():
        hindi = ClonedVoiceTTS(settings.tts_url, voice=settings.neutral_voice_profile, preset=settings.tts_preset)
    return NeutralVoiceTTS(hindi=hindi, hindi_voice=settings.neutral_hindi_voice or None)


def _asr(settings: Settings):
    """GPU recognition in the voice service (English + Hindi), the CPU recogniser as fallback (English)."""
    from .asr_local import LocalWhisperASR

    cpu = LocalWhisperASR(settings.whisper_model, settings.whisper_device, settings.whisper_compute, settings.whisper_threads)
    if settings.asr_backend != "service":
        return cpu
    from .asr_remote import ServiceASR

    return ServiceASR(settings.tts_url, fallback=cpu)


def cmd_devices(args, settings) -> int:
    from .audio_io import list_devices

    for d in list_devices():
        flags = ("in" if d["inputs"] else "") + ("/out" if d["outputs"] else "")
        mark = "*" if d["default_in"] or d["default_out"] else " "
        print(f"{mark} {d['index']:>3}  {d['name']}  [{d['hostapi']}, {flags.strip('/')}]")
    print("\nSet input_device / output_device in data/config.toml (index or name fragment).")
    return 0


def cmd_enroll(args, settings) -> int:
    from .enroll import run_interactive

    try:
        name = args.name or ("me-hi" if args.language == "hi" else "me")
        profile = run_interactive(VoiceProfiles(settings.voices_dir), name, settings.input_device, settings.silero_model,
                                  language=args.language)
    except VoiceError as exc:
        print(f"Not saved: {exc}")
        return 1
    print(f"Saved and activated {profile['id']} ({profile['quality']['seconds']:.0f} s, SNR {profile['quality']['snr_db']:.0f} dB).")
    for w in profile["quality"]["warnings"]:
        print(f"  note: {w}")
    print("Hear it with:  vox say \"Testing my cloned voice.\"")
    return 0


def cmd_voices(args, settings) -> int:
    profiles = VoiceProfiles(settings.voices_dir)
    if args.action == "check":
        try:
            r = httpx.post(f"{settings.tts_url}/voices/{args.id}/similarity", json={}, timeout=600.0)
        except httpx.HTTPError as exc:
            print(f"The voice service is not reachable ({type(exc).__name__}); start scripts\\start_tts.ps1.")
            return 1
        if r.status_code != 200:
            print(r.json().get("detail", r.text))
            return 1
        d = r.json()
        verdict = "sounds like the same speaker" if d["same_speaker"] else "below the same-speaker threshold"
        print(f"{d['voice']}: similarity {d['similarity']:.3f} (threshold {d['threshold']}) - {verdict}.")
        print("A score is a check, not a substitute for listening: try  vox say \"...\"")
        return 0 if d["same_speaker"] else 2
    try:
        if args.action == "use":
            profiles.set_active(args.id)
        elif args.action == "revoke":
            profiles.revoke(args.id)
            try:
                httpx.post(f"{settings.tts_url}/forget", json={}, timeout=3.0)
            except httpx.HTTPError:
                pass
            print(f"Deleted {args.id} and its recording.")
    except VoiceError as exc:
        print(exc)
        return 1
    active = profiles.active_id()
    for p in profiles.list():
        print(f"{'*' if p['id'] == active else ' '} {p['id']:<16} {p['quality']['seconds']:>5.0f} s  {p['source']}")
    if not profiles.list():
        print("No voice profiles. Run: vox enroll")
    return 0


def cmd_say(args, settings) -> int:
    text = " ".join(args.text)

    async def go() -> int:
        from .audio_io import Speaker
        from .tts_client import ClonedVoiceTTS
        from .tts_neutral import NeutralVoiceTTS

        tts = NeutralVoiceTTS() if args.neutral else ClonedVoiceTTS(settings.tts_url, preset=args.preset or settings.tts_preset)
        speaker = None if args.out else Speaker(settings.output_device)
        pcm = b""
        try:
            async for chunk in tts.synth(text):
                pcm += chunk
                if speaker:
                    speaker.play(chunk, tts.sample_rate)  # play while the rest is still being generated
        except (httpx.HTTPError, RuntimeError) as exc:
            print(f"Could not synthesize: {exc}\nIs the voice service running (scripts\\start_tts.ps1)? Use --neutral for the Windows voice.")
            return 1
        if args.out:
            from .disclosure import labelled_wav

            voice = "neutral" if args.neutral else (VoiceProfiles(settings.voices_dir).active_id() or "unknown")
            Path(args.out).write_bytes(labelled_wav(pcm, tts.sample_rate, voice))
            print(f"Wrote {args.out} (labelled as AI-generated speech)")
        else:
            await speaker.drain()
            speaker.close()
        return 0

    return asyncio.run(go())


def cmd_ask(args, settings) -> int:
    """One text request through the real agent, with terminal confirmations."""
    from .agent import Say, ToolFinished, ToolStarted

    async def confirm(intent, tool, a) -> bool:
        try:
            return (await asyncio.to_thread(input, f"  ? Allow: {intent}  [y/N] ")).strip().lower() in ("y", "yes")
        except EOFError:  # no terminal to ask on: decline
            return False

    from .language import reply_language

    agent, _ = _agent(settings, confirm)
    text = " ".join(args.text)

    async def go() -> int:
        async for ev in agent.run(text, reply_language(args.language or settings.language, text)):
            if isinstance(ev, Say):
                print(ev.text, end="", flush=True)
            elif isinstance(ev, ToolStarted):
                print(f"\n  [{ev.name} {json.dumps(ev.args)} ({ev.risk})]")
            elif isinstance(ev, ToolFinished):
                print(f"  [{ev.name}: {'ok' if ev.result.get('ok') else 'failed'} - {ev.decision}]")
        print()
        return 0

    return asyncio.run(go())


def cmd_llm_server(args, settings) -> int:
    """Run the assistant's own small LLM (llama-server) in the foreground."""
    from .llm_server import run

    return run(settings)


def cmd_serve(args, settings) -> int:
    from .webapp import main

    main(settings)
    return 0


def cmd_listen(args, settings) -> int:
    """Hands-free voice assistant on the default microphone and speakers."""
    from voxvortex.audio import SileroVAD

    from .asr_local import LocalWhisperASR
    from .audio_io import Microphone, Speaker
    from .tts_client import ClonedVoiceTTS
    from .tts_neutral import NeutralVoiceTTS
    from .voice import VoiceLoop

    colors = {"heard": "you", "say": None}

    def show(ev: dict) -> None:
        # The launcher consumes this machine-readable copy and mirrors accepted hands-free
        # turns into the Assistant panel. Keep the normal lines below useful to humans.
        if os.getenv("VOXA_EVENT_STREAM") == "1":
            print("@@VOXA_EVENT@@" + json.dumps(ev, ensure_ascii=False, default=str), flush=True)
        k = ev["kind"]
        if k == "heard":
            print(f"\nyou: {ev['text']}")
        elif k == "say":
            print(ev["text"], end="", flush=True)
        elif k == "tool":
            print(f"\n  [{ev['name']} {json.dumps(ev['args'])} ({ev['risk']})]")
        elif k == "tool_result":
            print(f"  [{ev['name']}: {'ok' if ev['ok'] else 'failed'} - {ev['decision']}]")
        elif k == "confirm_request":
            print(f"\n  ? waiting for yes/no: {ev['intent']}")
        elif k in ("interrupted", "muted", "unmuted", "reminder", "voice_mode", "guest_mode", "tts_fallback", "error"):
            print(f"\n  ({k}{': ' + str({a: b for a, b in ev.items() if a != 'kind'}) if len(ev) > 1 else ''})")
        elif k == "wake":
            print(f"\n  (listening - {ev['via']})")
        elif k == "language":
            print(f"\n  (language: {ev['mode']})")
        elif k == "ignored":
            print("\n  (heard, but ignored because the wake phrase was missing)")

    async def go() -> int:
        import threading

        from .hotkey import GlobalHotkey
        from .tools.knowledge import warm_index

        if args.language:
            settings.language = args.language
        threading.Thread(target=warm_index, args=(settings,), name="notes-index", daemon=True).start()
        agent, store = _agent(settings, None)
        asr = _asr(settings)
        spotter = LocalWhisperASR(settings.wake_spotter_model, settings.whisper_device, settings.whisper_compute, 4) \
            if settings.wake_spotter_model and not args.always else None
        print("Loading speech recognition…")
        await asyncio.gather(asr.warm(), *([spotter.warm()] if spotter else []))
        cloned = ClonedVoiceTTS(settings.tts_url, preset=settings.tts_preset)
        try:
            await cloned.health()
        except httpx.HTTPError:
            print("Cloned-voice service is not running; using the neutral voice. Start scripts\\start_tts.ps1 for your voice.")
            cloned = None
        loop = VoiceLoop(agent=agent, asr=asr, tts=cloned, neutral_tts=_neutral_tts(settings), speaker=Speaker(settings.output_device),
                         vad=SileroVAD(settings.silero_model), settings=settings, reminders=store, spotter=spotter,
                         mode="always" if args.always else "wake", end_silence_ms=settings.end_silence_ms,
                         on_event=show)
        hotkey = None
        if settings.hotkey:
            running = asyncio.get_running_loop()
            hotkey = GlobalHotkey(settings.hotkey, lambda: running.call_soon_threadsafe(loop.wake, "hotkey"))
            if not hotkey.start():
                print(f"Hotkey not available: {hotkey.error}")
                hotkey = None
        mic = Microphone(settings.input_device)
        mic.start()
        hint = "every utterance" if args.always else f"say \"{settings.wake_phrase}\" first"
        if hotkey:
            hint += f", or press {settings.hotkey}"
        print(f"Listening ({hint}). Use a headset so I don't hear myself. Ctrl+C to stop.")
        if loop.guest:
            print("Guest mode is on: neutral voice, and the first reply says it is an AI.")
        try:
            await loop.run(mic.frames())
        finally:
            mic.stop()
            loop.speaker.close()
            if hotkey:
                hotkey.stop()
        return 0

    try:
        return asyncio.run(go())
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


def cmd_doctor(args, settings) -> int:
    from .tools.system import summarize_gpus  # noqa: F401  (import check)

    problems = 0

    def line(ok: bool | None, label: str, detail: str = "") -> None:
        nonlocal problems
        problems += ok is False
        print(f"  [{'ok' if ok else ('--' if ok is None else 'XX')}] {label}{(' - ' + detail) if detail else ''}")

    print("VoxVortex Desktop checks")
    try:
        r = httpx.get(f"{settings.llm_base_url}/models", timeout=3.0)
        served = ", ".join(m.get("id", "?") for m in r.json().get("data", [])) if r.status_code == 200 else ""
        line(r.status_code == 200, f"LLM {settings.llm_base_url}", served)
    except httpx.HTTPError as exc:
        line(False, f"LLM {settings.llm_base_url}", f"{type(exc).__name__} - start it with: vox llm-server")
    try:
        hw = httpx.get(f"{settings.dual_gpu_url}/api/hardware", timeout=3.0).json()
        st = httpx.get(f"{settings.dual_gpu_url}/api/status", timeout=3.0).json()
        from .tools.system import summarize_gpus

        for g in summarize_gpus(hw, st, settings.gpu_temp_limit_c):
            line(True, f"GPU {g['gpu']}", f"{g['model'] or 'no model'}, {g['vram_used_gb']}/{g['vram_total_gb']} GB, {g['temp_junction_c']} C")
    except Exception as exc:
        line(False, "Dual GPU telemetry", type(exc).__name__)
    try:
        h = httpx.get(f"{settings.tts_url}/health", timeout=3.0).json()
        line(True, f"cloned-voice service {settings.tts_url}", "model loaded" if h["model_loaded"] else "up (model loads on first request)")
        voice_gpu = (h.get("runtime") or {}).get("gpu") or ""
        try:  # a Dual_gpu lane with a model on the same card: Windows pages one of them out of VRAM when both are used
            st = httpx.get(f"{settings.dual_gpu_url}/api/status", timeout=3.0).json()
            hw = httpx.get(f"{settings.dual_gpu_url}/api/hardware", timeout=3.0).json()
            from .tools.system import summarize_gpus

            shared = [g for g in summarize_gpus(hw, st, settings.gpu_temp_limit_c) if g["model"] and g["gpu"] == voice_gpu]
            if shared:
                line(None, "voice model shares a GPU", f"{voice_gpu} also runs {shared[0]['model']}: when both are busy "
                     "the voice is ~1-2 s slower and the lane can be left partly in system RAM (see README)")
        except Exception:
            pass
    except httpx.HTTPError:
        line(None, f"cloned-voice service {settings.tts_url}", "not running (scripts\\start_tts.ps1); neutral voice still works")
    active = VoiceProfiles(settings.voices_dir).active()
    line(active is not None, "active voice profile", active["id"] if active else "run: vox enroll")
    line(Path(settings.silero_model).is_file(), "Silero VAD model", str(settings.silero_model))
    try:
        from faster_whisper.utils import download_model

        download_model(settings.whisper_model, local_files_only=True)
        line(True, f"Whisper model {settings.whisper_model}", "cached")
    except Exception:
        line(None, f"Whisper model {settings.whisper_model}", "will download on first use")
    try:
        from .audio_io import list_devices

        d = list_devices()
        line(any(x["inputs"] for x in d), "microphone", f"{sum(1 for x in d if x['inputs'])} input device(s)")
    except Exception as exc:
        line(False, "audio devices", str(exc))
    import os
    import shutil

    agent = shutil.which(settings.coding_agent[0]) if settings.coding_agent else None
    line(agent is not None or None, f"coding agent `{settings.coding_agent[0] if settings.coding_agent else '-'}`",
         agent or "not on PATH; delegate_coding_task will say so")
    bad = [p for p in os.environ.get("PATH", "").split(";") if p.count('"') % 2]
    line(None if bad else True, "PATH entries", f"stray quote in {bad[0]!r}: cmd.exe cannot see later PATH entries (npm, node)"
         if bad else "no unbalanced quotes")
    if settings.hotkey:
        from .hotkey import GlobalHotkey

        hk = GlobalHotkey(settings.hotkey, lambda: None)
        free = hk.start()
        hk.stop()
        line(free or None, f"hotkey {settings.hotkey}", "free" if free else f"{hk.error} (set hotkey in data/config.toml)")
    print("\nAll required parts look fine." if not problems else f"\n{problems} problem(s) above.")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vox", description="VoxVortex Desktop: a local assistant in your own voice")
    parser.add_argument("--data-dir", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("devices", help="list audio devices").set_defaults(fn=cmd_devices)
    p = sub.add_parser("enroll", help="record a voice reference with a guided script")
    p.add_argument("--name", default=None, help="default: me (English), me-hi (Hindi)")
    p.add_argument("--language", choices=["en", "hi"], default="en",
                   help="the language you read in; a Hindi reference is used for Hindi speech")
    p.set_defaults(fn=cmd_enroll)
    p = sub.add_parser("voices", help="list / switch / delete / score voice profiles")
    p.add_argument("action", nargs="?", choices=["list", "use", "revoke", "check"], default="list")
    p.add_argument("id", nargs="?")
    p.set_defaults(fn=cmd_voices)
    p = sub.add_parser("say", help="speak text in the cloned voice")
    p.add_argument("text", nargs="+")
    p.add_argument("--neutral", action="store_true")
    p.add_argument("--preset", choices=["Draft", "Balanced", "Quality"])
    p.add_argument("--out", help="write a WAV instead of playing")
    p.set_defaults(fn=cmd_say)
    languages = ["auto", "en", "hi", "hinglish"]
    p = sub.add_parser("ask", help="send one typed request to the agent")
    p.add_argument("text", nargs="+")
    p.add_argument("--language", choices=languages, help="reply language (default: setting `language`)")
    p.set_defaults(fn=cmd_ask)
    sub.add_parser("serve", help="open the local web app").set_defaults(fn=cmd_serve)
    sub.add_parser("llm-server", help="run the assistant's own small LLM (llama-server)").set_defaults(fn=cmd_llm_server)
    p = sub.add_parser("listen", help="hands-free voice assistant")
    p.add_argument("--always", action="store_true", help="treat every utterance as a request (no wake phrase)")
    p.add_argument("--language", choices=languages, help="reply language to start with; switch any time by voice")
    p.set_defaults(fn=cmd_listen)
    sub.add_parser("doctor", help="check that every part is reachable").set_defaults(fn=cmd_doctor)
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # model text can hold characters the Windows console codepage cannot print
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if args.command == "voices" and args.action in ("use", "revoke", "check") and not args.id:
        parser.error(f"voices {args.action} needs a profile id")
    return args.fn(args, Settings.load(args.data_dir))


if __name__ == "__main__":
    sys.exit(main())
