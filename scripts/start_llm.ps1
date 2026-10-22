# The assistant's own small LLM (llama-server, Qwen3.5-4B + vision projector on the R9700, port 8083).
# Settings: llama_server, llm_gguf, llm_mmproj, llm_gpu, llm_ctx, llm_base_url in data/config.toml.
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $root '.venv\Scripts\python.exe') -m vox_assistant.cli llm-server
exit $LASTEXITCODE
