# Distilled model download

The trained QLoRA adapter (~154MB packaged) is too large for a normal git
commit (GitHub's 100MB per-file limit), so it is published as a
**GitHub Release asset** instead of being stored in this directory.

Download it from the repository's Releases page:
https://github.com/ceferra/adele-annotation-distillation/releases

Asset: `distilled-qwen2.5-7b-adele-lora.tar.gz`

Contents (PEFT LoRA adapter for `Qwen/Qwen2.5-7B-Instruct`):
- `adapter_config.json`, `adapter_model.safetensors` — the LoRA weights
- `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`,
  `added_tokens.json`, `special_tokens_map.json` — tokenizer files
- `README.md` — model card (training data, hyperparameters, evaluation)

See the main [README](../README.md#distilled-model) for loading instructions.
