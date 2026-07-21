# Distilled model download

The trained QLoRA adapter (~154MB packaged) is not stored in this directory.

## Canonical location: Hugging Face Hub

https://huggingface.co/ceferra/qwen2.5-7b-adele-annotator

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "ceferra/qwen2.5-7b-adele-annotator")
tok = AutoTokenizer.from_pretrained("ceferra/qwen2.5-7b-adele-annotator")
```

## Mirror: GitHub Release

Also available as a GitHub Release asset (useful if you don't want a
Hugging Face dependency), on this repository's Releases page:
https://github.com/ceferra/adele-annotation-distillation/releases

Asset: `distilled-qwen2.5-7b-adele-lora.tar.gz`

Contents (PEFT LoRA adapter for `Qwen/Qwen2.5-7B-Instruct`):
- `adapter_config.json`, `adapter_model.safetensors` — the LoRA weights
- `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`,
  `added_tokens.json`, `special_tokens_map.json` — tokenizer files
- `README.md` — model card (training data, hyperparameters, evaluation)

See the main [README](../README.md#distilled-model) for loading instructions.
