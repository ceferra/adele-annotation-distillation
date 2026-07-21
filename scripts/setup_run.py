from delean_batch_manager import DeLeAnBatchManager
from delean_batch_manager.core.utils.clients import create_openai_client

# Dummy client: we never call OpenAI, only used locally to build input files
# and parse local vLLM output files, which don't need network access.
client = create_openai_client(api_key="unused-local-only")

manager = DeLeAnBatchManager(
    client=client,
    base_folder="/home/administrador/delean_run/batch_runs/algebra",
    source_data_path="/home/administrador/delean_run/algebra_prepared.csv",
    rubrics_folder="/tmp/delean-batch-manager/rubrics",
    openai_model="Qwen/Qwen2.5-72B-Instruct-AWQ",
    max_completion_tokens=2000,
)

# vLLM's offline run_batch tool only accepts this exact endpoint string
manager.endpoint = "/v1/chat/completions"

manager.create_input_files()
print("Input files created:", len(manager._input_files))
for f in manager._input_files:
    print(" -", f)
