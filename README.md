# deepset-llamastack
## How to use
1. Open vllm Server `vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --tensor-parallel-size  8      --trust-remote-code     --gpu-memory-utilization 0.92        --max-model-len 40000 --enable-auto-tool-choice --tool-call-parser llama4_pythonic  --max-num-seqs 2 --enforce-eager`
2. `docker compose --file deepset.yaml up` will start the llama-stack server with demo UI.
3. Click the demo UI link, eg.https://ef4594a70e32d08986.gradio.live
4. Connect the llama-stack server to Haystack in the webUI, in this case, the model name is `vllm/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`.
