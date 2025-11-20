# ToM
Official repository of paper *ToM: Leveraging Tree-oriented MapReduce for Long-Context Reasoning in Large Language Models*
# ToM Project

End-to-end pipeline for processing long documents, structuring them with LLMs, reasoning over the resulting trees, and evaluating QA outputs.

- `HSP/`: text chunking, DeepSeek API calls to build hierarchical structures, and a RAPTOR-style regrouping pass.
- `Reasoning/`: asynchronous MapReduce reasoning over the tree structure.
- `Evaluate/`: simple F1 calculation script.

## Project Layout

```
ToM/
├── HSP/
│   ├── main.py          # chunking + similarity filtering + DeepSeek inference + tree merge
│   ├── raptor.py        # RAPTOR clustering and tree rebuilding
│   ├── util.py          # helpers for chunking, parsing, merging
│   └── prompts.py       # prompt templates for structuring
├── Reasoning/
│   ├── api.py           # DeepSeek / GPT-4o / qwq API wrappers
│   ├── pipeline.py      # async MapReduce pipeline
│   ├── prompts.py       # map/reduce prompts
│   └── main.py          # batch processing entry point
├── Evaluate/
│   └── caculate_f1.py   # QA F1 scoring
└── requirements.txt
```

## Environment

1. Python ≥ 3.9.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Export the DeepSeek API key (used by both HSP and Reasoning):

```bash
export DEEPSEEK_API_KEY="your_key"
```

`Reasoning/api.py` also includes optional integrations for GPT-4o (`call_gpt4`) and `qwq`; configure their credentials as needed.

## HSP Module

### 1. Chunking + Structuring (`HSP/main.py`)

- `split_text_into_token_blocks` (in `util.py`) uses `tiktoken` to split text by token count.
- `filter_blocks_by_similarity` filters blocks via BGE embeddings + cosine similarity.
- `run_inference` runs multi-threaded DeepSeek calls using the prompt defined in `HSP/prompts.py`.

Before running, set:

```python
input_file = "<path to JSONL with context>"
output_dir = "<directory for outputs>"
top_k_values = [3]  # any list of k values you need
```

Then:

```bash
python HSP/main.py
```

Key outputs:
- `processed_results_topk_*.json`: merged trees for each top-k setting.
- `structured_blocks.json`, `debug_blocks_*.json`: intermediate data for debugging.

### 2. RAPTOR Clustering (`HSP/raptor.py`)

Consumes the trees produced above and performs bottom-up clustering to build higher-level summaries.

Set `input_file` / `output_file` near the bottom of the script and run:

```bash
python HSP/raptor.py
```

## Reasoning Module

`Reasoning/main.py` loads a tree dataset and runs asynchronous MapReduce reasoning.

```python
config = {
    'input_path': '<path to tree JSON>',
    'output_path': '<path to store predictions>',
    'model_path': None,          # remain None for API mode
    'max_workers': 4,
    'max_concurrent': 10
}
```

Run:

```bash
python Reasoning/main.py
```

How it works:
- `pipeline.py` traverses the tree level by level, spawning map tasks and reduce tasks via DeepSeek batch APIs.
- Each result entry contains `prediction` plus `structured_info` (key_info, reasoning_process, answer, confidence_score).

## Evaluation

`Evaluate/caculate_f1.py` computes F1 between `prediction` and ground-truth answers:

```bash
python Evaluate/caculate_f1.py
```

Set `file_path` to a JSON file containing entries like:

```json
{
  "question": "...",
  "answer": ["..."],
  "prediction": "..."
}
```

The script reports average F1 (ignoring blank predictions).

## Tips & Customization

- **Concurrency tuning**: adjust thread counts in `HSP/main.py` (`ThreadPoolExecutor`) and batch sizes in `Reasoning/pipeline.py` for your hardware.
- **Prompts**: edit `HSP/prompts.py` or `Reasoning/prompts.py` to change formatting requirements or reasoning style.
- **Retry/backoff**: `chat`, `run_inference`, and related helpers already include retry logic; tweak retry counts or delays if needed.

## Contributing

Feel free to fork and open pull requests. Common extension points:
- Add new API wrappers in `Reasoning/api.py`.
- Enhance evaluation scripts in `Evaluate/`.
- Improve chunking or tree-merging strategies in `HSP/util.py`.

Bug reports and feature requests are welcome through GitHub Issues.
