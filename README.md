# Refactor‑Agent SDK Demo

This repository shows how to build a **multi‑agent workflow** with the new
**OpenAI Agents SDK** to automatically refactor functions that exceed a given
McCabe complexity threshold.

## How it works

| Step | Agent | Purpose |
|------|-------|---------|
| 1 | **Code‑Analysis‑Agent** | Parses each source file (or asks the LLM for non‑Python) and reports complexity per function |
| 2 | **Refactor‑Agent** | Rewrites one complex function at a time, keeping the public API |
| 3 | **QA‑Agent** | Verifies functional equivalence and checks whether complexity ≤ threshold |
| 4 | **Local Radon Check** | Guarantees the threshold is met (Python only) even if QA is overly optimistic |

The loop between Refactor and QA continues until the function passes or
`MAX_REFAC_ITERATIONS` is reached.

## Quick start

```bash
# 1. Install deps (prefer a virtual environment)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set your API key (or copy .env.example → .env)
export OPENAI_API_KEY=sk-...

# 3. Add files to the input/ folder and run
python refactor_agent_sdk.py
```

Refactored files are written to **output/**, preserving the original subtree.

## Configuration

Most knobs live in *refactor_agent_sdk.py* but can be overridden via env vars
(see *.env.example*):

* `COMPLEXITY_THRESHOLD` – target McCabe complexity (default 10)
* `MAX_REFAC_ITERATIONS` – retries per function (default 3)
* `LOG_LEVEL` – INFO by default; set `DEBUG` for verbose agent traffic

## Tested with

* Python 3.11
* openai ≥ 1.24 & openai‑agents ≥ 0.0.12

Feel free to open issues or suggestions! 🎉

