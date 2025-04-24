#!/usr/bin/env python3
"""Parallel refactor pipeline built with **LangGraph**

Compared with *refactor_agent_sdk.py* (single‑thread loop) this version:

* maps the Refactor/QA sub‑graph **concurrently** across all files in *input/*
* collects the *per‑file* summaries into a shared **LangGraph state** without
  risking race conditions.

The trick: every branch writes its result to a **concurrent.Futures queue**
that is consumed by a *single* aggregator node – so the shared state is only
mutated in one place.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langgraph.graph import Graph
from langgraph.state import BaseState, Mutable
from openai import OpenAI
from pydantic import BaseModel
from radon.complexity import cc_visit

# ---------------------------------------------------------------------------
# Environment / logging ------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("graph")

if not (api_key := os.getenv("OPENAI_API_KEY")):
  raise SystemExit("OPENAI_API_KEY missing – see .env.example")

client = OpenAI(api_key=api_key)

# ---------------------------------------------------------------------------
# Parameters -----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"

COMPLEXITY_THRESHOLD = int(os.getenv("COMPLEXITY_THRESHOLD", 10))
MAX_REFAC_ITERATIONS = int(os.getenv("MAX_REFAC_ITERATIONS", 3))

# ---------------------------------------------------------------------------
# Data models ----------------------------------------------------------------
class FunctionInfo(BaseModel):
  name: str
  complexity: int
  snippet: str


@dataclass
class FileSummary:
  path: str
  refactored: int
  warnings: List[str] = field(default_factory=list)


class GlobalState(BaseState):
  """LangGraph shared state – only the aggregator mutates it."""

  summaries: Mutable[List[FileSummary]]


# ---------------------------------------------------------------------------
# Helper functions -----------------------------------------------------------

def analyse_python(code: str) -> List[FunctionInfo]:
  return [
    FunctionInfo(
        name=f.name,
        complexity=f.complexity,
        snippet="\n".join(code.splitlines()[f.lineno - 1 : f.endline]),
    )
    for f in cc_visit(code)
  ]


def cyclomatic_complexity(code: str) -> int | None:
  try:
    return cc_visit(code)[0].complexity
  except Exception:
    return None


def strip_fence(txt: str) -> str:
  lines = txt.strip().splitlines()
  if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
    lines = lines[1:-1]
  return "\n".join(lines)

# ---------------------------------------------------------------------------
# Agent wrappers (sync for brevity – ThreadPool takes care of I/O wait) ------

def gpt(prompt: str, *, sys: str = "You are an AI assistant.") -> str:
  resp = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
  )
  return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Graph nodes ----------------------------------------------------------------

def node_refactor(file_path: Path, func: FunctionInfo) -> str:
  """Try to refactor one function; may raise on failure."""
  prompt = (
    "Refactor the following Python function so its McCabe complexity ≤ {thr}. "
    "Keep the API identical. Return only code.\n\n```python\n{code}\n```"
  ).format(thr=COMPLEXITY_THRESHOLD, code=func.snippet)

  for _ in range(MAX_REFAC_ITERATIONS):
    new_code = strip_fence(gpt(prompt))
    if (cc := cyclomatic_complexity(new_code)) is not None and cc <= COMPLEXITY_THRESHOLD:
      return new_code
    prompt = (
      "The previous refactor still has complexity {cc} (> {thr}). "
      "Please refactor further. Return code only.\n\n```python\n{code}\n```"
    ).format(cc=cc or -1, thr=COMPLEXITY_THRESHOLD, code=new_code)
  raise RuntimeError("Refactor failed to reach threshold")


async def worker(file_path: Path, out_q: "queue.Queue[FileSummary]"):
  code = file_path.read_text("utf-8")
  funcs = analyse_python(code)
  new_source = code
  refactored = 0
  warnings: List[str] = []

  for func in funcs:
    if func.complexity <= COMPLEXITY_THRESHOLD:
      continue
    try:
      replacement = node_refactor(file_path, func)
      new_source = new_source.replace(func.snippet, replacement)
      refactored += 1
    except Exception as e:
      warnings.append(f"{func.name}: {e}")

  # write result file
  dest = OUTPUT_DIR / file_path.relative_to(INPUT_DIR)
  dest.parent.mkdir(parents=True, exist_ok=True)
  dest.write_text(new_source, "utf-8")

  out_q.put(FileSummary(path=str(file_path), refactored=refactored, warnings=warnings))


# ---------------------------------------------------------------------------
# LangGraph construction -----------------------------------------------------

def build_graph(files: List[Path]):
  g = Graph(state_type=GlobalState)
  summary_queue: "queue.Queue[FileSummary]" = queue.Queue()

  # Map step – run workers concurrently via ThreadPool
  async def map_node(state: GlobalState):  # type: ignore[override]
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
      await asyncio.gather(*(loop.run_in_executor(pool, worker, p, summary_queue) for p in files))
    return state

  # Reduce step – aggregate summaries (single‑thread, so no race)
  def reduce_node(state: GlobalState):  # type: ignore[override]
    while not summary_queue.empty():
      state.summaries.append(summary_queue.get())
    return state

  g.add_node("map", map_node)
  g.add_node("reduce", reduce_node)
  g.set_entry_point("map")
  g.set_finish_point("reduce")
  return g


# ---------------------------------------------------------------------------
# Main -----------------------------------------------------------------------
async def main():
  if not INPUT_DIR.exists():
    raise SystemExit("input/ folder missing")
  files = [p for p in INPUT_DIR.rglob("*.py") if p.is_file()]
  if not files:
    raise SystemExit("No Python files in input/")

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  graph = build_graph(files)
  final_state: GlobalState = await graph.run_async(initial_state={"summaries": []})

  # Pretty‑print summary
  for s in final_state.summaries:
    status = f"{s.refactored} refactored" if s.refactored else "none refactored"
    log.info("%s → %s", s.path, status)
    for w in s.warnings:
      log.warning("%s :: %s", s.path, w)

  log.info("Done. Output in %s", OUTPUT_DIR)


if __name__ == "__main__":
  asyncio.run(main())
