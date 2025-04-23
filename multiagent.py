from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, TypeVar

from agents import Agent, Runner  # OpenAI Agents SDK
from agents.model_settings import ModelSettings
from pydantic import BaseModel, RootModel
from radon.complexity import cc_visit

# ---------------------------------------------------------------------------
# Configuration --------------------------------------------------------------
os.environ["OPENAI_API_KEY"] = "SET_YOUR_API_KEY_HERE"

SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"

COMPLEXITY_THRESHOLD = 10
MAX_REFAC_ITERATIONS = 3
MODEL_SETTINGS = ModelSettings(temperature=0.0)

# ---------------------------------------------------------------------------
# Logging --------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)-14s | %(message)s",
    datefmt="%H:%M:%S",
)
logger_main = logging.getLogger("main")
for noisy in ("httpx", "openai"):
  logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Pydantic schemas -----------------------------------------------------------
class FunctionInfo(BaseModel):
  name: str
  complexity: int
  snippet: str

class FunctionList(RootModel[List[FunctionInfo]]):
  pass

class QAResult(BaseModel):
  passed: bool
  feedback: str

# ---------------------------------------------------------------------------
# Utils ----------------------------------------------------------------------

def read_file(p: Path) -> str:
  return p.read_text("utf-8")

def write_file(p: Path, content: str):
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(content, "utf-8")

def detect_language(p: Path) -> str:
  return "python" if p.suffix.lower() == ".py" else p.suffix.lstrip(".")

def strip_code_fence(code: str) -> str:
  """Remove leading/trailing ``` blocks if the LLM wrapped its answer."""
  lines = code.strip().splitlines()
  if lines and lines[0].startswith("```") and lines[-1].startswith("```"):
    lines = lines[1:-1]
  return "\n".join(lines)

def cyclomatic_complexity(code: str) -> int | None:
  try:
    return cc_visit(code)[0].complexity
  except Exception:
    return None

# Generic wrapper so each LLM call logs once before/after --------------------
async def run_agent(agent: Agent, *, tag: str):
  log = logging.getLogger(tag)
  log.info("➜  requesting %s", agent.name.replace("-", " "))
  result = await Runner.run(agent, "")
  out = result.final_output
  log.info("✓  done (%s)", f"{len(out)} chars" if isinstance(out, (str, bytes)) else type(out).__name__)
  return out

# Helper to normalise output -------------------------------------------------
T = TypeVar("T")

def ensure_parsed(obj: Any, model: type[T]) -> T:
  if isinstance(obj, model):
    return obj
  if isinstance(obj, (str, bytes)):
    return model.model_validate_json(obj)
  return model.model_validate_json(json.dumps(obj))

# ---------------------------------------------------------------------------
# Agent builders -------------------------------------------------------------

def build_analysis_agent(code: str, language: str) -> Agent:
  prompt = (
    "You are a static code analysis agent.\n"
    "For every function in the {language} source below, output ONLY JSON conforming to the schema {{name, complexity, snippet}}.\n\n"
    "```{language}\n{code}\n```"
  ).format(language=language, code=code)
  return Agent("Code-Analysis-Agent", prompt, output_type=FunctionList, model_settings=MODEL_SETTINGS)

def build_refactor_agent(func_code: str, language: str, feedback: str) -> Agent:
  prompt = (
    "Refactor the following function so the McCabe complexity ≤ {thr}.\n"
    "Keep the public signature and behaviour unchanged.\n"
    "Return ONLY the new function body as code – no prose, no JSON.\n\n"
    "```{language}\n{code}\n```\n{feedback}"
  ).format(thr=COMPLEXITY_THRESHOLD, language=language, code=func_code, feedback=feedback)
  return Agent("Refactor-Agent", prompt, model_settings=MODEL_SETTINGS)

def build_qa_agent(orig: str, new: str, language: str) -> Agent:
  prompt = (
    "Compare the original and refactored versions.\n\n"
    "1. Are they functionally equivalent?\n"
    "2. Is the McCabe complexity of version B ≤ {thr}?\n\n"
    "Respond with JSON matching the schema {{ passed: bool, feedback: string }}.\n\n"
    "ORIGINAL:\n```{language}\n{orig}\n```\n\n"
    "VERSION B (refactored):\n```{language}\n{new}\n```"
  ).format(thr=COMPLEXITY_THRESHOLD, language=language, orig=orig, new=new)
  return Agent("QA-Agent", prompt, output_type=QAResult, model_settings=MODEL_SETTINGS)

# ---------------------------------------------------------------------------
# Core workflow --------------------------------------------------------------
async def analyse_functions(code: str, language: str) -> List[Dict[str, Any]]:
  if language == "python":
    return [
      {
        "name": f.name,
        "complexity": f.complexity,
        "snippet": "\n".join(code.splitlines()[f.lineno - 1 : f.endline]),
      }
      for f in cc_visit(code)
    ]
  parsed = ensure_parsed(
      await run_agent(build_analysis_agent(code, language), tag="Analysis-Agent"),
      FunctionList,
  )
  return [fi.model_dump() for fi in parsed]

async def refactor_function(func: Dict[str, Any], language: str) -> str:
  original, feedback = func["snippet"], ""
  for attempt in range(1, MAX_REFAC_ITERATIONS + 1):
    raw_code = await run_agent(build_refactor_agent(original, language, feedback), tag="Refactor-Agent")
    new_code = strip_code_fence(raw_code)

    qa = ensure_parsed(
        await run_agent(build_qa_agent(original, new_code, language), tag="QA-Agent"),
        QAResult,
    )

    if language == "python":
      new_cc = cyclomatic_complexity(new_code)
      if new_cc is None:
        qa.passed = False
        qa.feedback = "Parser failed – please output valid Python code."
      elif new_cc > COMPLEXITY_THRESHOLD:
        qa.passed = False
        qa.feedback = (
          f"Cyclomatic complexity still {new_cc} (> {COMPLEXITY_THRESHOLD}). Refactor further."
        )

    if qa.passed:
      return new_code

    feedback = qa.feedback or "Please try again."
    logging.getLogger("QA-Agent").info(
        "✖  QA failed – retrying (%d/%d) :: %s", attempt, MAX_REFAC_ITERATIONS, feedback
    )

  raise RuntimeError("QA failed after maximum iterations")

async def process_file(path: Path):
  rel = path.relative_to(INPUT_DIR)
  logger_main.info("Processing %s", rel)
  language, source = detect_language(path), read_file(path)
  functions = await analyse_functions(source, language)
  todo = [f for f in functions if f["complexity"] > COMPLEXITY_THRESHOLD]
  logger_main.info("%d/%d functions exceed threshold", len(todo), len(functions))

  if not todo:
    dest = OUTPUT_DIR / rel
    write_file(dest, source)
    logger_main.info("Copied unchanged → %s", dest)
    return

  new_source = source
  for func in todo:
    try:
      new_code = await refactor_function(func, language)
      new_source = new_source.replace(func["snippet"], new_code)
    except Exception as e:
      logger_main.warning("Skipping %s: %s", func["name"], e)

  dest = OUTPUT_DIR / rel
  write_file(dest, new_source)
  logger_main.info("Written → %s", dest)

async def main():
  if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY environment variable not set")
  if not INPUT_DIR.exists():
    raise SystemExit(f"Input directory not found: {INPUT_DIR}")
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  files = [p for p in INPUT_DIR.rglob("*.*") if p.is_file()]
  if not files:
    logger_main.warning("No input files found in %s", INPUT_DIR)
    return
  await asyncio.gather(*(process_file(f) for f in files))
  logger_main.info("All done → %s", OUTPUT_DIR)

if __name__ == "__main__":
  asyncio.run(main())
