from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from latest_ai_development.config.settings import get_settings
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow, WorkflowResult


app = FastAPI(
    title="Latest AI Development",
    version="0.1.0",
    description="LangChain production template with two-agent sequential orchestration",
    root_path="/testing",
)


class TopicRequest(BaseModel):
    topic: str = Field(..., description="Topic to research and convert into a markdown report.")
    current_year: int | None = Field(default=None, description="Optional year context.")
    save_output: bool = Field(default=True, description="Whether to save the final report.")


class TopicResponse(BaseModel):
    topic: str
    current_year: int
    research_notes: str
    final_report: str
    output_file: str | None = None


def _default_year() -> int:
    return datetime.now(timezone.utc).year


def _execute(inputs: dict[str, Any]) -> WorkflowResult:
    orchestration = LatestAiDevelopmentWorkflow()
    return orchestration.kickoff(inputs=inputs)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/ask", response_model=TopicResponse)
def ask(payload: TopicRequest) -> TopicResponse:
    try:
        inputs = payload.model_dump()
        if not inputs.get("current_year"):
            inputs["current_year"] = _default_year()

        result = _execute(inputs)
        return TopicResponse(**result.to_dict())

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run() -> None:
    """
    Direct execution entrypoint for the two-agent orchestration.
    """
    parser = argparse.ArgumentParser(description="Run the agent orchestration directly.")
    parser.add_argument(
        "topic",
        nargs="?",
        default="Clawdbot",
        help="Topic to research and convert into a markdown report.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=_default_year(),
        help="Year context for the orchestration.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without saving report.md",
    )
    args = parser.parse_args()

    result = _execute(
        {
            "topic": args.topic,
            "current_year": args.year,
            "save_output": not args.no_save,
        }
    )
    print(result.final_report)


def train() -> None:
    """
    Lightweight repeated execution utility.
    Runs the orchestration multiple times and stores a JSON summary.
    """
    parser = argparse.ArgumentParser(description="Run multiple orchestration iterations.")
    parser.add_argument("n_iterations", type=int, help="Number of orchestration runs.")
    parser.add_argument("filename", type=str, help="File to save the summary.")
    parser.add_argument(
        "--topic",
        default="AI Agents",
        help="Topic used during repeated runs.",
    )
    args = parser.parse_args()

    runs: list[dict[str, Any]] = []
    for index in range(args.n_iterations):
        result = _execute(
            {
                "topic": args.topic,
                "current_year": _default_year(),
                "save_output": True,
            }
        )
        runs.append(
            {
                "iteration": index + 1,
                "topic": result.topic,
                "output_file": result.output_file,
                "research_notes_present": bool(result.research_notes.strip()),
                "final_report_present": bool(result.final_report.strip()),
            }
        )

    output_path = Path(args.filename)
    output_path.write_text(
        json.dumps({"runs": runs}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Training summary saved to {output_path}")


def replay() -> None:
    """
    Replay-style rerun utility.
    Accepts a JSON payload file path or a raw topic string.
    """
    parser = argparse.ArgumentParser(description="Replay an orchestration run.")
    parser.add_argument("task_id", type=str, help="Payload file path or raw topic.")
    args = parser.parse_args()

    candidate_path = Path(args.task_id)
    if candidate_path.exists() and candidate_path.is_file():
        inputs = json.loads(candidate_path.read_text(encoding="utf-8"))
        if not isinstance(inputs, dict):
            raise ValueError("Replay payload file must contain a JSON object.")
    else:
        inputs = {
            "topic": args.task_id,
            "current_year": _default_year(),
            "save_output": True,
        }

    result = _execute(inputs)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def test() -> None:
    """
    Lightweight orchestration test utility.
    Preserves the command shape while validating execution.
    """
    parser = argparse.ArgumentParser(description="Run a lightweight orchestration test.")
    parser.add_argument("n_iterations", type=int, help="Number of test iterations.")
    parser.add_argument("eval_llm", type=str, help="Evaluation model label for reporting.")
    parser.add_argument(
        "--topic",
        default="AI Agents",
        help="Topic used during test runs.",
    )
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for index in range(args.n_iterations):
        result = _execute(
            {
                "topic": args.topic,
                "current_year": _default_year(),
                "save_output": False,
            }
        )
        summaries.append(
            {
                "iteration": index + 1,
                "eval_llm": args.eval_llm,
                "topic": result.topic,
                "research_notes_present": bool(result.research_notes.strip()),
                "final_report_present": bool(result.final_report.strip()),
            }
        )

    print(json.dumps({"tests": summaries}, indent=2, ensure_ascii=False))


def run_with_trigger() -> None:
    """
    Execute the orchestration from a JSON trigger payload string.
    """
    parser = argparse.ArgumentParser(description="Run orchestration from JSON trigger payload.")
    parser.add_argument("payload", type=str, help="JSON payload string.")
    args = parser.parse_args()

    try:
        inputs = json.loads(args.payload)
    except json.JSONDecodeError as exc:
        raise ValueError("run_with_trigger expects a valid JSON object string.") from exc

    if not isinstance(inputs, dict):
        raise ValueError("Trigger payload must be a JSON object.")

    if "topic" not in inputs:
        for candidate in ("subject", "query", "prompt"):
            value = inputs.get(candidate)
            if value:
                inputs["topic"] = value
                break

    if not inputs.get("topic"):
        raise ValueError("Trigger payload must contain 'topic' or an equivalent field.")

    if "current_year" not in inputs:
        inputs["current_year"] = _default_year()

    if "save_output" not in inputs:
        inputs["save_output"] = True

    result = _execute(inputs)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "latest_ai_development.main:app",
        host=settings.api_host,
        port=settings.port,
        reload=False,
    )