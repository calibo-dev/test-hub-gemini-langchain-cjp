from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

from latest_ai_development.config.settings import get_settings
from latest_ai_development.config.validators import validate_configuration
from latest_ai_development.tracking import initialize_langsmith_tracing
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow

# Load settings
settings = get_settings()

# Validate configuration at startup
validate_configuration()

# Initialize workflow controller
workflow = LatestAiDevelopmentWorkflow()

API_ROOT_PATH = settings.normalized_api_root_path


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_langsmith_tracing()
    yield


# Initialize FastAPI application
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="LangChain workflow for automated research and report generation",
    root_path=API_ROOT_PATH,
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    topic: str


@app.get("/health")
def health() -> dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, str]:
    """
    Execute the research + reporting workflow.
    """

    inputs = {
        "topic": request.topic,
        "current_year": datetime.now().year,
        "knowledge_context": "",
    }

    result = workflow.kickoff(inputs)

    return {
        "topic": result.get("topic"),
        "report": result.get("report"),
    }


if API_ROOT_PATH:
    app.add_api_route(
        f"{API_ROOT_PATH}/health",
        health,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_api_route(
        f"{API_ROOT_PATH}/ask",
        ask,
        methods=["POST"],
        include_in_schema=False,
    )


# CLI-compatible entrypoint
def run() -> None:
    import uvicorn

    uvicorn.run(
        "latest_ai_development.main:app",
        host=settings.api_host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
