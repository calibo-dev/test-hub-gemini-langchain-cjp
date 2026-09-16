from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from latest_ai_development.config.settings import get_settings
from latest_ai_development.config.validators import validate_configuration
from latest_ai_development.logging_utils import configure_logging
from latest_ai_development.tracing import initialize_langsmith_tracing
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow

# Load settings
settings = get_settings()
configure_logging(getattr(settings, "log_level", "INFO"))
logger = logging.getLogger(__name__)

# Validate configuration at startup
try:
    validate_configuration()
except Exception:
    logger.exception("Startup configuration validation failed")
    raise

# Initialize workflow controller
try:
    workflow = LatestAiDevelopmentWorkflow()
except Exception:
    logger.exception("Workflow initialization failed")
    raise

API_ROOT_PATH = settings.normalized_api_root_path


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_langsmith_tracing()
    yield


# Initialize FastAPI application
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="LangChain workflow for metadata-driven Snowflake ak_country queries",
    root_path=API_ROOT_PATH,
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    user_query: str


@app.get("/health")
def health() -> dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, str]:
    """
    Execute the ak_country metadata-driven query workflow.
    """
    flow_run_id = uuid4().hex[:12]

    logger.info(
        "Request started | flow_run_id=%s | user_query=%s",
        flow_run_id,
        request.user_query,
    )

    inputs = {
        "user_query": request.user_query,
        "current_year": datetime.now().year,
        "knowledge_context": "",
    }

    try:
        try:
            result = workflow.kickoff(inputs, flow_run_id=flow_run_id)
        except TypeError as exc:
            if "unexpected keyword argument 'flow_run_id'" not in str(exc):
                raise
            result = workflow.kickoff(inputs)
    except Exception as exc:
        logger.exception(
            "Request failed | flow_run_id=%s | user_query=%s",
            flow_run_id,
            request.user_query,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed. flow_run_id={flow_run_id}",
        ) from exc

    logger.info(
        "Request completed | flow_run_id=%s | user_query=%s",
        flow_run_id,
        request.user_query,
    )

    return {
        "user_query": request.user_query,
        "response": result.get("response"),
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
        log_level=str(getattr(settings, "log_level", "INFO")).lower(),
    )


if __name__ == "__main__":
    run()
