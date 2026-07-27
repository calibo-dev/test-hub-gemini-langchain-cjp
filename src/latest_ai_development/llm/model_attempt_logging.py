from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelFallbackMiddleware

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelAttempt:
    stage: str
    provider: str
    model_id: str


def model_attempt_from_section(stage: str, section_config: dict[str, Any]) -> ModelAttempt:
    return ModelAttempt(
        stage=stage,
        provider=str(section_config.get("provider", "")).strip(),
        model_id=str(section_config.get("modelId", "")).strip(),
    )


def get_model_attempts(
    stage: str,
    model_config: dict[str, Any],
) -> tuple[ModelAttempt, list[ModelAttempt]]:
    primary_config = model_config.get("primary", {}) if isinstance(model_config, dict) else {}
    fallback_config = model_config.get("fallback") if isinstance(model_config, dict) else None

    primary_attempt = model_attempt_from_section(stage, primary_config)
    fallback_attempts = [
        model_attempt_from_section(stage, fallback_config)
    ] if isinstance(fallback_config, dict) else []

    return primary_attempt, fallback_attempts


def with_model_attempt_logging(
    runnable: Any,
    attempt: ModelAttempt,
    *,
    retry: bool = False,
) -> Any:
    return runnable.with_listeners(
        on_start=lambda *_: log_model_retry(attempt) if retry else log_model_try(attempt),
        on_end=lambda *_: log_model_success(attempt),
        on_error=lambda run, *_: log_model_failure(attempt, getattr(run, "error", None)),
    )


class ModelAttemptLoggingMiddleware(AgentMiddleware):
    def __init__(self, primary_attempt: ModelAttempt) -> None:
        super().__init__()
        self.primary_attempt = primary_attempt

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        log_model_try(self.primary_attempt)
        try:
            response = handler(request)
        except Exception as exc:
            log_model_failure(self.primary_attempt, exc)
            raise

        log_model_success(self.primary_attempt)
        return response

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        log_model_try(self.primary_attempt)
        try:
            response = await handler(request)
        except Exception as exc:
            log_model_failure(self.primary_attempt, exc)
            raise

        log_model_success(self.primary_attempt)
        return response


class LoggedModelFallbackMiddleware(ModelFallbackMiddleware):
    def __init__(
        self,
        primary_attempt: ModelAttempt,
        fallback_attempts: list[ModelAttempt],
        first_model: Any,
        *additional_models: Any,
    ) -> None:
        super().__init__(first_model, *additional_models)
        self.primary_attempt = primary_attempt
        self.fallback_attempts = fallback_attempts

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        last_exception: Exception

        log_model_try(self.primary_attempt)
        try:
            response = handler(request)
        except Exception as exc:
            log_model_failure(self.primary_attempt, exc)
            last_exception = exc
        else:
            log_model_success(self.primary_attempt)
            return response

        for index, fallback_model in enumerate(self.models):
            attempt = self.fallback_attempts[index]
            log_model_retry(attempt)
            try:
                response = handler(request.override(model=fallback_model))
            except Exception as exc:
                log_model_failure(attempt, exc)
                last_exception = exc
                continue

            log_model_success(attempt)
            return response

        raise last_exception

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        last_exception: Exception

        log_model_try(self.primary_attempt)
        try:
            response = await handler(request)
        except Exception as exc:
            log_model_failure(self.primary_attempt, exc)
            last_exception = exc
        else:
            log_model_success(self.primary_attempt)
            return response

        for index, fallback_model in enumerate(self.models):
            attempt = self.fallback_attempts[index]
            log_model_retry(attempt)
            try:
                response = await handler(request.override(model=fallback_model))
            except Exception as exc:
                log_model_failure(attempt, exc)
                last_exception = exc
                continue

            log_model_success(attempt)
            return response

        raise last_exception


def log_model_try(attempt: ModelAttempt) -> None:
    logger.info(
        "Trying model | stage=%s | provider=%s | modelId=%s",
        attempt.stage,
        attempt.provider,
        attempt.model_id,
    )


def log_model_retry(attempt: ModelAttempt) -> None:
    logger.info(
        "Retrying with model | stage=%s | provider=%s | modelId=%s",
        attempt.stage,
        attempt.provider,
        attempt.model_id,
    )


def log_model_success(attempt: ModelAttempt) -> None:
    logger.info(
        "Model succeeded | stage=%s | provider=%s | modelId=%s",
        attempt.stage,
        attempt.provider,
        attempt.model_id,
    )


def log_model_failure(attempt: ModelAttempt, error: Any) -> None:
    logger.warning(
        "Model failed | stage=%s | provider=%s | modelId=%s | error=%s",
        attempt.stage,
        attempt.provider,
        attempt.model_id,
        _format_error(error),
    )


def _format_error(error: Any) -> str:
    if error is None:
        return "unknown"

    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"

    return str(error)
