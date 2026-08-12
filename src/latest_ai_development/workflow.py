from __future__ import annotations

import logging
import time
from uuid import uuid4

from latest_ai_development.config.settings import get_settings, get_workflow_config
from latest_ai_development.logging_utils import configure_logging
from latest_ai_development.stages.stage_registry import STAGE_REGISTRY
from latest_ai_development.tracing import (
    initialize_langsmith_tracing,
    trace_stage_execution,
    trace_workflow,
)

logger = logging.getLogger(__name__)


class LatestAiDevelopmentWorkflow:
    """
    Workflow controller responsible for executing stages defined
    in workflow.yaml.
    """

    def __init__(self):

        self.settings = get_settings()
        configure_logging(getattr(self.settings, "log_level", "INFO"))
        self.workflow_config = get_workflow_config()

        self.stage_order = self.workflow_config.get("stages", [])

        if not self.stage_order:
            raise ValueError("workflow.yaml must define at least one stage")

    def kickoff(self, inputs: dict, flow_run_id: str | None = None):
        initialize_langsmith_tracing()
        return self._kickoff_traced(inputs, flow_run_id=flow_run_id)

    @trace_workflow()
    def _kickoff_traced(self, inputs: dict, flow_run_id: str | None = None):
        flow_run_id = str(flow_run_id or inputs.get("flow_run_id") or uuid4().hex[:12]).strip()
        if not flow_run_id:
            flow_run_id = uuid4().hex[:12]

        context = dict(inputs)
        context["flow_run_id"] = flow_run_id

        logger.info(
            "Workflow started | flow_run_id=%s | stages=%s",
            flow_run_id,
            self.stage_order,
        )

        for stage_name in self.stage_order:

            stage_class = STAGE_REGISTRY.get(stage_name)

            if not stage_class:
                logger.error(
                    "Workflow stage lookup failed | flow_run_id=%s | stage=%s",
                    flow_run_id,
                    stage_name,
                )
                raise ValueError(f"Unknown stage '{stage_name}'")

            stage_component = getattr(
                stage_class,
                "component_name",
                getattr(stage_class, "__name__", stage_name),
            )

            logger.info(
                "Stage initialization started | flow_run_id=%s | stage=%s | component=%s",
                flow_run_id,
                stage_name,
                stage_component,
            )
            init_started_at = time.perf_counter()
            try:
                stage = stage_class()
            except Exception:
                logger.exception(
                    "Stage initialization failed | flow_run_id=%s | stage=%s | "
                    "component=%s | elapsed=%.2fs",
                    flow_run_id,
                    stage_name,
                    stage_component,
                    time.perf_counter() - init_started_at,
                )
                raise

            logger.info(
                "Stage execution started | flow_run_id=%s | stage=%s | component=%s",
                flow_run_id,
                stage_name,
                stage_component,
            )
            execute_started_at = time.perf_counter()
            try:
                result = trace_stage_execution(
                    stage_name,
                    stage_component,
                    stage.invoke,
                    context,
                )
            except Exception:
                logger.exception(
                    "Stage execution failed | flow_run_id=%s | stage=%s | "
                    "component=%s | elapsed=%.2fs",
                    flow_run_id,
                    stage_name,
                    stage_component,
                    time.perf_counter() - execute_started_at,
                )
                raise

            if not isinstance(result, dict):
                logger.error(
                    "Stage returned invalid output | flow_run_id=%s | stage=%s | "
                    "component=%s | output_type=%s",
                    flow_run_id,
                    stage_name,
                    stage_component,
                    type(result).__name__,
                )
                raise ValueError(
                    f"Stage '{stage_name}' must return a dictionary output"
                )

            context.update(result)
            logger.info(
                "Stage completed | flow_run_id=%s | stage=%s | component=%s | "
                "elapsed=%.2fs | keys=%s",
                flow_run_id,
                stage_name,
                stage_component,
                time.perf_counter() - execute_started_at,
                sorted(result.keys()),
            )

        if self.workflow_config.get("execution", {}).get("save_output", False):
            report = context.get("report")
            if report:
                self.settings.output_dir_path.mkdir(parents=True, exist_ok=True)
                self.settings.report_output_path.write_text(
                    str(report),
                    encoding="utf-8",
                )
                logger.info(
                    "Report saved | flow_run_id=%s | output_path=%s",
                    flow_run_id,
                    self.settings.report_output_path,
                )
            else:
                logger.warning(
                    "Report save skipped because no report was produced | flow_run_id=%s",
                    flow_run_id,
                )

        logger.info(
            "Workflow completed | flow_run_id=%s | stages=%d",
            flow_run_id,
            len(self.stage_order),
        )

        return context
