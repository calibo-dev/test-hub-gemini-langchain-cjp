from __future__ import annotations

from latest_ai_development.config.settings import get_settings, get_workflow_config
from latest_ai_development.stages.stage_registry import STAGE_REGISTRY


class LatestAiDevelopmentWorkflow:
    """
    Workflow controller responsible for executing stages defined
    in workflow.yaml.
    """

    def __init__(self):

        self.settings = get_settings()
        self.workflow_config = get_workflow_config()

        self.stage_order = self.workflow_config.get("stages", [])

        if not self.stage_order:
            raise ValueError("workflow.yaml must define at least one stage")

    def kickoff(self, inputs: dict):

        context = dict(inputs)

        for stage_name in self.stage_order:

            stage_class = STAGE_REGISTRY.get(stage_name)

            if not stage_class:
                raise ValueError(f"Unknown stage '{stage_name}'")

            stage = stage_class()

            result = stage.invoke(context)

            if not isinstance(result, dict):
                raise ValueError(
                    f"Stage '{stage_name}' must return a dictionary output"
                )

            context.update(result)

        return context