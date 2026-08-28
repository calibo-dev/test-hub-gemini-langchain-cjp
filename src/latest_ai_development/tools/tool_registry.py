from __future__ import annotations

from typing import Callable, List

from langchain_core.tools import BaseTool

from latest_ai_development.tools.custom_tool import search_tool
from latest_ai_development.tools.country_agentic_tool import CountryAgenticTool

_STAGE_TOOL_FACTORIES: dict[str, list[Callable[[], BaseTool]]] = {
    "metadata_fetching_agent": [CountryAgenticTool],
    "query_execution_agent": [CountryAgenticTool],
}

_GENERIC_TOOLS: list[BaseTool] = [
    search_tool,
]


def get_tools(stage_name: str | None = None) -> List[BaseTool]:
    '''Return the list of tools available to the workflow or a specific stage.

    When a stage requests tools by name, it only receives the tools declared for
    that stage. Callers that pass no stage name receive the general-purpose tools.
    '''

    if stage_name:
        factories = _STAGE_TOOL_FACTORIES.get(stage_name, [])
        return [factory() for factory in factories]

    return list(_GENERIC_TOOLS)
