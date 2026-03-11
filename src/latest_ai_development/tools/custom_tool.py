from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import tool


class CustomTopicHintInput(BaseModel):
    topic: str = Field(..., description="Topic for which a hint should be generated.")


@tool("get_topic_hint", args_schema=CustomTopicHintInput)
def get_topic_hint(topic: str) -> str:
    """
    Return a simple reusable hint for the given topic.

    This is a starter example showing how to define a LangChain-compatible tool.
    It is intentionally lightweight and can later be replaced with real tools
    such as search, retrieval, database lookup, or internal API calls.
    """
    clean_topic = topic.strip()

    if not clean_topic:
        return "No topic provided."

    return (
        f"Focus research on the most relevant recent developments, practical use cases, "
        f"risks, and decision-making insights related to '{clean_topic}'."
    )