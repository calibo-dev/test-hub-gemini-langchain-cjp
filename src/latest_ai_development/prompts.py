from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate


def _format_instructions(items: list[str] | None) -> str:
    if not items:
        return "- Follow the task contract carefully."
    return "\n".join(f"- {item}" for item in items)


def _agent_block(agent_config: dict[str, Any]) -> str:
    return (
        f"Role: {agent_config.get('role', 'N/A')}\n"
        f"Goal: {agent_config.get('goal', 'N/A')}\n"
        f"Backstory: {agent_config.get('backstory', 'N/A')}\n"
        f"Instructions:\n{_format_instructions(agent_config.get('instructions'))}"
    )


def build_research_prompt(
    agent_config: dict[str, Any],
    task_config: dict[str, Any],
    user_preferences: str = "",
) -> ChatPromptTemplate:
    system_prompt = f"""
You are the Researcher Agent in a sequential two-agent LangChain orchestration.

Agent Definition
{_agent_block(agent_config)}

User Preference Context
{user_preferences or "No additional user preferences provided."}

Rules
- Perform research only.
- Do not write the final polished markdown report.
- Produce structured research notes that the reporting analyst agent can consume.
- Be accurate, relevant, concise, and practical.
- Use the provided knowledge context when it is relevant.
- Clearly identify assumptions, risks, and gaps when information is uncertain.
""".strip()

    human_prompt = f"""
Task Description
{task_config.get("description", "")}

Expected Output
{task_config.get("expected_output", "")}

Inputs
- Topic: {{topic}}
- Current Year: {{current_year}}

Knowledge Context
{{knowledge_context}}

Return structured research notes with these sections:

1. Topic Overview
2. Key Findings
3. Important Details
4. Risks / Assumptions
5. Suggested Report Outline
""".strip()

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )


def build_reporting_prompt(
    agent_config: dict[str, Any],
    task_config: dict[str, Any],
    user_preferences: str = "",
) -> ChatPromptTemplate:
    system_prompt = f"""
You are the Reporting Analyst Agent in a sequential two-agent LangChain orchestration.

Agent Definition
{_agent_block(agent_config)}

User Preference Context
{user_preferences or "No additional user preferences provided."}

Rules
- Convert the research notes into a polished markdown report.
- Use only supported information from the provided research notes and context.
- Keep the report professional, clear, well-structured, and easy to scan.
- Do not invent unsupported claims.
- Do not wrap the output in markdown code fences.
- The final output should be suitable for saving directly as report.md.
""".strip()

    human_prompt = f"""
Task Description
{task_config.get("description", "")}

Expected Output
{task_config.get("expected_output", "")}

Inputs
- Topic: {{topic}}
- Current Year: {{current_year}}

Research Notes
{{research_notes}}

Knowledge Context
{{knowledge_context}}

Write the final markdown report.
""".strip()

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )