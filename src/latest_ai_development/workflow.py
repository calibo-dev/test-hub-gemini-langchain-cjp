from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover
    ChatOllama = None  # type: ignore[assignment]

from latest_ai_development.config.settings import (
    get_agents_config,
    get_models_config,
    get_settings,
    get_tasks_config,
)
from latest_ai_development.prompts import build_reporting_prompt, build_research_prompt
from latest_ai_development.secrets_manager import resolve_openai_api_key


@dataclass
class WorkflowResult:
    topic: str
    current_year: int
    research_notes: str
    final_report: str
    output_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LatestAiDevelopmentWorkflow:
    """
    LangChain implementation of the template behavior:

    1. Researcher agent gathers structured findings
    2. Reporting analyst agent converts findings into report.md
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.agents_config = get_agents_config()
        self.tasks_config = get_tasks_config()
        self.models_config = get_models_config()
        self.llm = self._build_llm()

    def kickoff(self, inputs: dict[str, Any]) -> WorkflowResult:
        return self.run(inputs)

    def run(self, inputs: dict[str, Any]) -> WorkflowResult:
        topic = str(inputs.get("topic", "")).strip()
        if not topic:
            raise ValueError("Missing required input: 'topic'")

        current_year = int(inputs.get("current_year") or datetime.now(timezone.utc).year)
        save_output = bool(inputs.get("save_output", True))

        user_preferences = self._load_user_preferences()
        knowledge_context = self._load_knowledge_context()

        research_notes = self._run_research_stage(
            topic=topic,
            current_year=current_year,
            user_preferences=user_preferences,
            knowledge_context=knowledge_context,
        )

        final_report = self._run_reporting_stage(
            topic=topic,
            current_year=current_year,
            research_notes=research_notes,
            user_preferences=user_preferences,
            knowledge_context=knowledge_context,
        )

        output_file = None
        if save_output:
            output_file = self._save_report(final_report)

        return WorkflowResult(
            topic=topic,
            current_year=current_year,
            research_notes=research_notes,
            final_report=final_report,
            output_file=output_file,
        )

    def _build_llm(self) -> BaseChatModel:
        provider = self.settings.provider.strip().upper()
        providers_cfg = self.models_config.get("providers", {})

        if provider == "OPENAI":
            openai_cfg = providers_cfg.get("openai", {})
            model_name = openai_cfg.get(
                "chat_model",
                self.models_config.get("default_model", "gpt-4.1-mini"),
            )
            temperature = float(openai_cfg.get("temperature", 0.7))
            max_tokens = int(openai_cfg.get("max_tokens", 4000))

            api_key = self.settings.openai_api_key or resolve_openai_api_key()
            if not api_key:
                raise ValueError(
                    "OPENAI provider selected but no API key was found. "
                    "Set OPENAI_API_KEY or OPENAI_API_KEY_SECRET."
                )

            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )

        if provider == "OLLAMA":
            if ChatOllama is None:
                raise ImportError(
                    "langchain-ollama is required for PROVIDER=OLLAMA. "
                    "Add 'langchain-ollama' to pyproject.toml."
                )

            ollama_cfg = providers_cfg.get("ollama", {})
            model_name = ollama_cfg.get("chat_model", "llama3.2")
            temperature = float(ollama_cfg.get("temperature", 0.7))

            return ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=self.settings.ollama_base_url,
            )

        raise ValueError(
            f"Unsupported PROVIDER '{self.settings.provider}'. "
            "Allowed values are OPENAI or OLLAMA."
        )

    def _run_research_stage(
        self,
        topic: str,
        current_year: int,
        user_preferences: str,
        knowledge_context: str,
    ) -> str:
        agent_config = self._render_mapping(
            self.agents_config["researcher"],
            topic=topic,
            current_year=current_year,
        )
        task_config = self._render_mapping(
            self.tasks_config["research_task"],
            topic=topic,
            current_year=current_year,
        )

        prompt = build_research_prompt(
            agent_config=agent_config,
            task_config=task_config,
            user_preferences=user_preferences,
        )
        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke(
            {
                "topic": topic,
                "current_year": current_year,
                "knowledge_context": knowledge_context,
            }
        )
        return result.strip()

    def _run_reporting_stage(
        self,
        topic: str,
        current_year: int,
        research_notes: str,
        user_preferences: str,
        knowledge_context: str,
    ) -> str:
        agent_config = self._render_mapping(
            self.agents_config["reporting_analyst"],
            topic=topic,
            current_year=current_year,
        )
        task_config = self._render_mapping(
            self.tasks_config["reporting_task"],
            topic=topic,
            current_year=current_year,
        )

        prompt = build_reporting_prompt(
            agent_config=agent_config,
            task_config=task_config,
            user_preferences=user_preferences,
        )
        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke(
            {
                "topic": topic,
                "current_year": current_year,
                "research_notes": research_notes,
                "knowledge_context": knowledge_context,
            }
        )
        return result.strip()

    def _render_mapping(
        self,
        mapping: dict[str, Any],
        topic: str,
        current_year: int,
    ) -> dict[str, Any]:
        rendered: dict[str, Any] = {}
        for key, value in mapping.items():
            rendered[key] = self._render_value(value, topic, current_year)
        return rendered

    def _render_value(self, value: Any, topic: str, current_year: int) -> Any:
        if isinstance(value, str):
            return value.format(topic=topic, current_year=current_year)
        if isinstance(value, list):
            return [self._render_value(item, topic, current_year) for item in value]
        if isinstance(value, dict):
            return {
                key: self._render_value(item, topic, current_year)
                for key, item in value.items()
            }
        return value

    def _load_user_preferences(self) -> str:
        preference_file = self.settings.context_path / "user_preference.txt"
        if not preference_file.exists():
            return "No user preference file found."
        return preference_file.read_text(encoding="utf-8").strip()

    def _load_knowledge_context(self, limit: int = 5) -> str:
        context_dir = self.settings.context_path
        if not context_dir.exists():
            return "No local knowledge context found."

        supported_suffixes = {".txt", ".md", ".rst"}
        files: list[Path] = []

        for path in sorted(context_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name == "user_preference.txt":
                continue
            if path.suffix.lower() not in supported_suffixes:
                continue
            files.append(path)

        if not files:
            return "No additional knowledge files found."

        sections: list[str] = []
        for file_path in files[:limit]:
            try:
                content = file_path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                sections.append(f"# {file_path.name}\nUnable to read file: {exc}")
                continue

            sections.append(f"# {file_path.name}\n{content or '<empty file>'}")

        return "\n\n".join(sections)

    def _save_report(self, content: str) -> str:
        output_path = self.settings.report_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return str(output_path)


def run_workflow(inputs: dict[str, Any]) -> WorkflowResult:
    return LatestAiDevelopmentWorkflow().run(inputs)