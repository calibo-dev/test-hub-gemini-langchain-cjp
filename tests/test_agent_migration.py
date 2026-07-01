from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

from latest_ai_development.agents.research_agent_builder import extract_final_message_text


class FakeAgent:
    def __init__(self, response: str = "research notes") -> None:
        self.response = response
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return {"messages": [SimpleNamespace(content=self.response)]}


class FakeChain:
    def __init__(self, response: str = "report") -> None:
        self.response = response
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self.response


def test_research_stage_uses_agent_and_passes_runtime_context(monkeypatch):
    from latest_ai_development.stages import research_stage as module

    fake_agent = FakeAgent("final research")
    monkeypatch.setattr(module, "build_research_agent", lambda: fake_agent)
    monkeypatch.setattr(
        module,
        "get_stages_config",
        lambda: {
            "research": {
                "system_prompt": "Research",
                "instructions": [],
                "output_sections": ["Overview"],
            }
        },
    )

    stage = module.ResearchStage()
    result = stage.invoke(
        {
            "topic": "AI agents",
            "current_year": 2026,
            "knowledge_context": "Use internal preferences as context.",
        }
    )

    message = fake_agent.calls[0]["messages"][0]
    assert message["role"] == "user"
    assert "AI agents" in message["content"]
    assert "2026" in message["content"]
    assert "Use internal preferences as context." in message["content"]
    assert result["research_notes"] == "final research"


def test_extract_final_message_text_supports_content_blocks():
    result = {
        "messages": [
            SimpleNamespace(
                content_blocks=[
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ]
            )
        ]
    }

    assert extract_final_message_text(result) == "first\nsecond"


def test_reporting_stage_passes_runtime_context(monkeypatch):
    from latest_ai_development.stages import reporting_stage as module

    fake_chain = FakeChain("final report")
    monkeypatch.setattr(module, "build_reporting_chain", lambda: fake_chain)

    stage = module.ReportingStage()
    result = stage.invoke(
        {
            "topic": "AI agents",
            "research_notes": "notes",
            "current_year": 2026,
            "knowledge_context": "context",
        }
    )

    assert fake_chain.calls[0] == {
        "topic": "AI agents",
        "research_notes": "notes",
        "current_year": 2026,
        "knowledge_context": "context",
    }
    assert result["report"] == "final report"


def test_workflow_writes_report_when_save_output_enabled(monkeypatch, tmp_path):
    from latest_ai_development import workflow as module

    class ResearchStage:
        def invoke(self, context):
            return {"research_notes": "notes"}

    class ReportingStage:
        def invoke(self, context):
            return {"report": "saved report"}

    settings = SimpleNamespace(
        output_dir_path=tmp_path,
        report_output_path=tmp_path / "report.md",
    )
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        module,
        "get_workflow_config",
        lambda: {
            "stages": ["research", "reporting"],
            "execution": {"save_output": True},
        },
    )
    monkeypatch.setattr(
        module,
        "STAGE_REGISTRY",
        {"research": ResearchStage, "reporting": ReportingStage},
    )

    result = module.LatestAiDevelopmentWorkflow().kickoff({"topic": "AI agents"})

    assert result["report"] == "saved report"
    assert settings.report_output_path.read_text(encoding="utf-8") == "saved report"


def test_workflow_does_not_write_report_when_save_output_disabled(monkeypatch, tmp_path):
    from latest_ai_development import workflow as module

    class ReportingStage:
        def invoke(self, context):
            return {"report": "unsaved report"}

    settings = SimpleNamespace(
        output_dir_path=tmp_path,
        report_output_path=tmp_path / "report.md",
    )
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        module,
        "get_workflow_config",
        lambda: {"stages": ["reporting"], "execution": {"save_output": False}},
    )
    monkeypatch.setattr(module, "STAGE_REGISTRY", {"reporting": ReportingStage})

    result = module.LatestAiDevelopmentWorkflow().kickoff({"topic": "AI agents"})

    assert result["report"] == "unsaved report"
    assert not settings.report_output_path.exists()


def test_fastapi_root_path_uses_api_root_path_not_context(monkeypatch, tmp_path):
    from latest_ai_development.config import settings as settings_module

    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("CONTEXT", str(tmp_path / "knowledge"))
    monkeypatch.setenv("API_ROOT_PATH", "")

    settings_module.get_settings.cache_clear()
    settings_module.get_workflow_config.cache_clear()
    settings_module.get_stages_config.cache_clear()
    settings_module.get_models_config.cache_clear()

    if "latest_ai_development.main" in sys.modules:
        main = importlib.reload(sys.modules["latest_ai_development.main"])
    else:
        main = importlib.import_module("latest_ai_development.main")

    assert main.settings.context == str(tmp_path / "knowledge")
    assert main.app.root_path == ""


def test_ask_keeps_public_response_shape(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")

    from latest_ai_development import main

    class FakeWorkflow:
        def kickoff(self, inputs):
            assert inputs["topic"] == "AI agents"
            assert "current_year" in inputs
            assert inputs["knowledge_context"] == ""
            return {"topic": inputs["topic"], "report": "api report"}

    monkeypatch.setattr(main, "workflow", FakeWorkflow())

    response = main.ask(main.AskRequest(topic="AI agents"))

    assert response == {"topic": "AI agents", "report": "api report"}
