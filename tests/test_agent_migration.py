from __future__ import annotations

import importlib
import logging
import sys
from types import SimpleNamespace

import pytest

from latest_ai_development import secrets_manager as secrets_module
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

    monkeypatch.setenv("DEBUG", "false")

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

    monkeypatch.setenv("DEBUG", "false")

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


def test_workflow_logs_stage_failure_with_flow_run_id(monkeypatch, caplog, tmp_path):
    from latest_ai_development import workflow as module

    monkeypatch.setenv("DEBUG", "false")

    class FailingStage:
        component_name = "research_agent"

        def invoke(self, context):
            raise RuntimeError("boom")

    settings = SimpleNamespace(
        output_dir_path=tmp_path,
        report_output_path=tmp_path / "report.md",
        log_level="INFO",
        provider="OPENAI",
    )
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        module,
        "get_workflow_config",
        lambda: {"stages": ["research"], "execution": {"save_output": False}},
    )
    monkeypatch.setattr(module, "STAGE_REGISTRY", {"research": FailingStage})
    caplog.set_level(logging.ERROR, logger="latest_ai_development.workflow")

    with pytest.raises(RuntimeError):
        module.LatestAiDevelopmentWorkflow().kickoff(
            {"topic": "AI agents"},
            flow_run_id="flow-123",
        )

    assert any(
        "Stage execution failed" in record.message
        and "flow_run_id=flow-123" in record.message
        and "component=research_agent" in record.message
        for record in caplog.records
    )


def test_research_stage_logs_agent_failure(monkeypatch, caplog):
    from latest_ai_development.stages import research_stage as module

    class FailingAgent:
        def invoke(self, payload):
            raise RuntimeError("agent boom")

    monkeypatch.setattr(module, "build_research_agent", lambda: FailingAgent())
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
    caplog.set_level(logging.ERROR, logger="latest_ai_development.stages.research_stage")

    stage = module.ResearchStage()

    with pytest.raises(RuntimeError):
        stage.invoke(
            {
                "topic": "AI agents",
                "current_year": 2026,
                "knowledge_context": "",
                "flow_run_id": "flow-456",
            }
        )

    assert any(
        "Research stage execution failed" in record.message
        and "flow_run_id=flow-456" in record.message
        and "component=research_agent" in record.message
        for record in caplog.records
    )


def test_reporting_stage_logs_chain_failure(monkeypatch, caplog):
    from latest_ai_development.stages import reporting_stage as module

    class FailingChain:
        def invoke(self, payload):
            raise RuntimeError("chain boom")

    monkeypatch.setattr(module, "build_reporting_chain", lambda: FailingChain())
    caplog.set_level(logging.ERROR, logger="latest_ai_development.stages.reporting_stage")

    stage = module.ReportingStage()

    with pytest.raises(RuntimeError):
        stage.invoke(
            {
                "topic": "AI agents",
                "research_notes": "notes",
                "current_year": 2026,
                "knowledge_context": "",
                "flow_run_id": "flow-789",
            }
        )

    assert any(
        "Reporting stage execution failed" in record.message
        and "flow_run_id=flow-789" in record.message
        and "component=reporting_chain" in record.message
        for record in caplog.records
    )


def test_normalize_api_context_matches_crewai_behavior():
    from latest_ai_development.config.settings import normalize_api_context

    assert normalize_api_context(None) == ""
    assert normalize_api_context("") == ""
    assert normalize_api_context("/") == ""
    assert normalize_api_context("testing") == "/testing"
    assert normalize_api_context("/testing/") == "/testing"


def test_normalize_provider_name_supports_crewai_provider_names():
    from latest_ai_development.config.settings import normalize_provider_name

    assert normalize_provider_name("OPENAI") == "openai"
    assert normalize_provider_name("ANTHROPICAI") == "anthropicai"
    assert normalize_provider_name("ANTHROPIC") == "anthropicai"
    assert normalize_provider_name("GEMINIAI") == "geminiai"
    assert normalize_provider_name("GEMINI") == "geminiai"


def test_get_llm_builds_anthropic_provider(monkeypatch):
    from latest_ai_development.llm import llm_factory as module

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSecretsManager:
        def get_secret(self, secret_name, secret_key=None):
            assert secret_name == "anthropic-secret"
            return "anthropic-key"

    monkeypatch.setitem(
        sys.modules,
        "langchain_anthropic",
        SimpleNamespace(ChatAnthropic=FakeChatAnthropic),
    )
    monkeypatch.setattr(secrets_module, "SecretsManager", FakeSecretsManager)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(
            anthropicai_api_key_secret="anthropic-secret",
        ),
    )
    llm = module.get_llm(
        {
            "primary": {
                "provider": "ANTHROPICAI",
                "modelId": "claude-test",
                "generationDefaults": {
                    "temperature": 0.2,
                    "maxOutputTokens": 123,
                },
            }
        }
    )

    assert isinstance(llm, FakeChatAnthropic)
    assert llm.kwargs == {
        "model": "claude-test",
        "temperature": 0.2,
        "api_key": "anthropic-key",
        "max_tokens": 123,
    }


def test_get_llm_builds_gemini_provider_alias(monkeypatch):
    from latest_ai_development.llm import llm_factory as module

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSecretsManager:
        def get_secret(self, secret_name, secret_key=None):
            assert secret_name == "gemini-secret"
            return "gemini-key"

    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        SimpleNamespace(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI),
    )
    monkeypatch.setattr(secrets_module, "SecretsManager", FakeSecretsManager)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(geminiai_api_key_secret="gemini-secret"),
    )
    llm = module.get_llm(
        {
            "primary": {
                "provider": "GEMINI",
                "modelId": "gemini-test",
                "generationDefaults": {
                    "temperature": 0.4,
                    "maxOutputTokens": 456,
                },
            }
        }
    )

    assert isinstance(llm, FakeChatGoogleGenerativeAI)
    assert llm.kwargs == {
        "model": "gemini-test",
        "temperature": 0.4,
        "api_key": "gemini-key",
        "max_output_tokens": 456,
    }


def test_fastapi_root_path_uses_context_prefix(monkeypatch):
    from latest_ai_development.config import settings as settings_module

    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("CONTEXT", "testing")
    monkeypatch.setenv("API_ROOT_PATH", "")

    settings_module.get_settings.cache_clear()
    settings_module.get_workflow_config.cache_clear()
    settings_module.get_stages_config.cache_clear()

    if "latest_ai_development.main" in sys.modules:
        main = importlib.reload(sys.modules["latest_ai_development.main"])
    else:
        main = importlib.import_module("latest_ai_development.main")

    route_paths = {getattr(route, "path", "") for route in main.app.routes}

    assert main.settings.context == "testing"
    assert main.API_ROOT_PATH == "/testing"
    assert main.app.root_path == "/testing"
    assert "/testing/health" in route_paths
    assert "/testing/ask" in route_paths


def test_fastapi_api_root_path_overrides_context(monkeypatch):
    from latest_ai_development.config import settings as settings_module

    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("CONTEXT", "/context")
    monkeypatch.setenv("API_ROOT_PATH", "/api-root")

    settings_module.get_settings.cache_clear()
    settings_module.get_workflow_config.cache_clear()
    settings_module.get_stages_config.cache_clear()

    if "latest_ai_development.main" in sys.modules:
        main = importlib.reload(sys.modules["latest_ai_development.main"])
    else:
        main = importlib.import_module("latest_ai_development.main")

    assert main.API_ROOT_PATH == "/api-root"
    assert main.app.root_path == "/api-root"


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


def test_runtime_dirs_do_not_create_context_path(tmp_path):
    from latest_ai_development.config.settings import ensure_runtime_dirs

    settings = SimpleNamespace(
        context_path=tmp_path / "external-context",
        output_dir_path=tmp_path / "output",
    )

    ensure_runtime_dirs(settings)

    assert not settings.context_path.exists()
    assert settings.output_dir_path.is_dir()
