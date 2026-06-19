from turtle import tilt

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from hmr.domain import ReaderAnswer, ReaderKnowledge
from hmr.llm.base import LLMClient, ReaderLLMService
from hmr.llm.prompted_service import PromptedReaderLLMService

from prompt_manager.prompt_loader import PromptLoader
from prompt_manager.domain import PromptDefinition


def mock_all_llm_service_interface(test_func):
    test_func_with_mock_environment = test_func
    for interface in ReaderLLMService.__abstractmethods__:
        mock = Mock(callable=interface)

def greet(name):
    return f"Hello {name}"

class TestPromptRenderInLLMService:

    @pytest.fixture
    def mock_llm_client(self):
        return Mock(spec=LLMClient)

    @pytest.fixture
    def prompt_loader(self):
        loader = PromptLoader(Path(__file__).resolve().parent.parent / "promptTemplates")
        return loader

    def test_all_prompt_file_could_be_loaded(self, prompt_loader):
        prompt_list = prompt_loader.list_all_prompt()
        for prompt in prompt_list:
            yaml_path, md_path = prompt_loader.prompts[prompt]
            assert isinstance(yaml_path, Path), f"Prompt head file {prompt} is not pointing to a Path object"
            assert isinstance(md_path, Path), f"Prompt body file {prompt} is not pointing to a Path object"
            loaded = prompt_loader.get_prompt(prompt)
            assert loaded is not None, f"Failed to load prompt file {prompt}"


    @patch.object(PromptedReaderLLMService, "_json_call", return_value={})
    def test_prompt_render_in_llmservice(self, mock_llm_client, prompt_loader):
        llm_service = PromptedReaderLLMService(mock_llm_client, prompt_loader)

        mocks = {}

        for interface in ReaderLLMService.__abstractmethods__:
            original_method = getattr(llm_service, interface)
            mock = Mock(wraps=original_method, name=interface)
            setattr(llm_service, interface, mock)
            mocks[interface] = mock

        fake_knowledge = ReaderKnowledge(
            summary="summary",
            entities=["entity1", "entity2", "entity3"],
            relations=["relation1", "relation2", "relation3"],
            exceptions=["exception1", "exception2", "exception3"],
        )

        fake_knowledges = [
            ReaderKnowledge(
                summary=f"summary{i}",
                entities=["entity{i}"],
                relations=["relation{i}"],
                exceptions=["exception{i}"],
            ) for i in range(3)]

        llm_service.extract_knowledge(text="text", title="title")
        llm_service.evaluate_activation(knowledge=fake_knowledge, question="\n".join(["question1", "question2", "question3"]))
        llm_service.aggregate_children_knowledge(children_knowledge=fake_knowledges, title="child-title")
        llm_service.answer_question(knowledge=fake_knowledge, question="question", reader_id="reader-id", title="title")
        llm_service.answer_backward_inquiry(knowledge=fake_knowledge, question="question", reader_id="reader-id", title="title")
        llm_service.build_capability_questions(knowledge=fake_knowledge, title="title")
        llm_service.detect_information_gaps(text="text", knowledge=fake_knowledge, title="title")
        llm_service.detect_information_gaps_from_knowledge(knowledge=fake_knowledge, title="title")
        llm_service.integrate_knowledge(original_knowledge=fake_knowledge, complete_answers=[], title="title")
        llm_service.estimate_capability_from_children(children_capabilities=[], title="title")
        llm_service.merge_answers(question="question", answers=[ReaderAnswer(reader_id="reader id", title="title", answer="answer", confidence=1.0, source_excerpt="")])
        
        for interface, mock in mocks.items():
            mock.assert_called_once()
