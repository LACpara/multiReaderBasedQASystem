import pathlib
import pytest
import sys
import os

from unittest.mock import Mock

workspace = pathlib.Path(__file__).resolve().parent.parent
if workspace not in sys.path:
    sys.path.insert(0, str(workspace))


from hmr.config import IngestionConfig
from hmr.reader_builder import ReaderTreeBuilder
from hmr.llm.base import ReaderLLMService
from hmr.llm.openai_compatible import OpenAICompatibleLLMClient
from hmr.storage.base import KnowledgeStore
from hmr.vector.base import VectorIndex
from hmr.domain import ReaderNode, ReaderKnowledge
from hmr.complexity import ComplexityEstimator


@pytest.fixture
def llm_service():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="../.env")
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    model_name = os.environ.get("MODEL_NAME")

    assert api_key is not None, "api_key is not found in environment"
    assert base_url is not None, "base_url is not found in environment"
    assert model_name is not None, "model_name is not found in environment"

    return OpenAICompatibleLLMClient(api_key, model_name, base_url)

@pytest.fixture
def mock_knowledge_store():
    mock_knowledge_store_ = Mock(spec=KnowledgeStore)
    return mock_knowledge_store_


@pytest.fixture
def mock_vector_index():
    mock_vector_index_ = Mock(spec=VectorIndex)
    return mock_vector_index_


@pytest.fixture
def redaer_builder(llm_service, mock_knowledge_store, mock_vector_index):
    config = IngestionConfig()
    reader_builder_ = ReaderTreeBuilder(
        config=config,
        llm_service=llm_service,
        store=mock_knowledge_store,
        vector_index=mock_vector_index
    )
    return reader_builder_


def test_reader_tree_build(llm_service, mock_knowledge_store, mock_vector_index):
    mock_llm = Mock(spec=ReaderLLMService)
    fake_knowledge = ReaderKnowledge(
        summary="test_summary",
        source_excerpt="test source content"
    )
    mock_llm.extract_knowledge.return_value = fake_knowledge
    mock_llm.build_capability_questions.return_value = ["QA1", "QA2", "QA3"]

    mock_complexity = Mock(spec=ComplexityEstimator)
    mock_complexity.score.return_value = 0.1
    
    mock_knowledge_store.init_schema
    mock_knowledge_store.upsert_reader
    mock_knowledge_store.get_reader
    mock_knowledge_store.list_children
    mock_knowledge_store.list_document_readers
    mock_knowledge_store.save_query_result
    mock_knowledge_store.close

    mock_vector_index.upsert_reader
    mock_vector_index.query
    mock_vector_index.close

    config = IngestionConfig()
    builder = ReaderTreeBuilder(
        config=config,
        llm_service=llm_service,
        store=mock_knowledge_store,
        vector_index=mock_vector_index,
        complexity_estimator=mock_complexity
    )