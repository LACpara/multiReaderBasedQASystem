import pathlib
import pytest
import sys
import os

from unittest.mock import Mock, call

workspace = pathlib.Path(__file__).resolve().parent.parent
if workspace not in sys.path:
    sys.path.insert(0, str(workspace))


from hmr.domain import (
    ReaderNode, VectorCandidate, RetrievalResult, ReaderAnswer,
    ActivationDecision, ReaderKnowledge
)
from hmr.retrieval_engine import RetrievalEngine
from hmr.llm.prompted_service import ReaderLLMService
from hmr.storage.sqlite_store import KnowledgeStore
from hmr.vector.base import VectorIndex
from hmr.config import RetrievalConfig


def make_mock_reader(reader_id, parent_id=None, title="Title", knowledge=None):
    return ReaderNode(
        reader_id=reader_id,
        document_id="doc1",
        title=title,
        parent_id=parent_id,
        depth=0,
        ordinal=0,
        text="some text",
        knowledge=knowledge or ReaderKnowledge(summary=""),
        child_ids=[]
    )
    

def test_ask_single_leaf_activation():
    """一个候选，激活成功，叶子节点直接回答，合并并保存"""
    # 1. Mock 依赖
    mock_llm = Mock(spec=ReaderLLMService)
    mock_store = Mock(spec=KnowledgeStore)
    mock_vector = Mock(spec=VectorIndex)

    config = RetrievalConfig(
        top_k=3,
        activation_threshold=0.6,
        max_answers=5
    )

    engine = RetrievalEngine(config, mock_llm, mock_store, mock_vector)

    # 候选
    candidate = VectorCandidate(reader_id="r1", score=0.9, document="d1")
    mock_vector.query.return_value = [candidate]

    reader = make_mock_reader("r1")
    mock_store.get_reader.return_value = reader

    # 激活评估
    decision = ActivationDecision(score=0.8, should_answer=True, sub_question="What is X?", reason="")
    mock_llm.evaluate_activation.return_value = decision

    answer = ReaderAnswer(
        reader_id="r1",
        answer="X is ...",
        confidence=0.9,
        source_excerpt="excerpt",
        title=reader.title
    )
    mock_llm.answer_question.return_value = answer

    merged = "Merged: X is ..."
    mock_llm.merge_answers.return_value = merged

    # 2. 执行
    result = engine.ask("What is X?")

    # 3. 断言
    # 向量查询以 depth=0 作为条件
    expected_condition = {"depth": {"$eq": 0}}
    mock_vector.query.assert_called_once_with("What is X?", top_k=3, where=expected_condition)

    # 获取 reader
    mock_store.get_reader.assert_called_once_with("r1")

    # 激活评估
    mock_llm.evaluate_activation.assert_called_once_with(reader.knowledge, "What is X?")

    # 回答
    mock_llm.answer_question.assert_called_once_with(
        reader.knowledge, "What is X?", reader_id="r1", title=reader.title
    )

    # 合并
    mock_llm.merge_answers.assert_called_once_with("What is X?", [answer])

    # 保存结果
    mock_store.save_query_result.assert_called_once()
    saved_result = mock_store.save_query_result.call_args[0][0]
    assert saved_result.question == "What is X?"
    assert saved_result.answer == merged
    assert saved_result.candidates == [candidate]
    assert saved_result.activated_answers == [answer]

    # 返回值正确
    assert result.question == "What is X?"
    assert result.answer == merged
    assert result.activated_answers == [answer]


def test_ask_max_answers_limit():
    """多个候选，部分激活，达到 max_answers=2 后提前停止"""
    mock_llm = Mock(spec=ReaderLLMService)
    mock_store = Mock(spec=KnowledgeStore)
    mock_vector = Mock(spec=VectorIndex)

    config = RetrievalConfig(
        top_k=5,
        activation_threshold=0.5,
        max_answers=2
    )

    engine = RetrievalEngine(config, mock_llm, mock_store, mock_vector)

    candidates = [VectorCandidate(reader_id=f"r{i}", score=0.9, document=f"d{i}") for i in range(4)]
    mock_vector.query.return_value = candidates

    readers = [make_mock_reader(f"r{i}") for i in range(4)]
    mock_store.get_reader.side_effect = readers

    # 让前三个都激活，但 max_answers=2，第三个不应被回答
    decisions = [
        ActivationDecision(score=0.8, should_answer=True,  sub_question="sq1", reason=""),
        ActivationDecision(score=0.7, should_answer=True,  sub_question="sq2", reason=""),
        ActivationDecision(score=0.9, should_answer=True,  sub_question="sq3", reason=""),
        ActivationDecision(score=0.4, should_answer=False, sub_question="sq4", reason=""),
    ]
    mock_llm.evaluate_activation.side_effect = decisions

    answers = [
        ReaderAnswer(reader_id="r0", answer="A0", confidence=0.8, source_excerpt="", title="T0"),
        ReaderAnswer(reader_id="r1", answer="A1", confidence=0.7, source_excerpt="", title="T1"),
    ]
    mock_llm.answer_question.side_effect = answers  # 只返回前两个
    mock_llm.merge_answers.return_value = "merged"

    result = engine.ask("Q")

    # 验证 evaluate_activation 被调用 2 次
    # 注意循环内逻辑：for candidate in candidates: 先 get_reader, evaluate, 再 passes, 若通过才回答问题并 append，然后检查 len(answers) >= max_answers  break。
    # 所以前两个通过且回答，第三个也通过了评估，但在回答问题前发现已经有两个答案了，所以 break，因此 evaluate_activation 调用了三次。
    assert mock_llm.evaluate_activation.call_count == 2

    # answer_question 只被调用了两次
    assert mock_llm.answer_question.call_count == 2

    # merge 合并这两个答案
    mock_llm.merge_answers.assert_called_once_with("Q", answers)

    # 保存结果
    mock_store.save_query_result.assert_called_once()
    saved = mock_store.save_query_result.call_args[0][0]
    assert len(saved.activated_answers) == 2


def test_ask_recursive_non_leaf():
    """非叶子节点激活后，递归调用子查询，最终合并"""
    mock_llm = Mock(spec=ReaderLLMService)
    mock_store = Mock(spec=KnowledgeStore)
    mock_vector = Mock(spec=VectorIndex)

    config = RetrievalConfig()
    config.top_k = 2
    config.activation_threshold = 0.5
    config.max_answers = 10

    engine = RetrievalEngine(config, mock_llm, mock_store, mock_vector)

    # 顶层候选
    root_candidate = VectorCandidate(reader_id="root", score=0.8, document="doc")
    mock_vector.query.return_value = [root_candidate]

    root_reader = make_mock_reader("root", title="Root")
    mock_store.get_reader.return_value = root_reader

    # 根节点激活
    root_decision = ActivationDecision(score=0.9, should_answer=True, sub_question="Sub Q?", reason="")
    mock_llm.evaluate_activation.return_value = root_decision

    # 递归子查询的 mock 设置：当 _ask 被递归调用时，我们直接 mock 整个 _ask 方法有点麻烦，
    # 更干净的做法是 mock 内部递归调用的依赖，即让 vector_index.query 根据不同的 where 条件返回不同结果。
    # 由于我们测试的是顶层 ask，递归发生在内部，我们可以准备两次 vector_index.query 的返回值。
    child_candidate = VectorCandidate(reader_id="child1", score=0.7, document="doc")
    child_answer = ReaderAnswer(reader_id="child1", answer="Child Answer", confidence=0.95, source_excerpt="child", title="Child")
    child_reader = make_mock_reader("child1", title="Child")
    child_decision = ActivationDecision(score=0.8, should_answer=True, sub_question="Sub Sub", reason="")

    root_reader.child_ids.append(child_reader.reader_id)

    # 配置 vector_index.query 的 side_effect
    mock_vector.query.side_effect = [
        [root_candidate],          # 第一次顶层调用
        [child_candidate]          # 第二次递归调用
    ]

    # 配置 get_reader 分别返回 root 和 child
    mock_store.get_reader.side_effect = [root_reader, child_reader]

    # 配置 evaluate_activation 两次
    mock_llm.evaluate_activation.side_effect = [root_decision, child_decision]

    # 配置 answer_question （子节点是叶子）
    mock_llm.answer_question.return_value = child_answer

    # 配置 merge_answers：第一次是递归里的合并（单个 child 答案），第二次是顶层合并（一个 root 答案）
    root_answer = ReaderAnswer(
        reader_id="root",
        answer="Merged child",   # 这个值由递归 _ask 返回的 ReaderAnswer.answer 填充
        confidence=0.95,         # 递归返回的 ReaderAnswer.confidence
        source_excerpt="child",
        title="Root"
    )
    mock_llm.merge_answers.side_effect = ["Merged child", "Final merged"]

    result = engine.ask("Root Q")

    # 验证两次查询的条件
    assert mock_vector.query.call_count == 2
    mock_vector.query.assert_any_call("Root Q", top_k=2, where={"depth": {"$eq": 0}})
    mock_vector.query.assert_any_call("Sub Q?", top_k=2, where={"parent_id": {"$eq": "root"}})

    # 根节点没有直接 answer_question，而是递归，所以 answer_question 只被 child 调用一次
    mock_llm.answer_question.assert_called_once_with(
        child_reader.knowledge, "Sub Sub", reader_id="child1", title="Child"
    )

    # merge 调用了两次：子层和根层
    assert mock_llm.merge_answers.call_count == 2

    # 顶层结果包含根候选和激活答案
    saved = mock_store.save_query_result.call_args[0][0]
    assert len(saved.activated_answers) == 1
    assert saved.activated_answers[0].reader_id == "root"
    assert saved.answer == "Final merged"


def test_ask_no_candidates():
    """无候选时，空答案，merge 仍被调用，保存空结果"""
    mock_llm = Mock(spec=ReaderLLMService)
    mock_store = Mock(spec=KnowledgeStore)
    mock_vector = Mock(spec=VectorIndex)
    mock_vector.query.return_value = []
    mock_llm.merge_answers.return_value = "No answer"

    config = RetrievalConfig()
    engine = RetrievalEngine(config, mock_llm, mock_store, mock_vector)

    result = engine.ask("Q")
    mock_llm.merge_answers.assert_called_once_with("Q", [])
    mock_store.save_query_result.assert_called_once()
    assert result.activated_answers == []
    assert result.answer == "No answer"


def test_ask_all_deactivated():
    """所有候选都不激活，答案为空"""
    mock_llm = Mock(spec=ReaderLLMService)
    mock_store = Mock(spec=KnowledgeStore)
    mock_vector = Mock(spec=VectorIndex)

    candidates = [VectorCandidate(reader_id="r1", score=0.3, document="doc")]
    mock_vector.query.return_value = candidates
    reader = make_mock_reader("r1")
    mock_store.get_reader.return_value = reader
    decision = ActivationDecision(score=0.2, should_answer=False, sub_question="", reason="")
    mock_llm.evaluate_activation.return_value = decision
    mock_llm.merge_answers.return_value = "No answer"

    engine = RetrievalEngine(RetrievalConfig(), mock_llm, mock_store, mock_vector)
    result = engine.ask("Q")

    mock_llm.answer_question.assert_not_called()
    mock_llm.merge_answers.assert_called_once_with("Q", [])
    assert result.activated_answers == []