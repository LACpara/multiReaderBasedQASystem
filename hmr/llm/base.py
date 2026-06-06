from __future__ import annotations

from abc import abstractmethod, ABC

from hmr.domain import ActivationDecision, ReaderAnswer, ReaderKnowledge, CompleteAnswer


class LLMClient(ABC):
    """Low-level provider boundary for actual remote LLM calls."""

    @abstractmethod
    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024, json_require: bool = False) -> str:
        """Return a raw completion string from the configured model provider."""


class ReaderLLMService(ABC):
    """High-level semantic operations required by the core Reader system."""

    @abstractmethod
    def extract_knowledge(self, text: str, *, title: str) -> ReaderKnowledge:
        """Turn a text span into structured local Reader knowledge."""

    @abstractmethod
    def build_capability_questions(self, knowledge: ReaderKnowledge, *, title: str) -> list[str]:
        """Generate questions that describe what this Reader can answer."""

    @abstractmethod
    def evaluate_activation(self, knowledge: ReaderKnowledge, question: str) -> ActivationDecision:
        """Decide whether the Reader should answer the external question."""

    @abstractmethod
    def answer_question(
        self,
        knowledge: ReaderKnowledge,
        question: str,
        *,
        reader_id: str,
        title: str,
    ) -> ReaderAnswer:
        """Produce one grounded partial answer from local Reader knowledge."""

    @abstractmethod
    def merge_answers(self, question: str, answers: list[ReaderAnswer]) -> str:
        """Merge activated Readers' answers into a compact final response."""
    
    @abstractmethod
    def aggregate_children_knowledge(
        self,
        children_knowledge: list[ReaderKnowledge],
        *,
        title: str
    ) -> ReaderKnowledge:
        """聚合多个子节点的知识，构建更高层次的知识表示"""
    
    @abstractmethod
    def estimate_capability_from_children(
        self,
        children_capabilities: list[list[str]],  # 每个子节点的 capability_questions
        *,
        title: str
    ) -> list[str]:
        """基于子节点的能力，估计父节点的回答能力"""
    
    @abstractmethod
    def detect_information_gaps(
        self,
        text: str,
        knowledge: ReaderKnowledge,
        *,
        title: str
    ) -> list[str]:
        """检测文本中的信息缺口，返回需要向上游询问的问题列表"""
    
    @abstractmethod
    def integrate_knowledge(
        self,
        original_knowledge: ReaderKnowledge,
        complete_answers: list[CompleteAnswer],
        *,
        title: str
    ) -> ReaderKnowledge:
        """将获取的完整答案整合到原有知识中"""
    
    @abstractmethod
    def answer_backward_inquiry(
        self,
        knowledge: ReaderKnowledge,
        question: str,
        *,
        reader_id: str,
        title: str
    ) -> tuple[str, str | None, float]:
        """回答逆向求知问题，返回(已回答内容, 剩余问题, 置信度)"""

    @abstractmethod
    def detect_information_gaps_from_knowledge(
        self,
        knowledge: ReaderKnowledge,
        *,
        title: str
    ) -> list[str]:
        """从知识中检测信息缺口（父节点专用），返回需要向上游询问的问题列表"""
