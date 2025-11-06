from abc import ABC, abstractmethod
from typing import Any
from .data_provider import Sample


class AbstractEvaluator(ABC):

    @property
    @abstractmethod
    def task_name(self) -> str:
        # ex. SALMONN_SAKURA_Gender_exp1
        pass

    def preprocess(self, text: str) -> Any:
        """
        default: str(text).strip().lower()
        """
        return str(text).strip().lower()

    @abstractmethod
    def evaluate(self, predictions: list[str], samples: list[Sample]) -> list[Any]:
        """
        default: verify lengths only
        return: list of evaluation results
        """
        if len(predictions) != len(samples):
            raise ValueError(f"|pred| ({len(predictions)}) and |samples| ({len(samples)}) must be the same")
        pass
