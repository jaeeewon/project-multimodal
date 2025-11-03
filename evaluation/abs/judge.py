from abc import ABC, abstractmethod


class AbstractJudge(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def judge_batch(self, prompts: list[list[dict[str, str]]]) -> list[str]:
        pass
