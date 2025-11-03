from abc import ABC, abstractmethod


class AbstractJudge(ABC):
    @abstractmethod
    def judge_batch(self, prompts: list[list[dict[str, str]]]) -> list[str]:
        pass
