from abc import ABC, abstractmethod
from typing import Iterator, Any

Sample = dict[str, Any]


class AbstractDataProvider(ABC):
    @property
    @abstractmethod
    def data_id(self) -> str:
        # ex. "librispeech:test-clean"
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        # yield sample
        pass

    def get_all_samples(self) -> list[Sample]:
        return list(self)
