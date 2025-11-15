from datetime import datetime
from abc import ABC, abstractmethod
from typing import Iterator, Any

Sample = dict[str, Any]
max_taken_seconds = 30


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

    def len(self, filter: dict) -> list[Sample]:
        n = 0
        for sample in iter(self):
            if any(f not in sample or sample[f] != filter[f] for f in filter):
                continue
            n += 1
        return n

    def take(self, n: int, filter: dict):
        samples = []
        for sample in iter(self):

            if any(f not in sample or sample[f] != filter[f] for f in filter):
                continue

            curr = int(datetime.utcnow().timestamp())
            if "status" in sample and sample["status"] == "taken" and curr - int(sample["takenAt"]) > max_taken_seconds:
                sample["takenAt"] = curr
                samples.append(sample)
            else:
                sample["takenAt"] = curr
                sample["status"] = "taken"
                samples.append(sample)

            if len(samples) == n:
                s = samples
                samples = []
                yield s
        
        if samples:
            yield samples

    def get_all_samples(self) -> list[Sample]:
        return list(self)
