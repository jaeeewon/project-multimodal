import random, numpy as np, torch, os
from abc import ABC, abstractmethod
from typing import List, Callable
from tqdm import tqdm
from .data_provider import AbstractDataProvider, Sample
from datetime import datetime

os.environ["MXNET_FC_TRUE_FP16"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def setup_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)


class AbstractModel(ABC):
    def __init__(self, model_name: str, seed: int = 42):
        self.model_name = model_name
        setup_seeds(seed)
        self.model = None
        self.tokenizer = None

    @property
    def model_id(self) -> str:
        return self.model_name

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def _inference(self, batch: List[Sample]) -> List[str]:
        pass

    def infer(
        self, data_provider: AbstractDataProvider, batch_size: int = 8, callback_fn: Callable = None
    ) -> List[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("models or tokenizer are not loaded")

        predicted: List[str] = []
        batch: List[Sample] = []

        total = len(data_provider)
        iterator = iter(data_provider)

        for i, sample in tqdm(enumerate(iterator), total=total, desc=f"{self.model_id}:{data_provider.data_id}"):
            sample["takenAt"] = int(datetime.utcnow().timestamp())
            sample["status"] = "taken"
            batch.append(sample)

            if len(batch) == batch_size or (i + 1) == total:
                try:
                    batch_preds = self._inference(batch)
                    predicted.extend(batch_preds)
                except Exception as e:
                    print(f"failed to infer: {e}")
                    batch_preds = []
                    predicted.extend([f"_inference: {e}"] * len(batch))
                finally:
                    for sample, inference in zip(batch, batch_preds):
                        sample["status"] = "inferenced"
                        sample["inference"] = inference
                        if callback_fn:
                            callback_fn(sample, inference)
                    batch.clear()

        return predicted
