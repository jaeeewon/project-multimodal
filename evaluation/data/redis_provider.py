import json
from collections.abc import Generator
from ..abs.data_provider import AbstractDataProvider, Sample
from ..utils.connection import RedisConnectionManager

required_type_enum = {
    0: "strict",
    1: "skip_missing_with_warn",
    2: "skip_missing_without_warn",
}


class RedisDataProvider(AbstractDataProvider):
    def __init__(
        self, host: str, port: int, db: int, key_prefix: str, required_fields: list[str] = None, required_type: int = 0
    ):
        self.key_prefix = key_prefix
        self._redis_conn = RedisConnectionManager.get_connection(host=host, port=port, db=db)
        self.keys: list[str] = sorted(self._redis_conn.keys(f"{self.key_prefix}:*"))
        self.required_fields = required_fields or []
        self.required_type = required_type

        if not self.keys:
            print(f'no keys found for the given prefix "{self.key_prefix}".')
            input("press enter to continue or ctrl+c to abort > ")

    @property
    def data_id(self) -> str:
        return self.key_prefix

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Generator[Sample]:
        for key in self.keys:
            data = self._redis_conn.hgetall(key)

            sample: Sample = {k: v for k, v in data.items()}
            sample["key"] = key

            missing_fields = [field for field in self.required_fields if field not in sample]
            if missing_fields:
                if required_type_enum[self.required_type] == "strict":
                    raise ValueError(f"[{self.key_prefix}] missing required fields {missing_fields} in sample {key}")
                elif "skip_missing" in required_type_enum[self.required_type]:
                    if required_type_enum[self.required_type] == "skip_missing_with_warn":
                        print(
                            f"[{self.key_prefix}] warning: missing required fields {missing_fields} in sample {key}; skip this sample"
                        )
                    continue
                else:
                    raise NotImplementedError(f"invalid required_type: {self.required_type}")

            if "ground_truth" not in sample:
                print(f"[{self.key_prefix}] Warning: 'ground_truth' missing in sample {key}. Skipping.")
                continue

            if "input" in sample and isinstance(sample["input"], str):
                try:
                    sample["input"] = json.loads(sample["input"])
                except json.JSONDecodeError:
                    pass

            yield sample

    def insert_samples(self, samples: list[Sample]):
        for sample in samples:
            if "key" not in sample:
                raise ValueError("each samples must have 'key' field for insertion")
            key = sample["key"]
            data_to_store = {k: v for k, v in sample.items() if k != "key"}
            self._redis_conn.hset(key, mapping=data_to_store)

        # refresh keys list instead of manually appending
        self.keys: list[str] = sorted(self._redis_conn.keys(f"{self.key_prefix}:*"))

    def delete_samples(self, keys: list[str]):
        self._redis_conn.delete(*keys)

        # refresh keys list instead of manually removing
        self.keys: list[str] = sorted(self._redis_conn.keys(f"{self.key_prefix}:*"))


if __name__ == "__main__":
    # python -m evaluation.data.redis_provider
    data_provider = RedisDataProvider(
        host="salmonn.hufs.jae.one",
        port=6379,
        db=9,
        key_prefix="salmonn-13b:sakura_gender_multihop",
    )

    if len(data_provider) == 0 and input("no samples found. insert mock samples? (y/n) > ").lower() == "y":
        mock_samples = [
            {
                "key": f"{data_provider.key_prefix}:mock_sample_{i}",
                "question": f"what is {i} + {i}?",
                "ground_truth": str(i + i),
            }
            for i in range(10)
        ]
        data_provider.insert_samples(mock_samples)
        print(f"inserted {len(mock_samples)} mock samples")
    elif (
        len(data_provider) > 0
        and input(f"{len(data_provider)} samples found. display samples? (y/n) > ").lower() == "y"
    ):
        for i, sample in enumerate(data_provider):
            print(sample)
        if input("remove all samples? (y/n) > ").lower() == "y":
            data_provider.delete_samples(data_provider.keys)
            print("removed all samples")

    print(f"data_id: {data_provider.data_id}")
    print(f"len(samples): {len(data_provider)}")
