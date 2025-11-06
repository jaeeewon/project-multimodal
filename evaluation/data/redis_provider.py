from collections.abc import Generator
from typing import Any
from ..abs.data_provider import AbstractDataProvider, Sample
from ..utils.connection import RedisConnectionManager
from ..utils.document import Document
from ..types.redis_config import RedisConfig


class RedisDataProvider(AbstractDataProvider):
    skip_type_enum = {
        0: "strict",
        1: "skip_missing_with_warn",
        2: "skip_missing_without_warn",
    }
    skip_type_enum_rev = {"strict": 0, "skip_missing_with_warn": 1, "skip_missing_without_warn": 2}

    def __init__(
        self,
        redis_cfg: RedisConfig,
        key_prefix: str,
        required_fields: list[str] = None,
        required_fields_type: int = 0,
        filter: dict[str, Any] = None,
        filter_type: int = 1,
    ):
        self.key_prefix = key_prefix
        self.key_pattern = f"{key_prefix}:*"
        self._redis_conn = RedisConnectionManager.get_connection(redis_cfg)
        self.required_fields = required_fields or []
        self.required_fields_type = required_fields_type
        self._filter = filter or {}
        self.filter_type = filter_type

        self.redis_cfg = redis_cfg

        if not self.__len__():
            print(f'no keys found for the given prefix "{self.key_prefix}".')
            input("press enter to continue or ctrl+c to abort > ")

    @property
    def data_id(self) -> str:
        return self.key_prefix

    def __len__(self) -> int:
        count = 0
        cursor = 0

        while cursor != 0 or count == 0:
            cursor, keys = self._redis_conn.scan(cursor=cursor, match=self.key_pattern, count=1000)

            if not self._filter:
                count += len(keys)
            else:
                for key in keys:
                    sample = self._redis_conn.hgetall(key)

                    if self._filter and len([k for k, v in self._filter.items() if k not in sample or sample[k] != v]):
                        continue

                    count += 1

            if cursor == 0:
                break

        return count

    def __iter__(self) -> Generator[Sample]:
        cursor = 0

        while cursor != 0 or cursor == 0:
            cursor, keys = self._redis_conn.scan(cursor=cursor, match=self.key_pattern, count=1000)

            for key in keys:
                sample = self._redis_conn.hgetall(key)

                if self._filter and len([k for k, v in self._filter.items() if k not in sample or sample[k] != v]):
                    filter_type_str = RedisDataProvider.skip_type_enum[self.filter_type]
                    if filter_type_str == "strict":
                        raise ValueError(f"[{self.key_prefix}] sample {key} does not match filter {self._filter}")
                    elif filter_type_str == "skip_missing_with_warn":
                        print(
                            f"[{self.key_prefix}] warning: sample {key} does not match filter {self._filter}; skip this sample"
                        )
                    continue

                missing_fields = [field for field in self.required_fields if field not in sample]
                if missing_fields:
                    required_type_str = RedisDataProvider.skip_type_enum[self.required_fields_type]
                    if required_type_str == "strict":
                        raise ValueError(
                            f"[{self.key_prefix}] missing required fields {missing_fields} in sample {key}"
                        )
                    elif "skip_missing" in required_type_str:
                        if required_type_str == "skip_missing_with_warn":
                            print(
                                f"[{self.key_prefix}] warning: missing required fields {missing_fields} in sample {key}; skip this sample"
                            )
                        continue
                    else:
                        raise NotImplementedError(f"invalid required_type: {self.required_type}")

                yield Document(self.redis_cfg, key)

            if cursor == 0:
                break

    def get_all_keys(self):
        cursor = 0
        _keys = []

        while cursor != 0 or not _keys:
            cursor, keys = self._redis_conn.scan(cursor=cursor, match=self.key_pattern, count=1000)
            _keys.extend(keys)

            if cursor == 0:
                break

        return _keys

    def insert_samples(self, samples: list[Sample]):
        for sample in samples:
            if "key" not in sample:
                raise ValueError("each samples must have 'key' field for insertion")
            key = sample["key"]
            data_to_store = {k: v for k, v in sample.items() if k != "key"}
            self._redis_conn.hset(key, mapping=data_to_store)

    def delete_samples(self, keys: list[str]):
        if not keys:
            raise ValueError("keys are empty")
        self._redis_conn.delete(*keys)

    def update_samples(self, keys: list[str], values: list[dict]):
        for k, vs in zip(keys, values):
            self._redis_conn.hset(k, mapping=vs)


if __name__ == "__main__":
    # python -m evaluation.data.redis_provider
    data_provider = RedisDataProvider(
        redis_cfg=RedisConfig(host="salmonn.hufs.jae.one", port=6379, db=9),
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
            data_provider.delete_samples(data_provider.get_all_keys())
            print("removed all samples")

    print(f"data_id: {data_provider.data_id}")
    print(f"len(samples): {len(data_provider)}")
