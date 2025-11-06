from util.sakura_datasets import get_sakura_ds
from .redis_provider import RedisDataProvider
from ..types.redis_config import RedisConfig


class SakuraDataProvider(RedisDataProvider):
    def __init__(self, redis_cfg: RedisConfig, key_prefix: str, required_fields: list[str] = None, filter: dict = None):
        super().__init__(
            redis_cfg=redis_cfg,
            key_prefix=key_prefix,
            required_fields=required_fields,
            required_fields_type=RedisDataProvider.skip_type_enum_rev["strict"],
            filter=filter,
            filter_type=RedisDataProvider.skip_type_enum_rev["skip_missing_without_warn"],
        )

    def insert_ds(self, is_exp=False):
        if input(f"[SakuraDataProvider] are you sure to insert sakura? (is_exp={is_exp}) | (y/n) > ") != "y":
            print("[SakuraDataProvider] insertion cancelled")
            return

        ds = [
            {**d, "key": f"{self.data_id}:{i}", "status": "initialized"}
            for i, d in enumerate(get_sakura_ds(is_exp=is_exp))
        ]
        self.insert_samples(ds)

    def delete_ds(self):
        if input(f"[SakuraDataProvider] are you sure to delete sakura? | (y/n) > ") != "y":
            print("[SakuraDataProvider] deletion cancelled")
            return
        self.delete_samples(self.get_all_keys())


if __name__ == "__main__":
    sakura_provider = SakuraDataProvider(
        redis_cfg=RedisConfig(host="salmonn.hufs.jae.one", port=6379, db=9),
        key_prefix="salmonn-13b:sakura",
        filter={"set": "language", "hop": "single"},
    )

    sakura_provider.delete_ds()
    sakura_provider.insert_ds()

    for sakura in sakura_provider:
        # test: instant update
        # sakura_provider.update_samples([sakura["key"]], [{"a": "b"}])
        print(sakura)
        break

    print(f"data_id: {sakura_provider.data_id}")
    print(f"len(samples): {len(sakura_provider)}")
    sakura_provider.status(keys=["status", "bin_inference"])
