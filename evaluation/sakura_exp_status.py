from .data.sakura import SakuraDataProvider
from .types.redis_config import RedisConfig
import os, time

if __name__ == "__main__":
    # python -m evaluation.sakura_exp_status
    data_provider = SakuraDataProvider(
        redis_cfg=RedisConfig(host="salmonn.hufs.jae.one", port=6379, db=11),
        key_prefix="salmonn-13b:SLMN13-SK",
    )

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        data_provider.status()
        time.sleep(3)
