from typing import TypedDict


class RedisConfig(TypedDict):
    host: str  # = "salmonn.hufs.jae.one"
    port: int  # = 6379
    db: int  # = 9
    decode_responses: bool
