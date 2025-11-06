import json
from .connection import RedisConnectionManager
from ..types.redis_config import RedisConfig

str_indent = 1


class Document:
    def __init__(self, redis_cfg: RedisConfig, key: str):
        self._redis_cfg = redis_cfg
        self._key = key

    def __repr__(self):
        return f"Document(db={self._redis_cfg.get('db')}, key='{self._key}')"

    def __str__(self):
        data = json.dumps(self.to_dict(), indent=2)
        _str = f"Document(db={self._redis_cfg.get('db')}, key='{self._key}')\n{data}"
        return _str.replace("\n", "\n" + " " * str_indent)

    def __getitem__(self, field: str):
        if field == "key":
            return self._key
        db = RedisConnectionManager.get_connection(self._redis_cfg)
        value = db.hget(self._key, field)
        if value is None:
            raise KeyError(f"field '{field}' not found for key '{self._key}'")

        return value

    def __setitem__(self, field: str, value):
        if field == "key":
            raise KeyError("field 'key' is read-only")
        db = RedisConnectionManager.get_connection(self._redis_cfg)
        db.hset(self._key, field, value)

    def __delitem__(self, field: str):
        db = RedisConnectionManager.get_connection(self._redis_cfg)
        result = db.hdel(self._key, field)
        if result == 0:
            raise KeyError(f"field '{field}' not found for key '{self._key}'")

    def __contains__(self, field: str) -> bool:
        db = RedisConnectionManager.get_connection(self._redis_cfg)
        return db.hexists(self._key, field)

    def delete(self, get=False):
        db = RedisConnectionManager.get_connection(self._redis_cfg)
        return db.hgetdel(self._key) if get else db.delete(self._key)

    def to_dict(self):
        db = RedisConnectionManager.get_connection(self._redis_cfg)

        return db.hgetall(self._key)
