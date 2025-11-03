import json
import hashlib
from typing import Optional, Any
from .connection import RedisConnectionManager


class CacheManager:
    def __init__(self, host: str, port: int, db: int):
        self.redis_conn = RedisConnectionManager.get_connection(host=host, port=port, db=db, decode_responses=False)

    def _generate_key(self, *args: str) -> str:
        combined = ":".join(args)
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_infer_key(self, model_id: str, data_id: str) -> str:
        return self._generate_key("infer", model_id, data_id)

    def get_eval_key(self, evaluator_task_name: str, infer_key: str) -> str:
        return self._generate_key("eval", evaluator_task_name, infer_key)

    def get(self, key: str) -> Optional[Any]:
        cached_data = self.redis_conn.get(key)

        if cached_data:
            print(f"[CacheManager] cache hit for key {key[-6:]}")
            return json.loads(cached_data.decode("utf-8"))

        print(f"[CacheManager] cache miss for key {key[-6:]}")
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            self.redis_conn.set(key, json.dumps(value).encode("utf-8"), ex=ttl)
            print(f"[CacheManager] cache set for key {key[-6:]}")
        except Exception as e:
            print(f"[CacheManager] failed to set cache for key {key[-6:]}: {e}")


if __name__ == "__main__":
    # python -m evaluation.utils.cache
    cache_manager = CacheManager(host="salmonn.hufs.jae.one", port=6379, db=10)

    model_id = "model_123"
    data_id = "data_456"
    evaluator_task_name = "task_xyz"

    infer_key = cache_manager.get_infer_key(model_id, data_id)
    eval_key = cache_manager.get_eval_key(evaluator_task_name, infer_key)

    # set cache
    cache_manager.set(infer_key, {"result": "inference_result"}, ttl=3600)
    cache_manager.set(eval_key, {"score": 95}, ttl=3600)

    # get cache
    infer_result = cache_manager.get(infer_key)
    eval_result = cache_manager.get(eval_key)

    print("Inference Result:", infer_result)
    print("Evaluation Result:", eval_result)
