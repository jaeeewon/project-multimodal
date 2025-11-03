import redis
import threading

# DB9: test db
# DB10: cache db


class RedisConnectionManager:
    _connections: dict[str, redis.Redis] = {}

    _lock = threading.Lock()

    @classmethod
    def get_connection(
        cls, host: str = "localhost", port: int = 6379, db: int = 0, decode_responses: bool = True
    ) -> redis.Redis:
        key = f"{host}:{port}:{db}:{decode_responses}"

        conn = cls._connections.get(key)
        if conn:
            try:
                if conn.ping():
                    return conn
            except redis.exceptions.ConnectionError:
                pass

        with cls._lock:
            conn = cls._connections.get(key)
            if conn:
                try:
                    if conn.ping():
                        return conn
                except redis.exceptions.ConnectionError:
                    pass

            try:
                conn = redis.Redis(host=host, port=port, db=db, decode_responses=decode_responses)
                conn.ping()
                cls._connections[key] = conn
                return conn
            except Exception as e:
                raise ConnectionError(f"failed to create redis connection for {key}: {e}")


if __name__ == "__main__":
    # python -m evaluation.utils.connection
    redis_conn = RedisConnectionManager.get_connection(host="salmonn.hufs.jae.one", port=6379, db=9)
    redis_conn.set("test_key", "test_value")
    value = redis_conn.get("test_key")
    print(f"successfully connected to redis. test_key: {value}")
    assert value == "test_value"
    print("redis connection test passed")
