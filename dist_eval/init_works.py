import redis
import uuid
import os
import time
from typing import Callable, Any
from collections import Counter
import pandas as pd


class SalmonnRedis:
    PENDING_QUEUE = "task_{}:pending"
    PROCESSING_QUEUE = "task_{}:processing"
    TASK_HASH_PREFIX = "task_{}:"

    def __init__(self, host="localhost", port=6379, db=0, decode_responses=True):
        self.client = redis.Redis(
            host=host, port=port, db=db, decode_responses=decode_responses
        )

    def initialize_tasks(self, task_name: str, task_data: list[dict[str, str]]):
        PENDING_QUEUE = self.PENDING_QUEUE.format(task_name)
        PROCESSING_QUEUE = self.PROCESSING_QUEUE.format(task_name)
        TASK_HASH_PREFIX = self.TASK_HASH_PREFIX.format(task_name)

        print(f"initializing {task_name} using {len(task_data)} tasks...")

        # assert empty queues
        pending_len = self.client.llen(PENDING_QUEUE)
        processing_len = self.client.llen(PROCESSING_QUEUE)
        assert (
            pending_len == 0 and processing_len == 0
        ), f"target Qs are not empty! pending: {pending_len}, processing: {processing_len}"
        # self.client.delete(PENDING_QUEUE, PROCESSING_QUEUE)

        for key in self.client.scan_iter(f"{TASK_HASH_PREFIX}*"):
            self.client.delete(key)

        for task in task_data:
            task_id = str(uuid.uuid4())
            task_key = f"{TASK_HASH_PREFIX}{task_id}"
            task.update({"id": task_id, "status": "initialized"})
            self.client.hset(task_key, mapping=task)

            self.client.lpush(PENDING_QUEUE, task_id)

        print(f"'{PENDING_QUEUE}' has {self.client.llen(PENDING_QUEUE)} tasks now")

    def start_worker(
        self,
        task_name: str,
        device: str,
        fn: Callable[[dict[str, Any]], Any],
    ):
        worker_id = (
            f"worker-{os.getpid()}-{os.environ.get('CUDA_VISIBLE_DEVICES', device)}"
        )
        log = lambda msg: print(f"[{worker_id}] {msg}")

        PENDING_QUEUE = self.PENDING_QUEUE.format(task_name)
        PROCESSING_QUEUE = self.PROCESSING_QUEUE.format(task_name)
        TASK_HASH_PREFIX = self.TASK_HASH_PREFIX.format(task_name)

        while True:
            task_id = None
            try:
                # pending -> processing
                task_id = self.client.brpoplpush(
                    PENDING_QUEUE, PROCESSING_QUEUE, timeout=0
                )

                task_key = f"{TASK_HASH_PREFIX}{task_id}"
                self.client.hset(task_key, "status", f"'{worker_id}' processing...")
                task_data = self.client.hgetall(task_key)
                out = fn(task_data)
                self.client.hset(task_key, "infer", out)
                self.client.hset(task_key, "status", "completed")
                self.client.lrem(PROCESSING_QUEUE, 1, task_id)

            except Exception as e:
                self.client.hset(task_key, "status", "failed")
                self.client.lrem(PROCESSING_QUEUE, 1, task_id)
                log(f"'{task_id}' failed and removed.")
                log(f"err: {str(e)}")

    def statistics(self, task_name: str, overwrite=False):
        TASK_HASH_PREFIX = self.TASK_HASH_PREFIX.format(task_name)
        passkeys = [
            self.PENDING_QUEUE.format(task_name),
            self.PROCESSING_QUEUE.format(task_name),
        ]

        task_keys = [
            k
            for k in self.client.scan_iter(f"{TASK_HASH_PREFIX}*")
            if k not in passkeys
        ]
        total_tasks = len(task_keys)

        if total_tasks == 0:
            print("no data")
            return

        status_counts = Counter()
        completed_tasks = 0

        for key in task_keys:
            task_data = self.client.hgetall(key)

            status = task_data.get("status")

            if overwrite and (status == "failed" or "worker" in status):
                self.client.hset(key, "status", "initialized")
                task_id = task_data.get("id")
                PENDING_QUEUE = self.PENDING_QUEUE.format(task_name)
                self.client.lpush(PENDING_QUEUE, task_id)
                status = "initialized"

            status_counts[status] += 1

            if status == "completed":
                completed_tasks += 1

        title = f"===== Statistics of {task_name} task ====="

        print(title)
        print(f"total: {total_tasks}")

        for status, count in status_counts.items():
            print(f"- {status}: {count} tasks ({count/total_tasks:.2%})")
        print("=" * len(title))


if __name__ == "__main__":
    pass
    # ===== monitor multiple tasks =====
    # tasks = [
    #     "en2ja",
    #     "en2de",
    #     "LibriSpeech-ASR-test-clean",
    #     "LibriSpeech-ASR-test-other",
    #     "en2zh",
    # ]
    # for i, task in enumerate(tasks):
    #     tasks[i] = task, SalmonnRedis(host="192.168.219.101", db=i)

    # while True:
    #     for task in tasks:
    #         task[1].statistics(task[0])
    #     time.sleep(10)

    # ===== monitor status =====
    # r = SalmonnRedis(host="192.168.219.101", db=1)
    # while True:
    #     r.statistics("en2de")
    #     time.sleep(5)

    # ===== initialize CoVoST2 tasks =====
    # ts = pd.read_csv("repr_exp/table3/CoVoST2/tr/test.tsv", sep="\t")
    # print(ts)
    # ts_list = ts.to_dict(orient="records")
    # [{'client_id': '5a5fee9cf1e1c4754ed6dc8a8cd254d2bc595a82ec0b57f9a7fe7afd9bc3d98fc67c2a081eff159030b42ad5816731eb367bc0f7020e8bcdee99c24aeb9030a3', 'path': 'common_voice_en_15790583.mp3', 'sentence': 'Two heads are better than one.', 'up_votes': 2, 'down_votes': 0, 'age': nan, 'gender': nan, 'accent': nan},]
    # is equal to test-splitted covost_v2.en_<to>.tsv and order matched

    # ===== initialize en2ja tasks =====
    # ja = pd.read_csv("repr_exp/table3/CoVoST2/tr/covost_v2.en_ja.tsv", sep="\t")
    # ja = ja[ja["split"] == "test"].to_dict(orient="records")
    # # print(ja)
    # r = SalmonnRedis(host="192.168.219.101", db=0)
    # r.initialize_tasks("en2ja", ja)

    # ===== initialize en2de tasks =====
    # de = pd.read_csv("repr_exp/table3/CoVoST2/tr/covost_v2.en_de.tsv", sep="\t")
    # de = de[de["split"] == "test"].to_dict(orient="records")
    # # print(ja)
    # r = SalmonnRedis(host="192.168.219.101", db=1)
    # r.initialize_tasks("en2de", de)

    # ===== initialize LibriSpeech ASR tasks =====
    # from librispeech import get_librispeech_list

    # libri = get_librispeech_list()
    # for i, subset in enumerate(libri):
    #     task_name = f"LibriSpeech-ASR-{subset}"
    #     r = SalmonnRedis(host="192.168.219.101", db=i + 2)
    #     r.initialize_tasks(task_name, libri[subset])
    # 2: test-clean, 3: test-other

    # ===== initialize en2zh tasks =====
    # de = pd.read_csv("repr_exp/table3/CoVoST2/tr/covost_v2.en_zh-CN.tsv", sep="\t")
    # de = de[de["split"] == "test"].to_dict(orient="records")
    # # print(ja)
    # r = SalmonnRedis(host="192.168.219.101", db=4)
    # r.initialize_tasks("en2zh", de)

    # ===== initialize GigaSpeech ASR tasks =====
