import redis
import uuid
import os
import sys
import time
from typing import Callable, Any
from collections import Counter
import pandas as pd
import torch


class SalmonnRedis:
    PENDING_QUEUE = "task_{}:pending"
    PROCESSING_QUEUE = "task_{}:processing"
    TASK_HASH_PREFIX = "task_{}:"

    def __init__(self, host="localhost", port=6379, db=0, decode_responses=True):
        self.client = redis.Redis(
            host=host, port=port, db=db, decode_responses=decode_responses
        )

    def initialize_tasks(
        self, task_name: str, task_data: list[dict[str, str]], push_anyway=False
    ):
        PENDING_QUEUE = self.PENDING_QUEUE.format(task_name)
        PROCESSING_QUEUE = self.PROCESSING_QUEUE.format(task_name)
        TASK_HASH_PREFIX = self.TASK_HASH_PREFIX.format(task_name)

        print(f"initializing {task_name} using {len(task_data)} tasks...")

        # assert empty queues
        pending_len = self.client.llen(PENDING_QUEUE)
        processing_len = self.client.llen(PROCESSING_QUEUE)
        if not push_anyway:
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
        fn: Callable[[dict[str, Any]], dict[str, Any]],
    ):
        device_id = os.environ.get("CUDA_VISIBLE_DEVICES", device)
        device_name = torch.cuda.get_device_name()
        worker_id = f"worker-{os.getpid()}-{device_name}-{device_id}"
        log = lambda msg: print(f"[{worker_id}] {msg}")

        PENDING_QUEUE = self.PENDING_QUEUE.format(task_name)
        PROCESSING_QUEUE = self.PROCESSING_QUEUE.format(task_name)
        TASK_HASH_PREFIX = self.TASK_HASH_PREFIX.format(task_name)

        while self.client.llen(PENDING_QUEUE) > 0:
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
                for k, v in out.items():
                    self.client.hset(task_key, k, v)
                self.client.hset(task_key, "status", "completed")
                self.client.lrem(PROCESSING_QUEUE, 1, task_id)

            except Exception as e:
                self.client.hset(task_key, "status", "failed")
                self.client.lpush(PENDING_QUEUE, task_id)
                self.client.lrem(PROCESSING_QUEUE, 1, task_id)
                log(f"'{task_id}' failed and removed.")
                log(f"err: {str(e)}")

    def statistics(self, task_name: str, overwrite=False, return_str=False):
        rtn_str = ""

        def print_or_store(msg):
            nonlocal rtn_str
            if return_str:
                rtn_str += msg + "\n"
            else:
                print(msg)

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
            print_or_store("no data")
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

        print_or_store(title)
        print_or_store(f"total: {total_tasks}")

        for status, count in status_counts.items():
            print_or_store(f"- {status}: {count} tasks ({count/total_tasks:.2%})")
        print_or_store("=" * len(title))

        if return_str:
            return rtn_str


if __name__ == "__main__":
    """"""
    # ===== monitor multiple tasks =====
    # tasks = [
    #     "en2ja",
    #     "en2de",
    #     "LibriSpeech-ASR-test-clean",
    #     "LibriSpeech-ASR-test-other",
    #     "en2zh",
    #     "GigaSpeech-ASR-test",
    #     "AudioCaps-AAC-test",
    # ]
    # for i, task in enumerate(tasks):
    #     tasks[i] = task, SalmonnRedis(host="192.168.219.101", db=i)

    # while True:
    #     for task in tasks:
    #         task[1].statistics(task[0])
    #     time.sleep(10)

    # ===== monitor lora-scaled librispeech asr tasks =====
    # ENTER_ALT_SCREEN = "\x1b[?1049h"
    # EXIT_ALT_SCREEN = "\x1b[?1049l"
    # CLEAR_SCREEN = "\x1b[2J"
    # CURSOR_HOME = "\x1b[H"

    # tasks = [
    #     "LibriSpeech-ASR-test-clean",
    #     "LibriSpeech-ASR-test-other",
    # ]
    # for i, task in enumerate(tasks):
    #     tasks[i] = task, SalmonnRedis(host="192.168.219.101", db=i + 2)

    # try:
    #     sys.stdout.write(ENTER_ALT_SCREEN)

    #     while True:
    #         sys.stdout.write(CLEAR_SCREEN)
    #         sys.stdout.write(CURSOR_HOME)

    #         sys.stdout.write(f"monitor LoRA-scaled ASR tasks\n\n")
    #         for task in tasks:
    #             for i in range(4):
    #                 task_name = f"{task[0]}-ls{i:02d}"
    #                 sys.stdout.write(task[1].statistics(task_name, return_str=True))
    #         sys.stdout.flush()

    #         time.sleep(10)

    # except KeyboardInterrupt:
    #     pass

    # finally:
    #     sys.stdout.write(EXIT_ALT_SCREEN)

    # ===== monitor lora-scaled gigaspeech asr tasks =====
    # ENTER_ALT_SCREEN = "\x1b[?1049h"
    # EXIT_ALT_SCREEN = "\x1b[?1049l"
    # CLEAR_SCREEN = "\x1b[2J"
    # CURSOR_HOME = "\x1b[H"

    # r = SalmonnRedis(host="192.168.219.101", db=5)

    # try:
    #     sys.stdout.write(ENTER_ALT_SCREEN)

    #     while True:
    #         sys.stdout.write(CLEAR_SCREEN)
    #         sys.stdout.write(CURSOR_HOME)

    #         sys.stdout.write(f"monitor LoRA-scaled ASR tasks\n\n")
    #         for i in range(4):
    #             task_name = f"GigaSpeech-ASR-test-ls{i:02d}"
    #             sys.stdout.write(r.statistics(task_name, return_str=True))
    #         sys.stdout.flush()

    #         time.sleep(10)

    # except KeyboardInterrupt:
    #     pass

    # finally:
    #     sys.stdout.write(EXIT_ALT_SCREEN)

    # ===== monitor lora-scaled librispeech pr tasks =====
    ENTER_ALT_SCREEN = "\x1b[?1049h"
    EXIT_ALT_SCREEN = "\x1b[?1049l"
    CLEAR_SCREEN = "\x1b[2J"
    CURSOR_HOME = "\x1b[H"

    r = SalmonnRedis(host="192.168.219.101", db=2)
    task = "LibriSpeech-PR-test-clean"

    try:
        sys.stdout.write(ENTER_ALT_SCREEN)

        while True:
            sys.stdout.write(CLEAR_SCREEN)
            sys.stdout.write(CURSOR_HOME)

            sys.stdout.write(f"monitor LoRA-scaled PR tasks\n\n")
            for i in range(5):
                task_name = task if i == 4 else f"{task}-ls{i:02d}"
                sys.stdout.write(r.statistics(task_name, return_str=True))
            sys.stdout.flush()

            time.sleep(10)

    except KeyboardInterrupt:
        pass

    finally:
        sys.stdout.write(EXIT_ALT_SCREEN)

    # ===== monitor status =====
    # r = SalmonnRedis(host="192.168.219.101", db=2)
    # while True:
    #     r.statistics("LibriSpeech-PR-test-clean")
    #     time.sleep(10)    

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
    # r = SalmonnRedis(host="192.168.219.101", db=5)
    # gis = []
    # for i in range(3):
    #     gi = pd.read_csv(
    #         f"repr_exp/table3/GigaSpeech/test_chunks_000{i}_metadata.csv", sep=","
    #     )
    #     # https://github.com/SpeechColab/GigaSpeech?tab=readme-ov-file#text-pre-processing
    #     # will be addressed after inference using under
    #     # https://github.com/SpeechColab/GigaSpeech/blob/main/utils/gigaspeech_scoring.py
    #     gi = gi.rename(columns={"sid": "file_name", "text_tn": "sentence"})
    #     gis += gi[["file_name", "sentence"]].to_dict(orient="records")
    #     # print(gi)
    # r.initialize_tasks("GigaSpeech-ASR-test", gis)

    # ===== initialize LibriSpeech ASR tasks - LoRA Scaling TEST =====
    # from librispeech import get_librispeech_list

    # libri = get_librispeech_list()
    # for i, subset in enumerate(libri):
    #     r = SalmonnRedis(host="192.168.219.101", db=i + 2)
    #     for ls in range(0, 4):
    #         task_name = f"LibriSpeech-ASR-{subset}-ls{ls:02d}"
    #         r.initialize_tasks(task_name, libri[subset])
    # 2: test-clean, 3: test-other

    # ===== initialize GigaSpeech ASR tasks - LoRA Scaling TEST =====
    # r = SalmonnRedis(host="192.168.219.101", db=5)
    # gis = []
    # for i in range(3):
    #     gi = pd.read_csv(
    #         f"repr_exp/table3/GigaSpeech/test_chunks_000{i}_metadata.csv", sep=","
    #     )
    #     # https://github.com/SpeechColab/GigaSpeech?tab=readme-ov-file#text-pre-processing
    #     # will be addressed after inference using under
    #     # https://github.com/SpeechColab/GigaSpeech/blob/main/utils/gigaspeech_scoring.py
    #     gi = gi.rename(columns={"sid": "file_name", "text_tn": "sentence"})
    #     gis += gi[["file_name", "sentence"]].to_dict(orient="records")
    #     # print(gi)

    # for ls in range(0, 4):
    #     task_name = f"GigaSpeech-ASR-test-ls{ls:02d}"
    #     r.initialize_tasks(task_name, gis)

    # ===== initialize AudioCaps AAC tasks =====
    # # original dataset located under
    # # "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv"
    # ac = pd.read_csv(
    #     "repr_exp/table3/AudioCaps/test.csv",
    #     sep=",",
    # )
    # ac = ac.to_dict(orient="records")
    # # print(ac)
    # r = SalmonnRedis(host="192.168.219.101", db=6)
    # r.initialize_tasks("AudioCaps-AAC-test", ac)

    # ===== initialize LibriSpeech PR tasks =====
    # from librispeech import get_librispeech_pr

    # ds = "test-clean"
    # libri = get_librispeech_pr()
    # task_name = f"LibriSpeech-PR-{ds}"
    # r = SalmonnRedis(host="192.168.219.101", db=2)
    # r.initialize_tasks(task_name, libri)
    # 2: test-clean, 3: test-other

    # ===== initialize LibriSpeech PR tasks - LoRA Scaling TEST =====
    # from librispeech import get_librispeech_pr

    # ds = "test-clean"
    # libri = get_librispeech_pr()
    # task_name = f"LibriSpeech-PR-{ds}"
    # r = SalmonnRedis(host="192.168.219.101", db=2)
    # for ls in range(0, 4):
    #     task_name = f"LibriSpeech-PR-{ds}-ls{ls:02d}"
    #     r.initialize_tasks(task_name, libri)
