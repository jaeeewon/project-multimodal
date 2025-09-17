import os
import ijson
from typing import Callable, Any
from collections.abc import Iterator


def inspect_anns(
    filename: str, fn: Callable[[dict[str, Any], dict, list], Any]
) -> tuple[dict, list]:
    """
    call by reference인 dict와 list를 이용해 `fn(ann, dict, list) -> Any`의 형태로 annotation을 분석함

    `tuple[dict, list]`의 형태로 반환
    """
    store_dict = {}
    store_list = []
    with open(filename, "rb") as f:
        anns: Iterator[dict] = ijson.items(f, "annotation.item")
        for ann in anns:
            fn(ann, store_dict, store_list)

    return (store_dict, store_list)


class Inspector:
    train_sets = [
        "salmonn_stage1_data.json",
        "salmonn_stage2_data.json",
        "salmonn_stage3_data.json",
    ]

    def __init__(self, train_sets=None):
        if train_sets is not None:
            self.train_sets = train_sets

    def _count_tasks(self, ann: dict, store_dict: dict, store_list: list):
        """
        ===== salmonn_stage1_data.json =====
        {'asr': 1191381, 'audiocaption': 435174} []

        ===== salmonn_stage2_data.json =====
        {'asr': 481241, 'audiocaption': 48267, 'audiocaption_v2': 19195, 'translation_ec': 289354, 'phone_recognition': 281239, 'emotion_recognition': 4090, 'music_description': 2643, 'QA': 648163, 'speech_separation': 64700, 'speaker_verification': 523411, 'gender_recognition': 28539} []

        ===== salmonn_stage3_data.json =====
        {'audio_story_telling': 48272} []
        """
        task = ann.get("task")
        if task not in store_dict:
            store_dict[task] = 0
        store_dict[task] += 1

    def _check_data_exists(self, ann: dict, store_dict: dict, store_list: list):
        task = ann.get("task")
        # {"path": "/LibriSpeech/train-clean-100/path_to_audio.flac", "text": "recognized", "task": "asr"}
        path = ann.get("path")
        splt_path = path.split("/")
        dataset = splt_path[1]  # LibriSpeech
        file_name = splt_path[-1]  # path_to_audio.flac

        key = "ready" if os.path.exists(path) else "not_ready"

        if task not in store_dict:
            store_dict[task] = {}
        if dataset not in store_dict[task]:
            if key == "not_ready":
                print(f"[{task}] {dataset} | {path} not exists!")
            store_dict[task][dataset] = {"ready": [], "not_ready": []}

        store_dict[task][dataset][key].append(file_name)

    def get_stats(self):
        for train_set in self.train_sets:
            print(f"===== {train_set} =====")
            store_dict, _ = inspect_anns(f"./ann/{train_set}", self._count_tasks)
            print(store_dict)

    def get_ready_rate(self, print_statics=True):
        for train_set in self.train_sets:
            print(f"===== {train_set} =====")
            store_dict, _ = inspect_anns(f"./ann/{train_set}", self._check_data_exists)
            if print_statics:
                for task in store_dict:
                    for dataset in store_dict[task]:
                        ready = len(store_dict[task][dataset]["ready"])
                        not_ready = len(store_dict[task][dataset]["not_ready"])
                        total = ready + not_ready
                        ready_rate = (ready / total) * 100 if total > 0 else 0
                        print(
                            f"[{task}] {dataset} | ready: {ready}, not_ready: {not_ready}, total: {total} | ready_rate: {ready_rate:.2f}%"
                        )


if __name__ == "__main__":
    inspector = Inspector()
    # inspector.get_stats()
    # inspector.get_ready_rate(print_statics=False)
