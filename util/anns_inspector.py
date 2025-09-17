import os
import ijson
import soundfile as sf
from typing import Callable, Any
from collections.abc import Iterator


def inspect_anns(
    filename: str, fn: Callable[[dict[str, Any], dict, dict, list], Any], cfg: dict = {}
) -> tuple[dict, list]:
    """
    call by reference인 dict와 list를 이용해 `fn(ann, cfg, dict, list) -> Any`의 형태로 annotation을 분석함

    `tuple[dict, list]`의 형태로 반환
    """
    store_dict = {}
    store_list = []
    with open(filename, "rb") as f:
        anns: Iterator[dict] = ijson.items(f, "annotation.item")
        for ann in anns:
            fn(ann, cfg, store_dict, store_list)

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

    def _count_tasks(self, ann: dict, cfg: dict, store_dict: dict, store_list: list):
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

    def _check_data_exists(
        self, ann: dict, cfg: dict, store_dict: dict, store_list: list
    ):
        task = ann.get("task")
        # {"path": "/LibriSpeech/train-clean-100/path_to_audio.flac", "text": "recognized", "task": "asr"}
        path = ann.get("path")
        prefix = cfg.get("prefix")
        splt_path = path.split("/")
        path_dir = "/".join(splt_path[:-1])
        dataset = splt_path[1]  # LibriSpeech

        if dataset == "CommonVoice":
            path = path.replace(".wav", ".mp3")
        # CommonVoice 데이터셋은 .mp3로 구성됐으므로 오직 빠진 파일이 있는지 체크하는 용도로만 사용함

        file_name = splt_path[-1]  # path_to_audio.flac
        real_path = f"{prefix}{path}"

        key = "ready" if os.path.exists(real_path) else "not_ready"

        if task not in store_dict:
            store_dict[task] = {}
        if dataset not in store_dict[task]:
            if key == "ready":
                inspect_audio = cfg.get("inspect_audio", True)
                if inspect_audio:
                    try:
                        info = sf.info(real_path)
                        print(f"===== [{task}] {dataset} | {path} =====")
                        print(f"file path: {path_dir}")
                        print(f"sampling rate: {info.samplerate}")
                        print(f"channels: {info.channels}")
                        print(f"format: {info.format_info} ({info.format})")
                        print(f"duration: {info.duration:.2f}s")
                    except Exception as e:
                        print(f"failed to load audio for {real_path}: {e}")
            else:
                print_not_exists = cfg.get("print_not_exists", True)
                if print_not_exists:
                    print(f"[{task}] {dataset} | {path_dir} required!")
            store_dict[task][dataset] = {"ready": [], "not_ready": []}

        store_dict[task][dataset][key].append(file_name)

    def get_stats(self):
        for train_set in self.train_sets:
            print(f"===== {train_set} =====")
            store_dict, _ = inspect_anns(f"./ann/{train_set}", self._count_tasks)
            print(store_dict)

    def get_ready_rate(
        self, print_statics=True, inspect_audio=True, print_not_exists=True, prefix="./"
    ):
        for train_set in self.train_sets:
            print(f"===== {train_set} =====")
            store_dict, _ = inspect_anns(
                f"./ann/{train_set}",
                self._check_data_exists,
                {
                    "inspect_audio": inspect_audio,
                    "print_not_exists": print_not_exists,
                    "prefix": prefix,
                },
            )
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
    inspector.get_ready_rate(
        print_statics=True,
        inspect_audio=False,
        print_not_exists=False,
        prefix="/home/jpong/Workspace/jaeeewon",
    )
