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


def count_tasks(ann: dict, store_dict: dict, store_list: list):
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


if __name__ == "__main__":
    train_sets = [
        "salmonn_stage1_data.json",
        "salmonn_stage2_data.json",
        "salmonn_stage3_data.json",
    ]

    for train_set in train_sets:
        print(f"===== {train_set} =====")
        stats = inspect_anns(f"./ann/{train_set}", count_tasks)
        print(*stats)
