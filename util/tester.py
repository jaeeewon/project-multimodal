from anns_inspector import inspect_anns

target_task = "translation_ec"


def covst2_test(ann: dict, cfg: dict, store_dict: dict, store_list: list):
    task = ann.get("task")
    if task != target_task:
        return

    if target_task not in store_dict:
        store_dict[target_task] = set()

    store_dict[target_task].add(ann.get("path").split("/")[2])


if __name__ == "__main__":
    store_dict, _ = inspect_anns("./ann/salmonn_stage2_data.json", fn=covst2_test)
    print(store_dict[target_task]) # {'train'}

# translation_ec의 데이터셋 종류 확인용