import os, json

sakura_path = "/home/jpong/Workspace/jaeeewon/repr/sakura"


def get_sakura_ds(is_exp=False, exclude_answer: bool = False):
    sakura = []

    for sets in os.listdir(os.path.join(sakura_path, "data")):
        if "." in sets:
            continue  # skip README.md

        path = os.path.join(sakura_path, "data", sets, "metadata.json")
        with open(path, "r") as f:
            metadata = json.load(f)

        for i, (k, v) in enumerate(metadata.items()):
            if is_exp and i % 50:
                continue

            if exclude_answer:
                v["single_instruction"] = v["single_instruction"].split(" (")[0]
                v["multi_instruction"] = v["multi_instruction"].split(" (")[0]

            sakura.append(
                {
                    "id": len(sakura),
                    "wav": os.path.join(sakura_path, k),
                    "query": v["single_instruction"],
                    "text": v["single_answer"],
                    "task": "sakura",
                    "hop": "single",
                    "set": sets.lower(),
                    "local_index": i,
                }
            )

            sakura.append(
                {
                    "id": len(sakura),
                    "wav": os.path.join(sakura_path, k),
                    "query": v["multi_instruction"],
                    "text": v["multi_answer"],
                    "task": "sakura",
                    "hop": "multi",
                    "set": sets.lower(),
                    "local_index": i,
                }
            )

    return sakura


def get_sakura_wrong_ds(multi_only=True):
    ds = []
    with open("lalms/open-source/qwen2-audio-instruct/sakura_llm-as-a-judge.json", "r") as f:
        evaled = json.load(f)
    for i, (sakura, ev) in enumerate(zip(get_sakura_ds(), evaled)):
        if ev == "incorrect":
            if multi_only and i % 2 == 0:  # even index is single
                continue
            ds.append(sakura)
    return ds


if __name__ == "__main__":
    ds = get_sakura_ds(exclude_answer=True)
    for d in ds:
        print(d['query'])

    with open("ann/sakura_exclude_answer.json", "w") as f:
        json.dump({"annotation": ds}, f, indent=4)
