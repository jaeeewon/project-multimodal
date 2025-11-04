import os, json

sakura_path = "/home/jpong/Workspace/jaeeewon/repr/sakura"


def get_sakura_ds():
    sakura = []

    for sets in os.listdir(os.path.join(sakura_path, "data")):
        if "." in sets:
            continue  # skip README.md

        path = os.path.join(sakura_path, "data", sets, "metadata.json")
        with open(path, "r") as f:
            metadata = json.load(f)

        for i, (k, v) in enumerate(metadata.items()):
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
    ds = get_sakura_ds()

    with open("ann/sakura.json", "w") as f:
        json.dump({"annotation": ds}, f, indent=4)
