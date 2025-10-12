import os, json

sakura_path = "/home/jpong/Workspace/jaeeewon/repr/sakura"

sakura = []

for sets in os.listdir(os.path.join(sakura_path, "data")):
    if "." in sets:
        continue  # skip README.md

    path = os.path.join(sakura_path, "data", sets, "metadata.json")
    with open(path, "r") as f:
        metadata = json.load(f)

    for k, v in metadata.items():
        sakura.extend(
            [
                {
                    "audio": os.path.join(sakura_path, k),
                    "question": v["single_instruction"],
                    "gt": v["single_answer"],
                    "source": sets,
                },
                {
                    "audio": os.path.join(sakura_path, k),
                    "question": v["multi_instruction"],
                    "gt": v["multi_answer"],
                    "source": sets,
                },
            ]
        )

with open("sakura_ignore_qwenaudio_meta.jsonl", "w") as f:
    for item in sakura:
        f.write(json.dumps(item) + "\n")
# {"audio": "clothoqa/audio_files/river_mouth3.wav", "gt": "yes", "source": "clothoaqa_test", "question": "Are there waves?"}
