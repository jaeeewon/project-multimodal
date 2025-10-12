import os, json

sakura_path = "/home/jpong/Workspace/jaeeewon/repr/sakura"

sakura = []

for sets in os.listdir(os.path.join(sakura_path, "data")):
    if "." in sets:
        continue  # skip README.md
    print(sets)

    path = os.path.join(sakura_path, "data", sets, "metadata.json")
    with open(path, "r") as f:
        metadata = json.load(f)

    for k, v in metadata.items():
        sakura.append(
            {
                "id": len(sakura),
                "audio": os.path.join(sakura_path, k),
                "query": f"<|audio_bos|><|AUDIO|><|audio_eos|>\n{v['single_instruction']}",
                "answer": v["single_answer"],
            }
        )

        sakura.append(
            {
                "id": len(sakura),
                "audio": os.path.join(sakura_path, k),
                "query": f"<|audio_bos|><|AUDIO|><|audio_eos|>\n{v['multi_instruction']}",
                "answer": v["multi_answer"],
            }
        )

# with open("sakura_ignore_qwen2audio_eval.jsonl", "w") as f:
#     for item in sakura:
#         f.write(json.dumps(item) + "\n")
# {"id": 0, "audio": "AIR-Bench/level-3/wav-v1/speech_dialogue_QA_fisher/568.38_596.46.wav", "query": "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\nWhat does the first speaker imply about the national debt?"}
