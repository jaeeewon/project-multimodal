from collections import Counter
import pandas as pd, json
import numpy as np


corr_counter = Counter()
incorr_counter = Counter()
fail_counter = Counter()
data = {"track": [], "hop": [], "accuracy": [], "size": [], "total": []}

# judgs = ["incorrect" if i % 13 == 0 else "correct" for i in range(4000)]

sakura_tracks = ["Emotion", "Language", "Animal", "Gender"]
for model, pth in [
    (
        "qwen2-audio-instruct",
        "lalms/open-source/qwen2-audio-instruct/sakura_llm-as-a-judge.json",
    ),
    (
        "qwen-audio-chat",
        "lalms/open-source/qwen-audio-chat/sakura_llm-as-a-judge.json",
    ),
]:
    with open(pth, "r") as f:
        judgs = json.load(f)
    for i, judg in enumerate(judgs):
        # (500 each for single and multi)
        # 0 ~ 999: Emotion
        # 1000 ~ 1999: Language
        # 2000 ~ 2999: Animal
        # 3000 ~ 3999: Gender
        track = sakura_tracks[i // 1000]
        hop = "multi" if i % 2 else "single"
        task_name = f"{model}-SAKURA-{track}-{hop}"
        if judg == "correct":
            corr_counter[task_name] += 1
        elif judg == "incorrect":
            incorr_counter[task_name] += 1
        else:
            fail_counter[task_name] += 1

        if i % 500 in [498, 499]:
            print(f"===== {task_name} =====")
            print(f"correct: {corr_counter[task_name]}")
            print(f"incorrect: {incorr_counter[task_name]}")
            print(f"failed: {fail_counter[task_name]}")
            total = (
                corr_counter[task_name]
                + incorr_counter[task_name]
                + fail_counter[task_name]
            )
            print(f"total: {total}")
            if total > 0:
                print(f"accuracy: {corr_counter[task_name] / total:.4f}")
                data["track"].append(track)
                data["hop"].append(hop)
                data["accuracy"].append(corr_counter[task_name] / total)
                data["size"].append(model)
                data["total"].append(total)
            print()

df = pd.DataFrame(data)

p = df["accuracy"]
n = df["total"]
df["ci"] = 1.96 * np.sqrt(p * (1 - p) / n + 1e-9)

df["accuracy"] = df["accuracy"] * 100
df["ci"] = df["ci"] * 100

df_acc = df.pivot_table(index="size", columns=["track", "hop"], values="accuracy")
df_ci = df.pivot_table(index="size", columns=["track", "hop"], values="ci")

track_order = ["Gender", "Language", "Emotion", "Animal"]
ordered_columns = [(track, hop) for track in track_order for hop in ["single", "multi"]]
df_acc = df_acc[ordered_columns]
df_ci = df_ci[ordered_columns]

df_acc[("Average", "single")] = df_acc.loc[:, (track_order, "single")].mean(axis=1)
df_acc[("Average", "multi")] = df_acc.loc[:, (track_order, "multi")].mean(axis=1)
df_ci[("Average", "single")] = df_ci.loc[:, (track_order, "single")].mean(axis=1)
df_ci[("Average", "multi")] = df_ci.loc[:, (track_order, "multi")].mean(axis=1)

new_columns = [f"{track} ({hop})" for track, hop in df_acc.columns]
df_acc.columns = new_columns
df_ci.columns = new_columns

df_final = df_acc.applymap("{:.1f}".format) + " ± " + df_ci.applymap("{:.1f}".format)

df_final.rename_axis(index="Size (B)", inplace=True)
df_final.columns.names = [None]

df_final.to_markdown("results/qwenaudio_sakura.md")
