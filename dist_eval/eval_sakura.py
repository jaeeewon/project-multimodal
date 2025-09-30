from init_works import SalmonnRedis
from collections import Counter
import pandas as pd
import numpy as np

r = SalmonnRedis(host="192.168.219.101", db=7)
sakura_tracks = ["Animal", "Emotion", "Gender", "Language"]
sakura_judge_pf = "-judge-qwen3"

corr_counter = Counter()
incorr_counter = Counter()
fail_counter = Counter()
data = {"track": [], "hop": [], "accuracy": [], "size": [], "total": []}

for pf, size in [("", 13), ("-7B", 7)]:
    for track in sakura_tracks:
        for hop in ["single", "multi"]:
            task_name = f"SAKURA-{track}-{hop}{pf}"
            TASK_HASH_PREFIX = SalmonnRedis.TASK_HASH_PREFIX.format(task_name)
            passkeys = [
                SalmonnRedis.PENDING_QUEUE.format(task_name),
                SalmonnRedis.PROCESSING_QUEUE.format(task_name),
                SalmonnRedis.PENDING_QUEUE.format(task_name) + sakura_judge_pf,
                SalmonnRedis.PROCESSING_QUEUE.format(task_name) + sakura_judge_pf,
            ]

            task_keys = [
                k
                for k in r.client.scan_iter(f"{TASK_HASH_PREFIX}*")
                if not k.startswith(tuple(passkeys))
            ]

            for key in task_keys:
                task_data = r.client.hgetall(key)

                status = task_data.get("status")

                if status == f"completed{sakura_judge_pf}":
                    judgement = task_data.get(f"Judgement{sakura_judge_pf}")
                    if judgement == "correct":
                        corr_counter[task_name] += 1
                    elif judgement == "incorrect":
                        incorr_counter[task_name] += 1
                    else:
                        fail_counter[task_name] += 1

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
                data["size"].append(size)
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

paper_data = [
    "59.8 ± 4.3",
    "48.6 ± 4.4",  # Gender (single, multi)
    "21.8 ± 3.6",
    "29.6 ± 4.0",  # Language (single, multi)
    "19.8 ± 3.5",
    "28.2 ± 3.9",  # Emotion (single, multi)
    "68.6 ± 4.1",
    "34.6 ± 4.2",  # Animal (single, multi)
    "42.5 ± 4.3",
    "35.3 ± 4.2",  # Average (single, multi)
]

df_paper = pd.DataFrame([paper_data], index=["Paper"], columns=df_final.columns)
df_paper.rename_axis(index="Size (B)", inplace=True)

df_final = pd.concat([df_final, df_paper])

df_final.to_markdown("results/salmonn_sakura.md")

"""
# SALMONN 14B with 8bit quantized LLM

===== SAKURA-Animal-single =====
correct: 365
incorrect: 135
failed: 0
total: 500
accuracy: 0.7300

===== SAKURA-Animal-multi =====
correct: 232
incorrect: 267
failed: 1
total: 500
accuracy: 0.4640

===== SAKURA-Emotion-single =====
correct: 155
incorrect: 341
failed: 4
total: 500
accuracy: 0.3100

===== SAKURA-Emotion-multi =====
correct: 159
incorrect: 339
failed: 2
total: 500
accuracy: 0.3180

===== SAKURA-Gender-single =====
correct: 269
incorrect: 231
failed: 0
total: 500
accuracy: 0.5380

===== SAKURA-Gender-multi =====
correct: 245
incorrect: 253
failed: 2
total: 500
accuracy: 0.4900

===== SAKURA-Language-single =====
correct: 110
incorrect: 385
failed: 5
total: 500
accuracy: 0.2200

===== SAKURA-Language-multi =====
correct: 110
incorrect: 383
failed: 7
total: 500
accuracy: 0.2200

# SALMONN 14B with 8bit quantized LLM

===== SAKURA-Animal-single-7B =====
correct: 340
incorrect: 159
failed: 1
total: 500
accuracy: 0.6800

===== SAKURA-Animal-multi-7B =====
correct: 173
incorrect: 323
failed: 4
total: 500
accuracy: 0.3460

===== SAKURA-Emotion-single-7B =====
correct: 98
incorrect: 399
failed: 3
total: 500
accuracy: 0.1960

===== SAKURA-Emotion-multi-7B =====
correct: 141
incorrect: 356
failed: 3
total: 500
accuracy: 0.2820

===== SAKURA-Gender-single-7B =====
correct: 300
incorrect: 200
failed: 0
total: 500
accuracy: 0.6000

===== SAKURA-Gender-multi-7B =====
correct: 244
incorrect: 255
failed: 1
total: 500
accuracy: 0.4880

===== SAKURA-Language-single-7B =====
correct: 103
incorrect: 396
failed: 1
total: 500
accuracy: 0.2060

===== SAKURA-Language-multi-7B =====
correct: 147
incorrect: 347
failed: 6
total: 500
accuracy: 0.2940
"""
