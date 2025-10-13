from init_works import SalmonnRedis
from collections import Counter
import pandas as pd, numpy as np, json


def concordance_correlation_coefficient(y_true, y_pred):
    """gen by gemini"""
    # NumPy 배열로 변환
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 평균 계산
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # 분산 계산
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    # 공분산 계산
    # ddof=0은 N으로 나누는 모공분산을 의미합니다.
    cov = np.cov(y_true, y_pred, ddof=0)[0, 1]

    # CCC 수식의 분자 계산
    # 2 * rho * sigma_x * sigma_y는 2 * cov(x, y)와 같습니다.
    numerator = 2 * cov

    # CCC 수식의 분모 계산
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    # CCC 계산
    ccc = numerator / denominator

    return ccc


r = SalmonnRedis(host="salmonn.hufs.jae.one", db=7)
sakura_tracks = ["Animal", "Emotion", "Gender", "Language"]
sakura_judge_pf = "-judge-qwen3"

corr_counter = Counter()
incorr_counter = Counter()
fail_counter = Counter()
data = {"track": [], "hop": [], "accuracy": [], "model": [], "total": []}

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
                data["model"].append(f"SALMONN-{size}B")
                data["total"].append(total)
            print()


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
                data["model"].append(model)
                data["total"].append(total)
            print()

df = pd.DataFrame(data)

p = df["accuracy"]
n = df["total"]
df["ci"] = 1.96 * np.sqrt(p * (1 - p) / n + 1e-9)

df["accuracy"] = df["accuracy"] * 100
df["ci"] = df["ci"] * 100

df_acc = df.pivot_table(index="model", columns=["track", "hop"], values="accuracy")
df_ci = df.pivot_table(index="model", columns=["track", "hop"], values="ci")

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

df_final.rename_axis(index="Model", inplace=True)
df_final.columns.names = [None]

papers = {
    "SALMONN-paper": [
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
    ],
    "qwen-audio-chat-paper": [
        "49.6 ± 4.4",
        "43.8 ± 4.3",  # Gender (single, multi)
        "87.6 ± 2.9",
        "40.6 ± 4.3",  # Language (single, multi)
        "63.2 ± 4.2",
        "37.0 ± 4.2",  # Emotion (single, multi)
        "92.2 ± 2.4",
        "66.0 ± 4.2",  # Animal (single, multi)
        "73.2 ± 3.9",
        "46.9 ± 4.4",  # Average (single, multi)
    ],
    "qwen2-audio-instruct-paper": [
        "88.0 ± 2.8",
        "47.2 ± 4.4",  # Gender (single, multi)
        "83.8 ± 3.2",
        "48.0 ± 4.4",  # Language (single, multi)
        "64.2 ± 4.2",
        "39.8 ± 4.3",  # Emotion (single, multi)
        "88.8 ± 2.8",
        "61.4 ± 4.3",  # Animal (single, multi)
        "81.2 ± 3.4",
        "49.1 ± 4.4",  # Average (single, multi)
    ],
}

ccc_results = {}
model_to_paper_map = {
    "SALMONN-13B": "SALMONN-paper",
    "SALMONN-7B": "SALMONN-paper",
    "qwen-audio-chat": "qwen-audio-chat-paper",
    "qwen2-audio-instruct": "qwen2-audio-instruct-paper",
}

for model_name, paper_key in model_to_paper_map.items():
    if model_name in df_acc.index:
        paper_accuracies = [float(s.split("±")[0].strip()) for s in papers[paper_key]]
        y_true = np.array(paper_accuracies)

        y_pred = df_acc.loc[model_name].values

        ccc = concordance_correlation_coefficient(y_true, y_pred)
        ccc_results[model_name] = ccc
        # df_final.loc[model_name] = df_final.loc[model_name].apply(
        #     lambda x: f"{x} / {ccc:.4f}"
        # )
        print(f"CCC for {model_name} (vs {paper_key}): {ccc:.4f}")

ccc_series = pd.Series(ccc_results, name="CCC")
df_final = df_final.join(ccc_series)

dfs = [
    pd.DataFrame(
        [paper_data], index=[paper_name], columns=df_final.columns[:-1]
    ).rename_axis(index="Model", inplace=False)
    for paper_name, paper_data in papers.items()
]

df_final = pd.concat([df_final] + dfs)
df_final.sort_index(inplace=True)
df_final["CCC"] = df_final["CCC"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

df_final.to_markdown("results/sakura.md")
