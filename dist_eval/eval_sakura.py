from init_works import SalmonnRedis
from collections import Counter, defaultdict
import pandas as pd, numpy as np, json, librosa, sys, math
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util.sakura_datasets import get_sakura_ds


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
corr_duration_counter = defaultdict(list[int])
incorr_counter = Counter()
incorr_duration_counter = defaultdict(list[int])
fail_counter = Counter()
fail_duration_counter = defaultdict(list[int])
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

            task_keys = [k for k in r.client.scan_iter(f"{TASK_HASH_PREFIX}*") if not k.startswith(tuple(passkeys))]

            for key in task_keys:
                task_data = r.client.hgetall(key)

                status = task_data.get("status")
                file = task_data.get("file")

                audio, sr = librosa.load(file, sr=16000)
                duration = audio.shape[0] / sr

                if status == f"completed{sakura_judge_pf}":
                    judgement = task_data.get(f"Judgement{sakura_judge_pf}")
                    if judgement == "correct":
                        corr_counter[task_name] += 1
                        corr_duration_counter[task_name].append(duration)
                    elif judgement == "incorrect":
                        incorr_counter[task_name] += 1
                        incorr_duration_counter[task_name].append(duration)
                    else:
                        fail_counter[task_name] += 1
                        fail_duration_counter[task_name].append(duration)

            print(f"===== {task_name} =====")
            print(f"correct: {corr_counter[task_name]}")
            print(
                f"correct avg duration: {np.mean(corr_duration_counter[task_name]) if corr_duration_counter[task_name] else 0:.4f}"
            )
            print(f"incorrect: {incorr_counter[task_name]}")
            print(
                f"incorrect avg duration: {np.mean(incorr_duration_counter[task_name]) if incorr_duration_counter[task_name] else 0:.4f}"
            )
            print(f"failed: {fail_counter[task_name]}")
            print(
                f"failed avg duration: {np.mean(fail_duration_counter[task_name]) if fail_duration_counter[task_name] else 0:.4f}"
            )
            total = corr_counter[task_name] + incorr_counter[task_name] + fail_counter[task_name]
            print(f"total: {total}")
            if total > 0:
                print(f"accuracy: {corr_counter[task_name] / total:.4f}")
                data["track"].append(track)
                data["hop"].append(hop)
                data["accuracy"].append(corr_counter[task_name] / total)
                data["model"].append(f"SALMONN-{size}B")
                data["total"].append(total)
            print()


sakura_ds = get_sakura_ds()
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
    for i, (judg, sakura_d) in enumerate(zip(judgs, sakura_ds)):
        # (500 each for single and multi)
        # 0 ~ 999: Emotion
        # 1000 ~ 1999: Language
        # 2000 ~ 2999: Animal
        # 3000 ~ 3999: Gender
        track = sakura_tracks[i // 1000]
        hop = "multi" if i % 2 else "single"
        task_name = f"{model}-SAKURA-{track}-{hop}"
        sound, sr = librosa.load(sakura_d["path"], sr=16000)
        duration = sound.shape[0] / sr
        if judg == "correct":
            corr_counter[task_name] += 1
            corr_duration_counter[task_name].append(duration)
        elif judg == "incorrect":
            incorr_counter[task_name] += 1
            incorr_duration_counter[task_name].append(duration)
        else:
            fail_counter[task_name] += 1
            fail_duration_counter[task_name].append(duration)

        if i % 500 in [498, 499]:
            print(f"===== {task_name} =====")
            print(f"correct: {corr_counter[task_name]}")
            print(
                f"correct avg duration: {np.mean(corr_duration_counter[task_name]) if corr_duration_counter[task_name] else 0:.4f}"
            )
            print(f"incorrect: {incorr_counter[task_name]}")
            print(
                f"incorrect avg duration: {np.mean(incorr_duration_counter[task_name]) if incorr_duration_counter[task_name] else 0:.4f}"
            )
            print(f"failed: {fail_counter[task_name]}")
            print(
                f"failed avg duration: {np.mean(fail_duration_counter[task_name]) if fail_duration_counter[task_name] else 0:.4f}"
            )
            total = corr_counter[task_name] + incorr_counter[task_name] + fail_counter[task_name]
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
    pd.DataFrame([paper_data], index=[paper_name], columns=df_final.columns[:-1]).rename_axis(
        index="Model", inplace=False
    )
    for paper_name, paper_data in papers.items()
]

df_final = pd.concat([df_final] + dfs)
df_final.sort_index(inplace=True)
df_final["CCC"] = df_final["CCC"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

df_final.to_markdown("results/sakura.md")

# merge keys for same model
corr_duration_counter_merged = defaultdict(list[int])
incorr_duration_counter_merged = defaultdict(list[int])
fail_duration_counter_merged = defaultdict(list[int])
for key in corr_duration_counter:
    splt = key.split("SAKURA-")
    base_key = splt[0]
    if not base_key:
        base_key = "SALMONN-{}B".format("7" if "-7B" in key else "13")
    corr_duration_counter_merged[base_key].extend(corr_duration_counter[key])
for key in incorr_duration_counter:
    splt = key.split("SAKURA-")
    base_key = splt[0]
    if not base_key:
        base_key = "SALMONN-{}B".format("7" if "-7B" in key else "13")
    incorr_duration_counter_merged[base_key].extend(incorr_duration_counter[key])
for key in fail_duration_counter:
    splt = key.split("SAKURA-")
    base_key = splt[0]
    if not base_key:
        base_key = "SALMONN-{}B".format("7" if "-7B" in key else "13")
    fail_duration_counter_merged[base_key].extend(fail_duration_counter[key])
corr_duration_counter = corr_duration_counter_merged
incorr_duration_counter = incorr_duration_counter_merged
fail_duration_counter = fail_duration_counter_merged

# the code below is written by Copilot

import matplotlib.pyplot as plt

# Collect all task names present in any of the counters
tasks = sorted(
    set(list(corr_duration_counter.keys()) + list(incorr_duration_counter.keys()) + list(fail_duration_counter.keys()))
)

if not tasks:
    print("No task duration data available to plot.")
else:
    # Determine global max duration to set bins
    all_durations = []
    for dlist in (
        list(corr_duration_counter.values())
        + list(incorr_duration_counter.values())
        + list(fail_duration_counter.values())
    ):
        all_durations.extend(dlist)
    max_dur = max(all_durations) if all_durations else 0.0
    # Use 1-second bins up to ceiling of max_dur (at least 10s for visibility)
    max_edge = max(10, math.ceil(max_dur))
    bins = np.arange(0, max_edge + 1, 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_widths = np.diff(bins)

    n_tasks = len(tasks)
    ncols = 2
    nrows = math.ceil(n_tasks / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 4 * nrows), squeeze=False)
    for idx, task in enumerate(tasks):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        corr_d = np.array(corr_duration_counter.get(task, []))
        incorr_d = np.array(incorr_duration_counter.get(task, []))
        fail_d = np.array(fail_duration_counter.get(task, []))

        corr_counts, _ = np.histogram(corr_d, bins=bins)
        incorr_counts, _ = np.histogram(incorr_d, bins=bins)
        fail_counts, _ = np.histogram(fail_d, bins=bins)

        total_counts = corr_counts + incorr_counts + fail_counts
        # accuracy per bin, avoid division by zero
        accuracy = np.zeros_like(total_counts, dtype=float)
        nonzero = total_counts > 0
        accuracy[nonzero] = corr_counts[nonzero] / total_counts[nonzero]
        accuracy[~nonzero] = np.nan  # show gaps where no samples

        # Left axis: accuracy (0-1)
        ax.plot(bin_centers, accuracy, marker="o", color="#1f77b4", linestyle="-", label="Accuracy")
        ax.set_ylim(0, 1)
        ax.set_xlim(bin_centers[0] - 0.5, bin_centers[-1] + 0.5)
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Accuracy")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Right axis: sample distribution (stacked bars)
        ax2 = ax.twinx()
        left_edges = bins[:-1]
        width = bin_widths

        p1 = ax2.bar(left_edges, corr_counts, width=width, align="edge", color="#2ca02c", alpha=0.7, label="Correct")
        p2 = ax2.bar(
            left_edges,
            incorr_counts,
            width=width,
            align="edge",
            bottom=corr_counts,
            color="#ff7f0e",
            alpha=0.7,
            label="Incorrect",
        )
        p3 = ax2.bar(
            left_edges,
            fail_counts,
            width=width,
            align="edge",
            bottom=(corr_counts + incorr_counts),
            color="#d62728",
            alpha=0.7,
            label="Failed",
        )

        max_count = total_counts.max() if total_counts.size else 0
        ax2.set_ylim(0, max(5, int(max_count * 1.15)))  # leave some headroom
        ax2.set_ylabel("Samples (count)")

        # Overall totals and accuracy in title
        overall_total = int(total_counts.sum())
        overall_acc = (
            corr_counter.get(task, 0)
            / (corr_counter.get(task, 0) + incorr_counter.get(task, 0) + fail_counter.get(task, 0))
            if (corr_counter.get(task, 0) + incorr_counter.get(task, 0) + fail_counter.get(task, 0)) > 0
            else 0.0
        )
        ax.set_title(f"{task} — N={overall_total}, overall acc={overall_acc:.3f}")

        # Legends: combine line and bar legends
        bars_legend = [p1, p2, p3]
        labels = ["Correct", "Incorrect", "Failed"]
        l1 = ax.legend([plt.Line2D([0], [0], color="#1f77b4", marker="o")], ["Accuracy"], loc="upper left")
        l2 = ax2.legend(bars_legend, labels, loc="upper right")
        ax.add_artist(l1)

    # Hide any empty subplots
    total_subplots = nrows * ncols
    for empty_idx in range(n_tasks, total_subplots):
        r = empty_idx // ncols
        c = empty_idx % ncols
        axes[r][c].axis("off")

    plt.tight_layout()
    out_path = "results/sakura_accuracy_by_time.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved accuracy-vs-duration plot to {out_path}")
    plt.show()
