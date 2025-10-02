import redis, jiwer
import matplotlib.pyplot as plt

_PENDING_QUEUE = "task_{}:pending"
_PROCESSING_QUEUE = "task_{}:processing"
_TASK_HASH_PREFIX = "task_{}:"

reference_transform = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)
hypothesis_transform = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

remove_kaldi_non_words = jiwer.RemoveKaldiNonWords()


def printer(key, answer, infer, type):
    print(f"===== {type} - {key} =====")
    print(f"answer: {' '.join(reference_transform(answer)[0])}")
    print(f"infer : {' '.join(hypothesis_transform(infer)[0])}")
    print(f"answer: {answer}")
    print(f"infer : {infer}")
    print(
        f"per   : {jiwer.wer(answer, infer, reference_transform=reference_transform, hypothesis_transform=hypothesis_transform):.2f}"
    )
    print("=" * (15 + len(key + type)))


redis_dbs = [
    ("LibriSpeech-PR-test-clean", 2),
]

skip = True

total_answers = [[], [], [], [], []]
total_infers = [[], [], [], [], []]
total_missed_eos = [0, 0, 0, 0, 0]
total_empty_answer = [0, 0, 0, 0, 0]
total_counts = [0, 0, 0, 0, 0]

missed_eos = 0
empty_answer = 0

for task, db in redis_dbs:
    r = redis.Redis(host="salmonn.hufs.jae.one", port=6379, db=db, decode_responses=True)
    for i in range(5):
        curr_missed_eos = 0
        curr_empty_answer = 0
        task_name = task if i == 4 else f"{task}-ls{i:02d}"

        TASK_HASH_PREFIX = _TASK_HASH_PREFIX.format(task_name)
        passkeys = [
            _PENDING_QUEUE.format(task_name),
            _PROCESSING_QUEUE.format(task_name),
        ]

        task_keys = [
            k for k in r.scan_iter(f"{TASK_HASH_PREFIX}*") if k not in passkeys
        ]
        total_tasks = len(task_keys)
        # total_counts[i] += total_tasks

        if total_tasks == 0:
            print("no data")
            exit()

        answers = []
        infers = []

        for key in task_keys:
            task_data = r.hgetall(key)

            if task_data.get("status") != "completed":
                continue
            total_counts[i] += 1

            answer = task_data.get("text").strip()
            infer = task_data.get("infer").strip()

            if skip and not infer.endswith("</s>"):
                curr_missed_eos += 1
                continue

            if skip and db == 5 and len(remove_kaldi_non_words(answer)) == 0:
                # skip empty answers in GigaSpeech
                # ex. <MUSIC>, <NOISE> ...
                curr_empty_answer += 1
                continue

            answers.append(answer)
            infers.append(infer)
            # if task_name == "GigaSpeech-ASR-test":
            #     printer(key, answer, infer, task_name)

        final_wer = jiwer.wer(
            answers,
            infers,
            reference_transform=reference_transform,
            hypothesis_transform=hypothesis_transform,
        )

        print(f"{task_name} final WER: {final_wer:.2f} ({final_wer*100:.2f}%)")
        if curr_missed_eos > 0:
            print(f"missed_eos: {curr_missed_eos}")
            total_missed_eos[i] += curr_missed_eos
        if db == 5 and curr_empty_answer > 0:
            print(f"empty_answer: {curr_empty_answer}")
            total_empty_answer[i] += curr_empty_answer

        curr_missed_eos, curr_empty_answer = 0, 0

        total_answers[i].extend(answers)
        total_infers[i].extend(infers)
print("=====")

total_wers = []

for i, task in enumerate(total_answers):
    final_wer = jiwer.wer(
        total_answers[i],
        total_infers[i],
        reference_transform=reference_transform,
        hypothesis_transform=hypothesis_transform,
    )
    print(
        f"final WER for LoRA-scaled {i}: {final_wer:.2f} ({final_wer*100:.2f}%) | missed_eos: {total_missed_eos[i]} | empty_answer: {total_empty_answer[i]} | {total_counts[i]}"
    )
    total_wers.append(final_wer)

data_usage_ratios = [
    1 - (total_empty_answer[i] + total_missed_eos[i]) / total_counts[i]
    for i in range(5)
]
lora_scales = [0, 1, 2, 3, 4]

# plot using gemini

fig, ax1 = plt.subplots()

color = "tab:blue"
ax1.set_xlabel("LoRA scaling factor")
ax1.set_ylabel("PER %", color=color)
ax1.plot(lora_scales, total_wers, "o-", color=color, label="PER")
ax1.tick_params(axis="y", labelcolor=color)
if skip:
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax1.set_xticks(lora_scales)
ax1.grid(True)

ax2 = ax1.twinx()
color = "tab:orange"
ax2.set_ylabel("Data Usage Ratio", color=color)
ax2.plot(lora_scales, data_usage_ratios, "s--", color=color, label="Data Usage Ratio")
ax2.tick_params(axis="y", labelcolor=color)

plt.title("[PR task] PER and Data Usage Ratio by LoRA Scaling Factor")
fig.tight_layout()
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper right")

plt.gca().invert_xaxis()
plt.savefig(
    f"repr_exp/figure3/LoRA_SF_PR_{'origin' if not skip else 'postprocessed'}.png"
)
