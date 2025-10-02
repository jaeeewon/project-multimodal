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
        f"wer   : {jiwer.wer(answer, infer, reference_transform=reference_transform, hypothesis_transform=hypothesis_transform):.2f}"
    )
    print("=" * (15 + len(key + type)))


redis_dbs = [
    ("LibriSpeech-ASR-test-clean", 2),
    # ("LibriSpeech-ASR-test-other", 3),
    # ("GigaSpeech-ASR-test", 5),
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
        total_counts[i] += total_tasks

        if total_tasks == 0:
            print("no data")
            exit()

        answers = []
        infers = []

        for key in task_keys:
            task_data = r.hgetall(key)

            # if task_data.get("status") != "completed":
            #     continued += 1
            #     continue

            answer = task_data.get("sentence").strip()
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
ax1.set_ylabel("WER %", color=color)
ax1.plot(lora_scales, total_wers, "o-", color=color, label="WER")
ax1.tick_params(axis="y", labelcolor=color)
if skip:
    ax1.set_yticks([0, 0.2, 0.4, 0.6])
ax1.set_xticks(lora_scales)
ax1.grid(True)

ax2 = ax1.twinx()
color = "tab:orange"
ax2.set_ylabel("Data Usage Ratio", color=color)
ax2.plot(lora_scales, data_usage_ratios, "s--", color=color, label="Data Usage Ratio")
ax2.tick_params(axis="y", labelcolor=color)

plt.title("[ASR task] WER and Data Usage Ratio by LoRA Scaling Factor")
fig.tight_layout()
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper right")

plt.gca().invert_xaxis()
plt.savefig(
    f"repr_exp/figure3/LoRA_SF_ASR_{'origin' if not skip else 'postprocessed'}.png"
)

"""
LibriSpeech-ASR-test-clean-ls00 final WER: 0.45 (44.73%)
missed_eos: 4
LibriSpeech-ASR-test-clean-ls01 final WER: 0.36 (35.80%)
missed_eos: 2
LibriSpeech-ASR-test-clean-ls02 final WER: 0.10 (9.69%)
missed_eos: 1
LibriSpeech-ASR-test-clean-ls03 final WER: 0.04 (3.52%)
missed_eos: 1
LibriSpeech-ASR-test-clean final WER: 0.02 (2.24%)
LibriSpeech-ASR-test-other-ls00 final WER: 0.59 (58.66%)
missed_eos: 11
LibriSpeech-ASR-test-other-ls01 final WER: 0.47 (46.76%)
missed_eos: 5
LibriSpeech-ASR-test-other-ls02 final WER: 0.19 (18.71%)
missed_eos: 2
LibriSpeech-ASR-test-other-ls03 final WER: 0.08 (7.53%)
missed_eos: 2
LibriSpeech-ASR-test-other final WER: 0.05 (5.12%)
GigaSpeech-ASR-test-ls00 final WER: 0.87 (87.20%)
missed_eos: 6259
empty_answer: 3708
GigaSpeech-ASR-test-ls01 final WER: 0.46 (46.02%)
missed_eos: 3208
empty_answer: 3941
GigaSpeech-ASR-test-ls02 final WER: 0.27 (26.98%)
missed_eos: 1677
empty_answer: 4718
GigaSpeech-ASR-test-ls03 final WER: 0.16 (16.15%)
missed_eos: 1100
empty_answer: 5156
GigaSpeech-ASR-test final WER: 0.12 (11.56%)
missed_eos: 1165
empty_answer: 4577
=====
final WER for LoRA-scaled 0: 0.78 (77.73%) | missed_eos: 6274 | empty_answer: 3708 | 31178
final WER for LoRA-scaled 1: 0.45 (44.94%) | missed_eos: 3215 | empty_answer: 3941 | 31178
final WER for LoRA-scaled 2: 0.24 (24.21%) | missed_eos: 1680 | empty_answer: 4718 | 31178
final WER for LoRA-scaled 3: 0.14 (13.87%) | missed_eos: 1103 | empty_answer: 5156 | 31178
final WER for LoRA-scaled 4: 0.10 (9.89%) | missed_eos: 1165 | empty_answer: 4577 | 31178
"""
