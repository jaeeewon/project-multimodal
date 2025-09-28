# 사용되지 않음
# 아까워서 남겨둠
# dist_eval/eval_wer.py 참고
import redis, jiwer
from collections import Counter

PENDING_QUEUE = "task_{}:pending"
PROCESSING_QUEUE = "task_{}:processing"
TASK_HASH_PREFIX = "task_{}:"

r = redis.Redis(host="192.168.219.101", port=6379, db=3, decode_responses=True)

print_types = [
    # "valid"
    # "speech_tagged",
    # "garbage_prefix_v2"
    # "garbage_prefix_v3",
    # "garbage_prefix_v1"
    # "garbage_prefix_v4"
    # "sorry_rejected"
    # "no_eos",
    "too_long",
]

insert_types = [
    # "valid", # ok
    # "speech_tagged", # tuned
    # "garbage_prefix_v2" # tuned
    # "garbage_prefix_v3" # tuned
    # "garbage_prefix_v1" # tuned
    # "garbage_prefix_v4" # tuned
    # "sorry_rejected" # deny insertion
    # "no_eos", # deny insertion
    "too_long",
    # "invalid",
]

# remove_words = ["<s>", "</s>"]

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
        # jiwer.RemoveSpecificWords(remove_words), # RemoveKaldiNonWords already removes <s>, </s>
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def printer(key, answer, infer, type):
    print(f"===== {type} - {key} =====")
    print(f"answer: {' '.join(reference_transform(answer)[0])}")
    print(f"infer : {' '.join(hypothesis_transform(infer)[0])}")
    print(f"answer: {answer}")
    print(f"infer : {infer}")
    print("=" * (15 + len(key + type)))


task_name = "LibriSpeech-ASR-test-other-ls00"

TASK_HASH_PREFIX = TASK_HASH_PREFIX.format(task_name)
passkeys = [
    PENDING_QUEUE.format(task_name),
    PROCESSING_QUEUE.format(task_name),
]

task_keys = [k for k in r.scan_iter(f"{TASK_HASH_PREFIX}*") if k not in passkeys]
total_tasks = len(task_keys)

if total_tasks == 0:
    print("no data")
    exit()

status_counts = Counter()

answers = []
infers = []

for key in task_keys:
    task_data = r.hgetall(key)

    answer = task_data.get("sentence").strip()
    infer = task_data.get("infer").strip()

    if not (infer.endswith("<s>") or infer.endswith("</s>")):
        status = "no_eos"
    elif infer.startswith("The transcription of the given speech is:"):
        status = "garbage_prefix_v1"
        infer = (
            infer[len("The transcription of the given speech is:") :]
            .strip()
            .split("\n")[0]
        )
    elif infer.startswith("The transcription of the speech is:"):
        status = "garbage_prefix_v4"
        infer = (
            infer[len("The transcription of the speech is:") :].strip().split("\n")[0]
        )
    elif "<Speech>" in infer and "</Speech>" in infer:
        # ~<Speech>the making of a loyal patriot</Speech>~일 경우, <Speech> </Speech> 안의 내용만 추출해야
        status = "speech_tagged"
        infer = infer.split("<Speech>", 1)[1].split("</Speech>", 1)[0].strip()
    elif infer.startswith("I'm sorry"):
        status = "sorry_rejected"
    elif ': "' in infer:
        # Here is the transcription of the speech: "the villages will save us in the end"</s>
        # Here is the transcription: "This snatcher had been an orphan for many years."
        status = "garbage_prefix_v2"
        infer = infer.split(': "', 1)[1].split('"', 1)[0].strip()
        # transcription이 두 번 나와서 앞에서도 자르고 뒤에서도 잘라야 함
    elif 'is "' in infer:
        status = "garbage_prefix_v3"
        infer = infer.split('is "', 1)[1].rsplit('"', 1)[0].strip()
        # The speech is: "This snatcher had been an orphan for many years."\n\nHere is the transcription: "This snatcher had been an orphan for many years."
    elif abs(len(answer) - len(infer)) > 10:
        status = "too_long"
    elif infer.endswith("</s>"):
        status = "valid"
    else:
        status = "invalid"

    wer = jiwer.wer(
        answer,
        infer,
        reference_transform=reference_transform,
        hypothesis_transform=hypothesis_transform,
    )
    if status in print_types and wer > 1:
        printer(key, answer, infer, f"{status} | wer: {wer:.2f}")
    if status in insert_types:
        answers.append(answer)
        infers.append(infer)

    status_counts[status] += 1

final_wer = jiwer.wer(
    answers,
    infers,
    reference_transform=reference_transform,
    hypothesis_transform=hypothesis_transform,
)

print(
    f"{task_name} final WER using {len(answers)} samples: {final_wer:.2f} ({final_wer*100:.2f}%)"
)
print(status_counts)
