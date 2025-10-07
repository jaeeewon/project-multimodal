import redis, sacrebleu, re, string

_PENDING_QUEUE = "task_{}:pending"
_PROCESSING_QUEUE = "task_{}:processing"
_TASK_HASH_PREFIX = "task_{}:"

r = redis.Redis(host="salmonn.hufs.jae.one", port=6379, db=8, decode_responses=True)
task_name = "IEMOCAP-ER"

TASK_HASH_PREFIX = _TASK_HASH_PREFIX.format(task_name)
passkeys = [
    _PENDING_QUEUE.format(task_name),
    _PROCESSING_QUEUE.format(task_name),
]

task_keys = [k for k in r.scan_iter(f"{TASK_HASH_PREFIX}*") if k not in passkeys]
total_tasks = len(task_keys)

if total_tasks == 0:
    print("no data")
    exit()

# emotions = {}
# answers = {}

ans_kv = {
    "neu": "Neutral",
    "hap": "Happy",
    "exc": "Happy",
    "sad": "Sad",
    "ang": "Angry",
}

skip_keys = ["xxx", "oth", "fru", "sur", "fea", "dis"]

correct = 0
total = 0

for key in task_keys:
    task_data = r.hgetall(key)

    answer = task_data.get("emotion")
    infer = task_data.get("infer")

    if answer in skip_keys:
        continue

    if infer.endswith("</s>"):
        infer = infer[:-4].strip()
    else:
        continue

    answer = ans_kv[answer]

    total += 1
    if infer == answer:
        correct += 1

    # if infer not in emotions:
    #     emotions[infer] = 0
    # emotions[infer] += 1

    # if answer not in answers:
    #     answers[answer] = 0
    # answers[answer] += 1

# print(f"emotions: {emotions}")
# {'Happy': 413, 'Neutral': 709, 'Angry': 506, 'Sad': 479, 'Fearful': 10, 'Embarrassed': 2, 'Anxious': 6, 'Positive': 1, 'Frustrated': 20, 'Excited': 9, 'Hopeful': 2, 'Negative': 1, 'Wow': 2, 'Ashamed': 1, 'Needy': 2, 'Sorry': 1, 'Surprised': 1, 'Disappointed': 1, 'Empathetic': 1, 'Tired': 1, 'Mysterious': 1, 'Haunting': 1}
# print(f"answers: {answers}")
# {'exc': 299, 'fru': 381, 'neu': 384, 'xxx': 520, 'hap': 143, 'sad': 245, 'sur': 18, 'ang': 170, 'fea': 10}
print(f"accuracy: {correct}/{total} = {correct/total*100:.2f}%") # accuracy: 849/1241 = 68.41%
