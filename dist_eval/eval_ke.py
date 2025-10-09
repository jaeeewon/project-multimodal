import redis

_PENDING_QUEUE = "task_{}:pending"
_PROCESSING_QUEUE = "task_{}:processing"
_TASK_HASH_PREFIX = "task_{}:"

r = redis.Redis(host="salmonn.hufs.jae.one", port=6379, db=8, decode_responses=True)
task_name = "Inspec-KE"

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

correct = 0
total = 0


def postprocess(value):
    value = value.strip()
    if value.startswith("and "):
        value = value[4:].strip()
    if value.endswith("."):
        value = value[:-1].strip()
    value = value.replace("-", " ")
    return value


for key in task_keys:
    task_data = r.hgetall(key)

    answers = [postprocess(k) for k in task_data.get("keyphrases").lower().split("; ")]
    infer = task_data.get("infer").lower()

    if infer.endswith("</s>"):
        infer = infer[:-4].strip()
    # else:
    #     print(f"skip (no </s>): [{task_data.get('path')}] {infer}")
    #     continue

    # skip no </s> - accuracy: 466/1469 = 31.72%
    # no skip </s> - accuracy: 470/1490 = 31.54%

    infer = [postprocess(k) for k in infer.split(",")]

    total += min(len(answers), 3)
    curr = min(
        sum(
            sum(k in a for a in answers) if len(k.split()) > 1 else k in answers
            for k in infer
        ),
        3,
    )
    correct += curr

    print(f"ANS: {answers}")
    print(f"INF: {infer}")
    print(f"COR: {curr}")
    print("=" * 20)
    print()

print(f"accuracy: {correct}/{total} = {correct/total*100:.2f}%")
# skip no </s> - accuracy: 466/1469 = 31.72%
# no skip </s> - accuracy: 470/1490 = 31.54%
