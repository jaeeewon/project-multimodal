import redis

_PENDING_QUEUE = "task_{}:pending"
_PROCESSING_QUEUE = "task_{}:processing"
_TASK_HASH_PREFIX = "task_{}:"

r = redis.Redis(host="salmonn.hufs.jae.one", port=6379, db=8, decode_responses=True)
task_name = "VoxCeleb1-SV"

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

for key in task_keys:
    task_data = r.hgetall(key)

    if not task_data.get("status") == "completed":
        continue

    infer = task_data.get("infer").lower()

    if infer.endswith("</s>"):
        infer = infer[:-4].strip()
    # else:
    #     print(f"skip (no </s>): [{task_data.get('path')}] {infer}")
    #     continue
    # 100% </s> exists

    total += 1
    if infer == "yes":
        correct += 1
    else:
        print(f"wrong: [{task_data.get('path')}] {infer}")

print(
    f"accuracy: {correct}/{total} = {correct/total*100:.2f}%"
)  # accuracy: 4864/4874 = 99.79%
