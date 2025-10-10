import redis, sacrebleu, re, string


def normalize(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_PENDING_QUEUE = "task_{}:pending"
_PROCESSING_QUEUE = "task_{}:processing"
_TASK_HASH_PREFIX = "task_{}:"

r = redis.Redis(host="salmonn.hufs.jae.one", port=6379, db=6, decode_responses=True)
task_name = "AudioCaps-Story-test"

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

diversities = 0
follow = 0
total = 0

for key in task_keys:
    task_data = r.hgetall(key)

    infer = task_data.get("infer").strip()

    if infer.endswith("</s>"):
        infer = infer[:-4].strip()
    # else:
    #     continue

    tokens = normalize(infer).split()
    # print(f"tokens: {len(tokens)} | unique: {len(set(tokens))}")

    total += 1
    if len(tokens) >= 50:
        follow += 1
        diversities += len(set(tokens))
    else:
        print(f"not follow: {infer}")


print(
    f"diversity_total: {diversities}/{total} = {diversities/total:.2f}%"
)  # diversity_total: 364261/4411 = 82.58%
print(
    f"diversity_followed: {diversities}/{follow} = {diversities/follow:.2f}%"
)  # diversity_followed: 364261/4408 = 82.64%
print(
    f"following rate: {follow}/{total} = {follow/total*100:.2f}%"
)  # following rate: 4408/4411 = 99.93%
