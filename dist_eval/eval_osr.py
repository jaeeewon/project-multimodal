import redis, re, jiwer

_PENDING_QUEUE = "task_{}:pending"
_PROCESSING_QUEUE = "task_{}:processing"
_TASK_HASH_PREFIX = "task_{}:"

r = redis.Redis(host="salmonn.hufs.jae.one", port=6379, db=8, decode_responses=True)
task_name = "LibriMix-OSR"

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

total = 0
correct = 0

answers = []
inferences = []


def normalize(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z' ]", "", text)
    text = " ".join(text.split())
    return text


for key in task_keys:
    task_data = r.hgetall(key)

    a1 = task_data.get("ans1")
    a2 = task_data.get("ans2")
    infer = task_data.get("infer")

    if infer.endswith("</s>"):
        infer = infer[:-4].strip()
    else:
        continue

    spl = infer.strip(".").split(".")
    l = len(spl)

    if l != 2:
        continue

    n1 = normalize(spl[0])
    n2 = normalize(spl[1])

    w1 = jiwer.wer([a1, a2], [n1, n2])
    w2 = jiwer.wer([a1, a2], [n2, n1])

    if w1 > w2:
        n1, n2 = n2, n1
    print(f"answ1: {a1}")
    print(f"norm1: {n1}")
    print(f"answ2: {a2}")
    print(f"norm2: {n2}")
    print("----")

    answers.extend([a1, a2])
    inferences.extend([n1, n2])

print(f"case1 wer: {jiwer.wer(answers, inferences)}")  # case1 wer: 0.22083954248998472
print(f"total counted: {len(answers)//2} / {total_tasks}")  # total counted: 2683 / 3000
