import redis, jiwer
from nltk.translate.meteor_score import single_meteor_score, meteor_score
from statistics import mean


transform = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

_PENDING_QUEUE = "task_{}:pending"
_PROCESSING_QUEUE = "task_{}:processing"
_TASK_HASH_PREFIX = "task_{}:"

r = redis.Redis(host="salmonn.hufs.jae.one", port=6379, db=6, decode_responses=True)
task_name = "AudioCaps-AAC-test"

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

answers = []
inferences = []
meteors = []

for key in task_keys:
    task_data = r.hgetall(key)

    answer = task_data.get("caption")
    infer = task_data.get("infer")

    if infer.endswith("</s>"):
        infer = infer[:-4].strip()
    else:
        continue

    # answers.append(transform(answer))
    # inferences.append(transform(infer))

    meteors.append(single_meteor_score(answer.split(), infer.split()))
    # meteor = single_meteor_score(answer.split(), infer.split())
    # print(f"ANS: {answer}")
    # print(f"INF: {infer}")
    # print(f"MET: {meteor}")
    # print("=====")

# meteors = meteor_score(inferences, answers)
print(mean(meteors)) # 0.19557955422256348
# print(meteor_score(inferences, answers))