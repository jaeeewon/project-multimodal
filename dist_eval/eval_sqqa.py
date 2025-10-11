import redis
from jiwer import wer

# from wiki_qa import get_wikiqa_sqqa

_PENDING_QUEUE = "task_{}:pending"
_PROCESSING_QUEUE = "task_{}:processing"
_TASK_HASH_PREFIX = "task_{}:"

r = redis.Redis(host="salmonn.hufs.jae.one", port=6379, db=8, decode_responses=True)
task_name = "WikiQA-SQQA"

TASK_HASH_PREFIX = _TASK_HASH_PREFIX.format(task_name)
passkeys = [
    _PENDING_QUEUE.format(task_name),
    _PROCESSING_QUEUE.format(task_name),
    _PENDING_QUEUE.format(task_name) + "-judge-qwen3",
    _PROCESSING_QUEUE.format(task_name) + "-judge-qwen3",
]

# wq = get_wikiqa_sqqa(skip_exist=True)
# id2ans = {str(d["question_id"]): d['answer'] for d in wq}

task_keys = [k for k in r.scan_iter(f"{TASK_HASH_PREFIX}*") if k not in passkeys]
total_tasks = len(task_keys)

if total_tasks == 0:
    print("no data")
    exit()

followed = 0
evaluated = []

for key in task_keys:
    task_data = r.hgetall(key)

    # r.hset(key, "answer", id2ans.get(task_data.get('question_id')))

    # if task_data.get("status") != "completed-judge-qwen3":
    #     continue

    infer = task_data.get("infer")
    answers = task_data.get("answer").lower().split("; ")
    judged = task_data.get("accuracy-judge-qwen3").lower().split("; ")

    if infer.endswith("</s>"):
        infer = infer[:-4].strip()
    # else:
    #     continue

    # evaled = []
    # for j in judged:
    #     try:
    #         evaled.append(json.loads(j.lower())["correct"])
    #     except:
    #         print(f"parsing error: {j}")
    #         evaled.append(False)
    #         # one exception,, haha
    #         # {"correct": true, "reason": "the standard answer does not mention the studio that produced 'annie'
    #     # qwen returns upper-cased boolean -> .lower() to parse using json.loads
    # evaluated.append(any(evaled))
    evaluated.append(any('{"correct": true, "reason": "' in j for j in judged))

    if any(wer(answer, infer) >= 0.3 for answer in answers):
        followed += 1
    # else:
    #     print(f"not followed: {transcript} -> {infer}")
    # 100% followed

print(
    f"accuracy: {sum(evaluated) / len(evaluated):.5f} ({sum(evaluated)}/{len(evaluated)})"
)  # accuracy: 0.36651 (232/633)
print(
    f"following rate: {followed / total_tasks:.5f} ({followed}/{total_tasks})"
)  # following rate: 1.00000 (633/633)
