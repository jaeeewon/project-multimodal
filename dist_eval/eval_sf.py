import redis
from tqdm import tqdm
from qwen3 import qwen3_api
from jiwer import wer
from slurp import get_slurp_sf

_PENDING_QUEUE = "task_{}:pending"
_PROCESSING_QUEUE = "task_{}:processing"
_TASK_HASH_PREFIX = "task_{}:"

USER_PROMPT = """
determine if the `our_slot` accurately reflects the `golden_slot` and respond only with 'Yes' or 'No'.
you can refer to the `slot_type` for better understanding.

- **our_slot**: `{our_slot}`
- **golden_slot**: `{golden_slot}`
- **slot_type**: `{slot_type}`
"""

SYSTEM_PROMPT = """
you will be given two slots: `our_slot` and `golden_slot`, along with a `slot_type` that describes the nature of the slot.
you don't need to flatter for our model and you don't need to be harsh for our model.
you must respond 'Yes' if `our_slot` accurately reflects `golden_slot`, otherwise respond 'No'.
you must respond strictly with either 'Yes' or 'No'.
you must not include any explanations or additional text.
"""

r = redis.Redis(host="salmonn.hufs.jae.one", port=6379, db=8, decode_responses=True)
task_name = "Slurp-SF"

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

followed = 0
inferred = []

# sps = get_slurp_sf()
# idx2transcript = {s["idx"]: s["transcript"] for s in sps}

for i, key in tqdm(enumerate(task_keys), total=total_tasks):
    task_data = r.hgetall(key)
    infer = task_data.get("infer")

    # if infer.endswith("</s>"):
    #     infer = infer[:-4].strip()
    # else:
    #     continue
    # 100% </s> exists

    answers = [s.split("=") for s in task_data.get("slots").lower().split("; ") if s]
    infer = [
        [v.strip() for v in i.split("=")]
        for i in infer.replace("</s>", "").lower().split("; ")
        if i
    ]

    # following rate
    # transcript = idx2transcript.get(int(task_data.get("idx")))
    # r.hset(key, "transcript", transcript)
    transcript = task_data.get("transcript")
    if any(wer(transcript, inf[1]) >= 0.3 for inf in infer):
        followed += 1
    # else:
    #     print(f"not followed: {transcript} -> {infer}")

    llm_as_a_judge = [
        qwen3_api(
            user_prompt=USER_PROMPT.format(
                our_slot=i[1], golden_slot=a[1], slot_type=a[0]
            ),
            system_prompt=SYSTEM_PROMPT,
        )
        for a, i in zip(answers, infer)
    ]
    # print("[inferred]")
    # for res in llm_as_a_judge:
    #     print(res)
    inferred.extend(llm_as_a_judge)
    if i % 200 == 0:
        print(
            f"[{i:04d} / {total_tasks}] accuracy: {inferred.count('Yes') / len(inferred):.5f}, following rate: {followed / (i + 1):.5f}"
        )

# {'Yes', 'No'}
print(f"accuracy: {inferred.count('Yes') / len(inferred):.5f}")  # accuracy: 0.39215
print(f"following rate: {followed / total_tasks:.5f}")  # following rate: 0.99627
