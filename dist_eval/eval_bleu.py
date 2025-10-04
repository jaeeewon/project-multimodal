import redis, sacrebleu, re, string


def normalize(text):
    text = text.lower()
    # lowercased and with punctuation removed (except for apostrophes and hyphens).
    excluded = set(string.punctuation) - {"'", "-"}
    text = text.translate(str.maketrans("", "", "".join(excluded)))
    return text


_PENDING_QUEUE = "task_{}:pending"
_PROCESSING_QUEUE = "task_{}:processing"
_TASK_HASH_PREFIX = "task_{}:"

r = redis.Redis(host="salmonn.hufs.jae.one", port=6379, db=0, decode_responses=True)
task_name = "en2ja"

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

ls = []

for key in task_keys:
    task_data = r.hgetall(key)

    answer = task_data.get("translation")
    infer = task_data.get("infer")

    if infer.endswith("</s>"):
        infer = infer[:-4].strip()
    # else:
    #     continue
    if not re.search(r"[\u3040-\u309F\u30A0-\u30FF]", infer):
        # 일본어 하나도 없으면
        continue

    # score = bleu4_score(answer, infer, "ja-mecab")
    # print(f"===== {key} =====")
    # print(f"answer: {answer}")
    # print(f"infer : {infer}")
    # print(f"BLEU-4: {score:.2f}")
    # ls.append(score)
    ls.append((answer, infer))


# print(f"avg BLEU-4: {sum(ls)/len(ls):.2f}")
corpus_bleu = sacrebleu.corpus_bleu(
    [normalize(hyp) for _, hyp in ls],
    [[normalize(ref) for ref, _ in ls]],
    tokenize="char",
)
print(f"BLEU4: {corpus_bleu.score:.2f}")
print("skipped:", total_tasks - len(ls), "/", total_tasks)
