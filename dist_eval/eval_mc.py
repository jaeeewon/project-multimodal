import redis, sacrebleu, re, string
from rouge_score import rouge_scorer
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

tokenizer = Tokenizer13a()


def normalize(text):
    text = text.lower()
    # lowercased and with punctuation removed (except for apostrophes and hyphens).
    excluded = set(string.punctuation) - {"'", "-"}
    text = text.translate(str.maketrans("", "", "".join(excluded)))
    return text


_PENDING_QUEUE = "task_{}:pending"
_PROCESSING_QUEUE = "task_{}:processing"
_TASK_HASH_PREFIX = "task_{}:"

r = redis.Redis(host="salmonn.hufs.jae.one", port=6379, db=8, decode_responses=True)
task_name = "MusicCaps-MC"

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

    answer = task_data.get("caption")
    infer = task_data.get("infer")

    if infer.endswith("</s>"):
        infer = infer[:-4].strip()
    # else:
    #     continue

    ls.append((answer, infer))


corpus_bleu = sacrebleu.corpus_bleu(
    [tokenizer(hyp) for _, hyp in ls],
    [[tokenizer(ref) for ref, _ in ls]],
    force=True,
    # tokenize="char"
).score

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
corpus_rouge_l = (
    sum(
        scorer.score(
            tokenizer(inference),
            tokenizer(answer),
        )["rougeL"].fmeasure
        for answer, inference in ls
    )
    / len(ls)
    * 100
)

print(f"BLEU4: {corpus_bleu:.2f}")
print(f"ROUGE-L: {corpus_rouge_l:.2f}")
print("skipped:", total_tasks - len(ls), "/", total_tasks)

# normalize <-> tokenizer
# skip no </s> - skipped: 306 / 2828
# BLEU4: 3.53 | ROUGE-L: 22.62 <-> BLEU4: 4.31 | ROUGE-L: 22.65
# no skip <s>
# BLEU4: 3.07 | ROUGE-L: 21.83 <-> BLEU4: 3.78 | ROUGE-L: 21.86