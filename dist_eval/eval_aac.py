import redis, jiwer, json
from nltk.translate.meteor_score import single_meteor_score
from statistics import mean

"""
SPIDEr = (SPICE + CIDEr) / 2

# https://github.com/salaniz/pycocoevalcap
# install requirements
apt install openjdk-8-jdk
java -version # openjdk version "1.8.0_462"
pip install pycocoevalcap

# error since memory limit
# replace '-Xmx8G' with '-Xmx80G' in pycocoevalcap/spice/spice.py, line 69
"""
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

transform = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
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

new_ref = {"images": [], "annotations": [], "info": ""}
new_cap = []
meteors = []

for i, key in enumerate(task_keys, start=1):
    task_data = r.hgetall(key)

    answer = task_data.get("caption")
    infer = task_data.get("infer_v2")

    if infer.endswith("</s>"):
        infer = infer[:-4].strip()
    else:
        continue

    meteors.append(
        single_meteor_score(transform(answer).split(), transform(infer).split())
    )
    new_ref["images"].append({"id": i})
    new_ref["annotations"].append(
        {"image_id": i, "id": i, "caption": transform(answer)}
    )
    new_cap.append({"image_id": i, "caption": transform(infer)})

annotation_file = "aac_annotation_ignore_.json"
results_file = "aac_results_ignore_.json"
with open(annotation_file, "w") as f:
    f.write(json.dumps(new_ref))
with open(results_file, "w") as f:
    f.write(json.dumps(new_cap))

coco = COCO(annotation_file)
coco_result = coco.loadRes(results_file)
coco_eval = COCOEvalCap(coco, coco_result)
coco_eval.evaluate()

# for metric, score in coco_eval.eval.items():
#     print(f"{metric}: {score:.3f}")
#     """
#     Bleu_1: 0.227
#     Bleu_2: 0.107
#     Bleu_3: 0.055
#     Bleu_4: 0.029
#     METEOR: 0.108
#     ROUGE_L: 0.217
#     CIDEr: 0.354
#     SPICE: 0.111
#     """

print(f"mean local meteors: {mean(meteors)}")
# 0.2152928432061087 if not skip </s>
# 0.21586046920735638 if skip </s>
print(f"SPIDEr: {(coco_eval.eval['SPICE'] + coco_eval.eval['CIDEr']) / 2}")
# 0.2326582523151935 if not skip </s>
# 0.23334328623335548 if skip </s>
print(
    f"skip: {(total_tasks - len(meteors)) / total_tasks:.2f}% ({total_tasks - len(meteors)} / {total_tasks})"
)  # skip: 0.01% (25 / 4411)
