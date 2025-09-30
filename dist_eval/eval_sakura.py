from init_works import SalmonnRedis
from collections import Counter

r = SalmonnRedis(host="192.168.219.101", db=7)
sakura_tracks = ["Animal", "Emotion", "Gender", "Language"]
sakura_judge_pf = "-judge-qwen3"

corr_counter = Counter()
incorr_counter = Counter()
fail_counter = Counter()

for track in sakura_tracks:
    for hop in ["single", "multi"]:
        task_name = f"SAKURA-{track}-{hop}"
        TASK_HASH_PREFIX = SalmonnRedis.TASK_HASH_PREFIX.format(task_name)
        passkeys = [
            SalmonnRedis.PENDING_QUEUE.format(task_name),
            SalmonnRedis.PROCESSING_QUEUE.format(task_name),
            SalmonnRedis.PENDING_QUEUE.format(task_name) + sakura_judge_pf,
            SalmonnRedis.PROCESSING_QUEUE.format(task_name) + sakura_judge_pf,
        ]

        task_keys = [
            k
            for k in r.client.scan_iter(f"{TASK_HASH_PREFIX}*")
            if not k.startswith(tuple(passkeys))
        ]

        for key in task_keys:
            task_data = r.client.hgetall(key)

            status = task_data.get("status")

            if status == f"completed{sakura_judge_pf}":
                judgement = task_data.get(f"Judgement{sakura_judge_pf}")
                if judgement == "correct":
                    corr_counter[task_name] += 1
                elif judgement == "incorrect":
                    incorr_counter[task_name] += 1
                else:
                    fail_counter[task_name] += 1

        print(f"===== {task_name} =====")
        print(f"correct: {corr_counter[task_name]}")
        print(f"incorrect: {incorr_counter[task_name]}")
        print(f"failed: {fail_counter[task_name]}")
        total = (
            corr_counter[task_name]
            + incorr_counter[task_name]
            + fail_counter[task_name]
        )
        print(f"total: {total}")
        if total > 0:
            print(f"accuracy: {corr_counter[task_name] / total:.4f}")
        print()

"""
# SALMONN 14B with 8bit quantized LLM

===== SAKURA-Animal-single =====
correct: 365
incorrect: 135
failed: 0
total: 500
accuracy: 0.7300

===== SAKURA-Animal-multi =====
correct: 232
incorrect: 265
failed: 3
total: 500
accuracy: 0.4640

===== SAKURA-Emotion-single =====
correct: 155
incorrect: 337
failed: 4
total: 496
accuracy: 0.3125

===== SAKURA-Emotion-multi =====
correct: 159
incorrect: 335
failed: 2
total: 496
accuracy: 0.3206

===== SAKURA-Gender-single =====
correct: 269
incorrect: 231
failed: 0
total: 500
accuracy: 0.5380

===== SAKURA-Gender-multi =====
correct: 245
incorrect: 253
failed: 2
total: 500
accuracy: 0.4900

===== SAKURA-Language-single =====
correct: 110
incorrect: 385
failed: 5
total: 500
accuracy: 0.2200

===== SAKURA-Language-multi =====
correct: 110
incorrect: 383
failed: 7
total: 500
accuracy: 0.2200
"""