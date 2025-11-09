from .data.sakura import SakuraDataProvider
from .types.redis_config import RedisConfig
from .models.salmonn import SALMONNModel
from .judges.vllm_judge import VllmJudge
from .evaluators.llm_as_judge import LLMEvaluator
import tempfile, yaml, datetime, os, traceback, re, pandas as pd
from multiprocessing import Process, Queue
from collections import Counter
from .salmonn_test import test_inference_fn

system_prompt = """You are a good judge. You will be given a question with list of possible options, a ground truth answer and a model generated response.
You have to determine whether the model generated answer is correct."""
user_prompt = """
You will be given a question with list of possible options, a ground truth answer and a model generated response. Determine whether the model generated response is correct based on the following criteria:
1. Since there is one and only one corect answer, it should be judged incorrect if the model do not choose any option from the option list or it choose more than one option.
2. If the model choose one option from the option list, it should be judged correct if the chosen option aligns with the ground truth answer, otherwise it should be judged incorrect.
3. Read the question, options, ground truth answer and model generated response carefully before making a decision.

Considering the following examples:
Question: What is the capital of France? (a) Paris (b) London (c) Berlin (d) Madrid
Ground truth answer: (a) Paris
If the model generated response is: "The capital of France is Tokyo.", it should be judged incorrect since it does not choose any option from the option list.
If the model generated response is: "The capital of France is Paris and London.", it should be judged incorrect since it chooses more than one option from the option list.
If the model generated response is: "The capital of France is London.", it should be judged incorrect since it chooses one option from the option list but the chosen option does not align with the ground truth answer.
If the model generated response is: "The capital of France is Paris.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
Another Question: What is the underlying emotion of the speaker? (a) Happy (b) Sad (c) Angry (d) Neutral
Ground truth answer: (a) Happy
If the model generated response is: "The speaker is happy.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
If the model generated response is: "The speaker expresses happiness.", it should be judged correct since "happiness" aligns with the ground truth answer "happy", and they are just different part of speech of the same word.
If the model generated response is: "Happiness," it should be judged correct since it is also a valid derivative of the ground truth answer "happy".

Now here is the question and the model generated response for you to judge:
Question: [QUESTION]
Ground truth answer: [GROUND_TRUTH_ANSWER]
Model generated response: [MODEL_GENERATED_RESPONSE]

Carefully make your decision based on the above criteria. Return your judgement with the following format:
Explanation: <Your explanation on your judgement>
Judgement: <Your judgement, either "correct" or "incorrect">
"""

CSV_FILE_PATH = "evaluation/experiment_results.csv"
MD_FILE_PATH = "evaluation/experiment_results.md"
MAX_HISTORY = 3


def load_data(path):
    columns = [
        "experiment",
        "base",
        "run_num",
        "category",
        "correct",
        "incorrect",
        "total",
        "accuracy",
        "accuracy_delta",
    ]

    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=columns)
    else:
        return pd.DataFrame(columns=columns)


def save_data(df):
    df_sorted = df.sort_values(by=["base", "category", "run_num"], ascending=[True, True, False])
    df_sorted.to_csv(CSV_FILE_PATH, index=False)
    df_sorted.to_markdown(MD_FILE_PATH, index=False)


def save_experiment(base_exp_id: str, data_provider: SakuraDataProvider):
    """
    this function, **especially saving results to csv**, is written by Gemini
    """
    data_provider.update_filter({"status": "evaluated"})
    df = load_data(CSV_FILE_PATH)

    if df.empty:
        df["run_num"] = pd.to_numeric(df["run_num"], errors="coerce").fillna(0).astype(int)

    new_rows = []
    correct_count = Counter()
    incorrect_count = Counter()
    total_count = Counter()

    # ===== extract_judgement@llm_judge.py =====
    for doc in iter(data_provider):
        pattern = r"Explanation: (.*?)\nJudgement: (.*?)(?:\n\n|$)"
        match = re.search(pattern, doc["evaluation"], re.DOTALL)

        if match:
            explanation = match.group(1)
            judgement = match.group(2).lower().strip()
        else:
            explanation = "No extracted explanation"
            judgement = "No extracted judgement"

        if judgement == "correct":
            correct_count[f"{doc['set']}-{doc['hop']}"] += 1
        elif judgement == "incorrect":
            incorrect_count[f"{doc['set']}-{doc['hop']}"] += 1
        else:
            print(f"Invalid judgement for key: {k}")
        total_count[f"{doc['set']}-{doc['hop']}"] += 1
    # ===== extract_judgement@llm_judge.py =====

    for category in correct_count:
        # print(f"===== {category} =====")
        # print(f"Correct count: {correct_count[category]}")
        # print(f"Incorrect count: {incorrect_count[category]}")
        # print(f"Total count: {total_count[category]}")
        # print(f"Accuracy: {(correct_count[category] / total_count[category]) * 100:.2f}%")

        # 3. 이 실험(base + category)의 과거 기록 찾기
        history = df[(df["base"] == base_exp_id) & (df["category"] == category)].sort_values(by="run_num")

        # 4. 새 실행 번호 및 이전 정확도 결정
        prev_accuracy = pd.NA
        if history.empty:
            new_run_num = 0
        else:
            new_run_num = history["run_num"].max() + 1
            prev_accuracy = history.iloc[-1]["accuracy"]  # 가장 마지막 실행의 정확도

        # 5. 새 실험 ID 생성 (e.g., test_run_1-2)
        final_exp_id = base_exp_id if new_run_num == 0 else f"{base_exp_id}-{new_run_num}"

        # 6. 새 실험 결과 계산
        correct = correct_count[category]
        total = total_count[category]
        incorrect = total - correct
        accuracy = round((correct / total) * 100, 2)

        # 7. [변경사항] 변화량 계산
        accuracy_delta = pd.NA
        if pd.notna(prev_accuracy):
            accuracy_delta = round(accuracy - prev_accuracy, 2)

        # 8. 새 결과 행(Row) 생성
        new_row = {
            "experiment": final_exp_id,
            "base": base_exp_id,
            "run_num": new_run_num,
            "category": category,
            "correct": correct,
            "incorrect": incorrect,
            "total": total,
            "accuracy": accuracy,
            "accuracy_delta": accuracy_delta,
        }

        new_rows.append(new_row)
        # print(f"  > {category} (Run {new_run_num}): Acc {accuracy}%, Delta {accuracy_delta}")

    # 9. 새 결과(들)를 기존 DataFrame에 추가
    df_new_rows = pd.DataFrame(new_rows)
    df_updated = pd.concat([df, df_new_rows], ignore_index=True)

    # 10. [변경사항] 최대 3개 보관 처리
    # groupby를 사용하여 각 (base_id, category) 그룹별로
    # 'run_num'이 가장 큰 3개만 선택합니다.
    def keep_latest(group):
        return group.nlargest(MAX_HISTORY, "run_num")

    df_final = df_updated.groupby(["base", "category"], as_index=False).apply(keep_latest).reset_index(drop=True)

    # 11. 최종 결과를 CSV에 덮어쓰기
    save_data(df_final)


def shared(device: str, data_provider: SakuraDataProvider, result_queue: Queue):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]

        status_keys = ["status", "set", "hop"]

        data_provider.status(keys=status_keys)
        # data_provider.insert_ds(is_exp=True)

        rtns = []

        if not len(data_provider):
            print(f"[{device}] inference already done")
        else:
            data = yaml.safe_load(open("configs/eval_13b.yaml"))
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=True) as tmpfile:
                data["model_class"]["device"] = "cuda:0"
                yaml.dump(data, tmpfile)
                model = SALMONNModel(config_path=tmpfile.name)

            # def callback_fn(sample, inference):
            #     ""

            start_time = datetime.datetime.now()
            infered = model.infer(
                data_provider,
                batch_size=data['run']['batch_size_eval'],
                # callback_fn=callback_fn,
                inference_fn=test_inference_fn
            )
            elapsed_time = datetime.datetime.now() - start_time

            print(f"[{device}] evaluated {len(infered)} samples for {elapsed_time}")

            rtns = infered

        data_provider.update_filter({"status": "inferenced"})

        if not len(data_provider):
            print(f"[{device}] judge already done")
        else:

            judge = VllmJudge()
            evaluator = LLMEvaluator(
                judge=judge,
                prompt_lambda=lambda prediction, sample: [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt.replace("[QUESTION]", sample["query"])
                        .replace("[GROUND_TRUTH_ANSWER]", sample["text"])
                        .replace("[MODEL_GENERATED_RESPONSE]", prediction),
                    },
                ],
                task_name=f"{data_provider.data_id}:llm_as_a_judge",
            )

            # def cb(tgt, ev):
            #     ""

            rtns = evaluator.evaluate_data_provider(data_provider, batch_size=8)  # , cb=cb)

        result_queue.put(rtns)

    except Exception as e:
        result_queue.put((e, traceback.format_exc()))


def main(exp_id: str, data_provider: SakuraDataProvider):
    devices = ["cuda:0", "cuda:2", "cuda:3"]
    result_queue = Queue()

    processes = [Process(name=device, target=shared, args=(device, data_provider, result_queue)) for device in devices]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    for process in processes:
        q = result_queue.get()
        if isinstance(q, tuple) and isinstance(q[0], Exception):
            print(f"[{process.name}] early exit by exception")
            print(f" - exception: {q[0]}")
            print(f" - traceback:\n{q[1]}")
        else:
            print(f"[{process.name}] processed {len(q)} documents totally")

    save_experiment(exp_id, data_provider)


if __name__ == "__main__":
    # python -m evaluation.sakura_exp

    model_name = "salmonn-13b"
    exp_id = "SLMN13-TEST"
    # exp_id = "SLMN13-SK-B3"
    # exp_id = "SLMN13-SK"
    # exp_id = "SLMN13-SK%50"

    # DB11: EXP
    data_provider = SakuraDataProvider(
        redis_cfg=RedisConfig(host="salmonn.hufs.jae.one", port=6379, db=11),
        key_prefix=f"{model_name}:{exp_id}",
        required_fields=["wav", "query"],
        filter={"status": "initialized"},
    )
    data_provider.delete_ds()
    data_provider.insert_ds(is_exp=True)
    # data_provider.insert_ds(is_exp=False)
    # data_provider.status()

    start_time = datetime.datetime.now()
    main(exp_id=exp_id, data_provider=data_provider)
    elapsed_time = datetime.datetime.now() - start_time

    print(f"===== totally elapsed for experiment {exp_id} - {elapsed_time} =====")
