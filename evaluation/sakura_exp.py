from .data.sakura import SakuraDataProvider
from .types.redis_config import RedisConfig
from .models.salmonn import SALMONNModel
from .judges.vllm_judge import VllmJudge
from .evaluators.llm_as_judge import LLMEvaluator
import tempfile, yaml, datetime, os, traceback, re, pandas as pd, argparse, time
from multiprocessing import Process, Queue, Event
from collections import Counter
from .salmonn_test import test_inference_fn, test_batch_inference

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


def save_experiment(base_exp_id: str, model_name: str, data_provider: SakuraDataProvider):
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
            print(f"Invalid judgement for key: {doc['key']}")
        total_count[f"{doc['set']}-{doc['hop']}"] += 1
    # ===== extract_judgement@llm_judge.py =====

    model_base = {
        "salmonn-13b": {
            "animal-multi": 46.4,
            "animal-single": 73.2,
            "emotion-multi": 31.8,
            "emotion-single": 31.2,
            "gender-multi": 49.2,
            "gender-single": 53.8,
            "language-multi": 22.0,
            "language-single": 22.6,
        },
        "salmonn-7b": {
            "animal-multi": 34.6,
            "animal-single": 68.2,
            "emotion-multi": 28.4,
            "emotion-single": 20,
            "gender-multi": 48.8,
            "gender-single": 60,
            "language-multi": 29.8,
            "language-single": 20.6,
        },
    }

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
        base = model_base[model_name].get(category)
        accuracy_delta = accuracy - base if base else pd.NA

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


def shared(
    device: str,
    config: dict,
    exp_ids: list[str],
    result_queue: Queue,
    error_event,
    model_name: str,
    save_exp: bool,
    inference_only: bool = True,
):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]

        model: SALMONNModel = None

        for i, exp_id in enumerate(exp_ids):
            print(f"[{device}] starting experiment {exp_id} ({i+1}/{len(exp_ids)})", flush=True)

            # DB11: EXP
            data_provider = SakuraDataProvider(
                redis_cfg=RedisConfig(host="salmonn.hufs.jae.one", port=6379, db=11),
                key_prefix=f"{model_name}:{exp_id}",
                required_fields=["wav", "query"],
                filter={},
            )

            assert len(
                data_provider
            ), f"no data stored in redis for {data_provider.key_prefix}\ncheck if you've inserted data first"

            status_keys = ["status", "set", "hop"]

            data_provider.status(keys=status_keys)
            data_provider.update_filter({"status": "initialized"})

            rtns = []

            if not len(data_provider):
                print(f"[{device}] inference already done")
            else:
                if model is None:
                    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=True) as tmpfile:
                        config["model_class"]["device"] = "cuda:0"
                        yaml.dump(config, tmpfile)
                        model = SALMONNModel(config_path=tmpfile.name)
                # def callback_fn(sample, inference):
                #     ""

                start_time = datetime.datetime.now()
                infered = model.infer(
                    data_provider,
                    batch_size=config["run"]["batch_size_eval"],
                    # callback_fn=callback_fn,
                    inference_fn=test_batch_inference,
                )
                elapsed_time = datetime.datetime.now() - start_time

                print(f"[{device}] evaluated {len(infered)} samples for {elapsed_time}", flush=True)

                rtns += infered

            if inference_only:
                continue

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

                rtns += evaluator.evaluate_data_provider(data_provider, batch_size=8)  # , cb=cb)

            if save_exp:
                data_provider.update_filter({})

                total_len = len(data_provider)
                evaled_len = data_provider.len({"status": "evaluated"})
                while evaled_len < total_len:
                    print(
                        f"[{device}] waiting for all data to be evaluated ({evaled_len}/{total_len})",
                        flush=True,
                    )
                    time.sleep(3)
                    evaled_len = data_provider.len({"status": "evaluated"})
                save_experiment(exp_id, model_name, data_provider)

        result_queue.put(rtns)

    except Exception as e:
        print(f"[{device}] exception: {e}", flush=True)
        error_event.set()
        result_queue.put((e, traceback.format_exc()))


def main(exp_ids: list[str], devices: list[str], config: dict, model_name: str, save_exp: bool, inference_only: bool):
    result_queue = Queue()

    error_event = Event()

    processes = [
        Process(
            name=device,
            target=shared,
            args=(
                device,
                config,
                exp_ids,
                result_queue,
                error_event,
                model_name,
                i == 0 and not inference_only and save_exp,  # save_exp
                inference_only,
            ),
        )
        for i, device in enumerate(devices)
    ]

    for process in processes:
        process.start()

    try:
        while True:
            if error_event.is_set():
                print("[main] terminating all processes due to error", flush=True)
                for process in processes:
                    process.terminate()
                break

            all_done = all(not process.is_alive() for process in processes)
            if all_done:
                break

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[main] terminating all processes due to keyboard interrupt", flush=True)
        for process in processes:
            process.terminate()

    for process in processes:
        process.join()

    if error_event.is_set():
        exit(1)

    for process in processes:
        q = result_queue.get()
        if isinstance(q, tuple) and isinstance(q[0], Exception):
            print(f"[{process.name}] early exit by exception")
            print(f" - exception: {q[0]}")
            print(f" - traceback:\n{q[1]}")
        else:
            print(f"[{process.name}] processed {len(q)} documents totally")


if __name__ == "__main__":
    # python -m evaluation.sakura_exp --device cuda:0 cuda:1 cuda:2 --model_name salmonn-7b --exp_ids SLMN7-SKR-LD-NZ-EARLY-30s-B8 --batch_size 8 --skip_confirm --save_exp
    parser = argparse.ArgumentParser(description="sakura experiment tool")
    parser.add_argument("--devices", type=str, nargs="+", required=True, help="devices to use")
    parser.add_argument(
        "--model_name", type=str, default="salmonn-7b", choices=["salmonn-7b", "salmonn-13b"], help="model name"
    )
    parser.add_argument("--exp_ids", type=str, nargs="+", required=True, help="experiment ids")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--skip_confirm", action="store_true", help="confirm before running the experiment")
    parser.add_argument("--save_exp", action="store_true", help="save experiment results")
    parser.add_argument("--inference_only", action="store_true", help="run inference only without evaluation")
    args = parser.parse_args()

    devices = args.devices
    model_name = args.model_name
    exp_ids = args.exp_ids
    batch_size = args.batch_size
    save_exp = args.save_exp
    inference_only = args.inference_only

    configs = {"salmonn-7b": "configs/eval_7b.yaml", "salmonn-13b": "configs/eval_13b.yaml"}

    config = yaml.safe_load(open(configs[model_name], "r"))
    config["run"]["batch_size_eval"] = batch_size

    if not args.skip_confirm:
        print("===== experiment configuration =====")
        confirm = input(
            f"exp_ids: '{exp_ids}', model: '{model_name}', devices: '{devices}', batch_size: {batch_size}.\nis it ok to run? (y/n) > "
        )
        if confirm.lower() != "y":
            print("experiment stopped by user")
            exit(0)

    start_time = datetime.datetime.now()
    main(
        exp_ids=exp_ids,
        devices=devices,
        config=config,
        model_name=model_name,
        save_exp=save_exp,
        inference_only=inference_only,
    )
    elapsed_time = datetime.datetime.now() - start_time

    print(f"===== totally elapsed for experiment {exp_ids} - {elapsed_time} =====")
