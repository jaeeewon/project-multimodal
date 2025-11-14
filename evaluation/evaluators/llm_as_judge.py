from ..abs.evaluator import AbstractEvaluator, Sample
from ..abs.judge import AbstractJudge
from ..abs.data_provider import AbstractDataProvider
from ..data.sakura import SakuraDataProvider
from ..types.redis_config import RedisConfig
from typing import Callable
from tqdm import tqdm


class LLMEvaluator(AbstractEvaluator):
    def __init__(self, judge: AbstractJudge, task_name: str, prompt_lambda=lambda: []):
        self._judge = judge
        self._task_name = task_name
        self._prompt_lambda = prompt_lambda

        print(f"[LLMEvaluator] initialized using judge model: {self._judge.model_name}")

    @property
    def task_name(self) -> str:
        return f"{self._task_name}:{self._judge.model_name}"

    def preprocess(self, text: str) -> str:
        # will be implemented soon
        return str(text).strip().lower()

    def evaluate(self, predictions: list[str], samples: list[Sample], batch_size: int = 1, use_tqdm=True) -> list[str]:
        super().evaluate(predictions, samples)  # verify lengths only

        judge_prompts = [
            self._prompt_lambda(prediction=pred, sample=sample) for pred, sample in zip(predictions, samples)
        ]

        return self._judge.judge_batch(judge_prompts, batch_size=batch_size, use_tqdm=use_tqdm)

    def evaluate_data_provider(
        self,
        data_provider: AbstractDataProvider,
        batch_size: int = 1,
        cb: Callable = None,
        **cfg,
    ) -> list[str]:
        """
        - {"status": "inferenced"} 만을 추론에 사용합니다
        - cb(tgt, ev)
        """
        # batch_size = cfg.get("batch_size", 1)
        evaluated = []

        filter = {"status": "inferenced"}

        for targets in tqdm(data_provider.take(batch_size, filter), total=data_provider.len(filter)):
            for tgt, ev in zip(
                targets,
                self.evaluate([tgt["inference"] for tgt in targets], targets, batch_size=batch_size, use_tqdm=False),
            ):
                if ev.startswith("judge_err"):
                    print(f"failed to evaluate {tgt}: {ev}")

                    tgt["status"] = "initialized"
                    continue

                tgt["status"] = "evaluated"
                tgt["evaluation"] = ev
                if cb:
                    cb(tgt, ev)
                evaluated.append(ev)

        return evaluated


if __name__ == "__main__":
    # python -m evaluation.evaluators.llm_as_judge

    from ..judges.vllm_judge import VllmJudge

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
        task_name="llm_as_a_judge-sakura",
    )

    sakura_provider = SakuraDataProvider(
        redis_cfg=RedisConfig(host="salmonn.hufs.jae.one", port=6379, db=9),
        key_prefix="salmonn-13b:sakura",
        filter={"set": "language", "hop": "single"},
    )

    # def cb(tgt, ev):
    #     ""

    results = evaluator.evaluate_data_provider(sakura_provider, batch_size=8)  # , cb=cb)
    for i, res in enumerate(results):
        print(f"--- Evaluation Result {i} ---")
        print(res)
