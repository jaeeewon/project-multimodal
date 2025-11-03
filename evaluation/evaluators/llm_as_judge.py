from ..abs.evaluator import AbstractEvaluator, Sample
from ..abs.judge import AbstractJudge


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

    def evaluate(self, predictions: list[str], samples: list[Sample], batch_size: int = 1) -> list[str]:
        super().evaluate(predictions, samples)  # verify lengths only

        judge_prompts = [
            self._prompt_lambda(prediction=pred, sample=sample) for pred, sample in zip(predictions, samples)
        ]

        return self._judge.judge_batch(
            judge_prompts,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    # python -m evaluation.evaluators.llm_as_judge

    from ..judges.vllm_judge import VllmJudge

    judge = VllmJudge()
    evaluator = LLMEvaluator(
        judge=judge,
        prompt_lambda=lambda prediction, sample: [
            {"role": "system", "content": "You are a helpful judge that evaluates model outputs."},
            {
                "role": "user",
                "content": f"Evaluate the following model output:\n\n{prediction}\n\n"
                f"Based on the reference answer:\n\n{sample['reference']}\n\n"
                f"Provide your evaluation.",
            },
        ],
        task_name="LLM_Judge_Test",
    )
    predictions = [
        "The capital of France is Paris.",
        "The largest planet in our solar system is Jupiter.",
        "Water boils at 0 degrees Celsius.",
    ]
    samples = [
        {"reference": "Paris is the capital city of France."},
        {"reference": "Jupiter is the biggest planet in our solar system."},
        {"reference": "At standard atmospheric pressure, water boils at 100°C."},
    ]
    results = evaluator.evaluate(predictions, samples, batch_size=2)
    for i, res in enumerate(results):
        print(f"--- Evaluation Result {i} ---")
        print(res)
