import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
from ..abs.judge import AbstractJudge


class VllmJudge(AbstractJudge):
    def __init__(self, model: str = "openai/gpt-oss-20b", seed: int = 42, temperature: float = 0.0, top_p: float = 1.0):
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self._client = AsyncOpenAI(base_url="http://salmonn.hufs.jae.one:8080/v1", api_key="salmonn")
        self._client.models.list()

    @property
    def model_name(self) -> str:
        return self.model

    def judge_batch(self, prompts: list[list[dict[str, str]]], batch_size: int, use_tqdm=True) -> list[str]:
        # ex. [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
        try:
            return asyncio.run(self._async_judge_batch(prompts, batch_size, use_tqdm))
        except Exception as e:
            print(f"[VLLM] failed to call api: {e}")
            return [str(e)] * len(prompts)

    async def _async_judge_batch(
        self, prompts: list[list[dict[str, str]]], batch_size: int, use_tqdm=True
    ) -> list[str]:
        results = []

        it = range(0, len(prompts), batch_size)

        for i in tqdm(it, desc="Calling VLLM") if use_tqdm else it:
            batch = prompts[i : i + batch_size]

            if not batch:
                continue

            tasks = [self._async_chat_completion(prompt) for prompt in batch]

            try:
                batch_results = await self._async_gather(tasks)
                results.extend(batch_results)
            except Exception as e:
                print(f"[VLLM] Failed to call api for a batch: {e}")
                results.extend([f"judge_err_vllm: {e}"] * len(batch))

        return results

    async def _async_chat_completion(self, prompt: list[dict[str, str]]) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self.model, messages=prompt, seed=self.seed, temperature=self.temperature, top_p=self.top_p
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[VLLM] failed to call api in _async_chat_completion: {e}")
            raise e

    async def _async_gather(self, tasks: list[asyncio.Task]) -> list[str]:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for res in results:
            if isinstance(res, Exception):
                final_results.append(f"vllm: {res}")
            else:
                final_results.append(res)
        return final_results


if __name__ == "__main__":
    # python -m evaluation.judges.vllm_judge
    judge = VllmJudge()
    prompts = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "1 + 1은 얼마야?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "1 + 2는 얼마야?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "1 + 3은 얼마야?"},
        ],
    ]

    print("===== batch_size=2 =====")
    results = judge.judge_batch(prompts, batch_size=2)
    for i, res in enumerate(results):
        print(f"--- Result {i} ---")
        print(res)

    print("\n===== batch_size=1 =====")
    results_b1 = judge.judge_batch(prompts, batch_size=1)
    for i, res in enumerate(results_b1):
        print(f"--- Result {i} ---")
        print(res)
