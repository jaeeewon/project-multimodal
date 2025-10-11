# current version: transformers==4.46.3
# Transformers is a library of pretrained natural language processing for inference and training. The latest version of transformers is recommended and transformers>=4.51.0 is required.
# https://github.com/QwenLM/Qwen3?tab=readme-ov-file#run-qwen3
import requests
import json

# run `python -m vllm.entrypoints.openai.api_server   --model Qwen/Qwen3-8B   --tensor-parallel-size 4   --host 0.0.0.0   --port 8080`


def qwen3_api(
    user_prompt: str,
    system_prompt: str = "",
) -> str:
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "Qwen/Qwen3-8B",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
        "seed": 42,
        # "temperature": 0.0,  # ValueError: `temperature` (=0.0) has to be a strictly positive float, otherwise your next token scores will be invalid. If you're looking for greedy decoding strategies, set `do_sample=False`.
    }

    try:
        response = requests.post(
            "http://salmonn.hufs.jae.one:8080/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
        )

        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return {"error": response.text if response else str(e)}


if __name__ == "__main__":
    import re

    user_prompt_template = """
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
    system_prompt = """You are a good judge. You will be given a question with list of possible options, a ground truth answer and a model generated response. 
    You have to determine whether the model generated answer is correct."""

    user_prompt = (
        user_prompt_template.replace(
            "[QUESTION]",
            "What is the capital of France? (a) Paris (b) London (c) Berlin (d) Madrid",
        )
        .replace("[GROUND_TRUTH_ANSWER]", "(a) Paris")
        .replace("[MODEL_GENERATED_RESPONSE]", "The capital of France is Paris.")
    )
    response = qwen3_api(user_prompt=user_prompt, system_prompt=system_prompt)
    pattern = r"Explanation: (.*?)\nJudgement: (.*?)(?:\n\n|$)"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        explanation = match.group(1)
        judgement = match.group(2)
    else:
        explanation = "No extracted explanation"
        judgement = "No extracted judgement"

    results = {
        "Explanation": explanation,
        "Judgement": judgement,
    }
    print(results)
