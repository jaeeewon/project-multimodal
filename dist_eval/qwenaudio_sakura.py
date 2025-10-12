import json, pandas as pd, re
from qwen3 import qwen3_api
from tqdm import tqdm

# from init_works import SalmonnRedis

# meta_path = "lalms/open-source/qwen2-audio-instruct/sakura_ignore_qwen2audio_meta.jsonl"
# inf_path = "lalms/open-source/qwen2-audio-instruct/sakura_ignore_qwen2audio_eval_251012135455.json"
# store_path = "lalms/open-source/qwen2-audio-instruct/sakura_llm-as-a-judge.json"
meta_path = "lalms/open-source/qwen-audio-chat/sakura_ignore_qwenaudio_meta.jsonl"
inf_path = "lalms/open-source/qwen-audio-chat/sakura_251012211038.json"
store_path = "lalms/open-source/qwen-audio-chat/sakura_llm-as-a-judge.json"


metas = []
with open(meta_path, "r") as f:
    for line in f:
        metas.append(json.loads(line))

with open(
    inf_path,
    "r",
) as f:
    infs = json.load(f)

assert len(metas) == len(infs)

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

evaled = []
for meta, inf in tqdm(zip(metas, infs), total=len(metas)):

    # qwen2audio
    # meta: {"id", "audio", "query", "answer"}
    # inf: {"idx", "response", "audio_path"}
    # question = meta["query"].replace("<|audio_bos|><|AUDIO|><|audio_eos|>\n", "")
    # response = inf["response"]
    # ground_truth_answer = meta["answer"]

    # qwenaudio
    # meta: {"audio", "question", "gt", "source"}
    # inf: {"gt", "response", "source", "audio_path"}
    question = meta["question"]
    response = inf["response"]
    ground_truth_answer = meta["gt"]



    user_prompt = (
        user_prompt_template.replace("[QUESTION]", question)
        .replace("[MODEL_GENERATED_RESPONSE]", response)
        .replace("[GROUND_TRUTH_ANSWER]", ground_truth_answer)
    )

    judgement = qwen3_api(user_prompt, system_prompt=system_prompt)

    # ===== extract_judgement@llm_judge.py =====
    pattern = r"Explanation: (.*?)\nJudgement: (.*?)(?:\n\n|$)"
    match = re.search(pattern, judgement, re.DOTALL)

    if match:
        explanation = match.group(1)
        judgement = match.group(2)
    else:
        explanation = "No extracted explanation"
        judgement = "No extracted judgement"
    evaled.append(judgement)

with open(
    store_path,
    "w",
) as f:
    json.dump(evaled, f, indent=4)