# CUDA_VISIBLE_DEVICES=1 python dist_eval/works.py

import os, re
from init_works import SalmonnRedis
from qwen3 import qwen3_api
from tqdm import tqdm

gpu_devices = (
    os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else None
)
device = "cuda:0"


def get_utils(device: str, lora_scaling: int = None, use_7B=False):
    import sys
    import os

    current_file_path = os.path.abspath(__file__)
    curr_dir = os.path.dirname(current_file_path)
    root_dir = os.path.join(curr_dir, "../")

    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    from util.remote_inference import Inference
    from util.bleu import bleu4_score
    from util.str import remove_puncs

    args = {
        "config_path": "configs/infer_config.yaml",
        "device": device,
        "use_7B": use_7B,
    }

    if lora_scaling in [0, 1, 2, 3]:
        args["lora_scaling"] = lora_scaling
        print(f"set lora_scaling: {args['lora_scaling']}")

    # if gpu_devices in ["0", "1", "2", "3"]:
    #     sync_gpu_to_lora = (
    #         input("sync gpu id to lora scaling? (Y/N) > ").strip().lower()
    #     )
    #     if sync_gpu_to_lora in ["y", "yes"]:
    #         args["lora_scaling"] = int(gpu_devices)
    #         print(f"[warning] sync lora_scaling to gpu: {args['lora_scaling']}")

    return (
        Inference(**args),
        bleu4_score,
        remove_puncs,
    )


inference, bleu4_score, remove_puncs = None, None, None  # get_utils(device)


def eval_en2ja(data: dict):
    print(f"evaluating {data['path']}...")
    path = "/home/jpong/Workspace/jaeeewon/CommonVoice/clips/" + data["path"]
    prompt = "Listen to the speech and translate it into Japanese."

    reference = data["translation"]
    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _reference = remove_puncs(reference)
    _result = remove_puncs(result)

    print(f"ref: {_reference}")
    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_en2de(data: dict):
    print(f"evaluating {data['path']}...")
    path = "/home/jpong/Workspace/jaeeewon/CommonVoice/clips/" + data["path"]
    prompt = "Listen to the speech and translate it into German."

    reference = data["translation"]
    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _reference = remove_puncs(reference)
    _result = remove_puncs(result)

    print(f"ref: {_reference}")
    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_en2zh(data: dict):
    print(f"evaluating {data['path']}...")
    path = "/home/jpong/Workspace/jaeeewon/CommonVoice/clips/" + data["path"]
    prompt = "Listen to the speech and translate it into Chinese."

    reference = data["translation"]
    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _reference = remove_puncs(reference)
    _result = remove_puncs(result)

    print(f"ref: {_reference}")
    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_librispeech_asr(data: dict):
    print(f"evaluating {data['path']}...")
    path = data["path"]
    prompt = "Recognize the speech and give me the transcription."

    reference = data["sentence"]
    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _reference = remove_puncs(reference)
    _result = remove_puncs(result)

    print(f"ref: {_reference}")
    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_gigaspeech_asr(data: dict):
    print(f"evaluating {data['file_name']}...")
    path = "/home/jpong/Workspace/jaeeewon/GigaSpeech/" + data["file_name"] + ".wav"
    prompt = "Recognize the speech and give me the transcription."

    reference = data["sentence"]
    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _reference = remove_puncs(reference)
    _result = remove_puncs(result)

    print(f"ref: {_reference}")
    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_audiocaps_aac(data: dict):
    print(f"evaluating {data['youtube_id']}...")
    path = "dataset/AudioCaps/test/" + data["youtube_id"] + ".wav"
    prompt = "Please describe the audio."
    prompt_v2 = "Please write down what your hear in the audio."

    reference = data["caption"]
    result = inference.infer_one_sample(wav_path=path, prompt=prompt)
    result_v2 = inference.infer_one_sample(wav_path=path, prompt=prompt_v2)

    _reference = remove_puncs(reference)
    _result = remove_puncs(result)
    _result_v2 = remove_puncs(result_v2)

    print(f"ref: {_reference}")
    print(f"res: {_result}")
    print(f"res_v2: {_result_v2}")
    print("=" * 20)

    return {"infer": result, "infer_v2": result_v2}


def eval_audiocaps_story(data: dict):
    print(f"evaluating {data['youtube_id']}...")
    path = "dataset/AudioCaps/test/" + data["youtube_id"] + ".wav"
    prompt = "Based on the audio, write a story in detail. Your story should be highly related to the audio."

    reference = data["caption"]
    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _result = remove_puncs(result)

    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_librispeech_pr(data: dict):
    print(f"evaluating {data['path']}...")
    path = f"/home/jpong/Workspace/jaeeewon/{data['path']}"
    prompt = "Provide the phonetic transcription for the speech."

    reference = data["text"]
    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _reference = remove_puncs(reference)
    _result = remove_puncs(result)

    print(f"ref: {_reference}")
    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_sakura(data: dict):
    print(f"evaluating {data['file']}...")
    path = data["file"]
    prompt = data["instruction"]

    reference = data["answer"]
    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _reference = remove_puncs(reference)
    _result = remove_puncs(result)

    print(f"ref: {_reference}")
    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_sakura_judge(data: dict):
    print(f"judging {data['file']}...")

    question = data["instruction"]
    response = data["infer"]
    ground_truth_answer = data["answer"]
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

    results = {
        "Explanation" + sakura_judge_pf: explanation,
        "Judgement" + sakura_judge_pf: judgement,
    }
    # ===== extract_judgement@llm_judge.py =====

    if judgement != "correct":
        print(f"answer: {ground_truth_answer}")
        print(f"model response: {response}")
        print(f"Explanation: {explanation}")
    print(f"Judgement: {judgement}")
    print("=" * 20)

    return results


def eval_iemocap_er(data: dict):
    print(f"evaluating {data['path']}...")
    path = data["path"]
    prompt = "Describe the emotion of the speaker in one word."

    reference = data["emotion"]
    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _reference = remove_puncs(reference)
    _result = remove_puncs(result)

    print(f"ref: {_reference}")
    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_musiccaps_mc(data: dict):
    print(f"evaluating {data['path']}...")
    path = data["path"]
    prompt = "Listen to this music clip and describe the music."

    reference = data["caption"]
    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _reference = remove_puncs(reference)
    _result = remove_puncs(result)

    print(f"ref: {_reference}")
    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_voxceleb1_sv(data: dict):
    print(f"evaluating {data['path']}...")
    path = data["path"]
    prompt = "Do you only hear the same person talking? Answer yes or no."

    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _result = remove_puncs(result)

    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_inspec_ke(data: dict):
    print(f"evaluating {data['path']}...")
    path = data["path"]
    prompt = "Give me only three keywords of the text."

    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _result = remove_puncs(result)

    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_wikiqa_sqqa(data: dict):
    print(f"evaluating {data['path']}...")
    path = data["path"]
    prompt = "Please answer the question in detail."

    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    _result = remove_puncs(result)

    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


def eval_slurp_sf(data: dict):
    print(f"evaluating {data['path']}...")
    path = data["path"]
    infer = []
    spl = data["slots"].strip().split("; ")
    for i, slot in enumerate(spl):
        splt = slot.strip().split("=")
        assert len(splt) == 2, f"invalid slot: {slot}"

        prompt = "According to the speech, what is the {}?".format(splt[0])
        result = inference.infer_one_sample(wav_path=path, prompt=prompt)
        _result = remove_puncs(result)
        infer.append(f"{splt[0]}={result}")
        print(f"{i+1}/{len(spl)} [{splt[0]}] | ans: {splt[1]} | res: {_result}")

    print("=" * 20)

    return {"infer": "; ".join(infer)}


def eval_librimix_osr(data: dict):
    print(f"evaluating {data['path']}...")
    path = data["path"]
    prompt = "Please write down what you hear each person says."

    result = inference.infer_one_sample(wav_path=path, prompt=prompt)

    a1, a2 = data["ans1"], data["ans2"]
    _a1, _a2 = remove_puncs(a1), remove_puncs(a2)
    _result = remove_puncs(result)

    print(f"ans1: {_a1}")
    print(f"ans2: {_a2}")
    print(f"res: {_result}")
    print("=" * 20)

    return {"infer": result}


# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=0)
# r.start_worker("en2ja", device, eval_en2ja)

# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=1)
# r.start_worker("en2de", device, eval_en2de)

# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=2) # cuda:0
# r.start_worker("LibriSpeech-ASR-test-clean", device, eval_librispeech_asr)

# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=3) # cuda:2
# r.start_worker("LibriSpeech-ASR-test-other", device, eval_librispeech_asr)

# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=4) # cuda:3
# r.start_worker("en2zh", device, eval_en2zh)

# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=5)
# r.start_worker("GigaSpeech-ASR-test", device, eval_gigaspeech_asr)

# if gpu_devices in ["0", "1", "2", "3"]:
#     ls = int(gpu_devices)
#     r = SalmonnRedis(host="salmonn.hufs.jae.one", db=2)
#     r.start_worker(
#         f"LibriSpeech-ASR-test-clean-ls{ls:02d}",
#         device,
#         eval_librispeech_asr,
#     )

#     r = SalmonnRedis(host="salmonn.hufs.jae.one", db=3)
#     r.start_worker(
#         f"LibriSpeech-ASR-test-other-ls{ls:02d}",
#         device,
#         eval_librispeech_asr,
#     )

# if gpu_devices in ["0", "1", "2", "3"]:
#     ls = int(gpu_devices)
#     for i in range(ls, ls + 4):
#         i %= 4
#         worker_name = f"GigaSpeech-ASR-test-ls{i:02d}"
#         r = SalmonnRedis(host="salmonn.hufs.jae.one", db=5)
#         print(f"===== start {worker_name} =====")
#         print(f"load model with lora scaling {i}")
#         if inference is not None:
#             del inference
#         inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=i)
#         r.start_worker(
#             worker_name,
#             device,
#             eval_gigaspeech_asr,
#         )
#         r.statistics(worker_name)
#         print(f"===== end {worker_name} =====")

# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=6)
# r.start_worker("AudioCaps-AAC-test", device, eval_audiocaps_aac)

# inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=4)
# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=2)
# r.start_worker("LibriSpeech-PR-test-clean", device, eval_librispeech_pr)

# if gpu_devices in ["0", "1", "2", "3"]:
#     ls = int(gpu_devices)
#     for i in range(ls, ls + 4):
#         i %= 4
#         worker_name = f"LibriSpeech-PR-test-clean-ls{i:02d}"
#         r = SalmonnRedis(host="salmonn.hufs.jae.one", db=2)
#         print(f"===== start {worker_name} =====")
#         print(f"load model with lora scaling {i}")
#         if inference is not None:
#             del inference
#         inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=i)
#         r.start_worker(
#             worker_name,
#             device,
#             eval_librispeech_pr,
#         )
#         r.statistics(worker_name)
#         print(f"===== end {worker_name} =====")

# inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=4)
# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=6)
# r.start_worker("AudioCaps-Story-test", device, eval_audiocaps_story)


# if inference is not None:
#     del inference

# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=7)
# sakura_tracks = ["Animal", "Emotion", "Gender", "Language"]
# if gpu_devices in ["0", "1", "2", "3"]:
#     ls = int(gpu_devices)
#     for i in range(ls, ls + 4):
#         i %= 4
#         track = sakura_tracks[i]
#         for hop in ["single", "multi"]:
#             for pf, use_7B in [
#                 ("", False),
#                 ("-7B", True),
#             ]:
#                 if inference is not None:
#                     del inference
#                 inference, bleu4_score, remove_puncs = get_utils(device, use_7B=use_7B)
#                 worker_name = f"SAKURA-{track}-{hop}{pf}"
#                 print(f"===== start {worker_name} =====")
#                 r.start_worker(
#                     worker_name,
#                     device,
#                     eval_sakura,
#                 )
#                 r.statistics(worker_name)
#                 print(f"===== end {worker_name} =====")

# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=7)
# user_prompt_template = """
#     You will be given a question with list of possible options, a ground truth answer and a model generated response. Determine whether the model generated response is correct based on the following criteria:
#     1. Since there is one and only one corect answer, it should be judged incorrect if the model do not choose any option from the option list or it choose more than one option.
#     2. If the model choose one option from the option list, it should be judged correct if the chosen option aligns with the ground truth answer, otherwise it should be judged incorrect.
#     3. Read the question, options, ground truth answer and model generated response carefully before making a decision.

#     Considering the following examples:
#     Question: What is the capital of France? (a) Paris (b) London (c) Berlin (d) Madrid
#     Ground truth answer: (a) Paris
#     If the model generated response is: "The capital of France is Tokyo.", it should be judged incorrect since it does not choose any option from the option list.
#     If the model generated response is: "The capital of France is Paris and London.", it should be judged incorrect since it chooses more than one option from the option list.
#     If the model generated response is: "The capital of France is London.", it should be judged incorrect since it chooses one option from the option list but the chosen option does not align with the ground truth answer.
#     If the model generated response is: "The capital of France is Paris.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
#     Another Question: What is the underlying emotion of the speaker? (a) Happy (b) Sad (c) Angry (d) Neutral
#     Ground truth answer: (a) Happy
#     If the model generated response is: "The speaker is happy.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
#     If the model generated response is: "The speaker expresses happiness.", it should be judged correct since "happiness" aligns with the ground truth answer "happy", and they are just different part of speech of the same word.
#     If the model generated response is: "Happiness," it should be judged correct since it is also a valid derivative of the ground truth answer "happy".

#     Now here is the question and the model generated response for you to judge:
#     Question: [QUESTION]
#     Ground truth answer: [GROUND_TRUTH_ANSWER]
#     Model generated response: [MODEL_GENERATED_RESPONSE]

#     Carefully make your decision based on the above criteria. Return your judgement with the following format:
#     Explanation: <Your explanation on your judgement>
#     Judgement: <Your judgement, either "correct" or "incorrect">
#     """
# system_prompt = """You are a good judge. You will be given a question with list of possible options, a ground truth answer and a model generated response.
# #                     You have to determine whether the model generated answer is correct."""
# sakura_tracks = ["Animal", "Emotion", "Gender", "Language"]
# sakura_judge_pf = "-judge-qwen3"

# # ===== init SAKURA judge tasks =====
# if gpu_devices == "0":  # prevent multiple tasks init
#     for pf in ["", "-7B"]:
#         for track in sakura_tracks:
#             for hop in ["single", "multi"]:
#                 task_name = f"SAKURA-{track}-{hop}{pf}"
#                 TASK_HASH_PREFIX = SalmonnRedis.TASK_HASH_PREFIX.format(task_name)
#                 passkeys = [
#                     SalmonnRedis.PENDING_QUEUE.format(task_name),
#                     SalmonnRedis.PROCESSING_QUEUE.format(task_name),
#                     SalmonnRedis.PENDING_QUEUE.format(task_name) + sakura_judge_pf,
#                     SalmonnRedis.PROCESSING_QUEUE.format(task_name) + sakura_judge_pf,
#                 ]

#                 task_keys = [
#                     k
#                     for k in r.client.scan_iter(f"{TASK_HASH_PREFIX}*")
#                     if not k.startswith(tuple(passkeys))
#                 ]

#                 for key in task_keys:
#                     task_data = r.client.hgetall(key)

#                     status = task_data.get("status")

#                     if "worker" not in status and status == "completed":
#                         task_id = task_data.get("id")
#                         PENDING_QUEUE = (
#                             SalmonnRedis.PENDING_QUEUE.format(task_name)
#                             + sakura_judge_pf
#                         )

#                         judge_key = f"Judgement{sakura_judge_pf}"
#                         if judge_key in task_data and task_data[judge_key] in [
#                             "correct",
#                             "incorrect",
#                         ]:
#                             continue
#                         r.client.lpush(PENDING_QUEUE, task_id)

# ===== start SAKURA judge workers =====
# for pf in ["", "-7B"]:
#     for track in sakura_tracks:
#         for hop in ["single", "multi"]:
#             task_name = f"SAKURA-{track}-{hop}{pf}"
#             print(f"===== start {task_name} =====")
#             r.start_worker(task_name, device, eval_sakura_judge, pf=sakura_judge_pf)

# inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=4)
# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=8)
# # r.start_worker("IEMOCAP-ER", device, eval_iemocap_er)

# inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=4)
# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=8)
# r.start_worker("MusicCaps-MC", device, eval_musiccaps_mc)

# inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=4)
# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=8)
# r.start_worker("VoxCeleb1-SV", device, eval_voxceleb1_sv)

# inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=4)
# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=8)
# r.start_worker("Inspec-KE", device, eval_inspec_ke)

# inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=4)
# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=8)
# r.start_worker("WikiQA-SQQA", device, eval_wikiqa_sqqa)

# inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=4)
# r = SalmonnRedis(host="salmonn.hufs.jae.one", db=8)
# r.start_worker("Slurp-SF", device, eval_slurp_sf)

inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=4)
r = SalmonnRedis(host="salmonn.hufs.jae.one", db=8)
r.start_worker("LibriMix-OSR", device, eval_librimix_osr)
