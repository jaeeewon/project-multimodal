# CUDA_VISIBLE_DEVICES=1 python dist_eval/works.py

import os
from init_works import SalmonnRedis

gpu_devices = (
    os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else None
)
device = "cuda:0"


def get_utils(device: str, lora_scaling: int = None):
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


# r = SalmonnRedis(host="192.168.219.101", db=0)
# r.start_worker("en2ja", device, eval_en2ja)

# r = SalmonnRedis(host="192.168.219.101", db=1)
# r.start_worker("en2de", device, eval_en2de)

# r = SalmonnRedis(host="192.168.219.101", db=2) # cuda:0
# r.start_worker("LibriSpeech-ASR-test-clean", device, eval_librispeech_asr)

# r = SalmonnRedis(host="192.168.219.101", db=3) # cuda:2
# r.start_worker("LibriSpeech-ASR-test-other", device, eval_librispeech_asr)

# r = SalmonnRedis(host="192.168.219.101", db=4) # cuda:3
# r.start_worker("en2zh", device, eval_en2zh)

# r = SalmonnRedis(host="192.168.219.101", db=5)
# r.start_worker("GigaSpeech-ASR-test", device, eval_gigaspeech_asr)

# if gpu_devices in ["0", "1", "2", "3"]:
#     ls = int(gpu_devices)
#     r = SalmonnRedis(host="192.168.219.101", db=2)
#     r.start_worker(
#         f"LibriSpeech-ASR-test-clean-ls{ls:02d}",
#         device,
#         eval_librispeech_asr,
#     )

#     r = SalmonnRedis(host="192.168.219.101", db=3)
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
#         r = SalmonnRedis(host="192.168.219.101", db=5)
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

# r = SalmonnRedis(host="192.168.219.101", db=6)
# r.start_worker("AudioCaps-AAC-test", device, eval_audiocaps_aac)

inference, bleu4_score, remove_puncs = get_utils(device, lora_scaling=4)
r = SalmonnRedis(host="192.168.219.101", db=2)
r.start_worker("LibriSpeech-PR-test-clean", device, eval_librispeech_pr)
