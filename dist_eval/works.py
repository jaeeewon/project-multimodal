# CUDA_VISIBLE_DEVICES=1 python dist_eval/works.py

from init_works import SalmonnRedis

device = "cuda:0"


def get_utils(device: str):
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

    return (
        Inference(config_path="configs/infer_config.yaml", device=device),
        bleu4_score,
        remove_puncs,
    )


inference, bleu4_score, remove_puncs = get_utils(device)


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

    return result


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

    return result


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

    return result


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

    return result


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

    return result


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

r = SalmonnRedis(host="192.168.219.101", db=5)
r.start_worker("GigaSpeech-ASR-test", device, eval_gigaspeech_asr)
