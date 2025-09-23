from datasets import load_dataset  # 3.6.0
import aac_datasets
from tqdm import tqdm
import os
import soundfile as sf
import librosa

download_path = "/home/jpong/Workspace/jaeeewon"

"""
datas = {
    "GigaSpeech": "speechcolab/gigaspeech",
    "WavCaps": "cvssp/WavCaps",
    "CommonVoice": "mozilla-foundation/common_voice_17_0"
}

for key in datas:
    ds = load_dataset(datas[key], cache_dir=f"{download_path}/{key}")
    print(key, ds)
"""

# ===== start of WavCaps =====

# print(
#     aac_datasets.WavCaps(
#         download_path,
#         download=True,
#         subset="audioset_no_audiocaps_v1",
#         force_download=True,
#     )
# )  # default
# print(
#     aac_datasets.WavCaps(
#         download_path,
#         download=True,
#         subset="freesound_no_clotho_v2",
#         force_download=True,
#     )
# )  # current | 더 다운받았지만 여전히 29.18%, 39.69%


# /home/jpong/Workspace/jaeeewon/WavCaps/Audio/AudioSet_SL/Y__-8wVOYH3c.flac
# WavCaps/AudioSet_SL이 돼야 함

# mv Audio/* .

# ===== end of WavCaps =====

# ===== start of AudioCaps =====
# requirements
# sudo apt install ffmpeg
# pip install yt-dlp

# print(aac_datasets.AudioCaps(download_path, download=True))

audiocaps = load_dataset(
    "OpenSound/AudioCaps",
    cache_dir="./hf",
)

target_set = "train"
output_dir = f"./dataset/AudioCaps/{target_set}"
os.makedirs(output_dir, exist_ok=True)

for item in tqdm(audiocaps[target_set], desc="extracting AudioCaps set"):
    sampling_rate = item["audio"]["sampling_rate"]
    audio_data = librosa.resample(
        y=item["audio"]["array"], orig_sr=sampling_rate, target_sr=16000
    )
    filename = f"{item['youtube_id']}.wav"
    output_path = os.path.join(output_dir, filename)

    sf.write(output_path, audio_data, sampling_rate)

print(f"extracted {len(audiocaps[target_set])} sounds of AudioCaps {target_set}")

# ===== end of AudioCaps =====


# ===== start of Clotho =====
# another way to prepare Clotho 24h data
# https://github.com/qiuqiangkong/audio_understanding/blob/main/scripts/download_clotho.sh

# print(aac_datasets.Clotho(download_path, download=True, version="v1", force_download=False, subset="dev"))
# print(aac_datasets.Clotho(download_path, download=True, version="v2", force_download=False, subset="dev"))
# print(aac_datasets.Clotho(download_path, download=True, version="v2.1", force_download=False, subset="dev"))
# print(aac_datasets.Clotho(download_path, download=True, version="v1", force_download=False, subset="val"))
# print(aac_datasets.Clotho(download_path, download=True, version="v1", force_download=False, subset="eval"))
# ===== end of Clotho


# ===== start of GigaSpeech M-set =====

# gs = load_dataset(
#     "speechcolab/gigaspeech",
#     "m",
#     token="hf_xxx"
# ) # download ("xs", "s", "m")

# # see structure
# print(gs)

# # load audio sample on the fly
# audio_input = gs["train"][0]["audio"]  # first decoded audio sample
# transcription = gs["train"][0]["text"]  # first transcription

# /home/jpong/.cache/huggingface/datasets/downloads/extracted/4d0d6f35f3cc6bdb9e26cd1242e2fb1e47b7eae2e140d1ce3e4351e7045bb6c7
# -> mv /home/jpong/.cache/huggingface/datasets/downloads/extracted/4d0d6f35f3cc6bdb9e26cd1242e2fb1e47b7eae2e140d1ce3e4351e7045bb6c7/*/* GigaSpeech/
# ad33dd78eefc95e72fd18615fda855b083e190323936e06ea3fbd1360bfcd766
# -> mv /home/jpong/.cache/huggingface/datasets/downloads/extracted/ad33dd78eefc95e72fd18615fda855b083e190323936e06ea3fbd1360bfcd766/*/* GigaSpeech/
# d508c69be8f72b78b242aeb0f3ed8abb0873979ce979c5f1f397914912563248
# -> mv /home/jpong/.cache/huggingface/datasets/downloads/extracted/d508c69be8f72b78b242aeb0f3ed8abb0873979ce979c5f1f397914912563248/*/* GigaSpeech/

# shit!!!!! run under
"""
SOURCE_DIR="/home/jpong/.cache/huggingface/datasets/downloads/extracted"
DEST_DIR="GigaSpeech"

for dir in "$SOURCE_DIR"/*/*/; do
  mv "$dir"* "$DEST_DIR"/
done
"""

# ===== end of GigaSpeech M-set =====
