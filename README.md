# reproduce [SALMONN: speech audio language music open neural network](https://arxiv.org/abs/2310.13289)


## environments
- CPU: Intel(R) Xeon(R) w7-3465X
- RAM: 64GB * 4
- GPU: RTX5090 32GB * 4

## step 1: prepare datasets
<details>
<summary>ongoing preparation</summary>

### CommonVoice - prepared but both converting and resampling are needed
- https://commonvoice.mozilla.org/en/datasets
- Common Voice Corpus 4	1/14/2020	38.6 GB	1,489	1,119	CC-0	51,072	MP3
- untar .tar.gz, convert .mp3 to .wav and (resample 48k to 16k)
- ln -s \<path-to-untarred>/clips \<path-to-CommonVoice>/train

### LibriSpeech - prepared
- symlinked what we've used training ESPnet ASR model

### GigaSpeech - prepared
- downloaded from huggingface and moved

### WavCaps - partially prepared (39.69% for QA and 29.18% for audiocaption)
- downloaded from aac_datasets
- symlink /DB/WavCaps WavCaps first
- located at /home/jpong/.cache/huggingface/hub/datasets--cvssp--WavCaps/snapshots/85a0c21e26fa7696a5a74ce54fada99a9b43c6de
- symlink using `ln -s /home/jpong/.cache/huggingface/hub/datasets--cvssp--WavCaps/snapshots/85a0c21e26fa7696a5a74ce54fada99a9b43c6de/Audio WavCaps`
    - WavCaps/AudioSet_SL | prepared
    - WavCaps/BBC_Sound_Effects | yet
    - WavCaps/FreeSound | yet

### Clotho - partially prepared (75.36%)
- downloaded from huggingface | Clotho-v1
- some checks are required,,

### AudioCaps - ing
- downloaded from huggingface | https://huggingface.co/datasets/OpenSound/AudioCaps
- symlink is planned

</details>

## step 2: TBD