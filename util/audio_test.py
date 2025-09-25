import torchaudio
import soundfile as sf

# wav_path = "/home/jpong/Workspace/jaeeewon/repr_salmonn/salmonn/resource/audio_demo/duck.wav" # 16kHz, stereo
# wav_path = "/home/jpong/Workspace/jaeeewon/GigaSpeech/AUD0000000007_S0000005.wav" # 16kHz, mono
wav_path = "/home/jpong/Workspace/jaeeewon/AudioCaps/train/4rDF7G9cO44.wav" # 24kHz, mono

sf_audio, sf_sr = sf.read(wav_path) # audio shape: (160000,)
ta_audio, ta_sr = torchaudio.load(wav_path)

if ta_sr != 16000:
    ta_audio = torchaudio.transforms.Resample(orig_freq=ta_sr, new_freq=16000)(ta_audio)
#     sr = 16000
ta_audio = ta_audio.double().squeeze().numpy() # audio shape: (16000,)
ta_audio_transposed = ta_audio.transpose() # audio shape: (16000,)
# audio = audio.numpy() # audio shape: (16000,)
print(f"[sf] audio shape: {sf_audio.shape}, dtype: {sf_audio.dtype}")
print(f"[ta] audio shape: {ta_audio.shape}, dtype: {ta_audio.dtype}")
print(f"[ta_t] audio shape: {ta_audio_transposed.shape}, dtype: {ta_audio_transposed.dtype}")
# if len(audio.shape) == 2: # stereo to mono
#     audio = audio[:, 0]
# if len(audio) < sr: # pad audio to at least 1s
#     sil = np.zeros(sr - len(audio), dtype=float)
#     audio = np.concatenate((audio, sil), axis=0)
# audio = audio[: sr * 30] # truncate audio to at most 30s
# print(f"audio shape: {audio.shape}")

# soundfile 대신 torchaudo로 음성을 불러오게 했는데, 채널이 2개 이상일 때 두 라이브러리에서의 형상이 다름

# [sf] audio shape: (160000, 2)
# [ta] audio shape: (2, 160000)
# [ta_t] audio shape: (160000, 2)

# 이에, transpose를 취해주니 그제서야 의도대로 동작