# types per each task by stage
## stage 1
### asr
```json
{"path": "/LibriSpeech/train-clean-100/path_to_audio.flac", "text": "recognized", "task": "asr"}
```
### audiocaption
```json
{"path": "/AudioCaps/train/file_name.wav", "text": "captioned", "task": "audiocaption"}
```

## stage 2
### asr
### audiocaption
### audiocaption_v2
```json
{"path": "/Clotho/train/file_name.wav", "text": "captioned", "task": "audiocaption_v2"}
```
### translation_ec
```json
{"path": "/CommonVoice/train/file_name.wav", "text": "chinese", "task": "translation_ec"}
```
### phone_recognition
```json
{"path": "/LibriSpeech/train-clean-100/path_to_audio.flac", "text": "recognized", "task": "phone_recognition"}
```
### emotion_recognition
```json
{"path": "/IEMOCAP/IEMOCAP_full_release/Session2/sentences/wav/Ses02M_impro04/file_name.wav", "text": "emotion", "task": "emotion_recognition"}
```
### music_description
```json
{"path": "/MusicCaps/file_name.wav", "text": "description", "task": "music_description"}
```
### QA
```json
{"path": "/MillionSongDatasetSpotify/file_name.wav", "text": "answer", "task": "QA", "Q": "question"}
```
### speech_separation
```json
{"path": "/LibriMix/mixdata/Libri2Mix/wav16k/max/train-360/mix_clean/file_name.wav", "text": "data", "task": "speech_separation"}
```
### speaker_verification
```json
{"path": "/Voxceleb1/dev/wav/id10001/path_to_audio.wav", "text": "Yes", "task": "speaker_verification", "expand_wav": ["/mnt/bn/audio-visual-llm-data/datasets/Voxceleb1/dev/wav/id10001/path_to_audio.wav"]}
```
### gender_recognition
```json
{"path": "/LibriSpeech/train-clean-100/path_to_audio.flac", "text": "Female", "task": "gender_recognition"}
```

## stage 3
### audio_story_telling
```json
{"path": "/AudioCaps/train/file_name.wav", "text": "story", "task": "audio_story_telling"}
```