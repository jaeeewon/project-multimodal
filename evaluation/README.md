# Refactor evaluation codes

- Encapsulate evaluation codes into `evaluation` folder
- Extensible structure for adding new tasks and models
- Instant checking for experiments

---

## model specification

- SALMONN(7B, 13B)
- Qwen-Audio
- Qwen2-Audio-Instruct

---

## tasks specification

- SAKURA
    - Gender
        - single-hop | 500 samples
        - multi-hop | 500 samples
    - Language
        - single-hop | 500 samples
        - multi-hop | 500 samples
    - Emotion
        - single-hop | 500 samples
        - multi-hop | 500 samples
    - Animal
        - single-hop | 500 samples
        - multi-hop | 500 samples
- ASR
    - LibriSpeech test-clean
    - LibriSpeech test-other
- and more...

## structure specification
- AbstractEvaluator
    - SAKURAEvaluator
    - and more...

---
