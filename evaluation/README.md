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

- AbstractModel
  - Model: 모델 로드와 추론 코드를 포함하는 AbsModel의 subclass
  - method
    - load_model(): 모델 로드
    - infer(): 모델 추론
  - properties
    - model_name: 모델 이름
    - model_path: 모델 경로(로드에 사용됨)
    - device: 모델이 로드될 장치
    - tokenizer: 모델에 사용되는 토크나이저
    - config: 모델 설정

- AbstractEvaluator
  - Evaluator: 특정 평가 task에 대한 평가 코드를 포함하는 AbsEvaluator의 subclass
  - method
    - preprocess(): 평가 전에 결과를 전처리하는 메서드
    - evaluate(): 모델 출력과 정답을 비교하여 평가 점수를 계산하는 메서드
  - properties
    - task_name: 평가 작업 이름
    - metric: 평가 지표 이름

- AbstractDataProvider # Iterable
  - DataProvider: 추론에 필요한 모든 요소들을 포함하는 AbsDataProvider의 subclass
  - data
    - key: 데이터셋 저장에 사용되는 키
    - audio_path: 추론에 사용되는 오디오 파일 경로
    - prompt: 추론에 사용되는 프롬프트
    - query: 추론에 사용되는 쿼리
    - input: 추론에 사용되는 추가 데이터 딕셔너리
    - ground_truth: 평가에 사용되는 정답

- Examples
  - AbstractModel.infer(DataProvider, callback_fn) -> list of inference results
  - AbstractEvaluator.evaluate(DataProvider) -> dict of evaluation results

---

### subclass

- RedisDataProvider(host, num_db, key, ds, insert=False)
  - Redis에서 데이터를 로드하는 DataProvider의 subclass

---
