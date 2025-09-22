# attempt 14

- bug: avg grad_norm is inf
- current change: rollback lr-family

## experimental settings
- randomly split `salmonn_stage1_data.json` into train, validation and test set with 99:0.5:0.5 ratio
- ~~use smaller speech model `whisper-large-v2` &rarr; `whisper-medium`~~
- ~~use smaller llm `vicuna-13b-v1.1` &rarr; `vicuna-7b-v1.1`~~
- load llm in 8bit for low resource
- use torchrun for distributed learning
- **reduced batch size `8` &rarr; `4`** and ~~doubled gradient accumulation `1` &rarr; `4`, not keeping the ratio~~
- ~~scaled `warmup_start_lr`, `init_lr` and `min_lr` by 0.75x and scaled `init_lr` by 0.5x~~
- ~~scaled `warmup_steps` by 3x~~
- ~~applied gradient clipping `1`~~
- ~~scaled gradient clipping `1` &rarr; `10`~~

## log

## result
