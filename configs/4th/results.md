# attempt 3

## experimental settings
- randomly split `salmonn_stage1_data.json` into train, validation and test set with 90:5:5 ratio
- use smaller speech model `whisper-large-v2` &rarr; `whisper-medium`
- use smaller llm `vicuna-13b-v1.1` &rarr; `vicuna-7b-v1.1`
- ~~load llm in 8bit for low resource~~
- use torchrun for distributed learning
- reduced batch size `8` &rarr; `6` and doubled gradient accumulation `1` &rarr; `2`, not keeping the ratio
- ~~scaled `warmup_start_lr`, `init_lr` and `min_lr` by 1.5x~~

## log

## result
