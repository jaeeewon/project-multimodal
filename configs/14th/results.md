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
```json
{"train_lr": "0.000", "train_loss": "3.284"}
{"valid_loss": 3.0861737728118896, "valid_agg_metrics": 0.39642372727394104, "valid_best_epoch": 0}
{"train_lr": "0.000", "train_loss": "3.109"}
{"valid_loss": 3.0639214515686035, "valid_agg_metrics": 0.39976343512535095, "valid_best_epoch": 1}
{"train_lr": "0.000", "train_loss": "3.091"}
{"valid_loss": 3.0458645820617676, "valid_agg_metrics": 0.4011201858520508, "valid_best_epoch": 2}
```