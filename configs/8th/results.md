# attempt 8

- bug: avg grad_norm is inf

## experimental settings
- randomly split `salmonn_stage1_data.json` into train, validation and test set with 90:5:5 ratio
- use smaller speech model `whisper-large-v2` &rarr; `whisper-medium`
- use smaller llm `vicuna-13b-v1.1` &rarr; `vicuna-7b-v1.1`
- ~~load llm in 8bit for low resource~~
- use torchrun for distributed learning
- reduced batch size `8` &rarr; `6` ~~and doubled gradient accumulation `1` &rarr; `2`~~, ~~not~~ keeping the ratio
- scaled `warmup_start_lr`, `init_lr` and `min_lr` by 0.75x ~~and scaled `init_lr` by 0.5x~~
- ~~scaled `warmup_steps` by 3x~~
- ~~applied gradient clipping `1`~~
- scaled gradient clipping `1` &rarr; `10`

## log

### train | 1st epoch | completed
```bash
Train: data epoch: [0]  [2999/3000]  eta: 0:00:00  lr: 0.000022  loss: 2.7363  grad_norm: 2.8635  time: 0.3651  data: 0.0000  max mem: 24757
Train: data epoch: [0] Total time: 0:18:14 (0.3647 s / it)
/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
2025-09-21 20:57:44,461 [INFO] Averaged stats: lr: 0.0000  loss: 3.3059  grad_norm: inf
```

## result

![loss_graph](attempt8_loss.svg)
- loss until epoch 1

![loss_graph](attempt8_grad_norm.svg)
- grad norm until epoch 1

![lr_graph](attempt8_lr.svg)
- lr until epoch 1

![loss_comparison_6_7_8](attempt8_loss_comparison.svg)
- compared to attempt 6, 7 and 8, it is clear that current config is the best

### first epoch
#### train | 1st epoch | completed
```json
{"train_lr": "0.000", "train_loss": "3.306", "train_grad_norm": "inf"}
```