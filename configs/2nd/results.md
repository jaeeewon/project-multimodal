# attempt 2

## experimental settings
- randomly split `salmonn_stage1_data.json` into train, validation and test set with 90:5:5 ratio
- use smaller speech model `whisper-large-v2` &rarr; `whisper-medium`
- use smaller llm `vicuna-13b-v1.1` &rarr; `vicuna-7b-v1.1`
- ~~load llm in 8bit for low resource~~
- use torchrun for distributed learning
- **reduced batch size `8` &rarr; `7`, not keeping the ratio**
<!-- - doubled gradient accumulation to 2 and halved batch size twice to 4, keeping the ratio -->

## log

### train | 1st epoch | completed
```log
Train: data epoch: [0]  [2999/3000]  eta: 0:00:00  lr: 0.000030  loss: 3.3003  time: 0.4025  data: 0.0000  max mem: 26767
Train: data epoch: [0] Total time: 0:20:12 (0.4042 s / it)
/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
2025-09-21 02:05:53,410 [INFO] Averaged stats: lr: 0.0000  loss: 3.310
```

### eval | 1st epoch | completed
```log
Eval: data epoch: [0]  [2440/2441]  eta: 0:00:00    time: 0.2441  data: 0.0001  max mem: 26767
Eval: data epoch: [0] Total time: 0:10:36 (0.2606 s / it)
2025-09-21 02:16:30,828 [INFO] Saving checkpoint at epoch 0 to /home/jpong/Workspace/jaeeewon/SALMONN_output/202509210144/checkpoint_best.pth.
2025-09-21 02:16:31,084 [INFO] Saving checkpoint at epoch 0 to /home/jpong/Workspace/jaeeewon/SALMONN_output/202509210144/checkpoint_0.pth
```

### train | 2nd epoch | completed
```log
Train: data epoch: [1]  [2999/3000]  eta: 0:00:00  lr: 0.000030  loss: 3.4354  time: 0.4031  data: 0.0000  max mem: 26811
Train: data epoch: [1] Total time: 0:20:13 (0.4046 s / it)
2025-09-21 02:36:45,225 [INFO] Averaged stats: lr: 0.0000  loss: 3.1111
```

### eval | 2nd epoch | completed
```log
Eval: data epoch: [1]  [2440/2441]  eta: 0:00:00    time: 0.2452  data: 0.0002  max mem: 26811
Eval: data epoch: [1] Total time: 0:10:34 (0.2601 s / it)
2025-09-21 02:47:23,336 [INFO] Saving checkpoint at epoch 1 to /home/jpong/Workspace/jaeeewon/SALMONN_output/202509210144/checkpoint_best.pth.
2025-09-21 02:47:23,722 [INFO] Saving checkpoint at epoch 1 to /home/jpong/Workspace/jaeeewon/SALMONN_output/202509210144/checkpoint_1.pth
```

## result

### first epoch
#### train | 1st epoch | completed
```json
{"train_lr": "0.000", "train_loss": "3.310"}
```
#### eval | 1st epoch | completed
```json
{"valid_loss": 3.076000690460205, "valid_agg_metrics": 0.3925953209400177, "valid_best_epoch": 0}
```
#### train | 2nd epoch | completed
```json
{"train_lr": "0.000", "train_loss": "3.111"}
```
#### eval | 2nd epoch | completed
```json
{"valid_loss": 3.0485262870788574, "valid_agg_metrics": 0.39634376764297485, "valid_best_epoch": 1}
```