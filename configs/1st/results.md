# attempt 1

## experimental settings
- randomly split `salmonn_stage1_data.json` into train, validation and test set with 80:10:10 ratio
- use smaller speech model `whisper-large-v2` &rarr; `whisper-medium`
- use smaller llm `vicuna-13b-v1.1` &rarr; `vicuna-7b-v1.1`
- load llm in 8bit for low resource
- use torchrun for distributed learning

## log

### train | 1st epoch | completed
```log
Train: data epoch: [0]  [2999/3000]  eta: 0:00:00  lr: 0.000030  loss: 3.5553  time: 0.4929  data: 0.0000  max mem: 26527
Train: data epoch: [0] Total time: 0:24:39 (0.4930 s / it)
/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
2025-09-21 00:22:31,434 [INFO] Averaged stats: lr: 0.0000  loss: 3.291
```

### eval | 1st epoch | completed
```log
Eval: data epoch: [0]  [4253/4254]  eta: 0:00:00    time: 0.2819  data: 0.0002  max mem: 26527
Eval: data epoch: [0] Total time: 0:20:32 (0.2897 s / it)
2025-09-21 00:43:10,500 [INFO] Saving checkpoint at epoch 0 to /home/jpong/Workspace/jaeeewon/SALMONN_output/202509202357/checkpoint_best.pth.
2025-09-21 00:43:10,770 [INFO] Saving checkpoint at epoch 0 to /home/jpong/Workspace/jaeeewon/SALMONN_output/202509202357/checkpoint_0.pth.
```

### train | 2nd epoch | crashed
```log
Train: data epoch: [1]  [ 285/3000]  eta: 0:22:16  lr: 0.000030  loss: 2.7421  time: 0.4907  data: 0.0000  max mem: 26527
[rank3]: Traceback (most recent call last):
[rank3]:   File "/home/jpong/Workspace/jaeeewon/repr_salmonn/salmonn/train.py", line 91, in <module>
[rank3]:     main()
[rank3]:   File "/home/jpong/Workspace/jaeeewon/repr_salmonn/salmonn/train.py", line 87, in main
[rank3]:     runner.train()
[rank3]:   File "/home/jpong/Workspace/jaeeewon/repr_salmonn/salmonn/runner.py", line 276, in train
[rank3]:     train_stats = self.train_epoch(cur_epoch)
[rank3]:   File "/home/jpong/Workspace/jaeeewon/repr_salmonn/salmonn/runner.py", line 116, in train_epoch
[rank3]:     samples = next(self.train_loader)
[rank3]:   File "/home/jpong/Workspace/jaeeewon/repr_salmonn/salmonn/utils.py", line 121, in __next__
[rank3]:     data = next(self.iter_loader)
[rank3]:   File "/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 734, in __next__
[rank3]:     data = self._next_data()
[rank3]:   File "/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1516, in _next_data
[rank3]:     return self._process_data(data, worker_id)
[rank3]:   File "/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1551, in _process_data
[rank3]:     data.reraise()
[rank3]:   File "/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/_utils.py", line 769, in reraise
[rank3]:     raise exception
[rank3]: soundfile.LibsndfileError: <exception str() failed>
W0921 00:45:35.396792 495335 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 495396 closing signal SIGTERM
W0921 00:45:35.397614 495335 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 495401 closing signal SIGTERM
W0921 00:45:35.398696 495335 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 495403 closing signal SIGTERM
E0921 00:45:35.400784 495335 site-packages/torch/distributed/elastic/multiprocessing/api.py:874] failed (exitcode: 1) local_rank: 3 (pid: 495404) of binary: /home/jpong/miniconda3/envs/salmonn/bin/python3.9
Traceback (most recent call last):
  File "/home/jpong/miniconda3/envs/salmonn/bin/torchrun", line 7, in <module>
    sys.exit(main())
  File "/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 357, in wrapper
    return f(*args, **kwargs)
  File "/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/distributed/run.py", line 901, in main
    run(args)
  File "/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/distributed/run.py", line 892, in run
    elastic_launch(
  File "/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/distributed/launcher/api.py", line 143, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/jpong/miniconda3/envs/salmonn/lib/python3.9/site-packages/torch/distributed/launcher/api.py", line 277, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
train.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-09-21_00:45:35
  host      : hufs_5090_4ea
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 495404)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
```

## result

### first epoch
#### train | 1st epoch | completed
```json
{"train_lr": "0.000", "train_loss": "3.291"}
```
#### eval | 1st epoch | completed
```json
{"valid_loss": 3.066974401473999, "valid_agg_metrics": 0.3930814266204834, "valid_best_epoch": 0}
```