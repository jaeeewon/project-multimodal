```bash
torchrun --nproc_per_node=4 train.py --cfg-path ../configs/config.yaml
```

```bash
python train.py --cfg-path ../configs/config.yaml
```

```bash
ValueError: The model corresponding to this feature extractor: WhisperFeatureExtractor was trained using a sampling rate of 16000. Please make sure that the provided `raw_speech` input was sampled with 16000 and not 32000.
```
- [audio, sr = sf.read(ann["path"], samplerate=16000)](salmonn/dataset.py)

```bash
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 254.00 MiB. GPU 2 has a total capacity of 31.37 GiB of which 225.88 MiB is free. Including non-PyTorch memory, this process has 28.23 GiB memory in use. Of the allocated memory 27.34 GiB is allocated by PyTorch, and 102.35 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

```bash
(base) jpong@hufs_5090_4ea:~/Workspace/jaeeewon$ du -sh whisper-medium
23G     whisper-medium
(base) jpong@hufs_5090_4ea:~/Workspace/jaeeewon$ du -sh whisper-large-v2
47G     whisper-large-v2
(base) jpong@hufs_5090_4ea:~/Workspace/jaeeewon$ du -sh vicuna-7b-v1.1
26G     vicuna-7b-v1.1
(base) jpong@hufs_5090_4ea:~/Workspace/jaeeewon$ du -sh vicuna-13b-v1.1
50G     vicuna-13b-v1.1
```

```bash
The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.
`low_cpu_mem_usage` was None, now default to True since model is quantized.
```

```bash
tensorboard --logdir=/home/jpong/Workspace/jaeeewon/SALMONN_output/202509202203
```

```bash
grep -o '"path"' /home/jpong/Workspace/jaeeewon/repr_salmonn/ann/salmonn_stage1_data.json_valid_ensured.json | wc -w -> 68165
# [eval stage] 68,165(ann_cnt) / 4(gpu_cnt) = 17,041.25(ann_per_gpu) -> 17,042@cuda:0 --> 분산 학습이 잘 되고 있는 중!!!!
```