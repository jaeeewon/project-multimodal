# create env
```bash
conda create -n qwen2audio python=3.11
conda activate qwen2audio

apt-get update
apt-get install openjdk-8-jdk

pip install "numpy<2.0"
pip install evaluate
pip install sacrebleu==1.5.1
pip install edit_distance
pip install editdistance
pip install jiwer
pip install scikit-image
pip install textdistance
pip install sed_eval
pip install more_itertools
pip install zhconv


pip install torch --extra-index-url=https://download.pytorch.org/whl/cu128
pip install accelerate
```

# run
```bash
checkpoint="Qwen/Qwen2-Audio-7B-Instruct"
ds="sakura"
python -m torch.distributed.launch --use-env \
    --nproc_per_node 4 --nnodes 1 \
    Qwen2-Audio/eval_audio/evaluate_sakura.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 24 \
    --num-workers 2
```