# create env
```bash
conda create -n qwenaudio python=3.11
conda activate qwenaudio

apt-get update
apt-get install openjdk-8-jdk

pip install "numpy<2.0"
# pip install evaluate sacrebleu==1.5.1
# pip install pycocoevalcap
# pip install edit_distance editdistance
# mkdir -p Qwen-Audio/eval_audio/caption_evaluation_tools
# git clone https://github.com/audio-captioning/caption-evaluation-tools.git Qwen-Audio/eval_audio/caption_evaluation_tools
# cd Qwen-Audio/eval_audio/caption_evaluation_tools/coco_caption/
# bash ./get_stanford_models.sh
# cd ../../../..
# pip install sacrebleu
# pip install sacrebleu[ja]
pip install sed_eval
pip install dcase_util
pip install -r Qwen-Audio/requirements.txt

pip install torch --extra-index-url=https://download.pytorch.org/whl/cu128
pip install accelerate
```

# modify
- transformers_stream_generator/main.py, line12
- from transformers.generation.beam_search import BeamSearchScorer

# run
```bash
ds="sakura"
python -m torch.distributed.launch --use-env \
    --nproc_per_node 4 --nnodes 1 \
    Qwen-Audio/eval_audio/evaluate_sakura.py \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```