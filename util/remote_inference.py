from omegaconf import OmegaConf
import torch
from transformers import WhisperFeatureExtractor

# <remote_import>
import sys
import os

current_file_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file_path))

salmonn_dir = os.path.join(root_dir, "salmonn")

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if salmonn_dir not in sys.path:
    sys.path.insert(0, salmonn_dir)


from salmonn.models.salmonn import SALMONN
from salmonn.utils import prepare_one_sample

# </remote_import>

salmonn_cfg = {
    "ckpt": "/home/jpong/Workspace/jaeeewon/SALMONN-7B/salmonn_7b_v0.pth",
    "llama_path": "/home/jpong/Workspace/jaeeewon/vicuna-7b-v1.5",
    "low_resource": False,
}


class Inference:
    def __init__(self, config_path: str, device: str, lora_scaling=4, use_7B=False):
        self.config = OmegaConf.load(config_path)
        if use_7B:
            self.config.model.ckpt = salmonn_cfg["ckpt"]
            self.config.model.llama_path = salmonn_cfg["llama_path"]
            self.config.model.low_resource = salmonn_cfg["low_resource"]
            # unset 8bit quantization for 7B model
            print(f"set use 7B model")
        self.config.model["lora_alpha"] = lora_scaling * self.config.model["lora_rank"]
        print(
            f"set lora_alpha to {self.config.model.lora_alpha} | lora_scaling: {lora_scaling}"
        )
        self.model = SALMONN.from_config(self.config.model)
        self.model.to(device=device)
        self.model.eval()
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(
            self.config.model.whisper_path
        )

    def infer_one_sample(self, wav_path: str, prompt: str):
        samples = prepare_one_sample(wav_path, self.wav_processor)
        prompt = [
            self.config.model.prompt_template.format(
                "<Speech><SpeechHere></Speech> " + prompt.strip()
            )
        ]
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return self.model.generate(samples, self.config.generate, prompts=prompt)[0]


if __name__ == "__main__":
    inference = Inference(config_path="configs/infer_config.yaml", device="cuda:0")
    infered = inference.infer_one_sample(
        "salmonn/resource/audio_demo/duck.wav",
        "Can you transcribe the speech into a written format?",
    )

    print(infered)
