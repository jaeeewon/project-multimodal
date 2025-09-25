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


class Inference:
    def __init__(self, config_path: str, device: str):
        self.config = OmegaConf.load(config_path)
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
