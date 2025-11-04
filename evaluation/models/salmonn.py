from pathlib import Path
import torch, sys

sys.path.append(str(Path(__file__).parent.parent.parent / "salmonn"))

from ..utils.document import Document
from omegaconf import OmegaConf
from ..abs.model import AbstractModel
from transformers import WhisperFeatureExtractor
from salmonn.models.salmonn import SALMONN
from salmonn.models.utils import StoppingCriteriaSub
from salmonn.utils import get_dataloader, prepare_one_sample, prepare_sample
from salmonn.dataset import SALMONNDataset


class SALMONNModel(AbstractModel):
    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        super().__init__(**self.config.abs_class)
        self.load_model()

    def load_model(self) -> None:
        self.model = SALMONN.from_config(self.config.model)
        self.model.to(device=self.config.model_class.device)
        self.model.eval()
        self.tokenizer = self.model.llama_tokenizer
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(self.config.model.whisper_path)

    def _inference(self, batch: list[Document]) -> list[str]:
        """
        batch: [ { "wav": [audio], "query": [text] }, ... ] # non-necessary keys and values will be discarded
        """
        samples = [prepare_one_sample(dta["wav"], self.wav_processor) for dta in batch]
        spectrograms = [s["spectrogram"].squeeze() for s in samples]
        raw_wavs = [s["raw_wav"].squeeze() for s in samples]
        padding_masks = [s["padding_mask"].squeeze() for s in samples]

        samples = {
            "spectrogram": torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True, padding_value=0.0),
            "raw_wav": torch.nn.utils.rnn.pad_sequence(raw_wavs, batch_first=True, padding_value=0.0),
            "padding_mask": torch.nn.utils.rnn.pad_sequence(padding_masks, batch_first=True, padding_value=0.0),
        }

        prompts = [
            self.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + dta["query"].strip())
            for dta in batch
        ]
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return self.model.generate(samples, self.config.generate, prompts=prompts, skip_special_tokens=True)


if __name__ == "__main__":
    import os
    from ..data.sakura import SakuraDataProvider, RedisConfig

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    model = SALMONNModel(config_path="configs/eval_13b.yaml")
    data_provider = SakuraDataProvider(
        redis_cfg=RedisConfig(host="salmonn.hufs.jae.one", port=6379, db=9),
        key_prefix="salmonn-13b:sakura",
        required_fields=["wav", "query"],
        filter={"set": "language", "hop": "single"},
    )

    def callback_fn(sample, inference):
        sample["status"] = "inferenced"
        # print(sample)

    model.infer(
        data_provider,
        batch_size=2,
        # callback_fn=lambda sample, inference: print(f"SALMONN output for {sample['query']}:\n {inference}"),
        callback_fn=callback_fn,
    )

    for sample in iter(data_provider):
        print("\n===== SALMONN output =====")
        print(f"{sample['query']}\n > {sample['inference']}")
