from omegaconf import OmegaConf
import os
from logit_processor import InteractiveLogitsProcessor

os.environ["MXNET_FC_TRUE_FP16"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
import torch, time, random, numpy as np, pandas as pd
from transformers import WhisperFeatureExtractor, StoppingCriteriaList
from sakura_datasets import get_sakura_ds, get_sakura_wrong_ds
from tqdm import tqdm

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
from salmonn.models.utils import StoppingCriteriaSub
from salmonn.utils import get_dataloader, prepare_one_sample, prepare_sample
from salmonn.dataset import SALMONNDataset

# </remote_import>


def setup_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)


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
        print(f"set lora_alpha to {self.config.model.lora_alpha} | lora_scaling: {lora_scaling}")
        self.model = SALMONN.from_config(self.config.model)
        self.model.to(device=device)
        self.model.eval()
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(self.config.model.whisper_path)

    def infer_one_sample(self, wav_path: str, prompt: str, skip_special_tokens=False):
        samples = prepare_one_sample(wav_path, self.wav_processor)
        prompt = [self.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())]
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return self.model.generate(
                samples,
                self.config.generate,
                prompts=prompt,
                skip_special_tokens=skip_special_tokens,
            )[0]

    def infer_samples(self, wav_path: list[str], prompt: str):
        samples = [prepare_one_sample(pth, self.wav_processor) for pth in wav_path]
        spectrograms = [s["spectrogram"].squeeze() for s in samples]
        raw_wavs = [s["raw_wav"].squeeze() for s in samples]
        padding_masks = [s["padding_mask"].squeeze() for s in samples]

        samples = {
            "spectrogram": torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True, padding_value=0.0),
            "raw_wav": torch.nn.utils.rnn.pad_sequence(raw_wavs, batch_first=True, padding_value=0.0),
            "padding_mask": torch.nn.utils.rnn.pad_sequence(padding_masks, batch_first=True, padding_value=0.0),
        }
        # print("spectrogram", samples["spectrogram"].size())
        # print("raw_wav", samples["raw_wav"].size())
        # print("padding_mask", samples["padding_mask"].size())
        # print("prepared samples for eval")
        # print(samples)

        prompt = [self.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())]
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # with torch.no_grad():
            return self.model.generate(samples, self.config.generate, prompts=prompt)

    def _get_sakura_loader(self):
        testsets = SALMONNDataset(get_sakura_ds(), self.wav_processor, task="sakura")
        test_loader = get_dataloader(
            dataset=testsets,
            config=self.config.run,
            is_train=False,
            use_distributed=False,
        )
        return test_loader

    def eval_sakura(self, scoring=True):
        dataloader = self._get_sakura_loader()

        ids, hyps, refs = [], [], []

        for samples in tqdm(dataloader):
            # print("spectrogram", samples["spectrogram"].size())
            # print("raw_wav", samples["raw_wav"].size())
            # print("padding_mask", samples["padding_mask"].size())
            _id = samples["idx"]
            ids.extend(_id)

            # Preprocess
            samples = prepare_sample(samples, cuda_enabled=torch.cuda.is_available())
            # print("prepared samples for eval")
            # print(samples)
            prompts = [
                self.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + q.strip())
                for q in samples["query"]
            ]
            # print("prompts:", prompts)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                # with torch.no_grad():
                results = self.model.generate(
                    samples,
                    self.config.generate,
                    prompts=prompts,
                    skip_special_tokens=True,
                )
            """print("prepared samples")
            print(samples)
            batch_size = samples["spectrogram"].shape[0]
            print(f"batch size: {batch_size}")

            spectrogram = samples["spectrogram"]
            raw_wav = samples.get("raw_wav", None)
            audio_padding_mask = samples.get("padding_mask", None)

            speech_embeds, speech_atts = self.model.encode_speech(
                spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
            )

            # Add prompt embeds + audio embed
            prompts = [
                "<Speech><SpeechHere></Speech> " + q.strip() for q in samples["query"]
            ]
            templated_prompts = [
                self.config.model.prompt_template.format(prompt) for prompt in prompts
            ]
            print(templated_prompts)

            speech_embeds, speech_atts = self.model.prompt_wrap(
                speech_embeds, speech_atts, templated_prompts, multi_prompt=True
            )
            bos = (
                torch.ones(
                    [batch_size, 1],
                    dtype=torch.int32,
                    device=speech_embeds.device,
                )
                * self.model.llama_tokenizer.bos_token_id
            )

            bos_embeds = self.model.llama_model.model.model.embed_tokens(bos)
            atts_bos = speech_atts[:, :1]

            embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
            attns = torch.cat([atts_bos, speech_atts], dim=1)

            generate_cfg = self.config.generate

            stop_words_ids = [torch.tensor([2]).cuda()]
            stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids)]
            )

            # Generation
            t1 = time.perf_counter()
            outputs = self.model.llama_model.generate(
                inputs_embeds=embeds,
                max_new_tokens=generate_cfg.get("max_new_tokens", 200),
                stopping_criteria=stopping_criteria,
                num_beams=generate_cfg.get("num_beams", 4),
                do_sample=generate_cfg.get("do_sample", False),
                min_length=generate_cfg.get("min_length", 1),
                temperature=generate_cfg.get("temperature", 1.0),
                top_p=generate_cfg.get("top_p", 0.9),
                repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
                length_penalty=generate_cfg.get("length_penalty", 1.0),
                attention_mask=attns,
            )
            print(f"inference time: {time.perf_counter() - t1}s")

            t1 = time.perf_counter()
            results = self.model.llama_tokenizer.batch_decode(
                outputs,
                add_special_tokens=False,
            )
            print(f"decoding time: {time.perf_counter() - t1}s")"""
            print(results)
            hyps.extend(results)

            if scoring:
                ref = samples["text"]
                refs.extend(ref)

        if scoring:
            print('scoring "hyps" and "refs"')

        df = pd.DataFrame({"id": ids, "hyps": results, "refs": ref})
        df.to_markdown("evaled_sakura.md")

    def experiment(self):
        testsets = SALMONNDataset(get_sakura_wrong_ds(), self.wav_processor, task="sakura")
        dataloader = get_dataloader(
            dataset=testsets,
            config=self.config.run,
            is_train=False,
            use_distributed=False,
        )

        ids, hyps, refs = [], [], []

        for samples in tqdm(dataloader):
            _id = samples["idx"]
            ids.extend(_id)

            samples = prepare_sample(samples, cuda_enabled=torch.cuda.is_available())
            prompts = [
                self.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + q.strip())
                for q in samples["query"]
            ]

            with torch.cuda.amp.autocast(dtype=torch.float16):
                batch_size = samples["spectrogram"].shape[0]

                spectrogram = samples["spectrogram"]
                raw_wav = samples.get("raw_wav", None)
                audio_padding_mask = samples.get("padding_mask", None)

                speech_embeds, speech_atts = self.model.encode_speech(
                    spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
                )

                if prompts is not None:
                    speech_embeds, speech_atts = self.model.prompt_wrap(
                        speech_embeds, speech_atts, prompts, multi_prompt=True
                    )

                bos = (
                    torch.ones(
                        [batch_size, 1],
                        dtype=torch.int32,
                        device=speech_embeds.device,
                    )
                    * self.model.llama_tokenizer.bos_token_id
                )
                bos_embeds = (
                    self.model.llama_model.model.embed_tokens(bos)
                    if not self.model.lora
                    else self.model.llama_model.model.model.embed_tokens(bos)
                )
                atts_bos = speech_atts[:, :1]

                embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
                attns = torch.cat([atts_bos, speech_atts], dim=1)

                stop_words_ids = [torch.tensor([2]).cuda()]
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

                generate_cfg = self.config.generate

                interactive_processor = InteractiveLogitsProcessor(tokenizer=self.model.llama_tokenizer, k=5)

                configs = {
                    "num_beams": 1,
                    "output_scores": True,
                    "return_dict_in_generate": True,
                    "logits_processor": [interactive_processor],
                    "do_sample": False,
                }

                print("answers:", samples["text"])

                texts = self.model.llama_tokenizer.batch_decode(
                    self.model.llama_model.generate(
                        inputs_embeds=embeds,
                        max_new_tokens=generate_cfg.get("max_new_tokens", 200),
                        stopping_criteria=stopping_criteria,
                        num_beams=generate_cfg.get("num_beams", 4),
                        do_sample=generate_cfg.get("do_sample", False),
                        min_length=generate_cfg.get("min_length", 1),
                        temperature=generate_cfg.get("temperature", 1.0),
                        top_p=generate_cfg.get("top_p", 0.9),
                        repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
                        length_penalty=generate_cfg.get("length_penalty", 1.0),
                        attention_mask=attns,
                    ),
                    add_special_tokens=False,
                    skip_special_tokens=True,
                )

                print("initial results:", texts)

                outputs = self.model.llama_model.generate(
                    inputs_embeds=embeds,
                    max_new_tokens=generate_cfg.get("max_new_tokens", 200),
                    stopping_criteria=stopping_criteria,
                    # num_beams=generate_cfg.get("num_beams", 4),
                    # do_sample=generate_cfg.get("do_sample", False),
                    min_length=generate_cfg.get("min_length", 1),
                    temperature=generate_cfg.get("temperature", 1.0),
                    top_p=generate_cfg.get("top_p", 0.9),
                    repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
                    length_penalty=generate_cfg.get("length_penalty", 1.0),
                    attention_mask=attns,
                    **configs,
                )

                texts = self.model.llama_tokenizer.batch_decode(
                    outputs.sequences, add_special_tokens=False, skip_special_tokens=True
                )

                print("=" * 50)
                print("questions:", samples["query"])
                print("answers:", samples["text"])
                print("final results:", texts)
            hyps.extend(texts)

            ref = samples["text"]
            refs.extend(ref)

        # print('scoring "hyps" and "refs"')

        df = pd.DataFrame({"id": ids, "hyps": hyps, "refs": refs})
        df.to_markdown("results/evaled_sakura_exp.md", index=False)


if __name__ == "__main__":
    setup_seeds(42)
    inference = Inference(config_path="configs/infer_config.yaml", device="cuda")

    inference.experiment()

    # infered = inference.infer_one_sample(
    #     "salmonn/resource/audio_demo/duck.wav",
    #     "Can you transcribe the speech into a written format?",
    # )

    # infered = inference.infer_samples(
    #     ["/home/jpong/Workspace/jaeeewon/repr/sakura/data/Emotion/audio/1078_IEO_ANG_HI.wav"], "What is the emotional tone of the speaker in the audio? (a) disgust (b) fear (c) sad (d) angry"
    # )
    # print(infered)

    # inference.eval_sakura()

    # for i in range(1, 51):
    #     t1 = time.perf_counter()
    #     print(f"===== batch={i} =====")
    #     samples = [
    #         "salmonn/resource/audio_demo/duck.wav",
    #     ] * i
    #     # samples = ["salmonn/resource/audio_demo/mountain.wav"] * i
    #     # samples = random.choices(
    #     #     [
    #     #         "salmonn/resource/audio_demo/duck.wav",
    #     #         "salmonn/resource/audio_demo/mountain.wav",
    #     #     ],
    #     #     k=i,
    #     # )
    #     infered = inference.infer_samples(
    #         samples,
    #         # "Can you transcribe the speech into a written format?",
    #         "What do you hear?",
    #     )
    #     print(infered)
    #     print(f"elapsed {time.perf_counter() - t1}s")
    #     print()
