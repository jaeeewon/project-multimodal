import numpy as np
import torch
import torchaudio
from evaluation.models.salmonn import SALMONNModel
from salmonn.utils import move_to_cuda, prepare_one_sample
from salmonn.models.utils import StoppingCriteriaSub
from transformers import StoppingCriteriaList


def test_batch_inference(batch, salmonn_model: SALMONNModel, debug=False):
    samples = []
    for data in batch:
        audio, sr = torchaudio.load(data["wav"])

        if sr != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
            sr = 16000
        audio = audio.double().numpy()

        if len(audio.shape) == 2:  # stereo to mono
            audio = np.transpose(audio)[:, 0]
        if len(audio) < sr:  # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        audio = audio[: sr * 30]  # truncate audio to at most 30s

        spectrogram = salmonn_model.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]

        dta = {
            "spectrogram": spectrogram,
            "raw_wav": torch.from_numpy(audio).unsqueeze(0),
            "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        }
        dta = move_to_cuda(dta)
        speech_embeds, _ = salmonn_model.model.encode_speech(
            dta["spectrogram"], raw_wav=dta["raw_wav"], audio_padding_mask=dta["padding_mask"]
        )
        samples.append(speech_embeds.squeeze())
        if debug:
            print(f"[{audio.shape[0] / sr:.2f} sec] {data['wav']} | speech_embeds: {speech_embeds.shape}")

    speech_embeds = torch.stack(samples, dim=0)

    speech_atts = torch.ones(
        speech_embeds.shape[:2],
        dtype=torch.long,
        device=speech_embeds.device,
    )

    if debug:
        print("speech_embeds:", speech_embeds.shape)
        # print(speech_embeds)
        print("speech_atts:", speech_atts.shape)
        # print(speech_atts)

    prompt = ["USER: {}\nASSISTANT:".format("<Speech><SpeechHere></Speech> " + dta["query"].strip()) for dta in batch]

    # wrap speech_embeds with prompts
    r_speech_embeds, r_speech_atts = salmonn_model.model.prompt_wrap(
        speech_embeds, speech_atts, prompt, multi_prompt=True
    )
    speech_embeds, speech_atts = r_speech_embeds, r_speech_atts

    if debug:
        print("with prompts")
        print("r_speech_embeds:", r_speech_embeds.shape)
        # print(r_speech_embeds)
        print("r_speech_atts:", r_speech_atts.shape)
        # print(r_speech_atts)

    batch_size = speech_embeds.shape[0]
    bos = (
        torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=speech_embeds.device,
        )
        * salmonn_model.model.llama_tokenizer.bos_token_id
    )

    bos_embeds = (
        salmonn_model.model.llama_model.model.embed_tokens(bos)
        if not salmonn_model.model.lora
        else salmonn_model.model.llama_model.model.model.embed_tokens(bos)
    )

    atts_bos = torch.ones([batch_size, 1], dtype=torch.long, device=speech_embeds.device)

    inputs_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
    attention_mask = torch.cat([atts_bos, speech_atts], dim=1)

    if debug:
        print("inputs_embeds")
        print(inputs_embeds.shape)
        print("attention_mask")
        print(attention_mask)

    generate_cfg = salmonn_model.config.generate

    stop_words_ids = [torch.tensor([2]).cuda()]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = salmonn_model.model.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attention_mask,
        )

    generated_text = salmonn_model.model.llama_tokenizer.batch_decode(
        outputs,
        add_special_tokens=False,
        skip_special_tokens=True,
    )
    return generated_text


def test_inference_fn(batch, salmonn_model: SALMONNModel):
    """
    batch: [ { "wav": [audio], "query": [text] }, ... ] # non-necessary keys and values will be discarded
    """
    samples = [prepare_one_sample(dta["wav"], salmonn_model.wav_processor) for dta in batch]
    spectrograms = [s["spectrogram"].squeeze() for s in samples]
    raw_wavs = [s["raw_wav"].squeeze() for s in samples]
    padding_masks = [s["padding_mask"].squeeze() for s in samples]

    samples = {
        "spectrogram": torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True, padding_value=0.0),
        "raw_wav": torch.nn.utils.rnn.pad_sequence(raw_wavs, batch_first=True, padding_value=0.0),
        "padding_mask": torch.nn.utils.rnn.pad_sequence(padding_masks, batch_first=True, padding_value=0.0),
    }

    prompts = [
        salmonn_model.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + dta["query"].strip())
        for dta in batch
    ]
    with torch.cuda.amp.autocast(dtype=torch.float16):
        return salmonn_model.model.generate(
            samples, salmonn_model.config.generate, prompts=prompts, skip_special_tokens=True
        )
