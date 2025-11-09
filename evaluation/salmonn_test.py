import torch
from evaluation.models.salmonn import SALMONNModel
from salmonn.utils import prepare_one_sample


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
