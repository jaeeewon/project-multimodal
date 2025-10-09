import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np
from nltk.tokenize import sent_tokenize
import librosa, time
import pyrubberband as pyrb
import requests
import edge_tts
from pydub import AudioSegment

url = "https://papago.naver.com/apis/tts/makeID"
clova_host = "https://papago.naver.com/apis/tts/{}"


class TTS:
    def __init__(self):
        return
        self.device = "cuda:0"

        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(
            self.device
        )

        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
        self.speaker_embeddings = (
            torch.tensor(embeddings_dataset[7306]["xvector"])
            .unsqueeze(0)
            .to(self.device)
        )

        self.max_token = 550  # max: 600

    def _generate(self, text):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        spectrogram = self.model.generate_speech(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            speaker_embeddings=self.speaker_embeddings,
        )

        with torch.no_grad():
            speech = self.vocoder(spectrogram)
        return speech.squeeze().cpu().numpy()

    def local_generate(self, text, speedup=True):
        speeches = []
        sentences = sentences = sent_tokenize(text)
        for i, sent in enumerate(sentences):
            speeches.append(self._generate(sent))
            # if i != len(sentences) - 1:
            #     speeches.append(np.zeros(int(0.25 * 16000), dtype=np.float32))
        speech = np.concatenate(speeches)

        if speedup and len(speech) > 16000 * 29.9:
            rate = len(speech) / (16000 * 29.9)
            print(f"speed up from {len(speech)/16000:.2f}s to 29.9s (x{rate:.2f})")
            # speech = librosa.effects.time_stretch(speech, rate=rate) # bad quality
            speech = pyrb.time_stretch(speech, 16000, rate=rate)
            # librosa: Fine element model, human kinetics, image segmentation and organ reconstruction
            # pyrb: Fine element model, human kidney for trauma research, image segmentation and organ reconstruction
            # answer: 'high-fidelity finite element model', 'kidney', 'trauma research', 'National Library of Medicine', 'image segmentation', 'organ reconstruction', 'software package', '2D VHF images', '3D polygonal representation', 'NURBS', 'polygonal surfaces', '3D hexahedral finite element mesh', 'hyperelastic material model', 'biological soft tissues', 'biomechanical research', 'Visible Human Female project', 'medical data set', 'physically based animation', 'nonuniform rational B-spline surfaces', 'viscoelastic model'

        return speech

    def clova_generate(self, text, path):
        form_data = {
            "alpha": 0,
            "pitch": 0,
            "speaker": "clara",
            "speed": 0,
            "text": text,
        }

        response = requests.post(url, data=form_data)
        # response.raise_for_status()

        if response.status_code != 200:
            raise Exception(f"status {response.status_code}: {response.text}")

        r = response.json()
        response = requests.get(clova_host.format(r["id"]), stream=True)
        response.raise_for_status()

        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def edge_generate(self, text, path, voice="en-US-AvaNeural"):
        # https://github.com/hasscc/hass-edge-tts/blob/9c41a93ce8f3366065de7a1cfa30242526bbdb31/custom_components/edge_tts/const.py#L114C6-L114C22
        res = edge_tts.Communicate(text, voice)

        sound_file = path + ".mp3"
        res.save_sync(sound_file)

        audio = AudioSegment.from_mp3(sound_file)

        audio = audio.set_frame_rate(16000)

        speech = np.array(audio.get_array_of_samples())

        if len(speech) > 16000 * 29.9:
            rate = len(speech) / (16000 * 29.9)
            print(f"speed up from {len(speech)/16000:.2f}s to 29.9s (x{rate:.2f})")
            # speech = librosa.effects.time_stretch(speech, rate=rate) # bad quality
            speech = pyrb.time_stretch(speech, 16000, rate=rate)

        sf.write(path, speech, 16000)


if __name__ == "__main__":
    tts = TTS()
    text = "A detailed finite element model of the human kidney for trauma research has been created directly from the National Library of Medicine Visible Human Female (VHF) Project data set. An image segmentation and organ reconstruction software package has been developed and employed to transform the 2D VHF images into a 3D polygonal representation. Nonuniform rational B-spline (NURBS) surfaces were then mapped to the polygonal surfaces, and were finally utilized to create a robust 3D hexahedral finite element mesh within a commercially available meshing software. The model employs a combined viscoelastic and hyperelastic material model to successfully simulate the behaviour of biological soft tissues. The finite element model was then validated for use in biomechanical research"
    t1 = time.perf_counter()
    tts.clova_generate(text, "clova_test.mp3")
    print("elapsed:", time.perf_counter() - t1)
