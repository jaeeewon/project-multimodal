import os, numpy as np, soundfile as sf
import edge_tts
from pydub import AudioSegment
import pyrubberband as pyrb

# 텍스트 입력
text = "A detailed finite element model of the human kidney for trauma research has been created directly from the National Library of Medicine Visible Human Female (VHF) Project data set. An image segmentation and organ reconstruction software package has been developed and employed to transform the 2D VHF images into a 3D polygonal representation. Nonuniform rational B-spline (NURBS) surfaces were then mapped to the polygonal surfaces, and were finally utilized to create a robust 3D hexahedral finite element mesh within a commercially available meshing software. The model employs a combined viscoelastic and hyperelastic material model to successfully simulate the behaviour of biological soft tissues. The finite element model was then validated for use in biomechanical research"  # "hello!!!!"
voice = "en-US-AvaNeural"  # https://github.com/hasscc/hass-edge-tts/blob/9c41a93ce8f3366065de7a1cfa30242526bbdb31/custom_components/edge_tts/const.py#L114C6-L114C22

tts = edge_tts.Communicate(text, voice)

sound_file = "test_edge.mp3"
tts.save_sync(sound_file)

audio = AudioSegment.from_mp3(sound_file)

audio = audio.set_frame_rate(16000)

speech = np.array(audio.get_array_of_samples())

if len(speech) > 16000 * 29.9:
    rate = len(speech) / (16000 * 29.9)
    print(f"speed up from {len(speech)/16000:.2f}s to 29.9s (x{rate:.2f})")
    # speech = librosa.effects.time_stretch(speech, rate=rate) # bad quality
    speech = pyrb.time_stretch(speech, 16000, rate=rate)

sf.write("test_edge.wav", speech, 16000)
