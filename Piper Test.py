import wave
from piper import PiperVoice
from piper import SynthesisConfig
import sounddevice as sd
import soundfile as sf


"""syn_config = SynthesisConfig(
    volume=1.0,  # half as loud
    length_scale=1.0,  # twice as slow
    noise_scale=1.2,  # more audio variation
    noise_w_scale=0.8,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)"""

text = "I have run simulations on every known element, and none can serve as a viable replacement for the paladium core,"

voice = PiperVoice.load("en_US-danny-low.onnx")
with wave.open("test.wav", "wb") as wav_file:
    voice.synthesize_wav(text, wav_file)


data, sr = sf.read("test.wav", dtype="int16")
sd.play(data, sr)
sd.wait()

