import wave
from piper import PiperVoice
from piper import SynthesisConfig
import sounddevice as sd
import soundfile as sf
from Configs.AppConfig import AppConfig

cfg = AppConfig.from_yaml("Configs/config.yaml")

class TTS:
    def __init__(self):
        self.voice = PiperVoice.load(cfg.tts.voice_path)
        self.syn_config = SynthesisConfig(
        speaker_id = cfg.tts.speaker_id,
        volume = cfg.tts.volume,  # half as loud
        normalize_audio = cfg.tts.normalize_audio) # use raw audio from voice

    def writeSayChunk(self, text, file):
        with wave.open(file, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file, syn_config=self.syn_config)

        data, sr = sf.read(file, dtype="int16")

        sd.play(data, sr)
        sd.wait()    
    