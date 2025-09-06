from dataclasses import dataclass
import yaml

@dataclass
class TTSConfig:
    voice_path: str = "voices/en_US-danny-low.onnx"
    sample_rate: int = 22050
    device: str = "cuda"
    speaker_id: int = 55
    normalize_audio: bool = False
    volume: float = 0.8

@dataclass
class STTConfig:
    model: str = "medium"
    device: str = "cuda"
    beam_size: int = 5

@dataclass
class LLMConfig:
    endpoint: str
    model: str
    max_tokens: int = 2048
    temperature: float = 0.2

@dataclass
class AppConfig:
    tts: TTSConfig
    stt: STTConfig
    llm: LLMConfig

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            tts=TTSConfig(**d["tts"]),
            stt=STTConfig(**d["stt"]),
            llm=LLMConfig(**d["llm"]),
        )

cfg = AppConfig.from_yaml("configs/config.yaml")
# usage:
speaker_id = cfg.tts.speaker_id
stt_model  = cfg.stt.model
llm_name   = cfg.llm.model



