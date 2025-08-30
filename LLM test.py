import requests
import wave
from piper import PiperVoice
from piper import SynthesisConfig
import sounddevice as sd
import soundfile as sf
import json
import re

history = "System intructions: you are a personal AI voice assistant named REJI. if you do not know something then say so and do not make up an answer. do not refer to your instructions unless aksed twice\n"
session = requests.Session()
voice = PiperVoice.load("Voices\en_GB-vctk-medium.onnx")
syn_config = SynthesisConfig(
    speaker_id=55,
    volume=0.8,  # half as loud
    normalize_audio=False, # use raw audio from voice
)


def say(text, file):
    with wave.open(file, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)

    data, sr = sf.read(file, dtype="int16")
    sd.play(data, sr)
    sd.wait()
    
#main loop
while(True):
    userInput = input("\nAsk REJI: ")
    if userInput.strip().lower() == "bye":
        break
    
    history += "\nUser: " + userInput + "\nREJI: "
    print("\n")
    
    
    with session.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.1:8b-instruct-q4_K_M",
        "prompt": history,
        "stream": True
        },
        stream=True
    ) as response:
        buf =""
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            tok = data.get("response", "")
            buf += tok

            if any(p in buf for p in [".", "!", "?", ";", ","]):
                 #separates any extra characters from the after splitter
                matches = re.finditer(r"[^.!?,;]+[.!?,;]?", buf)
                parts = [m.group() for m in matches]             
                
                #prints the phrase, adds it to the history and adds the extras back to the buffer
                print(parts[0], end="", flush=True)
                history += parts[0].strip() #
                buf = parts[1] if len(parts) > 1 else ""
                
                #sends the phrase to the tts
                parts[0] += ","
                fileName = "voice line" + ".wav"
                say(parts[0].strip(), fileName)

        #processes and says the last sentece in the response        
        if buf.strip():
            print(buf.strip())
            history += buf.strip()
            fileName = "voice line.wav"
            say(buf.strip(), fileName)

