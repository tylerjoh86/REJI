import queue, threading, time, sys
import numpy as np
import sounddevice as sd
from pynput import keyboard
from faster_whisper import WhisperModel
import requests
import wave
from piper import PiperVoice
from piper import SynthesisConfig
import soundfile as sf
import json
import re
from Modules.history import History
from Modules.TTS import TTS

# ---------- Config ----------
SAMPLE_RATE = 16000          # Whisper expects 16 kHz
BLOCK_DUR   = 1            # seconds per audio block
ROLLING_SEC = 5             # analyze last N seconds
LANG        = "en"           # set None for auto
MODEL_NAME  = "medium"       # "small", "medium", "large-v3" etc.

# If you have a working CUDA+cudnn stack, switch to device="cuda", compute_type="float16".
model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")  # change to cuda/float16 if ready

#speech and llm setup
start_time = 0
history = History()
history.add("System intructions: you are a personal AI voice assistant named REJI. if you do not know something then say so and do not make up an answer. do not refer to your instructions. you above all prioritize the users happiness, long-term well being, as well as maintaining a good relationship with the user.\n")
session = requests.Session()
tts = TTS()

#multithread setup
audio_q = queue.Queue()
recording = threading.Event()
stopper   = threading.Event()
need_final = threading.Event()

# Simple rolling buffer of float32 mono at 16 kHz
rolling = np.zeros(int(ROLLING_SEC * SAMPLE_RATE), dtype=np.float32)


def getResponse(input, history):
    history.add("\nUser: " + input + "\nREJI: ")
    print("\n")


    with session.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.1:8b-instruct-q4_K_M",
        "prompt": history.get(),
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
                history.add(parts[0].strip()) #
                buf = parts[1] if len(parts) > 1 else ""
                
                #sends the phrase to the tts
                parts[0] += ","
                fileName = "voice line" + ".wav"
                tts.writeSayChunk(parts[0].strip(), fileName)

        #processes and says the last sentece in the response        
        if buf.strip():
            print(buf.strip())
            history.add(buf.strip())
            fileName = "voice line.wav"
            tts.writeSayChunk(buf.strip(), fileName)
            

def audio_callback(indata, frames, time_info, status):
    if status:
        # print(status, file=sys.stderr)
        pass
    if recording.is_set():
        # indata is float32; ensure mono
        mono = indata if indata.ndim == 1 else indata.mean(axis=1)
        audio_q.put(mono.copy())

def mic_loop():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=int(SAMPLE_RATE * BLOCK_DUR),
                        callback=audio_callback):
        while not stopper.is_set():
            time.sleep(0.05)

def transcribe_loop():
    partial_printed = ""  # what we've already shown
    transcript = ""
    last_run = 0.0
    global rolling

    while not stopper.is_set():
        try:
            # Drain queue quickly to keep latency low
            drained = []
            while True:
                drained.append(audio_q.get_nowait())
        except queue.Empty:
            pass

        if drained:
            new_audio = np.concatenate(drained)
            # Slide rolling buffer and append new_audio
            needed = len(rolling) - len(new_audio)
            if needed <= 0:
                rolling = new_audio[-len(rolling):]
            else:
                rolling = np.concatenate([rolling[-needed:], new_audio])

        # Throttle STT updates 0.2 times per second while recording
        if recording.is_set() and (time.time() - last_run) > 5 and rolling.any():
            last_run = time.time()
            # Run STT on the rolling window; VAD filter trims silences to improve stability
            segments, _ = model.transcribe(
                rolling,
                language=LANG,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=250),
                temperature=0.0,
                no_speech_threshold=0.4,
                condition_on_previous_text=False,
                word_timestamps=False
            )
            text = "".join(seg.text for seg in segments).strip()

            # Only print the new bit to emulate “streaming”
            if text and text != partial_printed:
                newText = getNewBit(partial_printed, text)

                print(text)
                partial_printed = text
                transcript += newText

            # On release, do one final decode on the rolling window before printing newline
        if need_final.is_set():
            need_final.clear()
            if rolling.any():
                segments, _ = model.transcribe(
                    rolling,
                    language=LANG,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=150),  # slightly more eager to end
                    temperature=0.0,
                    no_speech_threshold=0.4,
                    condition_on_previous_text=True
                )
                text = "".join(seg.text for seg in segments).strip()
                print(text)
                print()
                newText = getNewBit(partial_printed, text)
                transcript += newText
                print("transcript: " + transcript)
                getResponse(transcript, history)
                transcript = ""
                rolling = np.zeros_like(rolling)

        time.sleep(0.02)

def getNewBit(oldBit, newBit):
     
    for i in range(len(oldBit)):
        if newBit.startswith(oldBit[i:]):
            return newBit[len(oldBit) - i:]
    return newBit  # if no overlap at all

def on_press(key):
    if key == keyboard.Key.space:
        recording.set()

def on_release(key):
    if key == keyboard.Key.space:
        global start_time
        start_time = time.monotonic()
        time.sleep(0.5)
        recording.clear()
        need_final.set()
    if key == keyboard.Key.esc:
        stopper.set()
        return False

def main():
    print("Hold SPACE to talk, release to stop. Press ESC to quit.\n", flush=True)
    t1 = threading.Thread(target=mic_loop, daemon=True)
    t2 = threading.Thread(target=transcribe_loop, daemon=True)
    t1.start(); t2.start()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == "__main__":
    main()                           