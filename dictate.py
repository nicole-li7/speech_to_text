import os
import tempfile
import threading
import tkinter as tk
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
from pynput import keyboard

SAMPLE_RATE = 16000
HOLD_THRESHOLD = 0.25   # seconds before a press counts as a hold (recording)
DOUBLE_TAP_WINDOW = 0.4  # seconds between taps to count as double-tap

is_recording = threading.Event()
audio_frames = []
kb = keyboard.Controller()
model = None

_ctrl_held = False
_recording_started = False
_last_tap_time = None
_hold_timer = None


# --- UI setup ---

root = tk.Tk()
root.title("")
root.overrideredirect(True)
root.attributes("-topmost", True)
root.geometry("180x52+60+60")
root.configure(bg="#1a1a2e")

frame = tk.Frame(root, bg="#1a1a2e", padx=10, pady=8)
frame.pack(fill="both", expand=True)

dot = tk.Label(frame, text="●", font=("Segoe UI", 13), bg="#1a1a2e", fg="#888888")
dot.pack(side="left", padx=(0, 6))

status_var = tk.StringVar(value="Loading...")
status_label = tk.Label(frame, textvariable=status_var, font=("Segoe UI", 11, "bold"),
                        bg="#1a1a2e", fg="#888888")
status_label.pack(side="left")

_dx, _dy = 0, 0

def on_drag_start(e):
    global _dx, _dy
    _dx, _dy = e.x, e.y

def on_drag_motion(e):
    root.geometry(f"+{root.winfo_x() + e.x - _dx}+{root.winfo_y() + e.y - _dy}")

root.bind("<ButtonPress-1>", on_drag_start)
root.bind("<B1-Motion>", on_drag_motion)


def set_status(text, color):
    status_var.set(text)
    status_label.config(fg=color)
    dot.config(fg=color)


# --- Audio ---

def audio_callback(indata, frames, time_info, status):
    if is_recording.is_set():
        audio_frames.append(indata.copy())


def transcribe_and_type(audio_data):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    try:
        wav.write(temp_path, SAMPLE_RATE, audio_data)
        result = model.transcribe(temp_path)
        text = result["text"].strip()
        if text:
            kb.type(text + " ")
            print(f"Typed: {text}")
        else:
            print("(nothing transcribed)")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        os.unlink(temp_path)
        root.after(0, lambda: set_status("Ready", "#00e676"))


# --- Keyboard ---

def begin_recording():
    global _recording_started
    if _ctrl_held:
        _recording_started = True
        audio_frames.clear()
        is_recording.set()
        root.after(0, lambda: set_status("Recording", "#ff1744"))


def on_press(key):
    global _ctrl_held, _hold_timer
    if key == keyboard.Key.ctrl_r and not _ctrl_held and not _recording_started:
        _ctrl_held = True
        _hold_timer = threading.Timer(HOLD_THRESHOLD, begin_recording)
        _hold_timer.start()


def on_release(key):
    global _ctrl_held, _recording_started, _last_tap_time, _hold_timer
    if key == keyboard.Key.ctrl_r:
        _ctrl_held = False

        if _recording_started:
            # was a hold — stop recording and transcribe
            _recording_started = False
            is_recording.clear()
            root.after(0, lambda: set_status("Processing...", "#ffab00"))
            if audio_frames:
                audio_data = np.concatenate(audio_frames, axis=0)
                threading.Thread(target=transcribe_and_type, args=(audio_data,), daemon=True).start()
            else:
                root.after(0, lambda: set_status("Ready", "#00e676"))
        else:
            # was a quick tap — cancel hold timer and check for double-tap
            if _hold_timer:
                _hold_timer.cancel()
                _hold_timer = None
            now = time.time()
            if _last_tap_time and (now - _last_tap_time) < DOUBLE_TAP_WINDOW:
                _last_tap_time = None
                kb.press(keyboard.Key.enter)
                kb.release(keyboard.Key.enter)
                print("Enter")
            else:
                _last_tap_time = now

    elif key == keyboard.Key.esc:
        root.after(0, root.destroy)


# --- Model loading ---

def load_model():
    global model
    model = whisper.load_model("base")
    root.after(0, lambda: set_status("Ready", "#00e676"))
    start_listeners()

def start_listeners():
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=audio_callback)
    stream.start()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

threading.Thread(target=load_model, daemon=True).start()

root.mainloop()
