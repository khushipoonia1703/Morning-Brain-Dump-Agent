"""Stage 1 — capture and transcribe.

Records from the default microphone until Enter is pressed, saves the audio
to recordings/, transcribes it with Whisper, and saves the text to transcripts/.

    python record.py                      # record, then transcribe
    python record.py recordings/dump_*.wav  # skip recording, transcribe an existing file

This is a workflow step, not an agent. The code decides every step.
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

SAMPLE_RATE = 16000
CHANNELS = 1
TRANSCRIBE_MODEL = "whisper-1"

RECORDINGS_DIR = Path("recordings")
TRANSCRIPTS_DIR = Path("transcripts")


def check_microphone():
    """Exit cleanly if there is no usable input device."""
    try:
        sd.check_input_settings(channels=CHANNELS, samplerate=SAMPLE_RATE)
    except Exception as e:
        print(f"No usable microphone found: {e}")
        print("Plug in a mic (or check your OS sound settings) and try again.")
        sys.exit(1)


def record_until_enter():
    """Record from the default mic and return the audio as one numpy array."""
    blocks = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"\n  audio warning: {status}", file=sys.stderr)
        blocks.append(indata.copy())

    stop = threading.Event()

    def wait_for_enter():
        input()
        stop.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()

    print("Recording. Press Enter to stop.")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=callback,
    ):
        started = time.time()
        while not stop.is_set():
            print(f"\r  {time.time() - started:5.1f}s", end="", flush=True)
            time.sleep(0.1)
        elapsed = time.time() - started

    print(f"\r  {elapsed:5.1f}s — stopped.")

    if not blocks:
        print("No audio was captured. Nothing to save.")
        sys.exit(1)

    return np.concatenate(blocks, axis=0)


def save_wav(audio, stamp):
    """Write the audio to recordings/dump_<stamp>.wav and return the path."""
    RECORDINGS_DIR.mkdir(exist_ok=True)
    path = RECORDINGS_DIR / f"dump_{stamp}.wav"
    sf.write(path, audio, SAMPLE_RATE)
    print(f"Saved audio: {path}")
    return path


def transcribe(wav_path):
    """Send the wav to Whisper and return the transcript text."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Add it to .env and try again.")
        print(f"Your recording is safe at: {wav_path}")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Transcribing with {TRANSCRIBE_MODEL} ...")
    try:
        with open(wav_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL,
                file=f,
            )
    except Exception as e:
        # Never lose a recording because a network call failed.
        print(f"Transcription failed: {e}")
        print(f"Your recording is safe at: {wav_path}")
        print(f"Retry just the transcription with:  python record.py {wav_path}")
        sys.exit(1)

    return response.text


def save_transcript(text, stamp):
    """Write the transcript to transcripts/dump_<stamp>.txt and return the path."""
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    path = TRANSCRIPTS_DIR / f"dump_{stamp}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def stamp_from_wav(wav_path):
    """Reuse the timestamp in dump_YYYY-MM-DD_HHMM.wav so both files match."""
    name = Path(wav_path).stem
    return name[len("dump_"):] if name.startswith("dump_") else name


def main():
    if len(sys.argv) > 1:
        # Transcribe a wav that already exists — the recovery path after a
        # failed API call.
        wav_path = Path(sys.argv[1])
        if not wav_path.exists():
            print(f"No such file: {wav_path}")
            sys.exit(1)
        stamp = stamp_from_wav(wav_path)
    else:
        check_microphone()
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        audio = record_until_enter()
        wav_path = save_wav(audio, stamp)

    text = transcribe(wav_path)
    transcript_path = save_transcript(text, stamp)

    print(f"Saved transcript: {transcript_path}\n")
    print(text)


if __name__ == "__main__":
    main()










































