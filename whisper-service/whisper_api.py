import gc
import logging
import os
import subprocess
import tempfile
import threading
import time

from faster_whisper import WhisperModel, decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps
from flask import Flask, jsonify, request

MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16")
IDLE_TIMEOUT = int(os.getenv("WHISPER_IDLE_TIMEOUT", 30 * 60))
# 1 = greedy decoding (fastest); 5 = beam search, fewer misrecognized words in noisy audio
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", 5))
# Whisper and its VAD work on 16 kHz mono: every upload is converted to it (convert_to_wav)
SAMPLING_RATE = 16000
# The speech intervals for the pause analysis (fluency). Whisper word timestamps stretch the words over the pauses,
# so the pauses are measured by the VAD: every silence from 250 ms (the usual lower bound of a pause), no padding
SPEECH_VAD = VadOptions(min_silence_duration_ms=250, speech_pad_ms=0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("whisper")
# huggingface_hub logs every HTTP request to the Hub on model load
logging.getLogger("httpx").setLevel(logging.WARNING)
# faster_whisper adds two lines to every transcription ("Processing audio with duration", "VAD filter removed"):
# the line "Transcribed ..." below has the length of the audio and the time
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

app = Flask(__name__)

model = None
last_used = time.monotonic()
# Serializes GPU work and guards model load/unload
lock = threading.Lock()


def get_model():
    global model
    if model is None:
        log.info("Loading model %s (%s, %s)...", MODEL_NAME, DEVICE, COMPUTE_TYPE)
        model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
        log.info("Model loaded")
    return model


def unload_model():
    global model
    model = None
    gc.collect()
    log.info("Model unloaded after %d s of inactivity", IDLE_TIMEOUT)


def watchdog():
    """Frees the GPU memory when nobody has transcribed for IDLE_TIMEOUT seconds."""
    while True:
        time.sleep(60)
        with lock:
            if model is not None and time.monotonic() - last_used > IDLE_TIMEOUT:
                unload_model()


threading.Thread(target=watchdog, daemon=True).start()


def convert_to_wav(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", input_path,
        "-ar", str(SAMPLING_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def speech_intervals(wav_path):
    """The intervals of speech in seconds; the gaps between them are the pauses."""
    audio = decode_audio(wav_path, sampling_rate=SAMPLING_RATE)
    return [
        {"start": round(t["start"] / SAMPLING_RATE, 3), "end": round(t["end"] / SAMPLING_RATE, 3)}
        for t in get_speech_timestamps(audio, SPEECH_VAD)
    ]


@app.post("/transcribe")
def transcribe():
    global last_used

    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "no file"}), 400

    language = request.form.get("language")
    if language in ("", "none"):
        language = None
    with_speech = request.form.get("speech") == "true"
    speech = None

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input")
            wav_path = os.path.join(tmp_dir, "audio.wav")
            file.save(input_path)

            try:
                convert_to_wav(input_path, wav_path)
            except subprocess.CalledProcessError as e:
                log.warning("ffmpeg failed for %s: %s", file.filename, e.stderr.strip())
                return jsonify({"error": "cannot decode audio: " + e.stderr.strip()}), 400

            with lock:
                try:
                    whisper = get_model()
                    start = time.monotonic()
                    segments, info = whisper.transcribe(
                        wav_path,
                        language=language,
                        task="transcribe",
                        beam_size=BEAM_SIZE,
                        temperature=0.0,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=700),
                        # with the previous text as the prompt one misrecognized phrase gets repeated in the next segments
                        condition_on_previous_text=False
                    )
                    # segments is a lazy generator: the actual decoding happens here
                    segments = [
                        {"start": s.start, "end": s.end, "text": s.text.strip()}
                        for s in segments
                    ]
                finally:
                    last_used = time.monotonic()

            # the VAD runs on the CPU: no need to hold the GPU lock
            if with_speech:
                speech = speech_intervals(wav_path)

        log.info("Transcribed %s (%s, %.0f s of audio) in %.2f s",
                 file.filename, info.language, info.duration, time.monotonic() - start)

        result = {
            "text": " ".join(s["text"] for s in segments),
            "language": info.language,
            "segments": segments
        }
        if speech is not None:
            result["speech"] = speech
        return jsonify(result)

    except Exception as e:
        log.exception("Transcription failed for %s", file.filename)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
