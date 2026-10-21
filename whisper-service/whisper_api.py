import bisect
import ctypes
import gc
import json
import logging
import mmap
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps
from flask import Flask, Response, jsonify, request
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

# The models: (repository of the Hugging Face Hub, revision, folder of the CTranslate2 model in it or None). Pinned to a
# revision: a model is downloaded only when the cache does not have it (a new model or a new revision set here), and
# a load asks the Hub nothing (faster-whisper asks it at every load of a model given by its name).
# The main model: every language without a model of its own, and the detection of the language
MAIN = "main"
MAIN_MODEL = ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478", None)
# Languages with a model of their own. Estonian: large-v3-turbo fine-tuned by TalTech on 1400 h of verbatim Estonian
# transcripts. Measured on 2026-10-12 with the options of /transcribe (FLEURS et test): 5.7% WER instead of 18.2% on
# the phrases without numbers (large-v3 splits compound words and garbles endings), 3.5 times faster, 2 GB of VRAM
# instead of 3.6. It writes numbers as words ("kahe tuhande üheteistkümnendal") and its English is worse than
# large-v3's, so only Estonian goes to it
LANGUAGE_MODELS = {
    "et": ("TalTechNLP/whisper-large-v3-turbo-et-verbatim-2604", "310c550922fc1d1bebf3fad41f9b33ede6652bdc", "ct2"),
}
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
# float16 measured on the RTX 5060 Ti (2026-10-08): 19-24% faster than int8_float16 with the same error rate
# (Estonian slightly better); the model takes ~3.6 GB of VRAM instead of ~2 GB
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
IDLE_TIMEOUT = int(os.getenv("WHISPER_IDLE_TIMEOUT", 30 * 60))
# 1 = greedy decoding (fastest); 5 = beam search, fewer misrecognized words in noisy audio
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", 5))
# Whisper and its VAD work on 16 kHz mono: every upload is converted to it (convert_to_wav)
SAMPLING_RATE = 16000
# The speech intervals for the pause analysis (fluency). Whisper word timestamps stretch the words over the pauses,
# so the pauses are measured by the VAD: every silence from 250 ms (the usual lower bound of a pause), no padding
SPEECH_VAD = VadOptions(min_silence_duration_ms=250, speech_pad_ms=0)
# the VAD filter of the decoding of a recording: the silences from 0.7 s are cut out
VOICE_VAD = VadOptions(min_silence_duration_ms=700)
# A recording up to one window of the model is decoded without timestamp tokens: the callers use only its text, and it
# is 3-10% faster. A longer one keeps them: without them the decoding moves in fixed 30 s windows and may cut a word
SHORT_AUDIO_SECONDS = 30

# The demuxers ffmpeg may use: audio containers only. A playlist (hls) or a concat list would make ffmpeg open other
# files or addresses of the Docker network; with the file protocol only, no address is opened anyway
ALLOWED_FORMATS = "mp3,wav,ogg,flac,mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,aac,amr,asf,aiff,caf,w64,wv"
INPUT_GUARD = ["-protocol_whitelist", "file", "-format_whitelist", ALLOWED_FORMATS]
CONVERT_TIMEOUT = 15 * 60

# Files (/transcribe-file): the longest audio accepted when the caller sets no lower limit
MAX_FILE_SECONDS = int(os.getenv("WHISPER_MAX_FILE_SECONDS", 4 * 3600))
# The speech is cut into chunks of one window of the model and the chunks are decoded in batches. Measured with
# float16 on an hour of speech: batch 8 decodes it 6 times faster than the sequential way and takes 3.3 GB of VRAM
# over the loaded model, batch 4 - 1.7 GB. The batch is chosen by the free VRAM (Ollama shares the GPU).
# Without timestamp tokens: with them the batched decoding made a third more errors on Russian (12.2% instead of
# 8.8% on an hour of a book). The time of the text comes from the word timestamps, aligned after the decoding
CHUNK_SECONDS = 30
BATCH_FREE_VRAM_MB = [(8, 4000), (4, 2200), (2, 1400)]
# the VAD of BatchedInferencePipeline; speech separated by a longer silence (music, a break) starts a new chunk
FILE_VAD = VadOptions(max_speech_duration_s=CHUNK_SECONDS, min_silence_duration_ms=160)
CHUNK_MAX_GAP_SECONDS = 5
# In batched decoding every chunk is decoded once, and the model sometimes stops early and skips a sentence (seen:
# 37-50 words, 1-2 times an hour). Such a chunk is decoded again the sequential way, which goes on after an early
# stop: a chunk with a stretch of speech this long without a word ...
REPAIR_GAP_SECONDS = 3
# ... or with fewer words per second of speech than this share of the median
REPAIR_RATE = 0.6
REPAIR_MIN_SPEECH_SECONDS = 5
REPAIR_MAX_SHARE = 0.25
# The subtitles (segments of the result) are made of the words: a cue ends with a sentence, a pause or these limits
CUE_MAX_SECONDS = 6
CUE_MAX_CHARS = 84
CUE_PAUSE_SECONDS = 1.0
INITIAL = re.compile(r"^\w\.$")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("whisper")
# huggingface_hub logs every HTTP request to the Hub on model load
logging.getLogger("httpx").setLevel(logging.WARNING)
# faster_whisper adds two lines to every transcription ("Processing audio with duration", "VAD filter removed"):
# the line "Transcribed ..." below has the length of the audio and the time
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

app = Flask(__name__)

# the loaded models by key: MAIN or a language of LANGUAGE_MODELS
models = {}
# when each model was used last, and the requests that use it now: a model is not unloaded under them
last_used = {}
active = Counter()
warming_up = False
# Guards loading, unloading and the counters. The transcriptions themselves run in parallel: CTranslate2 runs the work
# of the threads on the GPU one batch after another, so a short recording waits for a batch of a long file, not for
# the whole file (measured: 2.7-3.4 s instead of 1.7 s for 10 s of speech)
lock = threading.Lock()

# A load reads the whole model file into memory (large-v3 2.9 GB, the Estonian model 1.5 GB) and frees it once the model
# is on the GPU. glibc keeps what a thread frees in the arena of that thread, and the loads run in different request
# threads: every one kept a copy of the file, the process grew by 2.9 GB a load up to 15.6 GB of its 16 GB limit
# (2026-10-11) and held 7.7 GB with no model loaded. malloc_trim gives the free memory of all arenas back to the system:
# 8 loads and unloads in 4 threads stay at 0.8 GB instead of 6.4 GB (the Estonian model, measured 2026-10-20)
libc = ctypes.CDLL("libc.so.6")

# A load after a while without one took 10-25 s for large-v3 instead of 1-2.5 s. Windows takes away the memory of the
# WSL VM that nobody touched, even with plenty of its own free (seen: 12.3 -> 8.5 GB of the 68 GB of the VM within
# 40 idle minutes): the model file stays in the page cache, but every 4 KB page the load touches first - of the file,
# of the fresh memory for the weights - comes back by a fault of the host, and CTranslate2 loads in one thread.
# Touched from several threads the pages come back 5-8 times faster (the cached file 4.6 GB/s instead of 0.8, measured
# 2026-10-20), so the file is first read that way into as much fresh memory: freed at once, that memory is handed out
# to the load first. It costs ~0.4 s for large-v3 when nothing was taken away
PREHEAT_THREADS = 8
PREHEAT_PIECE = 64 << 20


def model_entry(key):
    return MAIN_MODEL if key == MAIN else LANGUAGE_MODELS[key]


def model_label(key):
    return model_entry(key)[0].split("/")[-1]


def model_path(key):
    """The folder of a model in the Hugging Face cache, downloaded on its first use."""
    repo, revision, folder = model_entry(key)
    files = [f"{folder}/*"] if folder else None
    try:
        path = snapshot_download(repo, revision=revision, allow_patterns=files, local_files_only=True)
    except LocalEntryNotFoundError:
        log.info("Downloading model %s...", repo)
        path = snapshot_download(repo, revision=revision, allow_patterns=files)
    return os.path.join(path, folder) if folder else path


def preheat(folder):
    """Reads the model file in parallel pieces into fresh memory and frees it, see PREHEAT_THREADS."""
    path = os.path.join(folder, "model.bin")
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        # private anonymous memory, like that of the weights
        memory = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE)
        try:
            with memoryview(memory) as view, ThreadPoolExecutor(PREHEAT_THREADS) as pool:
                list(pool.map(lambda offset: os.preadv(fd, [view[offset:offset + PREHEAT_PIECE]], offset),
                              range(0, size, PREHEAT_PIECE)))
        finally:
            memory.close()
    finally:
        os.close(fd)


def get_model(key=MAIN):
    """Under the lock."""
    if key not in models:
        log.info("Loading model %s (%s, %s)...", model_label(key), DEVICE, COMPUTE_TYPE)
        path = model_path(key)
        start = time.monotonic()
        try:
            preheat(path)
        except (OSError, ValueError) as e:
            # only slower without it
            log.warning("Preheating %s failed: %s", path, e)
        preheated = time.monotonic() - start
        models[key] = WhisperModel(path, device=DEVICE, compute_type=COMPUTE_TYPE)
        # the file read for the load is freed: see libc
        libc.malloc_trim(0)
        last_used[key] = time.monotonic()
        log.info("Model loaded in %.1f s (preheating %.1f s)", last_used[key] - start, preheated)
    return models[key]


def unload_model(key):
    del models[key]
    gc.collect()
    libc.malloc_trim(0)
    log.info("Model %s unloaded after %d s of inactivity", model_label(key), IDLE_TIMEOUT)


@contextmanager
def using_model(key=MAIN):
    """The model, loaded if needed; the idle time counts from the end of the use."""
    with lock:
        whisper = get_model(key)
        active[key] += 1
    try:
        yield whisper
    finally:
        with lock:
            active[key] -= 1
            last_used[key] = time.monotonic()


def watchdog():
    """Frees the GPU memory of a model nobody has used for IDLE_TIMEOUT seconds."""
    while True:
        time.sleep(60)
        with lock:
            for key in list(models):
                if active[key] == 0 and time.monotonic() - last_used[key] > IDLE_TIMEOUT:
                    unload_model(key)


threading.Thread(target=watchdog, daemon=True).start()


def model_for(language, audio, vad_filter):
    """The model to decode the audio with and the language to decode it in. Without a language the main model detects
    it from the first 30 s of speech, as its transcribe() would; a language with a model of its own goes to that one."""
    if language is None:
        with using_model() as whisper:
            language, _, _ = whisper.detect_language(audio, vad_filter=vad_filter, vad_parameters=VOICE_VAD)
    return (language if language in LANGUAGE_MODELS else MAIN), language


def convert_to_wav(input_path, output_path, max_seconds=None):
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *INPUT_GUARD, "-i", input_path]
    if max_seconds:
        command += ["-t", str(max_seconds)]
    command += ["-ar", str(SAMPLING_RATE), "-ac", "1", "-c:a", "pcm_s16le", output_path]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                   timeout=CONVERT_TIMEOUT)


def speech_intervals(audio):
    """The intervals of speech in seconds; the gaps between them are the pauses."""
    return [
        {"start": round(t["start"] / SAMPLING_RATE, 3), "end": round(t["end"] / SAMPLING_RATE, 3)}
        for t in get_speech_timestamps(audio, SPEECH_VAD)
    ]


@app.post("/warmup")
def warmup():
    """Loads a model ahead of a recording (the user pressed the microphone): the load takes 1-2.5 s, longer after
    a while without one (see PREHEAT_THREADS).
    Query: language (optional) - the speech to come; a language with a model of its own loads that one, any other
    the main model. Answers at once; the idle unload goes on as before, counting from now."""
    global warming_up
    language = request.args.get("language")
    key = language if language in LANGUAGE_MODELS else MAIN
    # the lock is held for the whole load: a warm-up during a load (its own or a transcription's) does not wait for it
    if warming_up or not lock.acquire(timeout=1):
        return jsonify({"loaded": False})
    try:
        loaded = key in models
        if loaded:
            last_used[key] = time.monotonic()
        else:
            warming_up = True
            threading.Thread(target=load_ahead, args=(key,), daemon=True).start()
    finally:
        lock.release()
    return jsonify({"loaded": loaded})


def load_ahead(key):
    global warming_up
    try:
        with lock:
            get_model(key)
    except Exception:
        log.exception("Warm-up failed")
    finally:
        warming_up = False


@app.post("/transcribe")
def transcribe():
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
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                log.warning("ffmpeg failed for %s: %s", file.filename, (e.stderr or "timeout").strip())
                return jsonify({"error": "cannot decode audio"}), 400

            audio = decode_audio(wav_path, sampling_rate=SAMPLING_RATE)
            key, language = model_for(language, audio, vad_filter=True)
            with using_model(key) as whisper:
                start = time.monotonic()
                segments, info = whisper.transcribe(
                    audio,
                    language=language,
                    task="transcribe",
                    beam_size=BEAM_SIZE,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters=VOICE_VAD,
                    # with the previous text as the prompt one misrecognized phrase gets repeated in the next segments
                    condition_on_previous_text=False,
                    without_timestamps=len(audio) / SAMPLING_RATE <= SHORT_AUDIO_SECONDS
                )
                # segments is a lazy generator: the actual decoding happens here
                segments = [
                    {"start": s.start, "end": s.end, "text": s.text.strip()}
                    for s in segments
                ]

            # the VAD runs on the CPU, after the model is released
            if with_speech:
                speech = speech_intervals(audio)

        log.info("Transcribed %s (%s, %s, %.0f s of audio) in %.2f s",
                 file.filename, info.language, model_label(key), info.duration, time.monotonic() - start)

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


@app.post("/transcribe-file")
def transcribe_file():
    """A long recording uploaded as a file: decoded in batches, which is several times faster than /transcribe.

    Form: file, language (empty or none - detected), max_seconds (the longest audio accepted).
    Before the work starts, a file that cannot be taken is refused with a status and {"error", "reason"}: reason
    "decode" (400, not an audio file ffmpeg can read), "video" (415, has a video track), "too_long" (413, with
    "seconds"). Then the answer is a stream of JSON lines (application/x-ndjson):
    {"stage": "converting"}, {"stage": "recognizing", "progress": 0.0-1.0}, {"stage": "checking"} and at the end
    {"result": {"text", "language", "duration", "segments": [{"start", "end", "text"}]}} or {"error", "reason"};
    the segments are subtitles (up to CUE_MAX_SECONDS) made of the word timestamps.
    Closing the connection stops the work after the batch being decoded.
    """
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "no file", "reason": "decode"}), 400

    language = request.form.get("language")
    if language in ("", "none"):
        language = None
    try:
        max_seconds = min(float(request.form.get("max_seconds", MAX_FILE_SECONDS)), MAX_FILE_SECONDS)
    except ValueError:
        return jsonify({"error": "max_seconds is not a number"}), 400
    name = file.filename

    tmp_dir = tempfile.mkdtemp()
    try:
        input_path = os.path.join(tmp_dir, "input")
        file.save(input_path)
        problem = probe(input_path, max_seconds)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    if problem:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        status, body = problem
        log.info("File %s refused: %s", name, body)
        return jsonify(body), status

    response = Response(cancellable(file_events(tmp_dir, input_path, language, max_seconds, name), name),
                        mimetype="application/x-ndjson")
    response.call_on_close(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))
    return response


def probe(path, max_seconds):
    """None for a file to transcribe, otherwise (status, body) of the refusal."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", *INPUT_GUARD, "-show_entries",
             "format=duration:stream=codec_type:stream_disposition=attached_pic", "-of", "json", path],
            capture_output=True, text=True, timeout=60)
        info = json.loads(result.stdout or "{}") if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        info = None
    if info is None:
        return 400, {"error": "cannot decode audio", "reason": "decode"}

    streams = info.get("streams", [])
    # the cover picture of an MP3 or M4A is a video stream too
    if any(s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic") for s in streams):
        return 415, {"error": "the file has a video track", "reason": "video"}
    if not any(s.get("codec_type") == "audio" for s in streams):
        return 400, {"error": "no audio track", "reason": "decode"}
    try:
        seconds = float(info.get("format", {}).get("duration") or 0)
    except ValueError:
        seconds = 0
    if seconds > max_seconds:
        return 413, {"error": "the audio is too long", "reason": "too_long", "seconds": round(seconds)}
    return None


def event(**fields):
    return json.dumps(fields, ensure_ascii=False) + "\n"


def cancellable(events, name):
    """The lines of file_events; the client closing the connection (a cancelled job) ends them at the next line."""
    started = time.monotonic()
    try:
        yield from events
    except GeneratorExit:
        log.info("File %s: the client went away after %.1f s, stopped", name, time.monotonic() - started)
        raise


def file_events(tmp_dir, input_path, language, max_seconds, name):
    started = time.monotonic()
    wav_path = os.path.join(tmp_dir, "audio.wav")
    yield event(stage="converting")
    try:
        # a second over the limit tells a file that is too long from one that just fits
        convert_to_wav(input_path, wav_path, max_seconds + 1)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning("ffmpeg failed for %s: %s", name, (e.stderr or "timeout").strip())
        yield event(error="cannot decode audio", reason="decode")
        return
    os.remove(input_path)
    audio = decode_audio(wav_path, sampling_rate=SAMPLING_RATE)
    duration = len(audio) / SAMPLING_RATE
    if duration > max_seconds:
        yield event(error="the audio is too long", reason="too_long", seconds=round(duration))
        return

    chunks = speech_chunks(audio)
    segments, detected, batch_size, repaired, key = [], language, 0, 0, MAIN
    if chunks:
        # the language is detected on the first chunk of speech
        first = chunks[0]
        key, language = model_for(
            language, audio[int(first["start"] * SAMPLING_RATE):int(first["end"] * SAMPLING_RATE)], vad_filter=False)
        yield event(stage="recognizing", progress=0)
        with using_model(key) as whisper:
            for batch_size in batch_sizes():
                try:
                    segments = []
                    generated, info = BatchedInferencePipeline(model=whisper).transcribe(
                        audio, language=language, task="transcribe", beam_size=BEAM_SIZE, temperature=0.0,
                        batch_size=batch_size, vad_filter=False,
                        clip_timestamps=[{"start": c["start"], "end": c["end"]} for c in chunks],
                        without_timestamps=True, word_timestamps=True, condition_on_previous_text=False)
                    reported = 0
                    for s in generated:
                        segments.append(segment(s))
                        progress = min(s.end / duration, 1)
                        if progress - reported >= 0.01:
                            reported = progress
                            yield event(stage="recognizing", progress=round(progress, 3))
                    detected = info.language
                    break
                except RuntimeError as e:
                    if "out of memory" not in str(e) or batch_size == 1:
                        raise
                    log.warning("Out of GPU memory with batch %d for %s: trying batch 1", batch_size, name)
                    yield event(stage="recognizing", progress=0)
            yield event(stage="checking")
            segments, repaired = repair(whisper, audio, chunks, segments, detected)

    log.info("Transcribed file %s (%s, %s, %.0f s of audio, %d chunks, batch %d, %d repaired) in %.1f s",
             name, detected, model_label(key), duration, len(chunks), batch_size, repaired, time.monotonic() - started)
    yield event(result={
        "text": " ".join(s["text"] for s in segments if s["text"]),
        "language": detected,
        "duration": round(duration, 1),
        "segments": cues(segments)
    })


def segment(s, offset=0.0):
    """A segment of faster-whisper with its words, the times moved by the offset."""
    return {"start": offset + s.start, "end": offset + s.end, "text": s.text.strip(),
            "words": [{"start": offset + w.start, "end": offset + w.end, "word": w.word} for w in s.words or []]}


def cues(segments):
    """The subtitles: the words grouped by sentences, pauses and the limits of a cue."""
    result, current = [], []

    def close():
        if current:
            result.append({"start": round(current[0]["start"], 2), "end": round(current[-1]["end"], 2),
                           "text": "".join(w["word"] for w in current).strip()})
            current.clear()

    for s in segments:
        if not s["words"]:
            close()
            if s["text"]:
                result.append({"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]})
            continue
        for w in s["words"]:
            if current and (w["start"] - current[-1]["end"] > CUE_PAUSE_SECONDS
                            or w["end"] - current[0]["start"] > CUE_MAX_SECONDS
                            or sum(len(c["word"]) for c in current) + len(w["word"]) > CUE_MAX_CHARS):
                close()
            current.append(w)
            # an initial ("L. T. Meade") does not end a sentence
            if w["word"].rstrip().endswith((".", "!", "?", "…")) and not INITIAL.match(w["word"].strip()):
                close()
        close()
    return result


def speech_chunks(audio):
    """The speech cut into chunks of up to CHUNK_SECONDS (seconds, with the short silences inside) and the seconds of
    speech in each chunk."""
    chunks = []
    for t in get_speech_timestamps(audio, FILE_VAD):
        start, end = t["start"] / SAMPLING_RATE, t["end"] / SAMPLING_RATE
        last = chunks[-1] if chunks else None
        if last and end - last["start"] <= CHUNK_SECONDS and start - last["end"] <= CHUNK_MAX_GAP_SECONDS:
            last["end"] = end
            last["speech"] += end - start
            last["pieces"].append((start, end))
        else:
            end = min(end, start + CHUNK_SECONDS)
            chunks.append({"start": start, "end": end, "speech": end - start, "pieces": [(start, end)]})
    return chunks


def batch_sizes():
    """The batch the free VRAM allows, then 1 as the fallback when the GPU runs out of memory anyway."""
    size = 4
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        free = int(out.strip().splitlines()[0])
        size = next((batch for batch, needed in BATCH_FREE_VRAM_MB if free >= needed), 1)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        log.warning("Cannot read the free GPU memory: batch %d", size)
    return [size] if size == 1 else [size, 1]


def words(text):
    return len(re.findall(r"\w+", text))


def chunk_of(starts, segment):
    """The chunk of a segment, by its middle: its start may come a little before the start of its chunk."""
    return max(bisect.bisect_right(starts, (segment["start"] + segment["end"]) / 2) - 1, 0)


def longest_uncovered(pieces, spans):
    """The longest stretch of speech (pieces of the VAD) that no span (a word) covers, in seconds."""
    longest = 0
    spans = sorted(spans, key=lambda s: s["start"])
    for start, end in pieces:
        position = start
        for s in spans:
            if s["end"] <= position or s["start"] >= end:
                continue
            longest = max(longest, s["start"] - position)
            position = max(position, s["end"])
        longest = max(longest, end - position)
    return longest


def repair(whisper, audio, chunks, segments, language):
    """Decodes again, the sequential way, the chunks that lost words: a stretch of speech without a word (the model
    skipped a sentence), or too few words for their speech."""
    starts = [c["start"] for c in chunks]
    by_chunk = [[] for _ in chunks]
    for s in segments:
        by_chunk[chunk_of(starts, s)].append(s)
    counts = [sum(words(s["text"]) for s in chunk_segments) for chunk_segments in by_chunk]
    rates = [counts[i] / c["speech"] for i, c in enumerate(chunks) if c["speech"] >= REPAIR_MIN_SPEECH_SECONDS]
    median = statistics.median(rates) if rates else 0
    suspicious = []
    for i, chunk in enumerate(chunks):
        rate = counts[i] / chunk["speech"] if chunk["speech"] else 0
        gap = longest_uncovered(chunk["pieces"], [w for s in by_chunk[i] for w in s["words"]])
        if gap >= REPAIR_GAP_SECONDS or (chunk["speech"] >= REPAIR_MIN_SPEECH_SECONDS and rate < REPAIR_RATE * median):
            suspicious.append((-gap, rate, i))
    # the worst first: the longest stretch without text, then the fewest words
    suspicious = sorted(suspicious)[:max(1, int(len(chunks) * REPAIR_MAX_SHARE))]

    replaced = {}
    for _, _, i in suspicious:
        chunk = chunks[i]
        piece = audio[int(chunk["start"] * SAMPLING_RATE):int(chunk["end"] * SAMPLING_RATE)]
        again, _ = whisper.transcribe(piece, language=language, task="transcribe", beam_size=BEAM_SIZE,
                                      temperature=0.0, vad_filter=True, vad_parameters=VOICE_VAD,
                                      condition_on_previous_text=False, word_timestamps=True)
        again = [segment(s, chunk["start"]) for s in again]
        if sum(words(s["text"]) for s in again) > counts[i]:
            replaced[i] = again
    log.info("Repair: %d suspicious chunks of %d (gaps %s s), %d decoded with more words", len(suspicious),
             len(chunks), [round(-gap, 1) for gap, _, _ in suspicious], len(replaced))
    if not replaced:
        return segments, 0

    result = []
    for s in segments:
        i = chunk_of(starts, s)
        if i not in replaced:
            result.append(s)
        elif replaced[i] is not None:
            # the new segments of the chunk take the place of its first old segment
            result.extend(replaced[i])
            replaced[i] = None
    # a chunk that had no segments at all
    for i, again in replaced.items():
        if again:
            result.extend(again)
    result.sort(key=lambda s: s["start"])
    return result, len(replaced)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
