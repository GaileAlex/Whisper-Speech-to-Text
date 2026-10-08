# Whisper Speech-to-Text

Speech recognition for [gaile.ee](https://gaile.ee/) (project [CV](https://github.com/GaileAlex/cv)): the voice
messages of the chat, their fluency analysis and the audio files of the speech-to-text page (up to hours long).
An audio file goes in, the text with timestamps comes out.
Runs on the GPU of the host with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and the Whisper
`large-v3` model.

## 🏗 Architecture

```
cv-app ──► cv-whisper-app (Spring Boot, :8389) ──► cv-whisper (Python, :5001) ──► GPU
```

- **cv-whisper-app** (`src/`) - the API for CV: takes the upload (up to 200 MB, streamed to disk), maps the language
  tag (`et-EE` → `et`), passes the file on and the errors of the Python service back (the progress of a long file
  as a stream); metrics and the traceId of CV.
- **cv-whisper** (`whisper-service/`) - Flask + gunicorn: ffmpeg converts the input (audio containers only) to 16 kHz
  mono WAV, faster-whisper transcribes it, Silero VAD finds the intervals of speech. The model is loaded on the first
  request (or ahead of it by `/warmup`) and unloaded after 30 minutes without requests (the GPU memory is free for
  the other services in between). The requests run in parallel on the one model: CTranslate2 takes their work batch
  by batch, so a voice message waits a couple of seconds behind a long file, not the whole file.
- A long file is cut into 30 s chunks of speech decoded in batches (6 times faster than the sequential decoding of
  a voice message); chunks that lost a sentence are found by the word timestamps and decoded again the sequential way.

## 🛠 Technologies

- **Java 25** with **Spring Boot 4.1** - Spring MVC on virtual threads, RestClient (JDK HttpClient), Actuator,
  Micrometer Tracing
- **Python 3.12** - faster-whisper 1.2 (CTranslate2 4.8), Flask, gunicorn
- **NVIDIA CUDA 12.9 + cuDNN 9**, model `large-v3` in `float16`
- **ffmpeg** - decoding of the audio formats
- **Docker Compose** - both services on the network `kokoro-net` of CV

## 📋 Prerequisites

- **NVIDIA GPU** (Blackwell needs CUDA 12.8+) and a driver for CUDA 12.9; Docker with GPU support
  (Docker Desktop on WSL 2 or the NVIDIA Container Toolkit)
- The Docker network `kokoro-net`: CV creates it, so CV goes up first (or `docker network create kokoro-net`)
- About 4 GB of GPU memory while the model is loaded, 3-4 GB more while a long file is decoded in a batch of 8
  (the batch is smaller when less is free); 3 GB of disk for the model in the Hugging Face cache

Development: **Java 25**; Maven comes with the wrapper (`mvnw`).

## 🚀 Running

```bash
docker compose up -d --build
```

- `cv-whisper-app` is published on `127.0.0.1:8389`, `cv-whisper` on `127.0.0.1:5001` (only the host itself; CV
  reaches them over `kokoro-net`).
- `docker-compose.yml` mounts the Hugging Face cache of the host so that the model (~3 GB) is downloaded only once:
  `C:/Users/Alex/.cache/huggingface`, or `HF_CACHE` in `.env` on another host.
- The volume `cuda-cache` keeps the CUDA kernels compiled for the GPU: without it the first request after a new
  container takes ~25 s longer.
- The first request after a start or an idle half hour loads the model (several seconds).

### Development

```bash
./mvnw spring-boot:run -Dspring-boot.run.profiles=dev
```

The `dev` profile listens on 8395 (Actuator on 8396) and calls the Python service at `http://localhost:5001`
(`WHISPER_URL`), e.g. the container of `docker compose up -d whisper`.

## 📚 API

### `POST /transcribe/{lang}`

A voice message. `lang`: `en-US`, `et-EE`, `ru`, ...; `none` - the language is detected. `multipart/form-data`:

| Field    | Required | Description                                                                   |
|----------|----------|-------------------------------------------------------------------------------|
| `file`   | yes      | audio in a format ffmpeg reads (webm, mp3, m4a, ogg, wav, flac, ...)          |
| `speech` | no       | `true` - also the intervals of speech (see below)                             |

```bash
curl -F file=@answer.webm -F speech=true http://localhost:8389/transcribe/en-US
```

```json
{
  "text": "Yesterday I went to the market. Then I met an old friend.",
  "language": "en",
  "segments": [
    {"start": 0.0, "end": 2.6, "text": "Yesterday I went to the market."},
    {"start": 3.8, "end": 5.9, "text": "Then I met an old friend."}
  ],
  "speech": [
    {"start": 0.21, "end": 2.48},
    {"start": 3.69, "end": 5.83}
  ]
}
```

- `segments` - Whisper's segments, seconds from the start.
- `speech` - only with `speech=true`: the intervals of speech found by the VAD (silences from 250 ms, no padding);
  the gaps between them are the pauses. Whisper's own timestamps stretch the words over the pauses, so CV measures
  the pauses and the tempo of the speaker by these intervals.

Errors come as `{"error": "..."}`:

| Status | When                                                         |
|--------|--------------------------------------------------------------|
| 400    | ffmpeg cannot decode the file                                |
| 500    | the transcription failed                                     |
| 502    | the Python service is not reachable                          |
| 504    | no answer within `whisper.timeout` (100 minutes)             |

A request without `file` gets the standard 400 of Spring Boot.

### `POST /transcribe-file/{lang}?maxSeconds=10800`

A long audio file (`multipart/form-data`, field `file`). A file that cannot be taken is refused at once with
`{"error", "reason"}`: 400 `decode` (not an audio file ffmpeg reads), 415 `video` (has a video track), 413
`too_long` (longer than `maxSeconds`, with `seconds`). Otherwise the answer is a stream of JSON lines
(`application/x-ndjson`):

```
{"stage": "converting"}
{"stage": "recognizing", "progress": 0}
{"stage": "recognizing", "progress": 0.412}
{"stage": "checking"}
{"result": {"text": "...", "language": "et", "duration": 3184.7, "segments": [{"start": 0.1, "end": 4.6, "text": "..."}]}}
```

`segments` are subtitles of up to 6 s made of the word timestamps. An error after the start comes as the last
line, `{"error", "reason"}`. Closing the connection stops the work.

### `POST /warmup`

Starts loading the model and answers at once: `{"loaded": true}` when it is loaded already. CV calls it when the
microphone is pressed, so the recording does not wait for the load (15-25 s).

## 📝 Configuration

`whisper-service` (environment of the container):

| Variable                | Default        | Description                                                     |
|-------------------------|----------------|-----------------------------------------------------------------|
| `WHISPER_MODEL`         | `large-v3`     | model name or path for faster-whisper                           |
| `WHISPER_DEVICE`        | `cuda`         | `cuda` or `cpu`                                                 |
| `WHISPER_COMPUTE_TYPE`  | `float16`      | CTranslate2 compute type (`int8_float16`: 20% slower, 1.7 GB less GPU memory) |
| `WHISPER_BEAM_SIZE`     | `5`            | 1 = greedy (faster), 5 = beam search (fewer errors in noise)    |
| `WHISPER_IDLE_TIMEOUT`  | `1800`         | seconds without requests before the model is unloaded           |
| `WHISPER_MAX_FILE_SECONDS` | `14400`     | the longest audio of `/transcribe-file`                         |

Spring Boot (`src/main/resources/application.properties`):

| Property                         | Value                   | Description                                         |
|----------------------------------|-------------------------|-----------------------------------------------------|
| `whisper.url` (`WHISPER_URL`)    | `http://cv-whisper:5001`| the Python service                                  |
| `whisper.timeout`                | `100m`                  | keep in sync with gunicorn `--timeout`              |
| `spring.mvc.async.request-timeout` | `100m`                | the stream of a long file                           |
| `spring.servlet.multipart.max-file-size` | `200MB`         | the largest upload                                  |

## 📈 Monitoring

Collected by the project [observability](https://github.com/GaileAlex/observability) (Grafana, Loki, Prometheus):

- `/actuator/health` and `/actuator/prometheus` of cv-whisper-app on the management port 8081 (inside the Docker
  network only), with the histogram of the transcription times;
- `[traceId-spanId]` in the log lines of cv-whisper-app: the traceId of the request of CV (`traceparent` header),
  so one search finds a voice message in the logs of both projects; nothing is exported;
- the Python service logs one line per transcription: file, language, length of the audio, time; for a long file
  also the chunks, the batch and the repaired chunks.

## 🧪 Testing

```bash
./mvnw verify
```

Unit tests of the controller (`@WebMvcTest`, the Python service mocked), of the request to the Python service
(`MockRestServiceServer`) and of the language mapping.

## 🔐 Dependencies

- GitHub Dependabot alerts and security updates are on; the automatic dependency submission reports the Maven
  dependencies the Spring Boot BOM brings (Tomcat, Jackson, ...), not only the ones in `pom.xml`.
- `pom.xml` may override versions of the Boot BOM (`tomcat.version`, `jackson-bom.version`) for an advisory that the
  current Boot release does not fix yet: drop the override when Boot catches up.
- `whisper-service/requirements.txt` pins the direct Python dependencies and PyAV (version 19 breaks
  faster-whisper 1.2.1, see the comment there); the other transitive ones (numpy, onnxruntime, huggingface-hub, ...)
  are the latest compatible at the time of the image build. A check of a running container:
  `docker exec cv-whisper pip freeze` against [OSV](https://osv.dev/).
