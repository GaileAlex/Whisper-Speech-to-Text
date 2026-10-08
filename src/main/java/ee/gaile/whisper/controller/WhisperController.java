package ee.gaile.whisper.controller;

import ee.gaile.whisper.dto.TranscriptionResult;
import ee.gaile.whisper.service.WhisperService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.ClientHttpResponse;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.http.HttpTimeoutException;
import java.util.Map;

/**
 * @author Aleksei Gaile 6 Nov 2025
 */
@RestController
@RequiredArgsConstructor
public class WhisperController {

    private final WhisperService whisperService;

    /**
     * @param speech true: the result has the intervals of speech too, see {@link TranscriptionResult#speech()}
     */
    @PostMapping(value = "/transcribe/{selectedLang}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public TranscriptionResult whisperTranscribe(@PathVariable("selectedLang") String selectedLang,
                                                 @RequestPart("file") MultipartFile file,
                                                 @RequestParam(value = "speech", defaultValue = "false") boolean speech) {
        return whisperService.transcribe(file, selectedLang, speech);
    }

    /**
     * A long recording uploaded as a file. The answer of the Python service is passed on as it comes: a refusal
     * with its status, or the stream of JSON lines with the progress and at the end the result.
     *
     * @param maxSeconds the longest audio the caller takes
     */
    @PostMapping(value = "/transcribe-file/{selectedLang}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<StreamingResponseBody> transcribeFile(@PathVariable("selectedLang") String selectedLang,
                                                                @RequestPart("file") MultipartFile file,
                                                                @RequestParam("maxSeconds") long maxSeconds)
            throws IOException {
        ClientHttpResponse response = whisperService.transcribeFile(file, selectedLang, maxSeconds);
        StreamingResponseBody body = out -> {
            try (response; InputStream in = response.getBody()) {
                copyFlushing(in, out);
            }
        };
        return ResponseEntity.status(response.getStatusCode())
                .contentType(response.getHeaders().getContentType())
                .body(body);
    }

    /**
     * Every piece is sent on at once: a line of progress must not wait in a buffer. When the client goes away the
     * write fails, and closing the answer of the Python service stops the transcription there.
     */
    static void copyFlushing(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[8192];
        int read;
        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
            out.flush();
        }
    }

    /**
     * Loads the model ahead of a recording; answers at once.
     */
    @PostMapping("/warmup")
    public Map<String, Object> warmup() {
        return whisperService.warmup();
    }

    /**
     * Passes the whisper service error (status and {"error": ...} body) through to the client.
     */
    @ExceptionHandler(RestClientResponseException.class)
    public ResponseEntity<String> handleWhisperError(RestClientResponseException e) {
        return ResponseEntity.status(e.getStatusCode())
                .headers(headers -> headers.setContentType(e.getResponseHeaders().getContentType()))
                .body(e.getResponseBodyAsString());
    }

    @ExceptionHandler(ResourceAccessException.class)
    public ResponseEntity<Map<String, String>> handleWhisperUnavailable(ResourceAccessException e) {
        if (e.getCause() instanceof HttpTimeoutException) {
            return ResponseEntity.status(HttpStatus.GATEWAY_TIMEOUT)
                    .body(Map.of("error", "Whisper service timed out"));
        }
        return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body(Map.of("error", "Whisper service is unavailable"));
    }

}
