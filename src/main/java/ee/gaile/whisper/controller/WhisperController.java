package ee.gaile.whisper.controller;

import ee.gaile.whisper.dto.TranscriptionResult;
import ee.gaile.whisper.service.WhisperService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.multipart.MultipartFile;

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

    @PostMapping("/transcribe")
    public TranscriptionResult transcribe(@RequestParam("file") MultipartFile file,
                                          @RequestParam(value = "lang", required = false) String lang,
                                          @RequestParam(value = "speech", defaultValue = "false") boolean speech) {
        return whisperService.transcribe(file, lang, speech);
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
