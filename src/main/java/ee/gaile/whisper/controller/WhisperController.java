package ee.gaile.whisper.controller;

import ee.gaile.whisper.dto.TranscriptionResult;
import ee.gaile.whisper.service.WhisperService;
import io.netty.handler.timeout.ReadTimeoutException;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClientRequestException;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Mono;

import java.util.Map;

/**
 * @author Aleksei Gaile 6 Nov 2025
 */
@CrossOrigin(origins = "*")
@RestController
@RequiredArgsConstructor
public class WhisperController {

    private final WhisperService whisperService;

    @PostMapping(value = "/transcribe/{selectedLang}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<TranscriptionResult> whisperTranscribe(@PathVariable("selectedLang") String selectedLang,
                                                       @RequestPart("file") MultipartFile file) {
        return whisperService.transcribe(file, selectedLang);
    }

    @PostMapping("/transcribe")
    public Mono<TranscriptionResult> transcribe(@RequestParam("file") MultipartFile file,
                                                @RequestParam(value = "lang", required = false) String lang) {
        return whisperService.transcribe(file, lang);
    }

    /**
     * Passes the whisper service error (status and {"error": ...} body) through to the client.
     */
    @ExceptionHandler(WebClientResponseException.class)
    public ResponseEntity<String> handleWhisperError(WebClientResponseException e) {
        return ResponseEntity.status(e.getStatusCode())
                .headers(headers -> headers.setContentType(e.getHeaders().getContentType()))
                .body(e.getResponseBodyAsString());
    }

    @ExceptionHandler(WebClientRequestException.class)
    public ResponseEntity<Map<String, String>> handleWhisperUnavailable(WebClientRequestException e) {
        if (e.getCause() instanceof ReadTimeoutException) {
            return ResponseEntity.status(HttpStatus.GATEWAY_TIMEOUT)
                    .body(Map.of("error", "Whisper service timed out"));
        }
        return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body(Map.of("error", "Whisper service is unavailable"));
    }

}
