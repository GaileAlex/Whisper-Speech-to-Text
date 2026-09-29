package ee.gaile.whisper.service;

import ee.gaile.whisper.dto.TranscriptionResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Locale;

@Slf4j
@Service
@RequiredArgsConstructor
public class WhisperService {

    private final WebClient whisperClient;

    /**
     * @param speech also the intervals of speech, for the pause analysis
     */
    public Mono<TranscriptionResult> transcribe(MultipartFile file, String lang, boolean speech) {
        String filename = file.getOriginalFilename();
        String language = toWhisperLanguage(lang);

        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("file", file.getResource());
        if (language != null) {
            builder.part("language", language);
        }
        if (speech) {
            builder.part("speech", "true");
        }

        log.info("Transcribing file: {} ({} bytes, language: {}{})",
                filename, file.getSize(), language != null ? language : "auto", speech ? ", speech intervals" : "");

        return whisperClient.post()
                .uri("/transcribe")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .bodyValue(builder.build())
                .retrieve()
                .bodyToMono(TranscriptionResult.class)
                .doOnNext(result -> {
                    log.info("Transcribed {}: language {}, {} segments, {} chars",
                            filename, result.language(), result.segments().size(), result.text().length());
                    log.debug("Transcription of {}: {}", filename, result.text());
                })
                .doOnError(error -> log.error("Transcription failed for {}: {}", filename, error.getMessage()));
    }

    /**
     * Converts a BCP 47 tag (en-US, et-EE, ...) to the ISO 639-1 code Whisper expects (en, et, ...).
     * Returns null for auto-detection.
     */
    static String toWhisperLanguage(String lang) {
        if (lang == null || lang.isBlank() || "none".equalsIgnoreCase(lang)) {
            return null;
        }
        return lang.strip().split("[-_]")[0].toLowerCase(Locale.ROOT);
    }

}
