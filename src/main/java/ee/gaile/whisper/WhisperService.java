package ee.gaile.whisper;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class WhisperService {

    private final WebClient whisperClient;

    public Mono<Map<String, Object>> whisperTranscribe(MultipartFile file, String lang) {
        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("file", file.getResource());
        log.info("Transcribing file: {}", file.getOriginalFilename());

        String normalized = normalizeLang(lang);
        if (normalized != null) {
            builder.part("language", normalized);
        }

        return whisperClient.post()
                .uri("/transcribe")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .bodyValue(builder.build())
                .retrieve()
                .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {
                })
                .timeout(Duration.ofMinutes(120))
                .doOnNext(result -> log.info("Transcription result for {}: {}", file.getOriginalFilename(), result))
                .doOnError(error -> log.error("Transcription failed for {}: {}", file.getOriginalFilename(), error.getMessage()));
    }

    public Mono<Map<String, Object>> whisperTranscribe(String selectedLang, MultipartFile file) {
        return whisperTranscribe(file, selectedLang);
    }

    private String normalizeLang(String lang) {
        if (lang == null || lang.isBlank() || "none".equalsIgnoreCase(lang)) {
            return null;
        }
        return switch (lang) {
            case "en-US" -> "en";
            case "et-EE" -> "et";
            case "ru-RU" -> "ru";
            default -> lang;
        };
    }

}
