package ee.gaile.whisper.service;

import ee.gaile.whisper.dto.TranscriptionResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.multipart.MultipartFile;

import java.util.Locale;

@Slf4j
@Service
@RequiredArgsConstructor
public class WhisperService {

    private final RestClient whisperClient;

    /**
     * @param speech also the intervals of speech, for the pause analysis
     * @throws RestClientException when the Python service fails or is unavailable
     */
    public TranscriptionResult transcribe(MultipartFile file, String lang, boolean speech) {
        String filename = file.getOriginalFilename();
        String language = toWhisperLanguage(lang);

        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("file", file.getResource());
        if (language != null) {
            parts.add("language", language);
        }
        if (speech) {
            parts.add("speech", "true");
        }

        log.info("Transcribing file: {} ({} bytes, language: {}{})",
                filename, file.getSize(), language != null ? language : "auto", speech ? ", speech intervals" : "");

        TranscriptionResult result;
        try {
            result = whisperClient.post()
                    .uri("/transcribe")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(parts)
                    .retrieve()
                    .body(TranscriptionResult.class);
        } catch (RestClientException e) {
            log.error("Transcription failed for {}: {}", filename, e.getMessage());
            throw e;
        }

        log.info("Transcribed {}: language {}, {} segments, {} chars",
                filename, result.language(), result.segments().size(), result.text().length());
        log.debug("Transcription of {}: {}", filename, result.text());
        return result;
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
