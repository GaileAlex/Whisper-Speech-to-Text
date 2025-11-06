package ee.gaile.whisper;

import lombok.RequiredArgsConstructor;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.io.InputStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * @author Aleksei Gaile 6 Nov 2025
 */
@Service
@RequiredArgsConstructor
public class WhisperService {

    private final WebClient whisperClient;

    public Mono<Map> whisperTranscribe(String selectedLang, MultipartFile file) {
        MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
        if (selectedLang.equals("en-US")) {
            selectedLang = "en";
        }
        if (selectedLang.equals("et-EE")) {
            selectedLang = "et";
        }

        bodyBuilder.part("file", file.getResource());
        bodyBuilder.part("language", selectedLang);

        return whisperClient.post()
                .uri("/transcribe")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .bodyValue(bodyBuilder.build())
                .retrieve()
                .bodyToMono(Map.class)
                .timeout(Duration.ofMinutes(1));
    }

    public Mono<Map<String, Object>> whisperTranscribe(MultipartFile file, String lang) {
        return Mono.create(sink -> {
            try (InputStream audioStream = file.getInputStream()) {

                List<String> chunkTexts = new ArrayList<>();
                int chunkSize = 100_000_000;
                byte[] buffer = new byte[chunkSize];
                int read;

                while ((read = audioStream.read(buffer)) != -1) {
                    byte[] chunk = Arrays.copyOf(buffer, read);
                    Map<String, Object> response = sendChunk(chunk, lang).block();
                    if (response != null && response.get("text") != null) {
                        chunkTexts.add(response.get("text").toString());
                    }
                }

                String finalText = String.join("\n", chunkTexts);
                sink.success(Map.of("text", finalText));

            } catch (Exception e) {
                sink.error(e);
            }
        });
    }

    private Mono<Map> sendChunk(byte[] audioBytes, String lang) {
        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("file", new ByteArrayResource(audioBytes) {
            @Override
            public String getFilename() {
                return "chunk.mp3";
            }
        });
        if (lang != null && !lang.isBlank()) {
            builder.part("language", normalizeLang(lang));
        }

        return whisperClient.post()
                .uri("/transcribe")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .bodyValue(builder.build())
                .retrieve()
                .bodyToMono(Map.class);
    }

    private String normalizeLang(String lang) {
        return switch (lang) {
            case "en-US" -> "en";
            case "et-EE" -> "et";
            case "ru-RU" -> "ru";
            default -> lang;
        };
    }

}
