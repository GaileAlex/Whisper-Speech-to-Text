package ee.gaile.whisper;

import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Map;

/**
 * @author Aleksei Gaile 6 Nov 2025
 */
@Service
@RequiredArgsConstructor
public class WhisperService {

    private final WebClient whisperClient;

    public Mono<Map> whisperTranslate(String selectedLang,  MultipartFile file) {
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

}
