package ee.gaile.whisper;

import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import reactor.core.publisher.Mono;

import java.util.Map;

/**
 * @author Aleksei Gaile 6 Nov 2025
 */
@RestController
@RequiredArgsConstructor
public class WhisperController {

    private final WhisperService whisperService;

    public record TtsRequest(String text, String speakerName, Integer speed) {
    }

    @PostMapping(value ="/transcribe/{selectedLang}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<Map> submitTtsJob(@PathVariable("selectedLang") String selectedLang,
                                  @RequestPart("file") MultipartFile file) {
        return whisperService.whisperTranslate(selectedLang, file);

    }

}
