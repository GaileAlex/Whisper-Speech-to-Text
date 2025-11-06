package ee.gaile.whisper;

import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
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
    public Mono<Map> whisperTranscribe(@PathVariable("selectedLang") String selectedLang,
                                       @RequestPart("file") MultipartFile file) {
        return whisperService.whisperTranscribe(selectedLang, file);

    }

    @PostMapping("/transcribe")
    public Mono<Map<String, Object>> transcribe(@RequestParam("file") MultipartFile file,
                                                @RequestParam(value = "lang", required = false) String lang) {
        return whisperService.whisperTranscribe(file, lang);
    }

}
