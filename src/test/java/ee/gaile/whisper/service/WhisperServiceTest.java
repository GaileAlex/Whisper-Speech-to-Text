package ee.gaile.whisper.service;

import ee.gaile.whisper.dto.TranscriptionResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClient;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.matchesRegex;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

class WhisperServiceTest {

    private static final MockMultipartFile FILE =
            new MockMultipartFile("file", "audio.webm", "audio/webm", new byte[]{1, 2, 3});

    private final RestClient.Builder client = RestClient.builder().baseUrl("http://whisper");
    private final MockRestServiceServer whisper = MockRestServiceServer.bindTo(client).build();
    private final WhisperService service = new WhisperService(client.build());

    @Test
    void transcribe_sendsTheLanguageAndTheSpeechFlag() {
        whisper.expect(requestTo("http://whisper/transcribe"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(content().contentTypeCompatibleWith(MediaType.MULTIPART_FORM_DATA))
                .andExpect(content().string(allOf(
                        containsString("name=\"file\"; filename=\"audio.webm\""),
                        matchesRegex("(?s).*name=\"language\".*\r\n\r\net\r\n.*"),
                        matchesRegex("(?s).*name=\"speech\".*\r\n\r\ntrue\r\n.*"))))
                .andRespond(withSuccess("""
                        {"text": "tere", "language": "et", "segments": [{"start": 0.0, "end": 1.5, "text": "tere"}],
                         "speech": [{"start": 0.5, "end": 1.2}]}""", MediaType.APPLICATION_JSON));

        TranscriptionResult result = service.transcribe(FILE, "et-EE", true);

        assertThat(result.text()).isEqualTo("tere");
        assertThat(result.speech()).containsExactly(new TranscriptionResult.Interval(0.5, 1.2));
        whisper.verify();
    }

    @Test
    void transcribe_throwsTheErrorOfTheWhisperService() {
        whisper.expect(requestTo("http://whisper/transcribe"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"error\": \"cannot decode audio\"}"));

        assertThatExceptionOfType(HttpClientErrorException.class)
                .isThrownBy(() -> service.transcribe(FILE, null, false))
                .satisfies(e -> assertThat(e.getResponseBodyAsString()).contains("cannot decode audio"));
    }

    @ParameterizedTest
    @CsvSource({
            "en-US, en",
            "et-EE, et",
            "ru-RU, ru",
            "de_DE, de",
            "et, et",
            "EN, en"
    })
    void toWhisperLanguage_convertsToIsoCode(String lang, String expected) {
        assertThat(WhisperService.toWhisperLanguage(lang)).isEqualTo(expected);
    }

    @ParameterizedTest
    @NullAndEmptySource
    @ValueSource(strings = {" ", "none", "NONE"})
    void toWhisperLanguage_returnsNullForAutoDetection(String lang) {
        assertThat(WhisperService.toWhisperLanguage(lang)).isNull();
    }

}
