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
import org.springframework.http.client.ClientHttpResponse;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClient;

import java.nio.charset.StandardCharsets;

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

    @Test
    void transcribeFile_sendsTheLanguageAndTheLimitAndReturnsTheStreamOpen() throws Exception {
        whisper.expect(requestTo("http://whisper/transcribe-file"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(content().string(allOf(
                        containsString("name=\"file\"; filename=\"audio.webm\""),
                        matchesRegex("(?s).*name=\"language\".*\r\n\r\net\r\n.*"),
                        matchesRegex("(?s).*name=\"max_seconds\".*\r\n\r\n10800\r\n.*"))))
                .andRespond(withSuccess("{\"stage\": \"converting\"}\n", MediaType.parseMediaType("application/x-ndjson")));

        try (ClientHttpResponse response = service.transcribeFile(FILE, "et-EE", 10800)) {
            assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
            assertThat(new String(response.getBody().readAllBytes(), StandardCharsets.UTF_8))
                    .isEqualTo("{\"stage\": \"converting\"}\n");
        }
        whisper.verify();
    }

    @Test
    void transcribeFile_returnsARefusalInsteadOfThrowing() throws Exception {
        whisper.expect(requestTo("http://whisper/transcribe-file"))
                .andRespond(withStatus(HttpStatus.CONTENT_TOO_LARGE)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"error\": \"the audio is too long\", \"reason\": \"too_long\", \"seconds\": 12000}"));

        try (ClientHttpResponse response = service.transcribeFile(FILE, null, 10800)) {
            assertThat(response.getStatusCode()).isEqualTo(HttpStatus.CONTENT_TOO_LARGE);
        }
    }

    @Test
    void warmup_returnsWhetherTheModelIsLoaded() {
        whisper.expect(requestTo("http://whisper/warmup"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withSuccess("{\"loaded\": false}", MediaType.APPLICATION_JSON));

        assertThat(service.warmup(null)).containsEntry("loaded", false);
    }

    @Test
    void warmup_passesTheLanguageAsWhisperNamesIt() {
        whisper.expect(requestTo("http://whisper/warmup?language=et"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withSuccess("{\"loaded\": true}", MediaType.APPLICATION_JSON));

        assertThat(service.warmup("et-EE")).containsEntry("loaded", true);
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
