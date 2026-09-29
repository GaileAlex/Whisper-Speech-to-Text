package ee.gaile.whisper.controller;

import ee.gaile.whisper.dto.TranscriptionResult;
import ee.gaile.whisper.service.WhisperService;
import io.netty.handler.timeout.ReadTimeoutException;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.RequestBuilder;
import org.springframework.test.web.servlet.ResultActions;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.web.reactive.function.client.WebClientRequestException;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Mono;

import java.net.ConnectException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.BDDMockito.given;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.request;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(WhisperController.class)
class WhisperControllerTest {

    private static final MockMultipartFile FILE =
            new MockMultipartFile("file", "audio.mp3", "audio/mpeg", new byte[]{1, 2, 3});

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private WhisperService whisperService;

    @Test
    void transcribe_returnsResult() throws Exception {
        given(whisperService.transcribe(any(), eq("et-EE"), eq(false))).willReturn(Mono.just(
                new TranscriptionResult("tere", "et", List.of(new TranscriptionResult.Segment(0.0, 1.5, "tere")),
                        null)));

        perform(MockMvcRequestBuilders.multipart("/transcribe").file(FILE).param("lang", "et-EE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.text").value("tere"))
                .andExpect(jsonPath("$.language").value("et"))
                .andExpect(jsonPath("$.segments[0].end").value(1.5))
                // the clients that did not ask for the speech intervals get the same result as before
                .andExpect(jsonPath("$.speech").doesNotExist());
    }

    @Test
    void transcribe_withPathLanguage_returnsResult() throws Exception {
        given(whisperService.transcribe(any(), eq("ru-RU"), eq(false))).willReturn(Mono.just(
                new TranscriptionResult("привет", "ru", List.of(), null)));

        perform(MockMvcRequestBuilders.multipart("/transcribe/ru-RU").file(FILE))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.text").value("привет"));
    }

    @Test
    void transcribe_withSpeech_returnsSpeechIntervals() throws Exception {
        given(whisperService.transcribe(any(), eq("en-US"), eq(true))).willReturn(Mono.just(
                new TranscriptionResult("I went home", "en", List.of(),
                        List.of(new TranscriptionResult.Interval(0.5, 1.2), new TranscriptionResult.Interval(1.9, 2.4)))));

        perform(MockMvcRequestBuilders.multipart("/transcribe/en-US").file(FILE).param("speech", "true"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.speech.length()").value(2))
                .andExpect(jsonPath("$.speech[1].start").value(1.9))
                .andExpect(jsonPath("$.speech[1].end").value(2.4));
    }

    @Test
    void transcribe_passesWhisperErrorThrough() throws Exception {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        given(whisperService.transcribe(any(), any(), anyBoolean())).willReturn(Mono.error(WebClientResponseException.create(
                400, "Bad Request", headers,
                "{\"error\":\"cannot decode audio\"}".getBytes(StandardCharsets.UTF_8), StandardCharsets.UTF_8)));

        perform(MockMvcRequestBuilders.multipart("/transcribe").file(FILE))
                .andExpect(status().isBadRequest())
                .andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.error").value("cannot decode audio"));
    }

    @Test
    void transcribe_returnsBadGatewayWhenWhisperIsDown() throws Exception {
        given(whisperService.transcribe(any(), any(), anyBoolean())).willReturn(Mono.error(requestException(
                new ConnectException("Connection refused"))));

        perform(MockMvcRequestBuilders.multipart("/transcribe").file(FILE))
                .andExpect(status().isBadGateway())
                .andExpect(jsonPath("$.error").value("Whisper service is unavailable"));
    }

    @Test
    void transcribe_returnsGatewayTimeoutOnWhisperTimeout() throws Exception {
        given(whisperService.transcribe(any(), any(), anyBoolean())).willReturn(Mono.error(requestException(
                ReadTimeoutException.INSTANCE)));

        perform(MockMvcRequestBuilders.multipart("/transcribe").file(FILE))
                .andExpect(status().isGatewayTimeout())
                .andExpect(jsonPath("$.error").value("Whisper service timed out"));
    }

    private ResultActions perform(RequestBuilder requestBuilder) throws Exception {
        MvcResult asyncResult = mockMvc.perform(requestBuilder)
                .andExpect(request().asyncStarted())
                .andReturn();
        return mockMvc.perform(asyncDispatch(asyncResult));
    }

    private static WebClientRequestException requestException(Throwable cause) {
        return new WebClientRequestException(cause, HttpMethod.POST, URI.create("http://whisper/transcribe"),
                new HttpHeaders());
    }

}
