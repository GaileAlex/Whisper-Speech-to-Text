package ee.gaile.whisper.controller;

import ee.gaile.whisper.dto.TranscriptionResult;
import ee.gaile.whisper.service.WhisperService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.ResourceAccessException;

import java.net.ConnectException;
import java.net.http.HttpTimeoutException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.BDDMockito.given;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
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
        given(whisperService.transcribe(any(), eq("et-EE"), eq(false))).willReturn(
                new TranscriptionResult("tere", "et", List.of(new TranscriptionResult.Segment(0.0, 1.5, "tere")),
                        null));

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe").file(FILE).param("lang", "et-EE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.text").value("tere"))
                .andExpect(jsonPath("$.language").value("et"))
                .andExpect(jsonPath("$.segments[0].end").value(1.5))
                // no speech field unless it was asked for: the response of the older clients is unchanged
                .andExpect(jsonPath("$.speech").doesNotExist());
    }

    @Test
    void transcribe_withPathLanguage_returnsResult() throws Exception {
        given(whisperService.transcribe(any(), eq("ru-RU"), eq(false))).willReturn(
                new TranscriptionResult("привет", "ru", List.of(), null));

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe/ru-RU").file(FILE))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.text").value("привет"));
    }

    @Test
    void transcribe_withSpeech_returnsSpeechIntervals() throws Exception {
        given(whisperService.transcribe(any(), eq("en-US"), eq(true))).willReturn(
                new TranscriptionResult("I went home", "en", List.of(),
                        List.of(new TranscriptionResult.Interval(0.5, 1.2), new TranscriptionResult.Interval(1.9, 2.4))));

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe/en-US").file(FILE).param("speech", "true"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.speech.length()").value(2))
                .andExpect(jsonPath("$.speech[1].start").value(1.9))
                .andExpect(jsonPath("$.speech[1].end").value(2.4));
    }

    @Test
    void transcribe_passesWhisperErrorThrough() throws Exception {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        given(whisperService.transcribe(any(), any(), anyBoolean())).willThrow(HttpClientErrorException.create(
                HttpStatus.BAD_REQUEST, "Bad Request", headers,
                "{\"error\":\"cannot decode audio\"}".getBytes(StandardCharsets.UTF_8), StandardCharsets.UTF_8));

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe").file(FILE))
                .andExpect(status().isBadRequest())
                .andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.error").value("cannot decode audio"));
    }

    @Test
    void transcribe_returnsBadGatewayWhenWhisperIsDown() throws Exception {
        given(whisperService.transcribe(any(), any(), anyBoolean())).willThrow(
                new ResourceAccessException("I/O error", new ConnectException("Connection refused")));

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe").file(FILE))
                .andExpect(status().isBadGateway())
                .andExpect(jsonPath("$.error").value("Whisper service is unavailable"));
    }

    @Test
    void transcribe_returnsGatewayTimeoutOnWhisperTimeout() throws Exception {
        given(whisperService.transcribe(any(), any(), anyBoolean())).willThrow(
                new ResourceAccessException("I/O error", new HttpTimeoutException("request timed out")));

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe").file(FILE))
                .andExpect(status().isGatewayTimeout())
                .andExpect(jsonPath("$.error").value("Whisper service timed out"));
    }

}
