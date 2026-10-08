package ee.gaile.whisper.controller;

import ee.gaile.whisper.dto.TranscriptionResult;
import ee.gaile.whisper.service.WhisperService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.mock.http.client.MockClientHttpResponse;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.ResourceAccessException;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.net.ConnectException;
import java.net.http.HttpTimeoutException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.BDDMockito.given;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(WhisperController.class)
class WhisperControllerTest {

    private static final MockMultipartFile FILE =
            new MockMultipartFile("file", "audio.mp3", "audio/mpeg", new byte[]{1, 2, 3});
    private static final MediaType NDJSON = MediaType.parseMediaType("application/x-ndjson");

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private WhisperService whisperService;

    @Test
    void transcribe_returnsResult() throws Exception {
        given(whisperService.transcribe(any(), eq("et-EE"), eq(false))).willReturn(
                new TranscriptionResult("tere", "et", List.of(new TranscriptionResult.Segment(0.0, 1.5, "tere")),
                        null));

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe/et-EE").file(FILE))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.text").value("tere"))
                .andExpect(jsonPath("$.language").value("et"))
                .andExpect(jsonPath("$.segments[0].end").value(1.5))
                // no speech field unless it was asked for: the response of the older clients is unchanged
                .andExpect(jsonPath("$.speech").doesNotExist());
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

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe/none").file(FILE))
                .andExpect(status().isBadRequest())
                .andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.error").value("cannot decode audio"));
    }

    @Test
    void transcribe_returnsBadGatewayWhenWhisperIsDown() throws Exception {
        given(whisperService.transcribe(any(), any(), anyBoolean())).willThrow(
                new ResourceAccessException("I/O error", new ConnectException("Connection refused")));

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe/none").file(FILE))
                .andExpect(status().isBadGateway())
                .andExpect(jsonPath("$.error").value("Whisper service is unavailable"));
    }

    @Test
    void transcribe_returnsGatewayTimeoutOnWhisperTimeout() throws Exception {
        given(whisperService.transcribe(any(), any(), anyBoolean())).willThrow(
                new ResourceAccessException("I/O error", new HttpTimeoutException("request timed out")));

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe/none").file(FILE))
                .andExpect(status().isGatewayTimeout())
                .andExpect(jsonPath("$.error").value("Whisper service timed out"));
    }

    @Test
    void transcribeFile_streamsTheLinesOfTheWhisperService() throws Exception {
        String lines = "{\"stage\": \"converting\"}\n{\"stage\": \"recognizing\", \"progress\": 0.5}\n"
                + "{\"result\": {\"text\": \"tere\", \"language\": \"et\", \"duration\": 1.5, \"segments\": []}}\n";
        given(whisperService.transcribeFile(any(), eq("et-EE"), eq(10800L)))
                .willReturn(answer(HttpStatus.OK, NDJSON, lines));

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe-file/et-EE").file(FILE)
                        .param("maxSeconds", "10800"))
                .andExpect(status().isOk())
                .andExpect(content().contentTypeCompatibleWith(NDJSON))
                .andExpect(content().string(lines));
    }

    @Test
    void copyFlushing_sendsEveryPieceAtOnce() throws Exception {
        List<String> flushed = new ArrayList<>();
        ByteArrayOutputStream sent = new ByteArrayOutputStream() {
            @Override
            public void flush() {
                flushed.add(toString(StandardCharsets.UTF_8));
            }
        };
        // a stream that gives one line per read, as the answer of Whisper comes
        InputStream lines = new SequenceInputStream(Collections.enumeration(List.of(
                new ByteArrayInputStream("{\"stage\": \"converting\"}\n".getBytes(StandardCharsets.UTF_8)),
                new ByteArrayInputStream("{\"stage\": \"checking\"}\n".getBytes(StandardCharsets.UTF_8)))));

        WhisperController.copyFlushing(lines, sent);

        assertThat(flushed).containsExactly("{\"stage\": \"converting\"}\n",
                "{\"stage\": \"converting\"}\n{\"stage\": \"checking\"}\n");
    }

    @Test
    void transcribeFile_passesARefusalThrough() throws Exception {
        String refusal = "{\"error\": \"the file has a video track\", \"reason\": \"video\"}";
        given(whisperService.transcribeFile(any(), any(), anyLong()))
                .willReturn(answer(HttpStatus.UNSUPPORTED_MEDIA_TYPE, MediaType.APPLICATION_JSON, refusal));

        mockMvc.perform(MockMvcRequestBuilders.multipart("/transcribe-file/none").file(FILE)
                        .param("maxSeconds", "10800"))
                .andExpect(status().isUnsupportedMediaType())
                .andExpect(jsonPath("$.reason").value("video"));
    }

    @Test
    void warmup_answersWhetherTheModelIsLoaded() throws Exception {
        given(whisperService.warmup()).willReturn(Map.of("loaded", false));

        mockMvc.perform(post("/warmup"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.loaded").value(false));
    }

    private static MockClientHttpResponse answer(HttpStatus status, MediaType type, String body) {
        MockClientHttpResponse response = new MockClientHttpResponse(body.getBytes(StandardCharsets.UTF_8), status);
        response.getHeaders().setContentType(type);
        return response;
    }

}
