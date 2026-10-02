package ee.gaile.whisper.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

import java.net.http.HttpClient;
import java.time.Duration;

@Configuration
public class WhisperClientConfig {

    /**
     * The request thread waits for the transcription: it is a virtual thread (spring.threads.virtual.enabled).
     * The upload is streamed from the temp file of the multipart request, not read into memory.
     */
    @Bean
    public RestClient whisperClient(@Value("${whisper.url}") String whisperUrl,
                                    @Value("${whisper.timeout}") Duration timeout) {
        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                // gunicorn speaks HTTP/1.1: no attempt to upgrade the connection to HTTP/2
                .version(HttpClient.Version.HTTP_1_1)
                .build();
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(timeout);

        return RestClient.builder()
                .baseUrl(whisperUrl)
                .requestFactory(requestFactory)
                .build();
    }

}
