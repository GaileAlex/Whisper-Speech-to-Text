package ee.gaile.whisper;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.handler.timeout.WriteTimeoutHandler;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

@Configuration
public class WebClientConfig {
    @Value("${whisper.url}")
    private String whisperUrl;

    @Bean
    public WebClient whisperClient() {
        HttpClient httpClient = HttpClient.create()
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 300_000)
                .doOnConnected(conn -> conn
                        .addHandlerLast(new ReadTimeoutHandler(300))
                        .addHandlerLast(new WriteTimeoutHandler(300))
                );

        return WebClient.builder()
                .baseUrl(whisperUrl)
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .build();
    }

}
