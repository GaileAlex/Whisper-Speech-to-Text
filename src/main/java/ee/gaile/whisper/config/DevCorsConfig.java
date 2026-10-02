package ee.gaile.whisper.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * dev/whisper.html is opened from the disk and calls the dev profile in the browser. In production only the server
 * of CV calls the API: no CORS there.
 */
@Configuration
@Profile("dev")
public class DevCorsConfig implements WebMvcConfigurer {

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/transcribe/**").allowedOrigins("*");
    }

}
