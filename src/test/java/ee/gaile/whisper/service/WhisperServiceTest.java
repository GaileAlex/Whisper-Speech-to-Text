package ee.gaile.whisper.service;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.assertj.core.api.Assertions.assertThat;

class WhisperServiceTest {

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
