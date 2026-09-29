package ee.gaile.whisper.dto;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.List;

/**
 * @param speech the intervals of speech (the gaps between them are the pauses); only when they were asked for
 */
public record TranscriptionResult(String text, String language, List<Segment> segments,
                                  @JsonInclude(JsonInclude.Include.NON_NULL) List<Interval> speech) {

    public record Segment(double start, double end, String text) {
    }

    /**
     * Seconds from the start of the audio.
     */
    public record Interval(double start, double end) {
    }

}
