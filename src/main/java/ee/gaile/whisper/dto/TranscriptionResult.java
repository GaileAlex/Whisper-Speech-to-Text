package ee.gaile.whisper.dto;

import java.util.List;

public record TranscriptionResult(String text, String language, List<Segment> segments) {

    public record Segment(double start, double end, String text) {
    }

}
