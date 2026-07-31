from unittest.mock import patch
import unittest

from localasr.transcriber import SenseVoiceTranscriber
from localasr.onnx_transcriber import (
    clean_sense_voice_text,
    extract_result_text,
    iter_diarization_windows,
    max_diarization_window_seconds,
)


class TranscriberTests(unittest.TestCase):
    def test_clean_sense_voice_text_removes_tags(self) -> None:
        self.assertEqual(clean_sense_voice_text("<|zh|><|NEUTRAL|>你好"), "你好")

    def test_extract_result_text_accepts_object(self) -> None:
        class Result:
            text = "你好"

        self.assertEqual(extract_result_text(Result()), "你好")

    def test_extract_segments_accepts_funasr_sentence_key(self) -> None:
        transcriber = SenseVoiceTranscriber.__new__(SenseVoiceTranscriber)
        transcriber.postprocess = lambda text: text.replace("<|zh|>", "")

        segments = transcriber._extract_segments(
            {
                "sentence_info": [
                    {
                        "sentence": "<|zh|>你好",
                        "start": 100,
                        "end": 900,
                        "spk": 1,
                    }
                ]
            },
            chunk_offset_ms=1000,
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "你好")
        self.assertEqual(segments[0].start_ms, 1100)
        self.assertEqual(segments[0].end_ms, 1900)
        self.assertEqual(segments[0].speaker, 1)

    def test_extract_segments_skips_punctuation_only_text(self) -> None:
        transcriber = SenseVoiceTranscriber.__new__(SenseVoiceTranscriber)
        transcriber.postprocess = lambda text: text

        segments = transcriber._extract_segments(
            {"sentence_info": [{"sentence": ".", "start": 100, "end": 200, "spk": 0}]},
            chunk_offset_ms=0,
        )

        self.assertEqual(segments, [])

    def test_diarization_windows_split_long_audio(self) -> None:
        audio = list(range(10))

        windows = list(iter_diarization_windows(audio, sample_rate=1, max_window_seconds=4))

        self.assertEqual([(start, chunk) for start, chunk in windows], [(0.0, [0, 1, 2, 3]), (4.0, [4, 5, 6, 7]), (8.0, [8, 9])])

    def test_max_diarization_window_seconds_env_bounds(self) -> None:
        with patch.dict("os.environ", {"LOCALASR_MAX_DIARIZATION_SECONDS": "5"}):
            self.assertEqual(max_diarization_window_seconds(), 10)
        with patch.dict("os.environ", {"LOCALASR_MAX_DIARIZATION_SECONDS": "9999"}):
            self.assertEqual(max_diarization_window_seconds(), 600)
        with patch.dict("os.environ", {"LOCALASR_MAX_DIARIZATION_SECONDS": "bad"}):
            self.assertEqual(max_diarization_window_seconds(), 30)


if __name__ == "__main__":
    unittest.main()
