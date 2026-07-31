from pathlib import Path
import tempfile
import unittest

from localasr.audio import find_audio_files
from localasr.config import AppConfig
from localasr.pipeline import (
    configured_audio_files,
    format_elapsed,
    format_srt_time,
    load_chunk_transcript,
    output_stem,
    render_srt,
    render_txt,
    safe_workdir_name,
    save_chunk_transcript,
    trim_transcript_to_window,
)
from localasr.transcriber import ChunkTranscript, Segment


class PipelineTests(unittest.TestCase):
    def test_find_audio_files_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp3").write_text("", encoding="utf-8")
            (root / "b.txt").write_text("", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "c.wav").write_text("", encoding="utf-8")

            found = find_audio_files(root, True, (".mp3", ".wav"))

            self.assertEqual([path.name for path in found], ["a.mp3", "c.wav"])

    def test_configured_audio_files_prefers_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "selected.m4a"
            ignored = root / "ignored.wav"
            selected.write_text("", encoding="utf-8")
            ignored.write_text("", encoding="utf-8")
            config = AppConfig(input_dir=root, output_dir=root / "out", audio_files=(selected,))

            self.assertEqual(configured_audio_files(config), [selected])

    def test_output_stem_falls_back_for_external_file(self) -> None:
        output = output_stem(Path("/tmp/external/interview.m4a"), Path("/tmp/in"), Path("/tmp/out"))

        self.assertEqual(output, Path("/tmp/out/interview"))

    def test_srt_render_uses_segment_timestamps(self) -> None:
        transcript = ChunkTranscript(
            index=0,
            source=Path("chunk.wav"),
            start_seconds=10,
            end_seconds=20,
            text="你好",
            segments=[Segment(text="你好", start_ms=10_500, end_ms=12_000, speaker=1)],
        )

        srt = render_srt([transcript])

        self.assertIn("00:00:10,500 --> 00:00:12,000", srt)
        self.assertIn("说话人 1: 你好", srt)

    def test_txt_render_keeps_speaker_labels(self) -> None:
        transcript = ChunkTranscript(
            index=0,
            source=Path("chunk.wav"),
            start_seconds=10,
            end_seconds=20,
            text="你好\n请介绍一下自己",
            segments=[
                Segment(text="你好", start_ms=10_500, end_ms=12_000, speaker=0),
                Segment(text="请介绍一下自己", start_ms=12_500, end_ms=15_000, speaker=1),
            ],
        )

        txt = render_txt([transcript])

        self.assertIn("说话人 0: 你好", txt)
        self.assertIn("说话人 1: 请介绍一下自己", txt)

    def test_txt_render_merges_consecutive_same_speaker_segments(self) -> None:
        transcript = ChunkTranscript(
            index=0,
            source=Path("chunk.wav"),
            start_seconds=10,
            end_seconds=20,
            text="",
            segments=[
                Segment(text="第一句。", start_ms=10_000, end_ms=11_000, speaker=1),
                Segment(text="第二句。", start_ms=11_000, end_ms=12_000, speaker=1),
                Segment(text="换人说话。", start_ms=12_000, end_ms=13_000, speaker=2),
            ],
        )

        txt = render_txt([transcript])

        self.assertIn("说话人 1: 第一句。第二句。", txt)
        self.assertIn("\n\n说话人 2: 换人说话。", txt)
        self.assertNotIn("说话人 1: 第二句。", txt)

    def test_txt_render_skips_punctuation_only_segments(self) -> None:
        transcript = ChunkTranscript(
            index=0,
            source=Path("chunk.wav"),
            start_seconds=10,
            end_seconds=20,
            text="",
            segments=[
                Segment(text=".", start_ms=10_000, end_ms=11_000, speaker=0),
                Segment(text="有效内容。", start_ms=11_000, end_ms=12_000, speaker=1),
            ],
        )

        txt = render_txt([transcript])

        self.assertNotIn("说话人 0", txt)
        self.assertIn("说话人 1: 有效内容。", txt)

    def test_txt_render_ignores_punctuation_when_merging_same_speaker(self) -> None:
        transcript = ChunkTranscript(
            index=0,
            source=Path("chunk.wav"),
            start_seconds=10,
            end_seconds=20,
            text="",
            segments=[
                Segment(text="第一句。", start_ms=10_000, end_ms=11_000, speaker=1),
                Segment(text=".", start_ms=11_000, end_ms=12_000, speaker=0),
                Segment(text="第二句。", start_ms=12_000, end_ms=13_000, speaker=1),
            ],
        )

        txt = render_txt([transcript])

        self.assertEqual(txt, "说话人 1: 第一句。第二句。\n")

    def test_format_srt_time(self) -> None:
        self.assertEqual(format_srt_time(3_723_004), "01:02:03,004")

    def test_format_elapsed(self) -> None:
        self.assertEqual(format_elapsed(4.2), "4秒")
        self.assertEqual(format_elapsed(62.0), "1分02秒")
        self.assertEqual(format_elapsed(3661.0), "1小时01分01秒")

    def test_safe_workdir_name_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "interview.m4a"
            audio.write_bytes(b"audio")

            self.assertEqual(safe_workdir_name(audio), safe_workdir_name(audio))

    def test_chunk_transcript_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "chunk_00001.json"
            transcript = ChunkTranscript(
                index=0,
                source=Path("chunk.wav"),
                start_seconds=0,
                end_seconds=60,
                text="你好",
                segments=[Segment(text="你好", start_ms=0, end_ms=1000, speaker=1)],
                warning="说话人识别失败，已降级为普通转写：示例",
            )

            save_chunk_transcript(cache_path, transcript)
            loaded = load_chunk_transcript(cache_path)

            self.assertEqual(loaded.text, "你好")
            self.assertEqual(loaded.warning, transcript.warning)
            self.assertEqual(loaded.segments[0].speaker, 1)

    def test_trim_transcript_to_window_filters_overlap_segments(self) -> None:
        transcript = ChunkTranscript(
            index=1,
            source=Path("chunk.wav"),
            start_seconds=585,
            end_seconds=1205,
            text="重复\n保留\n下段重复",
            segments=[
                Segment(text="重复", start_ms=586_000, end_ms=588_000),
                Segment(text="保留", start_ms=601_000, end_ms=603_000),
                Segment(text="下段重复", start_ms=1202_000, end_ms=1204_000),
            ],
        )

        trimmed = trim_transcript_to_window(transcript, keep_start_seconds=600, keep_end_seconds=1200)

        self.assertEqual(trimmed.start_seconds, 600)
        self.assertEqual(trimmed.end_seconds, 1200)
        self.assertEqual([segment.text for segment in trimmed.segments], ["保留"])
        self.assertEqual(trimmed.text, "保留")


if __name__ == "__main__":
    unittest.main()
