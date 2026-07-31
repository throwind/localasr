from pathlib import Path
import unittest

from localasr.service import TranscriptionRequest


class ServiceTests(unittest.TestCase):
    def test_request_from_mapping_normalizes_formats(self) -> None:
        request = TranscriptionRequest.from_mapping(
            {
                "input_dir": "/tmp/in",
                "output_dir": "/tmp/out",
                "formats": "txt,json",
                "audio_files": ["/tmp/in/a.m4a", "/tmp/in/b.wav"],
                "recursive": False,
                "chunk_seconds": 300,
                "boundary_search_seconds": 20,
                "overlap_seconds": 4,
                "silence_threshold_db": -40,
                "silence_min_duration": 0.8,
            }
        )

        self.assertEqual(request.input_dir, Path("/tmp/in"))
        self.assertEqual(request.output_dir, Path("/tmp/out"))
        self.assertEqual(request.audio_files, (Path("/tmp/in/a.m4a"), Path("/tmp/in/b.wav")))
        self.assertEqual(request.formats, ("txt", "json"))
        self.assertFalse(request.recursive)
        self.assertEqual(request.chunk_seconds, 300)
        self.assertEqual(request.boundary_search_seconds, 20)
        self.assertEqual(request.overlap_seconds, 4)
        self.assertEqual(request.silence_threshold_db, -40)
        self.assertEqual(request.silence_min_duration, 0.8)


if __name__ == "__main__":
    unittest.main()
