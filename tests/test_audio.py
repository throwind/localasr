import unittest

from localasr.audio import SilenceInterval, choose_cut_point, plan_chunk_ranges


class AudioPlanningTests(unittest.TestCase):
    def test_choose_cut_point_prefers_nearby_silence(self) -> None:
        cut = choose_cut_point(
            target_seconds=600,
            duration=1200,
            boundary_search_seconds=30,
            silence_intervals=[SilenceInterval(585, 591)],
        )

        self.assertEqual(cut, 588)

    def test_choose_cut_point_falls_back_to_target_without_silence(self) -> None:
        cut = choose_cut_point(
            target_seconds=600,
            duration=1200,
            boundary_search_seconds=30,
            silence_intervals=[SilenceInterval(500, 510)],
        )

        self.assertEqual(cut, 600)

    def test_plan_chunk_ranges_adds_overlap_but_keeps_core_window(self) -> None:
        ranges = plan_chunk_ranges(
            duration=1250,
            chunk_seconds=600,
            boundary_search_seconds=30,
            overlap_seconds=5,
            silence_intervals=[SilenceInterval(590, 594), SilenceInterval(1190, 1194)],
        )

        self.assertEqual(ranges[0], (0.0, 597.0, 0.0, 592.0))
        self.assertEqual(ranges[1], (587.0, 1197.0, 592.0, 1192.0))
        self.assertEqual(ranges[2], (1187.0, 1250, 1192.0, 1250))


if __name__ == "__main__":
    unittest.main()
