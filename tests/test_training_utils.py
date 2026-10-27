import unittest
from pathlib import Path

from train import RUN_COLUMNS, append_run, learning_rate


class LearningRateTests(unittest.TestCase):
    def test_schedule_endpoints(self) -> None:
        self.assertEqual(learning_rate(0, 1000, 100, 1e-3), 0.0)
        self.assertAlmostEqual(learning_rate(100, 1000, 100, 1e-3), 1e-3)
        self.assertAlmostEqual(learning_rate(1000, 1000, 100, 1e-3), 1e-4)

    def test_repeated_run_replaces_its_old_row(self) -> None:
        path = Path(__file__).parent / "_temporary_runs.csv"
        first = {column: "" for column in RUN_COLUMNS}
        first.update({"run_id": "S", "best_val_loss": 4.0})
        second = dict(first)
        second["best_val_loss"] = 3.5

        try:
            append_run(path, first)
            append_run(path, second)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("3.5", lines[1])
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

