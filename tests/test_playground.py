import unittest

from playground import train


class TrainingTests(unittest.TestCase):
    def test_hand_calculated_update(self):
        before, after = train(0.01, steps=1)
        self.assertEqual(before["gradient"], -50)
        self.assertEqual(before["loss"], 25)
        self.assertEqual(after["weight"], 2.5)
        self.assertEqual(after["loss"], 6.25)

    def test_one_update_solution(self):
        self.assertEqual(train(0.02, steps=1)[-1]["weight"], 3)
        self.assertEqual(train(0.02)[-1]["loss"], 0)

    def test_stable_rate_reduces_loss_each_step(self):
        losses = [row["loss"] for row in train(0.01)]
        self.assertTrue(all(b < a for a, b in zip(losses, losses[1:])))

    def test_boundary_oscillates(self):
        rows = train(0.04, steps=4)
        self.assertEqual([row["weight"] for row in rows], [2, 4, 2, 4, 2])
        self.assertTrue(all(row["loss"] == 25 for row in rows))

    def test_large_rate_diverges(self):
        losses = [row["loss"] for row in train(0.1, steps=4)]
        self.assertEqual(losses[:2], [25, 400])
        self.assertTrue(all(b > a for a, b in zip(losses, losses[1:])))

    def test_different_input_changes_optimal_rate(self):
        self.assertEqual(train(0.125, x=2, target=6, initial_weight=0)[-1]["weight"], 3)

    def test_overflow_stops_at_finite_measurements(self):
        rows = train(1e100, steps=10)
        self.assertEqual(len(rows), 2)

    def test_invalid_parameters(self):
        for rate in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                train(rate)
        for options in [{"steps": 0}, {"steps": 1001}, {"x": 0}, {"target": float("inf")}]:
            with self.assertRaises(ValueError):
                train(0.01, **options)


if __name__ == "__main__":
    unittest.main()
