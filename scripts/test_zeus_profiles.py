"""Tests of physical accounting, missing-data isolation and grade boundaries."""
import math
import unittest
from analyse_zeus_profiles import simulate, quantile, grade_record, imputed_value


class ProfilesTest(unittest.TestCase):
    params = dict(on_s=7200, off_s=1800, horizon_s=86400,
                  checkpoint_interval_work_s=900, checkpoint_cost_s=5, restart_cost_s=30)

    def run_case(self, work, checkpoint=False, **changes):
        return simulate(work, 100, checkpoint, **dict(self.params, **changes))

    def test_no_cutoff_energy_units(self):
        r = self.run_case(3600)
        self.assertEqual(r["completed"], 1)
        self.assertAlmostEqual(r["energy_gpu_kwh"], .1)
        self.assertEqual(r["time_s"], 3600)

    def test_no_recovery_cannot_finish_long_job(self):
        r = self.run_case(7500)
        self.assertEqual(r["completed"], 0)
        self.assertLess(r["active_s"], r["time_s"])

    def test_recovery_finishes_without_charging_outage_energy(self):
        r = self.run_case(7500, True)
        self.assertEqual(r["completed"], 1)
        self.assertEqual(r["interruptions"], 1)
        self.assertAlmostEqual(r["time_s"]-r["active_s"], 1800)
        self.assertGreater(r["active_s"], 7500)
        self.assertAlmostEqual(r["energy_gpu_kwh"], r["active_s"]*100/3600000)

    def test_saves_are_incremental_and_no_final_save_needed(self):
        r = self.run_case(1800, True, on_s=math.inf)
        self.assertEqual(r["saves"], 1)
        self.assertEqual(r["time_s"], 1805)

    def test_completion_at_cutoff(self):
        r = self.run_case(7200)
        self.assertEqual(r["completed"], 1)
        self.assertEqual(r["interruptions"], 0)

    def test_incomplete_save_does_not_preserve_progress(self):
        r = self.run_case(1000, True, on_s=902, horizon_s=5000)
        self.assertEqual(r["completed"], 0)
        self.assertEqual(r["saves"], 0)
        self.assertGreater(r["lost_work_s"], 0)

    def test_masked_value_cannot_influence_donor_mean(self):
        rows = [dict(configuration="a", time_s=999999), dict(configuration="b", time_s=10),
                dict(configuration="c", time_s=20)]
        self.assertEqual(imputed_value(rows, "a", "time_s"), (15, 2))
        rows[0]["time_s"] = -999999
        self.assertEqual(imputed_value(rows, "a", "time_s"), (15, 2))

    def test_quantile_and_threshold_equality(self):
        self.assertAlmostEqual(quantile([0, 10, 20, 30, 40], .2), 8)
        thresholds = {k:[1,2,3,4] for k in ("time_s", "energy_gpu_kwh", "carbon_g")}
        record = dict(time_s=1, energy_gpu_kwh=2, carbon_g=2)
        # score .5, half-up -> B; equality belongs to lower-cost grade.
        result = grade_record(record, thresholds, [2,1,1])
        self.assertEqual(result["grade_T"], 0)
        self.assertEqual(result["label"], "B")


if __name__ == "__main__":
    unittest.main()
