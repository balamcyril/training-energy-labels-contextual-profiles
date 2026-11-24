"""Independent boundary and selection checks for protocol 0.2."""
import unittest
from analyse_zeus_profiles import simulate
from validate_revision import strict_admissible, donor_estimate, method_sets, frontier


class RevisionTest(unittest.TestCase):
    params = dict(on_s=7200, off_s=1800, horizon_s=86400,
                  checkpoint_interval_work_s=900, checkpoint_cost_s=5, restart_cost_s=30)

    def test_shifted_start_restart_time_and_energy(self):
        r = simulate(5000, 100, False, **self.params, first_on_s=3600)
        self.assertEqual(r["completed"], 1)
        self.assertEqual(r["time_s"], 3600+1800+30+5000)
        self.assertEqual(r["active_s"], 3600+30+5000)
        self.assertEqual(r["lost_work_s"], 3600)
        self.assertAlmostEqual(r["energy_gpu_kwh"], (3600+30+5000)*100/3_600_000)

    def test_shifted_start_checkpoint_accounting(self):
        r = simulate(5000, 100, True, **self.params, first_on_s=3600)
        self.assertEqual(r["time_s"], 7740)
        self.assertEqual(r["active_s"], 5940)
        self.assertEqual(r["lost_work_s"], 885)
        self.assertEqual(r["saves"], 5)

    def test_shifted_start_exact_cutoff_is_success(self):
        r = simulate(3600, 100, False, **self.params, first_on_s=3600)
        self.assertEqual((r["completed"], r["interruptions"], r["time_s"]), (1,0,3600))

    def test_partial_save_in_first_window_is_lost(self):
        r = simulate(1000, 100, True, **self.params, first_on_s=902)
        self.assertEqual(r["time_s"], 3737)
        self.assertEqual(r["active_s"], 1937)
        self.assertEqual(r["lost_work_s"], 900)
        self.assertEqual(r["saves"], 1)

    def test_full_first_window_preserves_legacy_simulation(self):
        for checkpoint in (False, True):
            for work in (700, 7200, 9000, 20000):
                self.assertEqual(simulate(work, 100, checkpoint, **self.params),
                    simulate(work, 100, checkpoint, **self.params, first_on_s=7200))

    def test_mean_budget_is_not_all_path_budget(self):
        paths = [dict(completed=1,time_s=100,energy_gpu_kwh=e) for e in (.2,.6)]
        self.assertFalse(strict_admissible(paths, 100, .4))
        self.assertTrue(strict_admissible(paths, 100, .6))
        self.assertFalse(strict_admissible(paths, 99))
        paths[1]["completed"] = 0
        self.assertFalse(strict_admissible(paths, 100, .6))
        self.assertFalse(strict_admissible([], 100))

    def test_estimators_exclude_hidden_record(self):
        rows = [dict(configuration=str(i),time_s=v) for i,v in enumerate((99999,1,3,20))]
        self.assertEqual(donor_estimate(rows,"0","time_s","mean"), (8,3))
        self.assertEqual(donor_estimate(rows,"0","time_s","median"), (3,3))
        rows[0]["time_s"] = -99999
        self.assertEqual(donor_estimate(rows,"0","time_s","median"), (3,3))

    def test_comparators_share_constraints_and_keep_ties(self):
        ref = [dict(configuration=i,label=l,eligible=1) for i,l in (("a","A"),("b","B"),("c","B"))]
        context = [dict(configuration="a",label="A",eligible=0),
                   dict(configuration="b",label="C",eligible=1),
                   dict(configuration="c",label="B",eligible=1)]
        self.assertEqual(method_sets(ref, context), {"M1":["a"],"M2":["b","c"],"M3":["c"]})
        for r in context: r["eligible"] = 0
        self.assertEqual(method_sets(ref, context), {"M1":["a"],"M2":[],"M3":[]})

    def test_pareto_equal_costs_are_not_strict_dominance(self):
        rows = [dict(configuration=i,time_s=t,energy_gpu_kwh=e,carbon_g=2*e)
                for i,t,e in (("a",1,2),("b",2,1),("c",2,2),("d",1,2))]
        self.assertEqual([r["configuration"] for r in frontier(rows)], ["a","b","d"])


if __name__ == "__main__":
    unittest.main()
