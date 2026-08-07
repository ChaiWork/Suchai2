import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.expert_system.energy_evaluator import evaluate_energy_target, get_required_energy

class TestEnergyOverchargeBlock(unittest.TestCase):

    def test_mewtwo_ex_overcharge_block_when_bench_hungry(self):
        """Mewtwo ex (431) at 3 energy should trigger soft guidance (-0.20) if bench Spidops needs energy."""
        score, reason = evaluate_energy_target(
            target_card_id=431,
            target_current_energy=3,
            energy_card_id=1,  # Basic G Energy
            hand_ids=[1],
            my_prizes=6,
            opp_prizes=6,
            bench_ids=[401],      # Spidops
            bench_energies=[1],   # Spidops has 1/2 energy (hungry!)
            active_id=431,
            active_energy=3,
        )
        self.assertEqual(score, -0.20)
        self.assertIn("Target Already Powered", reason)
        print("  [PASS] Mewtwo ex 3/3 energy soft guidance (-0.20) triggered when bench Spidops needs energy!")

    def test_spidops_overcharge_block_when_active_hungry(self):
        """Spidops (401) on bench at 2 energy should trigger soft guidance (-0.20) if Active Mewtwo ex needs energy."""
        score, reason = evaluate_energy_target(
            target_card_id=401,
            target_current_energy=2,
            energy_card_id=1,
            hand_ids=[1],
            my_prizes=6,
            opp_prizes=6,
            bench_ids=[401],
            bench_energies=[2],   # Spidops has 2/2 energy
            active_id=431,        # Active Mewtwo ex
            active_energy=1,      # Active Mewtwo has 1/3 energy (hungry!)
        )
        self.assertEqual(score, -0.20)
        self.assertIn("Target Already Powered", reason)
        print("  [PASS] Spidops 2/2 energy soft guidance (-0.20) triggered when Active Mewtwo ex needs energy!")

    def test_mimikyu_type_restriction_violation(self):
        """Mimikyu (434) with Basic G Energy (1) should trigger type restriction prior penalty (-0.25)."""
        score, reason = evaluate_energy_target(
            target_card_id=434,
            target_current_energy=0,
            energy_card_id=1,     # Basic G Energy (invalid for Mimikyu)
            hand_ids=[1],
            my_prizes=6,
            opp_prizes=6,
            bench_ids=[431],
            bench_energies=[1],
            active_id=434,
            active_energy=0,
        )
        self.assertEqual(score, -0.25)
        self.assertIn("Energy Type Restriction", reason)
        print("  [PASS] Mimikyu Basic G energy type restriction violation (-0.25) verified!")

if __name__ == "__main__":
    unittest.main()
