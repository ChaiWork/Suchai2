import unittest
from src.reward_system.base_reward import calculate_base_strategic_reward_components
from src.reward_system.mewtwo_reward import calculate_mewtwo_strategic_reward
from src.expert_system.energy_evaluator import _dynamic_role_weight


class TestRewardFixes(unittest.TestCase):

    # -----------------------------------------------------------------------
    # 1. RETREAT: V(S') - V(S)
    # -----------------------------------------------------------------------

    def test_retreat_correct_play_positive(self):
        """Retreating a dying Articuno (20 HP, 1 energy) for a semi-ready Mewtwo ex (2 energy)
        should produce POSITIVE r_retreat_eff: readiness gain outweighs energy cost (hp < 40
        so energy_cost = 0.0, only survival + readiness delta applies)."""
        pre = {
            "active_id": 414, "active_energy": 1, "active_min_req_energy": 2,
            "active_hp": 20, "active_max_hp": 120, "my_active_energy": 1, "turn": 4,
        }
        post = {
            "active_id": 431, "active_energy": 2, "active_min_req_energy": 3,
            "active_hp": 280, "active_max_hp": 280, "my_active_energy": 2, "turn": 4,
        }
        comp = calculate_base_strategic_reward_components(pre, post, 12, 10, False, 0, "")
        self.assertGreater(
            comp["r_retreat_eff"], 0.0,
            f"Retreating dying Articuno for semi-ready Mewtwo should be POSITIVE, got {comp['r_retreat_eff']}"
        )

    def test_retreat_wasteful_healthy_penalized(self):
        """Retreating a healthy Articuno (120 HP, 1 energy) for a 0-energy Mewtwo ex should be
        NEGATIVE: post_readiness (0.0) < pre_readiness (0.5) and energy_cost applies (hp >= 80)."""
        pre = {
            "active_id": 414, "active_energy": 1, "active_min_req_energy": 2,
            "active_hp": 120, "active_max_hp": 120, "my_active_energy": 1, "turn": 3,
        }
        post = {
            "active_id": 431, "active_energy": 0, "active_min_req_energy": 3,
            "active_hp": 280, "active_max_hp": 280, "my_active_energy": 0, "turn": 3,
        }
        comp = calculate_base_strategic_reward_components(pre, post, 12, 10, False, 0, "")
        self.assertLess(
            comp["r_retreat_eff"], 0.0,
            f"Retreating healthy Articuno for 0-energy Mewtwo should be NEGATIVE, got {comp['r_retreat_eff']}"
        )

    def test_mewtwo_no_double_retreat_penalty(self):
        """mewtwo_reward.py should NOT flat-penalize an Articuno retreat (no double-counting).
        Only the healthy-powered-Mewtwo-ex switch-out guard remains in mewtwo_reward."""
        pre  = {"active_id": 414, "active_hp": 120, "my_active_energy": 1, "bench_ids": [431], "bench_energies": [2], "turn": 3}
        post = {"active_id": 431, "active_hp": 280, "my_active_energy": 2, "bench_ids": [414], "bench_energies": [0], "turn": 3}
        r = calculate_mewtwo_strategic_reward(pre, post, 12, 10, False, 0, "")
        self.assertGreaterEqual(r, -0.05, f"No flat retreat penalty for Articuno retreat, got {r}")

    # -----------------------------------------------------------------------
    # 2. MISSED ATTACK: gate on has_attack_option
    # -----------------------------------------------------------------------

    def test_missed_attack_with_legal_option(self):
        """has_attack_option=True + ready attacker: -0.25 penalty fires."""
        pre  = {"active_id": 431, "active_energy": 3, "active_min_req_energy": 3, "opp_active_hp": 280, "has_attack_option": True}
        post = {"active_id": 431, "active_energy": 3, "opp_active_hp": 280}
        comp = calculate_base_strategic_reward_components(pre, post, 0, 10, False, 0, "")
        self.assertEqual(comp["r_missed_attack"], -0.25)

    def test_missed_attack_no_legal_option(self):
        """has_attack_option=False (e.g. Mimikyu immunity): no missed attack penalty."""
        pre  = {"active_id": 431, "active_energy": 3, "active_min_req_energy": 3, "opp_active_hp": 280, "has_attack_option": False}
        post = {"active_id": 431, "active_energy": 3, "opp_active_hp": 280}
        comp = calculate_base_strategic_reward_components(pre, post, 0, 10, False, 0, "")
        self.assertEqual(comp["r_missed_attack"], 0.0,
                         f"No penalty without legal attack option, got {comp['r_missed_attack']}")

    # -----------------------------------------------------------------------
    # 3. STADIUM: value delta, not flat reward
    # -----------------------------------------------------------------------

    def test_stadium_upgrade_positive(self):
        """Playing TR Factory (0.10) over empty stadium (-1): delta = +0.10."""
        pre  = {"stadium_id": -1, "played_card_id": 1257}
        post = {"stadium_id": 1257}
        comp = calculate_base_strategic_reward_components(pre, post, 7, 10, False, 0, "")
        self.assertGreater(comp["r_stadium_value"], 0.0)

    def test_stadium_downgrade_negative(self):
        """Playing Battle Cage (0.08) over TR Factory (0.10): delta = -0.02, negative."""
        pre  = {"stadium_id": 1257, "played_card_id": 1264}
        post = {"stadium_id": 1264}
        comp = calculate_base_strategic_reward_components(pre, post, 7, 10, False, 0, "")
        self.assertLess(comp["r_stadium_value"], 0.0,
                        f"Playing Battle Cage over TR Factory should be negative, got {comp['r_stadium_value']}")

    # -----------------------------------------------------------------------
    # 4. SEARCH QUALITY: tempo-only = 0.03
    # -----------------------------------------------------------------------

    def test_search_quality_halved(self):
        """r_search_quality should be 0.03 (tempo-only) after redesign."""
        pre  = {"played_card_id": 1134}  # Transceiver
        post = {}
        comp = calculate_base_strategic_reward_components(pre, post, 7, 10, False, 0, "")
        self.assertAlmostEqual(comp["r_search_quality"], 0.03, places=5)

    # -----------------------------------------------------------------------
    # 5. DYNAMIC ARTICUNO ROLE WEIGHT
    # -----------------------------------------------------------------------

    def test_articuno_weight_active_no_mewtwo(self):
        """Articuno is active, Mewtwo not on bench: dynamic weight = 0.40."""
        w = _dynamic_role_weight(414, active_id=414, bench_ids={400, 434}, prizes=4, opp_prizes=4)
        self.assertAlmostEqual(w, 0.40, places=5)

    def test_articuno_weight_close_game(self):
        """Articuno is active, Mewtwo benched, prizes <= 2: weight = 0.25."""
        w = _dynamic_role_weight(414, active_id=414, bench_ids={431, 400}, prizes=2, opp_prizes=1)
        self.assertAlmostEqual(w, 0.25, places=5)

    def test_articuno_weight_bench(self):
        """Articuno on bench (Mewtwo active): static weight = 0.05."""
        w = _dynamic_role_weight(414, active_id=431, bench_ids={414}, prizes=4, opp_prizes=4)
        self.assertAlmostEqual(w, 0.05, places=5)

    # -----------------------------------------------------------------------
    # 6. ACTION CONVERSION: still gates on board development
    # -----------------------------------------------------------------------

    def test_action_conversion_not_unconditional(self):
        """Action type 7 without board development should NOT yield r_action_conv > 0."""
        pre  = {"active_id": 414, "active_energy": 0, "bench_size": 1, "energy": 0}
        post = {"active_id": 414, "active_energy": 0, "bench_size": 1, "energy": 0}
        comp = calculate_base_strategic_reward_components(pre, post, 7, 10, False, 0, "")
        self.assertEqual(comp["r_action_conv"], 0.0)


if __name__ == "__main__":
    unittest.main()
