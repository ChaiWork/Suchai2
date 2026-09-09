"""
Unit Tests for Dedicated Team Rocket Mewtwo ex Reward Engine & Expert Engine.
Ground Truth: Matches 60-card CSV list (deck lama.csv / deck_mewtwo.csv).
"""
import unittest
from src.configs.active_deck import set_active_deck, get_active_deck_name
from src.reward_system.reward_router import calculate_strategic_reward
from src.reward_system.mewtwo_reward import (
    calculate_mewtwo_strategic_reward,
    BASIC_G_ENERGY_ID, TR_ENERGY_ID, TAROUNTULA_ID, SPIDOPS_ID, MEWTWO_EX_ID,
    TRANSCEIVER_ID, BUG_CATCHING_SET_ID, BRAVE_BANGLE_ID, ARIANA_ID, GIOVANNI_ID,
    POKE_PAD_ID, NIGHT_STRETCHER_ID
)
from src.expert_system.expert_router import get_expert_bonus


class TestTeamRocketMewtwoDeckProfile(unittest.TestCase):

    def setUp(self):
        set_active_deck("MEWTWO")

    def test_active_deck_configuration(self):
        """Test that active deck name is set to MEWTWO."""
        self.assertEqual(get_active_deck_name(), "MEWTWO")

    def test_energy_attachment_rewards(self):
        """Test energy attachment reward evaluation for Mewtwo ex."""
        pre = {
            "attached_card_id": BASIC_G_ENERGY_ID,
            "attached_target_id": MEWTWO_EX_ID,
            "active_id": MEWTWO_EX_ID,
            "target_energy": 0,
            "my_active_energy": 0,
            "bench_ids": [],
            "bench_energies": []
        }
        post = {"bench_size": 0}
        debug = []
        reward = calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=8, step_idx=1, went_second=True, player_idx=0, debug_log=debug
        )
        self.assertGreaterEqual(reward, -0.10)
        reasons = [item[0] for item in debug]
        self.assertTrue(any("Energy_Attachment_Eval" in r for r in reasons))

    def test_turn1_going_first_supporter_guard(self):
        """Going first on Turn 1 prevents playing Supporters; playing Ariana should be penalized."""
        pre = {"played_card_id": ARIANA_ID, "turn": 1}
        post = {"bench_size": 2}
        debug = []
        reward = calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=7, step_idx=0, went_second=False, player_idx=0, debug_log=debug
        )
        self.assertLess(reward, 0.0)
        reasons = [item[0] for item in debug]
        self.assertIn("Invalid_Turn1_Going_First_Supporter", reasons)

    def test_wasteful_retreat_energy_discard(self):
        """Wasteful retreat evaluation moved to base_reward V(S')−V(S).
        mewtwo_reward.py NO LONGER emits a flat retreat penalty (was double-counting).
        Verify: no 'Penalty_Wasteful_Retreat' in debug, and reward is >= min bound (no spurious large penalty)."""
        pre = {"active_id": 414, "my_active_energy": 1, "active_hp": 120}
        post = {"active_id": 431, "my_active_energy": 0, "active_hp": 280}
        debug = []
        reward = calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=12, step_idx=3, went_second=True, player_idx=0, debug_log=debug
        )
        reasons = [item[0] for item in debug]
        # Flat retreat penalty removed from mewtwo_reward.py — handled by base_reward V(S')−V(S) instead
        self.assertNotIn("Penalty_Wasteful_Retreat_Discarding_Energy_From_Healthy_Active", reasons,
                         "Flat retreat penalty should no longer fire in mewtwo_reward (moved to base_reward)")
        # Safety guard for switching healthy powered Mewtwo ex out is still present,
        # but here we are retreating Articuno (414), not Mewtwo ex, so NO safety guard fires.
        self.assertGreaterEqual(reward, -0.10, f"No large penalty should fire for Articuno retreat, got {reward}")


    def test_expert_transceiver_prior(self):
        """Test expert guidance bonus for TR Transceiver play."""
        class MockOption:
            type = 7
            cardId = TRANSCEIVER_ID

        obs = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {"hand": [{"id": TRANSCEIVER_ID}], "bench": []},
                    {"hand": [], "bench": []}
                ],
                "turn": 2
            }
        }
        bonus, trigger = get_expert_bonus(obs, MockOption())
        self.assertGreater(bonus, 0.0)


if __name__ == "__main__":
    unittest.main()
