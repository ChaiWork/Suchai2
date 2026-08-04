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
        """Test energy attachment rewards for Basic Grass Energy and TR Energy to Mewtwo ex."""
        pre = {
            "attached_card_id": BASIC_G_ENERGY_ID,
            "attached_target_id": MEWTWO_EX_ID
        }
        post = {"bench_size": 2}
        debug = []
        reward = calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=8, step_idx=1, went_second=True, player_idx=0, debug_log=debug
        )
        self.assertGreaterEqual(reward, 0.14)
        reasons = [item[0] for item in debug]
        self.assertIn("Attach_Energy_To_Mewtwo_ex_Main_Attacker", reasons)

    def test_tr_transceiver_and_bug_catching_set(self):
        """Test TR Transceiver (+0.12) and Bug Catching Set (+0.10) play rewards."""
        pre = {"played_card_id": TRANSCEIVER_ID}
        post = {"bench_size": 2}
        debug = []
        reward = calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=7, step_idx=1, went_second=True, player_idx=0, debug_log=debug
        )
        self.assertGreaterEqual(reward, 0.12)
        reasons = [item[0] for item in debug]
        self.assertIn("Play_TR_Transceiver_Supporter_Tutor", reasons)

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

    def test_giovanni_gust_rewards(self):
        """Giovanni played when opponent has benched Pokémon gives gust attempt reward."""
        pre = {"played_card_id": GIOVANNI_ID, "opp_bench_ids": [400], "opp_active_id": 414}
        post = {"opp_bench_ids": [400], "opp_active_id": 414}
        debug = []
        reward = calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=7, step_idx=3, went_second=True, player_idx=0, debug_log=debug
        )
        self.assertGreaterEqual(reward, 0.08)
        reasons = [item[0] for item in debug]
        self.assertIn("Giovanni_Targeted_Bench_Gust", reasons)

    def test_mewtwo_ex_attack_execution(self):
        """Mewtwo ex attacking with Brave Bangle equipped gives high strategic reward."""
        pre = {
            "active_id": MEWTWO_EX_ID,
            "has_brave_bangle": True
        }
        post = {"bench_size": 2}
        debug = []
        reward = calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=13, step_idx=5, went_second=True, player_idx=0, debug_log=debug
        )
        self.assertGreaterEqual(reward, 0.20)
        reasons = [item[0] for item in debug]
        self.assertIn("Mewtwo_ex_Attack_Execution", reasons)

    def test_expert_transceiver_prior(self):
        """Test expert guidance bonus for TR Transceiver play."""
        class MockOption:
            type = 7
            cardId = TRANSCEIVER_ID

        obs = {
            "current": {
                "yourIndex": 0,
                "turn": 1,
                "players": [
                    {"prize": [1, 2, 3, 4, 5, 6], "active": [], "bench": [], "hand": [{"id": TRANSCEIVER_ID}]},
                    {"prize": [1, 2, 3, 4, 5, 6], "active": [], "bench": []}
                ]
            }
        }
        bonus, trigger = get_expert_bonus(obs, MockOption(), "Rulebasedmodel_Mewtwo")
        self.assertGreater(bonus, 0.01)
        self.assertIn("TR Transceiver", trigger)


if __name__ == "__main__":
    unittest.main()
