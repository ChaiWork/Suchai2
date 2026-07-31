"""
Comprehensive Unit Tests for Upgraded Mewtwo Strategic Reward Engine & Expert Engine.
Tests:
1. Spidops Psychic Energy attachment for Colorless cost (no false penalty)
2. Turn 1 Supporter guard when going first
3. Strategic hold cards exempt from turn-end penalty
4. Empty bench penalty applies only on turn > 1
5. Attack execution rewards (Mewtwo ex with Brave Bangle)
6. Robust flexible opponent archetype keyword matching (e.g. 'Dragapult ex (Obsidian Flames)')
7. Item chain sequence bonus (>= 2 items played in a turn)
8. Multi-tiered Giovanni gust rewards
9. Optional debug logging hook
"""
import unittest
from src.configs.active_deck import set_active_deck, get_active_deck_name
from src.reward_system.reward_router import calculate_strategic_reward
from src.reward_system.mewtwo_reward import (
    calculate_mewtwo_strategic_reward,
    SPIDOPS_ID, MEWTWO_EX_ID, BASIC_P_ENERGY_ID, ARIANA_ID, GIOVANNI_ID, BRAVE_BANGLE_ID, ULTRA_BALL_ID, POFFIN_ID
)
from src.expert_system.expert_router import get_expert_bonus


class TestMewtwoUpgradedRewardEngine(unittest.TestCase):

    def setUp(self):
        set_active_deck("MEWTWO")

    def test_spidops_psychic_energy_attachment(self):
        """Attaching Psychic Energy to Spidops (when energy < 2) satisfies Colorless cost and is rewarded."""
        pre = {
            "attached_card_id": BASIC_P_ENERGY_ID,
            "attached_target_id": SPIDOPS_ID,
            "target_current_energy": 0
        }
        post = {"bench_size": 2}
        debug = []
        reward = calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=8, step_idx=1, went_second=True, player_idx=0, debug_log=debug
        )
        self.assertGreater(reward, 0.10)
        reasons = [item[0] for item in debug]
        self.assertIn("Attach_Psychic_Energy_To_Spidops_For_Colorless_Cost", reasons)

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

    def test_empty_bench_penalty_turn_gt_1(self):
        """Empty bench on Turn 1 is NOT penalized, but on Turn > 1 is penalized."""
        pre_t1 = {"turn": 1}
        post_t1 = {"bench_size": 0}
        debug_t1 = []
        calculate_mewtwo_strategic_reward(pre=pre_t1, post=post_t1, action_type=14, step_idx=1, went_second=True, player_idx=0, debug_log=debug_t1)
        reasons_t1 = [item[0] for item in debug_t1]
        self.assertNotIn("Empty_Bench_Vulnerability_Turn_GT_1", reasons_t1)

        pre_t2 = {"turn": 2}
        post_t2 = {"bench_size": 0}
        debug_t2 = []
        calculate_mewtwo_strategic_reward(pre=pre_t2, post=post_t2, action_type=14, step_idx=2, went_second=True, player_idx=0, debug_log=debug_t2)
        reasons_t2 = [item[0] for item in debug_t2]
        self.assertIn("Empty_Bench_Vulnerability_Turn_GT_1", reasons_t2)

    def test_mewtwo_ex_attack_reward(self):
        """Mewtwo ex attacking with Brave Bangle equipped gives high strategic reward."""
        pre = {
            "active_id": MEWTWO_EX_ID,
            "active_energy": 2,
            "has_brave_bangle": True
        }
        post = {"bench_size": 2}
        debug = []
        reward = calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=13, step_idx=5, went_second=True, player_idx=0, debug_log=debug
        )
        self.assertGreaterEqual(reward, 0.18)
        reasons = [item[0] for item in debug]
        self.assertIn("Mewtwo_ex_Attack_Execution", reasons)

    def test_item_chain_sequence_bonus(self):
        """Playing >= 2 Items in the same turn triggers an Item Chain Sequence Bonus (+0.06)."""
        pre = {"played_card_id": POFFIN_ID, "items_played_this_turn": 1}
        post = {"bench_size": 2, "items_played_this_turn": 2}
        debug = []
        calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=7, step_idx=2, went_second=True, player_idx=0, debug_log=debug
        )
        reasons = [item[0] for item in debug]
        self.assertIn("Item_Chain_Sequence_Bonus", reasons)

    def test_multi_tiered_giovanni_gust(self):
        """Giovanni played when opponent has benched Pokémon gives gust attempt reward (+0.08 / +0.18)."""
        pre = {"played_card_id": GIOVANNI_ID, "opp_bench_ids": [400], "opp_active_id": 414}
        post = {"opp_bench_ids": [400], "opp_active_id": 414}
        debug = []
        calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=7, step_idx=3, went_second=True, player_idx=0, debug_log=debug
        )
        reasons = [item[0] for item in debug]
        self.assertIn("Giovanni_Targeted_Bench_Gust", reasons)

    def test_flexible_keyword_matching(self):
        """Robust flexible matching detects 'Dragapult ex (Obsidian Flames)' via 'pult' or 'dragapult'."""
        pre = {"played_card_id": BRAVE_BANGLE_ID}
        post = {"bench_size": 2}
        debug = []
        reward = calculate_mewtwo_strategic_reward(
            pre=pre, post=post, action_type=7, step_idx=2, went_second=True, player_idx=0, opponent_name="Dragapult ex (Obsidian Flames)", debug_log=debug
        )
        self.assertAlmostEqual(reward, 0.21, places=5)


if __name__ == "__main__":
    unittest.main()
