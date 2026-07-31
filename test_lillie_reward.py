"""
Unit tests for Lillie's Clefairy ex + Togekiss Strategic Reward Engine.
"""
import unittest
from src.reward_system.lillie_reward import (
    calculate_lillie_strategic_reward,
    POKE_PAD_ID,
    WONDROUS_PATCH_ID,
    ACCOMPANYING_FLUTE_ID,
    LILLIES_PEARL_ID,
    CLEFAIRY_EX_ID,
)


class TestLillieReward(unittest.TestCase):

    def test_poke_pad_reward(self):
        """Test that playing Poké Pad gives positive strategic reward (+0.15)."""
        pre = {"played_card_id": POKE_PAD_ID}
        post = {"bench_size": 2}
        reward = calculate_lillie_strategic_reward(
            pre=pre,
            post=post,
            action_type=7,  # PLAY
            step_idx=1,
            went_second=False,
            player_idx=0,
            opponent_name="Rulebasedmodel_Mewtwo"
        )
        # Expected: +0.15 (item play reward) + 0.05 (hand card play bonus) = +0.20 (bounded <= 0.25)
        self.assertGreater(reward, 0.0, "Poké Pad should yield a positive reward")
        self.assertAlmostEqual(reward, 0.20, places=4)

    def test_wondrous_patch_reward(self):
        """Test Wondrous Patch play action returns enhanced +0.25 reward (bounded to +0.25)."""
        pre = {"played_card_id": WONDROUS_PATCH_ID}
        post = {"bench_size": 2}
        reward = calculate_lillie_strategic_reward(
            pre=pre,
            post=post,
            action_type=7,  # PLAY
            step_idx=1,
            went_second=False,
            player_idx=0,
            opponent_name="Rulebasedmodel_Mewtwo"
        )
        self.assertAlmostEqual(reward, 0.25, places=4)

    def test_tank_matchup_flute_and_pearl_rewards(self):
        """Test Accompanying Flute and Lillie's Pearl give enhanced bonuses against heavy tanks."""
        pre = {"played_card_id": ACCOMPANYING_FLUTE_ID}
        post = {"bench_size": 2}
        reward_tank = calculate_lillie_strategic_reward(
            pre=pre,
            post=post,
            action_type=7,  # PLAY
            step_idx=1,
            went_second=False,
            player_idx=0,
            opponent_name="Rulebasedmodel_Abomasnow"
        )
        self.assertAlmostEqual(reward_tank, 0.25, places=4)

    def test_dragapult_pre_evolution_ko_reward(self):
        """Test knocking out Dreepy/Drakloak against Dragapult gives target KO reward."""
        pre = {"opp_pokemon": 3, "opp_bench_ids": [336, 101]}  # 336 = Dreepy
        post = {"opp_pokemon": 2, "opp_bench_ids": [101]}
        reward = calculate_lillie_strategic_reward(
            pre=pre,
            post=post,
            action_type=1,  # ATTACK / OTHER
            step_idx=1,
            went_second=False,
            player_idx=0,
            opponent_name="Rulebasedmodel_Dragapult"
        )
        self.assertGreaterEqual(reward, 0.25)

    def test_lucario_2prize_ex_penalty(self):
        """Test playing 2-prize ex Pokemon early against Lucario gives a penalty."""
        pre = {"played_card_id": CLEFAIRY_EX_ID}
        post = {"bench_size": 1, "turn": 2}
        reward = calculate_lillie_strategic_reward(
            pre=pre,
            post=post,
            action_type=7,  # PLAY
            step_idx=1,
            went_second=False,
            player_idx=0,
            opponent_name="Rulebasedmodel_Lucario"
        )
        self.assertLess(reward, 0.0, "Playing 2-prize ex early vs Lucario should yield a negative reward")


if __name__ == "__main__":
    unittest.main()
