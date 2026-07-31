"""
Unit tests for Marnie's Grimmsnarl ex Deck Profile (Reward & Expert Engines).
"""
import unittest
from src.configs.active_deck import set_active_deck, get_active_deck_name
from src.reward_system.reward_router import calculate_strategic_reward
from src.expert_system.expert_router import get_expert_bonus
from src.reward_system.grimmsnarl_reward import (
    IMPIDIMP_ID, GRIMMSNARL_EX_ID, MUNKIDORI_ID, SPIKEMUTH_GYM_ID, RARE_CANDY_ID, POFFIN_ID
)


class TestGrimmsnarlDeckProfile(unittest.TestCase):

    def setUp(self):
        set_active_deck("GRIMMSNARL")

    def test_active_deck_configuration(self):
        """Test that active deck name is GRIMMSNARL."""
        self.assertEqual(get_active_deck_name(), "GRIMMSNARL")

    def test_spikemuth_gym_reward(self):
        """Test playing Spikemuth Gym gives positive strategic reward (+0.22)."""
        pre = {"played_card_id": SPIKEMUTH_GYM_ID}
        post = {"bench_size": 2}
        reward = calculate_strategic_reward(
            pre=pre,
            post=post,
            action_type=7,  # PLAY
            step_idx=1,
            went_second=False,
            player_idx=0,
            opponent_name="Rulebasedmodel_Mewtwo"
        )
        self.assertGreater(reward, 0.20)

    def test_rare_candy_grimmsnarl_reward(self):
        """Test Rare Candy evolution gives high strategic reward."""
        pre = {"played_card_id": RARE_CANDY_ID}
        post = {"bench_size": 2}
        reward = calculate_strategic_reward(
            pre=pre,
            post=post,
            action_type=7,  # PLAY
            step_idx=1,
            went_second=False,
            player_idx=0,
            opponent_name="Rulebasedmodel_Mewtwo"
        )
        self.assertGreaterEqual(reward, 0.25)

    def test_expert_poffin_search_bonus(self):
        """Test expert guidance bonus for early Poffin play."""
        class MockOption:
            type = 7
            cardId = POFFIN_ID

        obs = {
            "current": {
                "yourIndex": 0,
                "turn": 1,
                "players": [
                    {"prize": [1, 2, 3, 4, 5, 6], "active": [], "bench": [], "hand": [{"id": POFFIN_ID}]},
                    {"prize": [1, 2, 3, 4, 5, 6], "active": [], "bench": []}
                ]
            }
        }
        bonus, trigger = get_expert_bonus(obs, MockOption(), "Rulebasedmodel_Mewtwo")
        self.assertGreater(bonus, 0.0)
        self.assertIn("Poffin", trigger)


if __name__ == "__main__":
    unittest.main()
