import unittest
from src.expert_system.search_strategy import (
    choose_best_search_target,
    score_search_target,
    BATTLE_CAGE_ID,
    SPIDOPS_ID,
    MEWTWO_EX_ID,
    GIOVANNI_ID,
    ARIANA_ID,
)

class TestSearchStrategy(unittest.TestCase):
    def setUp(self):
        self.mock_obs = {
            "current": {
                "turn": 1,
                "yourIndex": 0,
                "players": [
                    {
                        "prize": [1, 2, 3, 4, 5, 6],
                        "active": [{"cardId": 400}],
                        "bench": [],
                        "hand": [{"cardId": 1216}]
                    },
                    {
                        "prize": [1, 2, 3, 4, 5, 6],
                        "active": [{"cardId": 121}],
                        "bench": []
                    }
                ]
            }
        }

    def test_battle_cage_search_vs_dragapult(self):
        candidates = [401, 1086, BATTLE_CAGE_ID]
        best_cid, score = choose_best_search_target(
            self.mock_obs, candidate_card_ids=candidates, opponent_name="Rulebasedmodel_Dragapult"
        )
        self.assertEqual(best_cid, BATTLE_CAGE_ID)
        self.assertGreater(score, 0.0)

    def test_spidops_combo_completion(self):
        candidates = [SPIDOPS_ID, 1152, 1227]
        best_cid, score = choose_best_search_target(
            self.mock_obs, candidate_card_ids=candidates, opponent_name="Generic_Opponent"
        )
        self.assertEqual(best_cid, SPIDOPS_ID)
        self.assertGreater(score, 0.0)

    def test_mewtwo_ex_tutor_priority(self):
        candidates = [MEWTWO_EX_ID, 1152, 1227]
        best_cid, score = choose_best_search_target(
            self.mock_obs, candidate_card_ids=candidates, opponent_name="Generic_Opponent"
        )
        self.assertEqual(best_cid, MEWTWO_EX_ID)
        self.assertGreater(score, 0.0)

if __name__ == "__main__":
    unittest.main()
