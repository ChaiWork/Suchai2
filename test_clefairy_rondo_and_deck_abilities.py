"""
Unit test for Clefairy ex Strategic Full Moon Rondo & Deck.csv Abilities Analysis.
"""
import os
import sys

# Ensure src module can be imported
sys.path.insert(0, os.path.abspath("."))

from src.expert_system.base_expert import extract_board_context
from src.expert_system.lillie_expert import (
    evaluate_lillie_expert_bonus, CLEFAIRY_EX_ID, LATIAS_EX_ID, TOGEKISS_ID, SHAYMIN_ID, PSYDUCK_ID, ACCOMPANYING_FLUTE_ID, TOGEPI_ID
)


def test_full_moon_rondo_bench_scaling():
    """
    Verifies Full Moon Rondo damage calculation incorporates BOTH my bench AND opponent bench.
    """
    mock_obs = {
        "current": {
            "yourIndex": 0,
            "turn": 3,
            "players": [
                {
                    "prize": [1, 2, 3, 4, 5, 6],
                    "active": [{"cardId": CLEFAIRY_EX_ID, "energyCards": [1, 2], "hp": 190}],
                    "bench": [{"cardId": TOGEPI_ID}, {"cardId": TOGEPI_ID}, {"cardId": TOGEPI_ID}],
                },
                {
                    "prize": [1, 2, 3, 4, 5, 6],
                    "active": [{"cardId": 553, "hp": 130}],  # Archaludon (130 HP, Dragon)
                    "bench": [{"cardId": 400}, {"cardId": 401}],
                }
            ]
        }
    }

    ctx = extract_board_context(mock_obs)
    assert ctx["my_bench_count"] == 3, f"Expected my_bench_count=3, got {ctx['my_bench_count']}"
    assert ctx["opp_bench_count"] == 2, f"Expected opp_bench_count=2, got {ctx['opp_bench_count']}"

    # Test option: Play basic Pokemon onto my bench (ID 959 - Togepi)
    mock_option_bench = type("Option", (), {"optionType": 7, "cardId": TOGEPI_ID})()
    bonus, trigger = evaluate_lillie_expert_bonus(mock_obs, mock_option_bench, "Rulebasedmodel_Archaludon")
    print(f"[TEST 1] Benching Basic Pokémon | Bonus: {bonus:.2f} | Trigger: {trigger}")
    assert bonus > 0.0, "Expected positive bonus for benching Basic Pokémon under Clefairy"

    # Test option: Play Accompanying Flute (ID 1091) to force opponent basic onto opponent bench
    mock_option_flute = type("Option", (), {"optionType": 7, "cardId": ACCOMPANYING_FLUTE_ID})()
    bonus_flute, trigger_flute = evaluate_lillie_expert_bonus(mock_obs, mock_option_flute, "Rulebasedmodel_Archaludon")
    print(f"[TEST 2] Playing Accompanying Flute | Bonus: {bonus_flute:.2f} | Trigger: {trigger_flute}")
    assert bonus_flute > 0.0, "Expected positive bonus for Accompanying Flute"

    # Test option: Attack with Clefairy ex (OptionType 13 / ATTACK)
    mock_option_attack = type("Option", (), {"optionType": 13, "type": "Attack", "cardId": -1})()
    bonus_attack, trigger_attack = evaluate_lillie_expert_bonus(mock_obs, mock_option_attack, "Rulebasedmodel_Archaludon")
    print(f"[TEST 3] Clefairy Attack Execution | Bonus: {bonus_attack:.2f} | Trigger: {trigger_attack}")
    assert bonus_attack > 0.0, "Expected positive attack bonus"


def test_ability_counter_heuristics():
    """
    Verifies ability counter triggers for Latias ex, Shaymin, Togekiss, and Psyduck.
    """
    mock_obs = {
        "current": {
            "yourIndex": 0,
            "turn": 2,
            "players": [
                {
                    "prize": [1, 2, 3, 4, 5, 6],
                    "active": [{"cardId": TOGEPI_ID, "energyCards": [], "hp": 60}],
                    "bench": [],
                },
                {
                    "prize": [1, 2, 3, 4, 5, 6],
                    "active": [{"cardId": 338, "hp": 320}],  # Dragapult ex (Dragon / Spread)
                    "bench": [{"cardId": 336}],
                }
            ]
        }
    }

    # Test Latias ex (Skyliner) vs Snorlax Stall
    mock_latias_option = type("Option", (), {"optionType": 7, "cardId": LATIAS_EX_ID})()
    bonus_latias, trigger_latias = evaluate_lillie_expert_bonus(mock_obs, mock_latias_option, "Snorlax Stall")
    print(f"[TEST 4] Latias ex Skyliner vs Stall | Bonus: {bonus_latias:.2f} | Trigger: {trigger_latias}")
    assert bonus_latias >= 0.20, "Expected high bonus for Latias ex vs Stall"

    # Test Shaymin (Flower Curtain) vs Dragapult Spread
    mock_shaymin_option = type("Option", (), {"optionType": 7, "cardId": SHAYMIN_ID})()
    bonus_shaymin, trigger_shaymin = evaluate_lillie_expert_bonus(mock_obs, mock_shaymin_option, "Dragapult ex")
    print(f"[TEST 5] Shaymin Flower Curtain vs Spread | Bonus: {bonus_shaymin:.2f} | Trigger: {trigger_shaymin}")
    assert bonus_shaymin >= 0.18, "Expected Shaymin bonus vs Spread"

    # Test Psyduck (Damp) vs Self-KO engine (Electrode ex)
    mock_obs_selfko = {
        "current": {
            "yourIndex": 0,
            "turn": 1,
            "players": [
                {"prize": [1,2,3,4,5,6], "active": [{"cardId": TOGEPI_ID}], "bench": []},
                {"prize": [1,2,3,4,5,6], "active": [{"cardId": 101}], "bench": []} # Voltorb (Self-KO line)
            ]
        }
    }
    mock_psyduck_option = type("Option", (), {"optionType": 7, "cardId": PSYDUCK_ID})()
    bonus_psyduck, trigger_psyduck = evaluate_lillie_expert_bonus(mock_obs_selfko, mock_psyduck_option, "Electrode ex")
    print(f"[TEST 6] Psyduck Damp vs Self-KO Engine | Bonus: {bonus_psyduck:.2f} | Trigger: {trigger_psyduck}")
    assert bonus_psyduck >= 0.14, "Expected Psyduck bonus vs Self-KO"

    print("\n>>> ALL ABILITY COUNTER UNIT TESTS PASSED SUCCESSFULLY! <<<\n")


def test_pre_attack_card_play_precedence():
    """
    Verifies that when hand contains unplayed Supporters/Items, non-ending card plays receive higher prior than immediate attack.
    """
    mock_obs_with_supporter = {
        "current": {
            "yourIndex": 0,
            "turn": 3,
            "players": [
                {
                    "prize": [1, 2, 3, 4, 5, 6],
                    "active": [{"cardId": CLEFAIRY_EX_ID, "energyCards": [1, 2], "hp": 190}],
                    "bench": [{"cardId": TOGEPI_ID}],
                    "hand": [{"cardId": 1227}],  # Lillie's Determination in hand!
                },
                {
                    "prize": [1, 2, 3, 4, 5, 6],
                    "active": [{"cardId": 400, "hp": 220}], # High HP non-KO active
                    "bench": [],
                }
            ]
        }
    }

    # Option A: Play Lillie's Determination Supporter (ID 1227)
    opt_supporter = type("Option", (), {"optionType": 7, "cardId": 1227})()
    bonus_supporter, trig_supporter = evaluate_lillie_expert_bonus(mock_obs_with_supporter, opt_supporter, "Generic")

    # Option B: Attack with Clefairy ex (OptionType 13 / ATTACK)
    opt_attack = type("Option", (), {"optionType": 13, "type": "Attack", "cardId": -1})()
    bonus_attack, trig_attack = evaluate_lillie_expert_bonus(mock_obs_with_supporter, opt_attack, "Generic")

    print(f"[TEST 7] Supporter Play Bonus: {bonus_supporter:.2f} | Attack Bonus (when Supporter in hand): {bonus_attack:.2f} ({trig_attack})")
    assert bonus_supporter > bonus_attack, f"Expected Supporter play ({bonus_supporter}) > Attack ({bonus_attack}) before turn end"
    assert "Postpone" in trig_attack, "Expected Attack postponement trigger label"

    print("\n>>> PRE-ATTACK CARD PLAY PRECEDENCE TEST PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    test_full_moon_rondo_bench_scaling()
    test_ability_counter_heuristics()
    test_pre_attack_card_play_precedence()
