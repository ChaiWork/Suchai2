import os
import sys
import json
import time
import argparse
from collections import Counter

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("src")
sys.path.append("cg")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from main import agent as my_agent
from train import rule_based_opponent_agent, load_all_decks
from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_observation_class, all_card_data

card_db = {c.cardId: c.name for c in all_card_data()}

EASY_OPPONENTS = [
    "Rulebasedmodel",
    "Rulebasedmodel_Mewtwo_Easy",
    "Rulebasedmodel_Mewtwo",
    "Rulebasedmodel_Abomasnow",
    "Rulebasedmodel_Crustle",
    "Rulebasedmodel_Lopunny",
    "Rulebasedmodel_Typhlosion",
    "Rulebasedmodel_Marnie_Kangaskhan",
    "Rulebasedmodel_Honchkrow"
]

HARD_OPPONENTS = [
    "Rulebasedmodel_Alakazam",
    "Rulebasedmodel_Archaludon",
    "Rulebasedmodel_Dipplin",
    "Rulebasedmodel_Dragapult",
    "Rulebasedmodel_Grimmsnarl",
    "Rulebasedmodel_Iono",
    "Rulebasedmodel_Kangaskhan_Crustle",
    "Rulebasedmodel_Lucario",
    "Rulebasedmodel_Mewtwo_Wobbuffet",
    "Rulebasedmodel_Starmie",
    "Rulebasedmodel_Trevenant",
    "Rulebasedmodel_TR_Mewtwo",
    "Rulebasedmodel_Hydrapple_Ogerpon",
    "Rulebasedmodel_Grimmsnarl_ex",
    "Rulebasedmodel_Garchomp_ex",
    "Rulebasedmodel_Hydrapple_ex",
    "Rulebasedmodel_Ogerpon_ex",
    "Rulebasedmodel_Garchomp_ex_2",
    "Rulebasedmodel_HoOh_HeartGold",
    "Rulebasedmodel_Starmie_ex_2",
    "Rulebasedmodel_Metagross_Grass",
    "Rulebasedmodel_Tyranitar",
    "Rulebasedmodel_Crustle_Stall",
    "DRAGOPULT"
]

ALL_OPPONENTS = EASY_OPPONENTS + HARD_OPPONENTS

def load_deck(deck_name_or_path: str) -> list[int]:
    """Loads a 60-card deck list from file or preset opponent folder."""
    if os.path.exists(deck_name_or_path) and os.path.isfile(deck_name_or_path):
        csv_path = deck_name_or_path
    else:
        csv_path = os.path.join("decks", deck_name_or_path, "deck.csv")
        if not os.path.exists(csv_path):
            # Check easy / hard subfolders
            csv_easy = os.path.join("decks", "easy", deck_name_or_path, "deck.csv")
            csv_hard = os.path.join("decks", "hard", deck_name_or_path, "deck.csv")
            if os.path.exists(csv_easy):
                csv_path = csv_easy
            elif os.path.exists(csv_hard):
                csv_path = csv_hard
            else:
                csv_path = "deck.csv"
            
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        card_ids = [int(line.strip()) for line in f if line.strip()]
    if len(card_ids) < 60:
        card_ids.extend([1] * (60 - len(card_ids)))
    return card_ids[:60]

def run_single_match(opponent_name: str, game_idx: int = 1, save_replay: bool = True):
    """Runs a single test match against an opponent."""
    my_deck = load_deck("deck.csv")
    opp_deck = load_deck(opponent_name)

    start_time = time.time()
    obs, start_data = battle_start(opp_deck, my_deck)
    
    replay_steps = []
    turn_count = 0
    
    while True:
        curr = obs.get("current", {})
        if curr and curr.get("result", -1) >= 0:
            break
            
        curr_player = curr.get("yourIndex", 0)
        turn_count = curr.get("turn", 0)
        
        # 1. Obtain action from agent
        if curr_player == 1:  # Player 1 is My Agent (Suchai)
            try:
                selected = my_agent(obs)
            except Exception as e:
                selected = [0]
        else:  # Player 0 is Opponent
            try:
                selected = rule_based_opponent_agent(opponent_name, obs)
            except Exception as e:
                selected = [0]
                
        if not isinstance(selected, list):
            selected = [selected]

        options = obs.get("select", {}).get("option", [])
        num_opts = len(options) if options else 1
        if not selected or len(selected) == 0:
            selected = [0]
        elif selected[0] < 0 or selected[0] >= num_opts:
            selected = [0]

        step_record = [{
            "action": selected,
            "observation": obs,
            "selected": selected
        }]
        replay_steps.append(step_record)

        try:
            obs = battle_select(selected)
        except Exception:
            opts = obs.get("select", {}).get("option", [])
            advanced = False
            for opt_i in range(len(opts)):
                try:
                    obs = battle_select([opt_i])
                    advanced = True
                    break
                except Exception:
                    continue
            if not advanced:
                break

    result = obs.get("current", {}).get("result", -1)
    battle_finish()
    
    elapsed = time.time() - start_time
    
    if save_replay:
        os.makedirs("replays", exist_ok=True)
        replay_filename = f"replays/test_match_game_{game_idx}_{opponent_name}.json"
        replay_data = {
            "info": {"TeamNames": [opponent_name, "Suchai"]},
            "rewards": [1 if result == 0 else (-1 if result == 1 else 0), 1 if result == 1 else (-1 if result == 0 else 0)],
            "steps": replay_steps
        }
        with open(replay_filename, "w", encoding="utf-8") as rf:
            json.dump(replay_data, rf)
            
    return result, turn_count, elapsed

def run_benchmark(opponents: list[str], games_per_opp: int = 1):
    """Runs a full benchmark across a list of opponent archetypes."""
    print("==================================================================")
    print("        POKÉMON TCG AI AGENT — BENCHMARK EVALUATOR                ")
    print("==================================================================")
    print(f"Total Opponent Archetypes : {len(opponents)}")
    print(f"Games Per Opponent        : {games_per_opp}")
    print("------------------------------------------------------------------")
    
    overall_wins = 0
    overall_losses = 0
    overall_draws = 0
    overall_turns = 0
    
    results_summary = []
    
    for opp_idx, opp_name in enumerate(opponents, 1):
        wins, losses, draws = 0, 0, 0
        total_turns = 0
        print(f"[{opp_idx:2d}/{len(opponents)}] Testing vs {opp_name:<35}...", end="", flush=True)
        
        for g in range(1, games_per_opp + 1):
            res, turns, elapsed = run_single_match(opp_name, game_idx=g)
            total_turns += turns
            if res == 1:
                wins += 1
            elif res == 0:
                losses += 1
            else:
                draws += 1
                
        wr = (wins / games_per_opp) * 100.0
        overall_wins += wins
        overall_losses += losses
        overall_draws += draws
        overall_turns += total_turns
        
        status_icon = "🏆 WIN" if wr == 100 else ("❌ LOSS" if wr == 0 else f"{wr:.0f}%")
        print(f" {status_icon} ({wins}W-{losses}L-{draws}D, {total_turns/games_per_opp:.1f} turns)")
        results_summary.append((opp_name, wins, losses, draws, wr, total_turns / games_per_opp))

    total_games = len(opponents) * games_per_opp
    overall_wr = (overall_wins / total_games) * 100.0 if total_games > 0 else 0
    avg_turns = overall_turns / total_games if total_games > 0 else 0

    print("\n==================================================================")
    print("                 BENCHMARK SUMMARY REPORT                         ")
    print("==================================================================")
    print(f"{'Opponent Archetype':<35} | {'Record':<10} | {'Win Rate':<8} | {'Avg Turns'}")
    print("-" * 70)
    for opp_name, w, l, d, wr, turns in results_summary:
        print(f"{opp_name:<35} | {w}W-{l}L-{d}D     | {wr:5.1f}%   | {turns:.1f}")
    print("==================================================================")
    print(f"TOTAL BENCHMARK SCORE : {overall_wins}/{total_games} Wins ({overall_wr:.1f}% Overall Win Rate)")
    print(f"Average Game Length   : {avg_turns:.1f} Turns")
    print("==================================================================")

def main():
    parser = argparse.ArgumentParser(description="Test your AI Agent locally against rule-based opponents.")
    parser.add_argument("--opp", type=str, default="Rulebasedmodel_Hydrapple_Ogerpon", help="Opponent name, or 'all', 'easy', 'hard'")
    parser.add_argument("--games", type=int, default=1, help="Number of games per opponent")
    parser.add_argument("--all", action="store_true", help="Run benchmark against all easy and hard opponents")
    args = parser.parse_args()

    opp_arg = args.opp.lower()
    
    if args.all or opp_arg == "all":
        run_benchmark(ALL_OPPONENTS, args.games)
    elif opp_arg == "easy":
        run_benchmark(EASY_OPPONENTS, args.games)
    elif opp_arg == "hard":
        run_benchmark(HARD_OPPONENTS, args.games)
    else:
        run_benchmark([args.opp], args.games)

if __name__ == "__main__":
    main()
