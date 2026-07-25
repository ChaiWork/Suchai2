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
from train import rule_based_opponent_agent
from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_observation_class, all_card_data

card_db = {c.cardId: c.name for c in all_card_data()}

def load_deck(deck_name_or_path: str) -> list[int]:
    """Loads a 60-card deck list from file or preset opponent folder."""
    if os.path.exists(deck_name_or_path):
        csv_path = deck_name_or_path
    else:
        csv_path = os.path.join("decks", deck_name_or_path, "deck.csv")
        if not os.path.exists(csv_path):
            csv_path = "deck.csv"
            
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        card_ids = [int(line.strip()) for line in f if line.strip()]
    if len(card_ids) < 60:
        card_ids.extend([1] * (60 - len(card_ids)))
    return card_ids[:60]

def run_local_match(opponent_name: str, num_games: int = 1, save_replay: bool = True):
    """Runs local test matches between your agent and a chosen rule-based opponent."""
    print("==================================================================")
    print("        POKÉMON TCG AI AGENT — LOCAL TEST EVALUATOR               ")
    print("==================================================================")
    print(f"My Agent (Suchai) vs Opponent: {opponent_name}")
    print(f"Total Games to Simulate: {num_games}")
    print("------------------------------------------------------------------")

    my_deck = load_deck("deck.csv")
    opp_deck = load_deck(opponent_name)

    wins = 0
    losses = 0
    draws = 0
    total_turns = 0

    for game_idx in range(1, num_games + 1):
        print(f"\n🎮 [GAME {game_idx}/{num_games}] Starting Match vs {opponent_name}...")
        start_time = time.time()
        
        # Start battle in C++ engine
        obs, start_data = battle_start(opp_deck, my_deck)
        
        # Track replay steps
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
                    print(f"   ⚠️ My Agent Error on step {len(replay_steps)}: {e}")
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

            # Save step data to replay
            step_record = [{
                "action": selected,
                "observation": obs,
                "selected": selected
            }]
            replay_steps.append(step_record)

            # 2. Advance engine turn safely
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
        total_turns += turn_count
        
        # Determine outcome for Player 1 (Suchai)
        # result: 0 -> Player 0 Won, 1 -> Player 1 Won, 2 -> Draw
        if result == 1:
            wins += 1
            outcome_str = "🏆 WIN"
        elif result == 0:
            losses += 1
            outcome_str = "❌ LOSS"
        else:
            draws += 1
            outcome_str = "🤝 DRAW"
            
        print(f"   Finished in {turn_count} turns ({elapsed:.2f}s) | Outcome: {outcome_str}")
        
        # Save replay JSON
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
            print(f"   📁 Saved Match Replay: {replay_filename}")

    win_rate = (wins / num_games) * 100.0 if num_games > 0 else 0.0
    avg_turns = total_turns / num_games if num_games > 0 else 0
    print("\n==================================================================")
    print("                 FINAL EVALUATION SUMMARY                         ")
    print("==================================================================")
    print(f"Opponent Archetype : {opponent_name}")
    print(f"Record             : {wins} Wins | {losses} Losses | {draws} Draws")
    print(f"Win Rate           : {win_rate:.1f}%")
    print(f"Average Game Length: {avg_turns:.1f} Turns")
    print("==================================================================")

def main():
    parser = argparse.ArgumentParser(description="Test your AI Agent locally against any bot archetype.")
    parser.add_argument("--opp", type=str, default="Rulebasedmodel_Hydrapple_Ogerpon", help="Opponent bot archetype name")
    parser.add_argument("--games", type=int, default=1, help="Number of test games to play (default: 1)")
    args = parser.parse_args()
    
    run_local_match(args.opp, args.games)

if __name__ == "__main__":
    main()
