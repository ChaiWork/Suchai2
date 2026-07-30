#!/usr/bin/env python3
"""
Utility script to simulate a battle with your trained agent vs an opponent (e.g. Dragapult ex)
and export a visualizer JSON file (vis.json) for viewing in visualizer.html.
"""

import sys
import os
import json
import random
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import agent as my_agent
from train import rule_based_opponent_agent
from cg.game import battle_start, battle_finish, battle_select, visualize_data
from test_agent import load_deck

def run_match_and_export(opponent_name="Rulebasedmodel_Dragapult", output_json="vis.json"):
    print("==================================================================")
    print(f"       RUNNING MATCH: MY AGENT vs {opponent_name}                 ")
    print("==================================================================")
    
    my_deck = load_deck("deck.csv")
    opp_deck = load_deck(opponent_name)
    
    print("Initializing battle simulation...")
    obs_dict, _ = battle_start(opp_deck, my_deck)
    obs_log = [""]
    action_log = [None]
    step_count = 0
    
    while True:
        curr = obs_dict.get("current", {})
        if curr and curr.get("result", -1) >= 0:
            break
            
        curr_player = curr.get("yourIndex", 0)
        
        # 1. Obtain action from agent
        if curr_player == 1:  # Player 1 is My Agent (Suchai)
            try:
                selected = my_agent(obs_dict)
            except Exception as e:
                selected = [0]
        else:  # Player 0 is Opponent
            try:
                selected = rule_based_opponent_agent(opponent_name, obs_dict)
            except Exception as e:
                selected = [0]
                
        if not isinstance(selected, list):
            selected = [selected]

        options = obs_dict.get("select", {}).get("option", [])
        num_opts = len(options) if options else 1
        if not selected or len(selected) == 0 or selected[0] < 0 or selected[0] >= num_opts:
            selected = [0]

        obs_dict.pop("search_begin_input", None)
        obs_log.append(obs_dict)
        action_log.append(selected)
        
        try:
            obs_dict = battle_select(selected)
        except Exception:
            opts = obs_dict.get("select", {}).get("option", [])
            advanced = False
            for opt_i in range(len(opts)):
                try:
                    obs_dict = battle_select([opt_i])
                    advanced = True
                    break
                except Exception:
                    continue
            if not advanced:
                break
                
        step_count += 1
        if step_count % 30 == 0:
            print(f"Step {step_count}: Turn {curr.get('turn', 0)}...")

    result = obs_dict.get("current", {}).get("result", -1)
    winner_str = "MY AGENT WINS! 🏆" if result == 1 else ("OPPONENT WINS ❌" if result == 0 else "DRAW 🤝")
    print(f"\nGame Finished in {step_count} actions! Result: {winner_str}")

    print("Generating visualizer payload data...")
    try:
        vis = json.loads(visualize_data())
        for i in range(min(len(vis), len(obs_log))):
            vis[i]["obs"] = obs_log[i]
            vis[i]["action"] = [action_log[i], action_log[i]]
            
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(vis, f)
        print(f"SUCCESS: Saved battle replay to {output_json}!")
    except Exception as e:
        print(f"Visualizer extraction note: {e}")

    battle_finish()
    print("\nNext step: Open visualizer.html in your browser and select vis.json to watch!")

def run_batch_matches(opponent_name="Rulebasedmodel_Dragapult", num_matches=10, out_dir="replays"):
    os.makedirs(out_dir, exist_ok=True)
    print("==================================================================")
    print(f"       RUNNING BATCH MATCHES ({num_matches} Games): MY AGENT vs {opponent_name}")
    print("==================================================================")
    print(f"Replay output directory: {os.path.abspath(out_dir)}\n")

    results = []
    wins = 0
    losses = 0
    draws = 0

    for m_i in range(1, num_matches + 1):
        my_deck = load_deck("deck.csv")
        opp_deck = load_deck(opponent_name)

        obs_dict, _ = battle_start(opp_deck, my_deck)
        obs_log = [""]
        action_log = [None]
        step_count = 0

        while True:
            curr = obs_dict.get("current", {})
            if curr and curr.get("result", -1) >= 0:
                break

            curr_player = curr.get("yourIndex", 0)

            if curr_player == 1:
                try:
                    selected = my_agent(obs_dict)
                except Exception:
                    selected = [0]
            else:
                try:
                    selected = rule_based_opponent_agent(opponent_name, obs_dict)
                except Exception:
                    selected = [0]

            if not isinstance(selected, list):
                selected = [selected]

            options = obs_dict.get("select", {}).get("option", [])
            num_opts = len(options) if options else 1
            if not selected or len(selected) == 0 or selected[0] < 0 or selected[0] >= num_opts:
                selected = [0]

            obs_dict.pop("search_begin_input", None)
            obs_log.append(obs_dict)
            action_log.append(selected)

            try:
                obs_dict = battle_select(selected)
            except Exception:
                opts = obs_dict.get("select", {}).get("option", [])
                advanced = False
                for opt_i in range(len(opts)):
                    try:
                        obs_dict = battle_select([opt_i])
                        advanced = True
                        break
                    except Exception:
                        continue
                if not advanced:
                    break

            step_count += 1

        res = obs_dict.get("current", {}).get("result", -1)
        res_str = "WIN" if res == 1 else ("LOSS" if res == 0 else "DRAW")
        if res == 1:
            wins += 1
        elif res == 0:
            losses += 1
        else:
            draws += 1

        out_file = os.path.join(out_dir, f"match_{m_i:02d}_{res_str.lower()}.json")
        try:
            vis = json.loads(visualize_data())
            for i in range(min(len(vis), len(obs_log))):
                vis[i]["obs"] = obs_log[i]
                vis[i]["action"] = [action_log[i], action_log[i]]
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(vis, f)
            # Also keep default vis.json updated with latest replay
            with open("vis.json", "w", encoding="utf-8") as f:
                json.dump(vis, f)
        except Exception:
            pass

        battle_finish()
        results.append((m_i, res_str, step_count, out_file))
        print(f"Match {m_i:02d}/{num_matches:02d} | Result: {res_str} in {step_count} actions -> Saved {out_file}")

    wr = (wins / max(1, num_matches)) * 100.0
    print("\n==================================================================")
    print("                    BATCH MATCH SUMMARY REPORT                    ")
    print("==================================================================")
    print(f"Opponent:           {opponent_name}")
    print(f"Total Games Played: {num_matches}")
    print(f"Final Score:        {wins} Wins / {losses} Losses / {draws} Draws")
    print(f"Win Rate:           {wr:.1f}%")
    print(f"All Replays Folder: {os.path.abspath(out_dir)}")
    print("==================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run your AI Agent against a specified opponent and export replay JSON files.")
    parser.add_argument("--opp", type=str, default="Rulebasedmodel_Dragapult", help="Opponent name (e.g. Rulebasedmodel_Dragapult, DRAGOPULT, Rulebasedmodel_Starmie)")
    parser.add_argument("--num-matches", "-n", type=int, default=1, help="Number of matches to generate (e.g. 10)")
    parser.add_argument("--out-dir", type=str, default="replays", help="Output directory folder for batch replay JSON files")
    parser.add_argument("--out", type=str, default="vis.json", help="Output JSON file name for single match mode")
    args = parser.parse_args()

    if args.num_matches > 1:
        run_batch_matches(opponent_name=args.opp, num_matches=args.num_matches, out_dir=args.out_dir)
    else:
        run_match_and_export(opponent_name=args.opp, output_json=args.out)
