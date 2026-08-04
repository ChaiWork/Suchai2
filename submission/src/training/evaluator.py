import os
import sys
import math
import time
import queue
import random
import importlib


def rule_based_opponent_agent(opponent_name, obs):
    mapping = {
        "Rulebasedmodel": ["easy.main"],
        "Rulebasedmodel_Mewtwo_Easy": ["easy.mewtwo_agent_easy"],
        "Rulebasedmodel_Mewtwo": ["easy.mewtwo_agent"],
        "Rulebasedmodel_Abomasnow": ["easy.abomasnow_agent"],
        "Rulebasedmodel_Crustle": ["easy.crustle_agent"],
        "Rulebasedmodel_Lopunny": ["easy.lopunny_agent"],
        "Rulebasedmodel_Typhlosion": ["easy.typhlosion_agent"],
        "Rulebasedmodel_Marnie_Kangaskhan": ["easy.marnie_kangaskhan_agent"],
        "Rulebasedmodel_Honchkrow": ["easy.honchkrow_agent"],
        
        "Rulebasedmodel_Alakazam": ["hard.alakazam_agent"],
        "Rulebasedmodel_Archaludon": ["hard.archaludon_agent"],
        "Rulebasedmodel_Dipplin": ["hard.dipplin_agent"],
        "Rulebasedmodel_Dragapult": ["hard.dragapult_agent"],
        "Rulebasedmodel_Grimmsnarl": ["hard.grimmsnarl_agent"],
        "Rulebasedmodel_Iono": ["hard.iono_agent"],
        "Rulebasedmodel_Kangaskhan_Crustle": ["hard.kangaskhan_crustle_agent"],
        "Rulebasedmodel_Lucario": ["hard.lucario_agent"],
        "Rulebasedmodel_Mewtwo_Wobbuffet": ["hard.mewtwo_wobbuffet_agent"],
        "Rulebasedmodel_Starmie": ["hard.starmie_agent"],
        "Rulebasedmodel_Trevenant": ["hard.trevenant_agent"],
        "Rulebasedmodel_TR_Mewtwo": ["team_rocket_mewtwo_rule_agent"],
        
        "Rulebasedmodel_Hydrapple_Ogerpon": ["hard.hydrapple_ogerpon_agent"],
        "Rulebasedmodel_Grimmsnarl_ex": ["hard.grimmsnarl_ex_agent"],
        "Rulebasedmodel_Garchomp_ex": ["hard.garchomp_ex_agent"],
        "Rulebasedmodel_Hydrapple_ex": ["hard.hydrapple_ex_agent"],
        "Rulebasedmodel_Ogerpon_ex": ["hard.ogerpon_ex_agent"],
        "Rulebasedmodel_Garchomp_ex_2": ["hard.garchomp_ex_2_agent"],
        "Rulebasedmodel_HoOh_HeartGold": ["hard.hooh_heartgold_agent"],
        "Rulebasedmodel_Starmie_ex_2": ["hard.starmie_ex_2_agent"],
        "Rulebasedmodel_Metagross_Grass": ["hard.metagross_grass_agent"],
        "Rulebasedmodel_Tyranitar": ["hard.tyranitar_sandaconda_agent"],
        "Rulebasedmodel_Crustle_Stall": ["hard.crustle_stall_agent"],
        "DRAGOPULT": ["hard.dragapult_agent"]
    }
    
    modules = mapping.get(opponent_name, [opponent_name])
    for mod_path in modules:
        try:
            mod = importlib.import_module(f"Rulebasedmodel.{mod_path}")
            return mod.agent(obs)
        except Exception:
            pass
            
    raise ValueError(f"Unknown rule-based opponent: {opponent_name}")


def get_active_deck_csv_path() -> str:
    try:
        from src.configs.active_deck import get_active_deck_name
        deck_name = get_active_deck_name()
    except Exception:
        deck_name = os.getenv("ACTIVE_DECK", "MEWTWO").upper()

    if deck_name == "GRIMMSNARL":
        for path in ["deckgrimnai.csv", "deck_grimmsnarl.csv", "deck.csv"]:
            if os.path.exists(path):
                return path
    elif deck_name == "LILLIE":
        for path in ["deck clefairy.csv", "deck_lillie.csv", "deck.csv"]:
            if os.path.exists(path):
                return path
    else:  # MEWTWO
        for path in ["deck_mewtwo_battle_cage.csv", "deck lama.csv", "deck_mewtwo.csv", "deck.csv"]:
            if os.path.exists(path):
                return path
    return "deck.csv"


def load_all_decks():
    """Scans the workspace directories for deck.csv files and loads them.
    
    Returns:
        dict: A dictionary mapping folder name (deck name) to list of card IDs.
    """
    base_path = "decks"
    decks = {}
    
    # 1. Load active root deck (our agent's main deck)
    root_deck_path = get_active_deck_csv_path()
    if os.path.exists(root_deck_path):
        with open(root_deck_path, "r", encoding="utf-8-sig") as f:
            decks["Current (Self)"] = [int(line.strip()) for line in f if line.strip()]
            print(f"Active Deck Loaded for Current (Self): {root_deck_path}")
            
    # 2. Scan decks directory recursively for deck.csv files
    if os.path.exists(base_path):
        for root_dir, dirs, files in os.walk(base_path):
            if "deck.csv" in files:
                deck_file = os.path.join(root_dir, "deck.csv")
                folder_name = os.path.basename(root_dir)
                try:
                    with open(deck_file, "r", encoding="utf-8-sig") as f:
                        card_ids = [int(line.strip()) for line in f if line.strip()]
                        if len(card_ids) == 60:
                            decks[folder_name] = card_ids
                            print(f"Loaded deck '{folder_name}' from {deck_file}")
                        else:
                            print(f"Warning: Deck in {deck_file} has {len(card_ids)} cards (must be 60). Skipping.")
                except Exception as e:
                    print(f"Error loading deck from {deck_file}: {e}")
    return decks


def drain_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def wilson_score_interval(wins, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = wins / total
    denom = 1 + z**2 / total
    mean = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, mean - spread) * 100.0, min(1.0, mean + spread) * 100.0


def sample_league_opponent(active_elo, checkpoints, league_elos, sigma=150.0):
    weights = []
    for path in checkpoints:
        name = os.path.basename(path)
        elo = league_elos.get(name, 1500.0)
        w = math.exp(-((elo - active_elo) ** 2) / (2 * sigma ** 2))
        weights.append(max(0.01, w))
    w_sum = sum(weights)
    probs = [w / w_sum for w in weights]
    return random.choices(checkpoints, weights=probs, k=1)[0]


def draw_sprt_slider(llr, A, B, width=20):
    """Draws a text-based slider representing the Log-Likelihood Ratio (LLR) between A and B."""
    clipped_llr = max(A, min(B, llr))
    pct = (clipped_llr - A) / (B - A) if B > A else 0.5
    pos = int(round(pct * width))
    
    center_pct = -A / (B - A) if B > A else 0.5
    center_pos = int(round(center_pct * width)) if 0 <= center_pct <= 1 else -1
    
    slider_chars = []
    for i in range(width + 1):
        if i == pos:
            slider_chars.append("o")
        elif i == center_pos:
            slider_chars.append("+")
        else:
            slider_chars.append("-")
            
    slider_str = "".join(slider_chars)
    return f"REJECT [{A:.2f}] {slider_str} [{B:+.2f}] ACCEPT"


def run_sprt_evaluation(command_queues, result_queue, num_workers, sample_deck, opponent_decks, test_opponent_names,
                        alpha=0.05, beta=0.1, p0=0.50, p1=0.58, max_eval_games=40):
    drain_queue(result_queue)
    
    A = math.log(beta / (1.0 - alpha))
    B = math.log((1.0 - beta) / alpha)
    
    log_lik_ratio = 0.0
    wins, losses, draws = 0, 0, 0
    total_games = 0
    
    games_sent = 0
    games_received = 0
    active_evals = {}
    
    for w_idx in range(num_workers):
        opp_name = test_opponent_names[games_sent % len(test_opponent_names)]
        opp_deck = opponent_decks[opp_name]
        command_queues[w_idx].put(("EVAL", (sample_deck, opp_deck, opp_name)))
        active_evals[w_idx] = opp_name
        games_sent += 1
        
    print(f"SPRT Evaluation started (Max Games: {max_eval_games}). alpha={alpha}, beta={beta}, p0={p0}, p1={p1}")
    deck_stats = {name: [0, 0, 0] for name in test_opponent_names}
    decision = None
    
    while games_received < max_eval_games:
        msg, data = result_queue.get()
        if msg == "EVAL_COMPLETE":
            w_idx, opp_name, outcome = data
            games_received += 1
            total_games += 1
            
            if outcome == 1.0:
                wins += 1
                deck_stats[opp_name][0] += 1
                log_lik_ratio += math.log(p1 / p0)
            elif outcome == 0.0:
                losses += 1
                deck_stats[opp_name][1] += 1
                log_lik_ratio += math.log((1.0 - p1) / (1.0 - p0))
            elif outcome == 0.5:
                draws += 1
                deck_stats[opp_name][2] += 1
                
            denom = wins + losses
            win_rate = 100.0 * wins / denom if denom > 0 else 0.0
            slider = draw_sprt_slider(log_lik_ratio, A, B, width=20)
            outcome_str = "WIN" if outcome == 1.0 else "LOSS" if outcome == 0.0 else "DRAW"
            print(f"  Game {total_games:03d} | vs {opp_name:<20} | {outcome_str:<4} | WR: {win_rate:5.1f}% ({wins}W-{losses}L-{draws}D) | {slider} (LLR: {log_lik_ratio:+.3f})")
            sys.stdout.flush()
                
            if log_lik_ratio >= B:
                decision = True  # ACCEPT
                break
            elif log_lik_ratio <= A:
                decision = False  # REJECT
                break
                
            if games_sent < max_eval_games:
                opp_name = test_opponent_names[games_sent % len(test_opponent_names)]
                opp_deck = opponent_decks[opp_name]
                command_queues[w_idx].put(("EVAL", (sample_deck, opp_deck, opp_name)))
                active_evals[w_idx] = opp_name
                games_sent += 1
                
    time.sleep(0.5)
    drain_queue(result_queue)
    
    denom = wins + losses
    win_rate = 100.0 * wins / denom if denom > 0 else 0.0
    low_ci, high_ci = wilson_score_interval(wins, denom)
    
    print(f"Overall SPRT Evaluation complete. Total games: {total_games}")
    print(f"  -> Decision: {'ACCEPTED (Model Improved)' if decision else 'REJECTED (Model No Better)' if decision is not None else 'UNDECIDED'}")
    print(f"  -> Win Rate: {win_rate:.1f}% (Wins: {wins}, Losses: {losses}, Draws: {draws})")
    print(f"  -> 95% Wilson Confidence Interval: [{low_ci:.1f}%, {high_ci:.1f}%]")
    
    return decision, win_rate, wins, losses, draws, deck_stats
