import argparse
import csv
import glob
import math
import os
import random
import sys
import torch
import torch.nn
import torch.optim
import time

# Resolve cg-lib path dynamically for Kaggle vs Local environments
try:
    cg_lib_path = glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0]
    sys.path.append(cg_lib_path)
except IndexError:
    pass

# Add nested src/ folder to path for local execution compatibility
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.append(src_dir)

from model import MyModel, SparseVector, LearnInput
from agent import LearnSample, mcts_agent, random_agent
from plot_metrics import plot_metrics

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_observation_class, OptionType


def count_attached_energy(ps):
    count = 0
    if len(ps.active) > 0 and ps.active[0] is not None:
        count += len(ps.active[0].energyCards)
    for poke in ps.bench:
        if poke is not None:
            count += len(poke.energyCards)
    return count


def count_pokemon(ps):
    count = 0
    if len(ps.active) > 0 and ps.active[0] is not None:
        count += 1
    for poke in ps.bench:
        if poke is not None:
            count += 1
    return count


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with FIFO circular eviction and IS weights."""
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = 0.6   # prioritization exponent
        self.pos = 0       # circular buffer write position

    def add(self, sample: LearnSample, td_error: float):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
            self.priorities.append(priority)
        else:
            # Circular FIFO eviction — preserves recency over stale high-priority samples
            self.buffer[self.pos] = sample
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple[list[LearnSample], list[int], list[float]]:
        total = sum(self.priorities)
        n = len(self.buffer)
        probs = [p / total for p in self.priorities]
        indices = random.choices(range(n), weights=probs, k=batch_size)
        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights to correct for priority sampling bias
        weights = [(1.0 / (n * probs[i])) ** beta for i in indices]
        max_w = max(weights)
        weights = [w / max_w for w in weights]  # Normalize by max for stability

        return samples, indices, weights

    def update_priorities(self, indices: list[int], errors: list[float]):
        for idx, err in zip(indices, errors):
            if idx < len(self.priorities):
                self.priorities[idx] = (abs(err) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)


def progress(count: int, text: str):
    """Helper generator to display training/evaluation progress in terminal with time estimation."""
    current = 0
    start_time = time.time()
    while True:
        percent = 100 * current // count
        elapsed = time.time() - start_time
        
        # Estimate remaining time
        if current > 0:
            avg_time_per_item = elapsed / current
            est_total_time = avg_time_per_item * count
            est_remaining = est_total_time - elapsed
            
            # Format time as mm:ss
            elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
            rem_min, rem_sec = divmod(int(est_remaining), 60)
            time_str = f"[{elapsed_min:02d}:{elapsed_sec:02d}<{rem_min:02d}:{rem_sec:02d}]"
        else:
            time_str = f"[00:00<--:--]"
            
        sys.stderr.write(f"\r{text} {percent}% {time_str}   ")
        sys.stderr.flush()
        if current >= count:
            sys.stderr.write("\n")
            sys.stderr.flush()
            break
        yield current
        current += 1


def load_all_decks():
    """Scans the workspace directories for deck.csv files and loads them.
    
    Returns:
        dict: A dictionary mapping folder name (deck name) to list of card IDs.
    """
    base_path = "decks"
    decks = {}
    
    # 1. Load root deck (our agent's main deck)
    root_deck_path = "deck.csv"
    if os.path.exists(root_deck_path):
        with open(root_deck_path, "r", encoding="utf-8-sig") as f:
            decks["Current (Self)"] = [int(line.strip()) for line in f if line.strip()]
            
    # 2. Scan decks directory for additional deck.csv files
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                deck_file = os.path.join(item_path, "deck.csv")
                if os.path.exists(deck_file):
                    try:
                        with open(deck_file, "r", encoding="utf-8-sig") as f:
                            card_ids = [int(line.strip()) for line in f if line.strip()]
                            if len(card_ids) == 60:
                                decks[item] = card_ids
                                print(f"Loaded deck '{item}' from {deck_file}")
                            else:
                                print(f"Warning: Deck in {deck_file} has {len(card_ids)} cards (must be 60). Skipping.")
                    except Exception as e:
                        print(f"Error loading deck from {deck_file}: {e}")
    return decks


# League model cache — avoid re-loading the same checkpoint from disk every game
_league_cache = {}  # {path: model}


def get_league_model(path, device):
    """Load a league opponent model, caching recently used checkpoints."""
    if path not in _league_cache:
        m = MyModel(256, 4, 512, 2, 2).to(device)
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            m.load_state_dict(checkpoint["state_dict"])
        else:
            m.load_state_dict(checkpoint)
        m.eval()
        # Keep cache small — only last 3 opponents
        if len(_league_cache) >= 3:
            oldest = next(iter(_league_cache))
            del _league_cache[oldest]
        _league_cache[path] = m
    return _league_cache[path]

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the MCTS Pokemon TCG AI Agent.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of evaluation games against random agent (default: 50)")
    parser.add_argument("--self-play-episodes", type=int, default=100, help="Number of self-play games for data collection (default: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for model training (default: 128)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping based on evaluation win rate (default: 10)")
    args = parser.parse_args()

    # Reproducibility
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Load all available decks in the workspace
    opponent_decks = load_all_decks()
    if not opponent_decks:
        raise ValueError("No valid deck.csv found in root or subdirectories.")
        
    # sample_deck is our main active agent deck
    sample_deck = opponent_decks.get("Current (Self)")
    if not sample_deck:
        raise ValueError("Root deck.csv not found.")

    # Setup device, model, optimizer, and loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Hyperparameters: Epochs={args.epochs}, Eval-Episodes={args.eval_episodes}, Self-Play-Episodes={args.self_play_episodes}, Batch-Size={args.batch_size}, LR={args.lr}")
    
    model = MyModel(256, 4, 512, 2, 2)
    
    # Load checkpoint if exists
    checkpoint_path = "model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model weights from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                print(f"  -> Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"  -> Warning: Could not load checkpoint due to size mismatch ({e}). Starting from scratch.")
        
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Cosine warmup scheduler — 3 epoch warmup avoids instability at start
    def _lr_lambda(epoch):
        warmup_epochs = 3
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder value loss (used for non-IS-weighted logging only)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Kept for backward compat but unused by cross-entropy path
    
    # Mixed precision training scaler (CUDA only)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    os.makedirs("out", exist_ok=True)
    os.makedirs("out/runs", exist_ok=True)
    version = 1
    while os.path.exists(f"out/runs/run_{version}"):
        version += 1
    run_dir = f"out/runs/run_{version}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Creating run directory: {run_dir}")

    league_dir = "out/league"
    os.makedirs(league_dir, exist_ok=True)

    # Seed league with starting model (use run version to avoid collision)
    if os.path.exists("model.pth"):
        import shutil
        shutil.copy("model.pth", os.path.join(league_dir, f"model_epoch_0_run_{version}.pth"))
        print(f"Seeded league with initial model checkpoint as model_epoch_0_run_{version}.pth")

    # Initialize CSV files for metrics inside run_dir
    metrics_path = os.path.join(run_dir, "training_metrics.csv")
    with open(metrics_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "win_rate", "avg_loss", "avg_reward",
                         "r_prize_taken", "r_prize_lost", "r_kos", "r_own_kos",
                         "r_energy", "r_bench", "r_deckout", "r_terminal", "r_stall", "r_no_energy",
                         "avg_game_length", "policy_entropy"])

    deck_matchup_path = os.path.join(run_dir, "deck_matchup.csv")
    with open(deck_matchup_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "deck_name", "wins", "losses", "draws", "win_rate"])

    action_dist_path = os.path.join(run_dir, "action_distribution.csv")
    with open(action_dist_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "attack", "play", "attach", "evolve", "ability", "retreat", "end", "other"])

    best_win_rate = -1.0
    patience_counter = 0
    
    # 70/30 Train/Test Opponent Decks Split
    all_opponent_names = sorted([name for name in opponent_decks.keys() if name != "Current (Self)"])
    split_idx = int(0.7 * len(all_opponent_names))
    if len(all_opponent_names) >= 2:
        split_idx = max(1, min(len(all_opponent_names) - 1, split_idx))
    else:
        split_idx = len(all_opponent_names)

    train_opponent_names = all_opponent_names[:split_idx]
    test_opponent_names = all_opponent_names[split_idx:]
    if not test_opponent_names and all_opponent_names:
        test_opponent_names = all_opponent_names

    print(f"Train/Test split (70/30):")
    print(f"  -> Train Opponent Decks: {train_opponent_names}")
    print(f"  -> Test Opponent Decks: {test_opponent_names}")

    # Initialize rolling deck stats for curriculum / adaptive selection (only for training decks)
    rolling_wins = {name: 0.0 for name in train_opponent_names}
    rolling_games = {name: 0.0 for name in train_opponent_names}
    
    # Prioritized Experience Replay Buffer
    replay_buffer = PrioritizedReplayBuffer(capacity=50000)

    # Main training loop
    for counter in range(1, args.epochs + 1):
        print(f"\n--- Epoch {counter}/{args.epochs} ---")
        
        epoch_rewards = []
        
        # Per-epoch reward component accumulators
        rc = {"prize_taken": 0.0, "prize_lost": 0.0, "kos": 0.0, "own_kos": 0.0,
              "energy": 0.0, "bench": 0.0, "deckout": 0.0, "terminal": 0.0, "stall": 0.0,
              "no_energy": 0.0}
        rc_count = 0  # Number of steps accumulated
        total_game_length = 0
        total_games = 0
        
        # Per-epoch action type counters
        action_counts = {"attack": 0, "play": 0, "attach": 0, "evolve": 0,
                         "ability": 0, "retreat": 0, "end": 0, "other": 0}
        
        # Policy entropy accumulator for mode collapse detection
        entropy_accum = 0.0
        entropy_count = 0
        
        # Per-epoch deck matchup tracking (evaluation)
        epoch_deck_stats = {}
        
        # 1. Evaluation
        model.eval()
        win_rate = 0.0
        with torch.inference_mode():
            if args.eval_episodes > 0:
                results = [0, 0, 0]  # [Wins, Losses, Draws]
                deck_stats = {}      # {deck_name: [wins, losses, draws]}
                
                # Use only the test opponent decks split
                eval_opponent_names = test_opponent_names
                
                for name in eval_opponent_names:
                    deck_stats[name] = [0, 0, 0]

                for i in progress(args.eval_episodes, "Evaluating... "):
                    # Round-robin distribution of test opponent decks
                    opponent_name = eval_opponent_names[i % len(eval_opponent_names)]
                    opponent_deck = opponent_decks[opponent_name]
                    
                    obs, start_data = battle_start(sample_deck, opponent_deck)
                    if start_data.errorPlayer >= 0:
                        error = f"Deck error in battle between your deck and '{opponent_name}'."
                        if start_data.errorType == 1:
                            error += " The deck contains invalid card ID."
                        elif start_data.errorType == 2:
                            error += " You can include up to four cards with the same name."
                        elif start_data.errorType == 3:
                            error += " There are no Basic Pokémon in the deck."
                        elif start_data.errorType == 4:
                            error += " You can include only one Ace Spec card."
                        raise ValueError(error)
                        
                    your_index = i % 2
                    while True:
                        if obs["current"]["result"] >= 0:
                            break

                        if obs["current"]["yourIndex"] == your_index:
                            selected, _ = mcts_agent(obs, sample_deck, model)
                        else:
                            selected = random_agent(obs)
                        obs = battle_select(selected)
                    
                    battle_finish()

                    result = obs["current"]["result"]
                    if result == 2:  # Draw
                        results[2] += 1
                        deck_stats[opponent_name][2] += 1
                    elif result == your_index:  # Win
                        results[0] += 1
                        deck_stats[opponent_name][0] += 1
                    else:  # Lose
                        results[1] += 1
                        deck_stats[opponent_name][1] += 1
                
                denom = results[0] + results[1]
                win_rate = 100.0 * results[0] / denom if denom > 0 else 0.0
                print(f"Overall Evaluation win rate: {win_rate}% (Wins: {results[0]}, Losses: {results[1]}, Draws: {results[2]})", flush=True)
                
                # Check and save the best performing model
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_checkpoint = {
                        "epoch": counter,
                        "state_dict": model.state_dict(),
                        "win_rate": best_win_rate
                    }
                    torch.save(best_checkpoint, "best_model.pth")
                    torch.save(best_checkpoint, os.path.join(run_dir, "best_model.pth"))
                    print(f"  -> New best model checkpoint saved (Win Rate: {best_win_rate}%)")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"  -> No improvement in Win Rate for {patience_counter} consecutive epochs.")
                    if patience_counter >= args.patience:
                        print(f"Early stopping triggered: Win Rate has not improved for {args.patience} epochs.")
                        break
                    
                for name, stats in deck_stats.items():
                    d_denom = stats[0] + stats[1]
                    d_win_rate = 100.0 * stats[0] / d_denom if d_denom > 0 else 0.0
                    print(f"  -> vs '{name}': {d_win_rate:.1f}% (Wins: {stats[0]}, Losses: {stats[1]}, Draws: {stats[2]})", flush=True)
                    epoch_deck_stats[name] = {"wins": stats[0], "losses": stats[1], "draws": stats[2], "win_rate": d_win_rate}
            else:
                print("Skipping evaluation (eval-episodes=0).")

            # 2. Self Play Data Collection with League & Adaptive Selection
            if args.self_play_episodes > 0:
                # Use only the train opponent decks split
                eval_opponent_names = train_opponent_names
                
                for _ in progress(args.self_play_episodes, "Training Data Collecting... "):
                    # Decide opponent deck using Adaptive Matchup Selection (bias to lower win rates)
                    if not eval_opponent_names:
                        opponent_name = "Current (Self)"
                        opponent_deck = sample_deck
                    else:
                        if random.random() < 0.5:
                            opponent_name = "Current (Self)"
                            opponent_deck = sample_deck
                        else:
                            # Calculate opponent deck weights based on current rolling win rates
                            weights = []
                            for name in eval_opponent_names:
                                g = rolling_games[name]
                                w = rolling_wins[name]
                                wr = w / g if g > 0 else 0.5
                                # Invert winrate to prioritize playing against tougher decks
                                weights.append(max(0.1, 1.0 - wr))
                            
                            w_sum = sum(weights)
                            probs = [w / w_sum for w in weights]
                            opponent_name = random.choices(eval_opponent_names, weights=probs, k=1)[0]
                            opponent_deck = opponent_decks[opponent_name]
                    
                    # Decide opponent checkpoint path based on 40/30/20/10 distribution
                    use_league = False
                    league_model = None
                    r = random.random()
                    opponent_path = None
                    opponent_type = "Current"
                    
                    if r < 0.40:
                        opponent_type = "Current"
                    elif r < 0.70:
                        best_path = "best_model.pth"
                        if os.path.exists(best_path):
                            opponent_path = best_path
                            opponent_type = "Best Checkpoint"
                        else:
                            opponent_type = "Current"
                    elif r < 0.90:
                        prev_path = os.path.join(run_dir, f"model{counter-1}.pth")
                        if counter > 1 and os.path.exists(prev_path):
                            opponent_path = prev_path
                            opponent_type = "Previous Checkpoint"
                        else:
                            opponent_type = "Current"
                    else:
                        if os.path.exists(league_dir):
                            checkpoints = [os.path.join(league_dir, f) for f in os.listdir(league_dir) if f.endswith(".pth")]
                            if checkpoints:
                                opponent_path = random.choice(checkpoints)
                                opponent_type = f"Random Historical ({os.path.basename(opponent_path)})"
                                
                    if opponent_path is not None:
                        try:
                            league_model = get_league_model(opponent_path, device)
                            use_league = True
                        except Exception as e:
                            # Catch and skip size mismatch warnings silently or with warning
                            print(f"  -> Warning: Skipping checkpoint {opponent_path}: {e}")
                            use_league = False

                    obs, _ = battle_start(sample_deck, opponent_deck)
                    samples: list[list[tuple[LearnSample, dict]]] = [[], []]  # [Player0 samples, Player1 samples]
                    
                    while True:
                        if obs["current"]["result"] >= 0:
                            break

                        curr_player = obs["current"]["yourIndex"]
                        curr_deck = sample_deck if curr_player == 0 else opponent_deck
                        
                        # Inspect and store pre-action state metrics
                        obs_class = to_observation_class(obs)
                        state_ps = obs_class.current.players[curr_player]
                        opp_ps = obs_class.current.players[1 - curr_player]
                        
                        pre_metrics = {
                            "prizes": len(state_ps.prize),
                            "opp_prizes": len(opp_ps.prize),
                            "energy": count_attached_energy(state_ps),
                            "pokemon": count_pokemon(state_ps),
                            "opp_pokemon": count_pokemon(opp_ps),
                            "bench_size": len([p for p in state_ps.bench if p is not None]),
                            "deck_size": state_ps.deckCount,
                            "energy_attached_flag": obs_class.current.energyAttached  # True if energy was attached this turn
                        }

                        # Retrieve action and sample
                        if curr_player == 0:
                            selected, sample = mcts_agent(obs, curr_deck, model)
                            sample.pred_val = sample.value
                            samples[0].append((sample, pre_metrics))
                            
                            # Policy entropy for monitoring exploration/mode collapse
                            if sample is not None and hasattr(sample, 'policy') and len(sample.policy) > 0:
                                policy_probs = [max(1e-8, p) for p in sample.policy if p > 0]
                                p_sum = sum(policy_probs)
                                if p_sum > 0:
                                    entropy = -sum((p/p_sum) * math.log(p/p_sum) for p in policy_probs)
                                    entropy_accum += entropy
                                    entropy_count += 1
                        else:
                            if use_league:
                                selected, _ = mcts_agent(obs, curr_deck, league_model)
                            else:
                                selected, sample = mcts_agent(obs, curr_deck, model)
                                if opponent_name == "Current (Self)":
                                    sample.pred_val = sample.value
                                    samples[1].append((sample, pre_metrics))
                        
                        # Track action type distribution (Player 0 only)
                        if curr_player == 0:
                            for sel_idx in selected:
                                if sel_idx < len(obs_class.select.option):
                                    opt = obs_class.select.option[sel_idx]
                                    if opt.type == OptionType.ATTACK:
                                        action_counts["attack"] += 1
                                    elif opt.type == OptionType.PLAY:
                                        action_counts["play"] += 1
                                    elif opt.type == OptionType.ATTACH:
                                        action_counts["attach"] += 1
                                    elif opt.type == OptionType.EVOLVE:
                                        action_counts["evolve"] += 1
                                    elif opt.type == OptionType.ABILITY:
                                        action_counts["ability"] += 1
                                    elif opt.type == OptionType.RETREAT:
                                        action_counts["retreat"] += 1
                                    elif opt.type == OptionType.END:
                                        action_counts["end"] += 1
                                    else:
                                        action_counts["other"] += 1
                        
                        obs = battle_select(selected)
                    
                    battle_finish()

                    # Update rolling stats for Adaptive selection (from Player 0's perspective)
                    result = obs["current"]["result"]
                    if opponent_name != "Current (Self)":
                        rolling_games[opponent_name] += 1
                        if result == 0:  # Player 0 won
                            rolling_wins[opponent_name] += 1
                        elif result == 2:  # Draw
                            rolling_wins[opponent_name] += 0.5
                        
                        # Exponential decay so selector responds to current skill, not early history
                        DECAY = 0.9
                        for name in train_opponent_names:
                            rolling_wins[name] *= DECAY
                            rolling_games[name] *= DECAY

                    # Backpropagate and shape dense rewards
                    for i in range(2):
                        if i == 0 or (opponent_name == "Current (Self)" and not use_league):
                            player_samples = samples[i]
                            n_steps = len(player_samples)
                            if n_steps == 0:
                                continue
                            
                            # Terminal reward — normalized to [-1, 1] to match tanh output range
                            if result == 2:         # Draw
                                terminal_reward = 0.0
                            elif i == result:       # Win
                                terminal_reward = 1.0
                            else:                   # Loss
                                terminal_reward = -1.0
                            if i == 0:  # Track only from Player 0's perspective
                                rc["terminal"] += terminal_reward
                                total_game_length += obs_class.current.turn if (obs_class is not None and obs_class.current is not None) else 0
                                total_games += 1
                            
                            rewards = []
                            for step_idx in range(n_steps):
                                sample, pre = player_samples[step_idx]
                                
                                if step_idx < n_steps - 1:
                                    _, post = player_samples[step_idx + 1]
                                else:
                                    final_obs = to_observation_class(obs)
                                    final_ps = final_obs.current.players[i]
                                    final_opp_ps = final_obs.current.players[1 - i]
                                    post = {
                                        "prizes": len(final_ps.prize),
                                        "opp_prizes": len(final_opp_ps.prize),
                                        "energy": count_attached_energy(final_ps),
                                        "pokemon": count_pokemon(final_ps),
                                        "opp_pokemon": count_pokemon(final_opp_ps),
                                        "bench_size": len([p for p in final_ps.bench if p is not None]),
                                        "deck_size": final_ps.deckCount,
                                        "energy_attached_flag": True  # Game over — no penalty on final step
                                    }
                                
                                # Compute differences
                                prizes_taken = pre["prizes"] - post["prizes"]
                                prizes_lost = pre["opp_prizes"] - post["opp_prizes"]
                                opp_kos = pre["opp_pokemon"] - post["opp_pokemon"]
                                own_kos = pre["pokemon"] - post["pokemon"]
                                energy_attached = post["energy"] - pre["energy"]
                                
                                # Intermediate reward calculation — normalized to be proportional to ±1.0 terminal
                                # Previously these were 10–100× too large, causing tanh saturation and flat gradients.
                                step_reward = 0.0
                                r_stall = -0.005   # Mild per-step anti-stall (was -0.15)
                                r_prize_t = 0.0
                                r_prize_l = 0.0
                                r_ko = 0.0
                                r_own_ko = 0.0
                                r_en = 0.0
                                r_bench = 0.0
                                r_deck = 0.0
                                
                                if prizes_taken > 0:
                                    r_prize_t = prizes_taken * 0.20   # (was 12.0) Key win condition
                                if prizes_lost > 0:
                                    r_prize_l = prizes_lost * 0.15    # (was 8.0) Opponent taking prizes
                                if opp_kos > 0:
                                    r_ko = opp_kos * 0.10             # (was 6.0) Encourage aggression
                                if own_kos > 0:
                                    r_own_ko = own_kos * 0.08         # (was 5.0) Own KOs penalty
                                if energy_attached > 0:
                                    r_en = energy_attached * 0.03     # (was 1.5) Energy prerequisite to attacking
                                elif energy_attached < 0:
                                    r_en = energy_attached * 0.02     # (was 1.0) Lost energy
                                
                                # REMOVED: r_no_energy penalty
                                # The energy_attached_flag fires on every non-energy action within a turn,
                                # not just at end-of-turn. This fired 97% of steps and drowned all other signals.
                                # Energy attachment is already incentivized by r_en (+0.03 per energy attached).
                                r_no_energy = 0.0
                                
                                # Bench cushion penalty — mild nudge, not dominant signal
                                if post["bench_size"] <= 1:
                                    r_bench -= 0.02   # (was -1.0)
                                
                                # Reward for benching a Pokémon when bench was dangerously low
                                if pre["bench_size"] <= 1 and (post["bench_size"] > pre["bench_size"]):
                                    r_bench += 0.10    # (was +5.0)
                                
                                # Deck out penalty (apply penalty if deck size is critically low)
                                if post["deck_size"] <= 3:
                                    r_deck -= 0.10     # (was -5.0) Critical danger
                                elif post["deck_size"] <= 5:
                                    r_deck -= 0.04     # (was -2.0) Impending danger
                                
                                step_reward = r_stall + r_prize_t - r_prize_l + r_ko - r_own_ko + r_en + r_bench + r_deck
                                
                                # Accumulate reward components (Player 0 only)
                                if i == 0:
                                    rc["prize_taken"] += r_prize_t
                                    rc["prize_lost"] += r_prize_l
                                    rc["kos"] += r_ko
                                    rc["own_kos"] += r_own_ko
                                    rc["energy"] += r_en
                                    rc["bench"] += r_bench
                                    rc["deckout"] += r_deck
                                    rc["stall"] += r_stall
                                    rc["no_energy"] += r_no_energy
                                    rc_count += 1
                                
                                rewards.append(step_reward)
                            
                            # Propagate targets back using GAE(λ) — standard method from PPO/AlphaStar
                            GAMMA = 0.99    # Discount factor
                            LAMBDA = 0.95   # GAE lambda (was 0.98 — 0.95 is the standard sweet spot)

                            # Collect model predictions for bootstrapping
                            pred_values = [player_samples[s][0].pred_val for s in range(n_steps)]

                            # Standard GAE lambda-return calculation
                            returns = [0.0] * n_steps
                            gae = 0.0
                            for step_idx in reversed(range(n_steps)):
                                step_rew = rewards[step_idx]

                                if step_idx == n_steps - 1:
                                    next_val = terminal_reward  # Bootstrap from terminal
                                else:
                                    next_val = pred_values[step_idx + 1]

                                delta = step_rew + GAMMA * next_val - pred_values[step_idx]  # TD error
                                gae = delta + GAMMA * LAMBDA * gae  # GAE accumulation
                                returns[step_idx] = gae + pred_values[step_idx]  # λ-return = GAE + V(s)

                            # Assign targets and add to replay
                            for step_idx in range(n_steps):
                                sample, _ = player_samples[step_idx]
                                # Clamp target to [-1, 1] to match tanh output range
                                sample.value = max(-1.0, min(1.0, returns[step_idx]))
                                td_error = returns[step_idx] - sample.pred_val
                                replay_buffer.add(sample, td_error)
                                epoch_rewards.append(sample.value)
            else:
                print("Skipping self-play data collection (self-play-episodes=0).")

        # 3. Model Training using Prioritized Replay Buffer
        avg_loss = 0.0
        if len(replay_buffer) >= args.batch_size:
            print("Training Start.")
            model.train()
            batch_count = min(30, len(replay_buffer) // args.batch_size)
            print(f"Total training buffer size: {len(replay_buffer)}, Batch Count: {batch_count}")
            
            epoch_losses = []
            for i in range(batch_count):
                # Anneal IS beta from 0.4 → 1.0 over training epochs
                beta = min(1.0, 0.4 + 0.6 * (counter / args.epochs))
                samples, indices, is_weights = replay_buffer.sample(args.batch_size, beta=beta)
                
                input_enc = LearnInput()
                input_dec = LearnInput()
                mask = []
                label_enc = []
                label_dec = []
                
                for sample in samples:
                    input_enc.add(sample.sv_enc)
                    input_dec.add(sample.sv_dec)
                    label_enc.append(sample.value)
                    label_dec.extend(sample.policy)
                    for _ in range(len(sample.policy)):
                        mask.append(1.0)
                    for _ in range(64 - len(sample.policy)):
                        mask.append(0.0)
                        label_dec.append(0.0)
                        input_dec.offset.append(len(input_dec.index))

                # Convert to PyTorch tensors
                mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(args.batch_size, -1)
                label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device).view(args.batch_size, -1)
                label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device).view(args.batch_size, -1)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                    out_enc, out_dec = model(
                        torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                        torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                        torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                        torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                        torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                        torch.tensor(input_dec.offset, dtype=torch.int32, device=device)
                    )
                    
                    # Importance sampling weight tensor for PER bias correction
                    is_weight_tensor = torch.tensor(is_weights, dtype=torch.float32, device=device).view(args.batch_size, 1)

                    # Value loss with IS weighting (element-wise then weighted mean)
                    loss_enc_elem = torch.nn.functional.huber_loss(out_enc, label_tensor_enc, reduction="none", delta=0.2)
                    loss_enc = (loss_enc_elem * is_weight_tensor).mean()

                    # Policy loss: masked cross-entropy (matching visit-count probability targets)
                    # out_dec is raw logits (batch, 64), label_tensor_dec is probabilities (batch, 64)
                    # Mask invalid actions by setting their logits to -inf before softmax
                    masked_logits = out_dec + (1.0 - mask_tensor) * (-1e9)
                    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
                    loss_dec_ce = -(label_tensor_dec * log_probs * mask_tensor)
                    loss_dec_ce = (loss_dec_ce.sum(dim=-1, keepdim=True) * is_weight_tensor).mean()

                    loss = loss_enc + loss_dec_ce

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                epoch_losses.append(loss.item())
                
                errors = (out_enc - label_tensor_enc).abs().squeeze().tolist()
                # Handle single-item batch squeeze edge case
                if isinstance(errors, float):
                    errors = [errors]
                replay_buffer.update_priorities(indices, errors)
                
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            print(f"Training Finish. Average Loss: {avg_loss:.4f}")
        else:
            if args.self_play_episodes > 0:
                print(f"Skipping training: collected buffer size ({len(replay_buffer)}) less than batch size ({args.batch_size}).")

        # Compute average reward for this epoch
        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0

        # Save checkpoints AFTER training (not before)
        epoch_model_path = os.path.join(run_dir, f"model{counter}.pth")
        checkpoint = {
            "epoch": counter,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }
        torch.save(checkpoint, epoch_model_path)
        torch.save(checkpoint, "model.pth")
        
        # Save to league directory (include run version to avoid collision)
        league_model_path = os.path.join(league_dir, f"model_epoch_{counter}_run_{version}.pth")
        torch.save(checkpoint, league_model_path)
        print(f"Saved checkpoint: {epoch_model_path}, model.pth, and {league_model_path}")

        # Compute average reward components
        avg_gl = total_game_length / total_games if total_games > 0 else 0.0
        rc_div = max(1, rc_count)
        
        # Log metrics to CSV (expanded with entropy)
        avg_entropy = entropy_accum / max(1, entropy_count)
        with open(metrics_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([counter, win_rate, avg_loss, avg_reward,
                             rc["prize_taken"] / rc_div, rc["prize_lost"] / rc_div,
                             rc["kos"] / rc_div, rc["own_kos"] / rc_div,
                             rc["energy"] / rc_div, rc["bench"] / rc_div,
                             rc["deckout"] / rc_div, rc["terminal"] / max(1, total_games),
                             rc["stall"] / rc_div, rc["no_energy"] / rc_div, avg_gl,
                             avg_entropy])
        
        # Log deck matchup stats
        with open(deck_matchup_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for name, stats in epoch_deck_stats.items():
                writer.writerow([counter, name, stats["wins"], stats["losses"], stats["draws"], f"{stats['win_rate']:.1f}"])
        
        # Log action distribution
        with open(action_dist_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([counter, action_counts["attack"], action_counts["play"],
                             action_counts["attach"], action_counts["evolve"],
                             action_counts["ability"], action_counts["retreat"],
                             action_counts["end"], action_counts["other"]])
            
        # Step the scheduler
        scheduler.step()
        print(f"Epoch Metrics Logged -> Win Rate: {win_rate}%, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
    # Save the final model weights to model.pth only (best_model.pth is preserved from peak win rate)
    final_checkpoint = {
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(final_checkpoint, "model.pth")
    torch.save(final_checkpoint, os.path.join(run_dir, "model.pth"))
    print(f"\nTraining complete. Final weights saved to model.pth and inside {run_dir}")
    print(f"Best model (peak win rate) preserved in best_model.pth")


    # Generate plots
    print("Generating learning curves...")
    plot_metrics(metrics_path, run_dir, deck_matchup_path, action_dist_path)


if __name__ == "__main__":
    main()
