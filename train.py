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
from cg.api import to_observation_class


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
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = 0.6  # prioritization exponent

    def add(self, sample: LearnSample, td_error: float):
        priority = max(1e-5, abs(td_error)) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
            self.priorities.append(priority)
        else:
            idx = random.randint(0, len(self.buffer) - 1)
            self.buffer[idx] = sample
            self.priorities[idx] = priority

    def sample(self, batch_size: int) -> tuple[list[LearnSample], list[int]]:
        probs = [p / sum(self.priorities) for p in self.priorities]
        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        samples = [self.buffer[i] for i in indices]
        return samples, indices

    def update_priorities(self, indices: list[int], errors: list[float]):
        for idx, err in zip(indices, errors):
            if idx < len(self.priorities):
                self.priorities[idx] = max(1e-5, abs(err)) ** self.alpha

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


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the MCTS Pokemon TCG AI Agent.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of evaluation games against random agent (default: 50)")
    parser.add_argument("--self-play-episodes", type=int, default=100, help="Number of self-play games for data collection (default: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for model training (default: 128)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping based on evaluation win rate (default: 10)")
    args = parser.parse_args()

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder loss function
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Decoder loss function
    
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

    # Initialize CSV file for metrics inside run_dir
    metrics_path = os.path.join(run_dir, "training_metrics.csv")
    with open(metrics_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "win_rate", "avg_loss", "avg_reward"])

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
    replay_buffer = PrioritizedReplayBuffer(capacity=5000)

    # Main training loop
    for counter in range(1, args.epochs + 1):
        # Save checkpoints inside run_dir and root
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
        print(f"\n--- Epoch {counter}/{args.epochs} ---")
        print(f"Saved checkpoint: {epoch_model_path}, model.pth, and {league_model_path}")
        
        epoch_rewards = []
        
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
                win_rate = float(100 * results[0] // denom) if denom > 0 else 0.0
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
                    d_win_rate = float(100 * stats[0] // d_denom) if d_denom > 0 else 0.0
                    print(f"  -> vs '{name}': {d_win_rate}% (Wins: {stats[0]}, Losses: {stats[1]}, Draws: {stats[2]})", flush=True)
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
                    
                    # Decide whether to use a historical league checkpoint for opponent (Player 1)
                    use_league = False
                    league_model = None
                    if random.random() < 0.6:
                        if os.path.exists(league_dir):
                            checkpoints = [os.path.join(league_dir, f) for f in os.listdir(league_dir) if f.endswith(".pth")]
                            if checkpoints:
                                checkpoint_path = random.choice(checkpoints)
                                use_league = True
                                league_model = MyModel(256, 4, 512, 2, 2).to(device)
                                try:
                                    checkpoint = torch.load(checkpoint_path, map_location=device)
                                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                                        league_model.load_state_dict(checkpoint["state_dict"])
                                    else:
                                        league_model.load_state_dict(checkpoint)
                                    league_model.eval()
                                except Exception as e:
                                    print(f"  -> Warning: Skipping league checkpoint {checkpoint_path} due to size mismatch: {e}")
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
                            "energy": count_attached_energy(state_ps),
                            "pokemon": count_pokemon(state_ps),
                            "opp_pokemon": count_pokemon(opp_ps),
                            "bench_size": len([p for p in state_ps.bench if p is not None])
                        }

                        # Retrieve action and sample
                        if curr_player == 0:
                            selected, sample = mcts_agent(obs, curr_deck, model)
                            sample.pred_val = sample.value
                            samples[0].append((sample, pre_metrics))
                        else:
                            if use_league:
                                selected, _ = mcts_agent(obs, curr_deck, league_model)
                            else:
                                selected, sample = mcts_agent(obs, curr_deck, model)
                                if opponent_name == "Current (Self)":
                                    sample.pred_val = sample.value
                                    samples[1].append((sample, pre_metrics))
                        
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

                    # Backpropagate and shape dense rewards
                    for i in range(2):
                        if i == 0 or (opponent_name == "Current (Self)" and not use_league):
                            player_samples = samples[i]
                            n_steps = len(player_samples)
                            if n_steps == 0:
                                continue
                            
                            # Terminal reward
                            game_won = (i == result)
                            terminal_reward = 100.0 if game_won else -100.0
                            
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
                                        "energy": count_attached_energy(final_ps),
                                        "pokemon": count_pokemon(final_ps),
                                        "opp_pokemon": count_pokemon(final_opp_ps),
                                        "bench_size": len([p for p in final_ps.bench if p is not None])
                                    }
                                
                                # Compute differences
                                prizes_taken = pre["prizes"] - post["prizes"]
                                opp_kos = pre["opp_pokemon"] - post["opp_pokemon"]
                                own_kos = pre["pokemon"] - post["pokemon"]
                                energy_attached = post["energy"] - pre["energy"]
                                
                                # Intermediate reward calculation
                                step_reward = 0.0
                                if prizes_taken > 0:
                                    step_reward += prizes_taken * 8.0
                                if opp_kos > 0:
                                    step_reward += opp_kos * 5.0
                                if own_kos > 0:
                                    step_reward -= own_kos * 3.0
                                if energy_attached > 0:
                                    step_reward += energy_attached * 0.3
                                elif energy_attached < 0:
                                    step_reward += energy_attached * 1.0  # negative addition
                                
                                # Bench cushion penalty (apply penalty if bench is dangerously low: 0 or 1 Pokémon)
                                if post["bench_size"] <= 1:
                                    step_reward -= 2.0
                                
                                # Reward for benching a Pokémon when bench was dangerously low
                                if pre["bench_size"] <= 1 and (post["bench_size"] > pre["bench_size"]):
                                    step_reward += 5.0
                                
                                rewards.append(step_reward)
                            
                            # Propagate targets back
                            LAMBDA = 0.98
                            val = terminal_reward
                            for step_idx in reversed(range(n_steps)):
                                sample, _ = player_samples[step_idx]
                                step_rew = rewards[step_idx]
                                
                                target = (val + step_rew + sample.pred_val) * 0.5
                                val = (val + step_rew) * LAMBDA + sample.pred_val * (1.0 - LAMBDA)
                                
                                sample.value = target
                                td_error = target - sample.pred_val
                                replay_buffer.add(sample, td_error)
                                epoch_rewards.append(target)
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
                samples, indices = replay_buffer.sample(args.batch_size)
                
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

                out_enc, out_dec = model(
                    torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                    torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                    torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                    torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                    torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                    torch.tensor(input_dec.offset, dtype=torch.int32, device=device)
                )
                
                loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
                loss_dec = loss_fn_dec(out_dec, label_tensor_dec)
                loss_dec = loss_dec * mask_tensor
                loss_dec = loss_dec.sum() / float(args.batch_size)
                loss = loss_enc + loss_dec

                loss.backward()
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

        # Log metrics to CSV
        with open(metrics_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([counter, win_rate, avg_loss, avg_reward])
            
        # Step the scheduler
        scheduler.step()
        print(f"Epoch Metrics Logged -> Win Rate: {win_rate}%, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
    # Save the final model weights (latest/most trained) to both model.pth and best_model.pth
    final_checkpoint = {
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(final_checkpoint, "model.pth")
    torch.save(final_checkpoint, "best_model.pth")
    # Also save copies inside the run directory
    torch.save(final_checkpoint, os.path.join(run_dir, "model.pth"))
    torch.save(final_checkpoint, os.path.join(run_dir, "best_model.pth"))
    print(f"\nTraining complete. Final weights saved to model.pth, best_model.pth, and inside {run_dir}")


    # Generate plots
    print("Generating learning curves...")
    plot_metrics(metrics_path, run_dir)


if __name__ == "__main__":
    main()
