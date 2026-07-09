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


def progress(count: int, text: str):
    """Helper generator to display training/evaluation progress in terminal."""
    current = 0
    while True:
        percent = 100 * current // count
        sys.stderr.write(f"\r{text} {percent}%   ")
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
            decks["Dragapult (Self)"] = [int(line.strip()) for line in f if line.strip()]
            
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
    args = parser.parse_args()

    # Load all available decks in the workspace
    opponent_decks = load_all_decks()
    if not opponent_decks:
        raise ValueError("No valid deck.csv found in root or subdirectories.")
        
    # sample_deck is our main active agent deck
    sample_deck = opponent_decks.get("Dragapult (Self)")
    if not sample_deck:
        raise ValueError("Root deck.csv not found.")

    # Setup device, model, optimizer, and loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Hyperparameters: Epochs={args.epochs}, Eval-Episodes={args.eval_episodes}, Self-Play-Episodes={args.self_play_episodes}, Batch-Size={args.batch_size}, LR={args.lr}")
    
    model = MyModel(128, 2, 256, 1, 1)
    
    # Load checkpoint if exists
    checkpoint_path = "model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder loss function
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Decoder loss function
    
    os.makedirs("out", exist_ok=True)

    # Initialize CSV file for metrics
    metrics_path = "out/training_metrics.csv"
    with open(metrics_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "win_rate", "avg_loss", "avg_reward"])

    best_win_rate = -1.0

    # Main training loop
    for counter in range(1, args.epochs + 1):
        # Save checkpoints
        epoch_model_path = f"out/model{counter}.pth"
        torch.save(model.state_dict(), epoch_model_path)
        torch.save(model.state_dict(), "model.pth")
        print(f"\n--- Epoch {counter}/{args.epochs} ---")
        print(f"Saved checkpoint: {epoch_model_path} and model.pth")
        
        sample_list: list[LearnSample] = []
        epoch_rewards = []
        
        # 1. Evaluation
        model.eval()
        win_rate = 0.0
        with torch.inference_mode():
            if args.eval_episodes > 0:
                results = [0, 0, 0]  # [Wins, Losses, Draws]
                deck_stats = {}      # {deck_name: [wins, losses, draws]}
                
                # Exclude own deck from opponent list for evaluation if others exist
                eval_opponent_names = [name for name in opponent_decks.keys() if name != "Dragapult (Self)"]
                if not eval_opponent_names:
                    eval_opponent_names = list(opponent_decks.keys())
                
                for name in eval_opponent_names:
                    deck_stats[name] = [0, 0, 0]

                for i in progress(args.eval_episodes, "Evaluating... "):
                    # Round-robin distribution of opponent decks
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
                if win_rate >= best_win_rate:
                    best_win_rate = win_rate
                    torch.save(model.state_dict(), "best_model.pth")
                    print(f"  -> New best model checkpoint saved (Win Rate: {best_win_rate}%)")
                    
                for name, stats in deck_stats.items():
                    d_denom = stats[0] + stats[1]
                    d_win_rate = float(100 * stats[0] // d_denom) if d_denom > 0 else 0.0
                    print(f"  -> vs '{name}': {d_win_rate}% (Wins: {stats[0]}, Losses: {stats[1]}, Draws: {stats[2]})", flush=True)
            else:
                print("Skipping evaluation (eval-episodes=0).")

            # 2. Self Play Data Collection
            if args.self_play_episodes > 0:
                eval_opponent_names = [name for name in opponent_decks.keys() if name != "Dragapult (Self)"]
                
                for _ in progress(args.self_play_episodes, "Training Data Collecting... "):
                    # 50% chance of Dragapult vs Dragapult self-play, 50% Dragapult vs opponent decks
                    if random.random() < 0.5 or not eval_opponent_names:
                        opponent_name = "Dragapult (Self)"
                        opponent_deck = sample_deck
                    else:
                        opponent_name = random.choice(eval_opponent_names)
                        opponent_deck = opponent_decks[opponent_name]
                        
                    obs, _ = battle_start(sample_deck, opponent_deck)
                    samples: list[list[LearnSample]] = [[], []]  # [Player0 samples, Player1 samples]
                    while True:
                        if obs["current"]["result"] >= 0:
                            break

                        # Crucial fix: pass correct deck list of the active decision-making player
                        curr_player = obs["current"]["yourIndex"]
                        curr_deck = sample_deck if curr_player == 0 else opponent_deck
                        
                        selected, sample = mcts_agent(obs, curr_deck, model)
                        samples[curr_player].append(sample)
                        obs = battle_select(selected)
                    
                    battle_finish()

                    # Backpropagate actual game outcome rewards into the collected samples
                    for i in range(2):
                        LAMBDA = 0.9
                        value = 1.0 if i == obs["current"]["result"] else -1.0

                        for sample in reversed(samples[i]):
                            label = (value + sample.value) * 0.5
                            value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
                            sample.value = label
                            sample_list.append(sample)
                            epoch_rewards.append(label)
            else:
                print("Skipping self-play data collection (self-play-episodes=0).")

        # 3. Model Training
        avg_loss = 0.0
        if len(sample_list) >= args.batch_size:
            print("Training Start.")
            model.train()
            random.shuffle(sample_list)
            batch_count = len(sample_list) // args.batch_size
            print(f"Total training samples: {len(sample_list)}, Batch Count: {batch_count}")
            
            epoch_losses = []
            for i in range(batch_count):
                input_enc = LearnInput()
                input_dec = LearnInput()
                mask = []
                label_enc = []
                label_dec = []
                start = args.batch_size * i
                
                for j in range(start, start + args.batch_size):
                    sample = sample_list[j]
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
                
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            print(f"Training Finish. Average Loss: {avg_loss:.4f}")
        else:
            if args.self_play_episodes > 0:
                print(f"Skipping training: collected samples ({len(sample_list)}) less than batch size ({args.batch_size}).")

        # Compute average reward for this epoch
        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0

        # Log metrics to CSV
        with open(metrics_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([counter, win_rate, avg_loss, avg_reward])
            
        print(f"Epoch Metrics Logged -> Win Rate: {win_rate}%, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
        
    # Load and save the best model as the final model.pth for submission packaging
    if os.path.exists("best_model.pth"):
        import shutil
        shutil.copy("best_model.pth", "model.pth")
        print(f"\nFinal submission model updated to the best checkpoint (Win Rate: {best_win_rate}%)")
    else:
        torch.save(model.state_dict(), "model.pth")
        print("\nTraining complete. Final weights saved as model.pth")

    # Generate plots
    print("Generating learning curves...")
    plot_metrics(metrics_path, "out")


if __name__ == "__main__":
    main()
