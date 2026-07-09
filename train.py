import os
import sys
import random
import torch
import torch.optim
import torch.nn

# Ensure current directory is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models import MyModel
from encoders import SparseVector
from agent import mcts_agent, random_agent, LearnSample
from cg.game import battle_start, battle_finish, battle_select

class LearnInput:
    """
    Helper class to construct batch inputs for the neural network.
    """
    index: list[int]
    value: list[float]
    offset: list[int]

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []

    def add(self, sv: SparseVector):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)

def progress(count: int, text: str):
    """
    Progress bar indicator for training/evaluation iterations.
    """
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

def load_deck(path: str = None) -> list[int]:
    """
    Loads deck list from CSV file containing 60 card IDs.
    """
    if path is None:
        path = os.path.join(BASE_DIR, "deck.csv")
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def run_evaluation(model: MyModel, deck: list[int], num_games: int = 50) -> float:
    """
    Evaluates the current model against a random agent.
    Returns the win rate.
    """
    model.eval()
    results = [0, 0, 0]  # [win, lose, draw]
    with torch.inference_mode():
        for i in progress(num_games, "Evaluating... "):
            obs, start_data = battle_start(deck, deck)
            if start_data.errorPlayer >= 0:
                error = "Deck error."
                if start_data.errorType == 1:
                    error = "The deck contains invalid card ID."
                elif start_data.errorType == 2:
                    error = "You can include up to four cards with the same name in the deck, excluding basic Energy cards."
                elif start_data.errorType == 3:
                    error = "There are no Basic Pokémon in the deck."
                elif start_data.errorType == 4:
                    error = "You can include only one Ace Spec card in the deck."
                raise ValueError(error)
            
            your_index = i % 2
            while True:
                if obs["current"]["result"] >= 0:
                    break

                if obs["current"]["yourIndex"] == your_index:
                    selected, _ = mcts_agent(obs, deck, model, search_count=10)
                else:
                    selected = random_agent(obs)
                obs = battle_select(selected)
            
            battle_finish()

            result = obs["current"]["result"]
            if result == 2:  # Draw
                results[2] += 1
            elif result == your_index:  # Win
                results[0] += 1
            else:  # Lose
                results[1] += 1
    
    total_decided = results[0] + results[1]
    win_rate = 100 * results[0] // total_decided if total_decided > 0 else 0
    return win_rate

def collect_self_play_data(model: MyModel, deck: list[int], num_episodes: int = 100) -> list[LearnSample]:
    """
    Collects training data by playing the model against itself.
    """
    model.eval()
    sample_list: list[LearnSample] = []
    with torch.inference_mode():
        for _ in progress(num_episodes, "Training Data Collecting... "):
            obs, _ = battle_start(deck, deck)
            samples = [[], []]  # [Player0 samples, Player1 samples]
            while True:
                if obs["current"]["result"] >= 0:
                    break

                selected, sample = mcts_agent(obs, deck, model, search_count=10)
                samples[obs["current"]["yourIndex"]].append(sample)
                obs = battle_select(selected)
            
            battle_finish()

            # Temporal discounting and Bellman target updates
            for player_idx in range(2):
                LAMBDA = 0.9
                final_outcome = obs["current"]["result"]
                value = 1.0 if player_idx == final_outcome else -1.0

                for sample in reversed(samples[player_idx]):
                    label = (value + sample.value) * 0.5
                    value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
                    sample.value = label
                    sample_list.append(sample)
    return sample_list

def train_epoch(model: MyModel, sample_list: list[LearnSample], optimizer: torch.optim.Optimizer, device: torch.device):
    """
    Runs a training epoch on the collected self-play data.
    """
    model.train()
    random.shuffle(sample_list)
    BATCH_SIZE = 128
    batch_count = len(sample_list) // BATCH_SIZE
    
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)

    for i in range(batch_count):
        input_enc = LearnInput()
        input_dec = LearnInput()
        mask = []
        label_enc = []
        label_dec = []
        start = BATCH_SIZE * i
        
        for j in range(start, start + BATCH_SIZE):
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

        # Convert data to PyTorch tensors
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)
        label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)
        label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)

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
        loss_dec = (loss_dec * mask_tensor).sum() / float(BATCH_SIZE)
        loss = loss_enc + loss_dec

        loss.backward()
        optimizer.step()

def main_train(num_iterations: int = 5, save_dir: str = "out"):
    """
    Main training pipeline loop.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load deck list
    deck = load_deck()

    # Initialize model
    model = MyModel(d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for iteration in range(num_iterations):
        model_path = os.path.join(save_dir, f"model{iteration}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\n--- Iteration {iteration} (Model saved to {model_path}) ---")

        # 1. Evaluate current model
        win_rate = run_evaluation(model, deck, num_games=50)
        print(f"Evaluation win rate vs random: {win_rate}%")

        # 2. Collect self-play data
        sample_list = collect_self_play_data(model, deck, num_episodes=100)

        # 3. Train on collected data
        print("Training Start...")
        train_epoch(model, sample_list, optimizer, device)
        print("Training Finish.")

    # Save final model
    final_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")

if __name__ == "__main__":
    main_train()
