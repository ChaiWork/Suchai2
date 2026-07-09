import glob
import math
import os
import random
import sys
import torch
import torch.nn
import torch.optim

from model import MyModel, SparseVector, LearnInput
from agent import LearnSample, mcts_agent, random_agent

# Resolve cg-lib path dynamically for Kaggle vs Local environments
try:
    cg_lib_path = glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0]
    sys.path.append(cg_lib_path)
except IndexError:
    pass

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


def main():
    # Load deck list
    deck_path = "deck.csv"
    if os.path.exists(deck_path):
        with open(deck_path, "r") as f:
            sample_deck = [int(line.strip()) for line in f if line.strip()]
    else:
        sample_deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

    # Setup device, model, optimizer, and loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = MyModel(128, 2, 256, 1, 1)
    
    # Load checkpoint if exists
    checkpoint_path = "model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder loss function
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Decoder loss function
    
    os.makedirs("out", exist_ok=True)

    # Main training loop
    for counter in range(5):
        # Save checkpoints
        epoch_model_path = f"out/model{counter}.pth"
        torch.save(model.state_dict(), epoch_model_path)
        torch.save(model.state_dict(), "model.pth")
        print(f"\nSaved checkpoint: {epoch_model_path} and model.pth")
        
        sample_list: list[LearnSample] = []
        
        # 1. Evaluation
        model.eval()
        with torch.inference_mode():
            results = [0, 0, 0]  # [Wins, Losses, Draws]

            for i in progress(50, "Evaluating... "):
                obs, start_data = battle_start(sample_deck, sample_deck)
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
                        selected, _ = mcts_agent(obs, sample_deck, model)
                    else:
                        selected = random_agent(obs)
                    obs = battle_select(selected)
                
                battle_finish()

                if obs["current"]["result"] == 2:  # Draw
                    results[2] += 1
                elif obs["current"]["result"] == your_index:  # Win
                    results[0] += 1
                else:  # Lose
                    results[1] += 1
            
            denom = results[0] + results[1]
            win_rate = (100 * results[0] // denom) if denom > 0 else 0
            print(f"Evaluation win rate vs random: {win_rate}% (Wins: {results[0]}, Losses: {results[1]}, Draws: {results[2]})", flush=True)

            # 2. Self Play Data Collection
            for _ in progress(100, "Training Data Collecting... "):
                obs, _ = battle_start(sample_deck, sample_deck)
                samples: list[list[LearnSample]] = [[], []]  # [Player0 samples, Player1 samples]
                while True:
                    if obs["current"]["result"] >= 0:
                        break

                    selected, sample = mcts_agent(obs, sample_deck, model)
                    samples[obs["current"]["yourIndex"]].append(sample)
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

        # 3. Model Training
        print("Training Start.")
        model.train()
        random.shuffle(sample_list)
        BATCH_SIZE = 128
        batch_count = len(sample_list) // BATCH_SIZE
        print(f"Total training samples: {len(sample_list)}, Batch Count: {batch_count}")
        
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

            # Convert to PyTorch tensors
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
            loss_dec = loss_dec * mask_tensor
            loss_dec = loss_dec.sum() / float(BATCH_SIZE)
            loss = loss_enc + loss_dec

            loss.backward()
            optimizer.step()
            
        print("Training Finish.")
        
    # Save the final model weights at the end of training
    torch.save(model.state_dict(), "model.pth")
    print("Final model saved as model.pth")


if __name__ == "__main__":
    main()
