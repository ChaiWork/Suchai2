import os
import sys
import torch

# Setup sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models import MyModel
from encoders import SparseVector, get_encoder_input, get_decoder_input
from agent import mcts_agent, random_agent, LearnSample
from train import load_deck, LearnInput, train_epoch
from cg.game import battle_start, battle_finish, battle_select

def run_tests():
    print("Starting automated verification pipeline...")

    # 1. Load Deck Test
    print("\n--- 1. Testing load_deck ---")
    deck = load_deck()
    assert len(deck) == 60, f"Deck must contain 60 cards, found {len(deck)}"
    print("Deck loaded successfully. Length: 60")

    # 2. Model Initialization Test
    print("\n--- 2. Testing model initialization ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device detected: {device}")
    model = MyModel(d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1)
    model = model.to(device)
    model.eval()
    print("Model initialized and moved to device successfully.")

    # 3. Simulator & MCTS Playout Test
    print("\n--- 3. Testing simulator loading and single-step MCTS ---")
    try:
        obs, start_data = battle_start(deck, deck)
        if start_data.battlePtr is None or start_data.battlePtr == 0:
            raise ValueError(f"Battle failed to start. errorPlayer: {start_data.errorPlayer}, errorType: {start_data.errorType}")
        
        print("Simulator started battle successfully.")
        
        # Run one step of MCTS (use small search count=2 for speed)
        selected_options, sample = mcts_agent(obs, deck, model, search_count=2)
        print(f"MCTS Agent returned move: {selected_options}")
        assert isinstance(selected_options, list), "MCTS must return a list of integer indices"
        assert isinstance(sample, LearnSample), "MCTS must return a training sample"
        print("MCTS single step playout verified.")

        # Test advancing game step
        next_obs = battle_select(selected_options)
        print("Simulator advanced battle step successfully.")
        
    finally:
        battle_finish()
        print("Simulator session finished and memory released.")

    # 4. Training Step Test
    print("\n--- 4. Testing training step optimization ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # We will duplicate the collected sample to form a batch of size 128
    print("Duplicating sample to form a batch of size 128...")
    samples = []
    for _ in range(128):
        # Create a deep-ish copy by using new sample objects
        s = LearnSample(
            value=sample.value,
            policy=list(sample.policy),
            sv_enc=sample.sv_enc,
            sv_dec=sample.sv_dec
        )
        samples.append(s)
        
    print("Running train_epoch for 1 batch step...")
    train_epoch(model, samples, optimizer, device)
    print("Training step backward pass completed successfully!")

    print("\n=============================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=============================================")

if __name__ == "__main__":
    run_tests()
