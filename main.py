import os
import sys
import torch

# Kaggle environment path constraint handling
if os.path.exists('/kaggle_simulations/agent/'):
    sys.path.append('/kaggle_simulations/agent/')
else:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models import MyModel
from agent import mcts_agent
from train import load_deck

# Resolve paths relative to the agent directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
DECK_PATH = os.path.join(BASE_DIR, "deck.csv")

# Determine computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model architecture
model = MyModel(d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1)

# Load trained weights if available; fall back to random weights if not
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        sys.stderr.write(f"Successfully loaded trained weights from {MODEL_PATH}\n")
    except Exception as e:
        sys.stderr.write(f"Error loading weights from {MODEL_PATH}: {e}. Initializing with random weights.\n")
else:
    sys.stderr.write(f"No checkpoint found at {MODEL_PATH}. Initializing with random weights.\n")

model = model.to(device)
model.eval()

# Load the deck configured for this agent
deck = load_deck(DECK_PATH)

def agent(obs_dict: dict) -> list[int]:
    """
    Entry point for the Kaggle environment. Calls the MCTS agent to choose
    the optimal actions given the observation.
    """
    with torch.no_grad():
        selected_options, _ = mcts_agent(obs_dict, deck, model, search_count=10)
    return selected_options
