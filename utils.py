import os
import sys
import time
import torch
from cg.api import all_card_data, all_attack, CardType
from model import (
    MyModel,
    MODEL_D_MODEL,
    MODEL_NUM_HEADS,
    MODEL_D_FEEDFORWARD,
    MODEL_NUM_LAYERS_ENCODER,
    MODEL_NUM_LAYERS_DECODER,
)

# Global Card Database and Helpers (shared across worker and main script)
card_table = {c.cardId: c for c in all_card_data()}
attack_table = {a.attackId: a for a in all_attack()}
evolves_from_set = {c.evolvesFrom for c in all_card_data() if c.evolvesFrom}

def get_card_data(card_id: int):
    return card_table.get(card_id)

def is_defensive_blocker(c):
    if c is None or c.cardType != CardType.POKEMON:
        return False
    return c.basic and not c.ex and (c.name not in evolves_from_set)


class ProgressBar:
    """Helper class to display training/evaluation progress in terminal with time estimation."""
    def __init__(self, count: int, text: str):
        self.count = count
        self.text = text.ljust(30)
        self.start_time = time.time()

    def update(self, current: int, suffix: str = ""):
        percent = min(100, 100 * current // self.count) if self.count > 0 else 100
        elapsed = time.time() - self.start_time
        
        # Estimate remaining time
        if current > 0:
            avg_time_per_item = elapsed / current
            est_total_time = avg_time_per_item * self.count
            est_remaining = est_total_time - elapsed
            
            elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
            rem_min, rem_sec = divmod(int(max(0, est_remaining)), 60)
            time_str = f"[{elapsed_min:02d}:{elapsed_sec:02d}<{rem_min:02d}:{rem_sec:02d}, {avg_time_per_item:.1f}s/game]"
        else:
            time_str = f"[00:00<--:--, --s/game]"
            
        # 15-char width progress bar
        bar_width = 15
        filled_width = int(bar_width * current // self.count) if self.count > 0 else bar_width
        bar = "█" * filled_width + "░" * (bar_width - filled_width)
        
        suffix_str = f" | {suffix}" if suffix else ""
        sys.stderr.write(f"\r{self.text} {bar} {current}/{self.count} ({percent}%) {time_str}{suffix_str}   ")
        sys.stderr.flush()
        if current >= self.count:
            sys.stderr.write("\n")
            sys.stderr.flush()


def load_all_decks():
    """Scans the workspace directories for deck.csv files and loads them.
    Includes all pre-defined easy and hard opponent decks from OPPONENT_DECKS.
    
    Returns:
        dict: A dictionary mapping folder name (deck name) to list of card IDs.
    """
    try:
        from agent import OPPONENT_DECKS
        decks = dict(OPPONENT_DECKS)
    except ImportError:
        try:
            from src.agent import OPPONENT_DECKS
            decks = dict(OPPONENT_DECKS)
        except ImportError:
            decks = {}
    
    # 1. Load root deck (our agent's main deck)
    root_deck_path = "deck.csv"
    if os.path.exists(root_deck_path):
        with open(root_deck_path, "r", encoding="utf-8-sig") as f:
            decks["Current (Self)"] = [int(line.strip()) for line in f if line.strip()]
            
    # 2. Scan decks directory for additional deck.csv files
    base_path = "decks"
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
        m = MyModel(
            MODEL_D_MODEL,
            MODEL_NUM_HEADS,
            MODEL_D_FEEDFORWARD,
            MODEL_NUM_LAYERS_ENCODER,
            MODEL_NUM_LAYERS_DECODER
        ).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            m.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            m.load_state_dict(checkpoint, strict=False)
        m.eval()
        # Keep cache small — only last 3 opponents
        if len(_league_cache) >= 3:
            oldest = next(iter(_league_cache))
            del _league_cache[oldest]
        _league_cache[path] = m
    return _league_cache[path]
