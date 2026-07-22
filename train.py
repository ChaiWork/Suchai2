import argparse
import csv
import glob
import math
import os
import random
import sys

# Configure D drive for temporary storage and PyTorch cache to save space on C drive
for path in ["D:/temp", "D:/torch_cache"]:
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass

os.environ["TMPDIR"] = "D:/temp"
os.environ["TEMP"] = "D:/temp"
os.environ["TMP"] = "D:/temp"
os.environ["TORCH_HOME"] = "D:/torch_cache"

import torch
import torch.nn
import torch.optim
import time
import queue
import threading
import multiprocessing as mp

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

from model import (
    MyModel,
    SparseVector,
    LearnInput,
    MODEL_D_MODEL,
    MODEL_NUM_HEADS,
    MODEL_D_FEEDFORWARD,
    MODEL_NUM_LAYERS_ENCODER,
    MODEL_NUM_LAYERS_DECODER,
)
from agent import LearnSample, mcts_agent, random_agent, GPUInferenceClient
from plot_metrics import plot_metrics

from cg.game import battle_start, battle_finish, battle_select
from cg.api import (
    to_observation_class, OptionType, AreaType, SelectContext,
    CardType, EnergyType, all_card_data, all_attack
)

# Global Card Database and Helpers
card_table = {c.cardId: c for c in all_card_data()}
attack_table = {a.attackId: a for a in all_attack()}
evolves_from_set = {c.evolvesFrom for c in all_card_data() if c.evolvesFrom}

def get_card_data(card_id: int):
    return card_table.get(card_id)

def is_defensive_blocker(c):
    if c is None or c.cardType != CardType.POKEMON:
        return False
    return c.basic and not c.ex and (c.name not in evolves_from_set)

def can_attack(card_id: int, energy_count: int):
    c = get_card_data(card_id)
    if c is None or c.cardType != CardType.POKEMON:
        return False
    for aid in c.attacks:
        att = attack_table.get(aid)
        if att is not None and energy_count >= len(att.energies):
            return True
    return False


def count_attached_energy(ps):
    count = 0
    if len(ps.active) > 0 and ps.active[0] is not None:
        count += len(ps.active[0].energyCards)
    for poke in ps.bench:
        if poke is not None:
            count += len(poke.energyCards)
    return count


def count_active_energy(ps):
    if len(ps.active) > 0 and ps.active[0] is not None:
        return len(ps.active[0].energyCards)
    return 0


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


def rule_based_opponent_agent(opponent_name, obs):
    mapping = {
        "Rulebasedmodel": ["main", "easy.main"],
        "Rulebasedmodel_Iono": ["iono_agent", "hard.iono_agent"],
        "Rulebasedmodel_Dragapult": ["dragapult_agent", "hard.dragapult_agent"],
        "Rulebasedmodel_Mewtwo": ["mewtwo_agent", "easy.mewtwo_agent"],
        "Rulebasedmodel_Mewtwo_Easy": ["mewtwo_agent_easy", "easy.mewtwo_agent_easy"],
        "Rulebasedmodel_Abomasnow": ["abomasnow_agent", "easy.abomasnow_agent"],
        "Rulebasedmodel_Lucario": ["lucario_agent", "hard.lucario_agent"],
        "Rulebasedmodel_Crustle": ["crustle_agent", "easy.crustle_agent"],
        "Rulebasedmodel_Starmie": ["starmie_agent", "hard.starmie_agent"],
        "Rulebasedmodel_Dipplin": ["dipplin_agent", "hard.dipplin_agent"],
        "Rulebasedmodel_Mewtwo_Wobbuffet": ["mewtwo_wobbuffet_agent", "hard.mewtwo_wobbuffet_agent"]
    }
    
    modules = mapping.get(opponent_name, [opponent_name])
    for mod_path in modules:
        try:
            import importlib
            mod = importlib.import_module(f"Rulebasedmodel.{mod_path}")
            return mod.agent(obs)
        except Exception:
            pass
            
    raise ValueError(f"Unknown rule-based opponent: {opponent_name}")


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


class GPUInferenceServer:
    """Collects and batches NN evaluations from parallel game workers."""
    def __init__(self, model, parent_conns, model_lock, batch_size=64, timeout=0.005):
        self.model = model
        self.parent_conns = parent_conns
        self.model_lock = model_lock
        self.batch_size = batch_size
        self.timeout = timeout
        self.running = True
        self.thread = None

    def start(self, device):
        self.thread = threading.Thread(target=self._loop, args=(device,))
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _loop(self, device):
        while self.running:
            ready_conns = mp.connection.wait(self.parent_conns, timeout=self.timeout)
            if not ready_conns or not self.running:
                continue

            batch_conns = []
            batch_sv_enc = []
            batch_sv_dec = []

            for conn in ready_conns:
                if len(batch_conns) >= self.batch_size:
                    break
                try:
                    if conn.poll():
                        sv_enc, sv_dec = conn.recv()
                        batch_conns.append(conn)
                        batch_sv_enc.append(sv_enc)
                        batch_sv_dec.append(sv_dec)
                except (EOFError, OSError):
                    if conn in self.parent_conns:
                        self.parent_conns.remove(conn)

            if not batch_conns:
                continue

            input_enc = LearnInput()
            for sv in batch_sv_enc:
                input_enc.add(sv)

            orig_lens = []
            input_dec = LearnInput()
            for sv in batch_sv_dec:
                orig_len = len(sv.offset)
                orig_lens.append(orig_len)
                if orig_len < 256:
                    for _ in range(256 - orig_len):
                        sv.offset.append(len(sv.index))
                input_dec.add(sv)

            try:
                with self.model_lock:
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')), torch.inference_mode():
                        out_enc, out_dec = self.model(
                            torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                            torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                            torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                            torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                            torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                            torch.tensor(input_dec.offset, dtype=torch.int32, device=device)
                        )
                        values = out_enc.squeeze(-1).tolist()
                        policies = out_dec.tolist()
            except Exception as e:
                import traceback
                traceback.print_exc()
                values = [0.0] * len(batch_conns)
                policies = [[0.0] * 256] * len(batch_conns)

            for idx, conn in enumerate(batch_conns):
                val = values[idx]
                pol = policies[idx][:orig_lens[idx]]
                try:
                    conn.send((val, pol))
                except (OSError, IOError):
                    pass


def worker_loop(worker_id, command_queue, result_queue, inference_conn, device_str):
    """The game worker execution loop."""
    random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)
    device = torch.device(device_str)
    
    client = GPUInferenceClient(inference_conn)
    
    while True:
        try:
            cmd, args = command_queue.get()
        except KeyboardInterrupt:
            break
            
        if cmd == "STOP":
            break
            
        elif cmd == "PLAY_SELF":
            sample_deck, opponent_deck, opponent_type, opponent_path = args[:4]
            opponent_name = args[4] if len(args) > 4 else "Current (Self)"
            current_epoch = args[5] if len(args) > 5 else 0  # epoch counter for warmup schedule
            
            opp_model = None
            if opponent_path is not None:
                try:
                    _opp_model_module = MyModel(
                        MODEL_D_MODEL,
                        MODEL_NUM_HEADS,
                        MODEL_D_FEEDFORWARD,
                        MODEL_NUM_LAYERS_ENCODER,
                        MODEL_NUM_LAYERS_DECODER
                    ).to(device)
                    checkpoint = torch.load(opponent_path, map_location=device, weights_only=True)
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        _opp_model_module.load_state_dict(checkpoint["state_dict"], strict=False)
                    else:
                        _opp_model_module.load_state_dict(checkpoint, strict=False)
                    _opp_model_module.eval()
                    # FIX #10: wrap in GPUInferenceClient — mcts_agent expects .infer(), not nn.Module
                    opp_model = GPUInferenceClient(_opp_model_module, device)
                except Exception as e:
                    opp_model = None
            
            try:
                obs, start_data = battle_start(sample_deck, opponent_deck)
                if start_data.errorPlayer >= 0:
                    result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, [], -1, 0, {}, 0.0, 0, {}, {})))
                    continue
            except Exception as e:
                result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, [], -1, 0, {}, 0.0, 0, {}, {})))
                continue

            try:
                action_counts = {"attack": 0, "play": 0, "attach": 0, "evolve": 0, "ability": 0, "retreat": 0, "end": 0, "other": 0}
                played_cards = {}
                samples = [[], []]
                
                # Episode-state active spot lockout trackers
                episode_lockouts = [0, 0]
                episode_active_serials = [None, None]
                episode_attacked = [False, False]
                episode_last_turns = [None, None]
                
                while True:
                    if obs["current"]["result"] >= 0:
                        break
    
                    curr_player = obs["current"]["yourIndex"]
                    curr_deck = sample_deck if curr_player == 0 else opponent_deck
                    
                    obs_class = to_observation_class(obs)
                    state_ps = obs_class.current.players[curr_player]
                    opp_ps = obs_class.current.players[1 - curr_player]
                    
                    active_pk = state_ps.active[0] if (len(state_ps.active) > 0 and state_ps.active[0] is not None) else None
                    active_id = active_pk.id if active_pk else -1
                    active_energies = len(active_pk.energyCards) if active_pk else 0
                    
                    # Track active spot lockout at the episode level
                    active_serial = active_pk.serial if active_pk else None
                    turn_num = obs_class.current.turn
                    
                    if episode_active_serials[curr_player] != active_serial:
                        episode_active_serials[curr_player] = active_serial
                        episode_lockouts[curr_player] = 0
                        episode_attacked[curr_player] = False
                        
                    if episode_last_turns[curr_player] is not None and turn_num != episode_last_turns[curr_player]:
                        if not episode_attacked[curr_player]:
                            episode_lockouts[curr_player] += 1
                        episode_attacked[curr_player] = False
                    episode_last_turns[curr_player] = turn_num
                    
                    opp_active_pk = opp_ps.active[0] if (len(opp_ps.active) > 0 and opp_ps.active[0] is not None) else None
                    opp_active_id = opp_active_pk.id if opp_active_pk else -1
                    
                    bench_list = [p for p in state_ps.bench if p is not None]
                    bench_ids = [p.id for p in bench_list]
                    
                    has_attack_option = False
                    has_attach_option = False
                    if obs_class.select is not None and obs_class.select.option is not None:
                        for opt in obs_class.select.option:
                            if opt.type == OptionType.ATTACK:
                                has_attack_option = True
                            elif opt.type == OptionType.ATTACH:
                                has_attach_option = True

                    stadium_id = obs_class.current.stadium[0].id if (len(obs_class.current.stadium) > 0 and obs_class.current.stadium[0] is not None) else -1
                    
                    pre_metrics = {
                        "prizes": len(state_ps.prize),
                        "opp_prizes": len(opp_ps.prize),
                        "energy": count_attached_energy(state_ps),
                        "active_energy": count_active_energy(state_ps),
                        "pokemon": count_pokemon(state_ps),
                        "opp_pokemon": count_pokemon(opp_ps),
                        "bench_size": len(bench_list),
                        "deck_size": state_ps.deckCount,
                        "opp_deck_size": opp_ps.deckCount,
                        "energy_attached_flag": obs_class.current.energyAttached,
                        "active_id": active_id,
                        "active_energies": active_energies,
                        "opp_active_id": opp_active_id,
                        "active_hp": active_pk.hp if active_pk else 0,
                        "opp_active_hp": opp_active_pk.hp if opp_active_pk else 0,
                        "stadium_id": stadium_id,
                        "bench_ids": bench_ids,
                        "bench_energies": [len(p.energyCards) for p in state_ps.bench if p is not None],
                        "bench_damage": [p.maxHp - p.hp for p in state_ps.bench if p is not None],
                        "hand_size": len(state_ps.hand) if state_ps.hand is not None else 0,
                        "hand_ids": [c.id for c in state_ps.hand if c is not None] if state_ps.hand is not None else [],
                        "discard_size": len(state_ps.discard) if state_ps.discard is not None else 0,
                        "discard_energy": sum(1 for c in state_ps.discard if c is not None and c.id in [1, 5, 15]),
                        "turn": obs_class.current.turn,
                        "has_attack_option": has_attack_option,
                        "has_attach_option": has_attach_option,
                        "context": obs_class.select.context if (obs_class.select is not None) else None,
                        "lockout_turns": episode_lockouts[curr_player]
                    }
    
                    if curr_player == 0:
                        turn = obs_class.current.turn if (obs_class.current is not None) else 1
                        warmup_threshold = max(0, 5 - (current_epoch // 5))
                        
                        use_warmup = (turn <= warmup_threshold)
                        if use_warmup:
                            try:
                                rb_selected = rule_based_opponent_agent("Rulebasedmodel_Mewtwo", obs)
                                # Run MCTS with force_action to collect Behavioral Cloning data
                                temperature = 1.0 if turn <= 15 else 0.1
                                selected, sample = mcts_agent(obs, curr_deck, client, search_count=50, temperature=temperature, force_action=rb_selected)
                            except Exception as e:
                                use_warmup = False
                                
                        if not use_warmup:
                            # Temperature decay: explore fully early, exploit late-game
                            temperature = 1.0 if turn <= 15 else 0.1
                            selected, sample = mcts_agent(obs, curr_deck, client, search_count=50, temperature=temperature)
                            sample.pred_val = sample.value
                            
                            opt_type_val = -1
                            played_card_id = -1
                            attached_card_id = -1
                            attached_target_id = -1
                            evolved_card_id = -1
                            attack_id = -1
                            if selected and len(selected) > 0:
                                sel_idx = selected[0]
                                options = obs.get("select", {}).get("option", [])
                                if sel_idx < len(options):
                                    opt = options[sel_idx]
                                    opt_type_val = opt.get("type", -1)
                                    if opt_type_val == 7: # PLAY
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            played_card_id = hand[card_idx].id
                                            played_cards[played_card_id] = played_cards.get(played_card_id, 0) + 1
                                    elif opt_type_val == 8: # ATTACH
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            attached_card_id = hand[card_idx].id
                                        # Get target pokemon ID
                                        target_area = opt.get("inPlayArea", -1)
                                        target_idx = opt.get("inPlayIndex", -1)
                                        if target_area == 4: # ACTIVE
                                            if len(state_ps.active) > 0 and state_ps.active[0] is not None:
                                                attached_target_id = state_ps.active[0].id
                                        elif target_area == 5: # BENCH
                                            if 0 <= target_idx < len(state_ps.bench) and state_ps.bench[target_idx] is not None:
                                                attached_target_id = state_ps.bench[target_idx].id
                                    elif opt_type_val == 9: # EVOLVE
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            evolved_card_id = hand[card_idx].id
                                    elif opt_type_val == 13: # ATTACK
                                        attack_id = opt.get("attackId", -1)
                                        
                            pre_metrics["action_type"] = opt_type_val
                            pre_metrics["played_card_id"] = played_card_id
                            pre_metrics["attached_card_id"] = attached_card_id
                            pre_metrics["attached_target_id"] = attached_target_id
                            pre_metrics["evolved_card_id"] = evolved_card_id
                            pre_metrics["attack_id"] = attack_id
                            samples[0].append((sample, pre_metrics))

                        # Log actions to action_counts (always, even for warmup)
                        if selected and len(selected) > 0:
                            sel_idx = selected[0]
                            options = obs.get("select", {}).get("option", [])
                            if sel_idx < len(options):
                                opt_type = options[sel_idx].get("type")
                                if opt_type == 13:
                                    action_counts["attack"] += 1
                                elif opt_type == 7:
                                    action_counts["play"] += 1
                                elif opt_type == 8:
                                    action_counts["attach"] += 1
                                elif opt_type == 9:
                                    action_counts["evolve"] += 1
                                elif opt_type == 10:
                                    action_counts["ability"] += 1
                                elif opt_type == 12:
                                    action_counts["retreat"] += 1
                                elif opt_type == 14:
                                    action_counts["end"] += 1
                                else:
                                    action_counts["other"] += 1
                    else:
                        if opponent_name.startswith("Rulebasedmodel"):
                            try:
                                selected = rule_based_opponent_agent(opponent_name, obs)
                            except Exception as e:
                                selected = random_agent(obs)
                        elif opponent_type == "Current":
                            # Temperature decay mirrors player-0 for consistent training distribution
                            opp_temperature = 1.0 if turn_num <= 15 else 0.1
                            selected, sample = mcts_agent(obs, curr_deck, client, search_count=50, temperature=opp_temperature)
                            sample.pred_val = sample.value
                            
                            opt_type_val = -1
                            played_card_id = -1
                            attached_card_id = -1
                            attached_target_id = -1
                            evolved_card_id = -1
                            attack_id = -1
                            if selected and len(selected) > 0:
                                sel_idx = selected[0]
                                options = obs.get("select", {}).get("option", [])
                                if sel_idx < len(options):
                                    opt = options[sel_idx]
                                    opt_type_val = opt.get("type", -1)
                                    if opt_type_val == 7: # PLAY
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            played_card_id = hand[card_idx].id
                                    elif opt_type_val == 8: # ATTACH
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            attached_card_id = hand[card_idx].id
                                        # Get target pokemon ID
                                        target_area = opt.get("inPlayArea", -1)
                                        target_idx = opt.get("inPlayIndex", -1)
                                        if target_area == 4: # ACTIVE
                                            if len(state_ps.active) > 0 and state_ps.active[0] is not None:
                                                attached_target_id = state_ps.active[0].id
                                        elif target_area == 5: # BENCH
                                            if 0 <= target_idx < len(state_ps.bench) and state_ps.bench[target_idx] is not None:
                                                attached_target_id = state_ps.bench[target_idx].id
                                    elif opt_type_val == 9: # EVOLVE
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            evolved_card_id = hand[card_idx].id
                                    elif opt_type_val == 13: # ATTACK
                                        attack_id = opt.get("attackId", -1)
                                        
                            pre_metrics["action_type"] = opt_type_val
                            pre_metrics["played_card_id"] = played_card_id
                            pre_metrics["attached_card_id"] = attached_card_id
                            pre_metrics["attached_target_id"] = attached_target_id
                            pre_metrics["evolved_card_id"] = evolved_card_id
                            pre_metrics["attack_id"] = attack_id
                            samples[1].append((sample, pre_metrics))
                        elif opp_model is not None:
                            selected, sample = mcts_agent(obs, curr_deck, opp_model, search_count=50)
                        else:
                            selected = random_agent(obs)
                    
                    # Record if active player attacked in this step
                    if selected and len(selected) > 0:
                        sel_idx = selected[0]
                        options = obs.get("select", {}).get("option", [])
                        if sel_idx < len(options):
                            opt = options[sel_idx]
                            opt_type = opt.get("type", -1)
                            if opt_type == 13 or opt_type == OptionType.ATTACK:
                                episode_attacked[curr_player] = True
                                
                    obs = battle_select(selected)
                    
                battle_finish()
            except Exception as e:
                import traceback
                print(f"Error in worker {worker_id} simulation:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, [], -1, 0, {}, 0.0, 0, {}, {})))
                continue
            
            result = obs["current"]["result"]
            obs_class = to_observation_class(obs)
            final_turn = obs_class.current.turn if (obs_class is not None and obs_class.current is not None) else 0
            
            processed_samples = []
            rc_worker = {"prize_taken": 0.0, "prize_lost": 0.0, "kos": 0.0, "own_kos": 0.0,
                         "energy": 0.0, "bench": 0.0, "deckout": 0.0, "terminal": 0.0, "stall": 0.0,
                         "no_energy": 0.0, "strategic": 0.0}
            
            for i in range(2):
                player_samples = samples[i]
                n_steps = len(player_samples)
                if n_steps == 0:
                    continue
                    
                # Track if player went second (their first turn occurred on game turn 2)
                first_turn_of_player = player_samples[0][1].get("turn", 1)
                went_second = (first_turn_of_player == 2)
                
                num_attacks = sum(1 for _, pre in player_samples if pre.get("action_type") == 13)
                if i == result:
                    if num_attacks == 0:
                        # Zero reward for wins without attacking to prevent passive stall exploits.
                        # Force the model to learn that active attacking is the path to win.
                        terminal_reward = 0.0
                    else:
                        terminal_reward = 5.0
                        if went_second:
                            terminal_reward += 1.0  # Extra +1.0 reward for winning when starting second!
                elif result == 2:
                    terminal_reward = 0.0   # Draw: neutral outcome — not penalised like a loss
                elif result == -1:
                    terminal_reward = -5.0  # Error/invalid game — treat as loss
                else:
                    # FIX #5: Prize-aware loss — softer penalty for competitive losses
                    # -5.0 (0 prizes taken) up to -2.5 (5 prizes taken before losing)
                    start_prizes = player_samples[0][1].get("prizes", 6)
                    end_prizes = player_samples[-1][1].get("prizes", 6)
                    prizes_taken_total = max(0, start_prizes - end_prizes)
                    terminal_reward = max(-5.0, -5.0 + prizes_taken_total * 0.5)

                    
                rewards = []
                has_attacked_flag = False
                has_taken_prize_flag = False
                has_lost_prize_flag = False  # FIX #4: symmetric first-prize-lost penalty
                for step_idx in range(n_steps):
                    sample_obj, pre = player_samples[step_idx]
                    
                    if step_idx < n_steps - 1:
                        _, post = player_samples[step_idx + 1]
                    else:
                        final_obs = to_observation_class(obs)
                        final_ps = final_obs.current.players[i]
                        final_opp_ps = final_obs.current.players[1 - i]
                        final_active_pk = final_ps.active[0] if (len(final_ps.active) > 0 and final_ps.active[0] is not None) else None
                        final_active_id = final_active_pk.id if final_active_pk else -1
                        final_active_energies = len(final_active_pk.energyCards) if final_active_pk else 0
                        
                        final_opp_active_pk = final_opp_ps.active[0] if (len(final_opp_ps.active) > 0 and final_opp_ps.active[0] is not None) else None
                        final_opp_active_id = final_opp_active_pk.id if final_opp_active_pk else -1
                        
                        final_bench_list = [p for p in final_ps.bench if p is not None]
                        final_bench_ids = [p.id for p in final_bench_list]
                        final_stadium_id = final_obs.current.stadium[0].id if (len(final_obs.current.stadium) > 0 and final_obs.current.stadium[0] is not None) else -1
                        
                        post = {
                            "prizes": len(final_ps.prize),
                            "opp_prizes": len(final_opp_ps.prize),
                            "energy": count_attached_energy(final_ps),
                            "active_energy": count_active_energy(final_ps),
                            "pokemon": count_pokemon(final_ps),
                            "opp_pokemon": count_pokemon(final_opp_ps),
                            "bench_size": len(final_bench_list),
                            "deck_size": final_ps.deckCount,
                            "opp_deck_size": final_opp_ps.deckCount,
                            "energy_attached_flag": True,
                            "active_id": final_active_id,
                            "active_energies": final_active_energies,
                            "opp_active_id": final_opp_active_id,
                            "active_hp": final_active_pk.hp if final_active_pk else 0,
                            "opp_active_hp": final_opp_active_pk.hp if final_opp_active_pk else 0,
                            "stadium_id": final_stadium_id,
                            "bench_ids": final_bench_ids,
                            "bench_energies": [len(p.energyCards) for p in final_ps.bench if p is not None],
                            "bench_damage": [p.maxHp - p.hp for p in final_ps.bench if p is not None],
                            "hand_size": len(final_ps.hand) if final_ps.hand is not None else 0,
                            "hand_ids": [c.id for c in final_ps.hand if c is not None] if final_ps.hand is not None else [],
                            "discard_size": len(final_ps.discard) if final_ps.discard is not None else 0,
                            "discard_energy": sum(1 for c in final_ps.discard if c is not None and c.id in [1, 5, 15]),
                            "turn": final_obs.current.turn,
                            "has_attack_option": False,
                            "has_attach_option": False,
                            "lockout_turns": episode_lockouts[i]
                        }
                    
                    prizes_taken = pre["prizes"] - post["prizes"]
                    prizes_lost = pre["opp_prizes"] - post["opp_prizes"]
                    opp_kos = pre["opp_pokemon"] - post["opp_pokemon"]
                    own_kos = pre["pokemon"] - post["pokemon"]
                    energy_attached = post["energy"] - pre["energy"]
                    active_energy_attached = post.get("active_energy", 0) - pre.get("active_energy", 0)
                    bench_energy_attached = energy_attached - active_energy_attached
                    
                    # Read active spot lockout from pre state
                    lockout_turns = pre.get("lockout_turns", 0)
                        
                    # Stall penalty to prevent endless pass cycles
                    # If opponent is close to deckout, do NOT penalise stalling
                    opp_deck_size = pre.get("opp_deck_size", 40)
                    if opp_deck_size <= 5:
                        r_stall = 0.0  # Prevent reward farming
                    else:
                        r_stall = -0.02
                        if lockout_turns > 3:
                            r_stall -= min(0.20, 0.02 * (lockout_turns - 3))
                        
                    r_prize_t = prizes_taken * 0.5 if prizes_taken > 0 else 0.0
                    if prizes_taken > 0 and not has_taken_prize_flag:
                        r_prize_t += 0.50
                        has_taken_prize_flag = True
                    r_prize_l = prizes_lost * 0.35 if prizes_lost > 0 else 0.0  # Raised from 0.10 — symmetric with prize-taken to teach defensive play
                    # FIX #4: symmetric first-prize-lost penalty to match first-prize-taken bonus
                    if prizes_lost > 0 and not has_lost_prize_flag:
                        r_prize_l += 0.35
                        has_lost_prize_flag = True
                    r_ko = 0.0  # Removed: KO double-counts with r_prize_t (same game event triggers both)
                    r_own_ko = own_kos * 0.08 if own_kos > 0 else 0.0
                    
                    r_en = 0.0
                    # r_en deliberately omitted: type-aware r_strategic (below) provides
                    # the energy attachment signal. A separate baseline here partially
                    # cancels the type-mismatch penalty (audit item 1.3).
                        
                    # Principal RL Scientist - Optimized Mewtwo ex Reward Shaping
                    r_strategic = 0.0
                    action_type = pre.get("action_type", -1)

                    # Going second tactical exploitation bonus on first turn
                    if went_second and step_idx == 0:
                        if action_type == 13:  # ATTACK on turn 1 going second (exploiting going second attack rule)
                            r_strategic += 0.35
                        elif action_type in [7, 8]:  # PLAY / ATTACH on turn 1 going second
                            r_strategic += 0.15

                    is_tr_pokemon = lambda cid: (get_card_data(cid) is not None and 
                                                 get_card_data(cid).cardType == CardType.POKEMON and 
                                                 "rocket" in get_card_data(cid).name.lower())

                    # Tactic 1: Race the TR ability gate (Mewtwo ex needs 4 TR mons boarded to attack)
                    active_id_post = post.get("active_id", -1)
                    bench_ids_post = post.get("bench_ids", [])
                    tr_count = 0
                    if is_tr_pokemon(active_id_post):
                        tr_count += 1
                    for bid in bench_ids_post:
                        if is_tr_pokemon(bid):
                            tr_count += 1
                    
                    if tr_count < 4:
                        # Reward playing/boarding a TR Pokemon
                        if action_type == 7:  # PLAY
                            played_id = pre.get("played_card_id", -1)
                            if is_tr_pokemon(played_id):
                                r_strategic += 0.25
                        # FIX #3: Removed Tactic-1 search card bonus — it double-fired with the
                        # per-card contextual reward block added later (e.g. Ariana → +0.15 here
                        # + up to +0.30 in the Ariana branch = accidental +0.45 stacking).
                        # Penalize attaching to Mewtwo early if we have < 3 TR mons
                        if action_type == 8 and tr_count < 3:  # ATTACH
                            attached_target = pre.get("attached_target_id", -1)
                            if attached_target == 431:  # Mewtwo ex
                                r_strategic -= 0.10
                    
                    # 0. Active Spot Promotion Guard (SelectContext.TO_ACTIVE)
                    if pre.get("context") == SelectContext.TO_ACTIVE:
                        promoted_id = post.get("active_id", -1)
                        promoted_energy = post.get("active_energies", 0)
                        
                        promoted_c = get_card_data(promoted_id)
                        promoted_can_attack = can_attack(promoted_id, promoted_energy)
                        promoted_is_blocker = is_defensive_blocker(promoted_c)
                        
                        # Identify benched options before promotion
                        benched_options = []
                        for idx, bid in enumerate(pre.get("bench_ids", [])):
                            b_c = get_card_data(bid)
                            if b_c is not None and b_c.cardType == CardType.POKEMON:
                                b_energy = pre.get("bench_energies", [])[idx] if idx < len(pre.get("bench_energies", [])) else 0
                                b_can_attack = can_attack(bid, b_energy)
                                b_is_blocker = is_defensive_blocker(b_c)
                                benched_options.append((bid, b_energy, b_can_attack, b_is_blocker))
                                
                        if len(benched_options) > 0:
                            is_meaningful = promoted_can_attack or promoted_is_blocker
                            bench_has_ready_attacker = any(opt[2] for opt in benched_options)
                            
                            if bench_has_ready_attacker and not promoted_can_attack:
                                # Left a ready attacker on bench!
                                r_strategic -= 0.50
                            elif not is_meaningful:
                                # Promoted a non-blocker, non-attacker when we had other options
                                r_strategic -= 0.20
                            elif promoted_is_blocker:
                                # Promoted a blocker (meaningful play to stall/protect bench)
                                r_strategic += 0.20
                    
                    # 1. Stadium Establishment
                    if pre.get("stadium_id") != post.get("stadium_id") and post.get("stadium_id") != -1:
                        stadium_card = get_card_data(post.get("stadium_id"))
                        if stadium_card is not None and stadium_card.cardType == CardType.STADIUM:
                            if action_type == 7:  # PLAY
                                r_strategic += 0.10  # Minor board control reward
                    
                    # 2. Action Heuristics & Search Efficiency & Sequencing
                    if action_type == 7:  # PLAY
                        played_id = pre.get("played_card_id", -1)
                        played_card = get_card_data(played_id)
                        
                        if played_card is not None:
                            # Generic trainer/tool/stadium cards played without custom handler blocks below
                            if played_id in [1116, 1159, 1175, 1121, 1134, 1257]:
                                r_strategic += 0.08
                                
                            # Board development: play basic Pokemon to bench
                            if played_card.cardType == CardType.POKEMON:
                                if played_card.basic:
                                    if played_card.ex:
                                        r_strategic += 0.10  # Basic ex setup reward
                                    elif played_card.name in evolves_from_set:
                                        r_strategic += 0.15  # Basic evolvable setup reward
                                    else:
                                        r_strategic += 0.05  # Other basic setup reward
                                        
                            # Search efficiency: playing item/supporter with search/draw text
                            elif played_card.cardType in [CardType.ITEM, CardType.SUPPORTER] and any(
                                kw in played_card.name.lower() or kw in getattr(played_card, "text", "").lower() 
                                for kw in ["search", "draw", "look", "put"]
                            ):
                                # If our deck size is dangerously low, penalise draw/search cards to avoid self-deckout
                                my_deck_size = pre.get("deck_size", 40)
                                if my_deck_size <= 5:
                                    r_strategic -= 0.10  # Heavy penalty for draw/search when low on deck!
                                else:
                                    expected_min_diff = -2 if "Ultra Ball" in played_card.name else 0
                                    hand_diff = post["hand_size"] - pre["hand_size"]
                                    if hand_diff >= expected_min_diff:
                                        r_strategic += 0.10  # Search/draw success
                                    
                                    # Ultra Ball Overall Resource Efficiency Evaluation
                                    if "Ultra Ball" in played_card.name:
                                        pre_hand = list(pre.get("hand_ids", []))
                                        post_hand = list(post.get("hand_ids", []))
                                        if played_id in pre_hand:
                                            pre_hand.remove(played_id)
                                            
                                        from collections import Counter
                                        pre_counts = Counter(pre_hand)
                                        post_counts = Counter(post_hand)
                                        discarded_ids = []
                                        for cid, count in pre_counts.items():
                                            diff = count - post_counts.get(cid, 0)
                                            if diff > 0:
                                                discarded_ids.extend([cid] * diff)
                                                
                                        efficiency_score = 0.0
                                        for dcid in discarded_ids:
                                            dc = get_card_data(dcid)
                                            if dc is not None:
                                                if dc.cardType == CardType.BASIC_ENERGY:
                                                    # Check if we have Spidops in play to charge or Night Stretcher in hand
                                                    has_spidops = any(bid == 401 for bid in pre.get("bench_ids", [])) or pre.get("active_id") == 401
                                                    has_stretcher = 1097 in post_hand
                                                    if has_spidops or has_stretcher:
                                                        efficiency_score += 0.10
                                                elif post_counts.get(dcid, 0) > 0:
                                                    efficiency_score += 0.05  # Duplicate card discard
                                                elif dc.cardType == CardType.STADIUM and pre.get("stadium_id") == dcid:
                                                    efficiency_score += 0.08  # Duplicate stadium discard
                                                elif dc.cardType in [CardType.SUPPORTER, CardType.TOOL]:
                                                    efficiency_score -= 0.10  # Penalize unique Supporter/Tool discard
                                        r_strategic += efficiency_score
                                    else:
                                        r_strategic -= 0.05
                                    
                            # Correct use of Energy Switch
                            elif played_card.cardType == CardType.ITEM and "Energy Switch" in played_card.name:
                                active_c = get_card_data(post["active_id"])
                                if active_c is not None:
                                    active_att_costs = [len(attack_table.get(aid).energies) for aid in active_c.attacks if attack_table.get(aid)]
                                    min_att_cost = min(active_att_costs) if active_att_costs else 99
                                    if post["active_energies"] >= min_att_cost and pre["active_energies"] < min_att_cost:
                                        r_strategic += 0.25  # High swing turn reward
                                    elif post["active_energies"] > pre["active_energies"]:
                                        r_strategic += 0.10
                                    else:
                                        r_strategic -= 0.05
                                                       # Team Rocket's Transceiver play logic
                            elif "Transceiver" in played_card.name:
                                hand_diff = post["hand_size"] - pre["hand_size"]
                                if hand_diff >= 0:
                                    r_strategic += 0.10  # Transceiver search success
                                else:
                                    r_strategic -= 0.05
                                    
                            # Buddy-Buddy Poffin / Basic Search (1086): reward searching basic Pokemon for bench setup
                            elif played_id == 1086:
                                pre_bench  = len(pre.get("bench_ids", []))
                                post_bench = len(post.get("bench_ids", []))
                                mons_benched = post_bench - pre_bench
                                if mons_benched >= 2:
                                    r_strategic += 0.30  # Dual basic bench setup — max Spidops Rocket Rush value
                                elif mons_benched == 1:
                                    r_strategic += 0.20  # Single basic setup
                                else:
                                    r_strategic += 0.10  # Search item played

                            # Switch (1123): reward strategic switching to ready attacker or escaping bad matchup
                            elif played_id == 1123:
                                pre_active_id  = pre.get("active_id", -1)
                                post_active_id = post.get("active_id", -1)
                                post_active_c  = get_card_data(post_active_id)
                                is_post_ready  = post_active_c is not None and post.get("active_energies", 0) >= 1
                                if post_active_id in [414, 401, 431] and is_post_ready:
                                    r_strategic += 0.25  # Strategic switch into ready attacker (Articuno/Spidops/Mewtwo)
                                elif pre_active_id == 431 and pre.get("active_hp", 280) <= 140:
                                    r_strategic += 0.20  # Strategic escape from damaged Mewtwo ex
                                else:
                                    r_strategic += 0.10  # Standard switch mobility

                            # Bug Catching Set: rewards finding Grass Pokemon or Grass Energy into hand
                            elif played_id == 1094:  # Bug Catching Set
                                post_hand = post.get("hand_ids", [])
                                pre_hand  = pre.get("hand_ids", [])
                                grass_mon_ids = [400, 401]  # Tarountula, Spidops
                                grass_energy_id = 1  # Basic Grass Energy
                                found_grass_mon    = any(cid in post_hand and cid not in pre_hand for cid in grass_mon_ids)
                                found_grass_energy = post_hand.count(grass_energy_id) > pre_hand.count(grass_energy_id)
                                spidops_in_play = (pre.get("active_id") == 401 or 401 in pre.get("bench_ids", []))
                                if found_grass_mon and not spidops_in_play:
                                    r_strategic += 0.25  # Found Spidops line when not yet on field
                                elif found_grass_mon:
                                    r_strategic += 0.15  # Found TR mon for wider Rocket Rush board
                                elif found_grass_energy:
                                    r_strategic += 0.12  # Found Grass Energy for charging
                                else:
                                    r_strategic -= 0.08  # Whiffed — nothing found

                            # Night Stretcher: rewards recovering a high-value KO'd attacker or energy
                            elif played_id == 1097:  # Night Stretcher
                                pre_discard  = pre.get("discard_ids", [])
                                post_hand    = post.get("hand_ids", [])
                                pre_hand     = pre.get("hand_ids", [])
                                attacker_ids = [431, 401, 400]  # Mewtwo ex, Spidops, Tarountula
                                energy_ids   = [15, 1, 5]  # TR Energy, Grass, Psychic
                                got_attacker = any(cid in post_hand and cid not in pre_hand for cid in attacker_ids)
                                got_energy   = any(cid in post_hand and cid not in pre_hand for cid in energy_ids)
                                has_attacker_discard = any(cid in pre_discard for cid in attacker_ids)
                                prizes_opp = pre.get("prizes_remaining_opp", 3)
                                if got_attacker and prizes_opp <= 2:
                                    r_strategic += 0.40  # Emergency: recovered attacker when opponent is close to winning
                                elif got_attacker:
                                    r_strategic += 0.25  # Recovered a KO'd attacker
                                elif got_energy:
                                    r_strategic += 0.15  # Recovered energy (good secondary use)
                                elif has_attacker_discard and not got_attacker:
                                    r_strategic -= 0.10  # Should have recovered attacker but didn't
                                else:
                                    r_strategic += 0.05  # Neutral recovery

                            # Team Rocket's Archer: reward accelerating energy to useful Pokemon
                            elif played_id == 1217:  # Team Rocket's Archer
                                post_bench_energies = post.get("bench_energies", [])
                                pre_bench_energies  = pre.get("bench_energies", [])
                                bench_energy_gained = sum(post_bench_energies) - sum(pre_bench_energies)
                                active_energy_gained = post.get("active_energies", 0) - pre.get("active_energies", 0)
                                total_energy_gained = bench_energy_gained + active_energy_gained
                                post_bench = post.get("bench_ids", [])
                                spidops_bench_idx = next((i for i, bid in enumerate(post_bench) if bid == 401), None)
                                if spidops_bench_idx is not None and spidops_bench_idx < len(post_bench_energies):
                                    pre_sp_e = pre_bench_energies[spidops_bench_idx] if spidops_bench_idx < len(pre_bench_energies) else 0
                                    if post_bench_energies[spidops_bench_idx] > pre_sp_e:
                                        r_strategic += 0.25  # Accelerated to Spidops — Rocket Rush setup
                                    elif total_energy_gained > 0:
                                        r_strategic += 0.15  # Accelerated energy to some Pokemon
                                    else:
                                        r_strategic -= 0.10  # No energy accelerated — wasted Supporter slot
                                elif total_energy_gained > 0:
                                    r_strategic += 0.15
                                else:
                                    r_strategic -= 0.10

                            # Team Rocket's Ariana: reward benching TR Pokemon onto board
                            elif played_id == 1216:  # Team Rocket's Ariana
                                pre_bench_size  = len(pre.get("bench_ids", []))
                                post_bench_size = len(post.get("bench_ids", []))
                                tr_mons_added   = post_bench_size - pre_bench_size
                                hand_diff       = post["hand_size"] - pre["hand_size"]
                                if tr_mons_added >= 2:
                                    r_strategic += 0.30  # Flooded the board — massive Rocket Rush setup
                                elif tr_mons_added == 1:
                                    r_strategic += 0.20  # Benched 1 TR mon + drew cards
                                elif hand_diff >= 2:
                                    r_strategic += 0.10  # Drew cards but couldn't bench
                                else:
                                    r_strategic -= 0.05  # Wasted Supporter slot

                            # Team Rocket's Proton: reward drawing cards and filling hand
                            elif played_id == 1220:  # Team Rocket's Proton
                                hand_before = pre["hand_size"]
                                hand_after  = post["hand_size"]
                                hand_diff   = hand_after - hand_before
                                if hand_before <= 3 and hand_diff >= 2:
                                    r_strategic += 0.20  # Refilled a thin hand — high value
                                elif hand_diff >= 2:
                                    r_strategic += 0.10  # Normal draw — useful
                                elif hand_before >= 6:
                                    r_strategic -= 0.10  # Wasted draw with full hand
                                else:
                                    r_strategic += 0.05  # Neutral

                            # Lillie's Determination: reward late-game or thin-hand draw
                            elif played_id == 1227:  # Lillie's Determination
                                hand_before = pre["hand_size"]
                                hand_after  = post["hand_size"]
                                hand_diff   = hand_after - hand_before
                                if hand_before <= 3 and hand_diff >= 2:
                                    r_strategic += 0.20  # Thin hand refill — max value
                                elif hand_diff >= 2:
                                    r_strategic += 0.10  # Effective draw
                                elif hand_before >= 6:
                                    r_strategic -= 0.08  # Wasted with already full hand
                                else:
                                    r_strategic += 0.05

                            # Team Rocket's Petrel: reward recovering a high-value discarded card
                            elif played_id == 1219:  # Team Rocket's Petrel
                                pre_discard = pre.get("discard_ids", [])
                                post_hand   = post.get("hand_ids", [])
                                high_value_ids = [1216, 1218, 1220, 1217, 1227, 1129, 1116, 1097, 1159, 1175]
                                recovered_high_value = any(cid in post_hand and cid in pre_discard for cid in high_value_ids)
                                if recovered_high_value:
                                    r_strategic += 0.25  # Recovered a key Supporter or unique Item
                                elif post["hand_size"] > pre["hand_size"]:
                                    r_strategic += 0.10  # Recovered something useful
                                else:
                                    r_strategic -= 0.05  # Nothing useful to recover

                            # Poke Pad: reward reloading Supporter pile when discard has Supporters
                            elif played_id == 1152:  # Poke Pad
                                pre_discard = pre.get("discard_ids", [])
                                supporter_ids_all = [1216, 1217, 1218, 1219, 1220, 1227]
                                num_supporters_in_discard = sum(1 for cid in pre_discard if cid in supporter_ids_all)
                                if num_supporters_in_discard >= 3:
                                    r_strategic += 0.30  # Reloaded 3+ Supporters — massive tempo recovery
                                elif num_supporters_in_discard >= 1:
                                    r_strategic += 0.15  # Reloaded some Supporters
                                else:
                                    r_strategic -= 0.10  # No Supporters in discard — wasted single-copy card

                            # Sacred Ash: reward only when 4+ Pokemon are in discard pile
                            elif played_id == 1129:  # Sacred Ash
                                pre_discard = pre.get("discard_ids", [])
                                pokemon_ids_deck = [400, 401, 414, 431, 432, 434]
                                num_pokemon_discard = sum(1 for cid in pre_discard if cid in pokemon_ids_deck)
                                if num_pokemon_discard >= 4:
                                    r_strategic += 0.40  # Board rebuild from mass KO — max value
                                elif num_pokemon_discard >= 2:
                                    r_strategic += 0.20  # Meaningful recovery
                                else:
                                    r_strategic -= 0.15  # Night Stretcher would have been better — wasted single-copy

                            # Giovanni: reward playing with a charged TR active ready to attack
                            elif played_id == 1218:  # Team Rocket's Giovanni
                                active_c = get_card_data(pre.get("active_id"))
                                if active_c is not None:
                                    att_costs = [len(attack_table.get(aid).energies) for aid in active_c.attacks if attack_table.get(aid)]
                                    min_cost  = min(att_costs) if att_costs else 99
                                    is_charged = pre.get("active_energies", 0) >= min_cost
                                    is_tr = getattr(active_c, "name", "").startswith("Team Rocket")
                                    if is_charged and is_tr:
                                        r_strategic += 0.20  # Giovanni played with ready-to-attack TR Pokemon
                                    elif is_charged:
                                        r_strategic += 0.10  # Charged but non-TR active
                                    else:
                                        r_strategic -= 0.10  # Giovanni with un-charged active = wasted Supporter
                                else:
                                    r_strategic -= 0.05

                            # Fallback: general Supporter dead usage penalty
                            elif played_card.cardType == CardType.SUPPORTER and played_id not in [1216, 1217, 1218, 1219, 1220, 1227]:
                                hand_diff = post["hand_size"] - pre["hand_size"]
                                if hand_diff < 0:
                                    r_strategic -= 0.05
                            else:
                                r_strategic += 0.02
                            
                    elif action_type == 8:  # ATTACH
                        attached_id = pre.get("attached_card_id", -1)
                        target_id = pre.get("attached_target_id", -1)
                        
                        attached_card = get_card_data(attached_id)
                        target_card = get_card_data(target_id)
                        
                        if attached_card is not None and target_card is not None:
                            # FIX #1: Unified Mimikyu charging reward — was two sequential `if` blocks
                            # that both fired simultaneously (double reward +0.45 per step).
                            # FIX #2: Exclude Mimikyu from blocker penalty when charging vs ex opponent.
                            opp_active_card = get_card_data(pre.get("opp_active_id"))
                            opp_is_ex = opp_active_card is not None and getattr(opp_active_card, "ex", False)
                            opp_has_safeguard = opp_is_ex and any(
                                s.name == "Safeguard" or "ex" in s.text.lower()
                                for s in getattr(opp_active_card, "skills", [])
                            )
                            charging_mimikyu = (target_id == 434 and opp_is_ex)
                            if target_id == 434 and target_id in pre.get("bench_ids", []):
                                if opp_has_safeguard:
                                    r_strategic += 0.15  # Safeguard scenario — smaller bonus (Mimikyu immune but Safeguard covers it)
                                elif opp_is_ex:
                                    r_strategic += 0.30  # Standard ex-opponent Mimikyu charging reward

                            # Defensive/barrier blockers should never get energy attachments
                            # Exception: Mimikyu acts as an attacker vs ex opponents — don't penalise it
                            target_is_blocker = is_defensive_blocker(target_card) and not charging_mimikyu
                            if target_is_blocker:
                                if attached_card.cardType == CardType.SPECIAL_ENERGY:
                                    r_strategic -= 0.15  # Massive penalty for wasting Special Energy on blockers!
                                else:
                                    r_strategic -= 0.10  # Very high penalty for attaching basic energy to blockers!
                                    
                            # Special energy attachment logic
                            elif attached_card.cardType == CardType.SPECIAL_ENERGY:
                                current_energy = 0
                                if pre.get("active_id") == target_card.cardId:
                                    current_energy = pre.get("active_energy", 0)
                                else:
                                    benched_ids = pre.get("bench_ids", [])
                                    benched_energies = pre.get("bench_energies", [])
                                    target_bench_idx = pre.get("attached_target_bench_idx", -1)
                                    if target_bench_idx >= 0 and target_bench_idx < len(benched_energies):
                                        current_energy = benched_energies[target_bench_idx]
                                    else:
                                        try:
                                            b_idx = benched_ids.index(target_card.cardId)
                                            if b_idx < len(benched_energies):
                                                current_energy = benched_energies[b_idx]
                                        except ValueError:
                                            pass

                                max_attack_cost = 0
                                for aid in target_card.attacks:
                                    att = attack_table.get(aid)
                                    if att is not None:
                                        max_attack_cost = max(max_attack_cost, len(att.energies))

                                is_fully_charged = current_energy >= max_attack_cost

                                if target_card.cardId in [431, 401, 434, 414]:
                                    if is_fully_charged:
                                        r_strategic -= 0.03  # Calibrated penalty for overcharging
                                    else:
                                        r_strategic += 0.25  # Calibrated reward for Special/Double Energy attachment
                                        if pre.get("active_id") != target_card.cardId:
                                            r_strategic += 0.08  # Bench charging bonus
                                else:
                                    r_strategic -= 0.05  # Calibrated penalty
                                    
                            # Basic energy attachment logic
                            elif attached_card.cardType == CardType.BASIC_ENERGY:
                                current_energy = 0
                                if pre.get("active_id") == target_card.cardId:
                                    current_energy = pre.get("active_energy", 0)
                                else:
                                    benched_ids = pre.get("bench_ids", [])
                                    benched_energies = pre.get("bench_energies", [])
                                    target_bench_idx = pre.get("attached_target_bench_idx", -1)
                                    if target_bench_idx >= 0 and target_bench_idx < len(benched_energies):
                                        current_energy = benched_energies[target_bench_idx]
                                    else:
                                        try:
                                            b_idx = benched_ids.index(target_card.cardId)
                                            if b_idx < len(benched_energies):
                                                current_energy = benched_energies[b_idx]
                                        except ValueError:
                                            pass

                                max_attack_cost = 0
                                for aid in target_card.attacks:
                                    att = attack_table.get(aid)
                                    if att is not None:
                                        max_attack_cost = max(max_attack_cost, len(att.energies))

                                is_fully_charged = current_energy >= max_attack_cost

                                # Mewtwo ex (431) uses Psychic (5) or Special (15) energy.
                                if target_card.cardId == 431:
                                    if attached_card.cardId == 5: # Psychic
                                        if is_fully_charged:
                                            r_strategic -= 0.03
                                        else:
                                            r_strategic += 0.25
                                            if pre.get("active_id") != target_card.cardId:
                                                r_strategic += 0.08
                                    elif attached_card.cardId == 1: # Grass
                                        r_strategic -= 0.05
                                # Spidops/Tarountula (401/400) requires Grass energy (1)
                                elif target_card.cardId in [401, 400]:
                                    if attached_card.cardId == 1: # Grass
                                        if is_fully_charged:
                                            r_strategic -= 0.03
                                        else:
                                            r_strategic += 0.25
                                            if pre.get("active_id") != target_card.cardId:
                                                r_strategic += 0.08
                                    elif attached_card.cardId == 5: # Psychic
                                        r_strategic -= 0.03
                                # Mimikyu (434) requires Psychic energy (5)
                                elif target_card.cardId == 434:
                                    if attached_card.cardId == 5: # Psychic
                                        if is_fully_charged:
                                            r_strategic -= 0.03
                                        else:
                                            r_strategic += 0.25
                                            if pre.get("active_id") != target_card.cardId:
                                                r_strategic += 0.08
                                    elif attached_card.cardId == 1: # Grass
                                        r_strategic -= 0.03
                                # Articuno (414) powered by any basic energy
                                elif target_card.cardId == 414:
                                    if is_fully_charged:
                                        r_strategic -= 0.03
                                    else:
                                        r_strategic += 0.25
                                        if pre.get("active_id") != target_card.cardId:
                                            r_strategic += 0.08
                                else:
                                    if is_fully_charged:
                                        r_strategic -= 0.02
                                    else:
                                        target_is_attacker = target_card.ex or target_card.stage1 or target_card.stage2
                                        if target_is_attacker:
                                            r_strategic += 0.15
                                            if pre.get("active_id") != target_card.cardId:
                                                r_strategic += 0.05
                                        else:
                                            r_strategic += 0.05
                                    
                            # Tool attachments
                            elif attached_card.cardType == CardType.TOOL:
                                opp_active_card = get_card_data(pre.get("opp_active_id"))
                                is_opp_ex = opp_active_card is not None and opp_active_card.ex
                                
                                if "Brave Bangle" in attached_card.name or "Maximum Belt" in attached_card.name:
                                    if is_opp_ex and pre.get("active_id") == target_card.cardId:
                                        r_strategic += 0.20
                                    else:
                                        r_strategic -= 0.05
                                        
                    elif action_type == 9:  # EVOLVE
                        evolved_id = pre.get("evolved_card_id", -1)
                        evolved_card = get_card_data(evolved_id)
                        if evolved_card is not None:
                            if evolved_card.stage1 or evolved_card.stage2:
                                r_strategic += 0.15  # Evolution setup reward
                            else:
                                r_strategic += 0.08
                                
                    elif action_type == 10:  # ABILITY
                        r_strategic += 0.05
                        
                    elif action_type == 12:  # RETREAT
                        r_strategic -= 0.40  # Increased penalty to prevent panic retreat-loops
                        opp_active_card = get_card_data(pre.get("opp_active_id"))
                        active_card = get_card_data(pre.get("active_id"))
                        post_active_card = get_card_data(post.get("active_id"))
                        
                        # Strategic retreat from ex-immunity (e.g. Safeguard)
                        opp_has_immunity = opp_active_card is not None and any(s.name == "Safeguard" or "ex" in s.text.lower() for s in getattr(opp_active_card, "skills", []))
                        
                        if opp_has_immunity:
                            if active_card is not None and active_card.ex and post_active_card is not None and not post_active_card.ex:
                                r_strategic += 0.30  # Excellent retreat to non-ex attacker against immune opponent!
                            elif post_active_card is not None and is_defensive_blocker(post_active_card):
                                r_strategic += 0.20  # Defensive retreat to blocker to stall
                        elif active_card is not None and active_card.ex and pre.get("active_energies", 0) >= 3:
                            r_strategic -= 0.05
                            # Mewtwo ex mirror/KO prevention: do not retreat if we can KO opponent active!
                            if active_card.cardId == 431:
                                opp_hp = pre.get("opp_active_hp", 999)
                                if opp_hp <= 280:
                                    r_strategic -= 0.80  # Strong penalty for fleeing when KO is guaranteed
                                    
                        # Strategic retreat to Mimikyu Wall (434) against opponent ex Pokemon
                        opp_is_ex = opp_active_card is not None and getattr(opp_active_card, "ex", False)
                        if opp_is_ex and post_active_card is not None and post_active_card.cardId == 434:
                            r_strategic += 0.60  # Reward retreating to Mimikyu wall to stall/neutralise opponent ex!
                            
                        # KO Prevention: Strategic retreat from damaged Mewtwo ex (431) to a defensive blocker (Articuno/Wobbuffet/Mimikyu)
                        if active_card is not None and active_card.cardId == 431:
                            if pre.get("active_hp", 280) <= 140:  # Mewtwo ex is damaged / half HP
                                if post_active_card is not None and is_defensive_blocker(post_active_card):
                                    r_strategic += 0.50  # Reward retreating to Articuno/Wobbuffet/Mimikyu to save Mewtwo ex!
                            
                    elif action_type == 14:  # END (Turn Stall Decisions)
                        has_attack = pre.get("has_attack_option", False)
                        has_attach = pre.get("has_attach_option", False)
                        energy_already_attached = pre.get("energy_attached_flag", False)
                        
                        opp_active_card = get_card_data(pre.get("opp_active_id"))
                        active_card = get_card_data(pre.get("active_id"))
                        opp_immune = (opp_active_card is not None and active_card is not None and active_card.ex and 
                                      any(s.name == "Safeguard" or "ex" in s.text.lower() for s in getattr(opp_active_card, "skills", [])))
                        
                        if has_attack and not opp_immune:
                            r_strategic -= 0.40  # FIX #8: Reduced from -0.80 — was dominating PER sampling
                        elif has_attach and not energy_already_attached:
                            r_strategic -= 0.40  # FIX #8: Reduced from -0.80 — stall tax handles repeat passes
                        else:
                            r_strategic += 0.02  # Near-zero: just offsets stall tax, net ~0 per forced pass; prevents stall farming
                            
                        if post.get("bench_size", 0) == 0:
                            r_strategic -= 0.60  # Boosted: Heavy penalty for ending turn with 0 bench backup!
                            
                        # Penalty for ending turn with basic Pokemon in hand when bench is not full
                        # FIX #7: Removed dead card ID 463 (not in current deck)
                        has_basic_in_hand = any(cid in [400, 414, 431, 434, 432] for cid in post.get("hand_ids", []))
                        bench_size_post = post.get("bench_size", 0)
                        if has_basic_in_hand and bench_size_post < 5:
                            r_strategic -= 0.40  # Penalty for leaving basics in hand unplayed
                            
                    elif action_type == 13:  # ATTACK
                        opp_active_card = get_card_data(pre.get("opp_active_id"))
                        active_card = get_card_data(pre.get("active_id"))
                        opp_immune = (opp_active_card is not None and active_card is not None and active_card.ex and 
                                      any(s.name == "Safeguard" or "ex" in s.text.lower() for s in getattr(opp_active_card, "skills", [])))
                        
                        if opp_immune:
                            r_strategic -= 0.20  # Wastes turn to attack immune opponent
                        else:
                            att_id = pre.get("attack_id", -1)
                            if att_id == 560:  # Rocket Rush
                                # Count Team Rocket Pokemon in play (active + bench) in pre state
                                tr_count_pre = 0
                                active_id_pre = pre.get("active_id", -1)
                                bench_ids_pre = pre.get("bench_ids", [])
                                if is_tr_pokemon(active_id_pre):
                                    tr_count_pre += 1
                                for bid in bench_ids_pre:
                                    if is_tr_pokemon(bid):
                                        tr_count_pre += 1
                                
                                # Scale reward dynamically with board size:
                                # 2 TR mons -> 0.05, 3 -> 0.13, 4 -> 0.21, 5 -> 0.29, 6 -> 0.37
                                r_strategic += 0.05 + 0.08 * (tr_count_pre - 2) if tr_count_pre >= 2 else 0.02
                            elif att_id == 609:  # Rocket Mirror
                                # Reward based on amount of damage counters healed/redirected
                                pre_dmg = sum(pre.get("bench_damage", []))
                                post_dmg = sum(post.get("bench_damage", []))
                                healed = pre_dmg - post_dmg
                                if healed > 0:
                                    r_strategic += 0.02 * healed
                            elif active_card is not None and active_card.cardId == 431:
                                r_strategic += 0.35  # Calibrated: reward attacking with Mewtwo ex
                            elif active_card is not None and active_card.cardId == 414:  # Team Rocket's Articuno
                                r_strategic += 0.30  # Calibrated: reward Articuno Dark Frost rush
                                if pre.get("turn", 0) <= 3:
                                    r_strategic += 0.15  # Early turn aggro rush bonus
                            else:
                                r_strategic += 0.20  # Calibrated: reward standard attacks
                            if not has_attacked_flag:
                                r_strategic += 0.10
                                has_attacked_flag = True

                            # Early Turn 1-3 KO Aggro Rush Bonus
                            if pre.get("turn", 0) <= 3 and prizes_taken > 0:
                                r_strategic += 0.35  # Calibrated: Early Turn 1-3 Knockout bonus
                            
                    # Reward attaching Team Rocket's Energy (15) to Articuno (414) for 120 dmg Dark Frost
                    if action_type == 8:  # ATTACH
                        attached_cid = pre.get("attached_card_id", -1)
                        target_cid = pre.get("attached_target_id", -1)
                        if (attached_cid == 15 or pre.get("card_id") == 15) and (target_cid == 414 or pre.get("target_id") == 414):
                            r_strategic += 0.20  # Powering up Articuno Dark Frost

                    # Reward playing Giovanni (1218) during early turns (turn <= 3) for +30 damage 1-shot KO
                    if action_type == 6:  # PLAY SUPPORTER/CARD
                        played_cid = pre.get("played_card_id", -1)
                        if (played_cid == 1218 or pre.get("card_id") == 1218) and pre.get("turn", 0) <= 3:
                            r_strategic += 0.15  # Early Giovanni Aggro Damage Boost
                            
                    # 3. Bench Quality & Overextension Management
                    r_bench = 0.0
                    if post["bench_size"] == 0:
                        r_bench -= 0.05  # Strongly penalise having no backup Pokemon!
                        if post.get("turn", 0) <= 2:
                            r_bench -= 0.05  # Extra -0.05 penalty during early turns (turn <= 2)

                    elif post["bench_size"] == 5:
                        r_bench -= 0.02  # Overextension penalty
                        
                    # 4. Proper Attacker Timing & Charging
                    active_card = get_card_data(post.get("active_id"))
                    if active_card is not None:
                        if active_card.cardId == 431:  # Mewtwo ex
                            if post.get("active_energies", 0) < 2:
                                r_strategic -= 0.02  # Penalize having an uncharged Mewtwo ex active
                        elif active_card.ex or active_card.stage1 or active_card.stage2:
                            if post.get("active_energies", 0) < 2:
                                r_strategic -= 0.02  # Penalty for having a weak/undercharged attacker active

                            

                        
                    # 5. Opponent-Specific Counter Rewards
                    if i == 0:
                        opp_active_now = pre.get("opp_active_id", -1)
                        my_active_now = pre.get("active_id", -1)
                        
                        # --- ABOMASNOW COUNTERS ---
                        if opponent_name == "Rulebasedmodel_Abomasnow":
                            # Tactic 2: Snipe Snover pre-evolution for tempo blowout
                            if prizes_taken > 0 and opp_active_now == 722:
                                r_strategic += 0.35
                            # Reward attacking Snover (722) or Kyogre (721) -- easy prizes before full setup
                            if action_type == 13 and opp_active_now in [721, 722]:
                                r_strategic += 0.15
                            # Tactic 3: Exploit retreat-4 lock & Mimikyu Wall vs Abomasnow ex (723)
                            if opp_active_now == 723:
                                if my_active_now == 434:  # Mimikyu is active wall
                                    r_strategic += 0.15
                                    # Reward attacking with Mimikyu to chip or build pressure
                                    if action_type == 13: 
                                        r_strategic += 0.10
                                elif my_active_now == 401:  # Spidops is non-ex attacker (deals 180 dmg with Rocket Rush)
                                    # Reward Spidops active when ready to attack
                                    if pre.get("active_energies", 0) >= 2:
                                        r_strategic += 0.15
                                        if action_type == 13:  # ATTACK with Spidops
                                            r_strategic += 0.20
                                elif my_active_now == 431:  # Mewtwo ex active is a risk
                                    r_strategic -= 0.10
                                elif my_active_now == 400:
                                    r_strategic -= 0.10
                                
                                # Reward charging benched Tarountula/Spidops (400/401) to build Spidops attacker
                                attached_t_id = pre.get("attached_target_id", -1)
                                if action_type == 8 and (attached_t_id in [400, 401] or pre.get("target_id") in [400, 401]):
                                    r_strategic += 0.10
                                # Reward attaching to Mimikyu to tank/retreat
                                if action_type == 8 and (attached_t_id == 434 or pre.get("target_id") == 434):
                                    r_strategic += 0.10
                            # Reward patience (end turn) when Abomasnow is close to self-deckout via Hammer-lanche
                            if opp_deck_size <= 10 and action_type == 14:
                                    r_strategic += 0.15
                                    
                        # --- DRAGOPULT COUNTERS ---
                        elif opponent_name == "Rulebasedmodel_Dragapult":
                            # Snipe Dreepy (119) or Drakloak (120) pre-evolutions for tempo blowout
                            if prizes_taken > 0 and opp_active_now in [119, 120]:
                                r_strategic += 0.35
                            # Mimikyu Wall vs Dragapult ex (121)
                            # Dragapult ex is a Tera Pokemon. Mimikyu can copy Phantom Dive (200 dmg) with Gemstone Mimicry!
                            if opp_active_now == 121:
                                if my_active_now == 434:  # Mimikyu is immune active wall
                                    if pre.get("active_energies", 0) >= 2:
                                        r_strategic += 0.15  # Active and ready to copy Phantom Dive
                                        if action_type == 13:  # ATTACK (Gemstone Mimicry)
                                            r_strategic += 0.20
                                    else:
                                        r_strategic += 0.10  # Active wall but undercharged
                                elif my_active_now == 431:  # Mewtwo ex is a liability
                                    r_strategic -= 0.10
                                # Reward attaching energy to Mimikyu (434) to charge Gemstone Mimicry (needs [P][C])
                                attached_t_id = pre.get("attached_target_id", -1)
                                if action_type == 8 and (attached_t_id == 434 or pre.get("target_id") == 434):
                                    r_strategic += 0.15
                                # Defensive Hero's Cape attachment vs Dragapult
                                attached_c_id = pre.get("attached_card_id", -1)
                                if action_type == 8 and (attached_c_id == 1159 or pre.get("card_id") == 1159):
                                    r_strategic += 0.15  # High reward for creating a 380 HP tank vs Dragapult!
                                    
                                # Bench conservation: penalise overextension vs spread damage (> 2 benched mons)
                                b_size = post.get("bench_size", 0)
                                if b_size > 2:
                                    r_strategic -= 0.20 * (b_size - 2)

                        # --- IONO COUNTERS ---
                        elif opponent_name == "Rulebasedmodel_Iono":
                            # Reward playing TR Transceiver (1134), TR Ariana (1216), or TR Factory (1257) for hand recovery right after Iono disruption
                            played_c_id = pre.get("played_card_id", -1)
                            if action_type == 6 and (played_c_id in [1134, 1216, 1257] or pre.get("card_id") in [1134, 1216, 1257]):
                                if pre.get("hand_size", 0) <= 3:
                                    r_strategic += 0.25  # High reward for hand recovery after Iono disruption!
                                else:
                                    r_strategic += 0.15
                            # Snipe opponent Electric attackers (265-271) before full energy acceleration
                            if prizes_taken > 0 and opp_active_now in [265, 268, 269, 270, 271]:
                                r_strategic += 0.20

                    r_deck = 0.0
                    if post["deck_size"] == 0:
                        r_deck -= 0.50  # Penalize imminent deckout only
                        
                    step_reward = r_stall + r_prize_t - r_prize_l + r_ko - r_own_ko + r_en + r_bench + r_deck + r_strategic
                    step_reward = max(-1.0, min(1.0, step_reward))  # Clip step_reward to [-1.0, 1.0] for PER stability
                    
                    if i == 0:
                        rc_worker["prize_taken"] += r_prize_t
                        rc_worker["prize_lost"] += r_prize_l
                        rc_worker["kos"] += prizes_taken
                        rc_worker["own_kos"] += prizes_lost
                        rc_worker["energy"] += r_en
                        rc_worker["bench"] += r_bench
                        rc_worker["deckout"] += r_deck
                        rc_worker["stall"] += r_stall
                        rc_worker["strategic"] += r_strategic
                        
                    rewards.append(step_reward)
                    
                if i == 0:
                    rc_worker["terminal"] += terminal_reward
                    
                GAMMA = 0.99
                LAMBDA = 0.95
                # Scale MCTS value estimates from normalized [-1,1] to reward scale [-5,5]
                # to match step reward magnitudes; model outputs trained on targets / 5.0
                pred_values = [player_samples[s][0].pred_val * 5.0 for s in range(n_steps)]
                returns = [0.0] * n_steps
                gae = 0.0
                for step_idx in reversed(range(n_steps)):
                    step_rew = rewards[step_idx]
                    if step_idx == n_steps - 1:
                        delta = (step_rew + terminal_reward) - pred_values[step_idx]
                    else:
                        delta = step_rew + GAMMA * pred_values[step_idx + 1] - pred_values[step_idx]
                    gae = delta + GAMMA * LAMBDA * gae
                    returns[step_idx] = gae + pred_values[step_idx]
                    
                for step_idx in range(n_steps):
                    sample_obj, _ = player_samples[step_idx]
                    # Widen clip to ±5 so terminal reward (±5.0) signal survives the GAE return
                    sample_obj.value = max(-5.0, min(5.0, returns[step_idx]))
                    td_error = returns[step_idx] - sample_obj.pred_val
                    processed_samples.append((sample_obj, td_error))

            # Calculate policy entropy
            entropy_accum = 0.0
            entropy_count = 0
            for sample_obj, _ in samples[0]:
                if sample_obj is not None and hasattr(sample_obj, 'policy') and len(sample_obj.policy) > 0:
                    policy_probs = [max(1e-8, p) for p in sample_obj.policy if p > 0]
                    p_sum = sum(policy_probs)
                    if p_sum > 0:
                        entropy = -sum((p/p_sum) * math.log(p/p_sum) for p in policy_probs)
                        entropy_accum += entropy
                        entropy_count += 1
                        
            result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, processed_samples, result, final_turn, rc_worker, entropy_accum, entropy_count, action_counts, played_cards)))
            
        elif cmd == "EVAL":
            sample_deck, opponent_deck, opponent_name = args
            
            try:
                obs, start_data = battle_start(sample_deck, opponent_deck)
                if start_data.errorPlayer >= 0:
                    result_queue.put(("EVAL_COMPLETE", (worker_id, opponent_name, -1.0)))
                    continue
            except Exception as e:
                result_queue.put(("EVAL_COMPLETE", (worker_id, opponent_name, -1.0)))
                continue
                
            # Always evaluate our agent playing sample_deck (Player 0) against
            # the opponent playing opponent_deck (Player 1). The coin toss automatically
            # randomizes who goes first/second.
            your_index = 0
            while True:
                if obs["current"]["result"] >= 0:
                    break
                    
                if obs["current"]["yourIndex"] == your_index:
                    # Replicate Kaggle search count budget constraints during evaluation
                    turn = obs["current"].get("turn", 0)
                    active_is_walled = False
                    try:
                        my_active = obs["current"]["players"][your_index]["active"]
                        opp_active = obs["current"]["players"][1 - your_index]["active"]
                        if my_active and opp_active:
                            if my_active[0]["id"] == 431 and opp_active[0]["id"] == 434:
                                active_is_walled = True
                    except Exception:
                        pass
                        
                    is_main_context = False
                    try:
                        if obs.get("select") is not None and obs["select"].get("context") == SelectContext.MAIN:
                            is_main_context = True
                    except Exception:
                        pass
                        
                    if active_is_walled or is_main_context:
                        eval_search_count = 35
                    elif turn <= 3:
                        eval_search_count = 20
                    elif turn <= 8:
                        eval_search_count = 15
                    else:
                        eval_search_count = 10
                        
                    selected, _ = mcts_agent(obs, sample_deck, client, search_count=eval_search_count)
                else:
                    if opponent_name in ["Rulebasedmodel", "Rulebasedmodel_Iono", "Rulebasedmodel_Dragapult", "Rulebasedmodel_Mewtwo", "Rulebasedmodel_Mewtwo_Easy", "Rulebasedmodel_Abomasnow","Rulebasedmodel_Mewtwo_Wobbuffet"]:
                        try:
                            selected = rule_based_opponent_agent(opponent_name, obs)
                        except Exception as e:
                            selected = random_agent(obs)
                    else:
                        selected = random_agent(obs)
                obs = battle_select(selected)
                
            battle_finish()
            result = obs["current"]["result"]
            
            if result == 2:
                outcome = 0.5
            elif result == your_index:
                outcome = 1.0
            else:
                outcome = 0.0
                
            result_queue.put(("EVAL_COMPLETE", (worker_id, opponent_name, outcome)))


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


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the MCTS Pokemon TCG AI Agent.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Target evaluation games (default: 50)")
    parser.add_argument("--self-play-episodes", type=int, default=100, help="Number of self-play games for data collection (default: 100)")
    parser.add_argument("--self-play-ratio", type=float, default=0.5, help="Ratio of games played against Current (Self) vs rule-based bots (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for model training (default: 128)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping based on evaluation win rate (default: 10)")
    parser.add_argument("--num-workers", type=int, default=max(1, mp.cpu_count() - 1), help="Number of parallel worker processes")
    parser.add_argument("--disable-league", action="store_true", help="Disable league play and checkpoint saving in league directory")
    args = parser.parse_args()

    # Reproducibility
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    opponent_decks = load_all_decks()
    opponent_decks = {k: v for k, v in opponent_decks.items() if k in ["Current (Self)","Rulebasedmodel_Mewtwo_Easy","Rulebasedmodel_Mewtwo","Rulebasedmodel_Dragapult","Rulebasedmodel_Mewtwo_Wobbuffet","Rulebasedmodel_Abomasnow","Rulebasedmodel_Lucario","Rulebasedmodel_Crustle","Rulebasedmodel_Starmie","Rulebasedmodel_Dipplin","Rulebasedmodel_Iono"]}
    if not opponent_decks:
        raise ValueError("No valid deck.csv found in root or subdirectories.")
        
    sample_deck = opponent_decks.get("Current (Self)")
    if not sample_deck:
        raise ValueError("Root deck.csv not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Hyperparameters: Epochs={args.epochs}, Eval-Episodes={args.eval_episodes}, Self-Play-Episodes={args.self_play_episodes}, Self-Play-Ratio={args.self_play_ratio}, Batch-Size={args.batch_size}, LR={args.lr}")
    print(f"Workers count: {args.num_workers}")

    model = MyModel(
        MODEL_D_MODEL,
        MODEL_NUM_HEADS,
        MODEL_D_FEEDFORWARD,
        MODEL_NUM_LAYERS_ENCODER,
        MODEL_NUM_LAYERS_DECODER
    )
    
    checkpoint_path = "model.pth"
    start_epoch = 0  # Tracks loaded checkpoint epoch for LR schedule resume
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model weights from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
                start_epoch = checkpoint.get('epoch', 0)
                print(f"  -> Successfully loaded checkpoint from epoch {start_epoch}")
            else:
                model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print(f"  -> Warning: Could not load checkpoint due to size mismatch ({e}). Starting from scratch.")
        
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    def _lr_lambda(epoch):
        warmup_epochs = 5  # Increased from 3
        min_lr_ratio = 0.1  # Do not decay below 10% of peak learning rate
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale decay between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    # When resuming from a checkpoint (start_epoch > 0), PyTorch requires 'initial_lr'
    # to already exist in param_groups before LambdaLR can apply its multiplier.
    # Seed it from the current optimizer lr so the schedule resumes correctly.
    if start_epoch > 0:
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
    scheduler_last_epoch = start_epoch if start_epoch > 0 else -1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=scheduler_last_epoch)
    
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

    if not args.disable_league and os.path.exists("model.pth"):
        import shutil
        shutil.copy("model.pth", os.path.join(league_dir, f"model_epoch_0_run_{version}.pth"))
        print(f"Seeded league with initial model checkpoint as model_epoch_0_run_{version}.pth")

    elo_path = os.path.join(league_dir, "elo.csv")
    league_elos = {}
    if os.path.exists(elo_path):
        try:
            with open(elo_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 2:
                        league_elos[row[0]] = float(row[1])
        except Exception as e:
            print(f"Warning: Failed to load Elo file: {e}")
    if "active" not in league_elos:
        league_elos["active"] = 1500.0

    metrics_path = os.path.join(run_dir, "training_metrics.csv")
    with open(metrics_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "win_rate", "avg_loss", "avg_reward",
                         "r_prize_taken", "r_prize_lost", "r_kos", "r_own_kos",
                         "r_energy", "r_bench", "r_deckout", "r_terminal", "r_stall", "r_no_energy", "r_strategic",
                         "avg_game_length", "policy_entropy"])

    deck_matchup_path = os.path.join(run_dir, "deck_matchup.csv")
    with open(deck_matchup_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "deck_name", "wins", "losses", "draws", "win_rate"])

    action_dist_path = os.path.join(run_dir, "action_distribution.csv")
    with open(action_dist_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "attack", "play", "attach", "evolve", "ability", "retreat", "end", "other"])

    self_play_games_path = os.path.join(run_dir, "self_play_games.csv")
    with open(self_play_games_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "opponent_name", "result", "turns", "attacks", "plays", "attaches", "evolves", "abilities", "retreats", "ends", "other"])

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=run_dir)
        print("TensorBoard logging enabled.")
    except ImportError:
        tb_writer = None
        print("TensorBoard not installed. Skipping TensorBoard logging.")

    best_win_rate = -1.0
    patience_counter = 0
    
    all_opponent_names = sorted([name for name in opponent_decks.keys() if name != "Current (Self)"])
    
    # Train against all opponent decks (including Iono) to learn card-specific
    # counters and strategies, and evaluate against all decks to check progress.
    train_opponent_names = all_opponent_names
    test_opponent_names = ["Rulebasedmodel_Mewtwo_Easy", "Rulebasedmodel_Mewtwo", "Rulebasedmodel_Mewtwo_Wobbuffet", "Rulebasedmodel_Dragapult", "Rulebasedmodel_Abomasnow", "Rulebasedmodel_Lucario", "Rulebasedmodel_Crustle", "Rulebasedmodel_Starmie", "Rulebasedmodel_Dipplin", "Rulebasedmodel_Iono"]

    print(f"Opponent Decks Configuration:")
    print(f"  -> Train Opponent Decks: {train_opponent_names}")
    print(f"  -> Test Opponent Decks: {test_opponent_names}")

    rolling_wins = {name: 0.0 for name in train_opponent_names}
    rolling_games = {name: 0.0 for name in train_opponent_names}
    
    replay_buffer = PrioritizedReplayBuffer(capacity=50000)

    model_lock = threading.Lock()

    num_workers = args.num_workers
    print(f"Spawning {num_workers} parallel workers...")
    
    command_queues = [mp.Queue() for _ in range(num_workers)]
    result_queue = mp.Queue()
    
    parent_conns = []
    worker_conns = []
    for _ in range(num_workers):
        p_conn, c_conn = mp.Pipe(duplex=True)
        parent_conns.append(p_conn)
        worker_conns.append(c_conn)
        
    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_loop,
            args=(i, command_queues[i], result_queue, worker_conns[i], "cpu")
        )
        p.daemon = True
        p.start()
        workers.append(p)

    inference_server = GPUInferenceServer(model, parent_conns, model_lock, batch_size=args.batch_size)
    inference_server.start(device)
    print("GPU Inference Server started successfully.")

    counter = 0

    def handle_interrupt(signum, frame):
        print("\nTraining interrupted by user (Ctrl+C). Cleaning up and saving progress...")
        
        print("Stopping worker processes...")
        for q in command_queues:
            q.put(("STOP", None))
        for p in workers:
            try:
                p.join(timeout=2)
            except Exception:
                pass
            
        print("Stopping GPU inference server...")
        try:
            inference_server.stop()
        except Exception:
            pass
            
        with model_lock:
            final_checkpoint = {
                "epoch": counter,
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }
        torch.save(final_checkpoint, "model.pth")
        torch.save(final_checkpoint, os.path.join(run_dir, "model.pth"))
        
        if tb_writer is not None:
            tb_writer.close()
            
        print(f"\nSaved checkpoint (epoch {counter}) to model.pth and inside {run_dir}")
        print("Generating learning curves...")
        try:
            plot_metrics(metrics_path, run_dir, deck_matchup_path, action_dist_path)
        except Exception as e:
            print(f"Error generating plots: {e}")
        
        sys.exit(0)

    import signal
    signal.signal(signal.SIGINT, handle_interrupt)

    for counter in range(1, args.epochs + 1):
        print(f"\n--- Epoch {counter}/{args.epochs} ---")
        
        epoch_rewards = []
        epoch_action_counts = {"attack": 0, "play": 0, "attach": 0, "evolve": 0, "ability": 0, "retreat": 0, "end": 0, "other": 0}
        from collections import defaultdict
        epoch_card_plays = defaultdict(int)
        rc = {"prize_taken": 0.0, "prize_lost": 0.0, "kos": 0.0, "own_kos": 0.0,
              "energy": 0.0, "bench": 0.0, "deckout": 0.0, "terminal": 0.0, "stall": 0.0,
              "no_energy": 0.0, "strategic": 0.0}
        rc_count = 0
        total_game_length = 0
        total_games = 0
        
        entropy_accum = 0.0
        entropy_count = 0
        epoch_deck_stats = {}
        
        win_rate = 0.0
        if args.eval_episodes > 0:
            active_elo = league_elos.get("active", 1500.0)
            decision, win_rate, wins, losses, draws, deck_stats = run_sprt_evaluation(
                command_queues, result_queue, num_workers, sample_deck, opponent_decks, test_opponent_names,
                alpha=0.05, beta=0.1, p0=0.50, p1=0.58, max_eval_games=args.eval_episodes
            )
            
            for name, stats in deck_stats.items():
                d_denom = stats[0] + stats[1]
                d_win_rate = 100.0 * stats[0] / d_denom if d_denom > 0 else 0.0
                epoch_deck_stats[name] = {"wins": stats[0], "losses": stats[1], "draws": stats[2], "win_rate": d_win_rate}
                
            if decision is True or (decision is None and win_rate > best_win_rate):
                best_win_rate = win_rate
                with model_lock:
                    best_checkpoint = {
                        "epoch": counter,
                        "state_dict": model.state_dict(),
                        "win_rate": best_win_rate,
                        "elo": active_elo
                    }
                torch.save(best_checkpoint, "best_model.pth")
                torch.save(best_checkpoint, os.path.join(run_dir, "best_model.pth"))
                print(f"  -> Saved new best model checkpoint (Win Rate: {best_win_rate:.1f}%)")
                patience_counter = 0
            elif decision is False:
                patience_counter += 1
                print(f"  -> Model rejected. No improvement for {patience_counter} consecutive epochs.")
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered.")
                    break
        else:
            print("Skipping evaluation (eval-episodes=0).")

        if args.self_play_episodes > 0:
            drain_queue(result_queue)
            games_sent = 0
            games_received = 0
            active_tasks = {}
            active_elo = league_elos.get("active", 1500.0)
            league_completed = 0
            league_failed = 0
            
            league_checkpoints = []
            if not args.disable_league:
                league_checkpoints = [
                    os.path.join(league_dir, f) for f in os.listdir(league_dir) if f.endswith(".pth")
                ]
            
            def get_next_self_play_args():
                # League scheduling: start with mostly rule-based opponents so the model
                # learns real strategy before facing league models. Early league checkpoints
                # are nearly random -- 60% league in epoch 1 = 60% random opponents, which stalls.
                # Ramp from 0% to 50% league over the first 10 epochs, then hold at 50%.
                can_use_league = bool(league_checkpoints)
                league_prob = min(0.50, 0.05 * counter)  # 0% at epoch 0, 50% at epoch 10+
                use_league = can_use_league and bool(train_opponent_names) and (random.random() < league_prob)

                if use_league:
                    opp_path = sample_league_opponent(active_elo, league_checkpoints, league_elos)
                    opponent_name = os.path.basename(opp_path)
                    opponent_deck = sample_deck
                    opponent_type = "League"
                    opponent_path = opp_path
                else:
                    # Select a rule-based opponent based on inverse win-rates
                    weights = []
                    for name in train_opponent_names:
                        g = rolling_games[name]
                        w = rolling_wins[name]
                        wr = w / g if g > 0 else 0.5
                        weights.append(max(0.1, 1.0 - wr))
                    w_sum = sum(weights)
                    probs = [w / w_sum for w in weights]
                    
                    # Cap Abomasnow selection probability at 35% to avoid defensive collapse / replay buffer flooding
                    if "Rulebasedmodel_Abomasnow" in train_opponent_names:
                        abo_idx = train_opponent_names.index("Rulebasedmodel_Abomasnow")
                        if probs[abo_idx] > 0.35:
                            diff = probs[abo_idx] - 0.35
                            probs[abo_idx] = 0.35
                            other_indices = [idx for idx in range(len(probs)) if idx != abo_idx]
                            other_sum = sum(probs[idx] for idx in other_indices)
                            if other_sum > 0:
                                for idx in other_indices:
                                    probs[idx] += diff * (probs[idx] / other_sum)
                            else:
                                for idx in other_indices:
                                    probs[idx] += diff / len(other_indices)

                    opponent_name = random.choices(train_opponent_names, weights=probs, k=1)[0]
                    opponent_deck = opponent_decks[opponent_name]
                    opponent_type = "Rulebased"
                    opponent_path = None

                return opponent_name, sample_deck, opponent_deck, opponent_type, opponent_path

            for w_idx in range(num_workers):
                if games_sent < args.self_play_episodes:
                    opp_name, s_deck, o_deck, opp_type, opp_path = get_next_self_play_args()
                    command_queues[w_idx].put(("PLAY_SELF", (s_deck, o_deck, opp_type, opp_path, opp_name, counter)))
                    active_tasks[w_idx] = (opp_name, opp_type, opp_path)
                    games_sent += 1

            epoch_self_play_wins = 0.0
            epoch_self_play_games = 0.0
            pbar = ProgressBar(args.self_play_episodes, "Training Data Collecting... ")
            pbar.update(0)
            
            while games_received < args.self_play_episodes:
                msg, data = result_queue.get()
                if msg == "PLAY_SELF_COMPLETE":
                    w_idx, samples, result, final_turn, rc_worker, e_accum, e_count = data[:7]
                    action_counts = data[7] if len(data) > 7 else {}
                    played_cards = data[8] if len(data) > 8 else {}
                    
                    games_received += 1
                    
                    if action_counts:
                        for act_k, act_v in action_counts.items():
                            epoch_action_counts[act_k] += act_v
                    if played_cards:
                        for card_id, count in played_cards.items():
                            epoch_card_plays[card_id] += count
                            
                    opp_name, opp_type, opp_path = active_tasks[w_idx]
                    
                    if opp_type == "League":
                        if result >= 0:
                            league_completed += 1
                        else:
                            league_failed += 1
                    
                    # Append game outcome and action distribution to self_play_games.csv
                    with open(self_play_games_path, mode="a", newline="", encoding="utf-8-sig") as f_sp:
                        sp_writer = csv.writer(f_sp)
                        sp_writer.writerow([
                            counter,
                            opp_name,
                            result,
                            final_turn,
                            action_counts.get("attack", 0),
                            action_counts.get("play", 0),
                            action_counts.get("attach", 0),
                            action_counts.get("evolve", 0),
                            action_counts.get("ability", 0),
                            action_counts.get("retreat", 0),
                            action_counts.get("end", 0),
                            action_counts.get("other", 0)
                        ])
                    
                    if result >= 0:
                        if result == 0:
                            epoch_self_play_wins += 1.0
                            epoch_self_play_games += 1.0
                        elif result == 1:
                            epoch_self_play_games += 1.0
                        elif result == 2:
                            epoch_self_play_wins += 0.5
                            epoch_self_play_games += 1.0

                    if opp_name != "Current (Self)" and result >= 0:
                        if opp_name in rolling_games:
                            rolling_games[opp_name] += 1
                            if result == 0:
                                rolling_wins[opp_name] += 1
                            elif result == 2:
                                rolling_wins[opp_name] += 0.5
                                
                            # Decay only the current opponent's rolling stats — not all opponents
                            # (global decay was incorrectly pushing all win-rates toward 0.5)
                            rolling_wins[opp_name] *= 0.95
                            rolling_games[opp_name] *= 0.95

                    if opp_path is not None and "model" in os.path.basename(opp_path) and result >= 0:
                        opp_name_file = os.path.basename(opp_path)
                        active_elo = league_elos.get("active", 1500.0)
                        opp_elo = league_elos.get(opp_name_file, 1500.0)
                        
                        exp_active = 1.0 / (1.0 + 10.0 ** ((opp_elo - active_elo) / 400.0))
                        exp_opp = 1.0 - exp_active
                        
                        s_active = 1.0 if result == 0 else 0.0 if result == 1 else 0.5
                        s_opp = 1.0 - s_active
                        
                        K = 32
                        active_elo += K * (s_active - exp_active)
                        opp_elo += K * (s_opp - exp_opp)
                        
                        league_elos["active"] = active_elo
                        league_elos[opp_name_file] = opp_elo
                        
                        os.makedirs(os.path.dirname(elo_path), exist_ok=True)
                        with open(elo_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(["checkpoint", "elo"])
                            for name_ch, elo in league_elos.items():
                                writer.writerow([name_ch, elo])

                    for sample_obj, td_error in samples:
                        replay_buffer.add(sample_obj, td_error)
                        epoch_rewards.append(sample_obj.value)

                    for k in rc:
                        if k in rc_worker:
                            rc[k] += rc_worker[k]
                    # FIX #6: Count only player-0 steps — rc[] is accumulated for i==0 only,
                    # using len(samples) (both players combined) was dividing by ~2x the correct count.
                    player0_steps = len(samples) // 2 if len(samples) > 0 else 1
                    rc_count += player0_steps
                    total_game_length += final_turn
                    total_games += 1
                    
                    entropy_accum += e_accum
                    entropy_count += e_count
                    
                    if games_sent < args.self_play_episodes:
                        opp_name, s_deck, o_deck, opp_type, opp_path = get_next_self_play_args()
                        command_queues[w_idx].put(("PLAY_SELF", (s_deck, o_deck, opp_type, opp_path, opp_name, counter)))
                        active_tasks[w_idx] = (opp_name, opp_type, opp_path)
                        games_sent += 1
                    
                    wr = (epoch_self_play_wins / epoch_self_play_games * 100.0) if epoch_self_play_games > 0 else 0.0
                    avg_turns = (total_game_length / total_games) if total_games > 0 else 0.0
                    pbar.update(games_received, suffix=f"WinRate: {wr:.1f}% | AvgTurns: {avg_turns:.1f}")
            
            wr = (epoch_self_play_wins / epoch_self_play_games * 100.0) if epoch_self_play_games > 0 else 0.0
            avg_turns = (total_game_length / total_games) if total_games > 0 else 0.0
            pbar.update(args.self_play_episodes, suffix=f"WinRate: {wr:.1f}% | AvgTurns: {avg_turns:.1f}")
            if league_completed > 0 or league_failed > 0:
                print(f"\n  -> League self-play stats: {league_completed} games completed successfully, {league_failed} games failed.")
        else:
            print("Skipping self-play data collection (self-play-episodes=0).")

        avg_loss = 0.0
        last_loss_enc_val = 0.0      # Captured per-batch; used for TensorBoard after tensor del
        last_loss_dec_ce_val = 0.0
        trained_this_epoch = False
        if len(replay_buffer) >= args.batch_size:
            trained_this_epoch = True
            print("Training Start.")
            model.train()
            batch_count = min(50, len(replay_buffer) // args.batch_size)
            print(f"Total training buffer size: {len(replay_buffer)}, Batch Count: {batch_count}")
            
            epoch_losses = []
            for i in range(batch_count):
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
                    for _ in range(256 - len(sample.policy)):
                        mask.append(0.0)
                        label_dec.append(0.0)
                        input_dec.offset.append(len(input_dec.index))

                mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(args.batch_size, -1)
                label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device).view(args.batch_size, -1)
                label_tensor_enc = label_tensor_enc / 5.0  # Normalise GAE returns [-5,5] to [-1,1] for unbounded value head (audit fix)
                label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device).view(args.batch_size, -1)

                with model_lock:
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
                        
                        is_weight_tensor = torch.tensor(is_weights, dtype=torch.float32, device=device).view(args.batch_size, 1)

                        loss_enc_elem = torch.nn.functional.huber_loss(out_enc, label_tensor_enc, reduction="none", delta=1.0)
                        loss_enc = (loss_enc_elem * is_weight_tensor).mean()

                        # Cast decoder output to float32 explicitly for safe log_softmax computation under autocast
                        out_dec_fp32 = out_dec.float()
                        masked_logits = out_dec_fp32 + (1.0 - mask_tensor) * (-1e4)
                        log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
                        loss_dec_ce = -(label_tensor_dec * log_probs * mask_tensor)
                        loss_dec_ce = (loss_dec_ce.sum(dim=-1, keepdim=True) * is_weight_tensor).mean()

                        loss = loss_enc + loss_dec_ce

                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                epoch_losses.append(loss.item())

                errors = (out_enc - label_tensor_enc).abs().squeeze().tolist()
                if isinstance(errors, float):
                    errors = [errors]
                replay_buffer.update_priorities(indices, errors)

                # Capture scalar values before freeing tensors — TensorBoard logging uses these below
                last_loss_enc_val = loss_enc.item()
                last_loss_dec_ce_val = loss_dec_ce.item()

                # Explicitly free training tensors each step to avoid
                # accumulating GPU/CPU memory across batch iterations.
                del out_enc, out_dec, loss, loss_enc, loss_dec_ce
                del mask_tensor, label_tensor_enc, label_tensor_dec, is_weight_tensor
                
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            print(f"Training Finish. Average Loss: {avg_loss:.4f}")
        else:
            if args.self_play_episodes > 0:
                print(f"Skipping training: collected buffer size ({len(replay_buffer)}) less than batch size ({args.batch_size}).")

        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0

        epoch_model_path = os.path.join(run_dir, f"model{counter}.pth")
        with model_lock:
            checkpoint = {
                "epoch": counter,
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }
        torch.save(checkpoint, epoch_model_path)
        torch.save(checkpoint, "model.pth")
        
        if not args.disable_league:
            league_model_path = os.path.join(league_dir, f"model_epoch_{counter}_run_{version}.pth")
            torch.save(checkpoint, league_model_path)
            print(f"Saved checkpoint: {epoch_model_path}, model.pth, and {league_model_path}")
        else:
            print(f"Saved checkpoint: {epoch_model_path} and model.pth (League saving disabled)")

        # Prune league directory — keep only the most recent 10 checkpoints to
        # prevent the league candidate list from growing unbounded each epoch.
        MAX_LEAGUE_SIZE = 30
        league_files = sorted(
            [f for f in os.listdir(league_dir) if f.endswith(".pth")],
            key=lambda f: os.path.getmtime(os.path.join(league_dir, f))
        )
        for old_file in league_files[:-MAX_LEAGUE_SIZE]:
            try:
                os.remove(os.path.join(league_dir, old_file))
            except OSError:
                pass

        # Release stale league model weights from the in-process cache so
        # Python can GC the tensors and reclaim RAM.
        _league_cache.clear()

        avg_gl = total_game_length / total_games if total_games > 0 else 0.0
        avg_entropy = entropy_accum / max(1, entropy_count)
        rc_div = max(1, rc_count)
        
        with open(metrics_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([counter, win_rate, avg_loss, avg_reward,
                             rc["prize_taken"] / rc_div, rc["prize_lost"] / rc_div,
                             rc["kos"] / rc_div, rc["own_kos"] / rc_div,
                             rc["energy"] / rc_div, rc["bench"] / rc_div,
                             rc["deckout"] / rc_div, rc["terminal"] / rc_div,  # FIX #9: normalize per-step like all other rc fields
                             rc["stall"] / rc_div, rc["no_energy"] / rc_div, rc["strategic"] / rc_div, avg_gl,
                             avg_entropy])
        
        with open(deck_matchup_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for name, stats in epoch_deck_stats.items():
                writer.writerow([counter, name, stats["wins"], stats["losses"], stats["draws"], f"{stats['win_rate']:.1f}"])
        
        with open(action_dist_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([counter,
                             epoch_action_counts["attack"],
                             epoch_action_counts["play"],
                             epoch_action_counts["attach"],
                             epoch_action_counts["evolve"],
                             epoch_action_counts["ability"],
                             epoch_action_counts["retreat"],
                             epoch_action_counts["end"],
                             epoch_action_counts["other"]])
        
        if trained_this_epoch:
            scheduler.step()
        
        active_elo = league_elos.get("active", 1500.0)
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/Policy", last_loss_dec_ce_val, counter)  # last training batch value
            tb_writer.add_scalar("Loss/Value",  last_loss_enc_val,    counter)  # last training batch value
            tb_writer.add_scalar("MCTS/AverageDepth", avg_gl, counter)
            tb_writer.add_scalar("MCTS/Entropy", avg_entropy, counter)
            tb_writer.add_scalar("League/ActiveElo", active_elo, counter)
            tb_writer.add_scalar("Generalization/OOD_WinRate", win_rate, counter)

        print(f"Epoch Metrics Logged -> Win Rate: {win_rate:.1f}%, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, Active Elo: {active_elo:.1f}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Log and print card play statistics for the epoch
        card_stats_path = os.path.join(run_dir, "card_play_stats.csv")
        file_exists = os.path.exists(card_stats_path)
        with open(card_stats_path, mode="a", newline="", encoding="utf-8-sig") as cs_f:
            cs_writer = csv.writer(cs_f)
            if not file_exists:
                cs_writer.writerow(["epoch", "card_id", "card_name", "play_count"])
            for cid, count in sorted(epoch_card_plays.items(), key=lambda x: x[1], reverse=True):
                c_obj = card_table.get(cid)
                c_name = c_obj.name if c_obj is not None else f"Card #{cid}"
                cs_writer.writerow([counter, cid, c_name, count])
        
        # Print top 5 cards played this epoch
        top_cards = sorted(epoch_card_plays.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_cards:
            print("  Top Cards Played:")
            for cid, count in top_cards:
                c_obj = card_table.get(cid)
                c_name = c_obj.name if c_obj is not None else f"Card #{cid}"
                print(f"    - {c_name:<25}: {count} plays")

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    print("Stopping worker processes...")
    for q in command_queues:
        q.put(("STOP", None))
    for p in workers:
        p.join()
        
    print("Stopping GPU inference server...")
    inference_server.stop()
        
    with model_lock:
        final_checkpoint = {
            "epoch": args.epochs,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }
    torch.save(final_checkpoint, "model.pth")
    torch.save(final_checkpoint, os.path.join(run_dir, "model.pth"))
    
    if tb_writer is not None:
        tb_writer.close()
        
    print(f"\nTraining complete. Final weights saved to model.pth and inside {run_dir}")
    print(f"Best model (peak win rate) preserved in best_model.pth")

    print("Generating learning curves...")
    plot_metrics(metrics_path, run_dir, deck_matchup_path, action_dist_path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
