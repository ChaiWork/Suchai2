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
from cg.api import to_observation_class, OptionType, AreaType, SelectContext


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
    if opponent_name == "Rulebasedmodel":
        from Rulebasedmodel.main import agent
        return agent(obs)
    elif opponent_name == "Rulebasedmodel_Iono":
        from Rulebasedmodel.iono_agent import agent
        return agent(obs)
    elif opponent_name == "Rulebasedmodel_Dragapult":
        from Rulebasedmodel.dragapult_agent import agent
        return agent(obs)
    elif opponent_name == "Rulebasedmodel_Mewtwo":
        from Rulebasedmodel.mewtwo_agent import agent
        return agent(obs)
    elif opponent_name == "Rulebasedmodel_Mewtwo_Easy":
        from Rulebasedmodel.mewtwo_agent_easy import agent
        return agent(obs)
    elif opponent_name == "Rulebasedmodel_Abomasnow":
        from Rulebasedmodel.abomasnow_agent import agent
        return agent(obs)
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
                if orig_len < 128:
                    for _ in range(128 - orig_len):
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
                policies = [[0.0] * 128] * len(batch_conns)

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
            
            opp_model = None
            if opponent_path is not None:
                try:
                    opp_model = MyModel(
                        MODEL_D_MODEL,
                        MODEL_NUM_HEADS,
                        MODEL_D_FEEDFORWARD,
                        MODEL_NUM_LAYERS_ENCODER,
                        MODEL_NUM_LAYERS_DECODER
                    ).to(device)
                    checkpoint = torch.load(opponent_path, map_location=device, weights_only=True)
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        opp_model.load_state_dict(checkpoint["state_dict"], strict=False)
                    else:
                        opp_model.load_state_dict(checkpoint, strict=False)
                    opp_model.eval()
                except Exception as e:
                    opp_model = None
            
            try:
                obs, start_data = battle_start(sample_deck, opponent_deck)
                if start_data.errorPlayer >= 0:
                    result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, [], -1, 0, {}, 0.0, 0, {})))
                    continue
            except Exception as e:
                result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, [], -1, 0, {}, 0.0, 0, {})))
                continue

            try:
                action_counts = {"attack": 0, "play": 0, "attach": 0, "evolve": 0, "ability": 0, "retreat": 0, "end": 0, "other": 0}
                samples = [[], []]
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
                        "energy_attached_flag": obs_class.current.energyAttached,
                        "active_id": active_id,
                        "active_energies": active_energies,
                        "opp_active_id": opp_active_id,
                        "stadium_id": stadium_id,
                        "bench_ids": bench_ids,
                        "hand_size": len(state_ps.hand) if state_ps.hand is not None else 0,
                        "discard_size": len(state_ps.discard) if state_ps.discard is not None else 0,
                        "turn": obs_class.current.turn,
                        "has_attack_option": has_attack_option,
                        "has_attach_option": has_attach_option
                    }
    
                    if curr_player == 0:
                        selected, sample = mcts_agent(obs, curr_deck, client, search_count=200)
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
                        samples[0].append((sample, pre_metrics))
                        
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
                        if opponent_name in ["Rulebasedmodel", "Rulebasedmodel_Iono", "Rulebasedmodel_Dragapult", "Rulebasedmodel_Mewtwo", "Rulebasedmodel_Mewtwo_Easy","Rulebasedmodel_Abomasnow"]:
                            try:
                                selected = rule_based_opponent_agent(opponent_name, obs)
                            except Exception as e:
                                selected = random_agent(obs)
                        elif opponent_type == "Current":
                            selected, sample = mcts_agent(obs, curr_deck, client, search_count=200)
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
                            selected, sample = mcts_agent(obs, curr_deck, opp_model, search_count=120)
                        else:
                            selected = random_agent(obs)
                    
                    obs = battle_select(selected)
                    
                battle_finish()
            except Exception as e:
                import traceback
                print(f"Error in worker {worker_id} simulation:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, [], -1, 0, {}, 0.0, 0, {})))
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
                    
                num_attacks = sum(1 for _, pre in player_samples if pre.get("action_type") == 13)
                if i == result:
                    if num_attacks == 0:
                        # Still reward a win even without attacks (e.g. deck-out, prize KO via ability).
                        # 0.0 was silencing the win signal entirely — agent never learned winning is good.
                        # 2.0 keeps a positive gradient while reserving the full +5 bonus for attack-led wins.
                        terminal_reward = 2.0
                    else:
                        terminal_reward = 5.0
                elif result == 2:
                    terminal_reward = 0.0   # Draw: neutral outcome — not penalised like a loss
                elif result == -1:
                    terminal_reward = -5.0  # Error/invalid game — treat as loss
                else:
                    terminal_reward = -5.0

                    
                rewards = []
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
                            "energy_attached_flag": True,
                            "active_id": final_active_id,
                            "active_energies": final_active_energies,
                            "opp_active_id": final_opp_active_id,
                            "stadium_id": final_stadium_id,
                            "bench_ids": final_bench_ids,
                            "hand_size": len(final_ps.hand) if final_ps.hand is not None else 0,
                            "discard_size": len(final_ps.discard) if final_ps.discard is not None else 0,
                            "turn": final_obs.current.turn,
                            "has_attack_option": False,
                            "has_attach_option": False
                        }
                    
                    prizes_taken = pre["prizes"] - post["prizes"]
                    prizes_lost = pre["opp_prizes"] - post["opp_prizes"]
                    opp_kos = pre["opp_pokemon"] - post["opp_pokemon"]
                    own_kos = pre["pokemon"] - post["pokemon"]
                    energy_attached = post["energy"] - pre["energy"]
                    active_energy_attached = post.get("active_energy", 0) - pre.get("active_energy", 0)
                    bench_energy_attached = energy_attached - active_energy_attached
                    
                    # Stall penalty to prevent endless pass cycles
                    r_stall = -0.02
                        
                    r_prize_t = prizes_taken * 5.0 if prizes_taken > 0 else 0.0
                    r_prize_l = prizes_lost * 0.15 if prizes_lost > 0 else 0.0
                    r_ko = opp_kos * 0.10 if opp_kos > 0 else 0.0
                    r_own_ko = own_kos * 0.08 if own_kos > 0 else 0.0
                    
                    r_en = 0.0
                    if active_energy_attached > 0:
                        r_en += active_energy_attached * 0.10  # Reduced from 0.50 to prevent over-focus
                    if bench_energy_attached > 0:
                        r_en += bench_energy_attached * 0.02   # Reduced from 0.03
                        
                    # Principal RL Scientist - Optimized Mewtwo ex Reward Shaping
                    r_strategic = 0.0
                    action_type = pre.get("action_type", -1)
                    
                    # 1. Stadium Establishment
                    if pre.get("stadium_id") != 1257 and post.get("stadium_id") == 1257:
                        r_strategic += 0.25  # Prioritize establishing Team Rocket's Factory (+0.25)
                    
                    # 2. Action Heuristics & Search Efficiency & Sequencing
                    if action_type == 7:  # PLAY
                        played_id = pre.get("played_card_id", -1)
                        # Board development: play Tarountula or Mewtwo ex to bench
                        if played_id == 400: # Tarountula
                            r_strategic += 0.15  # Target setup
                        elif played_id == 431: # Mewtwo ex
                            r_strategic += 0.10
                        # Search efficiency: playing Transceiver (1134), Ariana (1216), or Ultra Ball (1121)
                        elif played_id in [1134, 1216, 1121, 1220]:
                            expected_min_diff = -2 if played_id == 1121 else 0
                            hand_diff = post["hand_size"] - pre["hand_size"]
                            if hand_diff >= expected_min_diff:
                                r_strategic += 0.20  # Boosted search/draw success
                            else:
                                r_strategic -= 0.05  # Penalize dead supporter/failed search
                        # Correct use of Energy Switch (1116)
                        elif played_id == 1116:
                            # Verify if Energy Switch enabled an active Mewtwo ex to attack (reach 3 energies)
                            if post["active_id"] == 431 and post["active_energies"] >= 3 and pre["active_energies"] < 3:
                                r_strategic += 0.25  # High swing turn reward
                            elif post["active_energies"] > pre["active_energies"]:
                                r_strategic += 0.10
                            else:
                                r_strategic -= 0.05
                        # General Supporter dead usage penalty
                        elif played_id in [1217, 1219, 1227]:  # Exclude Giovanni (1218)
                            hand_diff = post["hand_size"] - pre["hand_size"]
                            if hand_diff < 0:
                                r_strategic -= 0.05
                        elif played_id == 1218:  # Giovanni
                            # Reward using Giovanni to switch to a fully charged benched Mewtwo ex or Spidops
                            # if active is currently weak or a non-attacker (e.g. Articuno/Mimikyu or Mewtwo ex undercharged).
                            benched_ids = pre.get("bench_ids", [])
                            bench_energy = pre.get("energy", 0) - pre.get("active_energy", 0)
                            active_charged = (pre.get("active_id") == 431 and pre.get("active_energy", 0) >= 3) or (pre.get("active_id") == 401 and pre.get("active_energy", 0) >= 2)
                            
                            has_bench_attacker = False
                            for bid in benched_ids:
                                if bid in [431, 401]:
                                    has_bench_attacker = True
                                    break
                            
                            if has_bench_attacker and bench_energy >= 2 and not active_charged:
                                r_strategic += 0.25  # Strategic Giovanni play reward!
                        else:
                            r_strategic += 0.02
                            
                    elif action_type == 8:  # ATTACH
                        attached_id = pre.get("attached_card_id", -1)
                        target_id = pre.get("attached_target_id", -1)
                        
                        # Articuno and Mimikyu are defensive/barrier blockers and should never get energy attachments
                        if target_id in [414, 434] and attached_id in [1, 5, 15]:
                            if attached_id == 15:
                                r_strategic -= 3.0  # Massive penalty for wasting Special Energy on blockers!
                            else:
                                r_strategic -= 2.0  # Very high penalty for attaching basic energy to blockers!
                                
                        # Efficient Team Rocket Energy usage
                        elif attached_id == 15:  # Team Rocket's Energy
                            if target_id in [431, 401]:  # Mewtwo ex or Spidops
                                r_strategic += 0.25  # Boosted efficiency
                                if pre.get("active_id") != target_id or pre.get("active_energies", 0) >= 3:
                                    r_strategic += 0.15  # Extra reward for charging benched Mewtwo ex/Spidops to fuel Erasure Ball!
                            elif target_id in [414, 272]:
                                r_strategic -= 1.50  # Severe penalty for wasting Special Energy on Clefairy/Articuno
                                
                        # Proper basic energy attachment
                        elif attached_id == 5:  # Psychic Energy
                            if target_id == 431:  # Mewtwo ex
                                r_strategic += 0.20
                                if pre.get("active_id") != 431 or pre.get("active_energies", 0) >= 3:
                                    r_strategic += 0.15  # Extra reward for charging benched Mewtwo ex to build resources for Erasure Ball!
                            else:
                                r_strategic += 0.05
                        elif attached_id == 1:  # Grass Energy
                            if target_id in [400, 401]:  # Tarountula or Spidops
                                r_strategic += 0.20
                                if pre.get("active_id") != target_id or pre.get("active_energies", 0) >= 2:
                                    r_strategic += 0.10  # Extra reward for charging benched Spidops!
                            else:
                                r_strategic += 0.05
                                
                        # Tool Optimizations
                        elif attached_id == 1175:  # Brave Bangle
                            # Reward attaching to Active Spidops facing an ex opponent
                            if target_id == 401 and pre.get("opp_active_id") in [431, 272] and target_id == pre.get("active_id"):
                                r_strategic += 0.20
                            else:
                                r_strategic -= 0.05
                        elif attached_id == 1158:  # Maximum Belt
                            # Reward attaching to Active Mewtwo ex facing an ex opponent
                            if target_id == 431 and pre.get("opp_active_id") in [431, 272] and target_id == pre.get("active_id"):
                                r_strategic += 0.25
                            else:
                                r_strategic -= 0.05
                                
                    elif action_type == 9:  # EVOLVE
                        evolved_id = pre.get("evolved_card_id", -1)
                        if evolved_id == 401:  # Spidops
                            r_strategic += 0.30  # Crucial evolution setup
                        else:
                            r_strategic += 0.15
                            
                    elif action_type == 10:  # ABILITY
                        # Encourage using Rocket's Factory ability to draw cards
                        r_strategic += 0.10
                        
                    elif action_type == 12:  # RETREAT
                        # Base penalty for retreating to prevent infinite retreat loops and energy waste
                        r_strategic -= 0.15
                        # Strategic retreat from Mimikyu ex-immunity
                        if pre.get("opp_active_id") == 434 and pre.get("active_id") in [431, 272] and post.get("active_id") not in [431, 272]:
                            r_strategic += 0.30  # Excellent retreat to non-ex attacker against Mimikyu!
                        # Defensive retreat to Mimikyu barrier to stall and set up
                        elif post.get("active_id") == 434:
                            if pre.get("active_id") in [431, 401] and pre.get("active_energies", 0) < 3:
                                r_strategic += 0.20
                        elif pre.get("active_id") == 431 and pre.get("active_energies", 0) >= 3:
                            r_strategic -= 0.05
                            
                    elif action_type == 14:  # END (Turn Stall Decisions)
                        # Check if ending the turn was a bad decision (stalling) or a good decision
                        has_attack = pre.get("has_attack_option", False)
                        has_attach = pre.get("has_attach_option", False)
                        energy_already_attached = pre.get("energy_attached_flag", False)
                        
                        # Active Mewtwo ex / Clefairy ex facing a Mimikyu is immune, so attacking is bad anyway.
                        opp_immune = (pre.get("opp_active_id") == 434 and pre.get("active_id") in [431, 272])
                        
                        if has_attack and not opp_immune:
                            r_strategic -= 5.0  # Massive penalty for stalling when we can attack!
                        elif has_attach and not energy_already_attached:
                            r_strategic -= 3.0  # Big penalty for leaving energy in hand unattached
                        else:
                            # Good decision: we ended the turn because we had no constructive moves remaining
                            # We can stall/pass until we find the suitable cards to win
                            r_strategic += 0.3  # Reward for good pass, offsetting flat stall penalty
                            
                        # Extra penalty for ending the turn with an empty bench (high bench-out risk!)
                        if post.get("bench_size", 0) == 0:
                            r_strategic -= 1.0  # Big penalty for ending turn with 0 bench backup!
                            
                    elif action_type == 13:  # ATTACK
                        att_id = pre.get("attack_id", -1)
                        
                        # Mimikyu ex-immunity check: attacking Mimikyu active with an ex active is useless
                        if pre.get("opp_active_id") == 434 and pre.get("active_id") in [431, 272]:
                            r_strategic -= 0.20  # Wastes turn to attack immune Mimikyu
                        else:
                            if att_id == 608:  # Erasure Ball
                                if pre.get("active_energies", 0) < 3:
                                    r_strategic -= 0.05
                                else:
                                    r_strategic += 0.25
                            elif att_id == 560:  # Rocket Rush
                                r_strategic += 0.20
                            else:
                                r_strategic += 0.15
                                
                    # 3. Bench Quality & Overextension Management
                    r_bench = 0.0
                    if post["bench_size"] == 0:
                        r_bench -= 0.50  # Strongly penalise having no backup Pokemon!
                        if post.get("turn", 0) <= 2:
                            r_bench -= 0.50  # Extra -0.50 penalty during early turns (turn <= 2) to prevent turn 1 bench-out!
                    elif 2 <= post["bench_size"] <= 4:
                        r_bench += 0.05  # Optimal board development reward
                    elif post["bench_size"] == 5:
                        r_bench -= 0.02  # Overextension penalty
                        
                    # 4. Proper Mewtwo Timing & Charging
                    if post.get("active_id") == 431:
                        if post.get("active_energies", 0) < 2:
                            r_strategic -= 0.02  # Relaxed: Penalty for having a weak/undercharged Mewtwo active
                        elif post.get("active_energies", 0) >= 3:
                            r_strategic += 0.10  # Reward for maintaining a fully charged Mewtwo active
                            
                    # 5. Maintaining Multiple Attackers (Backup Attacker)
                    has_backup_attacker = False
                    for b_id in post.get("bench_ids", []):
                        if b_id in [431, 401]:  # Benched Mewtwo ex or Spidops
                            has_backup_attacker = True
                            break
                    if has_backup_attacker and (post["energy"] - post.get("active_energies", 0)) >= 2:
                        r_strategic += 0.08  # Reward for backup attacker development
                        
                    r_deck = 0.0
                    if post["deck_size"] == 0:
                        r_deck -= 0.50  # Penalize imminent deckout only
                        
                    step_reward = r_stall + r_prize_t - r_prize_l + r_ko - r_own_ko + r_en + r_bench + r_deck + r_strategic
                    
                    if i == 0:
                        rc_worker["prize_taken"] += r_prize_t
                        rc_worker["prize_lost"] += r_prize_l
                        rc_worker["kos"] += r_ko
                        rc_worker["own_kos"] += r_own_ko
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
                pred_values = [player_samples[s][0].pred_val for s in range(n_steps)]
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
                        
            result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, processed_samples, result, final_turn, rc_worker, entropy_accum, entropy_count, action_counts)))
            
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
                    selected, _ = mcts_agent(obs, sample_deck, client)
                else:
                    if opponent_name in ["Rulebasedmodel", "Rulebasedmodel_Iono", "Rulebasedmodel_Dragapult", "Rulebasedmodel_Mewtwo", "Rulebasedmodel_Mewtwo_Easy", "Rulebasedmodel_Abomasnow"]:
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
            slider_chars.append("●")
        elif i == center_pos:
            slider_chars.append("┼")
        else:
            slider_chars.append("─")
            
    slider_str = "".join(slider_chars)
    return f"REJECT [{A:.2f}] {slider_str} [{B:+.2f}] ACCEPT"


def run_sprt_evaluation(command_queues, result_queue, num_workers, sample_deck, opponent_decks, test_opponent_names,
                        alpha=0.05, beta=0.1, p0=0.50, p1=0.55):
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
        
    print(f"SPRT Evaluation started. alpha={alpha}, beta={beta}, p0={p0}, p1={p1}")
    deck_stats = {name: [0, 0, 0] for name in test_opponent_names}
    
    max_eval_games = 200
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
    # Filter opponent decks to exclude inefficient random agent models ,"Rulebasedmodel_Mewtwo_Easy","Rulebasedmodel_Mewtwo",
    opponent_decks = {k: v for k, v in opponent_decks.items() if k in ["Current (Self)","Rulebasedmodel_Abomasnow"]}
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=start_epoch)
    
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
    test_opponent_names = all_opponent_names

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
                alpha=0.05, beta=0.1, p0=0.50, p1=0.55
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
                if not train_opponent_names or random.random() < args.self_play_ratio:
                    if league_checkpoints and random.random() < 0.5:
                        opp_path = sample_league_opponent(active_elo, league_checkpoints, league_elos)
                        opponent_name = os.path.basename(opp_path)
                        opponent_deck = sample_deck
                        opponent_type = "League"
                        opponent_path = opp_path
                    else:
                        opponent_name = "Current (Self)"
                        opponent_deck = sample_deck
                        opponent_type = "Current"
                        opponent_path = None
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
                    opponent_name = random.choices(train_opponent_names, weights=probs, k=1)[0]
                    opponent_deck = opponent_decks[opponent_name]
                    opponent_type = "Rulebased"
                    opponent_path = None
                            
                return opponent_name, sample_deck, opponent_deck, opponent_type, opponent_path

            for w_idx in range(num_workers):
                if games_sent < args.self_play_episodes:
                    opp_name, s_deck, o_deck, opp_type, opp_path = get_next_self_play_args()
                    command_queues[w_idx].put(("PLAY_SELF", (s_deck, o_deck, opp_type, opp_path, opp_name)))
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
                    
                    games_received += 1
                    
                    if action_counts:
                        for act_k, act_v in action_counts.items():
                            epoch_action_counts[act_k] += act_v
                            
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
                    rc_count += len(samples) if len(samples) > 0 else 1
                    total_game_length += final_turn
                    total_games += 1
                    
                    entropy_accum += e_accum
                    entropy_count += e_count
                    
                    if games_sent < args.self_play_episodes:
                        opp_name, s_deck, o_deck, opp_type, opp_path = get_next_self_play_args()
                        command_queues[w_idx].put(("PLAY_SELF", (s_deck, o_deck, opp_type, opp_path, opp_name)))
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
                    for _ in range(128 - len(sample.policy)):
                        mask.append(0.0)
                        label_dec.append(0.0)
                        input_dec.offset.append(len(input_dec.index))

                mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(args.batch_size, -1)
                label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device).view(args.batch_size, -1)
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
        MAX_LEAGUE_SIZE = 10
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
                             rc["deckout"] / rc_div, rc["terminal"] / max(1, total_games),
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
