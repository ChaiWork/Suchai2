import argparse
import csv
import glob
import math
import os
import random
import sys
import time
import queue
import threading
import multiprocessing as mp

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
try:
    from expert_knowledge import get_expert_bonus, EXPERT_WEIGHT, USE_EXPERT_GUIDANCE
except ImportError:
    from src.expert_knowledge import get_expert_bonus, EXPERT_WEIGHT, USE_EXPERT_GUIDANCE

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_observation_class, OptionType, AreaType, SelectContext, CardType, EnergyType

from training.card_database import (
    card_table, attack_table, evolves_from_set, get_card_data,
    is_defensive_blocker, can_attack, count_effective_energy_cards,
    count_attached_energy, count_active_energy, count_pokemon
)
from training.replay_buffer import PrioritizedReplayBuffer
from training.logger import ProgressBar
from training.inference_server import GPUInferenceServer, get_league_model, _league_cache
from training.gae import compute_gae_advantages
from training.evaluator import (
    rule_based_opponent_agent, load_all_decks, drain_queue,
    wilson_score_interval, sample_league_opponent, draw_sprt_slider,
    run_sprt_evaluation
)
from training.worker import worker_loop


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the MCTS Pokemon TCG AI Agent.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Target evaluation games (default: 50)")
    parser.add_argument("--self-play-episodes", type=int, default=100, help="Number of self-play games for data collection (default: 100)")
    parser.add_argument("--self-play-ratio", type=float, default=0.5, help="Ratio of games played against Current (Self) vs rule-based bots (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for model training (default: 128)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping based on evaluation win rate (default: 10)")
    default_workers = min(8, max(1, mp.cpu_count() - 1)) if sys.platform == "win32" else max(1, mp.cpu_count() - 1)
    parser.add_argument("--num-workers", type=int, default=default_workers, help="Number of parallel worker processes")
    parser.add_argument("--disable-league", action="store_true", help="Disable league play and checkpoint saving in league directory")
    args = parser.parse_args()

    # Reproducibility
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    opponent_decks = load_all_decks()
    allowed_opponents = [
        "Current (Self)",
        # --- EASY OPPONENT DECKS ---
        "Rulebasedmodel_Mewtwo_Easy", "Rulebasedmodel_Mewtwo", "Rulebasedmodel_Abomasnow",
        "Rulebasedmodel_Crustle", "Rulebasedmodel_Honchkrow", "Rulebasedmodel_Lopunny",
        "Rulebasedmodel_Marnie_Kangaskhan", "Rulebasedmodel_Typhlosion",
        # --- HARD OPPONENT DECKS ---
        "Rulebasedmodel_Dragapult", "Rulebasedmodel_Lucario", "Rulebasedmodel_Starmie",
        "Rulebasedmodel_Dipplin", "Rulebasedmodel_Iono", "Rulebasedmodel_Archaludon",
        "Rulebasedmodel_Alakazam", "Rulebasedmodel_Kangaskhan_Crustle", "Rulebasedmodel_Grimmsnarl",
        "Rulebasedmodel_Trevenant", "Rulebasedmodel_Mewtwo_Wobbuffet", "Rulebasedmodel_Garchomp_ex",
        "Rulebasedmodel_Garchomp_ex_2", "Rulebasedmodel_Grimmsnarl_ex", "Rulebasedmodel_HoOh_HeartGold",
        "Rulebasedmodel_Hydrapple_ex", "Rulebasedmodel_Hydrapple_Ogerpon", "Rulebasedmodel_Metagross_Grass",
        "Rulebasedmodel_Ogerpon_ex", "Rulebasedmodel_Starmie_ex_2"
    ]
    opponent_decks = {k: v for k, v in opponent_decks.items() if k in allowed_opponents}
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
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

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
        writer.writerow(["epoch", "win_rate", "avg_loss", "value_loss", "policy_loss", "avg_reward",
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

    expert_guidance_path = os.path.join(run_dir, "expert_guidance.csv")
    with open(expert_guidance_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "opponent", "trigger", "action_type", "bonus",
            "game_result", "turn", "my_prizes", "opp_prizes"
        ])

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
                can_use_league = bool(league_checkpoints)
                max_league_prob = getattr(args, "self_play_ratio", 0.10)
                league_prob = min(max_league_prob, 0.02 * counter)
                use_league = can_use_league and bool(train_opponent_names) and (random.random() < league_prob)

                if use_league:
                    opp_path = sample_league_opponent(active_elo, league_checkpoints, league_elos)
                    opponent_name = os.path.basename(opp_path)
                    opponent_deck = sample_deck
                    opponent_type = "League"
                    opponent_path = opp_path
                else:
                    weights = []
                    for name in train_opponent_names:
                        g = rolling_games[name]
                        w = rolling_wins[name]
                        wr = w / g if g > 0 else 0.5
                        base_w = max(0.1, 1.0 - wr)
                        weights.append(base_w)
                    w_sum = sum(weights)
                    probs = [w / w_sum for w in weights]
                    
                    for i_op in range(len(probs)):
                        if probs[i_op] > 0.25:
                            diff = probs[i_op] - 0.25
                            probs[i_op] = 0.25
                            other_indices = [idx for idx in range(len(probs)) if idx != i_op]
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
                    played_cards  = data[8] if len(data) > 8 else {}
                    expert_log    = data[9] if len(data) > 9 else []
                    
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
                    
                    if expert_log:
                        with open(expert_guidance_path, mode="a", newline="", encoding="utf-8-sig") as f_eg:
                            eg_writer = csv.writer(f_eg)
                            for entry in expert_log:
                                eg_writer.writerow([
                                    counter,
                                    opp_name,
                                    entry.get("trigger", "NONE"),
                                    entry.get("action_type", -1),
                                    entry.get("bonus", 0.0),
                                    result,
                                    entry.get("turn", -1),
                                    entry.get("my_prizes", -1),
                                    entry.get("opp_prizes", -1),
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
        avg_val_loss = 0.0
        avg_pol_loss = 0.0
        last_loss_enc_val = 0.0
        last_loss_dec_ce_val = 0.0
        trained_this_epoch = False
        if len(replay_buffer) >= args.batch_size:
            trained_this_epoch = True
            if device.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            print("Training Start.")
            model.train()
            batch_count = min(50, len(replay_buffer) // args.batch_size)
            print(f"Total training buffer size: {len(replay_buffer)}, Batch Count: {batch_count}")
            
            epoch_losses = []
            epoch_val_losses = []
            epoch_pol_losses = []
            for i in range(batch_count):
                try:
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

                        out_dec_fp32 = out_dec.float()
                        masked_logits = out_dec_fp32 + (1.0 - mask_tensor) * (-1e4)
                        log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
                        loss_dec_ce = -(label_tensor_dec * log_probs * mask_tensor)
                        loss_dec_ce = (loss_dec_ce.sum(dim=-1, keepdim=True) * is_weight_tensor).mean()

                        # Policy Entropy Regularization: H(pi) = -sum(p * log_p) to prevent policy collapse
                        probs = torch.exp(log_probs) * mask_tensor
                        policy_entropy = -(probs * log_probs * mask_tensor).sum(dim=-1, keepdim=True).mean()
                        ENTROPY_WEIGHT = 0.01

                        loss = loss_enc + loss_dec_ce - ENTROPY_WEIGHT * policy_entropy

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
                    epoch_val_losses.append(loss_enc.item())
                    epoch_pol_losses.append(loss_dec_ce.item())

                    errors = (out_enc.detach() - label_tensor_enc).abs().squeeze().cpu().tolist()
                    if isinstance(errors, float):
                        errors = [errors]
                    replay_buffer.update_priorities(indices, errors)

                    last_loss_enc_val = loss_enc.item()
                    last_loss_dec_ce_val = loss_dec_ce.item()

                    del out_enc, out_dec, loss, loss_enc, loss_dec_ce
                    del mask_tensor, label_tensor_enc, label_tensor_dec, is_weight_tensor
                    if device.type == 'cuda' and i % 10 == 0:
                        torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        import gc
                        gc.collect()
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        print(f"[WARNING] CUDA Out of Memory on training batch {i}. Cleared CUDA cache & continuing safely.")
                        continue
                    else:
                        raise e
                
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses) if epoch_val_losses else 0.0
            avg_pol_loss = sum(epoch_pol_losses) / len(epoch_pol_losses) if epoch_pol_losses else 0.0
            print(f"Training Finish. Average Loss: {avg_loss:.4f} (Value Loss: {avg_val_loss:.4f}, Policy Loss: {avg_pol_loss:.4f})")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
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

        try:
            from src.configs.active_deck import get_active_deck_name
            active_deck_tag = get_active_deck_name().lower()
        except Exception:
            active_deck_tag = os.getenv("ACTIVE_DECK", "MEWTWO").lower()

        deck_model_name = f"model_{active_deck_tag}.pth"
        torch.save(checkpoint, deck_model_name)
        
        if not args.disable_league:
            league_model_path = os.path.join(league_dir, f"model_epoch_{counter}_run_{version}.pth")
            torch.save(checkpoint, league_model_path)
            print(f"Saved checkpoint: {epoch_model_path}, model.pth, {deck_model_name}, and {league_model_path}")
        else:
            print(f"Saved checkpoint: {epoch_model_path}, model.pth, and {deck_model_name}")

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

        _league_cache.clear()

        avg_gl = total_game_length / total_games if total_games > 0 else 0.0
        avg_entropy = entropy_accum / max(1, entropy_count)
        rc_div = max(1, rc_count)
        
        with open(metrics_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([counter, win_rate, avg_loss, avg_val_loss, avg_pol_loss, avg_reward,
                             rc["prize_taken"] / rc_div, rc["prize_lost"] / rc_div,
                             rc["kos"] / rc_div, rc["own_kos"] / rc_div,
                             rc["energy"] / rc_div, rc["bench"] / rc_div,
                             rc["deckout"] / rc_div, rc["terminal"] / rc_div,
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
            tb_writer.add_scalar("Loss/Policy", last_loss_dec_ce_val, counter)
            tb_writer.add_scalar("Loss/Value",  last_loss_enc_val,    counter)
            tb_writer.add_scalar("MCTS/AverageDepth", avg_gl, counter)
            tb_writer.add_scalar("MCTS/Entropy", avg_entropy, counter)
            tb_writer.add_scalar("League/ActiveElo", active_elo, counter)
            tb_writer.add_scalar("Generalization/OOD_WinRate", win_rate, counter)

        print(f"Epoch Metrics Logged -> Win Rate: {win_rate:.1f}%, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, Active Elo: {active_elo:.1f}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
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
    
    try:
        from src.configs.active_deck import get_active_deck_name
        active_deck_tag = get_active_deck_name().lower()
    except Exception:
        active_deck_tag = os.getenv("ACTIVE_DECK", "MEWTWO").lower()

    deck_final_name = f"model_{active_deck_tag}.pth"
    torch.save(final_checkpoint, deck_final_name)
    
    if tb_writer is not None:
        tb_writer.close()
        
    print(f"\nTraining complete. Final weights saved to model.pth, {deck_final_name}, and inside {run_dir}")
    print(f"Best model (peak win rate) preserved in best_model.pth")

    print("Generating learning curves...")
    plot_metrics(metrics_path, run_dir, deck_matchup_path, action_dist_path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
