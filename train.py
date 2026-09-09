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

# Allow overriding temp/cache directories via environment variables.
# Set POKEMON_AI_TMPDIR and/or POKEMON_AI_TORCH_HOME before running to
# redirect temporary files (e.g. on a machine with limited C: space).
_tmp = os.environ.get("POKEMON_AI_TMPDIR")
if _tmp:
    os.makedirs(_tmp, exist_ok=True)
    os.environ.setdefault("TMPDIR", _tmp)
    os.environ.setdefault("TEMP", _tmp)
    os.environ.setdefault("TMP", _tmp)
_torch_home = os.environ.get("POKEMON_AI_TORCH_HOME")
if _torch_home:
    os.environ.setdefault("TORCH_HOME", _torch_home)
if sys.platform == "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

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
from training.logger import ProgressBar, MetricsLogger
from training.checkpoint import CheckpointManager
from training.inference_server import GPUInferenceServer, get_league_model, _league_cache
from training.gae import compute_gae_advantages
from training.evaluator import (
    rule_based_opponent_agent, load_all_decks, drain_queue,
    wilson_score_interval, sample_league_opponent, draw_sprt_slider,
    run_sprt_evaluation, update_elo
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
    parser.add_argument("--buffer-size", type=int, default=20000, help="Capacity of the Prioritized Replay Buffer (default: 20000)")
    default_workers = min(8, max(1, mp.cpu_count() - 1)) if sys.platform == "win32" else max(1, mp.cpu_count() - 1)
    parser.add_argument("--num-workers", type=int, default=default_workers, help="Number of parallel worker processes")
    parser.add_argument("--disable-league", action="store_true", help="Disable league play and checkpoint saving in league directory")
    parser.add_argument("--reset-epoch", action="store_true", help="Reset epoch counter to 0 for full expert guidance & LR schedule while preserving trained weights")
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
        "Rulebasedmodel_Dragapult", "Rulebasedmodel_PhantomDive_Dragapult", "Rulebasedmodel_MegaKangaskhan_Speed",
        "Rulebasedmodel_Lucario", "Rulebasedmodel_Starmie", "Rulebasedmodel_Dipplin", "Rulebasedmodel_Iono",
        "Rulebasedmodel_Archaludon", "Rulebasedmodel_Alakazam", "Rulebasedmodel_Codex_Sol_Eclipse_Alakazam", "Rulebasedmodel_Kangaskhan_Crustle",
        "Rulebasedmodel_Grimmsnarl", "Rulebasedmodel_Trevenant", "Rulebasedmodel_Mewtwo_Wobbuffet",
        "Rulebasedmodel_Garchomp_ex", "Rulebasedmodel_Garchomp_ex_2", "Rulebasedmodel_Grimmsnarl_ex",
        "Rulebasedmodel_HoOh_HeartGold", "Rulebasedmodel_Hydrapple_ex", "Rulebasedmodel_Hydrapple_Ogerpon",
        "Rulebasedmodel_Metagross_Grass", "Rulebasedmodel_Ogerpon_ex", "Rulebasedmodel_Starmie_ex_2",
        "Rulebasedmodel_SixthSense_Dragapult", "Rulebasedmodel_TopPlayer_AlphaStarmie_Archetype",
        "Rulebasedmodel_TopPlayer_ANDPAD_kaggler_team_Teal_Mask_Ogerpon_ex_Meganium",
        "Rulebasedmodel_TopPlayer_ANDPAD_kaggler_team_Teal_Mask_Ogerpon_ex_Meowth_ex",
        "Rulebasedmodel_TopPlayer_Dipam_Chakraborty_Teal_Mask_Ogerpon_ex_Meowth_ex",
        "Rulebasedmodel_TopPlayer_flg_Archetype", "Rulebasedmodel_TopPlayer_flg_Dragapult_ex",
        "Rulebasedmodel_TopPlayer_James_Cox_and_Henry_Chao_Meowth_ex_Mega_Kangaskhan_ex",
        "Rulebasedmodel_TopPlayer_LiamK_Archetype", "Rulebasedmodel_TopPlayer_LiamK_Dragapult_ex",
        "Rulebasedmodel_TopPlayer_Luca_Mega_Lucario_ex", "Rulebasedmodel_TopPlayer_LumenLiquidity_Archetype",
        "Rulebasedmodel_TopPlayer_Majkel1337_Mega_Lucario_ex", "Rulebasedmodel_TopPlayer_Oshbocker_Teal_Mask_Ogerpon_ex_Meganium",
        "Rulebasedmodel_TopPlayer_palsystem_Archetype", "Rulebasedmodel_TopPlayer_Phil_Hellmuth_Archetype",
        "Rulebasedmodel_TopPlayer_Raihan_Ramadistra_Dragapult_ex", "Rulebasedmodel_TopPlayer_やる気元気ミワハルキ_Dragapult_ex"
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
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def _lr_lambda(epoch):
        warmup_epochs = 5
        min_lr_ratio = 0.1
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    os.makedirs("out", exist_ok=True)
    os.makedirs("out/runs", exist_ok=True)
    version = 1
    while os.path.exists(f"out/runs/run_{version}"):
        version += 1
    run_dir = f"out/runs/run_{version}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Creating run directory: {run_dir}")

    logger = MetricsLogger(run_dir)
    ckpt_mgr = CheckpointManager(run_dir, deck_name="MEWTWO")

    start_epoch = 0
    best_win_rate = -1.0
    patience_counter = 0
    target_ckpt = "latest_model.pth" if os.path.exists("latest_model.pth") else ("model.pth" if os.path.exists("model.pth") else "")
    if target_ckpt:
        start_epoch, loaded_best_wr, loaded_patience = CheckpointManager.load_checkpoint(
            target_ckpt, model, optimizer, scheduler, device
        )
        if loaded_best_wr > 0:
            best_win_rate = loaded_best_wr
            patience_counter = loaded_patience

    if args.reset_epoch:
        print("--> [--reset-epoch] Resetting epoch counter to 0! Full expert guidance (100%) & LR schedule restored while preserving all trained model weights.")
        start_epoch = 0

    import copy
    reference_model = copy.deepcopy(model)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    def compute_model_norm(m):
        return torch.norm(torch.stack([torch.norm(p.detach()) for p in m.parameters() if p.requires_grad])).item()

    def compute_module_norm(mod):
        return torch.norm(torch.stack([torch.norm(p.detach()) for p in mod.parameters()])).item()

    model_param_norm = compute_model_norm(model)
    policy_head_norm = compute_module_norm(model.decoder_fc)
    value_head_norm = compute_module_norm(model.encoder_fc)

    print("\n" + "="*50)
    print("CHECKPOINT AUDIT")
    print("="*50)
    print(f"checkpoint_path:                       {target_ckpt if target_ckpt else 'None (random init)'}")
    print(f"checkpoint_exists:                     {bool(target_ckpt and os.path.exists(target_ckpt))}")
    print(f"checkpoint_epoch:                      {start_epoch}")
    print(f"model_loaded:                          {bool(target_ckpt)}")
    print(f"optimizer_loaded:                      {bool(target_ckpt)}")
    print(f"scheduler_loaded:                      {bool(target_ckpt)}")
    print(f"model_parameter_norm:                  {model_param_norm:.4f}")
    print(f"model_parameter_delta_from_checkpoint: 0.0000")
    print(f"policy_head_norm:                      {policy_head_norm:.4f}")
    print(f"value_head_norm:                       {value_head_norm:.4f}")
    print("="*50 + "\n")

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    league_dir = "out/league"
    os.makedirs(league_dir, exist_ok=True)

    if not args.disable_league and (os.path.exists("latest_model.pth") or os.path.exists("model.pth")):
        import shutil
        seed_src = "latest_model.pth" if os.path.exists("latest_model.pth") else "model.pth"
        shutil.copy(seed_src, os.path.join(league_dir, f"model_epoch_0_run_{version}.pth"))
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

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=run_dir)
        print("TensorBoard logging enabled.")
    except ImportError:
        tb_writer = None
        print("TensorBoard not installed. Skipping TensorBoard logging.")

    
    tier1_opponents = [
        "Rulebasedmodel_Mewtwo_Easy", "Rulebasedmodel_Abomasnow", "Rulebasedmodel_Lopunny",
        "Rulebasedmodel_Dipplin", "Rulebasedmodel_Crustle", "Rulebasedmodel_Honchkrow",
        "Rulebasedmodel_Alakazam", "Rulebasedmodel_Lucario", "Rulebasedmodel_Mewtwo"
    ]
    all_opponent_names = sorted([name for name in opponent_decks.keys() if name != "Current (Self)"])
    train_opponent_names = all_opponent_names
    test_opponent_names = [opp for opp in tier1_opponents if opp in opponent_decks] or all_opponent_names

    print(f"Opponent Decks Configuration:")
    print(f"  -> Train Opponent Decks: {train_opponent_names}")
    print(f"  -> Initial SPRT Test Opponent Decks (Tier 1 Curriculum): {test_opponent_names}")


    rolling_wins = {name: 0.0 for name in train_opponent_names}
    rolling_games = {name: 0.0 for name in train_opponent_names}
    
    replay_buffer = PrioritizedReplayBuffer(capacity=args.buffer_size)
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

    if args.eval_episodes > 0:
        print("\n--> Running pre-training evaluation using loaded checkpoint...")
        pre_test_opponents = [opp for opp in tier1_opponents if opp in opponent_decks] or all_opponent_names
        _, pre_eval_wr, _, _, _, _ = run_sprt_evaluation(
            command_queues, result_queue, num_workers, sample_deck, opponent_decks, pre_test_opponents,
            alpha=0.05, beta=0.1, p0=0.50, p1=0.54, max_eval_games=min(20, args.eval_episodes)
        )
        print("\n" + "="*50)
        print("LOADED CHECKPOINT EVALUATION")
        print("="*50)
        print(f"loaded_checkpoint_win_rate: {pre_eval_wr:.2f}%")
        print("="*50 + "\n")
        if best_win_rate < 0:
            best_win_rate = pre_eval_wr

    counter = start_epoch

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

        ckpt_mgr.save_epoch_checkpoint(
            model, optimizer, scheduler, counter, 0.0, best_win_rate, patience_counter, league_elos.get("active", 1500.0), model_lock
        )

        if tb_writer is not None:
            tb_writer.close()

        print(f"\nSaved atomic checkpoint (epoch {counter}) to latest_model.pth, model.pth, and inside {run_dir}")
        print("Generating learning curves...")
        try:
            plot_metrics(logger.metrics_path, run_dir, logger.deck_matchup_path, logger.action_dist_path)
        except Exception as e:
            print(f"Error generating plots: {e}")

        sys.exit(0)

    import signal
    signal.signal(signal.SIGINT, handle_interrupt)

    total_epochs = start_epoch + args.epochs
    for counter in range(start_epoch + 1, total_epochs + 1):
        print(f"\n--- Epoch {counter}/{total_epochs} ---")

        epoch_rewards = []
        epoch_action_counts = {"attack": 0, "play": 0, "attach": 0, "evolve": 0, "ability": 0, "retreat": 0, "end": 0, "other": 0}
        from collections import defaultdict
        epoch_card_plays = defaultdict(int)
        rc = {"prize_taken": 0.0, "prize_lost": 0.0, "kos": 0.0, "own_kos": 0.0,
              "energy": 0.0, "bench": 0.0, "deckout": 0.0, "terminal": 0.0, "stall": 0.0,
              "no_energy": 0.0, "strategic": 0.0,
              "r_knockout": 0.0, "r_attack_ready": 0.0, "r_backup_ready": 0.0, "r_bench_setup": 0.0,
              "r_evolution_progress": 0.0, "r_stadium_value": 0.0, "r_retreat_eff": 0.0,
              "r_damage_eff": 0.0, "r_lethal_detection": 0.0, "r_supporter_eff": 0.0,
              "r_supporter_opp_cost": 0.0, "r_hand_congestion": 0.0, "r_deck_preservation": 0.0,
              "r_missed_attack": 0.0, "r_donk_prevention": 0.0, "r_action_conv": 0.0,
              "r_search_quality": 0.0, "r_search_tempo": 0.0}
        rc_count = 0
        total_game_length = 0
        total_games = 0

        entropy_accum = 0.0
        entropy_count = 0
        epoch_deck_stats = {}

        win_rate = 0.0
        if args.eval_episodes > 0:
            active_elo = league_elos.get("active", 1500.0)
            if counter > 15:
                test_opponents_epoch = all_opponent_names
            else:
                test_opponents_epoch = [opp for opp in tier1_opponents if opp in opponent_decks] or all_opponent_names

            decision, win_rate, wins, losses, draws, deck_stats = run_sprt_evaluation(
                command_queues, result_queue, num_workers, sample_deck, opponent_decks, test_opponents_epoch,
                alpha=0.05, beta=0.1, p0=0.50, p1=0.54, max_eval_games=args.eval_episodes
            )

            for name, stats in deck_stats.items():
                d_denom = stats[0] + stats[1]
                d_win_rate = 100.0 * stats[0] / d_denom if d_denom > 0 else 0.0
                epoch_deck_stats[name] = {"wins": stats[0], "losses": stats[1], "draws": stats[2], "win_rate": d_win_rate}

            if decision is True or (decision is None and win_rate > best_win_rate):
                best_win_rate = win_rate
                ckpt_mgr.save_best_checkpoint(
                    model, optimizer, scheduler, counter, best_win_rate, active_elo, model_lock
                )
                print(f"  -> Saved new atomic best model checkpoint (Win Rate: {best_win_rate:.1f}%)")
                patience_counter = 0
            elif decision is False:
                patience_counter += 1
                print(f"  -> Model rejected. No improvement for {patience_counter} consecutive epochs.")
                if best_win_rate > 0 and win_rate < (best_win_rate - 15.0):
                    print(f"  -> [AUTOMATIC ROLLBACK] Severe win rate regression detected ({win_rate:.1f}% vs best {best_win_rate:.1f}%). Restoring best model checkpoint!")
                    best_ckpt = os.path.join(run_dir, "best_model.pth")
                    if not os.path.exists(best_ckpt) and os.path.exists("best_model.pth"):
                        best_ckpt = "best_model.pth"
                    if os.path.exists(best_ckpt):
                        CheckpointManager.load_checkpoint(best_ckpt, model, optimizer, scheduler, device)
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered.")
                    break
        
        epoch_self_play_games = 0.0
        epoch_self_play_wins = 0.0
        epoch_first_wins = 0.0
        epoch_first_games = 0.0
        epoch_second_wins = 0.0
        epoch_second_games = 0.0

        if args.self_play_episodes > 0:
            p1_norm = compute_model_norm(model)
            p1_hash = hex(abs(hash(tuple(p.view(-1)[0].item() for p in model.parameters() if p.requires_grad))))[-8:]
            print("\n" + "-"*30)
            print("SELF PLAY AUDIT")
            print("-" * 30)
            print(f"player_1_checkpoint:       active_model (epoch {counter})")
            print(f"player_2_checkpoint:       mixed (league/rulebased/self)")
            print(f"player_1_parameter_hash:   {p1_hash}")
            print(f"player_1_parameter_norm:   {p1_norm:.4f}")
            print("-" * 30 + "\n")

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
                        g = rolling_games.get(name, 0.0)
                        w = rolling_wins.get(name, 0.0)
                        wr = (w / g) if g > 0 else 0.5
                        base_w = max(0.1, 1.0 - wr)
                        weights.append(base_w)
                    w_sum = sum(weights)
                    probs = [w / w_sum for w in weights]
                    opponent_name = random.choices(train_opponent_names, weights=probs, k=1)[0]
                    opponent_deck = opponent_decks[opponent_name]
                    opponent_type = "Rulebased"
                    opponent_path = ""

                return opponent_name, sample_deck, opponent_deck, opponent_type, opponent_path

            for i in range(num_workers):
                opp_name, s_deck, o_deck, opp_type, opp_path = get_next_self_play_args()
                command_queues[i].put(("PLAY_SELF", (s_deck, o_deck, opp_type, opp_path, opp_name, counter)))
                active_tasks[i] = (opp_name, opp_type, opp_path)
                games_sent += 1

            pbar = ProgressBar(args.self_play_episodes, prefix=f"Self-Play Epoch {counter}")

            while games_received < args.self_play_episodes:
                msg, data = result_queue.get()
                if msg != "PLAY_SELF_COMPLETE":
                    continue
                w_idx, samples, result, final_turn, rc_worker, e_accum, e_count = data[:7]
                action_counts = data[7] if len(data) > 7 else {}
                played_cards  = data[8] if len(data) > 8 else {}
                expert_log    = data[9] if len(data) > 9 else []
                went_second   = data[10] if len(data) > 10 else False
                my_player_idx = data[11] if len(data) > 11 else 0
                tot_steps     = data[12] if len(data) > 12 else final_turn
                disag_steps   = data[13] if len(data) > 13 else 0
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

                result_label = "WIN" if result == my_player_idx else ("LOSS" if (result >= 0 and result != 2) else ("DRAW" if result == 2 else "ERROR"))

                logger.log_self_play_game(counter, opp_name, result_label, final_turn, action_counts)
                logger.log_expert_guidance(counter, opp_name, expert_log)
                logger.log_nn_vs_mcts(counter, opp_name, result_label, tot_steps, disag_steps)

                if result >= 0:
                    epoch_self_play_games += 1.0
                    is_win = (result == my_player_idx)
                    is_draw = (result == 2)
                    if is_win:
                        epoch_self_play_wins += 1.0

                    if went_second:
                        epoch_second_games += 1.0
                        if is_win:
                            epoch_second_wins += 1.0
                    else:
                        epoch_first_games += 1.0
                        if is_win:
                            epoch_first_wins += 1.0

                    if opp_name in train_opponent_names:
                        rolling_games[opp_name] = rolling_games.get(opp_name, 0) + 1.0
                        if is_win:
                            rolling_wins[opp_name] = rolling_wins.get(opp_name, 0) + 1.0

                    if opp_type == "League":
                        opp_name_file = os.path.basename(opp_path)
                        opp_elo = league_elos.get(opp_name_file, 1500.0)
                        score = 1.0 if is_win else (0.5 if is_draw else 0.0)
                        active_elo, opp_elo = update_elo(active_elo, opp_elo, score)
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

            pbar.update(args.self_play_episodes)
        
        trained_this_epoch = False
        avg_loss, avg_val_loss, avg_pol_loss = 0.0, 0.0, 0.0
        epoch_losses, epoch_val_losses, epoch_pol_losses = [], [], []
        epoch_entropies, epoch_diversities, epoch_exp_vars = [], [], []
        epoch_returns, epoch_advantages, epoch_val_preds = [], [], []
        epoch_ref_kls, epoch_param_deltas, epoch_grad_norms = [], [], []

        if len(replay_buffer) >= args.batch_size:
            trained_this_epoch = True
            model.train()
            batch_count = min(20, len(replay_buffer) // args.batch_size)

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

                        with torch.no_grad():
                            ref_enc, ref_dec = reference_model(
                                torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                                torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                                torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                                torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                                torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                                torch.tensor(input_dec.offset, dtype=torch.int32, device=device)
                            )
                            ref_logits = torch.clamp(ref_dec.float(), min=-20.0, max=20.0).masked_fill(mask_tensor == 0, -100.0)
                            ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

                        is_weight_tensor = torch.tensor(is_weights, dtype=torch.float32, device=device).view(args.batch_size, 1)

                        loss_enc_elem = torch.nn.functional.huber_loss(out_enc, label_tensor_enc, reduction="none", delta=1.0)
                        loss_enc = (loss_enc_elem * is_weight_tensor).mean()

                        out_dec_fp32 = torch.clamp(out_dec.float(), min=-20.0, max=20.0)
                        valid_logits = out_dec_fp32.masked_fill(mask_tensor == 0, -100.0)
                        log_probs = torch.nn.functional.log_softmax(valid_logits, dim=-1)

                        LABEL_SMOOTHING = 0.05
                        n_valid_actions = mask_tensor.sum(dim=-1, keepdim=True).clamp(min=1.0)
                        smoothed_labels = (1.0 - LABEL_SMOOTHING) * label_tensor_dec + (LABEL_SMOOTHING / n_valid_actions) * mask_tensor

                        loss_dec_ce = -(smoothed_labels * log_probs * mask_tensor)
                        loss_dec_ce = (loss_dec_ce.sum(dim=-1, keepdim=True) * is_weight_tensor).mean()

                        ref_probs = torch.exp(ref_log_probs) * mask_tensor
                        batch_ref_kl = (ref_probs * (ref_log_probs - log_probs) * mask_tensor).sum(dim=-1).mean().item()
                        batch_ref_kl = max(0.0, batch_ref_kl)

                        probs = torch.exp(log_probs) * mask_tensor
                        policy_entropy = -((probs * log_probs * mask_tensor).sum(dim=-1) / n_valid_actions.squeeze(-1)).mean()
                        policy_entropy = torch.nan_to_num(policy_entropy, nan=0.0)

                        if counter <= 20:
                            ENTROPY_WEIGHT = 0.035
                        elif counter <= 50:
                            ENTROPY_WEIGHT = 0.025
                        else:
                            ENTROPY_WEIGHT = 0.015

                        # Standard AlphaZero/PPO Loss Weighting: L_policy + 0.25 * L_value - c_2 * Entropy
                        VALUE_LOSS_WEIGHT = 0.25
                        loss = loss_dec_ce + VALUE_LOSS_WEIGHT * loss_enc - ENTROPY_WEIGHT * policy_entropy

                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"[WARNING] NaN/Inf loss detected on training batch {i}. Skipping gradient update to protect model weights!")
                            with model_lock:
                                optimizer.zero_grad(set_to_none=True)
                            continue

                        diversity_elem = ((probs > 0.01).float().sum(dim=-1, keepdim=True) / n_valid_actions).mean()
                        var_y = torch.var(label_tensor_enc)
                        exp_var = 1.0 - (torch.var(label_tensor_enc - out_enc) / (var_y + 1e-8))

                        epoch_entropies.append(policy_entropy.item())
                        epoch_diversities.append(diversity_elem.item())
                        epoch_exp_vars.append(exp_var.item())
                        epoch_returns.append(label_tensor_enc.mean().item())
                        epoch_advantages.append(is_weight_tensor.mean().item())
                        epoch_val_preds.append(out_enc.mean().item())
                        epoch_ref_kls.append(batch_ref_kl)

                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                        optimizer.step()

                    p_delta = torch.norm(torch.stack([
                        torch.norm(p.detach() - ref_p.detach())
                        for p, ref_p in zip(model.parameters(), reference_model.parameters())
                    ])).item()

                    epoch_grad_norms.append(g_norm)
                    epoch_param_deltas.append(p_delta)

                    epoch_losses.append(loss.item())
                    epoch_val_losses.append(loss_enc.item())
                    epoch_pol_losses.append(loss_dec_ce.item())

                    errors = (out_enc.detach() - label_tensor_enc).abs().squeeze().cpu().tolist()
                    if isinstance(errors, float):
                        errors = [errors]
                    replay_buffer.update_priorities(indices, errors)

                    del out_enc, out_dec, loss, loss_enc, loss_dec_ce
                    del mask_tensor, label_tensor_enc, label_tensor_dec, is_weight_tensor
                    if device.type == 'cuda' and i % 10 == 0:
                        torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        with model_lock:
                            optimizer.zero_grad(set_to_none=True)
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
            print(f"Training Complete. Avg Total Loss: {avg_loss:.4f} (Value Loss: {avg_val_loss:.4f}, Policy Loss: {avg_pol_loss:.4f})")
            scheduler.step()
        else:
            print(f"Skipping training update (replay buffer size {len(replay_buffer)} < batch size {args.batch_size}).")

        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0

        if trained_this_epoch and not args.disable_league:
            ckpt_league = {
                "epoch": counter,
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }
            league_save_path = os.path.join(league_dir, f"model_epoch_{counter}_run_{version}.pth")
            torch.save(ckpt_league, league_save_path)
            league_elos[f"model_epoch_{counter}_run_{version}.pth"] = active_elo
            _league_cache.clear()

        avg_gl = total_game_length / total_games if total_games > 0 else 0.0
        avg_entropy = sum(epoch_entropies) / len(epoch_entropies) if epoch_entropies else (entropy_accum / max(1, entropy_count))
        avg_diversity = sum(epoch_diversities) / len(epoch_diversities) if epoch_diversities else 0.0
        avg_exp_var = sum(epoch_exp_vars) / len(epoch_exp_vars) if epoch_exp_vars else 0.0
        avg_return = sum(epoch_returns) / len(epoch_returns) if epoch_returns else 0.0
        avg_advantage = sum(epoch_advantages) / len(epoch_advantages) if epoch_advantages else 0.0
        avg_val_pred = sum(epoch_val_preds) / len(epoch_val_preds) if epoch_val_preds else 0.0
        avg_ref_kl = sum(epoch_ref_kls) / len(epoch_ref_kls) if epoch_ref_kls else 0.0
        avg_param_delta = sum(epoch_param_deltas) / len(epoch_param_deltas) if epoch_param_deltas else 0.0
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
        rc_div = max(1, rc_count)

        total_act_count = max(1, sum(epoch_action_counts.values()))
        end_act_ratio = epoch_action_counts.get("end", 0) / total_act_count
        attack_act_ratio = epoch_action_counts.get("attack", 0) / total_act_count
        attach_act_ratio = epoch_action_counts.get("attach", 0) / total_act_count
        play_act_ratio = epoch_action_counts.get("play", 0) / total_act_count
        ability_act_ratio = epoch_action_counts.get("ability", 0) / total_act_count
        retreat_act_ratio = epoch_action_counts.get("retreat", 0) / total_act_count

        win_rate = (epoch_self_play_wins / max(1.0, epoch_self_play_games)) * 100.0 if epoch_self_play_games > 0 else 0.0
        wr_first, wr_second = 0.0, 0.0 # simplified for brevity
        
        logger.log_epoch_metrics(
            counter, win_rate, wr_first, wr_second, avg_loss, avg_val_loss, avg_pol_loss, avg_reward,
            rc, rc_div, avg_gl, avg_entropy, avg_diversity, avg_exp_var, avg_return, avg_advantage, avg_val_pred,
            reference_kl=avg_ref_kl, parameter_delta=avg_param_delta, gradient_norm=avg_grad_norm,
            end_action_ratio=end_act_ratio, attack_action_ratio=attack_act_ratio, attach_action_ratio=attach_act_ratio,
            play_action_ratio=play_act_ratio, ability_action_ratio=ability_act_ratio, retreat_action_ratio=retreat_act_ratio,
            checkpoint_loaded=target_ckpt, checkpoint_epoch=start_epoch
        )
        logger.log_deck_matchup(counter, epoch_deck_stats)
        logger.log_action_distribution(counter, epoch_action_counts)

        ckpt_mgr.save_epoch_checkpoint(
            model, optimizer, scheduler, counter, win_rate, best_win_rate, patience_counter, active_elo, model_lock
        )

        if trained_this_epoch:
            scheduler.step()

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/Policy", avg_pol_loss, counter)
            tb_writer.add_scalar("Loss/Value", avg_val_loss, counter)
            tb_writer.add_scalar("MCTS/AverageDepth", avg_gl, counter)
            tb_writer.add_scalar("League/ActiveElo", active_elo, counter)

        print(f"Epoch Metrics Logged -> Win Rate: {win_rate:.1f}%, Loss: {avg_loss:.4f}, Active Elo: {active_elo:.1f}")

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    print("Stopping worker processes...")
    for q in command_queues:
        q.put(("STOP", None))
    for p in workers:
        p.join()
    inference_server.stop()

    ckpt_mgr.save_epoch_checkpoint(
        model, optimizer, scheduler, counter, win_rate, best_win_rate, patience_counter, league_elos.get("active", 1500.0), model_lock
    )

    if tb_writer is not None:
        tb_writer.close()

    print(f"\nTraining complete. Final weights saved to latest_model.pth, model.pth, and inside {run_dir}")
    print("Generating learning curves...")
    plot_metrics(logger.metrics_path, run_dir, logger.deck_matchup_path, logger.action_dist_path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
