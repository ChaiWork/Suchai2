"""
Research-Grade Training Metrics Logger for Pokémon TCG RL Agent.
Provides atomic, real-time CSV flushing and OS disk synchronization (os.fsync)
so metrics can be viewed live in external tools without buffering delays.
"""

import os
import csv
import sys
from typing import Dict, Any, List, Optional


class ProgressBar:
    """Terminal progress bar utility."""

    def __init__(self, total: int, prefix: str = "", length: int = 30):
        self.total = max(1, total)
        self.prefix = prefix
        self.length = length

    def update(self, current: int, suffix: str = ""):
        percent = min(100.0, max(0.0, 100.0 * (current / float(self.total))))
        filled_length = int(self.length * current // self.total)
        bar = "=" * filled_length + "-" * (self.length - filled_length)
        sys.stdout.write(f"\r{self.prefix} [{bar}] {percent:5.1f}% {suffix}")
        sys.stdout.flush()
        if current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


class MetricsLogger:

    """
    Manages CSV log files for reinforcement learning training:
    - training_metrics.csv
    - action_distribution.csv
    - deck_matchup.csv
    - self_play_games.csv
    - expert_guidance.csv
    """

    METRICS_HEADER = [
        "epoch", "win_rate", "win_rate_first", "win_rate_second", "avg_loss", "value_loss", "policy_loss", "avg_reward",
        "r_prize_taken", "r_prize_lost", "r_kos", "r_own_kos",
        "r_energy", "r_bench", "r_deckout", "r_terminal", "r_stall", "r_no_energy", "r_strategic",
        "r_knockout", "r_attack_ready", "r_backup_ready", "r_bench_setup", "r_evolution_progress", "r_stadium_value",
        "r_retreat_eff", "r_damage_eff", "r_lethal_detection", "r_supporter_eff", "r_supporter_opp_cost",
        "r_hand_congestion", "r_deck_preservation", "r_missed_attack", "r_donk_prevention", "r_action_conv",
        "r_search_quality", "r_search_tempo",
        "avg_game_length", "policy_entropy", "action_diversity", "explained_variance", "mean_return", "mean_advantage", "value_prediction_mean",
        "reference_kl", "parameter_delta", "gradient_norm", "end_action_ratio", "attack_action_ratio", "attach_action_ratio", "play_action_ratio", "ability_action_ratio", "retreat_action_ratio",
        "checkpoint_loaded", "checkpoint_epoch",
        # Architecture health metrics: NN vs MCTS decision quality
        "nn_mcts_disagreement_rate",  # Fraction of steps where MCTS chose different action than NN top-1
        "expert_scale_mean",           # Average expert confidence scale this epoch
        "expert_bonus_mean",           # Average absolute expert bonus magnitude
    ]

    ACTION_DIST_HEADER = [
        "epoch", "attack", "play", "attach", "evolve", "ability", "retreat", "end", "other"
    ]

    DECK_MATCHUP_HEADER = [
        "epoch", "opponent_name", "wins", "losses", "draws", "win_rate"
    ]

    SELF_PLAY_HEADER = [
        "epoch", "opponent_name", "result", "turns", "attack", "play", "attach", "evolve", "ability", "retreat", "end", "other"
    ]

    EXPERT_LOG_HEADER = [
        "epoch", "opponent_name", "trigger", "action_type", "bonus", "result", "turn", "my_prizes", "opp_prizes"
    ]

    NN_VS_MCTS_HEADER = [
        # Per-game record of when MCTS disagreed with the NN's raw top-1 choice
        "epoch", "opponent_name", "result", "total_steps",
        "disagreement_steps",    # Steps where MCTS chose different action than NN top-1
        "disagreement_rate",     # disagreement_steps / total_steps
        "nn_won_disagreements",  # Steps where final action matched NN, not MCTS-expert
    ]

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.metrics_path     = os.path.join(self.run_dir, "training_metrics.csv")
        self.action_dist_path = os.path.join(self.run_dir, "action_distribution.csv")
        self.deck_matchup_path = os.path.join(self.run_dir, "deck_matchup.csv")
        self.self_play_path   = os.path.join(self.run_dir, "self_play_games.csv")
        self.expert_log_path  = os.path.join(self.run_dir, "expert_guidance.csv")
        self.nn_vs_mcts_path  = os.path.join(self.run_dir, "nn_vs_mcts.csv")

        self._init_file(self.metrics_path, self.METRICS_HEADER)
        self._init_file(self.action_dist_path, self.ACTION_DIST_HEADER)
        self._init_file(self.deck_matchup_path, self.DECK_MATCHUP_HEADER)
        self._init_file(self.self_play_path, self.SELF_PLAY_HEADER)
        self._init_file(self.expert_log_path, self.EXPERT_LOG_HEADER)
        self._init_file(self.nn_vs_mcts_path, self.NN_VS_MCTS_HEADER)

    def _init_file(self, filepath: str, header: List[str]) -> None:
        """Creates CSV file with header if it does not already exist."""
        if not os.path.exists(filepath):
            with open(filepath, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass

    def _write_and_flush(self, filepath: str, row: List[Any]) -> None:
        """Appends a single row to CSV file and immediately flushes and syncs to disk."""
        with open(filepath, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

    def log_epoch_metrics(
        self,
        epoch: int,
        win_rate: float,
        wr_first: float,
        wr_second: float,
        avg_loss: float,
        val_loss: float,
        pol_loss: float,
        avg_reward: float,
        rc: Dict[str, float],
        rc_div: float,
        avg_gl: float,
        avg_entropy: float,
        avg_diversity: float,
        avg_exp_var: float,
        avg_return: float,
        avg_advantage: float,
        avg_val_pred: float,
        reference_kl: float = 0.0,
        parameter_delta: float = 0.0,
        gradient_norm: float = 0.0,
        end_action_ratio: float = 0.0,
        attack_action_ratio: float = 0.0,
        attach_action_ratio: float = 0.0,
        play_action_ratio: float = 0.0,
        ability_action_ratio: float = 0.0,
        retreat_action_ratio: float = 0.0,
        checkpoint_loaded: str = "",
        checkpoint_epoch: int = 0,
    ) -> None:
        """Logs comprehensive epoch statistics with immediate disk synchronization."""
        row = [
            epoch, win_rate, wr_first, wr_second, avg_loss, val_loss, pol_loss, avg_reward,
            rc.get("prize_taken", 0.0) / rc_div, rc.get("prize_lost", 0.0) / rc_div,
            rc.get("kos", 0.0) / rc_div, rc.get("own_kos", 0.0) / rc_div,
            rc.get("energy", 0.0) / rc_div, rc.get("bench", 0.0) / rc_div,
            rc.get("deckout", 0.0) / rc_div, rc.get("terminal", 0.0) / rc_div,
            rc.get("stall", 0.0) / rc_div, rc.get("no_energy", 0.0) / rc_div, rc.get("strategic", 0.0) / rc_div,
            rc.get("r_knockout", 0.0) / rc_div, rc.get("r_attack_ready", 0.0) / rc_div, rc.get("r_backup_ready", 0.0) / rc_div,
            rc.get("r_bench_setup", 0.0) / rc_div, rc.get("r_evolution_progress", 0.0) / rc_div,
            rc.get("r_stadium_value", 0.0) / rc_div, rc.get("r_retreat_eff", 0.0) / rc_div,
            rc.get("r_damage_eff", 0.0) / rc_div, rc.get("r_lethal_detection", 0.0) / rc_div,
            rc.get("r_supporter_eff", 0.0) / rc_div, rc.get("r_supporter_opp_cost", 0.0) / rc_div,
            rc.get("r_hand_congestion", 0.0) / rc_div, rc.get("r_deck_preservation", 0.0) / rc_div,
            rc.get("r_missed_attack", 0.0) / rc_div, rc.get("r_donk_prevention", 0.0) / rc_div,
            rc.get("r_action_conv", 0.0) / rc_div,
            rc.get("r_search_quality", 0.0) / rc_div, rc.get("r_search_tempo", 0.0) / rc_div,
            avg_gl, avg_entropy, avg_diversity, avg_exp_var, avg_return, avg_advantage, avg_val_pred,
            reference_kl, parameter_delta, gradient_norm, end_action_ratio, attack_action_ratio, attach_action_ratio, play_action_ratio, ability_action_ratio, retreat_action_ratio,
            checkpoint_loaded, checkpoint_epoch
        ]
        self._write_and_flush(self.metrics_path, row)

    def log_action_distribution(self, epoch: int, action_counts: Dict[str, int]) -> None:
        """Logs action type breakdown for the epoch."""
        row = [
            epoch,
            action_counts.get("attack", 0),
            action_counts.get("play", 0),
            action_counts.get("attach", 0),
            action_counts.get("evolve", 0),
            action_counts.get("ability", 0),
            action_counts.get("retreat", 0),
            action_counts.get("end", 0),
            action_counts.get("other", 0),
        ]
        self._write_and_flush(self.action_dist_path, row)

    def log_deck_matchup(self, epoch: int, deck_stats: Dict[str, Dict[str, Any]]) -> None:
        """Logs opponent win rate matchup statistics."""
        with open(self.deck_matchup_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for name, stats in deck_stats.items():
                writer.writerow([epoch, name, stats["wins"], stats["losses"], stats["draws"], f"{stats['win_rate']:.1f}"])
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

    def log_self_play_game(self, epoch: int, opp_name: str, result_label: str, turns: int, action_counts: Dict[str, int]) -> None:
        """Logs a single completed self-play game trajectory immediately."""
        row = [
            epoch,
            opp_name,
            result_label,
            turns,
            action_counts.get("attack", 0),
            action_counts.get("play", 0),
            action_counts.get("attach", 0),
            action_counts.get("evolve", 0),
            action_counts.get("ability", 0),
            action_counts.get("retreat", 0),
            action_counts.get("end", 0),
            action_counts.get("other", 0),
        ]
        self._write_and_flush(self.self_play_path, row)

    def log_expert_guidance(self, epoch: int, opp_name: str, expert_log: List[Dict[str, Any]]) -> None:
        """Logs expert guidance triggers immediately."""
        if not expert_log:
            return
        with open(self.expert_log_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for entry in expert_log:
                writer.writerow([
                    epoch,
                    opp_name,
                    entry.get("trigger", ""),
                    entry.get("action_type", ""),
                    f"{entry.get('bonus', 0.0):.4f}",
                    entry.get("result", ""),
                    entry.get("turn", 0),
                    entry.get("my_prizes", 6),
                    entry.get("opp_prizes", 6),
                ])
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

    def log_nn_vs_mcts(self, epoch: int, opp_name: str, result_label: str, total_steps: int, disagreement_steps: int) -> None:
        """Logs NN vs MCTS disagreement metrics for a single completed game."""
        disagreement_rate = (disagreement_steps / max(1, total_steps)) * 100.0
        row = [
            epoch,
            opp_name,
            result_label,
            total_steps,
            disagreement_steps,
            f"{disagreement_rate:.2f}%",
            total_steps - disagreement_steps,
        ]
        self._write_and_flush(self.nn_vs_mcts_path, row)
