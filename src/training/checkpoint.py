"""
Production-Grade Atomic Checkpointing System for Pokémon TCG RL Agent.
Guarantees zero file corruption via atomic temporary file replacement (os.replace).
Serializes complete state (model, optimizer, scheduler, epoch, best win rate, patience, elo)
to support crash recovery, reproducible experiments, and seamless training resume.
"""

import os
import torch
import shutil
from typing import Dict, Any, Optional, Tuple


def _atomic_torch_save(obj: Any, fpath: str) -> None:
    """
    Saves PyTorch object atomically by writing to a temporary file first,
    flushing file buffers to disk, and performing an atomic rename (os.replace).
    Prevents 0-byte or corrupted .pth files if training is interrupted.
    """
    directory = os.path.dirname(os.path.abspath(fpath))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = fpath + ".tmp"

    try:
        torch.save(obj, tmp_path)
        # Flush OS buffers
        with open(tmp_path, "a+b") as f:
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        # Atomic replace
        os.replace(tmp_path, fpath)
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise e


class CheckpointManager:
    """
    Manages complete training state checkpointing:
    - latest_model.pth (updated every epoch)
    - model.pth (backward compatibility alias)
    - best_model.pth (updated ONLY when evaluation win-rate achieves a new peak)
    - run_dir/checkpoints/model_epoch_{counter:03d}.pth (per-epoch history)
    """

    def __init__(self, run_dir: str, deck_name: str = "MEWTWO"):
        self.run_dir = run_dir
        self.deck_name = deck_name.lower()
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def save_epoch_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        win_rate: float,
        best_win_rate: float,
        patience_counter: int,
        active_elo: float,
        model_lock: Optional[Any] = None,
    ) -> Dict[str, str]:
        """
        Saves latest_model.pth, model.pth, and per-epoch formatted checkpoint atomically.
        Returns dictionary of saved file paths.
        """
        if model_lock is not None:
            with model_lock:
                state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        checkpoint = {
            "epoch": epoch,
            "state_dict": state_dict,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "win_rate": win_rate,
            "best_win_rate": best_win_rate,
            "patience_counter": patience_counter,
            "active_elo": active_elo,
            "deck_name": self.deck_name,
        }

        # 1. Per-epoch history checkpoint: run_dir/checkpoints/model_epoch_001.pth
        epoch_filename = f"model_epoch_{epoch:03d}.pth"
        epoch_path = os.path.join(self.checkpoints_dir, epoch_filename)
        _atomic_torch_save(checkpoint, epoch_path)

        # 2. Run directory latest_model.pth and model.pth
        run_latest = os.path.join(self.run_dir, "latest_model.pth")
        run_model = os.path.join(self.run_dir, "model.pth")
        _atomic_torch_save(checkpoint, run_latest)
        _atomic_torch_save(checkpoint, run_model)

        # 3. Root workspace aliases for main.py / package_submission compatibility
        _atomic_torch_save(checkpoint, "latest_model.pth")
        _atomic_torch_save(checkpoint, "model.pth")
        _atomic_torch_save(checkpoint, f"model_{self.deck_name}.pth")

        return {
            "epoch_path": epoch_path,
            "latest_path": run_latest,
            "root_latest": "latest_model.pth",
        }

    def save_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        win_rate: float,
        active_elo: float,
        model_lock: Optional[Any] = None,
    ) -> Dict[str, str]:
        """
        Saves best_model.pth atomically when evaluation win rate improves over prior best.
        """
        if model_lock is not None:
            with model_lock:
                state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        checkpoint = {
            "epoch": epoch,
            "state_dict": state_dict,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "win_rate": win_rate,
            "best_win_rate": win_rate,
            "active_elo": active_elo,
            "deck_name": self.deck_name,
        }

        run_best = os.path.join(self.run_dir, "best_model.pth")
        _atomic_torch_save(checkpoint, run_best)
        _atomic_torch_save(checkpoint, "best_model.pth")

        return {
            "run_best": run_best,
            "root_best": "best_model.pth",
        }

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, float, int]:
        """
        Loads a checkpoint robustly and restores model, optimizer, scheduler, epoch, best_win_rate, patience.
        Returns: (start_epoch, best_win_rate, patience_counter)
        """
        if not os.path.exists(checkpoint_path):
            return 0, -1.0, 0

        map_loc = device if device is not None else "cpu"
        print(f"Loading existing checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_loc, weights_only=True)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
                start_epoch = checkpoint.get("epoch", 0)
                best_win_rate = checkpoint.get("best_win_rate", -1.0)
                patience_counter = checkpoint.get("patience_counter", 0)

                if optimizer is not None and "optimizer_state" in checkpoint and checkpoint["optimizer_state"]:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state"])
                        print("  -> Restored optimizer momentum state")
                    except Exception as e:
                        print(f"  -> Note: Optimizer state skipped due to layout mismatch ({e})")

                if scheduler is not None and "scheduler_state" in checkpoint and checkpoint["scheduler_state"]:
                    try:
                        scheduler.load_state_dict(checkpoint["scheduler_state"])
                        print("  -> Restored learning rate scheduler state")
                    except Exception as e:
                        print(f"  -> Note: Scheduler state skipped ({e})")

                print(f"  -> Successfully resumed from epoch {start_epoch} (Best Win Rate: {best_win_rate:.1f}%)")
                return start_epoch, best_win_rate, patience_counter
            else:
                model.load_state_dict(checkpoint, strict=False)
                print(f"  -> Loaded legacy state_dict weights from {checkpoint_path}")
                return 0, -1.0, 0
        except Exception as e:
            print(f"  -> Warning: Could not load checkpoint {checkpoint_path} ({e}). Starting fresh.")
            return 0, -1.0, 0
