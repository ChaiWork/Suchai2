import unittest
import torch
import torch.nn.functional as F
import os
import sys
import shutil

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.append(p)

from model import MyModel, SparseVector, LearnInput, MODEL_D_MODEL, MODEL_NUM_HEADS, MODEL_D_FEEDFORWARD, MODEL_NUM_LAYERS_ENCODER, MODEL_NUM_LAYERS_DECODER
from training.logger import MetricsLogger
from training.checkpoint import CheckpointManager


class TestTrainingSmoke(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.join(root_dir, "out", "test_smoke_run")
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    def test_model_forward_backward_loss(self):
        """Smoke test for model forward, cross entropy loss with visit count targets, and backward step."""
        model = MyModel(MODEL_D_MODEL, MODEL_NUM_HEADS, MODEL_D_FEEDFORWARD, MODEL_NUM_LAYERS_ENCODER, MODEL_NUM_LAYERS_DECODER)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        batch_size = 2
        input_enc = LearnInput()
        input_dec = LearnInput()
        mask = []
        label_dec = []

        for _ in range(batch_size):
            sv_enc = SparseVector()
            for _ in range(24):
                sv_enc.word_start()
                sv_enc.add(1, 1.0)
            
            sv_dec = SparseVector()
            for a in range(256):
                sv_dec.word_start()
                sv_dec.add(10 + (a % 10), 1.0)
            
            input_enc.add(sv_enc)
            input_dec.add(sv_dec)

            for _ in range(5):
                mask.append(1.0)
                label_dec.append(0.2)
            for _ in range(256 - 5):
                mask.append(0.0)
                label_dec.append(0.0)

        input_enc_idx = torch.tensor(input_enc.index, dtype=torch.int32)
        input_enc_val = torch.tensor(input_enc.value, dtype=torch.float32)
        input_enc_off = torch.tensor(input_enc.offset, dtype=torch.int32)
        input_dec_idx = torch.tensor(input_dec.index, dtype=torch.int32)
        input_dec_val = torch.tensor(input_dec.value, dtype=torch.float32)
        input_dec_off = torch.tensor(input_dec.offset, dtype=torch.int32)

        out_enc, out_dec = model(input_enc_idx, input_enc_val, input_enc_off, input_dec_idx, input_dec_val, input_dec_off)
        
        self.assertEqual(out_enc.shape[0], batch_size)

        # Reshape logits to (batch_size, 256) and slice first 5 actions
        logits = out_dec.view(batch_size, 256)[:, :5]
        # Dummy MCTS visit count target distribution (non-negative, sums to 1.0)
        target_policy = torch.tensor([[0.5, 0.3, 0.2, 0.0, 0.0] for _ in range(batch_size)])
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss_ce = -(target_policy * log_probs).sum(dim=-1).mean()
        
        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(loss_ce))
        self.assertGreater(loss_ce.item(), 0.0)

    def test_metrics_logger_integration(self):
        """Smoke test for MetricsLogger with extended audit columns."""
        logger = MetricsLogger(self.test_dir)
        rc = {"prize_taken": 1.0, "kos": 1.0, "terminal": 1.0}
        
        logger.log_epoch_metrics(
            epoch=1, win_rate=90.0, wr_first=90.0, wr_second=90.0, avg_loss=0.5, val_loss=0.2, pol_loss=0.3, avg_reward=0.4,
            rc=rc, rc_div=1.0, avg_gl=15.0, avg_entropy=0.15, avg_diversity=0.8, avg_exp_var=0.5, avg_return=0.3, avg_advantage=0.2, avg_val_pred=0.3,
            reference_kl=0.02, parameter_delta=0.01, gradient_norm=0.5, end_action_ratio=0.1, attack_action_ratio=0.3, attach_action_ratio=0.3,
            play_action_ratio=0.2, ability_action_ratio=0.05, retreat_action_ratio=0.05, checkpoint_loaded="model.pth", checkpoint_epoch=1
        )
        
        self.assertTrue(os.path.exists(logger.metrics_path))
        with open(logger.metrics_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Header + 1 row
            self.assertIn("reference_kl", lines[0])
            self.assertIn("end_action_ratio", lines[0])


if __name__ == "__main__":
    unittest.main()
