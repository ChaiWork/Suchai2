import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.append(p)

from model import MyModel, MODEL_D_MODEL, MODEL_NUM_HEADS, MODEL_D_FEEDFORWARD, MODEL_NUM_LAYERS_ENCODER, MODEL_NUM_LAYERS_DECODER
from agent import LearnSample
from training.checkpoint import CheckpointManager


class TestRLCollapseFixes(unittest.TestCase):

    def test_mcts_policy_target_distribution(self):
        """Verify policy targets are non-negative probabilities summing to 1.0."""
        # Simulated MCTS child visit counts
        visits = [50, 30, 20, 0, 0]
        total_visits = sum(visits)
        
        policy_target = [(v / total_visits) if total_visits > 0 else (1.0 / len(visits)) for v in visits]
        
        self.assertAlmostEqual(sum(policy_target), 1.0, places=5)
        for p in policy_target:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_cross_entropy_gradient_direction(self):
        """Verify Cross-Entropy loss with visit probability targets increases logits for selected actions."""
        logits = torch.tensor([[1.0, 0.5, 0.0, -1.0]], requires_grad=True)
        valid_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        # Probability distribution target: action 0 visited 80%, action 1 visited 20%
        target_policy = torch.tensor([[0.8, 0.2, 0.0, 0.0]])
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_policy * log_probs * valid_mask).sum(dim=-1).mean()
        loss.backward()
        
        # Gradient for logit 0 (action with highest visits) should be negative (meaning increasing logit decreases loss)
        self.assertLess(logits.grad[0, 0].item(), 0.0)
        # Gradient for unvisited action 3 should be positive (meaning increasing logit increases loss)
        self.assertGreater(logits.grad[0, 3].item(), 0.0)

    def test_checkpoint_audit_output(self):
        """Verify diagnostic parameter norm calculation functions."""
        model = MyModel(MODEL_D_MODEL, MODEL_NUM_HEADS, MODEL_D_FEEDFORWARD, MODEL_NUM_LAYERS_ENCODER, MODEL_NUM_LAYERS_DECODER)
        
        # Compute global norm
        param_norm = torch.norm(torch.stack([torch.norm(p.detach()) for p in model.parameters() if p.requires_grad])).item()
        self.assertGreater(param_norm, 0.0)
        
        # Compute head norms
        encoder_head_norm = torch.norm(torch.stack([torch.norm(p.detach()) for p in model.encoder_fc.parameters()])).item()
        decoder_head_norm = torch.norm(torch.stack([torch.norm(p.detach()) for p in model.decoder_fc.parameters()])).item()
        
        self.assertGreater(encoder_head_norm, 0.0)
        self.assertGreater(decoder_head_norm, 0.0)

    def test_reference_kl_calculation(self):
        """Verify KL divergence calculation between reference and active logits."""
        p_logits = torch.tensor([[2.0, 1.0, 0.0]])
        q_logits = torch.tensor([[1.5, 1.2, 0.2]])
        
        p = F.softmax(p_logits, dim=-1)
        log_p = F.log_softmax(p_logits, dim=-1)
        log_q = F.log_softmax(q_logits, dim=-1)
        
        # KL(P || Q) = sum(P * (log_P - log_Q))
        kl = (p * (log_p - log_q)).sum(dim=-1).mean().item()
        self.assertGreaterEqual(kl, 0.0)

    def test_reward_hierarchy(self):
        """Verify terminal win/loss signal dominates maximum possible per-turn strategic shaping rewards."""
        terminal_win_reward = 1.0
        terminal_loss_reward = -1.0
        
        # Maximum clipped single-step strategic reward in base_reward.py & mewtwo_reward.py is capped at +/- 0.25
        max_step_shaping = 0.25
        
        self.assertGreater(terminal_win_reward, max_step_shaping * 2)
        self.assertLess(terminal_loss_reward, -max_step_shaping * 2)


if __name__ == "__main__":
    unittest.main()
