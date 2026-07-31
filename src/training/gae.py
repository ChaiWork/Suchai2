import numpy as np


def compute_gae_advantages(player_samples, rewards, terminal_reward, gamma=0.99, lam=0.95):
    """
    Computes Generalized Advantage Estimation (GAE) returns and targets for a sequence of player steps.
    Applies running mean/std advantage normalization for stable PPO policy updates.

    Args:
        player_samples: List of (sample_obj, pre_metrics) tuples for one player.
        rewards: List of computed step rewards.
        terminal_reward: Final terminal win/loss outcome (+1.0, -1.0, 0.0).
        gamma: Discount factor (default 0.99).
        lam: GAE decay factor (default 0.95).

    Returns:
        processed_samples: List of (sample_obj, td_error) tuples with updated sample_obj.value.
    """
    n_steps = len(player_samples)
    if n_steps == 0:
        return []

    # MCTS value estimates and model predictions are natively bounded in [-1.0, +1.0]
    pred_values = [player_samples[s][0].pred_val for s in range(n_steps)]
    advantages = [0.0] * n_steps
    returns = [0.0] * n_steps
    gae = 0.0

    for step_idx in reversed(range(n_steps)):
        step_rew = rewards[step_idx]
        if step_idx == n_steps - 1:
            delta = (step_rew + terminal_reward) - pred_values[step_idx]
        else:
            delta = step_rew + gamma * pred_values[step_idx + 1] - pred_values[step_idx]
        gae = delta + gamma * lam * gae
        advantages[step_idx] = gae
        returns[step_idx] = gae + pred_values[step_idx]

    # Dynamic Advantage Normalization: Zero-mean, unit-variance scaling for PPO
    adv_array = np.array(advantages, dtype=np.float32)
    std = adv_array.std()
    if std > 1e-8:
        norm_advantages = (adv_array - adv_array.mean()) / (std + 1e-8)
    else:
        norm_advantages = adv_array

    processed_samples = []
    for step_idx in range(n_steps):
        sample_obj, _ = player_samples[step_idx]
        # Enforce strict [-1.0, +1.0] target bounds for Value Network
        sample_obj.value = float(max(-1.0, min(1.0, returns[step_idx])))
        # Use normalized advantage as td_error for PPO & priority weighting
        td_error = float(norm_advantages[step_idx])
        processed_samples.append((sample_obj, td_error))

    return processed_samples
