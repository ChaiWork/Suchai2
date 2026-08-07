import numpy as np


def compute_gae_advantages(player_samples, rewards, terminal_reward, gamma=0.995, lam=0.98):
    """
    Computes Generalized Advantage Estimation (GAE) returns and targets for a sequence of player steps.
    Applies running mean/std advantage normalization for stable PPO policy updates.
    """
    # Filter out any None sample objects upfront
    valid_pairs = [(samp, rew) for (samp, _), rew in zip(player_samples, rewards) if samp is not None]
    if not valid_pairs:
        return []

    valid_samples, valid_rewards = zip(*valid_pairs)
    n_steps = len(valid_samples)

    pred_values = [
        getattr(valid_samples[s], "pred_val", getattr(valid_samples[s], "value", 0.0))
        for s in range(n_steps)
    ]
    advantages = [0.0] * n_steps
    returns = [0.0] * n_steps
    gae = 0.0

    for step_idx in reversed(range(n_steps)):
        step_rew = valid_rewards[step_idx]
        if step_idx == n_steps - 1:
            delta = (step_rew + terminal_reward) - pred_values[step_idx]
        else:
            delta = step_rew + gamma * pred_values[step_idx + 1] - pred_values[step_idx]
        gae = delta + gamma * lam * gae
        advantages[step_idx] = gae
        returns[step_idx] = gae + pred_values[step_idx]

    adv_array = np.array(advantages, dtype=np.float32)
    std = adv_array.std()
    if std > 1e-8:
        norm_advantages = (adv_array - adv_array.mean()) / (std + 1e-8)
    else:
        norm_advantages = adv_array

    processed_samples = []
    for step_idx in range(n_steps):
        sample_obj = valid_samples[step_idx]
        sample_obj.value = float(max(-1.0, min(1.0, returns[step_idx])))
        td_error = float(norm_advantages[step_idx])
        processed_samples.append((sample_obj, td_error))

    return processed_samples
