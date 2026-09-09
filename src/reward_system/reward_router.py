"""
Reward Router for Pokémon TCG RL Agent.
Routes reward calculations to the Team Rocket Mewtwo ex + Battle Cage reward engine.
Exposes a granular multi-component reward dictionary for independent logging.
"""
try:
    from reward_system.base_reward import calculate_base_strategic_reward_components
    from reward_system.mewtwo_reward import calculate_mewtwo_strategic_reward
except ImportError:
    from src.reward_system.base_reward import calculate_base_strategic_reward_components
    from src.reward_system.mewtwo_reward import calculate_mewtwo_strategic_reward


def calculate_strategic_reward_components(
    pre: dict,
    post: dict,
    action_type: int,
    step_idx: int,
    went_second: bool,
    player_idx: int,
    opponent_name: str,
) -> dict:
    """
    Computes all granular strategic reward components for the state transition S -> S'.

    Returns a dict of named reward terms that sum to the total step reward,
    capped later by calculate_strategic_reward().
    """
    comp = calculate_base_strategic_reward_components(
        pre, post, action_type, step_idx, went_second, player_idx, opponent_name
    )
    r_deck = calculate_mewtwo_strategic_reward(
        pre, post, action_type, step_idx, went_second, player_idx, opponent_name
    )
    comp["r_deck_specific"] = r_deck
    return comp


def calculate_strategic_reward(
    pre: dict,
    post: dict,
    action_type: int,
    step_idx: int,
    went_second: bool,
    player_idx: int,
    opponent_name: str,
) -> float:
    """Unified scalar reward entry point (clips total to [-0.50, +0.50])."""
    comp = calculate_strategic_reward_components(
        pre, post, action_type, step_idx, went_second, player_idx, opponent_name
    )
    return max(-0.50, min(0.50, sum(comp.values())))

