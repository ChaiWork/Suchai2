"""
Reward Router for Pokémon TCG RL Agent.
Dynamically routes reward calculations to the active deck profile reward engine.
"""
from src.configs.active_deck import get_active_deck_name
from src.reward_system.base_reward import calculate_base_strategic_reward
from src.reward_system.mewtwo_reward import calculate_mewtwo_strategic_reward
from src.reward_system.lillie_reward import calculate_lillie_strategic_reward


def calculate_strategic_reward(pre: dict, post: dict, action_type: int, step_idx: int, went_second: bool, player_idx: int, opponent_name: str) -> float:
    """
    Unified entry point for computing r_strategic.
    Combines base generic rewards with active deck-specific rewards.
    """
    r_base = calculate_base_strategic_reward(pre, post, action_type, step_idx, went_second, player_idx, opponent_name)

    deck = get_active_deck_name()
    if deck == "LILLIE":
        r_deck = calculate_lillie_strategic_reward(pre, post, action_type, step_idx, went_second, player_idx, opponent_name)
    else:  # Default MEWTWO
        r_deck = calculate_mewtwo_strategic_reward(pre, post, action_type, step_idx, went_second, player_idx, opponent_name)

    return r_base + r_deck
