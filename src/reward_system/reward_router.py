"""
Reward Router for Pokémon TCG RL Agent.
Dynamically routes reward calculations to the active deck profile reward engine.
Exposes granular multi-component reward dictionary for independent CSV logging.
"""
from src.configs.active_deck import get_active_deck_name
from src.reward_system.base_reward import calculate_base_strategic_reward_components, calculate_base_strategic_reward
from src.reward_system.mewtwo_reward import calculate_mewtwo_strategic_reward
from src.reward_system.lillie_reward import calculate_lillie_strategic_reward
from src.reward_system.grimmsnarl_reward import calculate_grimmsnarl_strategic_reward


def calculate_strategic_reward_components(pre: dict, post: dict, action_type: int, step_idx: int, went_second: bool, player_idx: int, opponent_name: str) -> dict:
    """
    Computes all granular strategic reward components for state transition S -> S'.
    """
    comp = calculate_base_strategic_reward_components(pre, post, action_type, step_idx, went_second, player_idx, opponent_name)

    deck = get_active_deck_name()
    if deck == "GRIMMSNARL":
        r_deck = calculate_grimmsnarl_strategic_reward(pre, post, action_type, step_idx, went_second, player_idx, opponent_name)
    elif deck == "LILLIE":
        r_deck = calculate_lillie_strategic_reward(pre, post, action_type, step_idx, went_second, player_idx, opponent_name)
    else:  # Default MEWTWO
        r_deck = calculate_mewtwo_strategic_reward(pre, post, action_type, step_idx, went_second, player_idx, opponent_name)

    comp["r_deck_specific"] = r_deck
    return comp


def calculate_strategic_reward(pre: dict, post: dict, action_type: int, step_idx: int, went_second: bool, player_idx: int, opponent_name: str) -> float:
    """Unified entry point returning scalar total reward for backward compatibility."""
    comp = calculate_strategic_reward_components(pre, post, action_type, step_idx, went_second, player_idx, opponent_name)
    return max(-0.50, min(0.50, sum(comp.values())))
