"""
Expert System Router for Pokémon TCG RL Agent.
Dynamically evaluates action prior bonuses based on ACTIVE_DECK and applies
training epoch confidence decay so neural network policies take over cleanly.
"""
from src.configs.active_deck import get_active_deck_name
from src.expert_system.base_expert import USE_EXPERT_GUIDANCE, EXPERT_WEIGHT
from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
from src.expert_system.lillie_expert import evaluate_lillie_expert_bonus
from src.expert_system.grimmsnarl_expert import evaluate_grimmsnarl_expert_bonus


def get_expert_bonus(obs, option, opponent_name: str = "", epoch: int = 0) -> tuple[float, str]:
    """
    Unified entry point for retrieving expert action prior bonus and trigger label.
    Includes epoch-based confidence decay so heuristics smoothly yield to NN policy.
    Returns: (weighted_bonus: float, trigger_name: str)
    """
    if not USE_EXPERT_GUIDANCE:
        return 0.0, "Disabled"

    # Confidence decay over 30 training epochs (from 1.0 down to 0.1)
    confidence = max(0.1, 1.0 - (epoch / 30.0))

    deck = get_active_deck_name()
    if deck == "GRIMMSNARL":
        bonus, triggered = evaluate_grimmsnarl_expert_bonus(obs, option, opponent_name)
    elif deck == "LILLIE":
        bonus, triggered = evaluate_lillie_expert_bonus(obs, option, opponent_name)
    else:  # Default MEWTWO
        bonus, triggered = evaluate_mewtwo_expert_bonus(obs, option, opponent_name)

    return bonus * EXPERT_WEIGHT * confidence, triggered
