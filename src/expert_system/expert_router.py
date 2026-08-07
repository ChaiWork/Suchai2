"""
Expert System Router for Pokémon TCG RL Agent.
Evaluates action prior bonuses for Team Rocket Mewtwo ex + Battle Cage and applies
training epoch confidence decay so neural network policies take over cleanly.
"""
from src.expert_system.base_expert import USE_EXPERT_GUIDANCE, EXPERT_WEIGHT
from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus


def get_expert_bonus(obs, option, opponent_name: str = "", epoch: int = 0) -> tuple[float, str]:
    """
    Unified entry point for retrieving expert action prior bonus and trigger label.
    Includes epoch-based confidence decay so heuristics smoothly yield to NN policy.
    Returns: (weighted_bonus: float, trigger_name: str)
    """
    if not USE_EXPERT_GUIDANCE:
        return 0.0, "Disabled"

    try:
        from src.expert_system.base_expert import expert_scale
    except ImportError:
        from expert_system.base_expert import expert_scale

    # Use stepped plateau schedule with strong guidance floor (0.60)
    # Prevents expert guidance starvation during high-epoch fine-tuning (Epoch 30+)
    confidence = max(0.60, expert_scale(epoch))

    bonus, triggered = evaluate_mewtwo_expert_bonus(obs, option, opponent_name)

    return bonus * EXPERT_WEIGHT * confidence, triggered

