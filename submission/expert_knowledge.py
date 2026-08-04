try:
    from expert_system.base_expert import USE_EXPERT_GUIDANCE, EXPERT_WEIGHT
    from expert_system.expert_router import get_expert_bonus
except ModuleNotFoundError:
    try:
        from src.expert_system.base_expert import USE_EXPERT_GUIDANCE, EXPERT_WEIGHT
        from src.expert_system.expert_router import get_expert_bonus
    except ModuleNotFoundError:
        USE_EXPERT_GUIDANCE = False
        EXPERT_WEIGHT = 0.0
        def get_expert_bonus(obs, option, opponent_name="GENERIC", epoch=1):
            return 0.0, ""

__all__ = ["USE_EXPERT_GUIDANCE", "EXPERT_WEIGHT", "get_expert_bonus"]
