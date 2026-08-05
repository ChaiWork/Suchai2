"""
Strategic and Step Reward Engine Facade for Pokémon TCG RL Agent.
Delegates strategic reward calculation to reward_router based on ACTIVE_DECK.
"""
from src.reward_system.reward_router import calculate_strategic_reward, calculate_strategic_reward_components


__all__ = ["calculate_strategic_reward", "calculate_strategic_reward_components"]
