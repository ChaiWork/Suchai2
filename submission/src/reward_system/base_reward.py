"""
Generic Domain-Agnostic RL Reward Engine.
Calculates terminal, prize, bench guard, and anti-stall rewards applicable to any deck archetype.
"""
from cg.api import CardType, OptionType


def calculate_base_strategic_reward(pre: dict, post: dict, action_type: int, step_idx: int, went_second: bool, player_idx: int, opponent_name: str) -> float:
    """
    Computes base deck-agnostic r_strategic value for state transitions.
    """
    r_strategic = 0.0

    # 1. Deck-Out Danger Warning Penalty
    deck_cnt_post = post.get("deck_count", 60)
    if deck_cnt_post <= 6 and action_type == 7:  # PLAY
        r_strategic -= 0.35

    # 2. Bench Size Protection Guard across ALL turns (prevents Donk / Bench Wipe)
    bench_size_curr = post.get("bench_size", 0)
    turn_curr = post.get("turn", 0)
    if bench_size_curr == 0:
        r_strategic -= 0.60 if turn_curr <= 5 else 0.40
    elif bench_size_curr == 1:
        r_strategic -= 0.25 if turn_curr <= 4 else 0.10
    elif bench_size_curr >= 2:
        r_strategic += 0.20

    # 3. Opponent Prize / KO Pressure
    opp_pk_lost = pre.get("opp_pokemon", 0) - post.get("opp_pokemon", 0)
    if opp_pk_lost > 0:
        r_strategic += 0.15

    # 4. Multi-Prize Card Loss Penalty (Deck-Agnostic Guard)
    prizes_lost = pre.get("opp_prizes", 6) - post.get("opp_prizes", 6)
    if prizes_lost >= 2:
        r_strategic -= 0.30  # Heavy deck-agnostic penalty for losing a 2-prize / Rule Box Pokémon

    return r_strategic
