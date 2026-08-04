"""
Generic Domain-Agnostic RL Reward Engine (Phase 4 Sparse Final Form).
Calculates pure sparse outcome rewards: prize deltas, KO rewards, bench guards,
deckout danger penalties, and terminal win/loss signals applicable to any deck archetype.
"""
from cg.api import CardType, OptionType


try:
    from src.expert_system.base_expert import expert_scale
except ImportError:
    try:
        from expert_system.base_expert import expert_scale
    except ImportError:
        def expert_scale(epoch=1):
            return 1.0


def calculate_base_strategic_reward(pre: dict, post: dict, action_type: int, step_idx: int, went_second: bool, player_idx: int, opponent_name: str) -> float:
    """
    Computes pure deck-agnostic r_strategic value for state transitions.
    Intermediate shaping signals (prize delta, KO, board setup, card conversion,
    supporter opportunity cost) decay smoothly over training epochs but retain a 
    minimum baseline floor (alpha_min = 0.20) so intermediate feedback never drops to zero.
    """
    r_strategic = 0.0

    # 1. Prize Delta Reward: +(prizes_taken) - (prizes_lost)
    prizes_taken = pre.get("my_prizes", 6) - post.get("my_prizes", 6)
    prizes_lost  = pre.get("opp_prizes", 6) - post.get("opp_prizes", 6)

    if prizes_taken > 0:
        r_strategic += prizes_taken * 0.25
    if prizes_lost > 0:
        r_strategic -= prizes_lost * 0.25

    # 2. KO Reward / Own KO Loss Penalty
    opp_pk_lost = pre.get("opp_pokemon", 0) - post.get("opp_pokemon", 0)
    my_pk_lost  = pre.get("my_pokemon", 0) - post.get("my_pokemon", 0)

    if opp_pk_lost > 0:
        r_strategic += opp_pk_lost * 0.15
    if my_pk_lost > 0:
        r_strategic -= my_pk_lost * 0.15

    # 3. Board Evolution & Setup Readiness Delta
    # Rewards developing bench HP, useful attached energy (capped at attack requirement), and evolution stages
    pre_en = pre.get("useful_energy", pre.get("total_energy", 0))
    post_en = post.get("useful_energy", post.get("total_energy", 0))
    pre_power = pre.get("my_hp", 0) + pre_en * 15 + pre.get("my_pokemon", 0) * 20
    post_power = post.get("my_hp", 0) + post_en * 15 + post.get("my_pokemon", 0) * 20
    power_delta = post_power - pre_power
    if power_delta > 0:
        r_strategic += min(0.20, power_delta / 250.0)

    # 4. Action Conversion & Hand Congestion Reward
    cards_played = post.get("cards_played_this_turn", 0) - pre.get("cards_played_this_turn", 0)
    if cards_played > 0 and action_type in (7, 8):  # PLAY or ATTACH
        r_strategic += min(0.15, cards_played * 0.05)

    # Hand Congestion Penalty (drawn lots of cards but holding many unplayed)
    hand_cnt_post = post.get("hand_count", 0)
    if hand_cnt_post > 10 and action_type == 7:  # PLAY draw engine
        r_strategic -= 0.05

    # 5. Supporter Opportunity Cost Penalty (ending turn with playable Supporter in hand uncast)
    supporter_played = post.get("supporter_played", False)
    has_supporter_in_hand = post.get("has_supporter_in_hand", False)
    if action_type == 0 and (not supporter_played) and has_supporter_in_hand:  # Action 0 = END_TURN
        r_strategic -= 0.15

    # 6. Prize Proximity / Damage Progress Reward
    opp_hp_loss = pre.get("opp_hp", 0) - post.get("opp_hp", 0)
    if opp_hp_loss > 0:
        r_strategic += min(0.20, (opp_hp_loss / 200.0) * 0.15)

    # 7. Bench Size Protection Guard across early turns (prevents Bench Wipe loss)
    bench_size_curr = post.get("bench_size", 0)
    turn_curr = post.get("turn", 0)
    if bench_size_curr == 0 and turn_curr <= 5:
        r_strategic -= 0.40

    # 8. Deck-Out Danger Warning Penalty (Escalating as deck count drops <= 5)
    deck_cnt_post = post.get("deck_count", 60)
    if deck_cnt_post <= 5:
        r_strategic -= 0.15 * ((6.0 - deck_cnt_post) / 5.0)

    # 9. Generic Board Control Improvement Delta (works for ANY stadium or bench protection card)
    board_ctrl_pre = pre.get("board_control", 0.0)
    board_ctrl_post = post.get("board_control", 0.0)
    if board_ctrl_post > board_ctrl_pre:
        r_strategic += min(0.08, (board_ctrl_post - board_ctrl_pre) * 0.05)
    elif action_type == 7 and post.get("stadium_played", False):
        r_strategic += 0.05  # Generic stadium deployment bonus

    # 10. Conditional Pass Penalty (small -0.05 penalty if powered attack was available but passed)
    act_ready = pre.get("active_attack_ready", False) or pre.get("my_active_energy", 0) >= 3
    if action_type == 0 and act_ready and opp_hp_loss == 0:  # Action 0 = END_TURN
        r_strategic -= 0.05

    # Apply curriculum decay with a non-zero baseline floor (alpha_min = 0.20)
    epoch_val = pre.get("epoch", 1)
    decay_factor = max(0.20, expert_scale(epoch_val))
    r_strategic *= decay_factor

    return max(-0.50, min(0.50, r_strategic))


def calculate_base_energy_reward(
    action_type: int,
    attached_card_id: int,
    target_card_id: int,
    target_current_energy: int,
    bench_ids: list,
    bench_energies: list,
    active_id: int,
    active_energy: int,
    active_hp: int,
    hand_ids: list,
    my_prizes: int,
    opp_prizes: int,
) -> float:
    """
    Generic deck-agnostic energy attachment reward.
    Called from any deck-specific reward module for action_type == 8.
    Bounded to [-0.25, +0.25].
    """
    if action_type != 8 or attached_card_id < 0 or target_card_id < 0:
        return 0.0
    try:
        try:
            from src.expert_system.energy_evaluator import compute_generic_energy_reward
        except ImportError:
            from expert_system.energy_evaluator import compute_generic_energy_reward
        
        is_active = (target_card_id == active_id)
        energy_added = 2 if attached_card_id in (15, 19) else 1
        energy_after = target_current_energy + energy_added

        # Reconstruct post bench energies for backup bonus calculation
        bench_energies_after = list(bench_energies)
        if not is_active:
            for i, b_id in enumerate(bench_ids):
                if b_id == target_card_id:
                    bench_energies_after[i] = bench_energies[i] + energy_added
                    break

        return compute_generic_energy_reward(
            target_card_id=target_card_id,
            energy_before=target_current_energy,
            energy_after=energy_after,
            attached_card_id=attached_card_id,
            is_active_target=is_active,
            bench_ids=bench_ids,
            bench_energies_before=bench_energies,
            bench_energies_after=bench_energies_after,
            active_id=active_id,
            active_energy_before=active_energy,
        )
    except Exception:
        return 0.0

