"""
Generic Domain-Agnostic RL Reward Engine (Granular Multi-Component Form).
Calculates orthogonal, state-transition-driven rewards for every strategic concept:
knockouts, board setup, damage efficiency, attack readiness, search quality, supporter timing, and rule guards.
Matches worker.py dictionary keys perfectly.
"""
try:
    from src.expert_system.base_expert import expert_scale
except ImportError:
    try:
        from expert_system.base_expert import expert_scale
    except ImportError:
        def expert_scale(epoch=1):
            return 1.0

SUPPORTER_IDS = frozenset({1216, 1218, 1219, 1220, 1227})
SEARCH_CARD_IDS = frozenset({1086, 1094, 1106, 1121, 1134, 1152})


def calculate_base_strategic_reward_components(pre: dict, post: dict, action_type: int, step_idx: int, went_second: bool, player_idx: int, opponent_name: str) -> dict:
    """
    Computes pure deck-agnostic granular reward components for state transitions S -> S'.
    Returns a dictionary mapping each strategic concept to its independent reward float.
    Uses exact key names matching worker.py:
      - active_hp / opp_active_hp
      - active_energy
      - hand_size / hand_ids
      - deck_size / opp_deck_size
      - bench_size / bench_ids
      - action_type / played_card_id
    """
    components = {
        "r_knockout": 0.0,
        "r_attack_ready": 0.0,
        "r_backup_ready": 0.0,
        "r_bench_setup": 0.0,
        "r_evolution_progress": 0.0,
        "r_stadium_value": 0.0,
        "r_retreat_eff": 0.0,
        "r_damage_eff": 0.0,
        "r_lethal_detection": 0.0,
        "r_supporter_eff": 0.0,
        "r_supporter_opp_cost": 0.0,
        "r_hand_congestion": 0.0,
        "r_deck_preservation": 0.0,
        "r_missed_attack": 0.0,
        "r_donk_prevention": 0.0,
        "r_action_conv": 0.0,
        "r_search_quality": 0.0,
        "r_search_tempo": 0.0,
    }

    # 1. Knockout Execution & Damage Efficiency & Lethal Detection
    pre_opp_pk = pre.get("opp_pokemon", 0)
    post_opp_pk = post.get("opp_pokemon", 0)
    if pre_opp_pk > post_opp_pk and post_opp_pk >= 0 and pre_opp_pk > 0:
        components["r_knockout"] = (pre_opp_pk - post_opp_pk) * 0.15

    pre_opp_hp = pre.get("opp_active_hp", 0)
    post_opp_hp = post.get("opp_active_hp", 0)
    if pre_opp_hp > post_opp_hp and pre_opp_hp > 0:
        opp_hp_loss = pre_opp_hp - post_opp_hp
        components["r_damage_eff"] = min(0.10, (opp_hp_loss / 200.0) * 0.10)
        if post_opp_hp <= 0:
            components["r_lethal_detection"] = 0.12

    # 2. Board Power & Attack Readiness
    pre_act_en = pre.get("active_energy", 0)
    post_act_en = post.get("active_energy", 0)
    if post_act_en >= 2 and pre_act_en < 2:
        components["r_attack_ready"] = 0.10

    pre_bench_size = pre.get("bench_size", 0)
    post_bench_size = post.get("bench_size", 0)
    pre_total_en = pre.get("energy", 0)
    post_total_en = post.get("energy", 0)
    bench_en = post_total_en - post_act_en
    if post_bench_size > 0 and bench_en >= 2 and post_total_en > pre_total_en:
        components["r_backup_ready"] = 0.08

    # 3. Early Bench Setup & Donk Guard
    turn_curr = post.get("turn", 1)
    if post_bench_size > pre_bench_size and turn_curr <= 3:
        components["r_bench_setup"] = min(0.08, 0.04 * (post_bench_size - pre_bench_size))

    if post_bench_size == 0 and turn_curr <= 2:
        components["r_donk_prevention"] = -0.20

    # 4. Evolution Progress
    if action_type == 9:  # EVOLVE
        components["r_evolution_progress"] = 0.08

    # 5. Supporter Play & Opportunity Cost
    played_card_id = pre.get("played_card_id", -1)
    if action_type == 7 and played_card_id in SUPPORTER_IDS:
        components["r_supporter_eff"] = 0.05

    post_hand_ids = post.get("hand_ids", [])
    has_supporter_in_hand = any(cid in SUPPORTER_IDS for cid in post_hand_ids)
    if action_type in (0, 14) and (played_card_id not in SUPPORTER_IDS) and has_supporter_in_hand:  # END_TURN
        components["r_supporter_opp_cost"] = -0.08

    # 6. Search Quality & Tempo
    if (action_type == 7 and played_card_id in SEARCH_CARD_IDS) or action_type == 3:
        components["r_search_quality"] = 0.06
        components["r_search_tempo"] = 0.05

    # 7. Action Conversion & Hand Congestion
    if action_type in (7, 8):  # PLAY or ATTACH
        components["r_action_conv"] = 0.04

    post_hand_size = post.get("hand_size", 0)
    if post_hand_size > 10 and action_type == 7:
        components["r_hand_congestion"] = -0.05

    # 8. Pure AlphaZero State-Transition Retreat Delta V(S -> S')
    if action_type == 12:  # RETREAT
        pre_act_en = pre.get("active_energy", 0)
        pre_min_req = pre.get("active_min_req_energy", 1)
        pre_readiness = min(1.0, pre_act_en / max(1, pre_min_req))
        
        post_act_en = post.get("active_energy", 0)
        post_min_req = post.get("active_min_req_energy", 1)
        post_readiness = min(1.0, post_act_en / max(1, post_min_req))
        
        pre_survival = pre.get("active_hp", 100) / max(1, pre.get("active_max_hp", 100))
        post_survival = post.get("active_hp", 100) / max(1, post.get("active_max_hp", 100))
        
        readiness_delta = post_readiness - pre_readiness
        survival_delta = post_survival - pre_survival
        
        board_delta = 0.60 * readiness_delta + 0.40 * survival_delta
        components["r_retreat_eff"] = max(-0.10, min(0.10, board_delta))

    # 9. Deck-Out Danger Warning
    deck_cnt_post = post.get("deck_size", 60)
    if deck_cnt_post <= 5:
        components["r_deck_preservation"] = -0.05 * ((6.0 - deck_cnt_post) / 5.0)

    # 10. Stadium Deployment
    if action_type == 7 and played_card_id in (1180, 1257, 1264):  # Prism Tower, TR Factory, Battle Cage
        components["r_stadium_value"] = 0.06

    # 11. Missed Attack Penalty (powered attacker passed without attacking)
    act_ready = (pre_act_en >= 2)
    if action_type in (0, 14) and act_ready and (pre_opp_hp - post_opp_hp <= 0):  # END_TURN
        components["r_missed_attack"] = -0.10

    # Apply curriculum decay
    epoch_val = pre.get("epoch", 1)
    decay_factor = max(0.20, expert_scale(epoch_val))
    for k in components:
        components[k] *= decay_factor

    return components


def calculate_base_strategic_reward(pre: dict, post: dict, action_type: int, step_idx: int, went_second: bool, player_idx: int, opponent_name: str) -> float:
    """Backward-compatible total scalar reward wrapper."""
    comp = calculate_base_strategic_reward_components(pre, post, action_type, step_idx, went_second, player_idx, opponent_name)
    return max(-0.50, min(0.50, sum(comp.values())))


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
    """Generic energy reward wrapper delegating to energy_evaluator."""
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
