"""
Generic Domain-Agnostic RL Reward Engine (Granular Multi-Component Form).
Calculates orthogonal, state-transition-driven rewards for every strategic concept:
knockouts, board setup, damage efficiency, attack readiness, search quality, supporter timing, and rule guards.
Matches worker.py dictionary keys perfectly.

  NOTE: No expert_scale decay is applied here. These are pure AlphaZero V(S')-V(S)
  signals and must remain at full strength throughout all training epochs.
  expert_scale decay belongs only in mewtwo_reward.py (deck-specific safety layer).
"""

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
        "r_slow_setup": 0.0,
        "r_donk_prevention": 0.0,
        "r_action_conv": 0.0,
        "r_search_quality": 0.0,
        "r_search_tempo": 0.0,
    }

    # 1. Knockout Execution & Damage Efficiency & Lethal Detection
    pre_opp_pk = pre.get("opp_pokemon", 0)
    post_opp_pk = post.get("opp_pokemon", 0)
    if pre_opp_pk > post_opp_pk and post_opp_pk >= 0 and pre_opp_pk > 0:
        components["r_knockout"] = (pre_opp_pk - post_opp_pk) * 0.25

    pre_opp_hp = pre.get("opp_active_hp", 0)
    post_opp_hp = post.get("opp_active_hp", 0)
    if pre_opp_hp > post_opp_hp and pre_opp_hp > 0:
        opp_hp_loss = pre_opp_hp - post_opp_hp
        components["r_damage_eff"] = min(0.15, (opp_hp_loss / 200.0) * 0.15)
        if post_opp_hp <= 0:
            components["r_lethal_detection"] = 0.20

    # 2. Board Power & Attack Readiness
    # Use API-driven minimum attack cost so r_attack_ready fires at the correct threshold
    # for every Pokémon (Mimikyu=1, Spidops/Articuno=2, Mewtwo ex=3) rather than a
    # hardcoded 2 that would treat a 1-cost Pokémon the same as a 4-cost one.
    pre_act_en = pre.get("active_energy", 0)
    post_act_en = post.get("active_energy", 0)
    _active_id = post.get("active_id", -1)
    try:
        try:
            from src.expert_system.energy_evaluator import get_min_required_energy
        except ImportError:
            from expert_system.energy_evaluator import get_min_required_energy
        _attack_threshold = get_min_required_energy(_active_id)
    except Exception:
        _attack_threshold = 2  # Safe fallback if cg.api cache unavailable
    if post_act_en >= _attack_threshold and pre_act_en < _attack_threshold:
        components["r_attack_ready"] = 0.20
    elif action_type == 8 and pre_act_en >= _attack_threshold and post_act_en > pre_act_en:
        # Active is ALREADY powered — harshly penalize over-attaching energy to active (-0.25)
        components["r_attack_ready"] = -0.25

    pre_bench_size = pre.get("bench_size", 0)
    post_bench_size = post.get("bench_size", 0)
    pre_bench_en = sum(pre.get("bench_energies", [])) if "bench_energies" in pre else max(0, pre.get("energy", 0) - pre_act_en)
    post_bench_en = sum(post.get("bench_energies", [])) if "bench_energies" in post else max(0, post.get("energy", 0) - post_act_en)

    # Symmetrical bench energy preparation reward: award +0.15 whenever energy on bench increases
    if post_bench_en > pre_bench_en:
        components["r_backup_ready"] = 0.15

    # 3. Early Bench Setup & Donk Guard
    turn_curr = post.get("turn", 1)
    if post_bench_size > pre_bench_size and turn_curr <= 3:
        components["r_bench_setup"] = min(0.10, 0.05 * (post_bench_size - pre_bench_size))

    if post_bench_size == 0 and turn_curr <= 2:
        components["r_donk_prevention"] = -0.20

    # 4. Evolution Progress
    if action_type == 9:  # EVOLVE
        components["r_evolution_progress"] = 0.10

    # 5. Supporter Play & Opportunity Cost
    played_card_id = pre.get("played_card_id", -1)
    if action_type == 7 and played_card_id in SUPPORTER_IDS:
        components["r_supporter_eff"] = 0.08

    post_hand_ids = post.get("hand_ids", [])
    has_supporter_in_hand = any(cid in SUPPORTER_IDS for cid in post_hand_ids)
    if action_type in (0, 14) and (played_card_id not in SUPPORTER_IDS) and has_supporter_in_hand:  # END_TURN
        components["r_supporter_opp_cost"] = -0.02

    # 6. Search Quality & Tempo
    # r_search_quality is a tempo-only signal (resources spent searching).
    # Card-level quality is already evaluated by mewtwo_expert search candidate priors.
    if (action_type == 7 and played_card_id in SEARCH_CARD_IDS) or action_type == 3:
        components["r_search_quality"] = 0.05   # quality lives in card-select priors
        components["r_search_tempo"] = 0.05

    # 7. Action Conversion & Hand Congestion
    # Require genuine board development (energy attached to active/bench or bench expanded)
    if action_type in (7, 8):  # PLAY or ATTACH
        pre_total_en = pre.get("energy", 0)
        post_total_en = post.get("energy", 0)
        if post_total_en > pre_total_en or post_bench_size > pre_bench_size or action_type == 9:
            components["r_action_conv"] = 0.08

    post_hand_size = post.get("hand_size", 0)
    if post_hand_size > 10 and action_type == 7:
        components["r_hand_congestion"] = -0.05

    # 8. State-Transition Retreat Delta: V(S') - V(S)
    # V(S) captures attacker readiness + active survival BEFORE the retreat.
    # V(S') captures readiness + survival AFTER the retreat (new active).
    # Energy discard is only an additional cost when the old active was healthy
    # (hp >= 80); if it was dying, the survival drop already accounts for it.
    # This correctly rewards retreating a 20-HP active to bring in a ready Mewtwo ex.
    if action_type == 12:  # RETREAT
        pre_act_en = pre.get("active_energy", 0)
        if pre_act_en == 0 and "my_active_energy" in pre:
            pre_act_en = pre.get("my_active_energy", 0)

        pre_min_req = pre.get("active_min_req_energy", 1)
        pre_readiness = min(1.0, pre_act_en / max(1, pre_min_req))

        post_act_en = post.get("active_energy", 0)
        if post_act_en == 0 and "my_active_energy" in post:
            post_act_en = post.get("my_active_energy", 0)

        post_min_req = post.get("active_min_req_energy", 1)
        post_readiness = min(1.0, post_act_en / max(1, post_min_req))

        pre_hp  = pre.get("active_hp", 100)
        pre_max = max(1, pre.get("active_max_hp", 100))
        post_hp  = post.get("active_hp", 100)
        post_max = max(1, post.get("active_max_hp", 100))

        pre_survival  = pre_hp  / pre_max
        post_survival = post_hp / post_max

        readiness_delta = post_readiness - pre_readiness
        survival_delta  = post_survival  - pre_survival

        # Only charge an energy cost penalty when the discarded active was healthy.
        # If pre_hp < 40 the active was dying — the survival loss already covers the cost.
        if pre_hp >= 80 and pre_act_en > 0:
            energy_cost = -0.10 * pre_act_en   # softer than -0.15; correctness depends on V delta
        else:
            energy_cost = 0.0

        board_delta = 0.60 * readiness_delta + 0.40 * survival_delta + energy_cost
        components["r_retreat_eff"] = max(-0.25, min(0.10, board_delta))

    # 9. Anti-Deckout Warning & Smooth Deck Preservation (Triggers smoothly at <= 8 cards)
    deck_cnt_post = post.get("deck_size", 60)
    if deck_cnt_post <= 8:
        preservation_scale = (8.0 - deck_cnt_post) / 8.0
        base_preservation = -0.04 * preservation_scale
        played_card_id = pre.get("played_card_id", -1)
        DRAW_SEARCH_CARDS = frozenset({1134, 1135, 1136, 1256, 1258, 1259})
        if action_type in (7, 3) and (played_card_id in DRAW_SEARCH_CARDS or played_card_id in SUPPORTER_IDS):
            base_preservation -= 0.10 * preservation_scale
        components["r_deck_preservation"] = max(-0.20, base_preservation)

    # 10. Stadium Deployment — state-transition delta: value(new_stadium) - value(replaced_stadium).
    # Replaces fixed +0.06 so that playing Battle Cage over TR Factory is penalized, not rewarded.
    _STADIUM_BENEFIT_MAP = {
        1257: 0.10,  # TR Factory — recurring draw engine; highest value
        1180: 0.08,  # Prism Tower — discard + draw utility
        1264: 0.08,  # Battle Cage — bench protection against snipers
        1263: 0.06,  # Generic stadium A
        1265: 0.06,  # Generic stadium B
    }
    _GENERIC_STADIUM_IDS = frozenset(_STADIUM_BENEFIT_MAP.keys())
    if action_type == 7 and played_card_id in _GENERIC_STADIUM_IDS:
        pre_stadium_id  = pre.get("stadium_id", -1)
        post_stadium_id = post.get("stadium_id", played_card_id)
        post_val = _STADIUM_BENEFIT_MAP.get(post_stadium_id, 0.0)
        pre_val  = _STADIUM_BENEFIT_MAP.get(pre_stadium_id, 0.0)
        stadium_delta = post_val - pre_val
        components["r_stadium_value"] = max(-0.08, min(0.10, stadium_delta))

    # 11. Missed Attack Penalty (powered attacker passed without attacking)
    # Requires a *legal* attack option in the pre-state, not just sufficient energy.
    # This avoids penalizing correct plays:
    #   - Bench + attach + Boss then KO next turn (Mimikyu immunity wall)
    #   - End turn when only available attack does 0 damage (immunity)
    act_ready = (pre_act_en >= _attack_threshold)
    had_legal_attack = pre.get("has_attack_option", False)
    if action_type in (0, 14) and act_ready and had_legal_attack and (pre_opp_hp - post_opp_hp <= 0):
        components["r_missed_attack"] = -0.25

    # 12. Slow Setup / Idle Pass Penalty (turns 2-5 with zero energy or empty bench)
    if action_type in (0, 14):
        total_board_en = pre.get("energy", 0)
        bench_cnt = pre.get("bench_size", 0)
        if 2 <= turn_curr <= 5 and (total_board_en == 0 or bench_cnt == 0):
            components["r_slow_setup"] = -0.10

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
