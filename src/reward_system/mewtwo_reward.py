"""
Team Rocket Mewtwo ex Outcome & Safety Reward Engine (Phase 2 Sparse RL).
Provides sparse, bounded strategic rewards and rule-safety guards for RL training.
Card-specific play rewards have been stripped in favor of outcome-based learning.
"""
from __future__ import annotations
from typing import Iterable, List, Dict, Optional

try:
    from src.expert_system.base_expert import expert_scale
except ImportError:
    try:
        from expert_system.base_expert import expert_scale
    except ImportError:
        def expert_scale(epoch=1):
            return 1.0

# Card ID Constants
BASIC_G_ENERGY_ID    = 1    # Basic {G} Energy
TR_ENERGY_ID         = 15   # Team Rocket's Energy
TAROUNTULA_ID        = 400  # Team Rocket's Tarountula
SPIDOPS_ID           = 401  # Team Rocket's Spidops
ARTICUNO_ID          = 414  # Team Rocket's Articuno
MEWTWO_EX_ID         = 431  # Team Rocket's Mewtwo ex
MIMIKYU_ID           = 434  # Team Rocket's Mimikyu

TRANSCEIVER_ID       = 1134 # Team Rocket's Transceiver
BUG_CATCHING_SET_ID  = 1094 # Bug Catching Set
BRAVE_BANGLE_ID      = 1175 # Brave Bangle
POKE_PAD_ID          = 1152 # Poké Pad
NIGHT_STRETCHER_ID   = 1159 # Night Stretcher
ARIANA_ID            = 1216 # Team Rocket's Ariana
GIOVANNI_ID          = 1218 # Team Rocket's Giovanni

SUPPORTER_IDS = frozenset({1216, 1218, 1219, 1220, 1227})
TR_POKEMON_IDS = frozenset({400, 401, 414, 431, 434})
EX_POKEMON_IDS = frozenset({431})

MIN_STRATEGIC_REWARD = -0.10
MAX_STRATEGIC_REWARD = 0.10

# Rule & Safety Penalties — bounded expert safety layer, NOT a replacement for RL learning.
# All values intentionally small so the neural network remains the final decision maker.
PENALTY_T1_GOING_FIRST_SUPPORTER        = -0.08
PENALTY_T1_EXPOSED_MEWTWO_EX            = -0.08
PENALTY_MEWTWO_EX_ATTACK_UNDER_POWER     = -0.08
PENALTY_EMPTY_BENCH_TURN_GT_1            = -0.05
REWARD_REACH_FOUR_ROCKET_POKEMON         = 0.06


def _count_tr_pokemon(bench_ids: Iterable[int], active_id: int) -> int:
    """Counts Team Rocket Pokémon currently in play (Active + Bench)."""
    count = 0
    for cid in [active_id] + list(bench_ids):
        if cid in TR_POKEMON_IDS:
            count += 1
    return count


# NOTE: USEFUL_ENERGY_THRESHOLDS was removed — energy cost lookup is now handled
# exclusively by energy_evaluator.get_required_energy() via the cg.api cache.



def compute_energy_attachment_reward(
    action_type: int,
    attached_card_id: int,
    target_card_id: int,
    is_active_target: bool,
    target_current_energy: int,
    bench_ids: list,
    active_id: int,
    active_hp: int,
    bench_energies: list | None = None,
    active_energy_before: int = 0,  # Fix Bug 2: must be passed to compute correct OpCost baseline
) -> float:
    if action_type != 8 or attached_card_id <= 0 or target_card_id <= 0:
        return 0.0

    try:
        from src.expert_system.energy_evaluator import compute_generic_energy_reward
    except ImportError:
        from expert_system.energy_evaluator import compute_generic_energy_reward

    energy_added = 2 if attached_card_id in (15, 19) else 1
    post_energy = target_current_energy + energy_added

    return compute_generic_energy_reward(
        target_card_id=target_card_id,
        energy_before=target_current_energy,
        energy_after=post_energy,
        attached_card_id=attached_card_id,
        is_active_target=is_active_target,
        bench_ids=bench_ids,
        bench_energies_before=bench_energies,
        active_id=active_id,
        active_energy_before=active_energy_before,  # Fix Bug 2: pass real active energy to evaluator
    )



def calculate_mewtwo_strategic_reward(
    pre: dict,
    post: dict,
    action_type: int,
    step_idx: int,
    went_second: bool,
    player_idx: int,
    opponent_name: str = "",
    debug_log: list | None = None
) -> float:
    """
    Computes sparse bounded strategic reward modifier r_strategic in range [-0.25, +0.25].
    Card-specific timing rewards are stripped; neural network learns optimal card plays
    directly from sparse outcome rewards (prizes taken/lost, KOs, win/loss).
    """
    r_strategic = 0.0
    turn = pre.get("turn", 1)

    def _add(val: float, reason: str):
        nonlocal r_strategic
        r_strategic += val
        if debug_log is not None:
            debug_log.append((reason, val))

    # -------------------------------------------------------------------------
    # 0. GLOBAL RULE & SAFETY GUARDS
    # -------------------------------------------------------------------------
    # A. Turn 1 going-first supporter guard (Illegal TCG rule)
    if turn == 1 and not went_second:
        played_id = pre.get("played_card_id", -1)
        if action_type == 7 and played_id in SUPPORTER_IDS:
            _add(PENALTY_T1_GOING_FIRST_SUPPORTER, "Invalid_Turn1_Going_First_Supporter")
            return max(MIN_STRATEGIC_REWARD, min(MAX_STRATEGIC_REWARD, r_strategic))

    # B. Exposed 2-Prize EX on Turn 1 going first
    if turn == 1 and not went_second:
        active_id = post.get("active_id", -1)
        pre_active_id = pre.get("active_id", -1)
        if active_id in EX_POKEMON_IDS and pre_active_id not in EX_POKEMON_IDS:
            _add(PENALTY_T1_EXPOSED_MEWTWO_EX, "T1_Going_First_Promoted_2Prize_EX_Active")

    # C. Wasteful retreat energy discard penalty or pointless active switching
    if action_type in (7, 12):  # PLAY (e.g. Switch card) or RETREAT
        pre_e = pre.get("my_active_energy", 0)
        post_e = post.get("my_active_energy", 0)
        pre_id = pre.get("active_id", -1)
        post_id = post.get("active_id", -1)
        pre_hp = pre.get("active_hp", 0)

        # Penalize swapping out healthy, fully powered Mewtwo ex (bounded prior — RL decides ultimately)
        if pre_id == MEWTWO_EX_ID and post_id != MEWTWO_EX_ID and pre_e >= 3 and pre_hp >= 100:
            _add(-0.08, "Penalty_Pointless_Switching_Healthy_Powered_Mewtwo_ex")

        if action_type == 12:
            # base_reward.py computes the full V(S')−V(S) retreat delta (readiness + survival + energy cost).
            # mewtwo_reward.py does NOT add a second flat penalty here — that would double-penalize every retreat.
            # The only safety guard remaining is the switch-out of a healthy powered Mewtwo ex above (line 152).
            pass  # intentionally empty — see comment above


    # D. Attack Execution Reward for Powered Mewtwo ex
    if action_type == 13:  # ATTACK
        act_id = pre.get("active_id", -1)
        act_energy = pre.get("my_active_energy", 0)
        bench = pre.get("bench_ids", [])
        if act_id == MEWTWO_EX_ID and act_energy >= 3 and _count_tr_pokemon(bench, act_id) >= 4:
            _add(0.08, "Reward_Attacking_With_Fully_Powered_Mewtwo_ex")

    # D. Power Saver ability requirement threshold check
    if action_type == 7:
        played_id = pre.get("played_card_id", -1)
        if played_id in TR_POKEMON_IDS:
            post_bench = post.get("bench_ids", [])
            post_active = post.get("active_id", -1)
            if _count_tr_pokemon(post_bench, post_active) == 4:
                _add(REWARD_REACH_FOUR_ROCKET_POKEMON, "Reached_Four_TR_Pokemon_Power_Saver_Threshold")

        # E. Evolution reward for Spidops (401)
        if played_id == 401:
            _add(0.05, "Evolved_Bench_Spidops_Attacker")

    # F. Unpowered Exposed Active Mewtwo ex Penalty (Power Saver Unmet & Energy < 3)
    post_act_id = post.get("active_id", -1)
    post_bench = post.get("bench_ids", [])
    post_energy = post.get("my_active_energy", 0)
    if (
        post_act_id == MEWTWO_EX_ID
        and (_count_tr_pokemon(post_bench, post_act_id) < 4 or post_energy < 3)
        and turn <= 4
    ):
        _add(-0.10, "Penalty_Exposed_Unpowered_Active_Mewtwo_ex_Power_Saver_Unmet")


    # -------------------------------------------------------------------------
    # 1. ENERGY ATTACHMENT EVALUATION (action_type == 8)
    # -------------------------------------------------------------------------
    if action_type == 8:
        attached_id = pre.get("attached_card_id", -1)
        target_id   = pre.get("attached_target_id", -1)
        target_energy = pre.get("target_energy", 0)
        bench_ids = pre.get("bench_ids", [])
        bench_energies = pre.get("bench_energies", [])
        active_id = pre.get("active_id", -1)
        active_energy = pre.get("my_active_energy", 0)  # canonical key — matches base_expert, mewtwo_expert
        active_hp = pre.get("active_hp", 100)
        hand_ids = pre.get("hand_ids", [])
        my_prizes_val  = pre.get("prizes", 6)
        opp_prizes_val = pre.get("opp_prizes", 6)

        is_active_target = (target_id == active_id)

        # Explicit overcharge penalty for ANY Pokemon (Active or Bench) already at or above max required energy
        try:
            from src.expert_system.energy_evaluator import get_required_energy
            req_energy = get_required_energy(target_id) if target_id > 0 else 2
        except Exception:
            req_energy = 2

        if target_energy >= req_energy:
            if is_active_target:
                _add(-0.25, f"Penalty_Active_Energy_Overcharge_Target_{target_id}")
            else:
                _add(-0.25, f"Penalty_Bench_Energy_Overcharge_Target_{target_id}")

        # Explicit Mimikyu energy cap penalty (max 1 energy on Mimikyu - ID 434)
        if target_id == 434 and target_energy >= 1:
            _add(-0.25, "Penalty_Mimikyu_Overcharge_Max_1_Energy")

        # Explicit Mimikyu energy type restriction penalty (TR Energy ID 15 only, never Basic G)
        if target_id == 434 and attached_id != 15 and attached_id > 0:
            _add(-0.25, "Penalty_Mimikyu_Wrong_Energy_Type_Restricted_to_TR_Energy")

        # Explicit reward for powering benched Spidops / Tarountula to supply Erasure Ball KO damage scaling
        if not is_active_target and target_id in (401, 400, 414, 431) and target_energy < 2:
            _add(0.30, "Reward_Benched_Energy_Fuel_For_Erasure_Ball_And_Backup")

        en_r = compute_energy_attachment_reward(
            action_type=8,
            attached_card_id=attached_id,
            target_card_id=target_id,
            is_active_target=is_active_target,
            target_current_energy=target_energy,
            bench_ids=bench_ids,
            active_id=active_id,
            active_hp=active_hp,
            bench_energies=bench_energies,
            active_energy_before=active_energy,  # Fix Bug 2: pass real active energy for OpCost baseline
        )
        _add(en_r, f"Energy_Attachment_Eval_target_{target_id}")

    # -------------------------------------------------------------------------
    # 2. ATTACK EXECUTION (action_type == 13)
    # -------------------------------------------------------------------------
    elif action_type == 13:
        act_id = pre.get("active_id", -1)
        bench_ids = pre.get("bench_ids", [])
        tr_count = _count_tr_pokemon(bench_ids, act_id)

        if act_id == MEWTWO_EX_ID and tr_count < 4:
            _add(PENALTY_MEWTWO_EX_ATTACK_UNDER_POWER, "Mewtwo_ex_Attack_With_Power_Saver_Unmet")

    # -------------------------------------------------------------------------
    # 3. PASS / TURN END SAFETY CHECK (action_type == 14)
    # -------------------------------------------------------------------------
    elif action_type == 14:
        if turn > 1 and post.get("bench_size", 0) == 0:
            _add(PENALTY_EMPTY_BENCH_TURN_GT_1, "Empty_Bench_Vulnerability_Turn_GT_1")

    # Scale strategic reward modifier by curriculum decay factor
    epoch_val = pre.get("epoch", 1)
    decay_factor = expert_scale(epoch_val)
    r_strategic *= decay_factor

    return max(MIN_STRATEGIC_REWARD, min(MAX_STRATEGIC_REWARD, r_strategic))
