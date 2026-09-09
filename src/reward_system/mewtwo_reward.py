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
NIGHT_STRETCHER_ID   = 1097 # Night Stretcher
ENERGY_SWITCH_ID     = 1116 # Energy Switch
SWITCH_ID            = 1123 # Switch
POKE_PAD_ID          = 1152 # Poké Pad
HEROS_CAPE_ID        = 1159 # Hero's Cape (ACE SPEC Tool)
BRAVE_BANGLE_ID      = 1175 # Brave Bangle
ARIANA_ID            = 1216 # Team Rocket's Ariana
GIOVANNI_ID          = 1218 # Team Rocket's Giovanni
PETREL_ID            = 1219 # Team Rocket's Petrel
PROTON_ID            = 1220 # Team Rocket's Proton
LILLIES_DETERM_ID    = 1227 # Lillie's Determination
BATTLE_CAGE_ID       = 1264 # Battle Cage

SUPPORTER_IDS = frozenset({1216, 1218, 1219, 1220, 1227})
TR_POKEMON_IDS = frozenset({400, 401, 414, 431, 434})
EX_POKEMON_IDS = frozenset({431})

MIN_STRATEGIC_REWARD = -0.25
MAX_STRATEGIC_REWARD = 0.25

# Rule & Safety Penalties — bounded expert safety layer, NOT a replacement for RL learning.
# All values intentionally small so the neural network remains the final decision maker.
PENALTY_T1_GOING_FIRST_SUPPORTER        = -0.08
PENALTY_T1_EXPOSED_MEWTWO_EX            = -0.08
PENALTY_MEWTWO_EX_ATTACK_UNDER_POWER     = -0.08
PENALTY_SPIDOPS_RETREAT_WHEN_CAN_ATTACK = -0.10
PENALTY_EMPTY_BENCH_TURN_GT_1            = -0.05
REWARD_REACH_FOUR_ROCKET_POKEMON         = 0.04  # Reduced 0.06→0.04: losses unlock Power Saver faster (turn 4.0) than wins (turn 4.3) — rush incentive was too strong


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
    if action_type in (7, 12):  # PLAY (e.g. Switch card, Giovanni) or RETREAT
        pre_e = pre.get("my_active_energy", 0)
        post_e = post.get("my_active_energy", 0)
        pre_id = pre.get("active_id", -1)
        post_id = post.get("active_id", -1)
        pre_hp = pre.get("active_hp", 0)

        # Penalize swapping out healthy, fully powered Mewtwo ex (bounded prior — RL decides ultimately)
        if pre_id == MEWTWO_EX_ID and post_id != MEWTWO_EX_ID and pre_e >= 3 and pre_hp >= 100:
            _add(-0.08, "Penalty_Pointless_Switching_Healthy_Powered_Mewtwo_ex")

        # Giovanni-triggered energy discard penalty:
        # Giovanni forces a retreat of the active Pokemon, discarding its energy silently.
        # This fires when action_type==7 (PLAY) + played_card==Giovanni + active energy went from >0 to 0
        # and the active slot changed (retreat actually happened).
        if action_type in (7, 10, 11):
            played_id = pre.get("played_card_id", -1)
            if played_id == GIOVANNI_ID:
                # Check if energy was discarded and the active slot changed (retreat happened)
                if pre_e >= 1 and post_e == 0 and pre_id != -1 and post_id != pre_id:
                    if pre_id in (SPIDOPS_ID, 401, 400):
                        _add(-0.40, "Penalty_Giovanni_Forced_Retreat_Discarded_Spidops_Energy")
                    elif pre_id == MEWTWO_EX_ID:
                        _add(-0.35, "Penalty_Giovanni_Forced_Retreat_Discarded_Mewtwo_Energy")
                    elif pre_id == ARTICUNO_ID:
                        _add(-0.20, "Penalty_Giovanni_Forced_Retreat_Discarded_Articuno_Energy")
                    else:
                        _add(-0.15, "Penalty_Giovanni_Forced_Retreat_Discarded_Active_Energy")
                opp_act_hp_gio = pre.get("opp_active_hp", 0)
                opp_bench_cnt_gio = len(pre.get("opp_bench_ids", []))
                if (opp_act_hp_gio >= 250 or "archaludon" in opponent_name.lower() or "starmie" in opponent_name.lower() or "abomasnow" in opponent_name.lower()) and opp_bench_cnt_gio > 0:
                    _add(0.08, "Reward_Giovanni_Gust_Bypassing_300HP_Tank_Active")

            if played_id in (BRAVE_BANGLE_ID, 1175, 1158):
                target_poke = pre.get("target_card_id", post.get("target_card_id", -1))
                if target_poke in (ARTICUNO_ID, 414, 400, 434):
                    _add(-0.35, "Penalty_Wasted_Brave_Bangle_On_Non_Attacker")
                elif target_poke == MEWTWO_EX_ID:
                    _add(0.08, "Reward_Equipped_Brave_Bangle_To_Mewtwo_ex")

            if played_id in (HEROS_CAPE_ID, 1159):
                target_poke = pre.get("target_card_id", post.get("target_card_id", -1))
                if target_poke == MEWTWO_EX_ID:
                    _add(0.12, "Reward_Equipped_Heros_Cape_To_Mewtwo_ex_380HP_Tank")
                elif target_poke in (ARTICUNO_ID, 414, MIMIKYU_ID, 434, TAROUNTULA_ID, 400):
                    _add(-0.15, "Penalty_Wasted_Heros_Cape_On_Non_Primary_Attacker")

        if action_type == 12:
            had_legal_attack = pre.get("has_attack_option", False)
            # Catastrophic retreat: retreated when attack was legally available this turn.
            # has_attack_option=True means ATTACK (type 13) was in the options list when retreat was chosen.
            # This is the strongest evidence of a loop mistake — fire a large training penalty.
            if had_legal_attack and pre_hp > 30:
                req_energy_ret = {SPIDOPS_ID: 2, MEWTWO_EX_ID: 3, ARTICUNO_ID: 2, MIMIKYU_ID: 1}.get(pre_id, 2)
                if pre_e >= req_energy_ret:
                    if pre_id in (SPIDOPS_ID, 401, 400):
                        _add(-0.50, "CATASTROPHIC_Penalty_Retreated_With_Attack_Available_Spidops_Had_Energy")
                    else:
                        _add(-0.40, "CATASTROPHIC_Penalty_Retreated_With_Attack_Available_Attacker_Had_Energy")
            elif pre_id == SPIDOPS_ID and (pre_e >= 2 or had_legal_attack) and pre_hp > 30:
                _add(PENALTY_SPIDOPS_RETREAT_WHEN_CAN_ATTACK, "Penalty_Spidops_Retreat_When_Able_To_Attack")
            # base_reward.py computes the full V(S')−V(S) retreat delta (readiness + survival + energy cost).
            # mewtwo_reward.py does NOT add a second flat penalty here — that would double-penalize every retreat.
            # The only safety guard remaining is the switch-out of a healthy powered Mewtwo ex above (line 153).
            pass


    # D. Attack Execution Reward for Powered Mewtwo ex
    if action_type == 13:  # ATTACK
        act_id = pre.get("active_id", -1)
        act_energy = pre.get("my_active_energy", 0)
        post_act_energy = post.get("my_active_energy", 0)
        bench = pre.get("bench_ids", [])
        opp_hp = pre.get("opp_active_hp", 100)
        opp_id = pre.get("opp_active_id", -1)
        is_tank_opp = (opp_hp > 160 or opp_id in EX_POKEMON_IDS)

        if act_id == MEWTWO_EX_ID and act_energy >= 3 and _count_tr_pokemon(bench, act_id) >= 4:
            _add(0.08, "Reward_Attacking_With_Fully_Powered_Mewtwo_ex")

        # CATASTROPHIC Penalty: Stripping energy from Benched Mewtwo ex or Benched Spidops during attack cost payment
        pre_bench_ids = pre.get("bench_ids", [])
        pre_bench_en  = pre.get("bench_energies", [])
        post_bench_ids = post.get("bench_ids", [])
        post_bench_en  = post.get("bench_energies", [])
        for p_bid, p_be in zip(pre_bench_ids, pre_bench_en):
            if p_bid == MEWTWO_EX_ID and p_be > 0:
                for post_bid, post_be in zip(post_bench_ids, post_bench_en):
                    if post_bid == MEWTWO_EX_ID and post_be < p_be:
                        _add(-0.50, "CATASTROPHIC_Penalty_Discarded_Energy_From_Benched_Mewtwo_ex")
            elif p_bid in (401, 400) and p_be > 0 and act_energy >= 2:
                for post_bid, post_be in zip(post_bench_ids, post_bench_en):
                    if post_bid in (401, 400) and post_be < p_be:
                        _add(-0.45, "CATASTROPHIC_Penalty_Discarded_Energy_From_Benched_Spidops")

        # Contextual Erasure Ball Discard Reward vs Overkill Penalty:
        energy_discarded_from_active = (act_energy - post_act_energy)
        if act_id == MEWTWO_EX_ID and energy_discarded_from_active >= 2:
            if not is_tank_opp and opp_hp <= 160:
                # Discarding energy against a <= 160 HP target was unnecessary overkill
                _add(-0.25, "Penalty_Unnecessary_Erasure_Ball_Discard_Overkill")
            elif is_tank_opp and post.get("prizes", 6) < pre.get("prizes", 6):
                # Discarding energy on a tank opponent achieved a crucial KO
                _add(0.15, "Reward_Lethal_Erasure_Ball_Discard_On_Tank_KO")

        # Spidops Weakness Counter Attack Reward (Darkness EX / Fighting EX)
        if act_id in (SPIDOPS_ID, 401) and act_energy >= 2:
            is_dark_or_fight = (opp_id in (646, 647, 648, 112, 104, 674, 678)) or any(k in opponent_name.lower() for k in ("grimmsnarl", "marnie", "darkness", "lucario", "fighting"))
            if is_dark_or_fight:
                _add(0.12, "Reward_Spidops_Attacking_Weakness_Darkness_Fighting_EX")

    # D. Power Saver ability requirement threshold check
    if action_type == 7:
        played_id = pre.get("played_card_id", -1)
        if played_id in TR_POKEMON_IDS:
            post_bench = post.get("bench_ids", [])
            post_active = post.get("active_id", -1)
            tr_cnt = _count_tr_pokemon(post_bench, post_active)
            if tr_cnt == 4:
                _add(REWARD_REACH_FOUR_ROCKET_POKEMON, "Reached_Four_TR_Pokemon_Power_Saver_Threshold")
            elif tr_cnt == 5:
                _add(0.03, "Reward_Five_TR_Pokemon_Post_KO_Power_Saver_Buffer")

        # E. Evolution reward for Spidops (401)
        # Reduced 0.05→0.03: expert already gives BONUS_EVOLVE_SPIDOPS=0.16 prior — trimming double-counting
        if played_id == 401:
            _add(0.03, "Evolved_Bench_Spidops_Attacker")

    # F. Unpowered Exposed Active Mewtwo ex Penalty (Power Saver Unmet & Energy < 3)
    post_act_id = post.get("active_id", -1)
    post_bench = post.get("bench_ids", [])
    post_energy = post.get("my_active_energy", 0)
    if (
        post_act_id == MEWTWO_EX_ID
        and (_count_tr_pokemon(post_bench, post_act_id) < 4 or post_energy < 3)
        and turn > 4
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

        # Explicit hard overcharge penalty for Spidops/Tarountula (>= 2 energy) and Mewtwo ex (>= 3 energy)
        if target_id in (401, 400) and target_energy >= 2:
            _add(-0.35, "Penalty_Spidops_Overcharge_Exceeds_2_Energy_Cap")
        elif target_id == 431 and target_energy >= 3:
            _add(-0.35, "Penalty_Mewtwo_ex_Overcharge_Exceeds_3_Energy_Cap")
        elif target_id == 434 and target_energy >= 1:
            _add(-0.35, "Penalty_Mimikyu_Overcharge_Max_1_Energy")

        # Explicit Mimikyu energy type restriction penalty (TR Energy ID 15 only, never Basic G)
        if target_id == 434 and attached_id != 15 and attached_id > 0:
            _add(-0.25, "Penalty_Mimikyu_Wrong_Energy_Type_Restricted_to_TR_Energy")

        # Explicit Articuno energy penalty (Articuno is a defensive pivot and cannot attack with Grass/TR energy)
        if target_id == ARTICUNO_ID:
            _add(-0.25, "Penalty_Attached_Energy_To_Articuno_Cannot_Attack")

        # Energy Concentration vs Splitting (P0 Forensic Fix):
        # When Mewtwo ex is in play and needs 1 more energy to reach 3, penalize attaching to secondary attackers
        # unless secondary attacker achieves immediate lethal KO attack readiness this turn.
        mewtwo_in_play = (active_id == MEWTWO_EX_ID) or (MEWTWO_EX_ID in bench_ids)
        mewtwo_energy = active_energy if active_id == MEWTWO_EX_ID else 0
        if not mewtwo_energy and MEWTWO_EX_ID in bench_ids and bench_energies:
            for bid, be in zip(bench_ids, bench_energies):
                if bid == MEWTWO_EX_ID:
                    mewtwo_energy = max(mewtwo_energy, be)
        
        if mewtwo_in_play and mewtwo_energy in (1, 2) and target_id != MEWTWO_EX_ID:
            # Check if Spidops target is achieving 2 energy for immediate attack
            is_spidops_readying = (target_id in (400, 401) and target_energy + (2 if attached_id in (15, 19) else 1) >= 2 and is_active_target)
            if not is_spidops_readying:
                _add(-0.15, "Penalty_Energy_Fragmentation_Split_Away_From_Mewtwo_ex")

        # Explicit reward for powering Mewtwo ex to 3 energy or Spidops to 2 energy
        opp_hp = pre.get("opp_active_hp", 0)
        opp_id = pre.get("opp_active_id", -1)
        is_tank_opp = (opp_hp >= 180 or opp_id in EX_POKEMON_IDS)
        if not is_active_target:
            if target_id == MEWTWO_EX_ID and target_energy < 3:
                _add(0.18, "Reward_Benched_Mewtwo_Energy_Fuel_3_Energy_Cap")
            elif target_id in (401, 400) and target_energy < 2:
                if is_tank_opp:
                    _add(0.12, "Reward_Bench_Energy_Fuel_Vs_Tank_Opponent")
                else:
                    _add(0.12, "Reward_Benched_Spidops_Energy_Fuel_2_Energy_Cap")

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
    # 1.B BATTLE CAGE CONTEXT-SENSITIVE TIMING REWARD
    # Evidence from 20 losses: 18/22 Battle Cage plays occurred vs single-target decks (0 bench dmg).
    # Reward: +0.08 when opponent has bench snipe threats (Dragapult, Froslass, etc.)
    # Penalty: -0.04 when played against decks with no bench damage capability
    # -------------------------------------------------------------------------
    if action_type == 7:
        played_id_cage = pre.get("played_card_id", -1)
        if played_id_cage == 1264:  # BATTLE_CAGE_ID
            bench_counter_keywords = ("dragapult", "froslass", "dusknoir", "dusclops", "munkidori", "dipplin", "phantom", "grimmsnarl", "marnie", "sixth", "spread")
            is_bench_counter_opp = any(k in opponent_name.lower() for k in bench_counter_keywords) or opponent_name == "DRAGAPULT"
            cage_bench = len([b for b in pre.get("bench_ids", []) if b > 0])
            cage_turn = pre.get("turn", 1)
            if is_bench_counter_opp:
                if cage_bench >= 3 and cage_turn >= 4:
                    _add(0.08, "Reward_Battle_Cage_Bench_Shield_Vs_Bench_Snipe")
                else:
                    _add(0.04, "Reward_Battle_Cage_Deployed_Vs_Bench_Snipe")
            else:
                _add(-0.04, "Penalty_Battle_Cage_Wasted_No_Bench_Snipe_Threat")

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
    # 2.B RETREAT EVALUATION (action_type == 12)
    # -------------------------------------------------------------------------
    elif action_type == 12:
        pre_act_id = pre.get("active_id", -1)
        pre_act_en = pre.get("my_active_energy", 0)
        pre_act_hp = pre.get("active_hp", 100)

        # Healthy Powered Mewtwo ex Retreat Penalty (Discards 2 Energy and loses attack tempo)
        if pre_act_id == MEWTWO_EX_ID and pre_act_en >= 3 and pre_act_hp > 120:
            _add(-0.35, "Penalty_Retreat_Healthy_Powered_Mewtwo_ex")

        # Power Saver Lockout Retreat Penalty (AL10 match_02_loss root cause fix)
        if pre_act_id == MEWTWO_EX_ID and pre_act_en >= 1:
            tr_count_r = _count_tr_pokemon(pre.get("bench_ids", []), pre_act_id)
            if tr_count_r < 4:
                _add(-0.25, "Penalty_Retreat_Power_Saver_Locked_Mewtwo_ex")

        # R2 (LATEST evidence): 5 Spidops waste retreats. Healthy (hp>30) retreat with energy
        # is genuine waste. Near-KO escape is legitimate and should not be penalized heavily.
        # Expert already fires -0.50; reward aligns directionally (-0.40 healthy, -0.08 near-KO).
        if pre_act_id in (400, 401) and pre_act_en >= 1:
            if pre_act_hp > 30:
                _add(-0.40, "Penalty_Spidops_Retreat_With_Energy_Healthy_HP")
            else:
                # Near-KO escape — small penalty preserves tactical flexibility
                _add(-0.08, "Penalty_Spidops_Low_HP_Retreat_With_Energy_Near_KO")

        # R1+R4 (LATEST evidence): 12 Articuno waste retreats.
        # Key finding: 7/12 occur when TR<4 — Articuno active still building Power Saver.
        # When TR>=4, Articuno active slot provides zero strategic value; energy waste is pure loss.
        # R4: Multi-energy (2+) accumulation pattern (match_06_win T=6/8/10) gets extra penalty.
        elif pre_act_id == ARTICUNO_ID and pre_act_en >= 1 and pre_act_hp > 30:
            tr_count_r = _count_tr_pokemon(pre.get("bench_ids", []), pre.get("active_id", -1))
            if pre_act_en >= 2 and tr_count_r >= 4:
                # R4: Multi-energy waste with full TR board — no justification
                _add(-0.20, "Penalty_Articuno_Multi_Energy_Retreat_TR_Full_Board_Waste")
            elif tr_count_r >= 4:
                # R1: TR requirement met — energy loss has no strategic offset
                _add(-0.15, "Penalty_Articuno_Retreat_Energy_Waste_TR_Count_Already_Met")
            else:
                # R1: TR < 4 — Articuno active may be helping TR count; softer penalty
                _add(-0.05, "Penalty_Articuno_Retreat_Energy_Waste_TR_Count_Partial")

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
