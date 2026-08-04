"""
Marnie's Grimmsnarl ex + Munkidori Strategic Reward Engine.
Ground Truth: Matches 60-card CSV list (deckgrimnai.csv).
AlphaZero State-Improvement Paradigm: Rewards state transitions and board improvement
using smooth mathematical soft-scaling (tanh) to prevent reward saturation and reward hacking.
"""
import math

# Card ID Constants (Marnie's Grimmsnarl ex Deck Ground Truth)
IMPIDIMP_ID         = 646   # Marnie's Impidimp
MORGREM_ID          = 647   # Marnie's Morgrem
GRIMMSNARL_EX_ID    = 648   # Marnie's Grimmsnarl ex
MUNKIDORI_ID        = 112   # Munkidori
FROSLASS_ID         = 104   # Froslass
SNORUNT_ID          = 860   # Snorunt

BASIC_DARK_ENERGY_ID= 7     # Basic {D} Dark Energy
SPIKEMUTH_GYM_ID    = 1259  # Spikemuth Gym
RARE_CANDY_ID       = 1079  # Rare Candy
POFFIN_ID           = 1086  # Buddy-Buddy Poffin
POKE_PAD_ID         = 1152  # Poké Pad
POKEGEAR_ID         = 1122  # Pokégear 3.0
NIGHT_STRETCHER_ID  = 1097  # Night Stretcher
UNFAIR_STAMP_ID     = 1080  # Unfair Stamp
LUCKY_HELMET_ID      = 1156  # Lucky Helmet

PETREL_ID           = 1219  # Team Rocket's Petrel
LILLIES_DETERM_ID   = 1227  # Lillie's Determination
BOSS_ORDERS_ID      = 1182  # Boss's Orders
DAWN_ID             = 1231  # Dawn

SUPPORTER_IDS  = frozenset({1219, 1227, 1182, 1231})
ITEM_IDS       = frozenset({1086, 1152, 1079, 1097, 1122, 1080, 1156})
KEY_BENCH_IDS  = frozenset({MUNKIDORI_ID, FROSLASS_ID, SNORUNT_ID})

# -------------------------------------------------------------------------
# CENTRALIZED REWARD TRANSITION CONFIGURATION (NO MAGIC NUMBERS)
# -------------------------------------------------------------------------
REWARD_TRANSITION_CONFIG = {
    "EMERGENCY_BENCH_PROTECTION": 0.15,
    "BENCH_QUALITY_IMPROVED": 0.08,
    "EMPTY_BENCH_VULNERABILITY": -0.15,
    "MUNKIDORI_WAR_VICTORY_3PLUS": 0.18,
    "MUNKIDORI_THIRD_BENCH_PLACED": 0.14,
    "MUNKIDORI_ENGINE_EXPANDED": 0.08,
    "KEY_BENCH_PROTECTED": 0.10,
    "KEY_BENCH_LOST": -0.16,
    "OVERBENCH_LOW_IMPACT_BASICS": -0.08,
    "SINGLE_PRIZE_BOARD_SETUP": 0.12,
    "STAGE2_GRIMMSNARL_COMPLETED": 0.20,
    "STAGE1_MORGREM_EVOLUTION": 0.10,
    "STAGE1_FROSLASS_EVOLUTION": 0.10,
    "ENERGY_ATTACK_READY_GRIMMSNARL": 0.08,
    "ENERGY_ATTACK_READY": 0.12,
    "ENERGY_OVERCHARGE_PENALTY": -0.08,
    "MULTI_PRIZE_SWING_TURN": 0.18,
    "BOSS_GUST_MUNKIDORI_WAR": 0.18,
    "SUCCESSFUL_BOSS_GUST": 0.14,
    "BOSS_FORCED_ACTIVE_RESET": 0.08,
    "WASTED_BOSS_GUST": -0.08,
    "UNFAIR_STAMP_PERFECT_DIG": 0.18,
    "UNFAIR_STAMP_KEY_FIND": 0.14,
    "UNFAIR_STAMP_RARE_CANDY": 0.10,
    "UNFAIR_STAMP_SEVERE_DISRUPTION": 0.16,
    "UNFAIR_STAMP_MODERATE_DISRUPTION": 0.10,
    "UNFAIR_STAMP_AHEAD_MISUSE": -0.08,
    "POKEPAD_DOUBLE_SUPPORTER_RECYCLE": 0.14,
    "POKEPAD_SINGLE_SUPPORTER_RECYCLE": 0.08,
    "NIGHT_STRETCHER_COMBO": 0.14,
    "NIGHT_STRETCHER_STD": 0.08,
    "SPIKEMUTH_GYM_SETUP": 0.16,
    "POKEGEAR_EARLY_FETCH": 0.10,
    "FROSLASS_SPREAD_HIGH": 0.16,
    "FROSLASS_SPREAD_STD": 0.08,
    "HAND_RECOVERY_CRITICAL": 0.15,
    "HAND_RECOVERY_DEAD": 0.10,
    "WASTED_RARE_CANDY_PENALTY": -0.10,
}


def calculate_grimmsnarl_strategic_reward(
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
    Computes bounded strategic reward modifier r_strategic in range (-0.25, +0.25) for Grimmsnarl deck.
    Uses state-improvement transitions (AlphaZero style) with smooth tanh soft-scaling.
    """
    r_raw = 0.0
    turn = pre.get("turn", 1)

    def _add(val: float, reason: str):
        nonlocal r_raw
        r_raw += val
        if debug_log is not None:
            debug_log.append((reason, val))

    # -------------------------------------------------------------------------
    # 1. STATE IMPROVEMENT: BENCH QUALITY, MUNKIDORI WAR & PRIZE TEMPO
    # -------------------------------------------------------------------------
    pre_bench = pre.get("bench_size", 0)
    post_bench = post.get("bench_size", 0)

    if pre_bench == 0 and post_bench > 0:
        _add(REWARD_TRANSITION_CONFIG["EMERGENCY_BENCH_PROTECTION"], "Emergency_Bench_Protection_Deployed")
    elif pre_bench < 3 and post_bench > pre_bench:
        _add(REWARD_TRANSITION_CONFIG["BENCH_QUALITY_IMPROVED"], "Bench_Quality_Improved")

    if post_bench == 0 and turn > 1:
        _add(REWARD_TRANSITION_CONFIG["EMPTY_BENCH_VULNERABILITY"], "Empty_Bench_Vulnerability_Turn_GT_1")

    # Munkidori War Victory (having more benched Munkidori than opponent & >= 3)
    pre_munkidori_count = sum(1 for pid in pre.get("my_bench_ids", []) if pid == MUNKIDORI_ID)
    post_munkidori_count = sum(1 for pid in post.get("my_bench_ids", []) if pid == MUNKIDORI_ID)
    opp_munkidori_count = sum(1 for pid in pre.get("opp_bench_ids", []) if pid == MUNKIDORI_ID)

    if post_munkidori_count > opp_munkidori_count and post_munkidori_count >= 3:
        _add(REWARD_TRANSITION_CONFIG["MUNKIDORI_WAR_VICTORY_3PLUS"], "Munkidori_War_Victory_3Plus")
    elif post_munkidori_count > pre_munkidori_count and post_munkidori_count == 3:
        _add(REWARD_TRANSITION_CONFIG["MUNKIDORI_THIRD_BENCH_PLACED"], "Munkidori_Third_Bench_Placed")
    elif post_munkidori_count > pre_munkidori_count:
        _add(REWARD_TRANSITION_CONFIG["MUNKIDORI_ENGINE_EXPANDED"], "Munkidori_Bench_Engine_Expanded")

    # Key Bench Engine Protection
    pre_key_count = sum(1 for pid in pre.get("my_bench_ids", []) if pid in KEY_BENCH_IDS)
    post_key_count = sum(1 for pid in post.get("my_bench_ids", []) if pid in KEY_BENCH_IDS)

    if post_key_count > pre_key_count:
        _add(REWARD_TRANSITION_CONFIG["KEY_BENCH_PROTECTED"], "Key_Bench_Pokemon_Protected")
    elif post_key_count == 0 and pre_key_count > 0:
        _add(REWARD_TRANSITION_CONFIG["KEY_BENCH_LOST"], "Key_Bench_Pokemon_Lost")

    # Penalty for over-benching low impact basics (e.g. 3 Impidimp + 3 Munkidori filling 5 bench slots)
    post_impidimp_count = sum(1 for pid in post.get("my_bench_ids", []) if pid == IMPIDIMP_ID)
    if post_impidimp_count >= 3 and post_munkidori_count >= 3 and post_bench >= 5:
        _add(REWARD_TRANSITION_CONFIG["OVERBENCH_LOW_IMPACT_BASICS"], "Overbench_Low_Impact_Basics")

    # Prize Tempo & Single-Prize Board Damage Setup
    my_prizes_taken = 6 - post.get("my_prizes", 6)
    opp_prizes_taken = 6 - post.get("opp_prizes", 6)
    post_has_ex = (GRIMMSNARL_EX_ID in post.get("my_bench_ids", []) or post.get("active_id") == GRIMMSNARL_EX_ID)
    counters_placed = post.get("damage_counters_placed", 0) - pre.get("damage_counters_placed", 0)

    if not post_has_ex and my_prizes_taken <= opp_prizes_taken and counters_placed > 0:
        _add(REWARD_TRANSITION_CONFIG["SINGLE_PRIZE_BOARD_SETUP"], "Single_Prize_Board_Damage_Setup")

    # -------------------------------------------------------------------------
    # 2. STATE IMPROVEMENT: EVOLUTION & ATTACKER COMPLETION
    # -------------------------------------------------------------------------
    pre_has_grimmsnarl = (GRIMMSNARL_EX_ID in pre.get("my_bench_ids", []) or pre.get("active_id") == GRIMMSNARL_EX_ID)

    if action_type in (7, 9):
        evolved_id = post.get("evolved_card_id", -1)
        if (evolved_id == GRIMMSNARL_EX_ID or post_has_ex) and not pre_has_grimmsnarl:
            _add(REWARD_TRANSITION_CONFIG["STAGE2_GRIMMSNARL_COMPLETED"], "Stage2_Grimmsnarl_ex_Completed")
        elif evolved_id == MORGREM_ID:
            _add(REWARD_TRANSITION_CONFIG["STAGE1_MORGREM_EVOLUTION"], "Stage1_Morgrem_Evolution")
        elif evolved_id == FROSLASS_ID:
            _add(REWARD_TRANSITION_CONFIG["STAGE1_FROSLASS_EVOLUTION"], "Stage1_Froslass_Evolution")

        # Penalty if Rare Candy was played but no evolution completed
        if pre.get("played_card_id") == RARE_CANDY_ID and not post_has_ex and evolved_id == -1:
            _add(REWARD_TRANSITION_CONFIG["WASTED_RARE_CANDY_PENALTY"], "Wasted_Rare_Candy_No_Evolution")

    # -------------------------------------------------------------------------
    # 3. STATE IMPROVEMENT: ATTACK READINESS & MULTI-PRIZE SWING
    # -------------------------------------------------------------------------
    pre_active_e = pre.get("active_energy", 0)
    post_active_e = post.get("active_energy", 0)

    if pre_active_e < 1 and post_active_e >= 1:
        if post.get("active_id") == GRIMMSNARL_EX_ID:
            _add(REWARD_TRANSITION_CONFIG["ENERGY_ATTACK_READY_GRIMMSNARL"], "Energy_Enables_1Cost_Attack_Grimmsnarl")
        else:
            _add(REWARD_TRANSITION_CONFIG["ENERGY_ATTACK_READY"], "Energy_Enables_1Cost_Attack")
    elif pre_active_e < 2 and post_active_e >= 2:
        if post.get("active_id") == GRIMMSNARL_EX_ID:
            _add(REWARD_TRANSITION_CONFIG["ENERGY_ATTACK_READY_GRIMMSNARL"], "Energy_Enables_Main_Attack_Grimmsnarl")
        else:
            _add(REWARD_TRANSITION_CONFIG["ENERGY_ATTACK_READY"], "Energy_Enables_Main_Attack")

    if post_active_e > 2 and action_type == 8:
        _add(REWARD_TRANSITION_CONFIG["ENERGY_OVERCHARGE_PENALTY"], "Overcharging_Energy_Penalty")

    if action_type == 13 and post.get("ko_occurred", False):
        ko_is_ex = post.get("ko_card_is_ex", False)
        if ko_is_ex and my_prizes_taken >= opp_prizes_taken - 1:
            _add(REWARD_TRANSITION_CONFIG["MULTI_PRIZE_SWING_TURN"], "Grimmsnarl_Multi_Prize_Swing_Turn")

    # -------------------------------------------------------------------------
    # 4. STATE IMPROVEMENT: MULTI-TIERED BOSS GUST & MUNKIDORI TARGETING
    # -------------------------------------------------------------------------
    if action_type == 7 and pre.get("played_card_id") == BOSS_ORDERS_ID:
        active_changed = (pre.get("opp_active_id") != post.get("opp_active_id"))
        opp_bench_count = len(pre.get("opp_bench_ids", []))
        opp_active_is_munkidori = (post.get("opp_active_id") == MUNKIDORI_ID)

        if active_changed and opp_active_is_munkidori:
            _add(REWARD_TRANSITION_CONFIG["BOSS_GUST_MUNKIDORI_WAR"], "Boss_Gust_Munkidori_War_Victory")
        elif active_changed and opp_bench_count > 0:
            _add(REWARD_TRANSITION_CONFIG["SUCCESSFUL_BOSS_GUST"], "Successful_Boss_Gust")
        elif active_changed:
            _add(REWARD_TRANSITION_CONFIG["BOSS_FORCED_ACTIVE_RESET"], "Boss_Forced_Active_Reset")
        else:
            _add(REWARD_TRANSITION_CONFIG["WASTED_BOSS_GUST"], "Wasted_Boss_Gust_No_Bench")

    # -------------------------------------------------------------------------
    # 5. UNFAIR STAMP DISRUPTION & POKÉ PAD SUPPORTER DENSITY LOOPS
    # -------------------------------------------------------------------------
    if action_type == 7:
        played_id = pre.get("played_card_id", -1)

        if played_id == UNFAIR_STAMP_ID:
            pre_hand = set(pre.get("hand_ids", []))
            post_hand = set(post.get("hand_ids", []))
            new_cards = post_hand - pre_hand

            found_spikemuth = SPIKEMUTH_GYM_ID in new_cards
            found_night_stretcher = NIGHT_STRETCHER_ID in new_cards
            found_supporter = any(cid in SUPPORTER_IDS for cid in new_cards)
            found_candy = RARE_CANDY_ID in new_cards

            if found_spikemuth and (found_night_stretcher or found_supporter):
                _add(REWARD_TRANSITION_CONFIG["UNFAIR_STAMP_PERFECT_DIG"], "Unfair_Stamp_Perfect_Dig_Spikemuth_Plus")
            elif found_spikemuth or found_night_stretcher or found_supporter:
                _add(REWARD_TRANSITION_CONFIG["UNFAIR_STAMP_KEY_FIND"], "Unfair_Stamp_Successful_Key_Find")
            elif found_candy:
                _add(REWARD_TRANSITION_CONFIG["UNFAIR_STAMP_RARE_CANDY"], "Unfair_Stamp_Rare_Candy_Found")

            # Opponent hand disruption reward / misuse penalty
            opp_pre_hand_size = pre.get("opp_hand_size", 0)
            opp_post_hand_size = post.get("opp_hand_size", 0)

            if opp_pre_hand_size >= 6 and opp_post_hand_size <= 3:
                _add(REWARD_TRANSITION_CONFIG["UNFAIR_STAMP_SEVERE_DISRUPTION"], "Unfair_Stamp_Severe_Opp_Hand_Shorten")
            elif opp_pre_hand_size >= 4 and opp_post_hand_size < opp_pre_hand_size:
                _add(REWARD_TRANSITION_CONFIG["UNFAIR_STAMP_MODERATE_DISRUPTION"], "Unfair_Stamp_Moderate_Opp_Hand_Shorten")
            elif opp_pre_hand_size <= 2 and my_prizes_taken > opp_prizes_taken:
                _add(REWARD_TRANSITION_CONFIG["UNFAIR_STAMP_AHEAD_MISUSE"], "Unfair_Stamp_Ahead_Misuse_Penalty")

        elif played_id == POKE_PAD_ID:
            pre_disc = set(pre.get("discard_ids", []))
            post_disc = set(post.get("discard_ids", []))
            supporters_removed = [cid for cid in (pre_disc - post_disc) if cid in SUPPORTER_IDS]

            if len(supporters_removed) >= 2:
                _add(REWARD_TRANSITION_CONFIG["POKEPAD_DOUBLE_SUPPORTER_RECYCLE"], "PokePad_Double_Supporter_Recycle")
            elif len(supporters_removed) == 1:
                _add(REWARD_TRANSITION_CONFIG["POKEPAD_SINGLE_SUPPORTER_RECYCLE"], "PokePad_Single_Supporter_Recycle")

        elif played_id == NIGHT_STRETCHER_ID:
            post_hand = set(post.get("hand_ids", []))
            pre_hand = set(pre.get("hand_ids", []))
            recovered_supporter = any(cid in SUPPORTER_IDS for cid in (post_hand - pre_hand))
            if recovered_supporter:
                _add(REWARD_TRANSITION_CONFIG["NIGHT_STRETCHER_COMBO"], "Unfair_Stamp_Night_Stretcher_Combo_Success")
            else:
                _add(REWARD_TRANSITION_CONFIG["NIGHT_STRETCHER_STD"], "Night_Stretcher_Standard_Recovery")

        elif played_id == SPIKEMUTH_GYM_ID:
            fetched_grimmsnarl = GRIMMSNARL_EX_ID in post.get("hand_ids", [])
            if fetched_grimmsnarl:
                _add(REWARD_TRANSITION_CONFIG["SPIKEMUTH_GYM_SETUP"], "Spikemuth_Gym_Core_Setup_Completed")

        elif played_id == POKEGEAR_ID and turn <= 2:
            _add(REWARD_TRANSITION_CONFIG["POKEGEAR_EARLY_FETCH"], "PokeGear_Early_Supporter_Fetch")

    elif action_type == 10:  # ABILITY
        _add(0.12, "Munkidori_Adrenaline_Brain_Damage_Move")

    # -------------------------------------------------------------------------
    # 6. FROSLASS DAMAGE SPREAD & CRITICAL HAND RECOVERY
    # -------------------------------------------------------------------------
    if counters_placed >= 3:
        _add(REWARD_TRANSITION_CONFIG["FROSLASS_SPREAD_HIGH"], "Froslass_Damage_Spread_High_Impact")
    elif counters_placed >= 1:
        _add(REWARD_TRANSITION_CONFIG["FROSLASS_SPREAD_STD"], "Froslass_Damage_Spread_Standard")

    if action_type == 7:
        pre_hand_size = pre.get("hand_size", 5)
        post_hand_size = post.get("hand_size", 0)

        if pre_hand_size <= 2 and post_hand_size >= 4:
            _add(REWARD_TRANSITION_CONFIG["HAND_RECOVERY_CRITICAL"], "Hand_Recovery_From_Critical_Hand_1_2")
        elif pre_hand_size <= 3 and post_hand_size >= 5:
            _add(REWARD_TRANSITION_CONFIG["HAND_RECOVERY_DEAD"], "Hand_Recovery_From_Dead_Hand")

    # Smooth mathematical soft-scaling via tanh (strictly bounded in (-0.25, +0.25) without hard clipping)
    r_strategic = 0.25 * math.tanh(r_raw / 0.25)

    if debug_log is not None:
        debug_log.append(("Raw_Strategic_Reward", r_raw))
        debug_log.append(("Tanh_Soft_Scaled_Strategic_Reward", r_strategic))

    return r_strategic
