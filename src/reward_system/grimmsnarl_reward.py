"""
Marnie's Grimmsnarl ex + Munkidori Strategic Reward Engine.
Ground Truth: Matches 60-card CSV list (deckgrimnai.csv).
AlphaZero State-Improvement Paradigm: Rewards state transitions and board improvement
(Froslass damage spread, key bench protection, Munkidori saturation, Punk Up energy awareness, Boss gusting)
rather than static card plays. Prevents reward hacking.
"""

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
    Computes bounded strategic reward modifier r_strategic in range [-0.25, +0.25] for Grimmsnarl deck.
    Uses state-improvement transitions (AlphaZero style) to avoid reward hacking.
    """
    r_strategic = 0.0
    turn = pre.get("turn", 1)

    def _add(val: float, reason: str):
        nonlocal r_strategic
        r_strategic += val
        if debug_log is not None:
            debug_log.append((reason, val))

    # -------------------------------------------------------------------------
    # 1. STATE IMPROVEMENT: BENCH QUALITY & PROTECTION
    # -------------------------------------------------------------------------
    pre_bench = pre.get("bench_size", 0)
    post_bench = post.get("bench_size", 0)

    if pre_bench == 0 and post_bench > 0:
        _add(0.20, "Emergency_Bench_Protection_Deployed")
    elif pre_bench < 3 and post_bench > pre_bench:
        _add(0.10, "Bench_Quality_Improved")

    if post_bench == 0 and turn > 1:
        _add(-0.20, "Empty_Bench_Vulnerability_Turn_GT_1")

    # Key Bench Engine Protection (Munkidori, Froslass, Snorunt)
    pre_key_count = sum(1 for pid in pre.get("my_bench_ids", []) if pid in KEY_BENCH_IDS)
    post_key_count = sum(1 for pid in post.get("my_bench_ids", []) if pid in KEY_BENCH_IDS)

    if post_key_count > pre_key_count:
        _add(0.15, "Key_Bench_Pokemon_Protected")
    elif post_key_count == 0 and pre_key_count > 0:
        _add(-0.20, "Key_Bench_Pokemon_Lost")

    # -------------------------------------------------------------------------
    # 2. STATE IMPROVEMENT: EVOLUTION & ATTACKER COMPLETION
    # -------------------------------------------------------------------------
    if action_type == 9 or (action_type == 7 and pre.get("played_card_id") == RARE_CANDY_ID):
        evolved_id = post.get("evolved_card_id", -1)
        if evolved_id == GRIMMSNARL_EX_ID or post.get("active_id") == GRIMMSNARL_EX_ID:
            _add(0.25, "Stage2_Grimmsnarl_ex_Completed")
        elif evolved_id == MORGREM_ID:
            _add(0.15, "Stage1_Morgrem_Evolution")
        elif evolved_id == FROSLASS_ID:
            _add(0.15, "Stage1_Froslass_Evolution")

    # -------------------------------------------------------------------------
    # 3. STATE IMPROVEMENT: ATTACK READINESS & PUNK UP AWARENESS
    # -------------------------------------------------------------------------
    pre_active_e = pre.get("active_energy", 0)
    post_active_e = post.get("active_energy", 0)

    if pre_active_e < 1 and post_active_e >= 1:
        if post.get("active_id") == GRIMMSNARL_EX_ID:
            _add(0.10, "Energy_Enables_1Cost_Attack_Grimmsnarl")  # Reduced due to Punk Up self-powering
        else:
            _add(0.15, "Energy_Enables_1Cost_Attack")
    elif pre_active_e < 2 and post_active_e >= 2:
        if post.get("active_id") == GRIMMSNARL_EX_ID:
            _add(0.10, "Energy_Enables_Main_Attack_Grimmsnarl")  # Reduced due to Punk Up self-powering
        else:
            _add(0.15, "Energy_Enables_Main_Attack")

    if post_active_e > 2 and action_type == 8:
        _add(-0.10, "Overcharging_Energy_Penalty")

    # -------------------------------------------------------------------------
    # 4. STATE IMPROVEMENT: MULTI-TIERED BOSS GUST TARGET VALUE
    # -------------------------------------------------------------------------
    if action_type == 7 and pre.get("played_card_id") == BOSS_ORDERS_ID:
        active_changed = (pre.get("opp_active_id") != post.get("opp_active_id"))
        opp_bench_count = len(pre.get("opp_bench_ids", []))
        new_active_is_basic = post.get("opp_active_is_basic", False)
        new_active_low_hp = post.get("opp_active_hp", 100) <= 60

        if active_changed and opp_bench_count > 0:
            if new_active_is_basic or new_active_low_hp:
                _add(0.22, "Successful_Boss_Gust_High_Value_Target")
            else:
                _add(0.18, "Successful_Boss_Gust")
        elif active_changed:
            _add(0.12, "Boss_Forced_Active_Reset")
        elif opp_bench_count >= 1:
            _add(0.08, "Boss_Targeted_Bench_Gust")
        else:
            _add(-0.08, "Wasted_Boss_Gust_No_Bench")

    # -------------------------------------------------------------------------
    # 5. STATE IMPROVEMENT: FROSLASS DAMAGE SPREAD & STADIUM SEEDING
    # -------------------------------------------------------------------------
    counters_placed = post.get("damage_counters_placed", 0) - pre.get("damage_counters_placed", 0)
    if counters_placed >= 3:
        _add(0.20, "Froslass_Damage_Spread_High_Impact")
    elif counters_placed >= 1:
        _add(0.12, "Froslass_Damage_Spread_Standard")

    if action_type == 7:
        played_id = pre.get("played_card_id", -1)
        if played_id == SPIKEMUTH_GYM_ID:
            _add(0.20, "Play_Spikemuth_Gym_Self_Damage_Seeding")
        elif played_id == UNFAIR_STAMP_ID:
            _add(0.15, "Play_Unfair_Stamp_Disruption")
        elif played_id == PETREL_ID:
            _add(0.12, "Play_Petrel_Supporter_Recovery")
        elif played_id == DAWN_ID:
            _add(0.12, "Play_Dawn_Mobility_Healing")

    elif action_type == 10:  # ABILITY
        _add(0.18, "Munkidori_Adrenaline_Brain_Damage_Move")

    # -------------------------------------------------------------------------
    # 6. STATE IMPROVEMENT: HAND RECOVERY
    # -------------------------------------------------------------------------
    if action_type == 7 and pre.get("hand_size", 5) <= 3 and post.get("hand_size", 0) >= 5:
        _add(0.15, "Hand_Recovery_From_Dead_Hand")

    # Bounded in range [-0.25, +0.25]
    clamped_r = max(-0.25, min(0.25, r_strategic))
    if debug_log is not None:
        debug_log.append(("Final_Clamped_Strategic_Reward", clamped_r))

    return clamped_r
