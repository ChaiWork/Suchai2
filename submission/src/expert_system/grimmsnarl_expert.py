"""
Marnie's Grimmsnarl ex + Munkidori Strategic Expert Engine.
Ground Truth: Matches 60-card CSV list (deckgrimnai.csv).
Acts purely as a lightweight action prior provider in range [-0.15, +0.20].
Centralized configuration parameters eliminate magic numbers and prevent MCTS over-dominance.
"""
from cg.api import OptionType
from src.expert_system.base_expert import extract_board_context

# Card ID Constants (Marnie's Grimmsnarl ex Deck)
IMPIDIMP_ID         = 646   # Marnie's Impidimp
MORGREM_ID          = 647   # Marnie's Morgrem
GRIMMSNARL_EX_ID    = 648   # Marnie's Grimmsnarl ex
MUNKIDORI_ID        = 112   # Munkidori
FROSLASS_ID         = 104   # Froslass
SNORUNT_ID          = 860   # Snorunt

BASIC_DARK_ENERGY_ID= 7     # Basic Dark Energy

SPIKEMUTH_GYM_ID    = 1259  # Spikemuth Gym
RARE_CANDY_ID       = 1079  # Rare Candy
POFFIN_ID           = 1086  # Buddy-Buddy Poffin
POKE_PAD_ID         = 1152  # Poké Pad
POKEGEAR_ID         = 1122  # Pokégear 3.0
NIGHT_STRETCHER_ID  = 1097  # Night Stretcher
UNFAIR_STAMP_ID     = 1080  # Unfair Stamp
LUCKY_HELMET_ID     = 1156  # Lucky Helmet

PETREL_ID           = 1219  # Team Rocket's Petrel
LILLIES_DETERM_ID   = 1227  # Lillie's Determination
BOSS_ORDERS_ID      = 1182  # Boss's Orders
DAWN_ID             = 1231  # Dawn

SUPPORTER_IDS = frozenset({1219, 1227, 1182, 1231})

# -------------------------------------------------------------------------
# CENTRALIZED EXPERT PRIOR CONFIGURATION (NO MAGIC NUMBERS)
# -------------------------------------------------------------------------
EXPERT_PRIOR_CONFIG = {
    "POFFIN_MULTI_BASIC": 0.18,
    "POFFIN_SINGLE_BASIC": 0.12,
    "POFFIN_LOW_VAL": 0.04,
    "POKEGEAR_EARLY": 0.15,
    "RARE_CANDY_READY": 0.20,
    "RARE_CANDY_SHORTCUT": 0.15,
    "RARE_CANDY_EARLY": 0.10,
    "SPIKEMUTH_HIGH_VAL": 0.18,
    "SPIKEMUTH_STANDARD": 0.10,
    "UNFAIR_STAMP_HUNT": 0.18,
    "UNFAIR_STAMP_DIG": 0.14,
    "UNFAIR_STAMP_STD": 0.08,
    "MUNKIDORI_1ST": 0.15,
    "MUNKIDORI_2ND": 0.14,
    "MUNKIDORI_3RD_CRITICAL": 0.20,
    "MUNKIDORI_4TH_LOW": 0.04,
    "IMPIDIMP_BASIC": 0.12,
    "EARLY_SUPPORTER_SETUP": 0.15,
    "SUPPORTER_REFRESH": 0.10,
    "BOSS_MUNKIDORI_WAR": 0.18,
    "BOSS_GUST_DISRUPTION": 0.14,
    "BOSS_GUST_SINGLE": 0.10,
    "POKEPAD_RECYCLE": 0.08,
    "NIGHT_STRETCHER_COMBO": 0.16,
    "NIGHT_STRETCHER_KEY_POKEMON": 0.12,
    "NIGHT_STRETCHER_ENERGY": 0.10,
    "ATTACH_ACTIVE_DARK": 0.12,
    "ATTACH_MUNKIDORI_DARK": 0.10,
    "STAGE2_GRIMMSNARL_COMEBACK": 0.20,
    "STAGE2_GRIMMSNARL_AHEAD": 0.12,
    "STAGE1_MORGREM": 0.10,
    "FROSLASS_EARLY": 0.14,
    "FROSLASS_STD": 0.10,
    "MUNKIDORI_ABILITY": 0.15,
    "ATTACK_HIGH_VALUE_KO": 0.20,
    "ATTACK_LOW_VALUE_KO": 0.14,
    "ATTACK_POSTPONE_SUPPORTER": 0.08,
    "ATTACK_EXECUTION": 0.15,
}

# Matchup archetype keywords
DRAGAPULT_KEYWORDS = ("dragapult", "drakloak", "dreepy")
CHARIZARD_KEYWORDS = ("charizard", "charmeleon", "charmander")
GARDEVOIR_KEYWORDS = ("gardevoir", "kirlia", "ralts")
CRUSTLE_KEYWORDS   = ("crustle", "dwebble", "abomasnow")


def evaluate_grimmsnarl_expert_bonus(obs, option, opponent_name: str) -> tuple[float, str]:
    """
    Evaluates context-aware heuristic action prior bonus for Marnie's Grimmsnarl ex deck.
    All bonus values are strictly bounded in [-0.15, +0.20] to prevent MCTS over-dominance.
    """
    ctx = extract_board_context(obs)
    my_active_id = ctx["my_active_id"]
    opp_active_id = ctx["opp_active_id"]
    my_bench_ids = ctx.get("my_bench_ids", [])
    opp_bench_ids = ctx.get("opp_bench_ids", [])
    my_bench_count = ctx["my_bench_count"]
    turn = ctx["turn"]

    my_prizes_taken = 6 - ctx.get("my_prizes", 6)
    opp_prizes_taken = 6 - ctx.get("opp_prizes", 6)

    opt_type_raw = getattr(option, "type", getattr(option, "optionType", None))
    if opt_type_raw is None and isinstance(option, dict):
        opt_type_raw = option.get("type", option.get("optionType", -1))

    opt_type_str = str(opt_type_raw).upper()
    IS_PLAY   = (opt_type_raw in (7, getattr(OptionType, "PLAY", 7))) or ("PLAY" in opt_type_str)
    IS_ATTACH = (opt_type_raw in (8, getattr(OptionType, "ATTACH", 8))) or ("ATTACH" in opt_type_str)
    IS_EVOLVE = (opt_type_raw in (9, getattr(OptionType, "EVOLVE", 9))) or ("EVOLVE" in opt_type_str)
    IS_ABILITY= (opt_type_raw in (10, getattr(OptionType, "ABILITY", 10))) or ("ABILITY" in opt_type_str)
    IS_ATTACK = (opt_type_raw in (13, getattr(OptionType, "ATTACK", 13))) or ("ATTACK" in opt_type_str)

    card_id = getattr(option, "cardId", -1) if not isinstance(option, dict) else option.get("cardId", -1)
    if card_id is None or card_id in (-1, 0):
        opt_index = getattr(option, "index", -1) if not isinstance(option, dict) else option.get("index", -1)
        my_ps = ctx.get("my_ps")
        if my_ps is not None and opt_index is not None and opt_index >= 0:
            hand = getattr(my_ps, "hand", []) if not isinstance(my_ps, dict) else my_ps.get("hand", [])
            if 0 <= opt_index < len(hand):
                card = hand[opt_index]
                if card is not None:
                    card_id = card.get("cardId", card.get("id", -1)) if isinstance(card, dict) else getattr(card, "cardId", getattr(card, "id", -1))

    bonus = 0.0
    triggered = "None"
    opp_lower = (opponent_name or "").lower()

    # Matchup context flags
    is_dragapult_matchup = any(kw in opp_lower for kw in DRAGAPULT_KEYWORDS)
    is_charizard_matchup = any(kw in opp_lower for kw in CHARIZARD_KEYWORDS)
    is_gardevoir_matchup = any(kw in opp_lower for kw in GARDEVOIR_KEYWORDS)
    is_crustle_matchup   = any(kw in opp_lower for kw in CRUSTLE_KEYWORDS)

    # -------------------------------------------------------------------------
    # 1. CORE PLAY ACTIONS
    # -------------------------------------------------------------------------
    if IS_PLAY:
        if card_id == POFFIN_ID and turn <= 3:
            my_hand_ids = set(ctx.get("my_hand_ids", []))
            field_ids = set(my_bench_ids + [my_active_id] + ctx.get("my_discard_ids", []))
            targets = {IMPIDIMP_ID, MUNKIDORI_ID, SNORUNT_ID}
            missing_targets = [tid for tid in targets if tid not in field_ids and tid not in my_hand_ids]

            if len(missing_targets) >= 2:
                bonus += EXPERT_PRIOR_CONFIG["POFFIN_MULTI_BASIC"]
                triggered = "Poffin_Search_Multiple_Basics_High_Value"
            elif len(missing_targets) == 1:
                bonus += EXPERT_PRIOR_CONFIG["POFFIN_SINGLE_BASIC"]
                triggered = "Poffin_Search_Single_Basic"
            else:
                bonus += EXPERT_PRIOR_CONFIG["POFFIN_LOW_VAL"]
                triggered = "Poffin_Low_Value_Use"

        elif card_id == POKEGEAR_ID and turn <= 2:
            bonus += EXPERT_PRIOR_CONFIG["POKEGEAR_EARLY"]
            triggered = "PokeGear_Early_Consistency_Engine"

        elif card_id == RARE_CANDY_ID and IMPIDIMP_ID in my_bench_ids:
            my_hand_ids = ctx.get("my_hand_ids", [])
            has_dark_energy = BASIC_DARK_ENERGY_ID in my_hand_ids
            has_attack_enablement = has_dark_energy or (my_active_id == IMPIDIMP_ID and turn >= 2)
            if has_attack_enablement:
                bonus += EXPERT_PRIOR_CONFIG["RARE_CANDY_READY"]
                triggered = "Rare_Candy_Grimmsnarl_Ready_To_Attack"
            elif turn >= 3:
                bonus += EXPERT_PRIOR_CONFIG["RARE_CANDY_SHORTCUT"]
                triggered = "Rare_Candy_Grimmsnarl_Shortcut"
            else:
                bonus += EXPERT_PRIOR_CONFIG["RARE_CANDY_EARLY"]
                triggered = "Rare_Candy_Grimmsnarl_Early_Evolution"

        elif card_id == SPIKEMUTH_GYM_ID:
            opp_bench_count = len(opp_bench_ids)
            if opp_bench_count >= 2 or (turn > 3 and opp_bench_count > 0):
                bonus += EXPERT_PRIOR_CONFIG["SPIKEMUTH_HIGH_VAL"]
                triggered = "Spikemuth_Gym_High_Value_Target"
            else:
                bonus += EXPERT_PRIOR_CONFIG["SPIKEMUTH_STANDARD"]
                triggered = "Spikemuth_Gym_Standard_Play"

        elif card_id == UNFAIR_STAMP_ID:
            my_hand_ids = ctx.get("my_hand_ids", [])
            has_spikemuth = (SPIKEMUTH_GYM_ID in my_bench_ids or SPIKEMUTH_GYM_ID in my_hand_ids)
            needs_spikemuth = not has_spikemuth
            needs_night_stretcher = NIGHT_STRETCHER_ID not in my_hand_ids
            needs_supporter = not any(cid in SUPPORTER_IDS for cid in my_hand_ids)

            if is_gardevoir_matchup and turn <= 3:
                bonus += EXPERT_PRIOR_CONFIG["UNFAIR_STAMP_HUNT"]
                triggered = "Unfair_Stamp_Gardevoir_Kirlia_Disruption"
            elif needs_spikemuth and turn <= 3:
                bonus += EXPERT_PRIOR_CONFIG["UNFAIR_STAMP_HUNT"]
                triggered = "Unfair_Stamp_Spikemuth_Gym_Hunt"
            elif needs_night_stretcher or needs_supporter:
                bonus += EXPERT_PRIOR_CONFIG["UNFAIR_STAMP_DIG"]
                triggered = "Unfair_Stamp_Resource_Dig"
            else:
                bonus += EXPERT_PRIOR_CONFIG["UNFAIR_STAMP_STD"]
                triggered = "Unfair_Stamp_Standard_Play"

        elif card_id == MUNKIDORI_ID and my_bench_count < 5:
            munkidori_count = my_bench_ids.count(MUNKIDORI_ID)
            if munkidori_count == 0:
                bonus += EXPERT_PRIOR_CONFIG["MUNKIDORI_1ST"]
                triggered = "Munkidori_First_Bench_Priority"
            elif munkidori_count == 1:
                bonus += EXPERT_PRIOR_CONFIG["MUNKIDORI_2ND"]
                triggered = "Munkidori_Second_Bench_Priority"
            elif munkidori_count == 2:
                bonus += EXPERT_PRIOR_CONFIG["MUNKIDORI_3RD_CRITICAL"]
                triggered = "Munkidori_Third_Bench_Critical"
            else:
                bonus += EXPERT_PRIOR_CONFIG["MUNKIDORI_4TH_LOW"]
                triggered = "Munkidori_Fourth_Bench_Low_Priority"

        elif card_id == IMPIDIMP_ID and my_bench_count < 5:
            bonus += EXPERT_PRIOR_CONFIG["IMPIDIMP_BASIC"]
            triggered = "Grimmsnarl_Bench_Basic_Impidimp"

        elif card_id in (PETREL_ID, LILLIES_DETERM_ID):
            my_hand_ids = ctx.get("my_hand_ids", [])
            if turn <= 2 and (SPIKEMUTH_GYM_ID not in my_bench_ids and RARE_CANDY_ID not in my_hand_ids):
                bonus += EXPERT_PRIOR_CONFIG["EARLY_SUPPORTER_SETUP"]
                triggered = "Early_Supporter_Setup_Priority"
            else:
                bonus += EXPERT_PRIOR_CONFIG["SUPPORTER_REFRESH"]
                triggered = "Grimmsnarl_Supporter_Draw_Refresh"

        elif card_id == BOSS_ORDERS_ID and len(opp_bench_ids) > 0:
            opp_munkidori_count = sum(1 for pid in opp_bench_ids if pid == MUNKIDORI_ID)
            if opp_munkidori_count > 0:
                bonus += EXPERT_PRIOR_CONFIG["BOSS_MUNKIDORI_WAR"]
                triggered = "Boss_Gust_Munkidori_War_Priority"
            elif is_charizard_matchup:
                bonus += EXPERT_PRIOR_CONFIG["BOSS_MUNKIDORI_WAR"]
                triggered = "Boss_Gust_Charizard_Pre_Evo_Target"
            elif len(opp_bench_ids) >= 2:
                bonus += EXPERT_PRIOR_CONFIG["BOSS_GUST_DISRUPTION"]
                triggered = "Grimmsnarl_Boss_Gust_Disruption"
            else:
                bonus += EXPERT_PRIOR_CONFIG["BOSS_GUST_SINGLE"]
                triggered = "Boss_Gust_Single_Target"

        elif card_id == POKE_PAD_ID:
            bonus += EXPERT_PRIOR_CONFIG["POKEPAD_RECYCLE"]
            triggered = "Grimmsnarl_PokePad_Recycle_Supporter"

        elif card_id == NIGHT_STRETCHER_ID:
            my_discard_ids = ctx.get("my_discard_ids", [])
            has_key_bench_in_discard = any(cid in (MUNKIDORI_ID, FROSLASS_ID, IMPIDIMP_ID) for cid in my_discard_ids)
            has_dark_energy_in_discard = BASIC_DARK_ENERGY_ID in my_discard_ids

            if has_key_bench_in_discard and has_dark_energy_in_discard:
                bonus += EXPERT_PRIOR_CONFIG["NIGHT_STRETCHER_COMBO"]
                triggered = "Night_Stretcher_Recovery_Pokemon_And_Energy"
            elif has_key_bench_in_discard:
                bonus += EXPERT_PRIOR_CONFIG["NIGHT_STRETCHER_KEY_POKEMON"]
                triggered = "Night_Stretcher_Recovery_Key_Pokemon"
            elif has_dark_energy_in_discard:
                bonus += EXPERT_PRIOR_CONFIG["NIGHT_STRETCHER_ENERGY"]
                triggered = "Night_Stretcher_Recovery_Energy"
            else:
                bonus += EXPERT_PRIOR_CONFIG["POFFIN_LOW_VAL"]
                triggered = "Night_Stretcher_Standard_Recovery"

    # -------------------------------------------------------------------------
    # 2. ENERGY ATTACHMENT
    # -------------------------------------------------------------------------
    elif IS_ATTACH:
        if card_id == BASIC_DARK_ENERGY_ID:
            if my_active_id in (IMPIDIMP_ID, MORGREM_ID, GRIMMSNARL_EX_ID):
                bonus += EXPERT_PRIOR_CONFIG["ATTACH_ACTIVE_DARK"]
                triggered = "Grimmsnarl_Dark_Energy_Attach_Active"
            elif MUNKIDORI_ID in my_bench_ids:
                bonus += EXPERT_PRIOR_CONFIG["ATTACH_MUNKIDORI_DARK"]
                triggered = "Grimmsnarl_Dark_Energy_Attach_Munkidori"

    # -------------------------------------------------------------------------
    # 3. EVOLUTION
    # -------------------------------------------------------------------------
    elif IS_EVOLVE:
        if card_id == GRIMMSNARL_EX_ID:
            if my_prizes_taken <= opp_prizes_taken:
                bonus += EXPERT_PRIOR_CONFIG["STAGE2_GRIMMSNARL_COMEBACK"]
                triggered = "Grimmsnarl_ex_Evolution_Comeback_Setup"
            else:
                bonus += EXPERT_PRIOR_CONFIG["STAGE2_GRIMMSNARL_AHEAD"]
                triggered = "Grimmsnarl_ex_Evolution_While_Ahead"
        elif card_id == MORGREM_ID:
            bonus += EXPERT_PRIOR_CONFIG["STAGE1_MORGREM"]
            triggered = "Morgrem_Stage1_Evolution"
        elif card_id == FROSLASS_ID:
            if turn <= 3 or is_dragapult_matchup:
                bonus += EXPERT_PRIOR_CONFIG["FROSLASS_EARLY"]
                triggered = "Froslass_Early_Evolution_Priority"
            else:
                bonus += EXPERT_PRIOR_CONFIG["FROSLASS_STD"]
                triggered = "Froslass_Stage1_Evolution"

    # -------------------------------------------------------------------------
    # 4. ABILITY & ATTACK EXECUTION
    # -------------------------------------------------------------------------
    elif IS_ABILITY:
        bonus += EXPERT_PRIOR_CONFIG["MUNKIDORI_ABILITY"]
        if is_crustle_matchup:
            bonus += 0.05
            triggered = "Munkidori_Adrenaline_Brain_Crustle_Bypass"
        else:
            triggered = "Munkidori_Adrenaline_Brain_Move_Damage"

    elif IS_ATTACK:
        my_hand_ids = ctx.get("my_hand_ids", [])
        has_unplayed_supporter = any(cid in SUPPORTER_IDS for cid in my_hand_ids)
        has_unplayed_poffin = POFFIN_ID in my_hand_ids
        can_ko = ctx.get("can_ko_active", False)
        opp_active_is_ex = ctx.get("opp_active_is_ex", False)

        if can_ko:
            if opp_active_is_ex or my_prizes_taken <= opp_prizes_taken:
                bonus += EXPERT_PRIOR_CONFIG["ATTACK_HIGH_VALUE_KO"]
                triggered = "Grimmsnarl_Attack_For_High_Value_KO"
            else:
                bonus += EXPERT_PRIOR_CONFIG["ATTACK_LOW_VALUE_KO"]
                triggered = "Grimmsnarl_Attack_For_Low_Value_KO"
        elif has_unplayed_supporter or has_unplayed_poffin:
            bonus += EXPERT_PRIOR_CONFIG["ATTACK_POSTPONE_SUPPORTER"]
            triggered = "Grimmsnarl_Attack_Postpone_For_Supporter"
        else:
            bonus += EXPERT_PRIOR_CONFIG["ATTACK_EXECUTION"]
            triggered = "Grimmsnarl_Attack_Execution"

    # Bound bonus range strictly in [-0.15, +0.20]
    bonus = max(-0.15, min(0.20, bonus))
    return bonus, triggered
