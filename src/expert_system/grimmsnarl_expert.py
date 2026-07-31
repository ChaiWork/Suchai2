"""
Marnie's Grimmsnarl ex + Munkidori Strategic Expert Engine.
Implements heuristic action prior bonuses for:
- Fast Rare Candy into Stage 2 Grimmsnarl ex with attack readiness
- Context-aware Spikemuth Gym placement and replacement counterplay
- Munkidori 3-bench saturation priority for 90 damage movement
- Gust targeting on opponent pre-evolution basics
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


def evaluate_grimmsnarl_expert_bonus(obs, option, opponent_name: str) -> tuple[float, str]:
    """
    Evaluates context-aware heuristic action prior bonus for Marnie's Grimmsnarl ex deck.
    """
    ctx = extract_board_context(obs)
    my_active_id = ctx["my_active_id"]
    opp_active_id = ctx["opp_active_id"]
    my_bench_ids = ctx.get("my_bench_ids", [])
    opp_bench_ids = ctx.get("opp_bench_ids", [])
    my_bench_count = ctx["my_bench_count"]
    turn = ctx["turn"]

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

    # -------------------------------------------------------------------------
    # 1. CORE PLAY ACTIONS
    # -------------------------------------------------------------------------
    if IS_PLAY:
        if card_id == POFFIN_ID and turn <= 3:
            bonus += 0.40
            triggered = "Poffin_Search_Impidimp_And_Munkidori"

        elif card_id == RARE_CANDY_ID and IMPIDIMP_ID in my_bench_ids:
            my_hand_ids = ctx.get("my_hand_ids", [])
            has_dark_energy = BASIC_DARK_ENERGY_ID in my_hand_ids
            has_attack_enablement = has_dark_energy or (my_active_id == IMPIDIMP_ID and turn >= 2)
            if has_attack_enablement:
                bonus += 0.50
                triggered = "Rare_Candy_Grimmsnarl_Ready_To_Attack"
            elif turn >= 3:
                bonus += 0.45
                triggered = "Rare_Candy_Grimmsnarl_Shortcut"
            else:
                bonus += 0.35
                triggered = "Rare_Candy_Grimmsnarl_Early_Evolution"

        elif card_id == SPIKEMUTH_GYM_ID:
            opp_bench_count = len(opp_bench_ids)
            if opp_bench_count >= 2 or (turn > 3 and opp_bench_count > 0):
                bonus += 0.45
                triggered = "Spikemuth_Gym_High_Value_Target"
            else:
                bonus += 0.30
                triggered = "Spikemuth_Gym_Standard_Play"

        elif card_id == MUNKIDORI_ID and my_bench_count < 5:
            munkidori_count = my_bench_ids.count(MUNKIDORI_ID)
            if munkidori_count < 3:
                bonus += 0.40
                triggered = "Munkidori_Bench_Saturation_Priority"
            else:
                bonus += 0.30
                triggered = "Munkidori_Bench_Basic"

        elif card_id == IMPIDIMP_ID and my_bench_count < 5:
            bonus += 0.35
            triggered = "Grimmsnarl_Bench_Basic_Impidimp"

        elif card_id in (PETREL_ID, LILLIES_DETERM_ID):
            bonus += 0.35
            triggered = "Grimmsnarl_Supporter_Draw_Refresh"

        elif card_id == BOSS_ORDERS_ID and len(opp_bench_ids) > 0:
            bonus += 0.38
            triggered = "Grimmsnarl_Boss_Gust_Disruption"

        elif card_id == POKE_PAD_ID:
            bonus += 0.30
            triggered = "Grimmsnarl_PokePad_Recycle_Supporter"

        elif card_id == NIGHT_STRETCHER_ID:
            bonus += 0.25
            triggered = "Grimmsnarl_Night_Stretcher_Recovery"

    # -------------------------------------------------------------------------
    # 2. ENERGY ATTACHMENT
    # -------------------------------------------------------------------------
    elif IS_ATTACH:
        if card_id == BASIC_DARK_ENERGY_ID:
            if my_active_id in (IMPIDIMP_ID, MORGREM_ID, GRIMMSNARL_EX_ID):
                bonus += 0.35
                triggered = "Grimmsnarl_Dark_Energy_Attach_Active"
            elif MUNKIDORI_ID in my_bench_ids:
                bonus += 0.30
                triggered = "Grimmsnarl_Dark_Energy_Attach_Munkidori"

    # -------------------------------------------------------------------------
    # 3. EVOLUTION
    # -------------------------------------------------------------------------
    elif IS_EVOLVE:
        if card_id == GRIMMSNARL_EX_ID:
            bonus += 0.45
            triggered = "Grimmsnarl_ex_Stage2_Evolution"
        elif card_id == MORGREM_ID:
            bonus += 0.30
            triggered = "Morgrem_Stage1_Evolution"
        elif card_id == FROSLASS_ID:
            bonus += 0.30
            triggered = "Froslass_Stage1_Evolution"

    # -------------------------------------------------------------------------
    # 4. ABILITY & ATTACK EXECUTION
    # -------------------------------------------------------------------------
    elif IS_ABILITY:
        bonus += 0.40
        triggered = "Munkidori_Adrenaline_Brain_Move_Damage"

    elif IS_ATTACK:
        my_hand_ids = ctx.get("my_hand_ids", [])
        has_unplayed_supporter = any(cid in SUPPORTER_IDS for cid in my_hand_ids)
        has_unplayed_poffin = POFFIN_ID in my_hand_ids
        can_ko = ctx.get("can_ko_active", False)

        if can_ko:
            bonus += 0.45
            triggered = "Grimmsnarl_Attack_For_KO"
        elif has_unplayed_supporter or has_unplayed_poffin:
            bonus += 0.20
            triggered = "Grimmsnarl_Attack_Postpone_For_Supporter"
        else:
            bonus += 0.40
            triggered = "Grimmsnarl_Attack_Execution"

    # Bound bonus range [-0.50, +0.45]
    bonus = max(-0.50, min(0.45, bonus))
    return bonus, triggered
