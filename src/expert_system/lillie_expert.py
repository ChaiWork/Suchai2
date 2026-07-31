"""
Lillie's Clefairy ex + Togekiss Specific Heuristic Expert Engine.
Implements opening priorities, evolution search, energy acceleration, supporter timing,
item management, tool placement, stadium control, and matchup-specific counterplay.
"""
from cg.api import CardType, OptionType, SelectContext
from src.expert_system.base_expert import extract_board_context

# Key Card ID Constants
CLEFAIRY_EX_ID       = 272   # Lillie's Clefairy ex (JTG #56)
TOGEKISS_ID          = 214   # Togekiss (SSP #72)
TOGEPI_ID            = 959   # Togepi (ASC #80 / SSP #70)
TOGETIC_ID           = 960   # Togetic (ASC #81 / SSP #71)
SMOOCHUM_ID          = 183   # Smoochum (SSP #75)
LATIAS_EX_ID         = 184   # Latias ex (SSP #76)
SHAYMIN_ID           = 343   # Shaymin (DRI #10)
PSYDUCK_ID           = 858   # Psyduck (ASC #39)
MIMIKYU_ID           = 434   # Team Rocket's Mimikyu (DRI #87)
IRON_BOULDER_ID      = 971   # Iron Boulder (SCR #71)

TELEPATHIC_ENERGY_ID = 19    # Telepath Psychic Energy (POR #87)
BASIC_PSYCHIC_ID     = 5     # Basic Psychic Energy (SVE #5)

MYSTERY_GARDEN_ID    = 1263  # Mystery Garden (MEG #122)
WONDROUS_PATCH_ID    = 1146  # Wondrous Patch (PFL #94)
LILLIES_PEARL_ID     = 1172  # Lillie's Pearl (JTG #151)
RARE_CANDY_ID        = 1079  # Rare Candy (SVI #191)
ULTRA_BALL_ID        = 1121  # Ultra Ball (SVI #196)
POKE_PAD_ID          = 1152  # Poké Pad (POR #81)
NIGHT_STRETCHER_ID   = 1097  # Night Stretcher (SFA #61)
AIR_BALLOON_ID       = 1174  # Air Balloon (BLK #79)
SWITCH_ID            = 1123  # Switch (SVI #194)
UNFAIR_STAMP_ID      = 1080  # Unfair Stamp (TWM #165)
ACCOMPANYING_FLUTE_ID= 1091  # Accompanying Flute (TWM #142)
POKEMON_CATCHER_ID   = 1124  # Pokémon Catcher (SVI #187)

COLRESS_TENACITY_ID  = 1194  # Colress's Tenacity (SFA #57)
HILDA_ID             = 1225  # Hilda (WHT #84)
LILLIES_DETERM_ID    = 1227  # Lillie's Determination (MEG #119)
MORTYS_CONVICTION_ID = 1187  # Morty's Conviction (TEF #155)
BOSS_ORDERS_ID       = 1182  # Boss's Orders (PAL #172)

# High-threat opponent evolution lines for matchup targeting
BENCH_SNIPE_KEYWORDS = ["dragapult", "alakazam", "kyurem", "greninja"]
EX_THREAT_KEYWORDS   = ["starmie", "archaludon", "slowking", "zoroark", "gholdengo", "grimmsnarl", "mewtwo"]


def evaluate_lillie_expert_bonus(obs, option, opponent_name: str) -> tuple[float, str]:
    """
    Evaluates heuristic action prior bonus for Lillie's Clefairy ex + Togekiss deck.
    """
    ctx = extract_board_context(obs)
    my_active_id = ctx["my_active_id"]
    opp_active_id = ctx["opp_active_id"]
    my_bench_ids = ctx.get("my_bench_ids", [])
    opp_bench_ids = ctx.get("opp_bench_ids", [])
    my_active_energy = ctx["my_active_energy"]
    my_bench_count = ctx["my_bench_count"]
    game_phase = ctx["game_phase"]
    turn = ctx["turn"]

    opt_type = getattr(option, "optionType", None)
    if opt_type is None and isinstance(option, dict):
        opt_type = option.get("optionType", -1)

    card_id = getattr(option, "cardId", -1) if not isinstance(option, dict) else option.get("cardId", -1)

    bonus = 0.0
    triggered = "None"
    opp_lower = (opponent_name or "").lower()

    # Matchup Flags
    is_bench_snipe_matchup = any(kw in opp_lower for kw in BENCH_SNIPE_KEYWORDS)
    is_ex_threat_matchup = any(kw in opp_lower for kw in EX_THREAT_KEYWORDS)

    # -------------------------------------------------------------------------
    # 1. PLAY ACTIONS (OptionType.PLAY / 7)
    # -------------------------------------------------------------------------
    if opt_type in (7, getattr(OptionType, "PLAY", 7)):

        # --- A. Basic Pokémon Setup ---
        if card_id in (CLEFAIRY_EX_ID, TOGEPI_ID, SMOOCHUM_ID, LATIAS_EX_ID, PSYDUCK_ID, SHAYMIN_ID):
            if is_bench_snipe_matchup and my_bench_count >= 3:
                bonus -= 0.15
                triggered = "Guard_Against_Dragapult_Bench_Snipe_Overbench"
            elif my_bench_count < 4:
                if card_id == CLEFAIRY_EX_ID:
                    bonus += 0.35
                    triggered = "Priority_Bench_Clefairy_ex"
                elif card_id == TOGEPI_ID:
                    bonus += 0.30
                    triggered = "Priority_Bench_Togepi"
                elif card_id == LATIAS_EX_ID:
                    bonus += 0.25
                    triggered = "Bench_Latias_ex_Free_Retreat_Aura"
                elif card_id == SMOOCHUM_ID and turn <= 2:
                    bonus += 0.30
                    triggered = "Early_Smoochum_Setup"
                else:
                    bonus += 0.15
                    triggered = f"Bench_Basic_{card_id}"

        # --- B. Supporter Priorities ---
        elif card_id == COLRESS_TENACITY_ID:
            bonus += 0.40
            triggered = "Colress_Tenacity_Search_Stadium_And_Energy"

        elif card_id == HILDA_ID:
            if TOGEPI_ID in my_bench_ids or TOGETIC_ID in my_bench_ids:
                bonus += 0.40
                triggered = "Hilda_Search_Togekiss_Evolution"
            else:
                bonus += 0.25
                triggered = "Hilda_Search_Energy"

        elif card_id == LILLIES_DETERM_ID:
            bonus += 0.30
            triggered = "Lillies_Determination_Hand_Refresh"

        elif card_id == MORTYS_CONVICTION_ID:
            if len(opp_bench_ids) >= 3:
                bonus += 0.35
                triggered = "Mortys_Conviction_Max_Draw"
            else:
                bonus += 0.20
                triggered = "Mortys_Conviction_Standard_Draw"

        elif card_id == BOSS_ORDERS_ID:
            if is_ex_threat_matchup or game_phase == "late":
                bonus += 0.35
                triggered = "Boss_Orders_Target_High_Value_Threat"
            else:
                bonus += 0.20
                triggered = "Boss_Orders_Gust"

        # --- C. Item Card Priorities ---
        elif card_id == RARE_CANDY_ID:
            if TOGEPI_ID in my_bench_ids:
                bonus += 0.45
                triggered = "Rare_Candy_Shortcut_To_Togekiss"

        elif card_id == ULTRA_BALL_ID:
            bonus += 0.30
            triggered = "Ultra_Ball_Search_Clefairy_ex_Or_Togekiss"

        elif card_id == WONDROUS_PATCH_ID:
            bonus += 0.35
            triggered = "Wondrous_Patch_Energy_Ramp"

        elif card_id == MYSTERY_GARDEN_ID:
            bonus += 0.35
            triggered = "Mystery_Garden_Stadium_Play"

        elif card_id == LILLIES_PEARL_ID:
            if my_active_id == CLEFAIRY_EX_ID or CLEFAIRY_EX_ID in my_bench_ids:
                bonus += 0.30
                triggered = "Equip_Lillies_Pearl_To_Clefairy_ex"

        elif card_id == NIGHT_STRETCHER_ID:
            bonus += 0.25
            triggered = "Night_Stretcher_Recovery"

        elif card_id == POKE_PAD_ID:
            # Rule Box Warning: Poké Pad only searches non-Rule Box Pokémon!
            bonus += 0.15
            triggered = "PokePad_Search_Non_Rulebox"

        elif card_id in (AIR_BALLOON_ID, SWITCH_ID):
            bonus += 0.20
            triggered = "Mobility_Switch_AirBalloon"

        elif card_id == UNFAIR_STAMP_ID:
            if game_phase in ("mid", "late"):
                bonus += 0.35
                triggered = "Unfair_Stamp_Disruption"

        elif card_id == ACCOMPANYING_FLUTE_ID:
            bonus += 0.15
            triggered = "Accompanying_Flute_Disruption"

        elif card_id == POKEMON_CATCHER_ID:
            bonus += 0.20
            triggered = "Pokemon_Catcher_Coin_Gust"

    # -------------------------------------------------------------------------
    # 2. ENERGY ATTACHMENT ACTIONS (OptionType.ATTACH / 8)
    # -------------------------------------------------------------------------
    elif opt_type in (8, getattr(OptionType, "ATTACH", 8)):
        if card_id == TELEPATHIC_ENERGY_ID:
            bonus += 0.40
            triggered = "Telepathic_Energy_Attach_Bench_Search_Bonus"
        elif my_active_id == CLEFAIRY_EX_ID and my_active_energy < 3:
            bonus += 0.30
            triggered = "Attach_Psychic_Energy_Active_Clefairy_ex"
        elif my_active_id == TOGEKISS_ID:
            bonus += 0.25
            triggered = "Attach_Energy_Togekiss"
        else:
            bonus += 0.15
            triggered = "Attach_Energy_Bench"

    # -------------------------------------------------------------------------
    # 3. EVOLUTION ACTIONS (OptionType.EVOLVE / 9)
    # -------------------------------------------------------------------------
    elif opt_type in (9, getattr(OptionType, "EVOLVE", 9)):
        if card_id == TOGEKISS_ID:
            bonus += 0.45
            triggered = "Evolve_To_Togekiss"
        elif card_id == TOGETIC_ID:
            bonus += 0.25
            triggered = "Evolve_To_Togetic"

    # -------------------------------------------------------------------------
    # 4. ATTACK ACTIONS (OptionType.ATTACK / 10)
    # -------------------------------------------------------------------------
    elif opt_type in (10, getattr(OptionType, "ATTACK", 10)):
        bonus += 0.45
        triggered = "Lillie_Deck_Attack_Execution"

    # Bound bonus range [-0.50, +0.45]
    bonus = max(-0.50, min(0.45, bonus))
    return bonus, triggered
