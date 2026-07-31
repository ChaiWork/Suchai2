"""
Lillie's Clefairy ex + Togekiss Specific Heuristic Expert Engine.
Implements opening priorities, evolution search, energy acceleration, supporter timing,
item management, tool placement, stadium control, and matchup-specific counterplay vs:
- Dragapult ex (Spread / Snipe)
- Crustle (Defensive Wall / Stall)
- Lucario (Fast Fighting Aggro)
- Grimmsnarl ex (High-HP Dark Tank)
"""
from cg.api import CardType, OptionType, SelectContext
from src.expert_system.base_expert import extract_board_context

# Key Card ID Constants (Lillie Deck)
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

# Specific Matchup Target Card ID Sets
DRAGAPULT_LINE_IDS   = {336, 337, 338}   # Dreepy, Drakloak, Dragapult ex
CRUSTLE_LINE_IDS     = {344, 345}        # Dwebble, Crustle
LUCARIO_LINE_IDS     = {447, 448}        # Riolu, Lucario
GRIMMSNARL_LINE_IDS  = {646, 647, 648}   # Impidimp, Morgrem, Grimmsnarl ex
# Card sets for ability counter checks
DRAGON_CARD_IDS  = {336, 337, 338, 553, 554, 804, 835, 646, 647} # Dreepy, Dragapult, Archaludon, Raging Bolt, Roaring Moon
SPREAD_DECKS     = {"dragapult", "lost box", "greninja", "abomasnow", "dipplin"}
STALL_DECKS      = {"snorlax", "stall", "control", "pidgeot control", "crustle", "iono"}
SELF_KO_CARDS    = {101, 102, 205, 206} # Voltorb, Electrode ex, Pineco, Forretress ex


def evaluate_lillie_expert_bonus(obs, option, opponent_name: str) -> tuple[float, str]:
    """
    Evaluates heuristic action prior bonus for Lillie's Clefairy ex + Togekiss deck.
    Includes explicit ability-based counter logic for Fairy Zone, Skyliner, Wonder Kiss,
    Flower Curtain, and Damp against major competitive archetypes.
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

    opt_type_raw = getattr(option, "type", getattr(option, "optionType", None))
    if opt_type_raw is None and isinstance(option, dict):
        opt_type_raw = option.get("type", option.get("optionType", -1))

    opt_type_str = str(opt_type_raw).upper()
    IS_PLAY   = (opt_type_raw in (7, getattr(OptionType, "PLAY", 7))) or ("PLAY" in opt_type_str)
    IS_ATTACH = (opt_type_raw in (8, getattr(OptionType, "ATTACH", 8))) or ("ATTACH" in opt_type_str)
    IS_EVOLVE = (opt_type_raw in (9, getattr(OptionType, "EVOLVE", 9))) or ("EVOLVE" in opt_type_str)
    IS_ABILITY= (opt_type_raw in (10, getattr(OptionType, "ABILITY", 10))) or ("ABILITY" in opt_type_str)
    IS_RETREAT= (opt_type_raw in (12, getattr(OptionType, "RETREAT", 12))) or ("RETREAT" in opt_type_str)
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

    # Board Ability Flags
    clefairy_in_play = (my_active_id == CLEFAIRY_EX_ID) or (CLEFAIRY_EX_ID in my_bench_ids)
    latias_in_play   = (my_active_id == LATIAS_EX_ID) or (LATIAS_EX_ID in my_bench_ids)
    togekiss_in_play = (my_active_id == TOGEKISS_ID) or (TOGEKISS_ID in my_bench_ids)
    shaymin_in_play  = (my_active_id == SHAYMIN_ID) or (SHAYMIN_ID in my_bench_ids)
    psyduck_in_play  = (my_active_id == PSYDUCK_ID) or (PSYDUCK_ID in my_bench_ids)

    # -------------------------------------------------------------------------
    # 0. MATCHUP IDENTIFICATION & FLAGS
    # -------------------------------------------------------------------------
    is_dragapult = "dragapult" in opp_lower or any(cid in DRAGAPULT_LINE_IDS for cid in [opp_active_id] + opp_bench_ids)
    is_crustle   = "crustle" in opp_lower or any(cid in CRUSTLE_LINE_IDS for cid in [opp_active_id] + opp_bench_ids)
    is_lucario   = "lucario" in opp_lower or any(cid in LUCARIO_LINE_IDS for cid in [opp_active_id] + opp_bench_ids)
    is_grimmsnarl= "grimmsnarl" in opp_lower or any(cid in GRIMMSNARL_LINE_IDS for cid in [opp_active_id] + opp_bench_ids)

    # -------------------------------------------------------------------------
    # 0b. ABILITY-BASED COUNTER ENGINE
    # -------------------------------------------------------------------------
    # A. Fairy Zone (Clefairy ex) vs Dragon Archetypes
    is_dragon_opp = any(cid in DRAGON_CARD_IDS for cid in [opp_active_id] + opp_bench_ids) or any(d in opp_lower for d in ("dragon", "dragapult", "roaring", "bolt", "archaludon", "regidrago", "goodra", "giratina"))
    if is_dragon_opp:
        if IS_PLAY and card_id == CLEFAIRY_EX_ID:
            bonus += 0.20
            triggered = "Ability_Fairy_Zone_Dragon_Counter_Setup"
        elif clefairy_in_play and IS_ATTACK and my_active_id == CLEFAIRY_EX_ID:
            bonus += 0.20
            triggered = "Ability_Fairy_Zone_Dragon_Weakness_Attack"
        elif clefairy_in_play and IS_PLAY and card_id == ACCOMPANYING_FLUTE_ID:
            bonus += 0.18
            triggered = "Ability_Fairy_Zone_Flute_Bench_Ramp"

    # B. Skyliner (Latias ex) vs Stall / Control / General Mobility
    is_stall_opp = any(s in opp_lower for s in STALL_DECKS)
    if is_stall_opp:
        if IS_PLAY and card_id == LATIAS_EX_ID:
            bonus += 0.20
            triggered = "Ability_Skyliner_Stall_Break_Priority"
    elif not latias_in_play and IS_PLAY and card_id == LATIAS_EX_ID and my_bench_count < 5:
        bonus += 0.12
        triggered = "Ability_Skyliner_General_Setup_Bonus"

    # C. Flower Curtain (Shaymin) vs Spread / Bench Damage Decks
    is_spread_opp = any(s in opp_lower for s in SPREAD_DECKS) or any(cid in (336, 337, 338) for cid in [opp_active_id] + opp_bench_ids)
    if is_spread_opp:
        if not shaymin_in_play and IS_PLAY and card_id == SHAYMIN_ID:
            bonus += 0.18
            triggered = "Ability_Flower_Curtain_Spread_Guard"

    # D. Wonder Kiss (Togekiss) Extra Prize Coin Flip Ramp
    if not togekiss_in_play and IS_EVOLVE and card_id == TOGEKISS_ID:
        bonus += 0.16
        triggered = "Ability_Wonder_Kiss_Evolution_Bonus"
    elif togekiss_in_play and IS_ATTACK:
        bonus += 0.14
        triggered = "Ability_Wonder_Kiss_Prize_Boost_Attack"

    # E. Damp (Psyduck) vs Self-KO Energy Acceleration Engines
    is_self_ko_opp = any(cid in SELF_KO_CARDS for cid in [opp_active_id] + opp_bench_ids)
    if is_self_ko_opp:
        if not psyduck_in_play and IS_PLAY and card_id == PSYDUCK_ID:
            bonus += 0.14
            triggered = "Ability_Damp_Self_KO_Lockout"

    # -------------------------------------------------------------------------
    # 1. MATCHUP 1: VS DRAGAPULT EX (Spread / Bench Snipe Counterplay)
    # -------------------------------------------------------------------------
    if is_dragapult:
        # A. Limit bench size to 3 to minimize Phantom Dive damage counter targets
        if IS_PLAY and card_id in (PSYDUCK_ID, SHAYMIN_ID, IRON_BOULDER_ID):
            if my_bench_count >= 3:
                bonus -= 0.15
                triggered = "Dragapult_Matchup_Bench_Cap_Guard"

        # B. Aggressive Gust on Dreepy (336) / Drakloak (337) before evolving
        if IS_PLAY and card_id in (BOSS_ORDERS_ID, POKEMON_CATCHER_ID):
            if any(cid in (336, 337) for cid in opp_bench_ids):
                bonus += 0.35
                triggered = "Dragapult_Matchup_Gust_PreEvolution_Target"

        # C. Preserve Night Stretcher for bench recovery after spread counters
        if IS_PLAY and card_id == NIGHT_STRETCHER_ID:
            if game_phase in ("mid", "late"):
                bonus += 0.25
                triggered = "Dragapult_Matchup_Night_Stretcher_Spread_Recovery"

    # -------------------------------------------------------------------------
    # 2. MATCHUP 2: VS CRUSTLE (Defensive Wall / Stall Counterplay)
    # -------------------------------------------------------------------------
    if is_crustle:
        # A. Gust benched non-wall support instead of tunneling into Crustle wall
        if IS_PLAY and card_id in (BOSS_ORDERS_ID, POKEMON_CATCHER_ID):
            if opp_active_id in (344, 345) and len(opp_bench_ids) > 0:
                bonus += 0.35
                triggered = "Crustle_Matchup_Bypass_Wall_Gust_Support"

        # B. Preserve Togekiss evolution pieces to bypass wall damage
        if IS_PLAY and card_id in (RARE_CANDY_ID, HILDA_ID):
            if TOGEPI_ID in my_bench_ids:
                bonus += 0.35
                triggered = "Crustle_Matchup_Fast_Togekiss_Wall_Bypass"

    # -------------------------------------------------------------------------
    # 3. MATCHUP 3: VS LUCARIO (Fast Fighting Aggro Counterplay)
    # -------------------------------------------------------------------------
    if is_lucario:
        # A. Priority Togekiss setup to out-scale Lucario early
        if IS_EVOLVE and card_id == TOGEKISS_ID:
            bonus += 0.45
            triggered = "Lucario_Matchup_Fast_Togekiss_Evolution"

        # B. Target Riolu (447) before evolving to Lucario
        if IS_PLAY and card_id in (BOSS_ORDERS_ID, POKEMON_CATCHER_ID):
            if 447 in opp_bench_ids:
                bonus += 0.35
                triggered = "Lucario_Matchup_Gust_Riolu_Target"

        # C. Maintain steady attacker stream (at least 2 Clefairy ex)
        if IS_PLAY and card_id == CLEFAIRY_EX_ID:
            if sum(1 for cid in my_bench_ids if cid == CLEFAIRY_EX_ID) < 2:
                bonus += 0.30
                triggered = "Lucario_Matchup_Maintain_Clefairy_Stream"

    # -------------------------------------------------------------------------
    # 4. MATCHUP 4: VS GRIMMSNARL EX (High-HP Dark Tank Counterplay)
    # -------------------------------------------------------------------------
    if is_grimmsnarl:
        # A. Gust benched Impidimp (646) / Morgrem (647) / Munkidori (112)
        if IS_PLAY and card_id in (BOSS_ORDERS_ID, POKEMON_CATCHER_ID):
            if any(cid in (646, 647, 112) for cid in opp_bench_ids):
                bonus += 0.35
                triggered = "Grimmsnarl_Matchup_Gust_Benched_Support"

        # B. Energy ramp on Clefairy ex for 2-turn or 1-turn high HP KO
        if IS_PLAY and card_id == WONDROUS_PATCH_ID:
            bonus += 0.30
            triggered = "Grimmsnarl_Matchup_Wondrous_Patch_High_HP_Ramp"

    # -------------------------------------------------------------------------
    # 5. FULL MOON RONDO STRATEGIC BENCH & DAMAGE CALCULATION
    # -------------------------------------------------------------------------
    opp_bench_count = ctx.get("opp_bench_count", len(opp_bench_ids))
    opp_active_hp = ctx.get("opp_active_hp", 999)
    total_benched = my_bench_count + opp_bench_count

    # Base damage calculation for Full Moon Rondo: 20 + 20 * (my_bench + opp_bench)
    # Lillie's Pearl adds +30 damage to Lillie's Pokemon ex
    pearl_equipped = (my_active_id == CLEFAIRY_EX_ID) or (LILLIES_PEARL_ID in my_bench_ids)
    rondo_base_dmg = 20 + 20 * total_benched + (30 if pearl_equipped else 0)

    # Check weakness: Dragon target gets 2x weakness multiplier due to Fairy Zone ability
    is_dragon_target = opp_active_id in (336, 337, 338, 553, 554, 804, 835) or "dragon" in opp_lower or "archaludon" in opp_lower
    weakness_mult = 2 if is_dragon_target else 1
    current_rondo_dmg = rondo_base_dmg * weakness_mult

    # Next rondo damage if 1 more Pokemon added to bench (either mine or opp's)
    next_rondo_dmg = (20 + 20 * (total_benched + 1) + (30 if pearl_equipped else 0)) * weakness_mult

    # Check if bench addition unlocks a KO on opp active
    ko_threshold_unlocked = (next_rondo_dmg >= opp_active_hp) and (current_rondo_dmg < opp_active_hp)

    # -------------------------------------------------------------------------
    # 6. CORE DECK HEURISTICS (Generic to Lillie Archetype)
    # -------------------------------------------------------------------------
    if IS_PLAY:
        if card_id in (CLEFAIRY_EX_ID, TOGEPI_ID, SMOOCHUM_ID, LATIAS_EX_ID, SHAYMIN_ID, PSYDUCK_ID, MIMIKYU_ID, IRON_BOULDER_ID):
            if my_bench_count < 5 and not (is_dragapult and my_bench_count >= 3):
                if ko_threshold_unlocked:
                    bonus += 0.45
                    triggered = f"Full_Moon_Rondo_Bench_KO_Threshold_{card_id}"
                elif my_active_id == CLEFAIRY_EX_ID or CLEFAIRY_EX_ID in my_bench_ids:
                    bonus += 0.35
                    triggered = f"Full_Moon_Rondo_Strategic_Bench_{card_id}"
                else:
                    bonus += 0.30
                    triggered = f"Lillie_Bench_Basic_{card_id}"

        elif card_id == ACCOMPANYING_FLUTE_ID:
            # Accompanying Flute forces opp basic onto opp bench, boosting Full Moon Rondo damage!
            if opp_bench_count < 5 and (my_active_id == CLEFAIRY_EX_ID or CLEFAIRY_EX_ID in my_bench_ids):
                if ko_threshold_unlocked:
                    bonus += 0.42
                    triggered = "Full_Moon_Rondo_Accompanying_Flute_KO_Setup"
                else:
                    bonus += 0.35
                    triggered = "Full_Moon_Rondo_Accompanying_Flute_Bench_Ramp"

        elif card_id == COLRESS_TENACITY_ID:
            bonus += 0.40
            triggered = "Colress_Tenacity_Search_Stadium_And_Energy"

        elif card_id == HILDA_ID:
            bonus += 0.35
            triggered = "Hilda_Search_Togekiss_Evolution"

        elif card_id in (LILLIES_DETERM_ID, MORTYS_CONVICTION_ID):
            bonus += 0.35
            triggered = f"Lillie_Supporter_Draw_Refresh_{card_id}"

        elif card_id in (ULTRA_BALL_ID, POKE_PAD_ID, NIGHT_STRETCHER_ID):
            bonus += 0.30
            triggered = f"Lillie_Item_Search_Recovery_{card_id}"

        elif card_id == RARE_CANDY_ID:
            if TOGEPI_ID in my_bench_ids:
                bonus += 0.40
                triggered = "Rare_Candy_Togekiss_Shortcut"

        elif card_id == WONDROUS_PATCH_ID:
            bonus += 0.30
            triggered = "Wondrous_Patch_Energy_Ramp"

        elif card_id == MYSTERY_GARDEN_ID:
            bonus += 0.30
            triggered = "Mystery_Garden_Stadium_Play"

        elif card_id == LILLIES_PEARL_ID:
            bonus += 0.32
            triggered = "Full_Moon_Rondo_Lillies_Pearl_Boost"

    elif IS_ATTACH:
        if card_id == TELEPATHIC_ENERGY_ID:
            bonus += 0.35
            triggered = "Telepathic_Energy_Attach_Bench_Search"
        elif my_active_id == CLEFAIRY_EX_ID and my_active_energy < 3:
            bonus += 0.30
            triggered = "Psychic_Energy_Attach_Clefairy"

    elif IS_EVOLVE:
        if card_id == TOGEKISS_ID:
            bonus += 0.45
            triggered = "Evolve_Togekiss_Priority"

    elif IS_ATTACK:
        # Check if hand has unplayed Supporters or setup Items/Stadiums that should be played BEFORE attacking
        my_hand_ids = ctx.get("my_hand_ids", [])
        has_unplayed_supporter = any(cid in (1227, 1225, 1194, 1187, 1182) for cid in my_hand_ids)
        has_unplayed_setup_item = any(cid in (1146, 1091, 1097, 1263, 1172, 1121) for cid in my_hand_ids)

        if my_active_id == CLEFAIRY_EX_ID:
            if current_rondo_dmg >= opp_active_hp:
                bonus += 0.45
                triggered = "Full_Moon_Rondo_Guaranteed_KO"
            elif has_unplayed_supporter or has_unplayed_setup_item:
                # Defer attack prior slightly so MCTS searches playing Supporters/Items FIRST
                bonus += 0.20
                triggered = "Full_Moon_Rondo_Attack_Postpone_For_Supporter_Play"
            else:
                bonus += 0.40
                triggered = "Full_Moon_Rondo_Attack_Execution"
        else:
            if has_unplayed_supporter or has_unplayed_setup_item:
                bonus += 0.15
                triggered = "Lillie_Deck_Attack_Postpone_For_Supporter_Play"
            else:
                bonus += 0.35
                triggered = "Lillie_Deck_Attack_Execution"

    # Bound bonus range [-0.50, +0.45]
    bonus = max(-0.50, min(0.45, bonus))
    return bonus, triggered
