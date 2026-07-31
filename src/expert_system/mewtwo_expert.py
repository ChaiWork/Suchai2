"""
Team Rocket Mewtwo ex + Spidops Specific Heuristic Expert Engine.
Contains all Team Rocket / Mewtwo heuristic rules extracted from expert_knowledge.py.
"""
from cg.api import CardType, OptionType, SelectContext

try:
    from src.training.card_database import get_card_data
except ImportError:
    try:
        from training.card_database import get_card_data
    except ImportError:
        def get_card_data(card_id):
            return None

# Configuration Flags for Ablation Testing & Feature Control
USE_EXPERT_GUIDANCE = True
EXPERT_WEIGHT = 0.10

# Pokémon card IDs used in context checks
MEWTWO_EX_ID = 431
SPIDOPS_ID = 401
MIMIKYU_ID = 434
ARTICUNO_ID = 414
TAROUNTULA_ID = 400
OGERPON_EX_ID = 96
CRUSTLE_ID = 345
STARMIE_EX_ID = 1031
GRIMMSNARL_EX_ID = 648
MUNKIDORI_ID = 112
FROSLASS_ID = 104
IMPIDIMP_ID = 646
MORGREM_ID = 647
SNORUNT_ID = 860

# Item & Tool card IDs
HEROS_CAPE_ID    = 1159   # ACE SPEC: +100 HP
MAXIMUM_BELT_ID  = 1158   # ACE SPEC: +50 dmg vs ex
BRAVE_BANGLE_ID  = 1175   # +30 dmg vs ex (non-ex attacker only)
HANDHELD_FAN_ID  = 1161   # Clears status effects
ENERGY_SWITCH_ID = 1116   # Energy transfer
TRANSCEIVER_ID   = 1134   # TR Supporter tutor
NIGHT_STRETCHER_ID = 1097 # Recovers Pokémon/Energy from discard

# Stadium card IDs
SPIKEMUTH_GYM_ID = 1259   # +30 to all Dark attacks
TR_FACTORY_ID    = 1257   # TR draw bonus

# Grimmsnarl-ex / Munkidori deck evolution lines
GRIMMSNARL_LINE_IDS = {646, 647, 648}  # Impidimp -> Morgrem -> Grimmsnarl-ex
WATER_RAMP_STALL_IDS = frozenset()     # Placeholder — detected by W-energy + low-HP basic stalling

# TR Supporter IDs — synergize with TR Factory (1257) for +2 draw bonus
TR_SUPPORTER_IDS = frozenset({1216, 1218, 1219, 1220})  # Ariana, Giovanni, Petrel, Proton

# Evolution line ID sets (pre-evolutions that should be denied)
DRAGAPULT_LINE_IDS = {336, 337, 338}   # Dreepy, Drakloak, Dragapult
ALAKAZAM_LINE_IDS = {65, 66, 67}       # Abra, Kadabra, Alakazam
LUCARIO_LINE_IDS  = {447, 448}         # Riolu, Lucario
STARMIE_LINE_IDS  = {120, 121}         # Staryu, Starmie


def _extract_context(obs):
    """
    Pull the minimal board-state fields needed for context-aware guards.
    Returns a dict with safe defaults on any access error.
    """
    ctx = {
        "my_prizes":        6,
        "opp_prizes":       6,
        "my_active_id":     -1,
        "opp_active_id":    -1,
        "opp_bench_ids":    [],
        "my_active_energy": 0,
        "game_phase":       "early",  # early / mid / late
        "my_ps":            None,
        "my_bench_count":   0,
        "tr_in_play":       0,       # count of TR Pokémon in active + bench
        "factory_active":   False,   # True if TR Factory (1257) is the active stadium
        "turn":             0,
    }
    try:
        current = getattr(obs, "current", None)
        if current is None and isinstance(obs, dict):
            current = obs.get("current")

        if current is not None:
            yi = getattr(current, "yourIndex", 0) if not isinstance(current, dict) else current.get("yourIndex", 0)
            players = getattr(current, "players", []) if not isinstance(current, dict) else current.get("players", [])
            ctx["turn"] = getattr(current, "turn", 0) if not isinstance(current, dict) else current.get("turn", 0)

            if len(players) >= 2:
                my_ps = players[yi]
                opp_ps = players[1 - yi]

                prizes_my = getattr(my_ps, "prize", []) if not isinstance(my_ps, dict) else my_ps.get("prize", [])
                prizes_opp = getattr(opp_ps, "prize", []) if not isinstance(opp_ps, dict) else opp_ps.get("prize", [])

                ctx["my_prizes"] = len(prizes_my)
                ctx["opp_prizes"] = len(prizes_opp)
                ctx["my_ps"] = my_ps

                my_active_list = getattr(my_ps, "active", []) if not isinstance(my_ps, dict) else my_ps.get("active", [])
                if my_active_list:
                    a = my_active_list[0]
                    if a is not None:
                        if isinstance(a, dict):
                            ctx["my_active_id"] = a.get("cardId", a.get("id", -1))
                            ctx["my_active_energy"] = len(a.get("energyCards", []))
                        else:
                            ctx["my_active_id"] = getattr(a, "cardId", getattr(a, "id", -1))
                            ctx["my_active_energy"] = len(getattr(a, "energyCards", []))

                opp_active_list = getattr(opp_ps, "active", []) if not isinstance(opp_ps, dict) else opp_ps.get("active", [])
                if opp_active_list:
                    a = opp_active_list[0]
                    if a is not None:
                        if isinstance(a, dict):
                            ctx["opp_active_id"] = a.get("cardId", a.get("id", -1))
                        else:
                            ctx["opp_active_id"] = getattr(a, "cardId", getattr(a, "id", -1))

                opp_bench = getattr(opp_ps, "bench", []) if not isinstance(opp_ps, dict) else opp_ps.get("bench", [])
                bench_ids = []
                for p in opp_bench:
                    if p is not None:
                        if isinstance(p, dict):
                            bench_ids.append(p.get("cardId", p.get("id", -1)))
                        else:
                            bench_ids.append(getattr(p, "cardId", getattr(p, "id", -1)))
                ctx["opp_bench_ids"] = bench_ids

                prizes = ctx["my_prizes"]
                if prizes <= 2:
                    ctx["game_phase"] = "late"
                elif prizes <= 4:
                    ctx["game_phase"] = "mid"
                else:
                    ctx["game_phase"] = "early"

                # Count benched Pokémon for bench-building heuristics
                my_bench = getattr(my_ps, "bench", []) if not isinstance(my_ps, dict) else my_ps.get("bench", [])
                my_bench_ids = []
                for p in my_bench:
                    if p is not None:
                        if isinstance(p, dict):
                            my_bench_ids.append(p.get("cardId", p.get("id", -1)))
                        else:
                            my_bench_ids.append(getattr(p, "cardId", getattr(p, "id", -1)))
                ctx["my_bench_ids"] = my_bench_ids
                ctx["my_bench_count"] = len(my_bench_ids)

                # Count TR Pokémon in play (active + bench) for Power Saver gate
                TR_POKEMON_IDS = frozenset({400, 401, 414, 431, 434})
                tr_count = 0
                if ctx["my_active_id"] in TR_POKEMON_IDS:
                    tr_count += 1
                for p in my_bench:
                    if p is not None:
                        pid = getattr(p, "cardId", getattr(p, "id", -1)) if not isinstance(p, dict) else p.get("cardId", p.get("id", -1))
                        if pid in TR_POKEMON_IDS:
                            tr_count += 1
                ctx["tr_in_play"] = tr_count

                # Count hand size for Secret Box / Supporter hand length checks
                my_hand = getattr(my_ps, "hand", []) if not isinstance(my_ps, dict) else my_ps.get("hand", [])
                ctx["my_hand_len"] = len(my_hand)

                # Check if TR Factory (1257) is the active stadium
                try:
                    stadium = getattr(current, "stadium", None)
                    if stadium is not None:
                        stadium_id = getattr(stadium, "cardId", getattr(stadium, "id", -1)) if not isinstance(stadium, dict) else stadium.get("cardId", stadium.get("id", -1))
                        ctx["factory_active"] = (stadium_id == 1257)
                except Exception:
                    pass

    except Exception:
        pass  # Return safe defaults on any attribute error

    return ctx



def evaluate_mewtwo_expert_bonus(obs, option, opponent_name="unknown"):
    """
    Evaluates a context-aware heuristic bonus in range [-0.10, +0.40] for an
    action option.  Serves as a soft prior guide for MCTS without hardcoding moves.

    The bonus is intentionally small so the neural-network prior remains
    dominant.  MCTS can always override a heuristic if enough simulations
    demonstrate a better line.

    Args:
        obs:           Observation class from to_observation_class()
        option:        A single select option from obs.select.option[i]
        opponent_name: Name of the current opponent deck archetype

    Returns:
        (bonus: float, trigger: str)
    """
    if not USE_EXPERT_GUIDANCE or not obs or not option:
        return 0.0, "NONE"

    bonus = 0.0
    triggered = "NONE"
    opp_lower = str(opponent_name).lower()
    opt_type   = getattr(option, "type", None)

    # Extract board-state context once for all guards
    ctx = _extract_context(obs)
    opp_prizes       = ctx["opp_prizes"]
    my_active_id     = ctx["my_active_id"]
    my_bench_ids     = ctx.get("my_bench_ids", [])
    opp_active_id    = ctx["opp_active_id"]
    opp_bench_ids    = ctx["opp_bench_ids"]
    my_active_energy = ctx["my_active_energy"]
    game_phase       = ctx["game_phase"]
    my_ps            = ctx["my_ps"]
    my_bench_count   = ctx["my_bench_count"]
    tr_in_play       = ctx["tr_in_play"]
    factory_active   = ctx["factory_active"]
    my_hand_len      = ctx["my_hand_len"]
    turn             = ctx["turn"]

    # Convenience: full set of visible opponent Pokémon IDs
    opp_pokemon_ids = set(opp_bench_ids) | {opp_active_id}

    # -------------------------------------------------------------------------
    # 00. Emergency Bench-Out Prevention Guard (Fixes 80% of Tournament Losses)
    # When bench size is 0 or 1, give massive priority to benching basic Pokemon
    # or playing search/tutor cards (Poffin, Ultra Ball, Bug Catching Set, Ariana).
    # -------------------------------------------------------------------------
    opt_card_id = getattr(option, "cardId", -1)
    if my_bench_count == 0:
        if opt_type in (7, OptionType.PLAY):
            if opt_card_id in (400, 414, 434, 464, 272, 431, 1086, 1121, 1134, 1094, 1092, 1216, 1220):
                bonus += 0.50
                triggered = "Emergency_Bench_Out_Protection_Prior"
        elif opt_type in (14, getattr(OptionType, "END", 14)):
            bonus -= 0.50
            triggered = "Penalize_End_Turn_With_Zero_Bench"
    elif my_bench_count == 1:
        if opt_type in (7, OptionType.PLAY):
            if opt_card_id in (400, 414, 434, 464, 272, 431, 1086, 1121, 1134, 1094, 1092, 1216, 1220):
                bonus += 0.30
                triggered = "Secondary_Bench_Expansion_Prior"
        elif opt_type in (14, getattr(OptionType, "END", 14)) and turn <= 4:
            bonus -= 0.35
            triggered = "Penalize_End_Turn_With_One_Bench"

    # Card search/tutor selection option check when bench count is 0 or 1
    if my_bench_count <= 1 and opt_card_id in (400, 414, 434, 464, 272, 431):
        bonus += 0.35
        triggered = "Search_Select_Basic_Pokemon_For_Bench"

    # -------------------------------------------------------------------------
    # 0a. Spidops Team Rocket 5-Pokemon Max Damage Scaling Heuristic
    # Spidops's attack damage scales directly with Team Rocket Pokemon in play.
    # When tr_in_play == 5, Spidops achieves maximum damage output (+0.35 bonus)!
    # -------------------------------------------------------------------------
    if my_active_id == SPIDOPS_ID or SPIDOPS_ID in my_bench_ids:
        if opt_type in (13, OptionType.ATTACK) and my_active_id == SPIDOPS_ID:
            if tr_in_play >= 5:
                bonus += 0.35
                triggered = "Spidops_Max_Damage_5_TR_Pokemon_Attack"
            elif tr_in_play == 4:
                bonus += 0.20
                triggered = "Spidops_High_Damage_4_TR_Pokemon_Attack"
        elif opt_type in (7, OptionType.PLAY) and tr_in_play < 5:
            opt_cid = getattr(option, "cardId", -1)
            TR_POKEMON_IDS_SET = {400, 401, 414, 431, 434, 464}
            if opt_cid in TR_POKEMON_IDS_SET:
                if tr_in_play == 4:
                    bonus += 0.30
                    triggered = "Bench_5th_TR_Pokemon_For_Max_Spidops_Damage"

    # -------------------------------------------------------------------------
    # 0b. Mewtwo ex, Clefairy ex, and Sneasel Strategic Prior Heuristics
    # - Mewtwo ex Power Saver gate (requires >= 4 TR Pokemon in play to attack)
    # - Lillie's Clefairy ex Fairy Zone (2x Weakness multiplier vs Dragon targets)
    # - Sneasel TR scaling attack & late-game finisher role
    # -------------------------------------------------------------------------
    if my_active_id == MEWTWO_EX_ID:
        if opt_type in (13, OptionType.ATTACK):
            if tr_in_play >= 4:
                bonus += 0.35
                triggered = "Mewtwo_Power_Saver_Satisfied_Attack"
            else:
                bonus -= 0.30
                triggered = "Mewtwo_Power_Saver_Locked_Penalty"
    elif opt_type in (7, OptionType.PLAY) and tr_in_play < 4 and (my_active_id == MEWTWO_EX_ID or MEWTWO_EX_ID in my_bench_ids):
        opt_cid = getattr(option, "cardId", -1)
        if opt_cid in {400, 401, 414, 431, 434, 464}:
            bonus += 0.25
            triggered = "Bench_TR_Pokemon_For_Mewtwo_Power_Saver"

    if getattr(option, "cardId", -1) == 272 and opt_type in (7, OptionType.PLAY):
        opp_card = get_card_data(opp_active_id)
        if opp_card and any(kw in getattr(opp_card, "name", "").lower() for kw in ["dragapult", "dreepy", "drakloak", "garchomp", "dragon"]):
            bonus += 0.35
            triggered = "Clefairy_ex_Fairy_Zone_Vs_Dragon_Opponent"

    if my_active_id == 464 and opt_type in (13, OptionType.ATTACK):
        if tr_in_play >= 4:
            bonus += 0.30
            triggered = "Sneasel_TR_Scaled_Attack"

    # -------------------------------------------------------------------------
    # 0. Generic Unseen Opponent Threat Evaluator (Zero-Shot Generalization)
    # Evaluates ANY opponent active/bench Pokemon by generic attributes
    # (HP, Energy Count, Prize Value) without requiring card ID matching.
    # -------------------------------------------------------------------------
    opp_active_card = get_card_data(opp_active_id)
    opp_active_hp = getattr(opp_active_card, "hp", 100) if opp_active_card else 100
    is_opp_ex = (opp_active_hp >= 200) or ("ex" in str(getattr(opp_active_card, "name", "")).lower())

    if is_opp_ex:  # High-HP EX Threat (Unseen or Known)
        if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1175:  # Brave Bangle (+0.12)
            bonus += 0.12
            triggered = "Generic_EX_Brave_Bangle_Threat_Prior"
        elif opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) in (414, 434):  # 1-Prize Wall defense
            if my_bench_count < 4:
                bonus += 0.15
                triggered = "Generic_EX_Wall_Defense_Prior"

    # -------------------------------------------------------------------------
    # 1. Universal Bench Snipe Defense — Repelling Veil & Urgent Spidops Evolution
    #
    # Opponents with Bench Snipe attacks (Dragapult ex Phantom Dive, Alakazam,
    # Kyurem ex, Greninja ex) deal 60+ damage directly to benched Basics,
    # one-shotting 50HP Tarountulas.
    #
    # Counter Strategy:
    # 1. Bench Articuno! Repelling Veil PREVENTS all damage counters/snipe
    #    from being placed on all Basic Team Rocket Pokemon on your bench!
    # 2. Priority evolve Tarountula (50 HP) -> Spidops (130 HP) to exceed 60HP.
    # 3. Use search cards (Ultra Ball, Bug Catching Set, Secret Box, Ariana)
    #    to fetch Articuno (414) or Spidops (401) immediately!
    # -------------------------------------------------------------------------
    BENCH_SNIPE_KEYWORDS = ["dragapult", "alakazam", "kyurem", "greninja", "dipplin"]
    is_bench_snipe_opponent = any(k in opp_lower for k in BENCH_SNIPE_KEYWORDS) or bool(DRAGAPULT_LINE_IDS & opp_pokemon_ids) or (opp_active_id == 14)

    if is_bench_snipe_opponent:
        # 1. Priority Bench Articuno (414) for Repelling Veil bench shield
        if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == ARTICUNO_ID:
            if my_bench_count < 5:
                bonus += 0.35
                triggered = "Articuno_RepellingVeil_Bench_Snipe_Shield"

        # 2. Retreat Active Articuno to Bench so Repelling Veil stays active
        elif opt_type in (12, OptionType.RETREAT) and my_active_id == ARTICUNO_ID:
            bonus += 0.30
            triggered = "Articuno_Retreat_To_Bench_Vs_Bench_Snipe"

        # 3. Priority Evolve Tarountula (50 HP) -> Spidops (130 HP) to survive 60 HP snipe
        elif opt_type in (9, OptionType.EVOLVE) and getattr(option, "cardId", -1) == SPIDOPS_ID:
            bonus += 0.35
            triggered = "Spidops_Evolve_To_Survive_Bench_Snipe"

        # 4. Search Cards to fetch Articuno (414) or Spidops (401)
        elif opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) in (1094, 1121, 1092, 1216, 1227):
            if ARTICUNO_ID not in my_bench_ids and my_active_id != ARTICUNO_ID:
                bonus += 0.25
                triggered = "Search_Articuno_Bench_Snipe_Shield"

        # 4. Evolution Denial: KO Dreepy/Drakloak on opponent bench before Dragapult ex evolves
        elif opt_type in (13, OptionType.ATTACK):
            target_name = str(getattr(option, "targetName", "")).lower()
            is_denial_target = any(k in target_name for k in ["dreepy", "drakloak"])
            if is_denial_target and bool(DRAGAPULT_LINE_IDS & opp_pokemon_ids):
                bonus += 0.15 if game_phase in ("mid", "late") else 0.08
                triggered = "KO_Dreepy_Drakloak"

    # -------------------------------------------------------------------------
    # 1b. Fire Deck Counter (Typhlosion / Centiskorch) — Non-Grass Opening
    # -------------------------------------------------------------------------
    FIRE_POKEMON_IDS = {352, 353, 354, 717, 934, 18, 19, 20}
    is_fire_opponent = any(k in opp_lower for k in ["typhlosion", "centiskorch", "fire", "charizard", "hooh"]) or bool(FIRE_POKEMON_IDS & opp_pokemon_ids)

    if is_fire_opponent:
        card_id = getattr(option, "cardId", getattr(option, "id", -1))
        if opt_type in (7, OptionType.PLAY) and card_id in (414, 272, 434):
            if turn <= 3:
                bonus += 0.20
                triggered = "Fire_Matchup_Articuno_Clefairy_Prior"
        elif opt_type in (7, OptionType.PLAY) and card_id in (400, 401) and turn <= 3:
            bonus -= 0.18
            triggered = "Fire_Matchup_Grass_Exposure_Penalty"

    # -------------------------------------------------------------------------
    # 1c. Iono / Hand Disruption Counter — Board Emptying Guard
    # -------------------------------------------------------------------------
    is_iono_opponent = "iono" in opp_lower or 1190 in opp_pokemon_ids
    if is_iono_opponent or my_hand_len >= 6:
        card_id = getattr(option, "cardId", getattr(option, "id", -1))
        if opt_type in (7, OptionType.PLAY) and card_id in (400, 414, 434, 464, 272, 1156, 1175, 1257):
            bonus += 0.16
            triggered = "Iono_Disruption_Board_Emptying_Prior"
        elif opt_type in (8, OptionType.ATTACH):
            bonus += 0.14
            triggered = "Iono_Disruption_Energy_Boarding_Prior"

    # -------------------------------------------------------------------------
    # 1d. Starmie ex Counter — Staryu Evolution Denial & Maximum Belt OHKO
    # -------------------------------------------------------------------------
    elif "starmie" in opp_lower or 120 in opp_bench_ids or opp_active_id in (120, 121, 1031):
        if opt_type in (13, OptionType.ATTACK):
            target_name = str(getattr(option, "targetName", "")).lower()
            if opp_active_id == 120 or "staryu" in target_name:
                bonus += 0.18
                triggered = "Staryu_Evolution_Denial_Attack"
        elif opt_type in (7, OptionType.PLAY):
            card_id = getattr(option, "cardId", -1)
            if card_id == 1218 and 120 in opp_bench_ids:  # Giovanni gust Staryu
                bonus += 0.15
                triggered = "Giovanni_Gust_Staryu_Prior"
            elif card_id == 1218 and len(opp_bench_ids) > 0:  # Giovanni gust to bypass un-killable active EX
                opp_active_card = get_card_data(opp_active_id)
                opp_hp = getattr(opp_active_card, "hp", 100) if opp_active_card else 100
                if opp_hp >= 200:
                    bonus += 0.30
                    triggered = "Giovanni_Gust_Unkillable_EX_Bypass_Prior"
            elif card_id == 1158 and opp_active_id in (121, 1031):  # Maximum Belt vs Starmie ex
                bonus += 0.14
                triggered = "Maximum_Belt_Vs_Starmie_EX"

    # -------------------------------------------------------------------------
    # 1c. 2nd Place Tournament Deck Heuristics (Secret Box, Clefairy ex, Sneasel)
    # -------------------------------------------------------------------------
    if opt_type in (7, OptionType.PLAY):
        card_id = getattr(option, "cardId", getattr(option, "id", -1))
        opt_name_str = str(getattr(option, "name", "")).lower()
        if card_id == 1092 or ("secret" in opt_name_str and "box" in opt_name_str):  # Secret Box (ACE SPEC)
            my_hand_len = ctx.get("my_hand_len", 5)
            if my_hand_len >= 4:
                bonus += 0.22 if my_bench_count < 3 else 0.15
                triggered = "Secret_Box_Pre_Supporter_Ramp"
            else:
                bonus -= 0.05
                triggered = "Secret_Box_Low_Hand_Penalty"
        elif card_id == 1227:  # Lillie's Determination (draw up to 6/8 cards)
            if my_hand_len <= 4 or my_prizes_rem > opp_prizes_rem:
                bonus += 0.18
                triggered = "Lillie_Determination_Max_Draw_Prior"
        elif card_id == 1218:  # Team Rocket's Giovanni (Gust targeted benched Pokemon)
            if has_attack_available or game_phase in ("mid", "late"):
                bonus += 0.18
                triggered = "Giovanni_Gust_KO_Prior"
        elif card_id == 1116:  # Energy Switch (accelerate energy to active attacker)
            if my_active_energy == 0 and has_attack_available:
                bonus += 0.18
                triggered = "Energy_Switch_Attacker_Prior"
        elif card_id in (1257, 50, 51):  # Team Rocket's Factory (Stadium draw engine)
            if my_hand_len <= 4:
                bonus += 0.12
                triggered = "TR_Factory_Draw_Engine_Prior"
        elif card_id == 1129:  # Sacred Ash late-game recovery
            if game_phase in ("mid", "late"):
                bonus += 0.10
                triggered = "Sacred_Ash_Late_Recycle"
            else:
                bonus -= 0.05
                triggered = "Sacred_Ash_Early_Waste_Penalty"
        elif card_id == 1175 and is_opp_ex and my_active_id == 401:  # Brave Bangle vs EX
            bonus += 0.08
            triggered = "Brave_Bangle_EX_Threshold"
        elif card_id == 1156:  # Lucky Helmet (draw engine on active/wall)
            if my_active_id in (434, 414) or my_active_id == 401:
                bonus += 0.08
                triggered = "Lucky_Helmet_Wall_Draw_Prior"
            else:
                bonus += 0.04
                triggered = "Lucky_Helmet_Equip_Prior"
        elif card_id in (1086, 1094):  # Buddy-Buddy Poffin (1086) or Bug Catching Set (1094)
            if turn <= 3 and my_bench_count < 4:
                bonus += 0.18  # Priority zero-discard setup items over Ultra Ball
                triggered = "Poffin_BugCatching_Early_Setup_Prior"
        elif card_id in (1094, 1121, 1092, 1216, 1227):  # Bug Catching Set, Ultra Ball, Secret Box, Ariana, Lillie
            if 400 in my_bench_ids and 401 not in my_bench_ids:
                bonus += 0.20
                triggered = "Search_To_Evolve_Benched_Tarountula_Prior"

        # Early Turn Supporter/Search setup for Tarountula (400), Mewtwo ex (431), Clefairy ex (272)
        if turn <= 4 and card_id in (1216, 1227, 1218, 1220, 1086, 1094, 1121, 1134):
            has_attacker = any(cid in my_bench_ids or my_active_id == cid for cid in (400, 431, 272))
            if not has_attacker or my_bench_count < 3:
                bonus += 0.25
                triggered = "Early_Turn_Supporter_Attacker_Setup"

    elif opt_type in (8, OptionType.ATTACH):
        card_id = getattr(option, "cardId", -1)
        target_id = getattr(option, "targetCardId", getattr(option, "targetId", -1))
        if target_id == 414:  # Target is Articuno (414)
            if card_id == 15 or "rocket" in str(getattr(option, "name", "")).lower():
                bonus -= 0.30
                triggered = "TR_Energy_Articuno_Waste_Penalty"
            elif card_id == 1:  # Basic {G} Energy
                bonus += 0.15
                triggered = "Grass_Energy_Articuno_Prior"
        elif target_id == 272:  # Target is Lillie's Clefairy ex (272)
            if card_id == 15 or "rocket" in str(getattr(option, "name", "")).lower():
                bonus -= 0.50  # TR Energy cannot be attached to non-TR Clefairy ex!
                triggered = "TR_Energy_Illegal_Clefairy_Penalty"
            elif card_id in (1, 5):  # Basic Green (1) or Basic Psychic (5) Energy
                bonus += 0.20
                triggered = "Clefairy_Green_Psychic_Energy_Prior"
        elif card_id == 5:  # Basic Psychic Energy (#5)
            if target_id in (400, 401):  # Tarountula (400) or Spidops (401)
                bonus -= 0.25  # Save Psychic Energy for Clefairy ex & Mewtwo ex!
                triggered = "Psychic_Energy_Spidops_Penalty"
            elif target_id in (272, 431):  # Clefairy ex (272) or Mewtwo ex (431)
                bonus += 0.18
                triggered = "Psychic_Energy_Attacker_Prior"


        # Dragapult ex vs Clefairy ex Unprotected Guard (-0.10)
        if card_id == 272 and "dragapult" in opp_lower and 414 not in my_bench_ids and my_active_id != 414:
            bonus -= 0.10
            triggered = "Clefairy_EX_Dragapult_Unprotected_Penalty"

    elif opt_type in (12, OptionType.RETREAT):
        if my_active_id in (434, 414, 400):  # 1-Prize Wall (Mimikyu 434, Articuno 414, Tarountula 400)
            bonus -= 0.20
            triggered = "Wall_Retreat_Suicide_Penalty"
        elif my_active_id == 431 and (tr_in_play < 4 or turn <= 4):  # Early Locked Active Mewtwo ex
            if 414 in my_bench_ids or 434 in my_bench_ids or 400 in my_bench_ids:
                bonus += 0.25
                triggered = "Mewtwo_EX_Early_Active_Emergency_Retreat"
        elif my_active_id == 464 and turn <= 4:  # Early Active Fragile Sneasel (60 HP)
            if 414 in my_bench_ids or 434 in my_bench_ids or 400 in my_bench_ids:
                bonus += 0.22
                triggered = "Sneasel_Early_Active_Emergency_Retreat"

    elif opt_type in (9, OptionType.EVOLVE):
        bonus += 0.35  # Priority evolution (Tarountula 50 HP -> Spidops 130 HP) before Supporter/Attack
        triggered = "Priority_Evolution_Tarountula_Spidops"

    elif opt_type in (14, getattr(OptionType, "END", 14)):
        select_opts = getattr(getattr(obs, "select", None), "option", [])
        has_setup_opt = any(getattr(o, "type", -1) in (7, 8, 9, OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE) for o in select_opts)
        if has_setup_opt:
            bonus -= 0.25
            triggered = "Premature_End_Turn_Penalty"

    elif opt_type in (13, OptionType.ATTACK):
        if my_active_id == 272:  # Clefairy ex attack priority
            opp_active_card = get_card_data(opp_active_id)
            opp_name = str(getattr(opp_active_card, "name", "")).lower() if opp_active_card else ""
            if any(kw in opp_name for kw in ["dragapult", "dreepy", "drakloak", "garchomp", "dragon"]):
                bonus += 0.25
                triggered = "Clefairy_EX_Fairy_Zone_Dragon_Attack_Prior"
            else:
                bonus += 0.15
                triggered = "Clefairy_EX_Attack_Bonus"
        elif my_active_id == 431:  # Mewtwo ex late-game finisher
            if game_phase == "late" and tr_in_play >= 4:
                bonus += 0.15
                triggered = "Mewtwo_EX_Late_Game_Finisher_Prior"
            else:
                bonus -= 0.15
                triggered = "Mewtwo_EX_Early_Exposure_Penalty"
        elif my_active_id == 464:  # Sneasel late-game snipe
            if game_phase == "late":
                bonus += 0.12
                triggered = "Sneasel_Late_Bench_Snipe"
            else:
                bonus -= 0.12
                triggered = "Sneasel_Early_Exposure_Penalty"
        elif my_active_id == 400:  # Unevolved Tarountula attack guard vs high HP
            opp_active_card = get_card_data(opp_active_id)
            opp_hp = getattr(opp_active_card, "hp", 999) if opp_active_card else 999
            if opp_hp >= 80 and my_hand_len >= 2:
                bonus -= 0.12
                triggered = "Tarountula_Hopeless_Attack_Penalty"

        # Hopeless Attack vs Un-killable High-HP EX when opponent is on Game Point (-0.20)
        opp_active_card = get_card_data(opp_active_id)
        opp_hp = getattr(opp_active_card, "hp", 999) if opp_active_card else 999
        opp_prizes_rem = ctx.get("opp_prizes", 6)
        if opp_hp >= 250 and opp_prizes_rem <= 1 and len(opp_bench_ids) > 0:
            bonus -= 0.20
            triggered = "Hopeless_Attack_Vs_Unkillable_EX_Penalty"

    # Action Ordering: Prioritize Play/Attach setup before declaring Attack (+0.12)
    has_attack_available = False
    try:
        select_opts = getattr(getattr(obs, "select", None), "option", [])
        has_attack_available = any(getattr(opt, "type", -1) in (13, OptionType.ATTACK) for opt in select_opts)
    except Exception:
        pass

    if opt_type in (7, OptionType.PLAY):
        card_id = getattr(option, "cardId", -1)
        opt_name_str = str(getattr(option, "name", "")).lower()
        if card_id in (1216, 1134) or ("ariana" in opt_name_str or "transceiver" in opt_name_str):
            if has_attack_available:
                bonus += 0.15
                triggered = "Ariana_Supporter_Before_Attack_Prior"
            elif turn <= 2:
                bonus += 0.10
                triggered = "Transceiver_Before_Tool_Prior"
        elif card_id in (1175, 1156) and turn <= 2:
            # Slight dampening of Tool equipping on T1 before searching Supporter
            bonus -= 0.04

    if opt_type in (7, OptionType.PLAY, 8, OptionType.ATTACH) and has_attack_available:
        bonus += 0.12
        triggered = "Pre_Attack_Setup_Prior"

    # -------------------------------------------------------------------------
    # 1d. Intelligent Context-Aware Search Priors (Transceiver, Ultra Ball, Poké Pad)
    # -------------------------------------------------------------------------
    if opt_type in (7, OptionType.PLAY):
        card_id = getattr(option, "cardId", -1)
        if card_id == 1134:  # TR Transceiver
            if any(id_ in opp_bench_ids for id_ in (120, 336, 65)):  # Staryu, Dreepy, Abra
                bonus += 0.08
                triggered = "Transceiver_Giovanni_Gust_Prior"
            elif turn <= 2 and my_bench_count < 3:
                bonus += 0.10
                triggered = "Transceiver_Proton_Turn1_Setup_Prior"
            elif my_hand_len <= 3:
                bonus += 0.06
                triggered = "Transceiver_Ariana_Draw_Prior"
        elif card_id == 1121:  # Ultra Ball
            if "dragapult" in opp_lower and 414 not in my_bench_ids:
                bonus += 0.08
                triggered = "Ultra_Ball_Articuno_Vs_Dragapult_Prior"
            elif game_phase in ("early", "mid"):
                bonus += 0.06
                triggered = "Ultra_Ball_Spidops_Evolve_Prior"
        elif card_id == 1152:  # Poké Pad
            if game_phase in ("mid", "late"):
                bonus += 0.05
                triggered = "Poke_Pad_Supporter_Recycle_Prior"

    # -------------------------------------------------------------------------
    # 2. Dipplin Counter — Stadium Control (Festival Grounds override)
    # -------------------------------------------------------------------------
    elif "dipplin" in opp_lower:
        if opt_type in (7, OptionType.PLAY):
            card_id = getattr(option, "cardId", -1)
            if card_id in (1257, 1258):  # Team Rocket's Factory / Spikemuth Gym
                bonus += 0.10
                triggered = "Override_Festival_Grounds"

    # -------------------------------------------------------------------------
    # 3. Grimmsnarl Counter — Spidops Retreat Trap
    # -------------------------------------------------------------------------
    elif "grimmsnarl" in opp_lower:
        if opt_type in (7, OptionType.PLAY):
            card_id = getattr(option, "cardId", -1)
            if card_id in (400, 401):  # Tarountula / Spidops
                # Most useful in early-to-mid game to establish trap
                if game_phase in ("early", "mid"):
                    bonus += 0.10
                    triggered = "Spidops_Retreat_Trap"

    # -------------------------------------------------------------------------
    # 4. Iono Counter — Hand Recovery Conservation
    # -------------------------------------------------------------------------
    elif "iono" in opp_lower:
        my_hand_len = 5
        try:
            yi = obs.current.yourIndex
            my_hand_len = len(obs.current.players[yi].hand)
        except Exception:
            pass

        # Reward search/recovery when hand is depleted (<= 2 cards)
        if my_hand_len <= 2 and opt_type in (7, OptionType.PLAY):
            card_id = getattr(option, "cardId", -1)
            if card_id in (1134, 1216, 1094, 1097, 1257, 1227):  # Transceiver, Ariana, Lillie's Determination, etc.
                bonus += 0.15 if card_id == 1227 else 0.10
                triggered = "Lillie_Hand_Reset_Recovery" if card_id == 1227 else "Iono_Recovery_Search"

    # -------------------------------------------------------------------------
    # 4a. Teal Mask Ogerpon ex (ID 96) Teal Dance + Energy Switch (ID 1116) Heuristic
    # -------------------------------------------------------------------------
    if opt_type in (10, OptionType.ABILITY):
        opt_str = (str(getattr(option, "name", "")) + " " + str(getattr(option, "text", ""))).lower()
        is_teal_dance = any(kw in opt_str for kw in ["teal", "dance", "ogerpon"])
        if not is_teal_dance:
            cid = getattr(option, "cardId", getattr(option, "id", -1))
            if cid == 96:
                is_teal_dance = True
            elif ctx.get("my_ps") is not None:
                try:
                    my_ps = ctx["my_ps"]
                    area = getattr(option, "inPlayArea", -1)
                    idx = getattr(option, "inPlayIndex", -1)
                    if area == 4 and len(my_ps.active) > 0 and getattr(my_ps.active[0], "cardId", getattr(my_ps.active[0], "id", -1)) == 96:
                        is_teal_dance = True
                    elif area == 5 and 0 <= idx < len(my_ps.bench) and getattr(my_ps.bench[idx], "cardId", getattr(my_ps.bench[idx], "id", -1)) == 96:
                        is_teal_dance = True
                except Exception:
                    pass
        if is_teal_dance:
            bonus += 0.25
            triggered = "Ogerpon_Teal_Dance_Draw_Accelerate"
    elif opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1116:  # Energy Switch
        bonus += 0.20
        triggered = "Energy_Switch_Tempo_Transfer"

    # -------------------------------------------------------------------------
    # 4a-2. Night Stretcher (ID 1097) Attacker Recycle Heuristic
    # -------------------------------------------------------------------------
    if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1097:
        bonus += 0.18
        triggered = "Night_Stretcher_Attacker_Recycle"

    # -------------------------------------------------------------------------
    # 4b. Universal Lillie's Determination (ID 1227) Hand Reset Heuristic
    # -------------------------------------------------------------------------
    if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1227:
        my_hand_len = 5
        try:
            yi = obs.current.yourIndex
            my_hand_len = len(obs.current.players[yi].hand)
        except Exception:
            pass

        if my_hand_len <= 2:
            bonus += 0.15
            triggered = "Lillie_Hand_Reset_Recovery"
        elif my_hand_len >= 6:
            bonus -= 0.05
            triggered = "Lillie_Wasteful_Reset_Penalty"

    # -------------------------------------------------------------------------
    # 4b-2. Universal 1-Per-Turn Supporter & Item Priority Heuristics
    # -------------------------------------------------------------------------
    supporter_played = False
    try:
        if obs and getattr(obs, "current", None):
            supporter_played = getattr(obs.current, "supporterPlayed", False)
    except Exception:
        pass

    if opt_type in (7, OptionType.PLAY):
        card_id = getattr(option, "cardId", -1)
        if not supporter_played:
            if card_id == 1218:  # Team Rocket's Giovanni (Double Switch / Gust: switches active TR Pokemon & gusts opp bench)
                if len(opp_bench_ids) > 0:
                    bonus += 0.20
                    triggered = "Giovanni_Double_Switch_Gust"
            elif card_id == 1216:  # Ariana — draw to 5 (or 8 when all TR Pokémon in play)
                if tr_in_play >= 4:
                    bonus += 0.20  # Full Ariana bonus: draw 8 cards when TR board is maxed!
                    triggered = "Ariana_Draw8_TR_Board_Full"
                else:
                    bonus += 0.15
                    triggered = "Draw_Supporter_Priority"
            elif card_id == 1227:  # Lillie's Determination (handled above in 4b, add Factory synergy here)
                if factory_active:
                    bonus += 0.10  # Extra draw when Factory is active
                    triggered = "Lillie_Factory_Combo"
                else:
                    bonus += 0.10
                    triggered = "Draw_Supporter_Priority"
            elif card_id == 1219:  # Petrel (search any Trainer)
                bonus += 0.15
                triggered = "Search_Supporter_Priority"
            elif card_id == 1220:  # Proton (search 3 Basic TR Pokémon; playable Turn 1 going first!)
                if turn <= 1:
                    bonus += 0.25  # Proton Turn 1 is a massive tempo play — search 3 TR Basics immediately!
                    triggered = "Proton_Turn1_Bench_Setup"
                elif my_bench_count < 3:
                    bonus += 0.20  # Urgently need bench Pokémon to reach Power Saver threshold
                    triggered = "Proton_Bench_Build_Power_Saver"
                else:
                    bonus += 0.15
                    triggered = "Search_Supporter_Priority"

            # TR Factory + TR Supporter combo: playing any TR Supporter with Factory in play draws +2 extra cards
            if card_id in TR_SUPPORTER_IDS and factory_active and not supporter_played:
                bonus += 0.10
                triggered = triggered + "_Factory_Combo" if triggered != "NONE" else "TR_Factory_Supporter_Combo"

        if card_id == 1159:  # Hero's Cape (+100 HP ACE SPEC Tool)
            bonus += 0.20
            triggered = "Hero_Cape_Plus100HP_Boost"
        elif card_id == 1134:  # Transceiver
            bonus += 0.15
            triggered = "Transceiver_Item_Search"
        elif card_id == 1094:  # Bug Catching Set
            bonus += 0.15
            triggered = "Bug_Catching_Set_Search"
        elif card_id == 1257:  # TR Factory (play it early to maximize combo value)
            if game_phase == "early" and not factory_active:
                bonus += 0.15
                triggered = "TR_Factory_Setup"

    # -------------------------------------------------------------------------
    # 4c. Universal Emergency Bench Search & Basic Placement (Prevent Lone Active Wipe)
    #
    # Having 0 benched Pokemon is an existential threat (1 KO = instant loss).
    # Highest priority when bench is empty is to search and play Basic Pokemon.
    # -------------------------------------------------------------------------
    if my_bench_count == 0:
        if opt_type in (7, OptionType.PLAY):
            card_id = getattr(option, "cardId", -1)
            if card_id in (1134, 1086, 1220, 1094):  # Transceiver, Poffin, Proton, Bug Catching Set
                bonus += 0.35
                triggered = "Turn1_Emergency_Bench_Search"
            elif card_id in (400, 414, 434, 431):  # Playing any Basic Pokemon from hand to bench
                bonus += 0.35
                triggered = "Emergency_Bench_Basic_Placement"

    # -------------------------------------------------------------------------
    # 4d. Priority Evolution (Tarountula -> Spidops)
    #
    # Evolving Tarountula (50 HP) to Spidops (130 HP) upgrades HP, attack,
    # and retreat-locking ability. Evolving active Tarountula prevents KO!
    # -------------------------------------------------------------------------
    if opt_type in (9, OptionType.EVOLVE):
        card_id = getattr(option, "cardId", -1)
        if card_id == SPIDOPS_ID:  # Spidops
            if my_active_id == TAROUNTULA_ID:
                bonus += 0.40  # Maximum priority: Evolve active 50 HP Tarountula immediately!
                triggered = "Active_Tarountula_Spidops_Evolution"
            else:
                bonus += 0.35  # High priority: Evolve bench Tarountula
                triggered = "Priority_Spidops_Evolution"

    # -------------------------------------------------------------------------
    # 4e. Power Saver Bench Gate — Core Win Condition Guard
    #
    # Power Saver requires 4+ TR Pokémon in play (Active + Bench).
    # The AI must not attack with Mewtwo ex before the threshold is met,
    # and must be rewarded for bench-building cards that reach it.
    # -------------------------------------------------------------------------

    # C1a: Penalize attacking with Mewtwo ex when Power Saver precondition not met
    if opt_type in (13, OptionType.ATTACK) and my_active_id == MEWTWO_EX_ID:
        if tr_in_play < 4:
            bonus -= 0.30  # Power Saver will fail (0 damage) — heavily discourage this attack!
            triggered = "Power_Saver_Bench_Not_Ready"

    # C1b: Reward bench-building when bench < 3 TR Pokémon (need 4 total for Power Saver)
    if my_bench_count < 3 and opt_type in (7, OptionType.PLAY):
        card_id = getattr(option, "cardId", -1)
        if card_id in (1086, 1220, 1152, 1094, 1134):  # Poffin, Proton, PokePad, Bug Catching Set, Transceiver
            bonus += 0.15
            triggered = "Bench_Build_Power_Saver_Prereq"

    # -------------------------------------------------------------------------
    # 4e-2. Bench Over-Charge Guard — Max 2 Energies Per Bench Pokémon
    #
    # Strategy: charge bench Pokémon to a maximum of 2 energies each.
    # - Attaching a 3rd energy to a bench Pokémon is wasteful because:
    #   * Retreat costs (Spidops=2, Mewtwo ex=3) will discard those energies
    #   * 2 energies is enough to enable Silky String (Spidops, 2G) on arrival
    #   * Energy is better used on the active attacker or spread across more bench
    # - Exception: active Pokémon may need up to 4 energies (Power Saver)
    # -------------------------------------------------------------------------
    if opt_type in (8, OptionType.ATTACH):
        attach_index = getattr(option, "inPlayIndex", -1)
        is_bench_target = (attach_index >= 1)  # index 0 = active, 1+ = bench slots
        if is_bench_target:
            try:
                bench_slot = attach_index - 1  # bench[0] = inPlayIndex 1, etc.
                if my_ps is not None:
                    bench_list = getattr(my_ps, "bench", []) if not isinstance(my_ps, dict) else my_ps.get("bench", [])
                    if bench_slot < len(bench_list) and bench_list[bench_slot] is not None:
                        target = bench_list[bench_slot]
                        if isinstance(target, dict):
                            bench_energies = len(target.get("energyCards", []))
                        else:
                            bench_energies = len(getattr(target, "energyCards", []))

                        if bench_energies >= 2:
                            # Already has 2 energies — adding a 3rd is over-charging!
                            bonus -= 0.30
                            triggered = "Bench_Overcharge_Penalty"
                        elif bench_energies == 1:
                            # First charge already done, second is fine — small bonus to fill to 2
                            bonus += 0.05
                            triggered = "Bench_Second_Energy_OK"
                        else:
                            # First energy on a bench Pokemon — good, encourage early prep
                            bonus += 0.10
                            triggered = "Bench_First_Energy_Prep"
            except Exception:
                pass  # Fail open — no penalty on access error

        # -------------------------------------------------------------------------
        # TR Energy Conservation Rule: Save TR Energy (15) for Critical Situations
        # -------------------------------------------------------------------------
        try:
            attached_card_id = getattr(option, "cardId", -1)
            if my_ps is not None:
                hand_cards = getattr(my_ps, "hand", []) if not isinstance(my_ps, dict) else my_ps.get("hand", [])
                hand_cids = [c.get("id") if isinstance(c, dict) else getattr(c, "id", -1) for c in hand_cards if c]
                has_basic_grass_in_hand = (1 in hand_cids)

                # Check if critical situation (active HP <= 60, sole opponent active with 0 bench, or active needs immediate attack)
                is_critical = False
                opp_bench_cnt = 99
                if opp_ps is not None:
                    opp_bench_list = getattr(opp_ps, "bench", []) if not isinstance(opp_ps, dict) else opp_ps.get("bench", [])
                    opp_bench_cnt = len([b for b in opp_bench_list if b])
                
                if opp_bench_cnt == 0:
                    is_critical = True  # Instant game-winning KO opportunity!

                if my_ps.active and my_ps.active[0] is not None:
                    a_obj = my_ps.active[0]
                    my_hp = a_obj.get("hp", 999) if isinstance(a_obj, dict) else getattr(a_obj, "hp", 999)
                    if my_hp <= 60 or has_attack_option:
                        is_critical = True

                # Special Priority: If opponent has 0 bench, attach TR Energy immediately for instant win KO!
                if attached_card_id == 15 and opp_bench_cnt == 0:
                    bonus += 0.35
                    triggered = "Instant_Win_KO_TR_Energy_Bonus"
                # If attaching Team Rocket Energy (15) while Basic Grass Energy (1) IS in hand and situation is NOT critical
                elif attached_card_id == 15 and has_basic_grass_in_hand and not is_critical:
                    bonus -= 0.30
                    triggered = "TR_Energy_Conservation_Penalty"
                # If attaching Basic Grass Energy (1) when TR Energy is also in hand, give bonus to conserve TR energy
                elif attached_card_id == 1 and (15 in hand_cids) and not is_critical:
                    bonus += 0.20
                    triggered = "Prefer_Basic_Grass_Energy_Bonus"
        except Exception:
            pass

        # Mewtwo ex (431) Green Energy Cap: Max 1 Green Energy (card ID 1) allowed!
        # 1 Green Energy fulfills 1 Colorless energy requirement (e.g. 1 TR Energy + 1 Green Energy = 3E for Psychic Burst).
        # Overcharging 2+ Green Energies on Mewtwo ex is forbidden (wastes Spidops' Grass energy).
        try:
            attached_card_id = getattr(option, "cardId", -1)
            target_is_mewtwo = False
            mewtwo_grass_count = 0
            mewtwo_obj = None

            if attach_index == 0 and my_ps is not None and my_ps.active and my_ps.active[0] is not None:
                a_obj = my_ps.active[0]
                aid = a_obj.get("id") if isinstance(a_obj, dict) else getattr(a_obj, "id", -1)
                if aid == MEWTWO_EX_ID:
                    target_is_mewtwo = True
                    mewtwo_obj = a_obj
            elif attach_index >= 1 and my_ps is not None:
                b_slot = attach_index - 1
                bench_list = getattr(my_ps, "bench", []) if not isinstance(my_ps, dict) else my_ps.get("bench", [])
                if b_slot < len(bench_list) and bench_list[b_slot] is not None:
                    b_obj = bench_list[b_slot]
                    bid = b_obj.get("id") if isinstance(b_obj, dict) else getattr(b_obj, "id", -1)
                    if bid == MEWTWO_EX_ID:
                        target_is_mewtwo = True
                        mewtwo_obj = b_obj

            if target_is_mewtwo and mewtwo_obj is not None:
                energies = mewtwo_obj.get("energyCards", []) if isinstance(mewtwo_obj, dict) else getattr(mewtwo_obj, "energyCards", [])
                for e in energies:
                    eid = e.get("id") if isinstance(e, dict) else getattr(e, "id", -1)
                    if eid == 1:
                        mewtwo_grass_count += 1

                if attached_card_id == 1:
                    if mewtwo_grass_count >= 1:
                        # Overcharging 2nd+ Green Energy on Mewtwo ex is forbidden!
                        bonus -= 0.35
                        triggered = "Mewtwo_Grass_Energy_Overcharge_Penalty"
                    else:
                        # 1st Green Energy is OK to fulfill 1 Colorless cost
                        bonus += 0.10
                        triggered = "Mewtwo_First_Grass_Energy_Colorless_OK"
        except Exception:
            pass

        # Articuno Energy Cap (Max 2 Energies): Articuno's Ice Beam attack costs 2 energy.
        # Attaching a 3rd+ energy to Articuno (Active or Bench) wastes energy.
        try:
            target_is_articuno = False
            articuno_current_e = 0
            if attach_index == 0 and my_ps is not None and my_ps.active and my_ps.active[0] is not None:
                a_obj = my_ps.active[0]
                aid = a_obj.get("id") if isinstance(a_obj, dict) else getattr(a_obj, "id", -1)
                if aid == ARTICUNO_ID:
                    target_is_articuno = True
                    articuno_current_e = len(a_obj.get("energyCards", [])) if isinstance(a_obj, dict) else len(getattr(a_obj, "energyCards", []))
            elif attach_index >= 1 and my_ps is not None:
                b_slot = attach_index - 1
                bench_list = getattr(my_ps, "bench", []) if not isinstance(my_ps, dict) else my_ps.get("bench", [])
                if b_slot < len(bench_list) and bench_list[b_slot] is not None:
                    b_obj = bench_list[b_slot]
                    bid = b_obj.get("id") if isinstance(b_obj, dict) else getattr(b_obj, "id", -1)
                    if bid == ARTICUNO_ID:
                        target_is_articuno = True
                        articuno_current_e = len(b_obj.get("energyCards", [])) if isinstance(b_obj, dict) else len(getattr(b_obj, "energyCards", []))

            if target_is_articuno and articuno_current_e >= 2:
                bonus -= 0.30
                triggered = "Articuno_Energy_Cap_Penalty"
        except Exception:
            pass

        # -------------------------------------------------------------------------
        # Articuno Role: NON-ATTACKER Defensive Wall / 1-Energy Pivot
        # - Articuno retreat cost = 1 energy (only 1 energy needed to retreat)
        # - STRICT ENERGY CAP = 1 Energy Max on Articuno!
        # - Attaching a 2nd+ energy to Articuno is a severe waste of energy.
        # -------------------------------------------------------------------------
        try:
            target_is_articuno = False
            articuno_current_e = 0
            if attach_index == 0 and my_ps is not None and my_ps.active and my_ps.active[0] is not None:
                a_obj = my_ps.active[0]
                aid = a_obj.get("id") if isinstance(a_obj, dict) else getattr(a_obj, "id", -1)
                if aid == ARTICUNO_ID:
                    target_is_articuno = True
                    articuno_current_e = len(a_obj.get("energyCards", [])) if isinstance(a_obj, dict) else len(getattr(a_obj, "energyCards", []))
            elif attach_index >= 1 and my_ps is not None:
                b_slot = attach_index - 1
                bench_list = getattr(my_ps, "bench", []) if not isinstance(my_ps, dict) else my_ps.get("bench", [])
                if b_slot < len(bench_list) and bench_list[b_slot] is not None:
                    b_obj = bench_list[b_slot]
                    bid = b_obj.get("id") if isinstance(b_obj, dict) else getattr(b_obj, "id", -1)
                    if bid == ARTICUNO_ID:
                        target_is_articuno = True
                        articuno_current_e = len(b_obj.get("energyCards", [])) if isinstance(b_obj, dict) else len(getattr(b_obj, "energyCards", []))

            if target_is_articuno:
                if articuno_current_e >= 1:
                    # STRICT CAP: Articuno never needs more than 1 energy (retreat cost = 1)
                    bonus -= 0.35
                    triggered = "Articuno_Max_1_Energy_Cap_Penalty"
                elif articuno_current_e == 0 and attach_index == 0:
                    # 1st energy on active Articuno is OK to enable 1-energy retreat
                    bonus += 0.05
                    triggered = "Articuno_1Energy_Retreat_Prep"
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 4f. Brilliant Articuno Repelling Veil Matchup Play & 1-Energy Pivot
    #
    # Articuno's Repelling Veil ability prevents ALL attack effects (like damage
    # counters) on your Basic TR Pokémon (Tarountula/Mimikyu/Mewtwo ex).
    #
    # Key Matchups where Repelling Veil is ⭐⭐⭐⭐⭐ Brilliant:
    #   1. Dragapult ex — stops Phantom Dive 6 damage counters on bench Basics
    #   2. Alakazam ex — blocks damage counter placement / manipulation
    #   3. Sableye / Spread decks — negates spread damage counter effects
    # -------------------------------------------------------------------------
    if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == ARTICUNO_ID:
        is_snipe_effect_deck = any(kw in opp_lower for kw in ["dragapult", "dreepy", "drakloak", "alakazam", "abra", "sableye", "spread", "shrouded"])
        if is_snipe_effect_deck:
            bonus += 0.35  # Brilliant Play: Priority bench Articuno to block Phantom Dive / spread effects!
            triggered = "Articuno_RepellingVeil_SnipeGuard_BrilliantPlay"
        elif my_bench_count < 4:
            bonus += 0.15
            triggered = "Articuno_Repelling_Veil_Setup"

    # -------------------------------------------------------------------------
    # 4f-2. Articuno 1-Energy Pivot Retreat to Bench Attacker
    #
    # When active Articuno has 1 energy (its retreat cost), retreat it immediately
    # if a benched attacker (Spidops with >=2E or Mewtwo ex with >=3E) is ready!
    # -------------------------------------------------------------------------
    if opt_type in (12, OptionType.RETREAT) and my_active_id == ARTICUNO_ID:
        if my_active_energy >= 1:
            benched_attacker_ready = False
            try:
                if my_ps is not None:
                    bench_list = getattr(my_ps, "bench", []) if not isinstance(my_ps, dict) else my_ps.get("bench", [])
                    for b in bench_list:
                        if b is not None:
                            bid = b.get("id") if isinstance(b, dict) else getattr(b, "id", -1)
                            be = len(b.get("energyCards", [])) if isinstance(b, dict) else len(getattr(b, "energyCards", []))
                            if (bid == SPIDOPS_ID and be >= 2) or (bid == MEWTWO_EX_ID and be >= 3):
                                benched_attacker_ready = True
                                break
            except Exception:
                benched_attacker_ready = True

            if benched_attacker_ready:
                bonus += 0.35  # Retreat Articuno with 1E to bring out ready bench attacker!
                triggered = "Articuno_Pivot_Retreat_To_Bench_Attacker"

    # -------------------------------------------------------------------------
    # 4g. Mimikyu Free-Retreat Pivot
    # -------------------------------------------------------------------------
    if opt_type in (12, OptionType.RETREAT) and my_active_id == MIMIKYU_ID:
        bonus += 0.15
        triggered = "Mimikyu_Free_Retreat_Pivot"

    # -------------------------------------------------------------------------
    # 4g-2. Spidops Attack Priority — Penalize Retreating Spidops With Energy
    #
    # Retreat costs (real card values):
    #   Mimikyu=0 (free), Tarountula=1, Articuno=1, Spidops=2, Mewtwo ex=3
    #
    # When Spidops is active with >= 2 energies, it can attack NOW:
    #   - Silky String: 2G -> 60 dmg
    #   - String Bomb:  3G -> 90 dmg
    # Retreating discards 2 energies (the retreat cost), wasting build-up turns.
    # -------------------------------------------------------------------------
    DECK_RETREAT_COSTS = {
        MIMIKYU_ID:    0,  # Free retreat
        TAROUNTULA_ID: 1,
        ARTICUNO_ID:   1,
        SPIDOPS_ID:    2,
        MEWTWO_EX_ID:  3,
    }
    active_retreat_cost = DECK_RETREAT_COSTS.get(my_active_id, 1)

    if opt_type in (13, OptionType.ATTACK) and my_active_id == SPIDOPS_ID:
        # Bonus for choosing to ATTACK with Spidops when it already has energy
        if my_active_energy >= 3:
            bonus += 0.35  # String Bomb ready — attack now!
            triggered = "Spidops_StringBomb_Attack_Now"
        elif my_active_energy >= 2:
            bonus += 0.25  # Silky String ready — attack now!
            triggered = "Spidops_SilkyString_Attack_Now"

    if opt_type in (12, OptionType.RETREAT) and my_active_id == SPIDOPS_ID:
        # Penalty: Retreating Spidops wastes energies equal to its retreat cost (2)
        energy_wasted = min(my_active_energy, active_retreat_cost)
        if energy_wasted >= 2:
            bonus -= 0.50  # Strong: 2 energies thrown away when could attack!
            triggered = "Spidops_Retreat_Energy_Waste"
        elif energy_wasted == 1:
            bonus -= 0.20
            triggered = "Spidops_Retreat_Minor_Waste"

    # -------------------------------------------------------------------------
    # 4g-3. Mewtwo ex Attack Priority — Penalize Retreating Mewtwo ex With Energy
    #
    # Mewtwo ex retreat cost = 3 energies.
    # Mewtwo ex attacks:
    #   - Psychic Burst: 3G -> 60 + 20x opponent energies
    #   - Power Saver:   4G -> 280 damage (requires 4 TR Pokemon in play)
    #
    # When Mewtwo ex has >= 3 energies, it can attack NOW with Psychic Burst.
    # When Mewtwo ex has 4 energies + TR bench ready, Power Saver 1-shots anything.
    # Retreating with 3 energies discards ALL 3 (retreat cost = 3) — huge waste!
    # -------------------------------------------------------------------------
    if opt_type in (13, OptionType.ATTACK) and my_active_id == MEWTWO_EX_ID:
        # Count how many TR Pokemon are in play (active + bench)
        tr_ids = frozenset({MEWTWO_EX_ID, SPIDOPS_ID, TAROUNTULA_ID, ARTICUNO_ID, MIMIKYU_ID})
        try:
            bench_ids_set = set()
            if my_ps is not None:
                bench_list = getattr(my_ps, "bench", []) if not isinstance(my_ps, dict) else my_ps.get("bench", [])
                for b in bench_list:
                    if b is not None:
                        bid = b.get("id") if isinstance(b, dict) else getattr(b, "id", -1)
                        bench_ids_set.add(bid)
            tr_in_play = 1 + sum(1 for bid in bench_ids_set if bid in tr_ids)  # 1 = active Mewtwo
        except Exception:
            tr_in_play = 1

        if my_active_energy >= 4 and tr_in_play >= 4:
            bonus += 0.40  # Power Saver ready — 280 damage 1-shot, ATTACK NOW!
            triggered = "Mewtwo_PowerSaver_Attack_Now"
        elif my_active_energy >= 3:
            bonus += 0.30  # Psychic Burst ready — attack now!
            triggered = "Mewtwo_PsychicBurst_Attack_Now"

    if opt_type in (12, OptionType.RETREAT) and my_active_id == MEWTWO_EX_ID:
        # Mewtwo ex retreat cost = 3, so any retreat with energy is costly.
        # The existing train.py KO-prevention guard handles the -0.80 penalty when
        # the opponent can be KO'd; here we add the energy-waste component.
        mewtwo_retreat_cost = DECK_RETREAT_COSTS.get(MEWTWO_EX_ID, 3)
        energy_wasted = min(my_active_energy, mewtwo_retreat_cost)
        if energy_wasted >= 3:
            bonus -= 0.40  # Retreating discards ALL 3 energies — severe waste!
            triggered = "Mewtwo_Retreat_Full_Energy_Waste"
        elif energy_wasted == 2:
            bonus -= 0.25
            triggered = "Mewtwo_Retreat_Partial_Waste"
        elif energy_wasted == 1:
            bonus -= 0.10
            triggered = "Mewtwo_Retreat_Minor_Waste"

    # -------------------------------------------------------------------------
    # 4h. Universal Crustle / ex Immunity Guard (Spidops Priority & ex Retreat)
    # -------------------------------------------------------------------------
    is_opp_crustle_active = (opp_active_id == CRUSTLE_ID) or ("crustle" in opp_lower)
    is_my_active_ex = (my_active_id == MEWTWO_EX_ID)

    if is_opp_crustle_active:
        # Penalty: NEVER attack Crustle with an ex Pokemon (Mewtwo ex deals 0 damage!)
        if opt_type in (13, OptionType.ATTACK) and is_my_active_ex:
            bonus -= 0.50
            triggered = "Crustle_EX_Immunity_Attack_Penalty"

        # Reward: Attack Crustle with Spidops (non-ex 1-shot KO!)
        elif opt_type in (13, OptionType.ATTACK) and my_active_id == SPIDOPS_ID:
            bonus += 0.20
            triggered = "Spidops_Attack_Crustle_KO"

        # Reward: Retreat or Switch away from ex attacker to Spidops/non-ex
        elif opt_type in (12, OptionType.RETREAT) and is_my_active_ex:
            bonus += 0.20
            triggered = "Retreat_EX_Vs_Crustle"
        elif opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) in (1123, 1218) and is_my_active_ex:
            bonus += 0.20
            triggered = "Switch_EX_Vs_Crustle"

        # Reward: Priority Spidops Evolution & Energy Attachment when facing Crustle
        elif opt_type in (9, OptionType.EVOLVE) and getattr(option, "cardId", -1) == SPIDOPS_ID:
            bonus += 0.20
            triggered = "Spidops_Evolution_Vs_Crustle"
        elif opt_type in (8, OptionType.ATTACH):
            attach_index = getattr(option, "inPlayIndex", -1)
            target_is_spidops = False
            try:
                if my_ps is not None:
                    if attach_index == 0 and my_ps.active and my_ps.active[0] is not None:
                        target_is_spidops = my_ps.active[0].id in (SPIDOPS_ID, TAROUNTULA_ID)
                    elif attach_index >= 1:
                        b_idx = attach_index - 1
                        if b_idx < len(my_ps.bench) and my_ps.bench[b_idx] is not None:
                            target_is_spidops = my_ps.bench[b_idx].id in (SPIDOPS_ID, TAROUNTULA_ID)
            except Exception:
                target_is_spidops = True
            if target_is_spidops:
                bonus += 0.20
                triggered = "Spidops_Energy_Vs_Crustle"

    # -------------------------------------------------------------------------
    # 4i. Universal Mega Starmie ex Counter (Giovanni Gust & Bench Protection)
    # -------------------------------------------------------------------------
    is_opp_starmie = (opp_active_id == STARMIE_EX_ID) or ("starmie" in opp_lower)

    if is_opp_starmie:
        # 1. Giovanni (1218) Double Switch / Gust: Pull benched Staryu (70 HP) active to KO before it evolves!
        if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1218:
            if len(opp_bench_ids) > 0:
                bonus += 0.25
                triggered = "Giovanni_Gust_Staryu_Denial"

        # 2. Priority evolution (Tarountula -> Spidops) to survive 50 bench snipe from Jetting Blow
        elif opt_type in (9, OptionType.EVOLVE) and getattr(option, "cardId", -1) == SPIDOPS_ID:
            bonus += 0.25
            triggered = "Spidops_Evolve_Vs_Starmie_Snipe_Guard"

        # 3a. Brave Bangle (+30 vs ex) — ONLY valid on Spidops/Articuno (non-Rule-Box attackers)
        #     Mewtwo ex has a Rule Box and CANNOT benefit from Brave Bangle!
        elif opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1175:
            if my_active_id in (SPIDOPS_ID, ARTICUNO_ID):
                bonus += 0.20
                triggered = "Brave_Bangle_Non_Ex_Attacker_Vs_Starmie"

        # 3b. Hero's Cape (+100 HP) — valid on any Pokémon
        elif opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1159:
            bonus += 0.20
            triggered = "Hero_Cape_Vs_Starmie"

    # -------------------------------------------------------------------------
    # 5. Kangaskhan / Crustle Counter — One-Shot Preparation & Avoid Low Damage
    # -------------------------------------------------------------------------
    elif "kangaskhan" in opp_lower or "crustle" in opp_lower:
        if opt_type in (8, OptionType.ATTACH):
            # Context guard: energy attachment bonus only when going to a key attacker
            target_is_attacker = False
            try:
                attach_index = getattr(option, "inPlayIndex", -1)
                if my_ps is not None:
                    if attach_index == 0 and my_ps.active and my_ps.active[0] is not None:
                        target_is_attacker = my_ps.active[0].id in (MEWTWO_EX_ID, SPIDOPS_ID)
                    elif attach_index >= 1:
                        bench_idx = attach_index - 1
                        if bench_idx < len(my_ps.bench) and my_ps.bench[bench_idx] is not None:
                            target_is_attacker = my_ps.bench[bench_idx].id in (MEWTWO_EX_ID, SPIDOPS_ID)
            except Exception:
                target_is_attacker = True  # Fail open: give benefit of doubt

            if target_is_attacker:
                bonus += 0.10
                triggered = "Mewtwo_Energy_Prep"

        elif opt_type in (7, OptionType.PLAY):
            card_id = getattr(option, "cardId", -1)
            if card_id == 1218:  # Giovanni Double Switch / Gust — gust opponent's best bench threat
                if len(opp_bench_ids) > 0:
                    bonus += 0.15
                    triggered = "Giovanni_Gust_Kangaskhan"
                elif game_phase == "late":
                    bonus += 0.08
                    triggered = "Giovanni_Gust_Late"

    # -------------------------------------------------------------------------
    # 6. Lucario Counter — Psychic Weakness Exploitation
    # -------------------------------------------------------------------------
    elif "lucario" in opp_lower or "riolu" in opp_lower:
        if opt_type in (13, OptionType.ATTACK):
            # Context guard: bonus only when opponent is weak target and we can attack
            opp_is_weak_target = opp_active_id in LUCARIO_LINE_IDS
            we_are_mewtwo      = my_active_id == MEWTWO_EX_ID
            we_can_attack      = my_active_energy >= 2

            if we_are_mewtwo and opp_is_weak_target and we_can_attack:
                bonus += 0.15
                triggered = "Psychic_Weakness_KO"
            elif opp_is_weak_target and we_can_attack:
                # We can still exploit weakness with another attacker
                bonus += 0.08
                triggered = "Weakness_KO_Other_Attacker"

    # -------------------------------------------------------------------------
    # 7. Mewtwo / Wobbuffet Counter — Gust Target / Bypass Wall
    # -------------------------------------------------------------------------
    elif "wobbuffet" in opp_lower or "mewtwo" in opp_lower:
        if opt_type in (7, OptionType.PLAY):
            card_id = getattr(option, "cardId", -1)
            if card_id in (1219, 1123):  # Petrel (gust) / Switch
                # Context guard: only reward gusting when there are better bench targets.
                # If Wobbuffet is the only Pokémon, attacking it directly may be correct.
                opp_has_bench_targets = len(opp_bench_ids) > 0
                if opp_has_bench_targets:
                    bonus += 0.15
                    triggered = "Bypass_Wobbuffet_Gust"

    # -------------------------------------------------------------------------
    # 8. Starmie Counter — Early Staryu KO
    # -------------------------------------------------------------------------
    elif "starmie" in opp_lower:
        if opt_type in (13, OptionType.ATTACK):
            target_name = str(getattr(option, "targetName", "")).lower()
            is_staryu_target = "staryu" in target_name

            # Context guard: Staryu denial is most impactful before Starmie sets up
            opp_has_starmie_line = bool(STARMIE_LINE_IDS & opp_pokemon_ids)

            if is_staryu_target and opp_has_starmie_line and game_phase in ("early", "mid"):
                bonus += 0.10
                triggered = "Early_Staryu_KO"

    # -------------------------------------------------------------------------
    # 9. Trevenant Counter — Defensive Tool Attachment
    # -------------------------------------------------------------------------
    elif "trevenant" in opp_lower:
        if opt_type in (7, OptionType.PLAY):
            card_id = getattr(option, "cardId", -1)
            if card_id in (1159, 1175):  # Hero's Cape / Brave Bangle
                # Tools are most valuable before our Pokémon are damaged
                if game_phase in ("early", "mid"):
                    bonus += 0.10
                    triggered = "Defensive_Tool_Attach"

    # -------------------------------------------------------------------------
    # 10. Alakazam Counter — Abra Denial & Anti-Bench-Overextension
    # -------------------------------------------------------------------------
    elif "alakazam" in opp_lower:
        if opt_type in (13, OptionType.ATTACK):
            target_name = str(getattr(option, "targetName", "")).lower()
            is_abra_target = "abra" in target_name or "kadabra" in target_name

            # Context guard: denying Abra only matters if opponent has Alakazam setup potential
            opp_has_alakazam_line = bool(ALAKAZAM_LINE_IDS & opp_pokemon_ids)
            meaningful_denial = opp_has_alakazam_line and opp_prizes <= 5

            if is_abra_target and meaningful_denial:
                bonus += 0.15
                triggered = "KO_Abra_Denial"

        # Bench Overextension Guard vs Alakazam (Alakazam deals extra damage per benched Pokemon)
        elif opt_type in (7, OptionType.PLAY):
            if my_bench_count >= 3:
                bonus -= 0.10
                triggered = "Alakazam_Bench_Limit_Guard"

    # -------------------------------------------------------------------------
    # NEW 1. Hero's Cape Placement Rules (from 88413528 / 88414072 loss analysis)
    #
    # Hero's Cape (+100 HP ACE SPEC) MUST go on Mewtwo-ex[TR] (280->380HP).
    # Placing it on Tarountula (50HP basic with 0 attacks) is a wasted ACE SPEC.
    # -------------------------------------------------------------------------
    if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == HEROS_CAPE_ID:
        # Detect the target of the tool attachment
        try:
            tool_target_id = -1
            if my_ps is not None:
                active_list = getattr(my_ps, "active", []) if not isinstance(my_ps, dict) else my_ps.get("active", [])
                if active_list and active_list[0] is not None:
                    tool_target_id = active_list[0].get("id") if isinstance(active_list[0], dict) else getattr(active_list[0], "id", -1)
        except Exception:
            tool_target_id = -1

        if tool_target_id == MEWTWO_EX_ID:
            bonus += 0.35
            triggered = "HeroCape_Correct_Target_MewtwoEx"
        elif tool_target_id in (TAROUNTULA_ID, MIMIKYU_ID):
            # WRONG: Hero's Cape on a basic non-attacker wastes the ACE SPEC
            bonus -= 0.40
            triggered = "HeroCape_Wrong_Target_NonAttacker"

    # -------------------------------------------------------------------------
    # NEW 2. Brave Bangle Correct Placement (from 88415303 loss analysis)
    #
    # Brave Bangle (+30 dmg vs ex Pokémon) ONLY works for non-ex attackers.
    # Mewtwo-ex has a Rule Box so Brave Bangle does NOT apply to it.
    # Best targets: Spidops (130HP, attacks ex) > Articuno (120HP) > Tarountula.
    # Worst target: Mimikyu (60HP, no realistic attacking role).
    # -------------------------------------------------------------------------
    if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == BRAVE_BANGLE_ID:
        try:
            bangle_target_id = -1
            if my_ps is not None:
                active_list = getattr(my_ps, "active", []) if not isinstance(my_ps, dict) else my_ps.get("active", [])
                if active_list and active_list[0] is not None:
                    bangle_target_id = active_list[0].get("id") if isinstance(active_list[0], dict) else getattr(active_list[0], "id", -1)
        except Exception:
            bangle_target_id = -1

        if bangle_target_id == SPIDOPS_ID:
            bonus += 0.25
            triggered = "BraveBangle_Correct_Spidops"
        elif bangle_target_id == ARTICUNO_ID:
            bonus += 0.10
            triggered = "BraveBangle_Articuno_Acceptable"
        elif bangle_target_id == MIMIKYU_ID:
            bonus -= 0.25
            triggered = "BraveBangle_Wrong_Target_Mimikyu"

    # -------------------------------------------------------------------------
    # NEW 3. Grimmsnarl-ex Matchup Counter Rules (from 88413528 / 88415303 losses)
    #
    # Strategy vs Grimmsnarl-ex / Munkidori spread deck:
    # 1. KO Munkidori FIRST — it spreads Poison and charges Grimmsnarl-ex on bench.
    # 2. Replace Spikemuth Gym immediately — it gives +30 to all Dark attacks.
    # 3. Giovanni gust on Morgrem/Impidimp before they evolve to Grimmsnarl-ex.
    # -------------------------------------------------------------------------
    is_grimmsnarl_opponent = (
        any(kw in opp_lower for kw in ["grimmsnarl", "impidimp", "morgrem", "munkidori"])
        or bool(GRIMMSNARL_LINE_IDS & opp_pokemon_ids)
        or MUNKIDORI_ID in opp_pokemon_ids
    )

    if is_grimmsnarl_opponent:
        # 3a. Reward attacking Munkidori — Munkidori spread stops when it is KO'd
        if opt_type in (13, OptionType.ATTACK) and opp_active_id == MUNKIDORI_ID:
            if my_active_energy >= 1:
                bonus += 0.30
                triggered = "Grimmsnarl_Counter_KO_Munkidori_Active"

        # 3b. Reward Giovanni gust on evolving Impidimp/Morgrem (deny Grimmsnarl-ex)
        elif opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1218:  # TR-Giovanni
            opp_has_impidimp_morgrem = bool({IMPIDIMP_ID, MORGREM_ID} & set(opp_bench_ids))
            if opp_has_impidimp_morgrem and game_phase in ("early", "mid"):
                bonus += 0.30
                triggered = "Grimmsnarl_Counter_Giovanni_Gust_Evolution_Line"

        # 3c. Reward replacing Spikemuth Gym with TR-Factory (deny +30 Dark)
        elif opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == TR_FACTORY_ID:
            try:
                current = getattr(obs, "current", None)
                if current is not None:
                    stadium_list = getattr(current, "stadium", []) if not isinstance(current, dict) else current.get("stadium", [])
                    active_stadium_ids = [s.get("id") if isinstance(s, dict) else getattr(s, "id", -1) for s in stadium_list]
                    if SPIKEMUTH_GYM_ID in active_stadium_ids:
                        bonus += 0.35
                        triggered = "Grimmsnarl_Replace_Spikemuth_Gym"
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # NEW 4. Attack Stalling Basics Immediately (from 88414072 water ramp loss)
    #
    # Opponent stalling with a 70HP basic for 8 turns while building 400HP
    # Stage 2 ex walls is a losing line. ATTACK the stalling basic immediately!
    # If opponent active <= 100 HP and we have energy, attack to take early prizes.
    # -------------------------------------------------------------------------
    if opt_type in (13, OptionType.ATTACK) and my_active_energy >= 1:
        try:
            current = getattr(obs, "current", None)
            if current is not None:
                yi = getattr(current, "yourIndex", 0) if not isinstance(current, dict) else current.get("yourIndex", 0)
                players = getattr(current, "players", []) if not isinstance(current, dict) else current.get("players", [])
                if len(players) >= 2:
                    opp_ps = players[1 - yi]
                    opp_active_list = getattr(opp_ps, "active", []) if not isinstance(opp_ps, dict) else opp_ps.get("active", [])
                    if opp_active_list and opp_active_list[0] is not None:
                        opp_act = opp_active_list[0]
                        opp_act_hp = opp_act.get("hp", 999) if isinstance(opp_act, dict) else getattr(opp_act, "hp", 999)
                        opp_act_max_hp = opp_act.get("maxHp", 999) if isinstance(opp_act, dict) else getattr(opp_act, "maxHp", 999)
                        # Reward attacking if opponent active is a low-HP basic (stalling)
                        if opp_act_max_hp <= 100 and game_phase in ("early", "mid") and opp_prizes >= 4:
                            if triggered == "NONE":  # Only apply if no stronger rule already fired
                                bonus += 0.20
                                triggered = "Attack_Stalling_Low_HP_Basic"
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # NEW 5. Free-Retreat Pivot When Active Has 0 Energy (from 88415303 loss)
    #
    # When active Pokémon has 0 energy and faces a high-HP threat, use Mimikyu
    # (free retreat) or retreat to preserve position. Staying in active with 0
    # energy means we cannot attack or deal damage.
    # -------------------------------------------------------------------------
    if opt_type in (12, OptionType.RETREAT) and my_active_energy == 0:
        # Retreating a 0-energy active is almost always correct (it cannot attack)
        if my_active_id in (TAROUNTULA_ID, MIMIKYU_ID, ARTICUNO_ID):
            if my_active_id == MIMIKYU_ID:
                bonus += 0.25  # Free retreat — no energy cost at all
                triggered = "Pivot_Mimikyu_Free_Retreat_0Energy"
            elif my_active_id in (TAROUNTULA_ID, ARTICUNO_ID):
                bonus += 0.15  # 1 energy retreat — small but correct
                triggered = "Pivot_Basic_Retreat_0Energy"

    # -------------------------------------------------------------------------
    # NEW 6. Giovanni Gust on Evolving Stage 1 (universal — vs water ramp / other evo decks)
    #
    # When opponent has a Stage 1 Pokémon on the bench that could evolve to a
    # 300-400HP Stage 2 ex, gusting it with Giovanni and KO'ing it is priority.
    # -------------------------------------------------------------------------
    if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1218:  # TR-Giovanni
        if not is_grimmsnarl_opponent:  # Grimmsnarl rule already handled this above
            # Check if opponent bench has a Stage 1 evolution Pokémon (medium HP)
            try:
                current = getattr(obs, "current", None)
                if current is not None:
                    yi = getattr(current, "yourIndex", 0) if not isinstance(current, dict) else current.get("yourIndex", 0)
                    players = getattr(current, "players", []) if not isinstance(current, dict) else current.get("players", [])
                    if len(players) >= 2:
                        opp_ps = players[1 - yi]
                        opp_bench = getattr(opp_ps, "bench", []) if not isinstance(opp_ps, dict) else opp_ps.get("bench", [])
                        for b in opp_bench:
                            if b is not None:
                                b_hp = b.get("maxHp", 0) if isinstance(b, dict) else getattr(b, "maxHp", 0)
                                b_cur_hp = b.get("hp", 0) if isinstance(b, dict) else getattr(b, "hp", 0)
                                # A Stage 1 typically has 80-130 HP range
                                if 80 <= b_hp <= 150 and game_phase in ("early", "mid") and opp_prizes >= 4:
                                    bonus += 0.20
                                    triggered = "Giovanni_Gust_Evolving_Stage1"
                                    break
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # NEW 7. Full Metal Lab / Defense Stadium Contestation (Replay 88472281)
    # -------------------------------------------------------------------------
    if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) in (1257, 1258, 1259):
        try:
            current = getattr(obs, "current", None)
            if current is not None:
                stadium_list = getattr(current, "stadium", []) if not isinstance(current, dict) else current.get("stadium", [])
                active_stadium_ids = [s.get("id") if isinstance(s, dict) else getattr(s, "id", -1) for s in stadium_list]
                if 1244 in active_stadium_ids:  # Full Metal Lab
                    bonus += 0.30
                    triggered = "Override_Full_Metal_Lab"
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # NEW 8. Dusknoir Cursed Blast Burst Denial (Replay 88474377 / 88485012)
    # -------------------------------------------------------------------------
    is_dusknoir_opponent = any(kw in opp_lower for kw in ["dusknoir", "dusclops", "duskull"])
    if is_dusknoir_opponent:
        if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1218:  # Giovanni Gust
            if any(k in str(getattr(option, "targetName", "")).lower() for k in ["duskull", "dusclops"]):
                bonus += 0.30
                triggered = "Giovanni_Gust_Duskull_Denial"

    # -------------------------------------------------------------------------
    # NEW 9. Spiky Energy Recoil Suicide Guard (Replay 88505471)
    # -------------------------------------------------------------------------
    if opt_type in (13, OptionType.ATTACK):
        try:
            current = getattr(obs, "current", None)
            if current is not None:
                yi = getattr(current, "yourIndex", 0) if not isinstance(current, dict) else current.get("yourIndex", 0)
                players = getattr(current, "players", []) if not isinstance(current, dict) else current.get("players", [])
                if len(players) >= 2:
                    my_ps = players[yi]
                    opp_ps = players[1 - yi]
                    my_act = my_ps.active[0] if my_ps.active and my_ps.active[0] else None
                    opp_act = opp_ps.active[0] if opp_ps.active and opp_ps.active[0] else None
                    if my_act and opp_act:
                        my_hp = my_act.get("hp", 999) if isinstance(my_act, dict) else getattr(my_act, "hp", 999)
                        tools = opp_act.get("tools", []) if isinstance(opp_act, dict) else getattr(opp_act, "tools", [])
                        tool_names = [str(t.get("name", "")).lower() for t in tools if t]
                        if any("spiky" in tn for tn in tool_names) and my_hp <= 50:
                            bonus -= 0.40
                            triggered = "Spiky_Energy_Recoil_Suicide_Guard"
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # NEW 10. Mega Lucario ex Counter Blueprint (Psychic Weakness & Riolu Gust)
    # -------------------------------------------------------------------------
    is_lucario_opp = any(kw in opp_lower for kw in ["lucario", "riolu", "solrock", "lunatone"])
    if is_lucario_opp:
        card_id = getattr(option, "cardId", -1)
        # Priority 1: Gust Riolu or Solrock/Lunatone via Giovanni Gust before Lucario ex evolves
        if opt_type in (7, OptionType.PLAY) and card_id == 1218:  # Giovanni Gust
            target_name = str(getattr(option, "targetName", "")).lower()
            if any(k in target_name for k in ["riolu", "solrock", "lunatone"]):
                bonus += 0.30
                triggered = "Giovanni_Gust_Riolu_Solrock_Denial"
        # Priority 2: Charge TR Mewtwo ex (431) to exploit Lucario's 2x Psychic Weakness for 3-prize KO
        elif opt_type in (8, OptionType.ATTACH):
            target_area = getattr(option, "targetArea", None)
            # If target card is Mewtwo ex (431)
            target_cid = getattr(option, "cardIdTarget", -1)
            if target_cid == 431 or target_area == 0:
                bonus += 0.35
                triggered = "Mewtwo_Psychic_Weakness_Vs_Lucario"
        # Priority 3: Promote TR Mewtwo ex to Active when loaded to take 3-prize KO
        elif opt_type in (11, OptionType.RETREAT):
            promote_cid = getattr(option, "cardId", -1)
            if promote_cid == 431:
                bonus += 0.30
                triggered = "Mewtwo_Promote_3Prize_Vs_Lucario"

    # -------------------------------------------------------------------------
    # NEW 11. Cynthia's Gible Evolution Denial (Replay 88477096)
    # -------------------------------------------------------------------------
    if "garchomp" in opp_lower or "gible" in opp_lower:
        if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1218:  # Giovanni Gust
            if "gible" in str(getattr(option, "targetName", "")).lower():
                bonus += 0.25
                triggered = "Giovanni_Gust_Gible_Denial"

    # -------------------------------------------------------------------------
    # NEW 12. Grimmsnarl / Froslass Engine Denial (Match Analysis)
    # -------------------------------------------------------------------------
    is_grimmsnarl_froslass_opp = any(kw in opp_lower for kw in ["grimmsnarl", "froslass", "munkidori", "snorunt"])
    if is_grimmsnarl_froslass_opp:
        # Priority 1: Target Froslass or Snorunt via Giovanni Gust to shut down damage spread
        if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == 1218:  # Giovanni Gust
            target_name = str(getattr(option, "targetName", "")).lower()
            if any(k in target_name for k in ["froslass", "snorunt"]):
                bonus += 0.35
                triggered = "Giovanni_Gust_Froslass_Denial"
        # Priority 2: Maintain full bench resilience vs Devolution TM plays
        elif opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) in (400, 401, 414, 434):
            try:
                current = getattr(obs, "current", None)
                if current is not None:
                    yi = getattr(current, "yourIndex", 0) if not isinstance(current, dict) else current.get("yourIndex", 0)
                    players = getattr(current, "players", []) if not isinstance(current, dict) else current.get("players", [])
                    if len(players) >= 2:
                        my_ps = players[yi]
                        bench_cnt = len([b for b in my_ps.bench if b]) if hasattr(my_ps, "bench") else 0
                        if bench_cnt < 4:
                            bonus += 0.20
                            triggered = "Grimmsnarl_Bench_Resilience"
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # NEW 13. Dragapult ex Tabletop Counter (Articuno Shield, Mimikyu Wall & 320 KO)
    # -------------------------------------------------------------------------
    is_dragapult_opp = any(kw in opp_lower for kw in ["dragapult", "dreepy", "drakloak"])
    if is_dragapult_opp:
        card_id = getattr(option, "cardId", -1)
        # Priority 1: Bench TR Articuno (414) for Repelling Veil bench damage shield vs Phantom Dive
        if opt_type in (7, OptionType.PLAY) and card_id == 414:
            bonus += 0.35
            triggered = "Articuno_Bench_Shield_Vs_Dragapult"
        # Priority 2: Bench/Search TR Mimikyu (434) as an ex-wall in mid/late game
        elif opt_type in (7, OptionType.PLAY) and card_id == 434:
            bonus += 0.30
            triggered = "Mimikyu_Ex_Wall_Vs_Dragapult"
        # Priority 3: Attach Brave Bangle (1175) to TR Mewtwo ex for 320 damage one-shot KO on Dragapult ex
        elif opt_type in (7, OptionType.PLAY) and card_id == 1175:
            bonus += 0.30
            triggered = "Brave_Bangle_320KO_Vs_Dragapult"
        # Priority 4: Play TR Proton (1220) early for setup consistency
        elif opt_type in (7, OptionType.PLAY) and card_id == 1220 and game_phase == "early":
            bonus += 0.25
            triggered = "Proton_Early_Consistency"

    # -------------------------------------------------------------------------
    # NEW 14. Universal Tactical Anti-Mistake Guards (Replay 88624710)
    # -------------------------------------------------------------------------
    try:
        current = getattr(obs, "current", None)
        if current is not None:
            yi = getattr(current, "yourIndex", 0) if not isinstance(current, dict) else current.get("yourIndex", 0)
            players = getattr(current, "players", []) if not isinstance(current, dict) else current.get("players", [])
            if len(players) >= 2:
                my_ps = players[yi]
                deck_cnt = my_ps.get("deckCount", 60) if isinstance(my_ps, dict) else getattr(my_ps, "deckCount", 60)
                
                # 14A. Self Deck-Out Danger Guard (-0.40 / -0.50 penalty)
                if deck_cnt <= 6:
                    card_id = getattr(option, "cardId", -1)
                    if opt_type in (7, OptionType.PLAY) and card_id in (1227, 1094, 1152, 1257, 1216, 1219, 1220):
                        bonus -= 0.45 if deck_cnt <= 4 else 0.35
                        triggered = "Self_Deckout_Danger_Guard"
                
                # 14B. Pointless Retreat Without Attacking Guard (-0.35 penalty)
                if opt_type in (11, OptionType.RETREAT):
                    if has_attack_option:
                        bonus -= 0.35
                        triggered = "Pointless_Retreat_Guard"

                # 14C. Energy Overcharging Guard (-0.35 penalty for 4+ energy on single unit)
                if opt_type in (8, OptionType.ATTACH):
                    my_act = my_ps.active[0] if my_ps.active and my_ps.active[0] else None
                    if my_act:
                        act_en = len(my_act.get("energies", [])) if isinstance(my_act, dict) else len(getattr(my_act, "energies", []))
                        target_area = getattr(option, "targetArea", None)
                        # If attaching to active that already has 3+ energies when bench has 0 energy Pokémon
                        if target_area in (0, AreaType.ACTIVE) or getattr(option, "area", None) == 0:
                            if act_en >= 3:
                                bench_list = getattr(my_ps, "bench", []) if not isinstance(my_ps, dict) else my_ps.get("bench", [])
                                uncharged_bench = any((len(b.get("energies", [])) == 0 if isinstance(b, dict) else len(getattr(b, "energies", [])) == 0) for b in bench_list if b)
                                if uncharged_bench:
                                    bonus -= 0.35
                                    triggered = "Overcharge_Energy_Guard"
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # NEW 15. Simbozz TR Mewtwo Domination Blueprint (Power Saver & Board Flood)
    # -------------------------------------------------------------------------
    try:
        current = getattr(obs, "current", None)
        if current is not None:
            yi = getattr(current, "yourIndex", 0) if not isinstance(current, dict) else current.get("yourIndex", 0)
            players = getattr(current, "players", []) if not isinstance(current, dict) else current.get("players", [])
            if len(players) >= 2:
                my_ps = players[yi]
                # Count total Team Rocket Pokemon in play (active + bench)
                tr_pkmn_ids = (400, 401, 414, 431, 434)
                my_act_id = my_ps.active[0].get("id") if (my_ps.active and my_ps.active[0]) else -1
                bench_list = getattr(my_ps, "bench", []) if not isinstance(my_ps, dict) else my_ps.get("bench", [])
                bench_ids = [b.get("id") if isinstance(b, dict) else getattr(b, "id", -1) for b in bench_list if b]
                
                all_board_ids = ([my_act_id] if my_act_id > 0 else []) + bench_ids
                tr_board_cnt = sum(1 for cid in all_board_ids if cid in tr_pkmn_ids)
                
                # Rule 15A. Power Saver Bench Fill: Prioritize boarding TR Pokemon when TR count < 4
                if tr_board_cnt < 4 and opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) in (400, 414, 434):
                    bonus += 0.35
                    triggered = "Power_Saver_Bench_Fill_Bonus"
                
                # Rule 15B. Power Saver Safety: Avoid promoting Mewtwo ex if TR count < 4
                if tr_board_cnt < 4 and opt_type in (11, OptionType.RETREAT):
                    # If promoting Mewtwo ex while Power Saver is unsatisfied
                    target_cid = getattr(option, "cardId", -1)
                    if target_cid == 431:
                        bonus -= 0.30
                        triggered = "Power_Saver_Unsatisfied_Promote_Penalty"
                        
                # Rule 15C. Turn 1-2 Explosive Board Flood (Proton 1220 & Transceiver 1134)
                if game_phase == "early" and tr_board_cnt < 4 and opt_type in (7, OptionType.PLAY):
                    card_id = getattr(option, "cardId", -1)
                    if card_id in (1220, 1134):
                        bonus += 0.35
                        triggered = "Turn1_Board_Flood_Proton_Transceiver"
    except Exception:
        pass

    # Bound bonus within [-0.50, +0.40]: allow meaningful penalties for critical errors
    bonus = max(-0.50, min(0.40, bonus))
    return bonus, triggered


