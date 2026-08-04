from cg.api import CardType, OptionType, SelectContext

# Configuration Flags for Ablation Testing & Feature Control
USE_EXPERT_GUIDANCE = True
EXPERT_WEIGHT = 0.10

# Pokémon card IDs used in context checks
MEWTWO_EX_ID = 431
SPIDOPS_ID = 401
MIMIKYU_ID = 434
ARTICUNO_ID = 414
TAROUNTULA_ID = 400
CRUSTLE_ID = 345
STARMIE_EX_ID = 1031

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
                ctx["my_bench_count"] = sum(1 for p in my_bench if p is not None)

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


def get_expert_bonus(obs, option, opponent_name="unknown"):
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
    opp_active_id    = ctx["opp_active_id"]
    opp_bench_ids    = ctx["opp_bench_ids"]
    my_active_energy = ctx["my_active_energy"]
    game_phase       = ctx["game_phase"]
    my_ps            = ctx["my_ps"]
    my_bench_count   = ctx["my_bench_count"]
    tr_in_play       = ctx["tr_in_play"]
    factory_active   = ctx["factory_active"]
    turn             = ctx["turn"]

    # Convenience: full set of visible opponent Pokémon IDs
    opp_pokemon_ids = set(opp_bench_ids) | {opp_active_id}

    # -------------------------------------------------------------------------
    # 1. Dragapult ex Counter — Repelling Veil Bench Shield & Evolution Denial
    #
    # Dragapult ex's Phantom Dive attack deals 200 active damage + 60 damage
    # counters spread across benched Basics (one-shotting 50HP Tarountulas).
    #
    # Counter Strategy:
    # 1. Bench Articuno! Repelling Veil PREVENTS Phantom Dive damage counters
    #    from being placed on all Basic Team Rocket Pokemon on your bench!
    # 2. Retreat active Articuno to bench so Repelling Veil protects the board.
    # 3. Priority evolve Tarountula (50 HP) -> Spidops (130 HP) to survive.
    # 4. Evolution Denial: KO Dreepy/Drakloak before Dragapult ex sets up.
    # -------------------------------------------------------------------------
    is_dragapult_opponent = ("dragapult" in opp_lower) or bool(DRAGAPULT_LINE_IDS & opp_pokemon_ids)

    if is_dragapult_opponent:
        # 1. Priority Bench Articuno for Repelling Veil bench shield
        if opt_type in (7, OptionType.PLAY) and getattr(option, "cardId", -1) == ARTICUNO_ID:
            if my_bench_count < 5:
                bonus += 0.35
                triggered = "Articuno_RepellingVeil_Vs_Dragapult"

        # 2. Retreat Active Articuno to Bench so Repelling Veil stays active
        elif opt_type in (12, OptionType.RETREAT) and my_active_id == ARTICUNO_ID:
            bonus += 0.30
            triggered = "Articuno_Retreat_To_Bench_Vs_Dragapult"

        # 3. Priority Evolve Tarountula -> Spidops to exceed 60 HP snipe threshold
        elif opt_type in (9, OptionType.EVOLVE) and getattr(option, "cardId", -1) == SPIDOPS_ID:
            bonus += 0.35
            triggered = "Spidops_Evolve_To_Survive_Phantom_Dive"

        # 4. Evolution Denial: KO Dreepy/Drakloak on opponent bench before Dragapult ex evolves
        elif opt_type in (13, OptionType.ATTACK):
            target_name = str(getattr(option, "targetName", "")).lower()
            is_denial_target = any(k in target_name for k in ["dreepy", "drakloak"])
            if is_denial_target and bool(DRAGAPULT_LINE_IDS & opp_pokemon_ids):
                bonus += 0.15 if game_phase in ("mid", "late") else 0.08
                triggered = "KO_Dreepy_Drakloak"

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

    # Bound bonus within [-0.30, +0.40]: allow meaningful penalties for critical errors
    # (old ceiling was +0.20 which silently truncated important heuristics like Starmie gust +0.25)
    bonus = max(-0.30, min(0.40, bonus))
    return bonus, triggered
