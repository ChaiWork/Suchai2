from cg.api import CardType, OptionType, SelectContext

# Configuration Flags for Ablation Testing & Feature Control
USE_EXPERT_GUIDANCE = True
EXPERT_WEIGHT = 0.10

# Pokémon card IDs used in context checks
MEWTWO_EX_ID = 431
SPIDOPS_ID = 401
MIMIKYU_ID = 434
ARTICUNO_ID = 414

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
    }
    try:
        current = obs.current
        yi = current.yourIndex
        my_ps  = current.players[yi]
        opp_ps = current.players[1 - yi]

        ctx["my_prizes"]  = len(my_ps.prize)
        ctx["opp_prizes"] = len(opp_ps.prize)
        ctx["my_ps"]      = my_ps

        if my_ps.active:
            a = my_ps.active[0]
            if a is not None:
                ctx["my_active_id"]     = a.id
                ctx["my_active_energy"] = len(a.energyCards)

        if opp_ps.active:
            a = opp_ps.active[0]
            if a is not None:
                ctx["opp_active_id"] = a.id

        ctx["opp_bench_ids"] = [p.id for p in opp_ps.bench if p is not None]

        prizes = ctx["my_prizes"]
        if prizes <= 2:
            ctx["game_phase"] = "late"
        elif prizes <= 4:
            ctx["game_phase"] = "mid"
        else:
            ctx["game_phase"] = "early"

    except Exception:
        pass  # Return safe defaults on any attribute error

    return ctx


def get_expert_bonus(obs, option, opponent_name="unknown"):
    """
    Evaluates a context-aware heuristic bonus in range [-0.10, +0.20] for an
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

    # Convenience: full set of visible opponent Pokémon IDs
    opp_pokemon_ids = set(opp_bench_ids) | {opp_active_id}

    # -------------------------------------------------------------------------
    # 1. Dragapult Counter — Evolution Denial / Dreepy-Drakloak KO
    # -------------------------------------------------------------------------
    if "dragapult" in opp_lower:
        if opt_type in (13, OptionType.ATTACK):
            target_name = str(getattr(option, "targetName", "")).lower()
            is_denial_target = any(k in target_name for k in ["dreepy", "drakloak"])

            # Context guard: only reward if opponent has meaningful evolution potential
            opp_has_evolution = bool(DRAGAPULT_LINE_IDS & opp_pokemon_ids)

            if is_denial_target and opp_has_evolution:
                # Bigger bonus in mid-game when evolution denial has maximum impact
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
    # 4c. Universal Turn 1-2 Emergency Bench Search (Prevent Lone Active Wipe)
    # -------------------------------------------------------------------------
    my_bench_count = 0
    try:
        if my_ps is not None:
            my_bench_count = len([p for p in my_ps.bench if p is not None])
    except Exception:
        pass

    if my_bench_count == 0 and opt_type in (7, OptionType.PLAY):
        card_id = getattr(option, "cardId", -1)
        if card_id in (1134, 1086, 1220):  # Transceiver, Poffin, Proton
            bonus += 0.20
            triggered = "Turn1_Emergency_Bench_Search"

    # -------------------------------------------------------------------------
    # 4d. Priority Evolution (Tarountula -> Spidops)
    # -------------------------------------------------------------------------
    if opt_type in (9, OptionType.EVOLVE):
        card_id = getattr(option, "cardId", -1)
        if card_id == 401:  # Spidops
            bonus += 0.15
            triggered = "Priority_Spidops_Evolution"

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
            if card_id == 1218:  # Giovanni +40 dmg
                # Context guard: only valuable when Mewtwo has enough energy to attack
                if my_active_id == MEWTWO_EX_ID and my_active_energy >= 2:
                    bonus += 0.15
                    triggered = "Giovanni_OneShot_Boost"
                elif game_phase == "late":
                    # Late game: Giovanni is almost always correct
                    bonus += 0.08
                    triggered = "Giovanni_OneShot_Boost_Late"

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
            my_bench_count = 0
            try:
                if my_ps is not None:
                    my_bench_count = len([p for p in my_ps.bench if p is not None])
            except Exception:
                pass
            if my_bench_count >= 3:
                bonus -= 0.10
                triggered = "Alakazam_Bench_Limit_Guard"

    # Strictly bound bonus score within [-0.10, +0.20]
    bonus = max(-0.10, min(0.20, bonus))
    return bonus, triggered
