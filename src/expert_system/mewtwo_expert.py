"""
Team Rocket Mewtwo ex + Spidops Refactored Heuristic Expert Engine.
Ground Truth: Matches 60-card CSV list (deck copy BARU CARD TAPI KECEWA.csv).
Provides small strategic action prior guidance (range [-0.20, +0.20]) for MCTS exploration.
"""
from cg.api import OptionType

# Card ID Constants (From Deck CSV Ground Truth)
BASIC_G_ENERGY_ID    = 1    # Basic {G} Energy
BASIC_P_ENERGY_ID    = 5    # Basic {P} Energy
TR_ENERGY_ID         = 15   # Team Rocket's Energy

TAROUNTULA_ID        = 400  # Team Rocket's Tarountula
SPIDOPS_ID           = 401  # Team Rocket's Spidops
ARTICUNO_ID          = 414  # Team Rocket's Articuno
MIMIKYU_ID           = 434  # Team Rocket's Mimikyu
MEWTWO_EX_ID         = 431  # Team Rocket's Mewtwo ex
SNEASEL_ID           = 464  # Team Rocket's Sneasel
CLEFAIRY_EX_ID       = 272  # Lillie's Clefairy ex

POFFIN_ID            = 1086 # Buddy-Buddy Poffin
SECRET_BOX_ID        = 1092 # Secret Box (ACE SPEC)
BUG_CATCHING_SET_ID  = 1094 # Bug Catching Set
ENERGY_SWITCH_ID     = 1116 # Energy Switch
ULTRA_BALL_ID        = 1121 # Ultra Ball
SACRED_ASH_ID        = 1129 # Sacred Ash
TRANSCEIVER_ID       = 1134 # Team Rocket's Transceiver
POKE_PAD_ID          = 1152 # Poké Pad
LUCKY_HELMET_ID      = 1156 # Lucky Helmet
BRAVE_BANGLE_ID      = 1175 # Brave Bangle

ARIANA_ID            = 1216 # Team Rocket's Ariana
GIOVANNI_ID          = 1218 # Team Rocket's Giovanni
PETREL_ID            = 1219 # Team Rocket's Petrel
PROTON_ID            = 1220 # Team Rocket's Proton
LILLIES_DETERM_ID    = 1227 # Lillie's Determination

TR_FACTORY_ID        = 1257 # Team Rocket's Factory

TR_POKEMON_IDS = frozenset({400, 401, 414, 431, 434, 464})
EX_POKEMON_IDS = frozenset({431, 272})


def _extract_context(obs) -> dict:
    """
    Extracts essential board context for Mewtwo deck heuristics.
    """
    ctx = {
        "my_prizes": 6,
        "opp_prizes": 6,
        "my_active_id": -1,
        "opp_active_id": -1,
        "my_bench_ids": [],
        "opp_bench_ids": [],
        "my_active_energy": 0,
        "game_phase": "early",
        "my_ps": None,
        "my_bench_count": 0,
        "tr_in_play": 0,
        "turn": 0,
        "my_hand_len": 0,
        "my_hand_ids": [],
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
                if my_active_list and my_active_list[0] is not None:
                    a = my_active_list[0]
                    if isinstance(a, dict):
                        ctx["my_active_id"] = a.get("cardId", a.get("id", -1))
                        ctx["my_active_energy"] = len(a.get("energyCards", []))
                    else:
                        ctx["my_active_id"] = getattr(a, "cardId", getattr(a, "id", -1))
                        ctx["my_active_energy"] = len(getattr(a, "energyCards", []))

                opp_active_list = getattr(opp_ps, "active", []) if not isinstance(opp_ps, dict) else opp_ps.get("active", [])
                if opp_active_list and opp_active_list[0] is not None:
                    a = opp_active_list[0]
                    ctx["opp_active_id"] = a.get("cardId", a.get("id", -1)) if isinstance(a, dict) else getattr(a, "cardId", getattr(a, "id", -1))

                opp_bench = getattr(opp_ps, "bench", []) if not isinstance(opp_ps, dict) else opp_ps.get("bench", [])
                bench_ids = []
                for p in opp_bench:
                    if p is not None:
                        bench_ids.append(p.get("cardId", p.get("id", -1)) if isinstance(p, dict) else getattr(p, "cardId", getattr(p, "id", -1)))
                ctx["opp_bench_ids"] = bench_ids

                prizes = ctx["my_prizes"]
                if prizes <= 2:
                    ctx["game_phase"] = "late"
                elif prizes <= 4:
                    ctx["game_phase"] = "mid"
                else:
                    ctx["game_phase"] = "early"

                my_bench = getattr(my_ps, "bench", []) if not isinstance(my_ps, dict) else my_ps.get("bench", [])
                my_bench_ids = []
                tr_count = 1 if ctx["my_active_id"] in TR_POKEMON_IDS else 0
                for p in my_bench:
                    if p is not None:
                        pid = p.get("cardId", p.get("id", -1)) if isinstance(p, dict) else getattr(p, "cardId", getattr(p, "id", -1))
                        my_bench_ids.append(pid)
                        if pid in TR_POKEMON_IDS:
                            tr_count += 1
                ctx["my_bench_ids"] = my_bench_ids
                ctx["my_bench_count"] = len(my_bench_ids)
                ctx["tr_in_play"] = tr_count

                my_hand = getattr(my_ps, "hand", []) if not isinstance(my_ps, dict) else my_ps.get("hand", [])
                ctx["my_hand_len"] = len(my_hand)
                hand_ids = []
                for c in my_hand:
                    if c is not None:
                        hand_ids.append(c.get("cardId", c.get("id", -1)) if isinstance(c, dict) else getattr(c, "cardId", getattr(c, "id", -1)))
                ctx["my_hand_ids"] = hand_ids
    except Exception:
        pass

    return ctx


def evaluate_mewtwo_expert_bonus(obs, option, opponent_name="unknown") -> tuple[float, str]:
    """
    Evaluates a strategic prior bonus in range [-0.20, +0.20] for Mewtwo deck.
    Serves as a soft prior guide for MCTS without hardcoding actions.
    """
    ctx = _extract_context(obs)
    my_active_id = ctx["my_active_id"]
    opp_bench_ids = ctx.get("opp_bench_ids", [])
    my_bench_ids = ctx.get("my_bench_ids", [])
    my_bench_count = ctx["my_bench_count"]
    tr_in_play = ctx["tr_in_play"]
    turn = ctx["turn"]
    game_phase = ctx["game_phase"]

    opt_type_raw = getattr(option, "type", getattr(option, "optionType", None))
    if opt_type_raw is None and isinstance(option, dict):
        opt_type_raw = option.get("type", option.get("optionType", -1))

    opt_type_str = str(opt_type_raw).upper()
    IS_PLAY   = (opt_type_raw in (7, getattr(OptionType, "PLAY", 7))) or ("PLAY" in opt_type_str)
    IS_ATTACH = (opt_type_raw in (8, getattr(OptionType, "ATTACH", 8))) or ("ATTACH" in opt_type_str)
    IS_EVOLVE = (opt_type_raw in (9, getattr(OptionType, "EVOLVE", 9))) or ("EVOLVE" in opt_type_str)
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

    # -------------------------------------------------------------------------
    # 1. PLAY ACTIONS (action_type == 7)
    # -------------------------------------------------------------------------
    if IS_PLAY:
        # --- A. Basic Pokémon Benching & Opening Setup ---
        if card_id in (TAROUNTULA_ID, ARTICUNO_ID, MIMIKYU_ID, SNEASEL_ID):
            if my_bench_count < 4:
                bonus += 0.15
                triggered = f"Mewtwo_Bench_Basic_{card_id}"
        elif card_id in EX_POKEMON_IDS:
            if turn <= 3 and ctx["my_active_energy"] == 0:
                bonus -= 0.15  # Avoid premature 2-prize ex exposure early
                triggered = f"Mewtwo_Avoid_Premature_ex_Exposure_{card_id}"
            elif my_bench_count < 4:
                bonus += 0.10
                triggered = f"Mewtwo_Bench_ex_{card_id}"

        # --- B. Supporter & Tutor Play ---
        elif card_id == TRANSCEIVER_ID:
            bonus += 0.18  # Search TR Supporter (Ariana / Giovanni / Petrel)
            triggered = "Mewtwo_Transceiver_Supporter_Tutor"

        elif card_id == ARIANA_ID:
            bonus += 0.16  # Team Rocket Pokémon search + draw
            triggered = "Mewtwo_Ariana_Search_Draw"

        elif card_id == GIOVANNI_ID and len(opp_bench_ids) > 0:
            bonus += 0.18  # Gust benched target
            triggered = "Mewtwo_Giovanni_Gust_Disruption"

        elif card_id == LILLIES_DETERM_ID:
            bonus += 0.15  # Hand refresh
            triggered = "Mewtwo_Lillie_Hand_Refresh"

        elif card_id == PROTON_ID:
            bonus += 0.12  # Additional draw engine
            triggered = "Mewtwo_Proton_Draw"

        elif card_id == PETREL_ID:
            bonus += 0.12  # Supporter recycling
            triggered = "Mewtwo_Petrel_Supporter_Recovery"

        # --- C. Search Items, Tools & Stadiums ---
        elif card_id == BUG_CATCHING_SET_ID:
            bonus += 0.15  # Search Grass Pokémon (Tarountula/Spidops) + Energy
            triggered = "Mewtwo_Bug_Catching_Set_Search"

        elif card_id == POFFIN_ID and my_bench_count < 4:
            bonus += 0.18  # Search 2 Basics <= 70 HP (Tarountula, Mimikyu, Sneasel)
            triggered = "Mewtwo_Poffin_Search_Basics"

        elif card_id == ULTRA_BALL_ID:
            bonus += 0.14  # Unrestricted Pokémon search
            triggered = "Mewtwo_Ultra_Ball_Search"

        elif card_id == SECRET_BOX_ID:
            bonus += 0.16  # ACE SPEC multi-type search engine
            triggered = "Mewtwo_Secret_Box_Search"

        elif card_id == ENERGY_SWITCH_ID:
            bonus += 0.15  # Energy acceleration transfer to active attacker
            triggered = "Mewtwo_Energy_Switch_Transfer"

        elif card_id == SACRED_ASH_ID:
            bonus += 0.12  # Recycle 5 Pokémon back to deck
            triggered = "Mewtwo_Sacred_Ash_Recovery"

        elif card_id == POKE_PAD_ID:
            bonus += 0.12  # Recycle Supporter to deck
            triggered = "Mewtwo_PokePad_Recycle"

        elif card_id == BRAVE_BANGLE_ID:
            if my_active_id in (SPIDOPS_ID, MEWTWO_EX_ID):
                bonus += 0.16  # +30 damage boost against ex targets
                triggered = "Mewtwo_Brave_Bangle_Equip_Attacker"

        elif card_id == LUCKY_HELMET_ID:
            if my_active_id in (ARTICUNO_ID, MIMIKYU_ID):
                bonus += 0.16  # Draw on damage equip on defensive wall
                triggered = "Mewtwo_Lucky_Helmet_Defensive_Wall"

        elif card_id == TR_FACTORY_ID:
            bonus += 0.15  # Stadium draw engine for Team Rocket cards
            triggered = "Mewtwo_TR_Factory_Stadium"

    # -------------------------------------------------------------------------
    # 2. ENERGY ATTACHMENT ACTIONS (action_type == 8)
    # -------------------------------------------------------------------------
    elif IS_ATTACH:
        attached_id = getattr(option, "cardId", -1) if not isinstance(option, dict) else option.get("cardId", -1)
        if attached_id == TR_ENERGY_ID:
            if tr_in_play >= 2:
                bonus += 0.20  # Counts as 2 Energy for TR Pokémon
                triggered = "Mewtwo_TR_Energy_Accelerate"
        elif attached_id == BASIC_G_ENERGY_ID:
            if my_active_id in (TAROUNTULA_ID, SPIDOPS_ID, ARTICUNO_ID):
                bonus += 0.15
                triggered = "Mewtwo_Grass_Energy_Attach"
        elif attached_id == BASIC_P_ENERGY_ID:
            if my_active_id in (CLEFAIRY_EX_ID, MEWTWO_EX_ID, MIMIKYU_ID):
                bonus += 0.15
                triggered = "Mewtwo_Psychic_Energy_Attach"

    # -------------------------------------------------------------------------
    # 3. EVOLUTION ACTIONS (action_type == 9)
    # -------------------------------------------------------------------------
    elif IS_EVOLVE:
        if card_id == SPIDOPS_ID:
            bonus += 0.20  # Main Stage 1 attacker evolution
            triggered = "Mewtwo_Evolve_Spidops"

    # -------------------------------------------------------------------------
    # 4. RETREAT ACTIONS (action_type == 12)
    # -------------------------------------------------------------------------
    elif IS_RETREAT:
        if my_active_id in EX_POKEMON_IDS and turn <= 4:
            bonus += 0.15  # Rescue early exposed 2-prize ex back to bench
            triggered = "Mewtwo_Retreat_Rescue_Exposed_ex"

    # -------------------------------------------------------------------------
    # 5. ATTACK EXECUTION (action_type == 13)
    # -------------------------------------------------------------------------
    elif IS_ATTACK:
        my_hand_ids = ctx.get("my_hand_ids", [])
        has_unplayed_supporter = any(cid in (1216, 1218, 1227, 1219, 1220) for cid in my_hand_ids)
        has_unplayed_search_item = any(cid in (1134, 1094, 1086, 1121, 1092) for cid in my_hand_ids)

        if has_unplayed_supporter or has_unplayed_search_item:
            bonus += 0.10  # Postpone attack slightly so MCTS searches Supporters/Items FIRST
            triggered = "Mewtwo_Postpone_Attack_For_Setup"
        else:
            bonus += 0.20
            triggered = "Mewtwo_Attack_Execution"

    # Bound bonus range [-0.20, +0.20]
    bonus = max(-0.20, min(0.20, bonus))
    return bonus, triggered
