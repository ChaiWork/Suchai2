"""
Team Rocket Mewtwo ex Heuristic Expert Engine (v3 Refined).
Ground Truth: Matches 60-card CSV list (deck lama.csv / deck_mewtwo.csv).
Provides phase-aware strategic action prior guidance (range [-0.20, +0.20]) for MCTS exploration.
"""

from cg.api import OptionType
try:
    from src.expert_system.search_strategy import score_search_target
except ImportError:
    from expert_system.search_strategy import score_search_target

# Card ID Constants (Ground Truth: deck lama.csv)
BASIC_G_ENERGY_ID    = 1    # Basic {G} Energy
TR_ENERGY_ID         = 15   # Team Rocket's Energy

TAROUNTULA_ID        = 400  # Team Rocket's Tarountula
SPIDOPS_ID           = 401  # Team Rocket's Spidops
ARTICUNO_ID          = 414  # Team Rocket's Articuno
MEWTWO_EX_ID         = 431  # Team Rocket's Mewtwo ex
MIMIKYU_ID           = 434  # Team Rocket's Mimikyu

POFFIN_ID            = 1086 # Buddy-Buddy Poffin
BUG_CATCHING_SET_ID  = 1094 # Bug Catching Set
ENERGY_SWITCH_ID      = 1095 # Energy Switch
SWITCH_ID            = 1097 # Switch
EARTHEN_VESSEL_ID     = 1106 # Earthen Vessel
ULTRA_BALL_ID        = 1121 # Ultra Ball
TRANSCEIVER_ID       = 1134 # Team Rocket's Transceiver
POKE_PAD_ID          = 1152 # Poké Pad
MAXIMUM_BELT_ID      = 1158 # Maximum Belt (ACE SPEC Tool)
NIGHT_STRETCHER_ID   = 1159 # Night Stretcher
BRAVE_BANGLE_ID      = 1175 # Brave Bangle
PRISM_TOWER_ID        = 1180 # Prism Tower

ARIANA_ID            = 1216 # Team Rocket's Ariana
GIOVANNI_ID          = 1218 # Team Rocket's Giovanni
PETREL_ID            = 1219 # Team Rocket's Petrel
PROTON_ID            = 1220 # Team Rocket's Proton
LILLIES_DETERM_ID    = 1227 # Lillie's Determination

TR_FACTORY_ID        = 1257 # Team Rocket's Factory
BATTLE_CAGE_ID       = 1264 # Battle Cage (Anti-Dragapult Bench Protection)

# Card Category Sets
TR_POKEMON_IDS         = frozenset({400, 401, 414, 431, 434})
EX_POKEMON_IDS         = frozenset({431})
SUPPORTER_IDS          = frozenset({1216, 1218, 1219, 1220, 1227})
TACTICAL_SUPPORTER_IDS = frozenset({1219, 1220})
SEARCH_ITEM_IDS        = frozenset({POFFIN_ID, BUG_CATCHING_SET_ID, ULTRA_BALL_ID, TRANSCEIVER_ID})
MAIN_ATTACKER_IDS      = frozenset({MEWTWO_EX_ID, SPIDOPS_ID})
SPIDOPS_LINE_IDS       = frozenset({SPIDOPS_ID, TAROUNTULA_ID})

# Energy cost requirements per attacker
ATTACK_ENERGY_COSTS = {
    MEWTWO_EX_ID: 3,  # Mewtwo ex Psywave requires 3 energy total
    SPIDOPS_ID: 2,     # Spidops Venomous Whip requires 2 energy total
    ARTICUNO_ID: 2,
    MIMIKYU_ID: 1,
    TAROUNTULA_ID: 1,
}


# Heuristic Tuning Constants (Prior Bonuses in range [-0.20, +0.20])
# Attack Bonuses & Penalties
BONUS_MEWTWO_ATTACK                  = 0.20
PENALTY_MEWTWO_POWER_SAVER_UNMET     = -0.20
BONUS_SPIDOPS_ATTACK                 = 0.14
BONUS_SPIDOPS_ATTACK_MEWTWO_PRIORITY = 0.12
BONUS_ARTICUNO_ATTACK                = 0.12
BONUS_GENERIC_ATTACK                 = 0.10
BONUS_UNDERPOWERED_ATTACK            = 0.02

# Supporter & Gust Bonuses
BONUS_GIOVANNI_WIN_CLOSE_EX          = 0.20
BONUS_GIOVANNI_WIN_CLOSE_OR_EX       = 0.16
BONUS_GIOVANNI_EARLY_GUST            = 0.14
BONUS_GIOVANNI_BENCH_GUST            = 0.12
BONUS_ARIANA_LOW_HAND                = 0.16
BONUS_LILLIE_LOW_HAND                = 0.16
BONUS_TACTICAL_SUPPORTER             = 0.10

# Search, Bench & Equipment Bonuses
BONUS_REACH_FOUR_TR_POKEMON          = 0.16
BONUS_TRANSCEIVER                    = 0.18
BONUS_BUG_CATCHING                   = 0.16
BONUS_EARLY_POFFIN                   = 0.18  # was 0.22 — capped at expert prior ceiling
BONUS_LATE_POFFIN                    = 0.08

BONUS_ULTRA_BALL                     = 0.14
BONUS_MAXIMUM_BELT                   = 0.18
BONUS_BRAVE_BANGLE_ACTIVE            = 0.16
BONUS_BRAVE_BANGLE_BENCH             = 0.12
BONUS_POKE_PAD                       = 0.14
BONUS_POKE_PAD_CLOSE                 = 0.16
BONUS_NIGHT_STRETCHER                = 0.12
BONUS_NIGHT_STRETCHER_CLOSE          = 0.16
BONUS_TR_FACTORY_EARLY               = 0.14
BONUS_TR_FACTORY_LATE                = 0.10

# Positioning & Energy Bonuses
BONUS_SWITCH_HEAVY_ATTACKER          = 0.14
PENALTY_SWITCH_MEWTWO_NO_ATTACK      = -0.18
BONUS_SWITCH_OPTIMIZE_CLOSE          = 0.12
BONUS_ENERGY_ACTIVE_MEWTWO           = 0.18
BONUS_ENERGY_BENCH_MEWTWO            = 0.15
BONUS_ENERGY_SPIDOPS                 = 0.14
BONUS_EVOLVE_SPIDOPS                 = 0.16


def _get_val(obj, key: str, default=None):
    """Safely retrieves a value from either a dict (checking 'cardId' and 'id') or object attribute."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key)
        if val is None and key == "cardId":
            val = obj.get("id")
        return val if val is not None else default
        
    val = getattr(obj, key, None)
    if val is None and key == "cardId":
        val = getattr(obj, "id", None)
    return val if val is not None else default


def _count_tr_pokemon(bench_ids: list[int], active_id: int) -> int:
    """Counts total Team Rocket Pokémon currently in play (Active + Bench)."""
    count = 0
    for cid in [active_id] + list(bench_ids):
        if cid in TR_POKEMON_IDS:
            count += 1
    return count


def _extract_context(obs) -> dict:
    """
    Extracts structured, state-aware board context from observation object or dictionary.
    Handles nested attribute-style and dict-style observation schemas safely.
    """
    ctx = {
        "turn": 1,
        "my_prizes": 6,
        "opp_prizes": 6,
        "my_active_id": -1,
        "my_active_energy": 0,
        "opp_active_id": -1,
        "opp_active_energy": 0,
        "my_bench_ids": [],
        "opp_bench_ids": [],
        "my_bench_count": 0,
        "opp_bench_count": 0,
        "my_hand_len": 0,
        "my_hand_ids": [],
    }

    current = _get_val(obs, "current")
    if current is None:
        return ctx

    ctx["turn"] = _get_val(current, "turn", 1)
    yi = _get_val(current, "yourIndex", 0)
    players = _get_val(current, "players", [])

    if len(players) >= 2:
        my_ps = players[yi]
        opp_ps = players[1 - yi]
        ctx["my_ps"] = my_ps

        prizes_my = _get_val(my_ps, "prize", [])
        prizes_opp = _get_val(opp_ps, "prize", [])
        ctx["my_prizes"] = len(prizes_my) if isinstance(prizes_my, (list, tuple, set)) else 6
        ctx["opp_prizes"] = len(prizes_opp) if isinstance(prizes_opp, (list, tuple, set)) else 6

        my_active_list = _get_val(my_ps, "active", [])
        if my_active_list and len(my_active_list) > 0 and my_active_list[0] is not None:
            active_card = my_active_list[0]
            ctx["my_active_id"] = _get_val(active_card, "cardId", -1)
            energies = _get_val(active_card, "energyCards", [])
            ctx["my_active_energy"] = len(energies) if isinstance(energies, (list, tuple)) else 0
            ctx["my_active_hp"] = _get_val(active_card, "hp", 280)

        opp_active_list = _get_val(opp_ps, "active", [])
        if opp_active_list and len(opp_active_list) > 0 and opp_active_list[0] is not None:
            active_card = opp_active_list[0]
            ctx["opp_active_id"] = _get_val(active_card, "cardId", -1)
            energies = _get_val(active_card, "energyCards", [])
            ctx["opp_active_energy"] = len(energies) if isinstance(energies, (list, tuple)) else 0

        my_bench = _get_val(my_ps, "bench", [])
        if isinstance(my_bench, (list, tuple)):
            ctx["my_bench_ids"] = [_get_val(b, "cardId", -1) for b in my_bench if b is not None]
            ctx["my_bench_count"] = len(ctx["my_bench_ids"])

        opp_bench = _get_val(opp_ps, "bench", [])
        if isinstance(opp_bench, (list, tuple)):
            ctx["opp_bench_ids"] = [_get_val(b, "cardId", -1) for b in opp_bench if b is not None]
            ctx["opp_bench_count"] = len(ctx["opp_bench_ids"])

        my_hand = _get_val(my_ps, "hand", [])
        if isinstance(my_hand, (list, tuple)):
            ctx["my_hand_ids"] = [_get_val(h, "cardId", -1) for h in my_hand if h is not None]
            ctx["my_hand_len"] = len(ctx["my_hand_ids"])

    return ctx


def _resolve_card_id_from_opt(obs: dict, opt, ctx: dict) -> int:
    """Safely extracts target cardId from an option object across all play & search contexts."""
    cid = _get_val(opt, "cardId", _get_val(opt, "id", -1))
    if cid is not None and cid > 0:
        return cid

    area = _get_val(opt, "area")
    idx = _get_val(opt, "index")
    if area is None or idx is None or idx < 0:
        return -1

    my_ps = ctx.get("my_ps")
    current = _get_val(obs, "current")
    select_obj = _get_val(obs, "select")

    if area == 1:  # DECK
        deck_cards = _get_val(select_obj, "deck", [])
        if deck_cards and 0 <= idx < len(deck_cards) and deck_cards[idx] is not None:
            return _get_val(deck_cards[idx], "cardId", _get_val(deck_cards[idx], "id", -1))

    elif area == 12:  # LOOKING
        looking_cards = _get_val(current, "looking", [])
        if looking_cards and 0 <= idx < len(looking_cards) and looking_cards[idx] is not None:
            return _get_val(looking_cards[idx], "cardId", _get_val(looking_cards[idx], "id", -1))

    elif area == 5 and my_ps:  # BENCH
        bench_cards = _get_val(my_ps, "bench", [])
        if bench_cards and 0 <= idx < len(bench_cards) and bench_cards[idx] is not None:
            return _get_val(bench_cards[idx], "cardId", _get_val(bench_cards[idx], "id", -1))

    elif area == 4 and my_ps:  # ACTIVE
        active_cards = _get_val(my_ps, "active", [])
        if active_cards and 0 <= idx < len(active_cards) and active_cards[idx] is not None:
            return _get_val(active_cards[idx], "cardId", _get_val(active_cards[idx], "id", -1))

    elif area == 2 and my_ps:  # HAND
        hand_cards = _get_val(my_ps, "hand", [])
        if hand_cards and 0 <= idx < len(hand_cards) and hand_cards[idx] is not None:
            return _get_val(hand_cards[idx], "cardId", _get_val(hand_cards[idx], "id", -1))

    return -1


def evaluate_mewtwo_expert_bonus(obs: dict, opt, opponent_name: str = "") -> tuple[float, str]:
    """
    Evaluates state-aware strategic prior bonus for MCTS exploration (bounded range [-0.20, +0.20]).
    Returns: (prior_bonus: float, explanation: str)
    """
    opt_type = _get_val(opt, "type")
    card_id = _get_val(opt, "cardId")

    if opt_type is None:
        return 0.0, ""

    ctx = _extract_context(obs)

    # Dynamic card_id resolution from hand index if option provides hand index instead of direct cardId
    if card_id is None or card_id < 0:
        card_idx = _get_val(opt, "index")
        if card_idx is not None and isinstance(card_idx, int) and 0 <= card_idx < len(ctx["my_hand_ids"]):
            card_id = ctx["my_hand_ids"][card_idx]
        else:
            target_id = _get_val(opt, "targetId")
            if target_id is not None and target_id >= 0:
                card_id = target_id
            else:
                card_id = -1

    turn = ctx["turn"]
    hand_len = ctx["my_hand_len"]
    active_id = ctx["my_active_id"]

    tr_count = _count_tr_pokemon(ctx["my_bench_ids"], active_id)
    early_game = turn <= 3 or ctx["my_prizes"] >= 5
    close_game = ctx["my_prizes"] <= 2 or ctx["opp_prizes"] <= 2

    # 1. PLAY CARD ACTIONS (OptionType.PLAY / 7)
    if opt_type in (getattr(OptionType, "PLAY", 7), 7):
        # Anti-deckout guard: suppress draw/search cards when deck count is low (<= 5)
        my_ps = ctx.get("my_ps")
        deck_cnt = _get_val(my_ps, "deckCount", 60) if my_ps else 60
        if deck_cnt <= 5 and card_id in (1134, 1135, 1136, 1256, 1258, 1259):
            return -0.25, "Prior Penalty: Anti-Deckout Guard (Deck Count <= 5)"
        # A. Team Rocket Pokémon Play & 4-Pokémon Power Saver Unlock
        if card_id in TR_POKEMON_IDS:
            if tr_count == 3:
                return 0.18, "Prior: Play 4th TR Pokemon Unlocking Mewtwo ex Power Saver Ability"
            if ctx["my_bench_count"] <= 1:
                return 0.16, "Prior: Play TR Pokemon Low Bench Recovery Guard"
            # FIX 2 (expert): Early Mewtwo ex and Tarountula setup priority
            if early_game:
                if card_id == MEWTWO_EX_ID:
                    return 0.20, "Prior: Early Mewtwo ex Main Attacker Deployment (Turn 1-3)"
                elif card_id == TAROUNTULA_ID:
                    return 0.16, "Prior: Early Tarountula Setup Deployment (Turn 1-3)"
            if card_id == SPIDOPS_ID or card_id == TAROUNTULA_ID:
                return 0.12, "Prior: Play Spidops Line Trap Territory Setup"
            return 0.08, "Prior: Play Team Rocket Pokemon Setup"


        # B. Search & Tutor Engines
        if card_id == TRANSCEIVER_ID:
            return BONUS_TRANSCEIVER, "Prior: TR Transceiver Supporter Tutor"

        if card_id == BUG_CATCHING_SET_ID:
            return BONUS_BUG_CATCHING, "Prior: Bug Catching Set Grass Search"

        if card_id == POFFIN_ID:
            if early_game or ctx["my_bench_count"] <= 1:
                return BONUS_EARLY_POFFIN, "Prior: Early Poffin Bench Fill"
            return BONUS_LATE_POFFIN, "Prior: Late Poffin Bench Fill"

        if card_id == ULTRA_BALL_ID:
            return BONUS_ULTRA_BALL, "Prior: Ultra Ball Key Search"

        # C. Equipment & Recovery
        if card_id == MAXIMUM_BELT_ID:
            return BONUS_MAXIMUM_BELT, "Prior: Equip Maximum Belt ACE SPEC Snipe Tool"

        if card_id == BRAVE_BANGLE_ID:
            is_fast_aggro = any(k in opponent_name.lower() for k in ("starmie", "ogerpon", "garchomp", "dipplin"))
            if is_fast_aggro:
                return 0.20, "Prior: Equip Brave Bangle Counter Fast 2-Energy Aggro"
            if active_id in MAIN_ATTACKER_IDS:
                return BONUS_BRAVE_BANGLE_ACTIVE, "Prior: Equip Brave Bangle to Active Main Attacker"
            if any(b in MAIN_ATTACKER_IDS for b in ctx["my_bench_ids"]):
                return BONUS_BRAVE_BANGLE_BENCH, "Prior: Equip Brave Bangle to Bench Main Attacker"
            return 0.06, "Prior: Equip Brave Bangle General"

        if card_id == POKE_PAD_ID:
            is_control = any(k in opponent_name.lower() for k in ("iono", "control", "snorlax"))
            if is_control or hand_len <= 3 or close_game:
                return 0.18, "Prior: Poké Pad Recycle Key Supporter vs Iono Control"
            return 0.08, "Prior: Poké Pad Supporter Recycling"

        if card_id == NIGHT_STRETCHER_ID:
            if close_game:
                return BONUS_NIGHT_STRETCHER_CLOSE, "Prior: Night Stretcher Key Recovery Close Game"
            return BONUS_NIGHT_STRETCHER, "Prior: Night Stretcher Recovery"

        if card_id == TR_FACTORY_ID:
            is_control = any(k in opponent_name.lower() for k in ("iono", "control", "snorlax"))
            if is_control or early_game:
                return 0.20, "Prior: Play TR Factory Stadium Counter Iono Hand Disruption"
            return 0.16, "Prior: Play TR Factory Stadium Draw Engine"

        if card_id == PRISM_TOWER_ID:
            return 0.16, "Prior: Play Prism Tower Recurring Discard & Draw"

        if card_id == BATTLE_CAGE_ID:
            bench_counter_keywords = ("dragapult", "froslass", "dusknoir", "dusclops", "munkidori")
            is_bench_counter_meta = any(k in opponent_name.lower() for k in bench_counter_keywords) or opponent_name == "DRAGAPULT"
            if is_bench_counter_meta:
                return 0.20, "Prior: Play Battle Cage Counter Dragapult/Froslass Bench Snipe"
            return 0.18, "Prior: Play Battle Cage Active/Bench Protection"

        # D. Positioning & Tempo Items
        if card_id == SWITCH_ID:
            has_bench_attacker = any(b in MAIN_ATTACKER_IDS for b in ctx["my_bench_ids"])
            active_is_main = active_id in MAIN_ATTACKER_IDS
            active_e = ctx.get("my_active_energy", 0)
            active_hp = ctx.get("my_active_hp", 280)

            # Discourage pointless switching of healthy, fully-powered active Mewtwo ex.
            # Bounded prior: RL will override this if retreat is genuinely strategic.
            if active_id == MEWTWO_EX_ID and active_e >= 3 and active_hp >= 100 and tr_count >= 4:
                return -0.12, "Prior Penalty: Switch out healthy powered Mewtwo ex (strongly discouraged)"

            # Penalize switching if neither active nor bench has energy
            if active_e == 0 and not has_bench_attacker:
                return -0.15, "Prior Penalty: Pointless Switch (Neither Active Nor Bench Has Energy)"
            if not active_is_main and has_bench_attacker:
                return BONUS_SWITCH_HEAVY_ATTACKER, "Prior: Switch Non-Attacker for Heavy Attacker"
            if close_game and has_bench_attacker:
                return BONUS_SWITCH_OPTIMIZE_CLOSE, "Prior: Switch to Optimize Damage Close Game"
            if close_game:
                return 0.10, "Prior: Switch Tempo Positioning Close Game"
            return 0.05, "Prior: Switch Tempo Positioning"

        # E. State-Aware Supporters
        if card_id == GIOVANNI_ID:
            if ctx["opp_bench_count"] > 0:
                has_ex_target = any(b in EX_POKEMON_IDS for b in ctx["opp_bench_ids"])
                if has_ex_target and close_game:
                    return BONUS_GIOVANNI_WIN_CLOSE_EX, "Prior: Giovanni Gust Target EX Prize Win Swing"
                elif has_ex_target or close_game:
                    bonus = BONUS_GIOVANNI_WIN_CLOSE_OR_EX
                    if early_game and not has_ex_target:
                        bonus = BONUS_GIOVANNI_EARLY_GUST  # Slight early-game dampener
                    return bonus, "Prior: Giovanni Gust Target EX or Win Close"
                return BONUS_GIOVANNI_BENCH_GUST, "Prior: Giovanni Bench Gust"
            return 0.02, "Prior: Giovanni Supporter Play"

        if card_id == ARIANA_ID:
            if hand_len <= 4:
                return BONUS_ARIANA_LOW_HAND, "Prior: Ariana Draw Refresh Low Hand"
            return 0.08, "Prior: Ariana Team Rocket Draw"

        if card_id == LILLIES_DETERM_ID:
            if hand_len <= 3:
                return BONUS_LILLIE_LOW_HAND, "Prior: Lillie Determination Emergency Draw"
            return 0.06, "Prior: Lillie Determination Draw"

        if card_id in TACTICAL_SUPPORTER_IDS:
            return BONUS_TACTICAL_SUPPORTER, "Prior: TR Tactical Supporter Play"

    # 2. ENERGY ATTACHMENT ACTIONS (OptionType.ATTACH / 8)
    elif opt_type in (getattr(OptionType, "ATTACH", 8), 8):
        if card_id > 0:
            in_area = _get_val(opt, "inPlayArea", _get_val(opt, "area", _get_val(opt, "targetArea", 4)))
            in_idx  = _get_val(opt, "inPlayIndex", _get_val(opt, "index", _get_val(opt, "targetIndex", 0)))

            target_poke = None
            my_ps = ctx.get("my_ps")
            if my_ps is not None:
                active_list = _get_val(my_ps, "active", [])
                bench_list  = _get_val(my_ps, "bench", [])
                if (in_area == 4 or in_area == "active") and active_list and active_list[0] is not None:
                    target_poke = active_list[0]
                elif (in_area == 5 or in_area == "bench") and bench_list and 0 <= in_idx < len(bench_list) and bench_list[in_idx] is not None:
                    target_poke = bench_list[in_idx]

            opt_target_energy = 0
            if target_poke is not None:
                energies = _get_val(target_poke, "energyCards", [])
                opt_target_energy = len(energies) if isinstance(energies, (list, tuple)) else 0
                opt_target = _get_val(target_poke, "cardId", _get_val(target_poke, "id", -1))
            else:
                if (in_area == 5 or in_area == "bench") and 0 <= in_idx < len(ctx.get("my_bench_ids", [])):
                    opt_target = ctx["my_bench_ids"][in_idx]
                else:
                    opt_target = _get_val(opt, "targetCardId", active_id) if not isinstance(opt, dict) else opt.get("targetCardId", active_id)
                opt_target_energy = ctx.get("target_energy", 0)

            # Extract board context for generic evaluator
            my_bench_ids     = ctx.get("my_bench_ids", [])
            my_prizes        = ctx.get("my_prizes", 6)
            opp_prizes       = ctx.get("opp_prizes", 6)
            my_bench_count   = ctx.get("my_bench_count", 0)
            hand_ids         = ctx.get("my_hand_ids", [])
            active_energy    = ctx.get("my_active_energy", 0)
            active_hp        = ctx.get("my_active_hp", 100)

            # Build bench energy list generically from my_ps
            bench_energies = []
            if my_ps is not None:
                bench_list_raw = _get_val(my_ps, "bench", [])
                for bp in bench_list_raw:
                    if bp is not None:
                        ben_en = _get_val(bp, "energyCards", [])
                        bench_energies.append(len(ben_en) if isinstance(ben_en, (list, tuple)) else 0)

            # --- Generic evaluation via energy_evaluator ---
            try:
                from src.expert_system.energy_evaluator import evaluate_energy_target
            except ImportError:
                from expert_system.energy_evaluator import evaluate_energy_target

            score, reason = evaluate_energy_target(
                target_card_id=opt_target,
                target_current_energy=opt_target_energy,
                energy_card_id=card_id,
                hand_ids=hand_ids,
                my_prizes=my_prizes,
                opp_prizes=opp_prizes,
                bench_ids=my_bench_ids,
                bench_energies=bench_energies,
                active_id=active_id,
                active_energy=active_energy,
                active_hp=active_hp,
            )
            return score, f"Prior: {reason}"



    # 3. EVOLVE ACTIONS (OptionType.EVOLVE / 9)
    elif opt_type in (getattr(OptionType, "EVOLVE", 9), 9):
        if card_id == SPIDOPS_ID or card_id > 0:
            if active_id == TAROUNTULA_ID or TAROUNTULA_ID in ctx["my_bench_ids"]:
                return BONUS_EVOLVE_SPIDOPS, "Prior: Evolve Tarountula to Spidops"
            return 0.10, "Prior: Evolve to Spidops"

    # 4. RETREAT ACTIONS (OptionType.RETREAT / 12)
    elif opt_type in (getattr(OptionType, "RETREAT", 12), 12):
        active_hp = ctx.get("my_active_hp", 280)
        active_energy = ctx.get("my_active_energy", 0)

        # Bounded prior discouragement — not a hard block. RL remains the final decision maker.
        # Rationale: retreating wastes energy and tempo; but the NN may discover edge cases
        # (type disadvantage, poisoned, opponent lethal threat) where retreat IS correct.
        if active_id == MEWTWO_EX_ID and active_energy >= 1 and active_hp >= 100:
            return -0.12, "Prior Penalty: Retreat healthy powered Mewtwo ex (energy waste, strongly discouraged)"
        elif active_energy >= 1 and active_hp >= 80:
            return -0.10, "Prior Penalty: Retreat healthy active with energy (wasteful)"
        elif active_hp <= 30:
            return 0.15, "Prior: Tactical retreat of low-HP active"
        return -0.08, "Prior Penalty: General wasteful retreat"

    # 5. ATTACK ACTIONS (OptionType.ATTACK / 13)
    elif opt_type in (getattr(OptionType, "ATTACK", 13), 13):
        req_energy = ATTACK_ENERGY_COSTS.get(active_id, 2)
        current_energy = ctx["my_active_energy"]

        if active_id == MEWTWO_EX_ID:
            if tr_count < 4:
                return PENALTY_MEWTWO_POWER_SAVER_UNMET, "Prior Penalty: Mewtwo ex Power Saver Unmet (<4 TR Pokemon)"
            if current_energy >= req_energy:
                # FIX 1 (expert): Powered Mewtwo MUST attack — highest possible prior
                opp_hp = ctx.get("opp_active_hp", 9999)
                if opp_hp <= 160:
                    return 0.20, "CRITICAL Prior: Mewtwo ex Fully Powered - Lethal Window - ATTACK NOW"
                return 0.20, "Prior: Mewtwo ex Fully Powered - Attack Immediately"
            return BONUS_UNDERPOWERED_ATTACK, "Prior: Underpowered Active Attack"

        if current_energy >= req_energy:
            if active_id == SPIDOPS_ID:
                # Venomous Whip scales: 20 + 20 per bench Pokemon (full bench=5 → 120 dmg)
                bench_size = ctx.get("my_bench_count", len(ctx["my_bench_ids"]))
                spidops_damage = 20 + (20 * bench_size)
                opp_hp = ctx.get("opp_active_hp", 9999)
                if bench_size == 5:
                    # Full bench — max damage 120, always attack!
                    return 0.20, f"Prior: Spidops Full Bench Attack 120 Damage"
                elif bench_size >= 3:
                    # Good bench — decent damage 80-100
                    if 0 < opp_hp <= spidops_damage:
                        return 0.20, f"Prior: Spidops Lethal Window Bench{bench_size}"
                    return 0.15, f"Prior: Spidops Good Bench Attack {spidops_damage} Damage"
                elif bench_size <= 1:
                    # Thin bench — lower damage (20-40), but attacking is still ALWAYS positive over passing
                    return 0.10, f"Prior: Spidops Thin Bench ({bench_size}) {spidops_damage} Damage Attack"
                mewtwo_bench_present = MEWTWO_EX_ID in ctx["my_bench_ids"]
                if mewtwo_bench_present:
                    return BONUS_SPIDOPS_ATTACK_MEWTWO_PRIORITY, "Prior: Spidops Attack (Mewtwo On Bench)"
                return BONUS_SPIDOPS_ATTACK, "Prior: Affordable Spidops Attack"


            if active_id == ARTICUNO_ID:
                return BONUS_ARTICUNO_ATTACK, "Prior: Affordable Articuno Snipe Attack"
            return BONUS_GENERIC_ATTACK, "Prior: Affordable Active Attack"
        else:
            return BONUS_UNDERPOWERED_ATTACK, "Prior: Underpowered Active Attack"

    # 6. SEARCH CANDIDATE CARD SELECTION (OptionType.CARD / 3)
    elif opt_type in (getattr(OptionType, "CARD", 3), 3):
        target_card_id = _resolve_card_id_from_opt(obs, opt, ctx)
        if target_card_id > 0:
            try:
                from src.expert_system.search_strategy import evaluate_search_target_prior, extract_search_context
            except ImportError:
                from expert_system.search_strategy import evaluate_search_target_prior, extract_search_context
            
            search_ctx = extract_search_context(obs)
            prior_score, reason = evaluate_search_target_prior(target_card_id, search_ctx)
            return prior_score, f"Search Candidate Prior: {reason}"
        return 0.0, "Search Candidate: Unknown Card"

    return 0.0, ""


# Backward-compatible API binding
get_expert_bonus = evaluate_mewtwo_expert_bonus


