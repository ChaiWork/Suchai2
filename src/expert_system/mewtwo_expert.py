"""
Team Rocket Mewtwo ex Heuristic Expert Engine (v3 Refined).
Ground Truth: Matches 60-card CSV list (deck lama.csv / deck_mewtwo.csv).
Provides phase-aware strategic action prior guidance (range [-0.20, +0.20]) for MCTS exploration.
"""

from cg.api import OptionType, SelectContext, AreaType
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
NIGHT_STRETCHER_ID   = 1097 # Night Stretcher
ENERGY_SWITCH_ID     = 1116 # Energy Switch
ULTRA_BALL_ID        = 1121 # Ultra Ball
SWITCH_ID            = 1123 # Switch
TRANSCEIVER_ID       = 1134 # Team Rocket's Transceiver
POKE_PAD_ID          = 1152 # Poké Pad
MAXIMUM_BELT_ID      = 1158 # Maximum Belt (ACE SPEC Tool)
HEROS_CAPE_ID        = 1159 # Hero's Cape (ACE SPEC Tool)
BRAVE_BANGLE_ID      = 1175 # Brave Bangle
PRISM_TOWER_ID       = 1180 # Prism Tower

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
BONUS_GIOVANNI_WIN_CLOSE_EX          = 0.24
BONUS_GIOVANNI_WIN_CLOSE_OR_EX       = 0.20
BONUS_GIOVANNI_EARLY_GUST            = 0.18
BONUS_GIOVANNI_BENCH_GUST            = 0.16
BONUS_ARIANA_LOW_HAND                = 0.28
BONUS_LILLIE_LOW_HAND                = 0.28
BONUS_TACTICAL_SUPPORTER             = 0.22

# Search, Bench & Equipment Bonuses
BONUS_REACH_FOUR_TR_POKEMON          = 0.22
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


def _count_effective_energy(cards) -> int:
    """Counts effective energy provided by attached cards (TR Energy #15 = 2, Basic = 1)."""
    if not cards:
        return 0
    total = 0
    for c in cards:
        if c is not None:
            cid = _get_val(c, "cardId", _get_val(c, "id", -1))
            if cid in (15, 19):  # Team Rocket's Energy provides 2 Colorless Energy
                total += 2
            else:
                total += 1
    return total


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
            ctx["my_active_energy"] = _count_effective_energy(energies)
            ctx["my_active_hp"] = _get_val(active_card, "hp", 280)

        opp_active_list = _get_val(opp_ps, "active", [])
        if opp_active_list and len(opp_active_list) > 0 and opp_active_list[0] is not None:
            active_card = opp_active_list[0]
            ctx["opp_active_id"] = _get_val(active_card, "cardId", -1)
            energies = _get_val(active_card, "energyCards", [])
            ctx["opp_active_energy"] = len(energies) if isinstance(energies, (list, tuple)) else 0
            ctx["opp_active_hp"] = _get_val(active_card, "hp", 100)
            ctx["opp_active_max_hp"] = _get_val(active_card, "maxHp", 100)

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


def _resolve_target_pokemon_id_from_opt(obs: dict, opt, ctx: dict) -> int:
    """Safely extracts the target in-play Pokémon cardId for Tool/Attach/Item targets."""
    target_cid = _get_val(opt, "targetCardId", _get_val(opt, "targetId", -1))
    if target_cid is not None and target_cid > 0:
        return target_cid

    in_area = _get_val(opt, "inPlayArea", _get_val(opt, "targetArea"))
    in_idx  = _get_val(opt, "inPlayIndex", _get_val(opt, "targetIndex"))
    my_ps = ctx.get("my_ps")
    if my_ps:
        if in_area in (5, "bench") and in_idx is not None:
            bench_cards = _get_val(my_ps, "bench", [])
            if 0 <= in_idx < len(bench_cards) and bench_cards[in_idx] is not None:
                return _get_val(bench_cards[in_idx], "cardId", _get_val(bench_cards[in_idx], "id", -1))
        elif in_area in (4, "active") or in_area is None:
            active_cards = _get_val(my_ps, "active", [])
            if active_cards and active_cards[0] is not None:
                return _get_val(active_cards[0], "cardId", _get_val(active_cards[0], "id", -1))
    return ctx.get("my_active_id", -1)


def evaluate_erasure_ball_discard(obs: dict, opt, ctx: dict, opponent_name: str = "") -> tuple[float, str]:
    """
    Contextual Erasure Ball Energy Discard Evaluation.
    Distinguishes NORMAL ATTACK (preserve energy against low HP targets)
    vs HIGH-DAMAGE / TANK KO ATTACK (allow controlled discard against high HP targets).
    Strictly forbids discarding energy from Benched Mewtwo ex.
    """
    opt_area = _get_val(opt, "area")
    opt_idx  = _get_val(opt, "index", 0)
    my_ps    = ctx.get("my_ps")
    opp_hp   = ctx.get("opp_active_hp", 100)
    opp_id   = ctx.get("opp_active_id", -1)
    
    # Check if target is a high HP / tanky opponent (HP > 160 or EX) requiring 280 damage boost
    is_tank_target = (opp_hp > 160 or opp_id in EX_POKEMON_IDS)

    # 1. Option is discarding energy from BENCH SPOT
    active_energy = ctx.get("my_active_energy", 0)
    if opt_area in (5, "bench", getattr(AreaType, "BENCH", 5)):
        if my_ps is not None:
            bench_list = _get_val(my_ps, "bench", [])
            if isinstance(bench_list, (list, tuple)) and 0 <= opt_idx < len(bench_list) and bench_list[opt_idx] is not None:
                target_poke_id = _get_val(bench_list[opt_idx], "cardId", _get_val(bench_list[opt_idx], "id", -1))
                
                # Benched Mewtwo ex: STRICTLY FORBIDDEN to discard reserve energy
                if target_poke_id == MEWTWO_EX_ID:
                    return -0.65, "Prior Penalty: CATASTROPHIC - FORBIDDEN to Discard Energy from Benched Mewtwo ex! Preserve Backup Attacker Energy!"
                
                # Benched Spidops: evaluate recycling / attack readiness
                elif target_poke_id in (SPIDOPS_ID, 401):
                    # If Active Mewtwo has >= 2 energy cards, ALWAYS pay from active and NEVER drain Spidops!
                    if active_energy >= 2:
                        return -0.65, "Prior Penalty: ABSOLUTE VETO - Active Mewtwo Has Energy, NEVER Discard Energy from Benched Spidops Sweeper!"
                    has_recovery = any(cid in (NIGHT_STRETCHER_ID, PETREL_ID, 1097, 1219) for cid in ctx.get("my_hand_ids", []))
                    if has_recovery:
                        return -0.20, "Prior Penalty: Discard Energy from Spidops (Energy Recovery Available in Hand)"
                    return -0.55, "Prior Penalty: FORBIDDEN - Do NOT Discard Energy from Benched Spidops Attacker!"
                
                # 1-Prize Non-Attacking Pivots (Tarountula, Articuno, Mimikyu)
                elif target_poke_id in (TAROUNTULA_ID, ARTICUNO_ID, MIMIKYU_ID, 400, 414, 434):
                    if is_tank_target:
                        return 0.30, "Prior: Discard Useless Energy from 1-Prize Pivot to Power 280 Damage Strike on Tank"
                    return -0.30, "Prior Penalty: Do NOT Discard Pivot Energy when Normal 160 Attack Already KOs"

        return -0.50, "Prior Penalty: Do NOT Discard Energy from Bench when Active Attacker Has Energy"

    # 2. Option is discarding energy from ACTIVE SPOT (Active Mewtwo ex using the attack)
    if opt_area in (4, "active", getattr(AreaType, "ACTIVE", 4)) or opt_area is None:
        if is_tank_target:
            return 0.55, f"CRITICAL Prior: Discard Energy from Active Mewtwo ex for Lethal 280 Damage on Tank Opponent (HP: {opp_hp})!"
        else:
            # Low HP opponent (<= 160 HP): Normal 160 attack KOs without discard!
            return -0.45, f"Prior Penalty: FORBIDDEN - Do NOT Discard Energy with Erasure Ball (Opponent has {opp_hp} <= 160 HP, Normal Attack KOs!)"

    return 0.0, ""


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

    has_ready_bench_attacker = False
    my_ps = ctx.get("my_ps")
    if my_ps is not None:
        bench_list_raw = _get_val(my_ps, "bench", [])
        for bp in bench_list_raw:
            if bp is not None:
                bid = _get_val(bp, "cardId", _get_val(bp, "id", -1))
                ben_en = _count_effective_energy(_get_val(bp, "energyCards", []))
                req_b = 3 if bid == MEWTWO_EX_ID else (2 if bid in (SPIDOPS_ID, 401) else 1)
                if bid == MEWTWO_EX_ID and tr_count < 4:
                    continue  # Power-Saver-locked Mewtwo cannot attack
                if ben_en >= req_b:
                    has_ready_bench_attacker = True
                    break

    # 0. ACTIVE SELECTION & PROMOTION CONTEXT EVALUATION
    select_obj = _get_val(obs, "select", {})
    select_context = _get_val(select_obj, "context", -1)
    
    is_setup_active = (select_context in (1, "1", "SetupActivePokemon", getattr(SelectContext, "SETUP_ACTIVE_POKEMON", 1)))
    is_setup_bench  = (select_context in (2, "2", "SetupBenchPokemon", getattr(SelectContext, "SETUP_BENCH_POKEMON", 2)))
    is_to_active    = (select_context in (3, 4, "3", "4", "ToActive", "PromoteActive", "Switch", getattr(SelectContext, "TO_ACTIVE", 4), getattr(SelectContext, "SWITCH", 3))) or (isinstance(opt, dict) and opt.get("context") in ("ToActive", "SetupActivePokemon", "PromoteActive", 3, 4, "3", "4"))

    # 0.D ATTACK ENERGY DISCARD CONTEXT EVALUATION (SelectContext.DISCARD_ENERGY_CARD = 26, 28, 29, 30, 23)
    is_discard_energy = (
        select_context in (26, 28, 29, 30, 23, "26", "28", "29", "30", "23",
                           "DiscardEnergyCard", "DiscardCardOrAttachedCard", "DiscardEnergy", "DetachFrom",
                           getattr(SelectContext, "DISCARD_ENERGY_CARD", 26))
        or (isinstance(select_context, str) and "discard" in select_context.lower() and "energy" in select_context.lower())
        or (isinstance(opt, dict) and (opt.get("type") in ("EnergyCard", 2, "AttachedCard") or opt.get("energyIndex") is not None))
    )
    if is_discard_energy:
        return evaluate_erasure_ball_discard(obs, opt, ctx, opponent_name=opponent_name)

    # A. INITIAL ACTIVE SETUP (Turn 1 Game Setup: Selecting Starting Active Pokemon from Hand)
    if is_setup_active:
        target_card_id = _resolve_card_id_from_opt(obs, opt, ctx)
        if target_card_id == MEWTWO_EX_ID:
            has_single_prize_basic = any(cid in (TAROUNTULA_ID, ARTICUNO_ID, MIMIKYU_ID, 400, 414, 434) for cid in ctx["my_hand_ids"])
            if has_single_prize_basic:
                return -0.60, "Prior Penalty: FORBIDDEN - Bench Mewtwo ex at Setup (Start 1-Prize Pivot Active to Charge Mewtwo on Bench!)"
            return 0.0, "Prior: Start Mewtwo ex (Only Basic in Hand)"
        elif target_card_id in (TAROUNTULA_ID, 400):
            return 0.45, "CRITICAL Prior: Start Tarountula Active (Evolves into Spidops Turn 2)"
        elif target_card_id in (ARTICUNO_ID, 414):
            return 0.45, "CRITICAL Prior: Start Articuno Active (120-HP Wall & Repelling Veil Bench Shield)"
        elif target_card_id in (MIMIKYU_ID, 434):
            return 0.35, "Prior: Start Mimikyu Active (0 Retreat Cost Pivot / Safeguard)"

    # B. INITIAL BENCH SETUP (Turn 1 Game Setup: Placing Benched Pokemon from Hand)
    if is_setup_bench:
        return 0.35, "Prior: Place Basic Pokemon on Bench at Setup"

    # C. MID-GAME PROMOTION & SWITCH (SelectContext.TO_ACTIVE / 4 or SWITCH / 3)
    if is_to_active:
        target_card_id = _resolve_card_id_from_opt(obs, opt, ctx)
        if target_card_id == MEWTWO_EX_ID:
            # Crustle EX Immunity Matchup Rule: NEVER promote Mewtwo ex against Crustle!
            opp_active_id = ctx.get("opp_active_id", -1)
            is_crustle = (opp_active_id in (344, 345, 407, 408)) or any(k in opponent_name.lower() for k in ("crustle", "dwebble"))
            if is_crustle:
                return -0.65, "Prior Penalty: ABSOLUTE VETO - Do NOT Promote Mewtwo ex vs Crustle (Crustle has EX Immunity Ability!)"

            mewtwo_energy = 0
            my_ps = ctx.get("my_ps")
            if my_ps:
                for b in _get_val(my_ps, "bench", []):
                    if b and _get_val(b, "cardId", _get_val(b, "id", -1)) == MEWTWO_EX_ID:
                        en = _get_val(b, "energyCards", [])
                        mewtwo_energy = max(mewtwo_energy, _count_effective_energy(en))
            if mewtwo_energy >= 3 and tr_count >= 4:
                return 0.52, "CRITICAL Prior: Promote Fully Charged Mewtwo ex (3 Energy + Power Saver Active!)"
            
            has_single_prize_pivot = any(cid in (414, 400, 401, 434) for cid in ctx["my_bench_ids"])
            if tr_count < 4 and has_single_prize_pivot:
                return -0.60, "Prior Penalty: FORBIDDEN - Do NOT Promote Power-Saver-Locked Mewtwo ex (TR count < 4)! Promote 1-Prize Wall instead!"
            
            is_fast_aggro_meta = any(k in opponent_name.lower() for k in ("starmie", "lopunny", "megakangaskhan", "speed", "garchomp", "abomasnow"))
            if is_fast_aggro_meta and has_single_prize_pivot:
                return -0.55, "Prior Penalty: FORBIDDEN - Do NOT Promote Unpowered 2-Prize Mewtwo ex vs Fast Aggro (Charge on Bench First!)"
            if has_single_prize_pivot:
                return -0.50, "Prior Penalty: FORBIDDEN - Do NOT Promote Unpowered 2-Prize Mewtwo ex to Active (Charge on Bench First!)"
            return -0.20, "Prior Penalty: Promote Unpowered 2-Prize Mewtwo ex"

        elif target_card_id == ARTICUNO_ID:
            is_fast_aggro_meta = any(k in opponent_name.lower() for k in ("starmie", "lopunny", "megakangaskhan", "speed", "garchomp"))
            if is_fast_aggro_meta:
                return 0.45, "CRITICAL Prior: Promote Articuno 120-HP Single-Prize Wall vs Fast Aggro (Starmie/Lopunny/MegaKangaskhan)"
            if tr_count >= 4:
                return 0.10, f"Prior: Promote Articuno Single-Prize Wall"
            return 0.35, f"Prior: Promote Articuno Single-Prize Wall (TR={tr_count}, builds Power Saver)"

        elif target_card_id == TAROUNTULA_ID:
            # Fragile 50-HP Tarountula Donk Prohibition:
            # Never promote 50-HP Tarountula into active if any other bench option exists (Articuno/Mimikyu/Mewtwo)
            has_other_bench = any(cid in (414, 401, 431, 434) for cid in ctx["my_bench_ids"] if cid != TAROUNTULA_ID)
            if has_other_bench:
                return -0.35, "Prior Penalty: FORBIDDEN - Do NOT Promote Fragile 50-HP Tarountula (Donk Target!)"
            return 0.0, "Prior: Promote Tarountula (Only Bench Target Available)"

        elif target_card_id == MIMIKYU_ID:
            return 0.35, "Prior: Promote Mimikyu 0-Retreat Cost Pivot"

        elif target_card_id == SPIDOPS_ID:
            spidops_energy = 0
            my_ps = ctx.get("my_ps")
            if my_ps:
                for b in _get_val(my_ps, "bench", []):
                    if b and _get_val(b, "cardId", _get_val(b, "id", -1)) == SPIDOPS_ID:
                        en = _get_val(b, "energyCards", [])
                        spidops_energy = max(spidops_energy, _count_effective_energy(en))
            has_powered_mewtwo = False
            if my_ps:
                for b in _get_val(my_ps, "bench", []):
                    if b and _get_val(b, "cardId", _get_val(b, "id", -1)) == MEWTWO_EX_ID:
                        en = _get_val(b, "energyCards", [])
                        if _count_effective_energy(en) >= 3 and tr_count >= 4:
                            has_powered_mewtwo = True
                            break
            opp_active_id = ctx.get("opp_active_id", -1)
            is_crustle = (opp_active_id in (344, 345, 407, 408)) or any(k in opponent_name.lower() for k in ("crustle", "dwebble"))
            if is_crustle:
                return 0.60, "CRITICAL Prior: Promote Spidops Counter Attacker vs Crustle (Single-Prize EX Immunity Counter!)"
            if has_powered_mewtwo and spidops_energy < 2:
                return -0.35, "Prior Penalty: Do NOT Promote Unpowered Spidops Over Fully Powered Mewtwo ex"
            if spidops_energy >= 2:
                is_darkness_or_grimmsnarl = any(k in opponent_name.lower() for k in ("grimmsnarl", "marnie", "darkness", "munkidori", "froslass", "honchkrow", "tyranitar")) or (ctx.get("opp_active_id", -1) in (646, 647, 648, 112, 104))
                if is_darkness_or_grimmsnarl:
                    return 0.55, "CRITICAL Prior: Promote Powered Spidops (Exploits 2x Grass Weakness on Darkness EX for 300+ Damage OHKO!)"
                is_fighting_matchup = any(k in opponent_name.lower() for k in ("lucario", "fighting", "hariyama", "riolu", "kgin"))
                if is_fighting_matchup:
                    return 0.50, "CRITICAL Prior: Promote Powered Spidops Counter Attacker vs Fighting Meta"
                return 0.45, "CRITICAL Prior: Promote Powered Spidops Main Attacker"
            is_fast_aggro_meta = any(k in opponent_name.lower() for k in ("starmie", "lopunny"))
            if is_fast_aggro_meta:
                return 0.40, "CRITICAL Prior: Promote Spidops ex Counter Attacker & Trap Territory vs Fast Aggro (Starmie/Lopunny)"
            return 0.15, "Prior: Promote 130-HP Spidops Single-Prize Pivot Wall"

    # 1. PLAY CARD ACTIONS (OptionType.PLAY / 7)
    if opt_type in (getattr(OptionType, "PLAY", 7), 7):
        # LETHAL KO WINDOW ATTACK URGENCY:
        # When active attacker is ready with enough energy to KO opponent active,
        # suppress playing non-essential setup items/abilities so the agent TAKES THE LETHAL KO IMMEDIATELY.
        opp_hp = ctx.get("opp_active_hp", 9999)
        active_energy = ctx.get("my_active_energy", 0)
        can_ko_immediately = (
            (active_id == SPIDOPS_ID and active_energy >= 2 and opp_hp <= 180) or
            (active_id == MEWTWO_EX_ID and active_energy >= 3 and tr_count >= 4 and opp_hp <= 270) or
            (active_id == ARTICUNO_ID and active_energy >= 1 and opp_hp <= 60)
        )
        if can_ko_immediately and card_id not in (1182, 1183):
            return -0.35, "Prior Penalty: Delaying Lethal Attack (KO Opponent Active Immediately!)"

        # Anti-deckout guard: suppress draw/search cards when deck count is low (<= 5)
        my_ps = ctx.get("my_ps")
        deck_cnt = _get_val(my_ps, "deckCount", 60) if my_ps else 60
        if deck_cnt <= 5 and card_id in (1134, 1135, 1136, 1256, 1258, 1259):
            return -0.25, "Prior Penalty: Anti-Deckout Guard (Deck Count <= 5)"
        # A. Team Rocket Pokémon Play & 4-Pokémon Power Saver Unlock & 5-Bench Spidops Scaling
        if card_id in TR_POKEMON_IDS:
            if ctx["my_bench_count"] == 0:
                return 0.45, "CRITICAL Prior: Donk/Bench-Out Prevention - Play Basic Pokemon to Empty Bench Immediately"
            # Dragapult / Grimmsnarl / Bench Snipe / Spread Counter Meta: Deploy Articuno Bench Shield (Repelling Veil)
            bench_counter_keywords = ("dragapult", "froslass", "dusknoir", "dusclops", "munkidori", "dipplin", "grimmsnarl", "marnie", "phantom", "sixth", "spread")
            is_bench_counter_meta = any(k in opponent_name.lower() for k in bench_counter_keywords) or opponent_name == "DRAGAPULT"
            if card_id == ARTICUNO_ID and is_bench_counter_meta and ARTICUNO_ID not in ctx["my_bench_ids"] and active_id != ARTICUNO_ID:
                return 0.45, "CRITICAL Prior: Deploy Articuno Bench Shield (Repelling Veil vs Dragapult/Grimmsnarl Spread)"
            # Crustle EX Immunity Matchup Rule: Prioritize playing Tarountula & Spidops ex
            opp_active_id = ctx.get("opp_active_id", -1)
            is_crustle = (opp_active_id in (344, 345, 407, 408)) or any(k in opponent_name.lower() for k in ("crustle", "dwebble"))
            if is_crustle and card_id == SPIDOPS_ID:
                return 0.60, "CRITICAL Prior: PRIORITIZE Evolve Spidops vs Crustle (Single-Prize EX Immunity Hard Counter!)"
            if is_crustle and card_id == TAROUNTULA_ID:
                return 0.40, "CRITICAL Prior: PRIORITIZE Deploy Tarountula vs Crustle (Build Spidops Attacker!)"

            if tr_count == 3:
                return 0.50, "CRITICAL Prior: Play 4th TR Pokemon Unlocking Mewtwo ex Power Saver Ability"
            if tr_count < 4:
                return 0.40, "CRITICAL Prior: Play TR Pokemon to Accelerate Power Saver Unlock"
            if ctx["my_bench_count"] < 2:
                return 0.35, "CRITICAL Prior: Mid-Game Bench Recovery - Rebuild Empty/Low Bench Immediately"
            if ctx["my_bench_count"] == 4 or (SPIDOPS_ID in ctx["my_bench_ids"] or active_id == SPIDOPS_ID):
                return 0.30, "CRITICAL Prior: Fill 5th Bench Slot to Maximize Spidops Damage & Board Power"
            elif ctx["my_bench_count"] <= 2:
                return 0.22, "Prior: Play TR Pokemon Low Bench Recovery Guard"
            # FIX 2 (expert): Early/Mid Mewtwo ex and Tarountula setup priority
            opp_active_hp = ctx.get("opp_active_hp", 100)
            if card_id == MEWTWO_EX_ID and (early_game or opp_active_hp >= 180):
                return 0.35, "CRITICAL Prior: Deploy Mewtwo ex Main Attacker vs High-HP/EX Opponent"
            elif card_id == TAROUNTULA_ID and early_game:
                return 0.25, "Prior: Early Tarountula Setup Deployment (Turn 1-3)"

            # Fast Aggro Matchup Rule: Prioritize deploying Articuno wall & Spidops Trap Territory vs Starmie/Lopunny
            is_fast_aggro_meta = any(k in opponent_name.lower() for k in ("starmie", "lopunny"))
            if is_fast_aggro_meta:
                if card_id == ARTICUNO_ID and ARTICUNO_ID not in ctx["my_bench_ids"] and active_id != ARTICUNO_ID:
                    return 0.40, "CRITICAL Prior: Deploy Articuno 120-HP Wall vs Fast Aggro (Starmie/Lopunny)"
                if card_id in (SPIDOPS_ID, TAROUNTULA_ID):
                    return 0.35, "CRITICAL Prior: Deploy Spidops Line Trap Territory vs Fast Aggro (Starmie/Lopunny)"

            if card_id == MEWTWO_EX_ID:
                return 0.45, "CRITICAL Prior: Bench Mewtwo ex (Prepare on Bench to Charge Towards 3 Energy!)"
            if card_id == SPIDOPS_ID:
                return 0.50, "CRITICAL Prior: Evolve Tarountula into Spidops Attacker (130 HP + Venomous Whip)!"
            if card_id == TAROUNTULA_ID:
                return 0.20, "Prior: Play Spidops Line Trap Territory Setup"
            return 0.10, "Prior: Play Team Rocket Pokemon Setup"


        # B. Search & Tutor Engines
        if card_id == TRANSCEIVER_ID:
            if ctx["my_bench_count"] <= 1 or tr_count < 4:
                return 0.45, "CRITICAL Prior: TR Transceiver Bench & Power Saver Tutor"
            return BONUS_TRANSCEIVER, "Prior: TR Transceiver Supporter Tutor"

        if card_id == BUG_CATCHING_SET_ID:
            if ctx["my_bench_count"] == 5 and tr_count >= 4:
                return -0.15, "Prior Penalty: Bench Full - Do Not Over-Cycle Bug Catching Set"
            if tr_count < 4:
                return 0.45, "CRITICAL Prior: Bug Catching Set - Power Saver Bench Fill Urgency"
            return BONUS_BUG_CATCHING, "Prior: Bug Catching Set Grass Search"

        if card_id == POFFIN_ID:
            if ctx["my_bench_count"] == 5:
                return -0.20, "Prior Penalty: Bench Full - Poffin Has No Legal Targets"
            if ctx["my_bench_count"] <= 1 or tr_count < 4 or early_game:
                if tr_count < 4:
                    return 0.48, "CRITICAL Prior: Poffin Immediate Bench Fill - Power Saver Fill Emergency"
                return 0.38, "CRITICAL Prior: Poffin Immediate Bench Fill"
            return BONUS_LATE_POFFIN, "Prior: Late Poffin Bench Fill"

        if card_id == ULTRA_BALL_ID:
            if ctx["my_bench_count"] == 5 and tr_count >= 4:
                return -0.10, "Prior Penalty: Bench Full & Power Saver Met - Ultra Ball Low Priority"
            has_draw_supporter = any(cid in (ARIANA_ID, LILLIES_DETERM_ID) for cid in ctx["my_hand_ids"])
            if has_draw_supporter and hand_len <= 4:
                return -0.20, "Prior Penalty: Play Draw Supporter (Ariana/Lillie) BEFORE Ultra Ball Discards Hand"
            if ctx["my_bench_count"] <= 1 or tr_count < 4:
                if tr_count < 4:
                    return 0.45, "CRITICAL Prior: Ultra Ball Immediate Power Saver Bench Fill"
                return 0.35, "CRITICAL Prior: Ultra Ball Immediate Bench Fill"
            return BONUS_ULTRA_BALL, "Prior: Ultra Ball Key Search"

        # C. Equipment & Recovery
        if card_id == MAXIMUM_BELT_ID:
            return BONUS_MAXIMUM_BELT, "Prior: Equip Maximum Belt ACE SPEC Snipe Tool"

        if card_id in (HEROS_CAPE_ID, 1159):
            target_poke_id = _resolve_target_pokemon_id_from_opt(obs, opt, ctx)
            if target_poke_id == MEWTWO_EX_ID:
                return 0.55, "CRITICAL Prior: Equip Hero's Cape (+100 HP) to Mewtwo ex Immediately (380 HP Anchor Tank!)"
            elif target_poke_id in (ARTICUNO_ID, 414, MIMIKYU_ID, 434, TAROUNTULA_ID, 400):
                return -0.40, "Prior Penalty: FORBIDDEN - Do NOT Waste Hero's Cape on Non-Primary Attacker!"
            return 0.25, "Prior: Equip Hero's Cape (+100 HP) ACE SPEC Tool"

        if card_id == BRAVE_BANGLE_ID or card_id == MAXIMUM_BELT_ID:
            target_poke_id = _resolve_target_pokemon_id_from_opt(obs, opt, ctx)

            # Strict Tool Target Prohibition: Do NOT attach damage boosting tools (Brave Bangle / Maximum Belt) to non-attackers!
            if target_poke_id in (ARTICUNO_ID, 414, TAROUNTULA_ID, 400, MIMIKYU_ID, 434):
                return -0.45, "Prior Penalty: FORBIDDEN - Do NOT Attach Damage Tool (Brave Bangle/Belt) to Articuno/Non-Attacker!"

            if target_poke_id == MEWTWO_EX_ID:
                return 0.35, "CRITICAL Prior: Attach Damage Tool (Brave Bangle/Belt) to Mewtwo ex Main Attacker"
            elif target_poke_id == SPIDOPS_ID:
                return 0.25, "Prior: Attach Damage Tool (Brave Bangle/Belt) to Spidops Attacker"

            opp_active_hp = ctx.get("opp_active_hp", 0)
            opp_active_id = ctx.get("opp_active_id", -1)
            is_tank_or_ex = opp_active_hp >= 180 or opp_active_id in EX_POKEMON_IDS
            is_fast_aggro = any(k in opponent_name.lower() for k in ("starmie", "ogerpon", "garchomp", "dipplin"))
            if is_tank_or_ex:
                return 0.25, "Prior: Equip Brave Bangle vs Tank/EX Opponent"
            if is_fast_aggro:
                return 0.25, "Prior: Equip Brave Bangle Counter Fast 2-Energy Aggro"
            if active_id in MAIN_ATTACKER_IDS:
                return BONUS_BRAVE_BANGLE_ACTIVE, "Prior: Equip Brave Bangle to Active Main Attacker"
            if any(b in MAIN_ATTACKER_IDS for b in ctx["my_bench_ids"]):
                return BONUS_BRAVE_BANGLE_BENCH, "Prior: Equip Brave Bangle to Bench Main Attacker"
            return 0.06, "Prior: Equip Brave Bangle General"

        if card_id == POKE_PAD_ID:
            is_control = any(k in opponent_name.lower() for k in ("iono", "control", "snorlax", "trevenant"))
            if is_control or hand_len <= 3 or close_game:
                return 0.30, "CRITICAL Prior: Poké Pad Recycle Key Supporter vs Iono/Trevenant Control"
            return 0.08, "Prior: Poké Pad Supporter Recycling"

        if card_id == NIGHT_STRETCHER_ID:
            if close_game:
                return BONUS_NIGHT_STRETCHER_CLOSE, "Prior: Night Stretcher Key Recovery Close Game"
            return BONUS_NIGHT_STRETCHER, "Prior: Night Stretcher Recovery"

        # Stadiums (TR Factory, Prism Tower, Battle Cage)
        # Check if opponent has a hostile stadium in play (Nighttime Mine, Spikemuth Gym, etc.)
        current_stadium = obs.get("current", {}).get("stadium", [])
        has_hostile_stadium = False
        if current_stadium and len(current_stadium) > 0 and current_stadium[0] is not None:
            stadium_id = _get_val(current_stadium[0], "cardId", _get_val(current_stadium[0], "id", -1))
            if stadium_id not in (TR_FACTORY_ID, BATTLE_CAGE_ID, PRISM_TOWER_ID, 1257, 1264, 1180):
                has_hostile_stadium = True

        if card_id == TR_FACTORY_ID:
            if has_hostile_stadium:
                return 0.55, "CRITICAL Prior: Overwrite Hostile Stadium (Nighttime Mine/Spikemuth Gym) with TR Factory!"
            is_control_or_dipplin = any(k in opponent_name.lower() for k in ("iono", "control", "snorlax", "dipplin", "trevenant", "alakazam"))
            if is_control_or_dipplin or early_game:
                return 0.40, "CRITICAL Prior: Play TR Factory Stadium Counter Iono/Trevenant/Alakazam Control"
            return 0.16, "Prior: Play TR Factory Stadium Draw Engine"

        if card_id == PRISM_TOWER_ID:
            if has_hostile_stadium:
                return 0.52, "CRITICAL Prior: Overwrite Hostile Stadium with Prism Tower!"
            return 0.16, "Prior: Play Prism Tower Recurring Discard & Draw"

        if card_id == BATTLE_CAGE_ID:
            if has_hostile_stadium:
                return 0.55, "CRITICAL Prior: Overwrite Hostile Stadium with Battle Cage!"
            bench_counter_keywords = ("dragapult", "froslass", "dusknoir", "dusclops", "munkidori", "dipplin", "phantom", "grimmsnarl", "marnie", "sixth", "spread")
            is_bench_counter_meta = any(k in opponent_name.lower() for k in bench_counter_keywords) or opponent_name == "DRAGAPULT"
            if is_bench_counter_meta:
                return 0.55, "CRITICAL Prior: Play Battle Cage Counter Dragapult/Grimmsnarl/Froslass Bench Snipe"
            return 0.04, "Prior: Play Battle Cage Stadium"

        # D. Positioning & Tempo Items
        if card_id == SWITCH_ID:
            active_e = ctx.get("my_active_energy", 0)
            active_hp = ctx.get("my_active_hp", 280)
            has_bench_attacker = any(b in MAIN_ATTACKER_IDS for b in ctx["my_bench_ids"])
            active_is_main = active_id in MAIN_ATTACKER_IDS

            # Critical Switch: If active is a 1-prize pivot (Articuno/Mimikyu/Tarountula) and Mewtwo ex on bench is fully charged (>= 3 energy, TR >= 4):
            has_fully_powered_mewtwo = False
            my_ps = ctx.get("my_ps")
            if my_ps:
                for b in _get_val(my_ps, "bench", []):
                    if b and _get_val(b, "cardId", _get_val(b, "id", -1)) == MEWTWO_EX_ID:
                        en = _get_val(b, "energyCards", [])
                        if _count_effective_energy(en) >= 3 and tr_count >= 4:
                            has_fully_powered_mewtwo = True
                            break

            if has_fully_powered_mewtwo and active_id != MEWTWO_EX_ID:
                return 0.50, "CRITICAL Prior: Switch to Promote Fully Charged Mewtwo ex Main Attacker!"

            # Spidops Switch Prohibition: Do not switch out healthy Spidops (HP > 30) unless a fully powered attacker is ready on bench
            if active_id in (SPIDOPS_ID, 401) and active_hp > 30 and not has_ready_bench_attacker:
                return -0.35, "Prior Penalty: FORBIDDEN - Do NOT Switch Out Spidops (Spidops Belongs in Active!)"

            if active_id == MEWTWO_EX_ID and active_e >= 3 and active_hp >= 100 and tr_count >= 4:
                return -0.20, "Prior Penalty: Switch out healthy powered Mewtwo ex (strongly discouraged)"

            if not has_ready_bench_attacker and active_hp > 30:
                return -0.20, "Prior Penalty: Pointless Switch - No Fully Powered Attacker on Bench"
            if not active_is_main and has_bench_attacker:
                return BONUS_SWITCH_HEAVY_ATTACKER, "Prior: Switch Non-Attacker for Heavy Attacker"
            if close_game and has_bench_attacker:
                return BONUS_SWITCH_OPTIMIZE_CLOSE, "Prior: Switch to Optimize Damage Close Game"
            if close_game:
                return 0.10, "Prior: Switch Tempo Positioning Close Game"
            return 0.05, "Prior: Switch Tempo Positioning"

        # E. State-Aware Supporters
        if card_id == GIOVANNI_ID:
            # CRITICAL: Giovanni forces a retreat of the active Pokemon as a side-effect.
            # This discards ALL attached energy from the active before we see any RETREAT option.
            # We must block Giovanni here if the active has energy that would be wasted.
            active_energy = ctx.get("my_active_energy", 0)
            active_hp_gio = ctx.get("my_active_hp", 280)

            # Giovanni-triggered energy discard prohibition:
            # If the active Pokemon has energy AND is healthy (not dying, HP > 30),
            # playing Giovanni will silently discard that energy — this is FORBIDDEN.
            if active_energy >= 1 and active_hp_gio > 30:
                # Exception: if no ready benched attacker exists anyway (total dead-end board),
                # Giovanni-gust may still be the only proactive play — softer penalty.
                if active_id in (SPIDOPS_ID, 401, TAROUNTULA_ID, 400):
                    return -0.45, "Prior Penalty: FORBIDDEN - Giovanni Forces Retreat Discarding Spidops/Tarountula Energy (Wastes Attached Energy!)"
                if active_id == MEWTWO_EX_ID and active_energy >= 2:
                    return -0.40, "Prior Penalty: FORBIDDEN - Giovanni Forces Retreat Discarding Mewtwo ex Energy (Wastes Attached Energy!)"
                if active_id == ARTICUNO_ID and active_energy >= 1:
                    return -0.25, "Prior Penalty: Giovanni Forces Retreat Discarding Articuno Energy (Wasteful)"

            if "archaludon" in opponent_name.lower():
                return 0.45, "CRITICAL Prior: Giovanni Gust vs Archaludon (Bypass 300 HP Metal Armor Active)"
            if ctx["opp_bench_count"] > 0:
                opp_active_max_hp = ctx.get("opp_active_max_hp", 100)
                opp_active_hp = ctx.get("opp_active_hp", 100)
                is_high_hp_tank = opp_active_max_hp >= 300 or opp_active_hp >= 250 or any(k in opponent_name.lower() for k in ("starmie", "abomasnow", "wally"))
                if is_high_hp_tank:
                    return 0.42, "CRITICAL Prior: Giovanni Gust vs 300+ HP Tank/Mega EX (Bypass Active Wall/Full Heal Supporters)"
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
            if hand_len <= 4 or tr_count < 4 or early_game:
                return 0.55, "CRITICAL Prior: Fast Ariana Draw (Accelerate TR Energy & TR Pokemon Setup!)"
            return 0.25, "Prior: Ariana Team Rocket Draw"

        if card_id == LILLIES_DETERM_ID:
            if hand_len <= 4:
                return 0.48, "CRITICAL Prior: Lillie Emergency Draw (Play BEFORE Discard Items)"
            return 0.08, "Prior: Lillie Determination Draw"

        if card_id == PROTON_ID:
            if tr_count < 4 or ctx["my_bench_count"] <= 2 or early_game:
                return 0.48, "CRITICAL Prior: Proton Search 3 TR Pokemon for Power Saver Setup"
            return 0.20, "Prior: Proton TR Pokemon Search"

        if card_id in TACTICAL_SUPPORTER_IDS:
            return BONUS_TACTICAL_SUPPORTER, "Prior: TR Tactical Supporter Play"

    # 2. ENERGY ATTACHMENT ACTIONS (OptionType.ATTACH / 8 or Target Selection context)
    select_context = ctx.get("select_context")
    is_attach_phase = (opt_type in (getattr(OptionType, "ATTACH", 8), 8)) or (select_context in (8, getattr(OptionType, "ATTACH", 8), "AttachEnergy"))

    if is_attach_phase:
        energy_cid = card_id if card_id in (1, 2, 3, 4, 5, 15, 19) else 1
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
            opt_target_energy = _count_effective_energy(energies)
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
                    bench_energies.append(_count_effective_energy(ben_en))

        target_is_active = (in_area == 4 or in_area == "active" or opt_target == active_id)
        has_ready_bench_attacker = False
        if my_ps is not None:
            bench_list_raw = _get_val(my_ps, "bench", [])
            for bp in bench_list_raw:
                if bp is not None:
                    bid = _get_val(bp, "cardId", _get_val(bp, "id", -1))
                    ben_en = _get_val(bp, "energyCards", [])
                    eff_e = _count_effective_energy(ben_en)
                    req_b = 3 if bid == MEWTWO_EX_ID else (2 if bid in (SPIDOPS_ID, 401) else 1)
                    if bid == MEWTWO_EX_ID and tr_count < 4:
                        continue
                    if eff_e >= req_b:
                        has_ready_bench_attacker = True
                        break

        # ---------------------------------------------------------------------
        # 1. HARD OVERCHARGE VETOES (Zero Wasted Energy Stacking)
        # ---------------------------------------------------------------------
        # A. Spidops / Tarountula: Max 2 Energy! Never attach 3rd, 4th, 5th energy!
        if opt_target in (SPIDOPS_ID, TAROUNTULA_ID, 401, 400) and opt_target_energy >= 2:
            return -0.60, "Prior Penalty: FORBIDDEN - Spidops/Tarountula already at max 2 energy! (Do NOT overcharge, energize other attackers/Mewtwo!)"

        # B. Mewtwo ex: Max 3 Energy!
        if opt_target == MEWTWO_EX_ID and opt_target_energy >= 3:
            return -0.60, "Prior Penalty: FORBIDDEN - Mewtwo ex already fully powered at 3 energy! (Psywave/Psystrike ready)"

        # C. Articuno: Benched Articuno receives ZERO energy. Active Articuno max 1 energy to pivot.
        if opt_target == ARTICUNO_ID:
            if target_is_active and opt_target_energy == 0 and has_ready_bench_attacker:
                return 0.20, "Prior: Attach 1 Energy to Active Articuno to Pay 1-Retreat Cost and Promote Ready Attacker"
            return -0.50, "Prior Penalty: FORBIDDEN - Do NOT Attach Energy to Articuno (Utility/Pivot cannot attack with Grass/TR energy)"

        # D. Mimikyu: Max 1 TR Energy.
        if opt_target == MIMIKYU_ID and (opt_target_energy >= 1 or energy_cid != TR_ENERGY_ID):
            return -0.45, "Prior Penalty: FORBIDDEN - Mimikyu Max 1 TR Energy Only"

        # ---------------------------------------------------------------------
        # Priority 0: Basic Grass Energy on Spidops vs Enhanced Hammer / Alakazam / Control Meta
        opp_act_id = ctx.get("opp_active_id", -1)
        is_hammer_control = any(k in opponent_name.lower() for k in ("alakazam", "hammer", "control", "xerosic", "snorlax", "dudunsparce")) or (opp_act_id in (741, 742, 743, 65, 66))
        if is_hammer_control and energy_cid == 1 and opt_target in (SPIDOPS_ID, TAROUNTULA_ID, 401, 400) and opt_target_energy < 2:
            return 0.55, "CRITICAL Prior: Power Spidops with Basic Grass Energy (Immune to Enhanced Hammer Discard vs Control Meta!)"

        # Priority 1: Power ACTIVE Attacker to Attack Threshold (Spidops 2 en, Mewtwo ex 3 en)
        if target_is_active and opt_target in (MEWTWO_EX_ID, SPIDOPS_ID):
            req_act = 3 if opt_target == MEWTWO_EX_ID else 2
            if opt_target_energy < req_act:
                p_name = "Mewtwo ex" if opt_target == MEWTWO_EX_ID else "Spidops"
                return 0.52, f"CRITICAL Prior: Power Active {p_name} to Attack (en={opt_target_energy}/{req_act})"

        # Priority 2: Complete Mewtwo ex (Active or Bench) to 3 Energy (1 TR Energy + 1 Grass Energy = 3)
        if opt_target == MEWTWO_EX_ID and opt_target_energy < 3:
            en_gain = 2 if energy_cid in (15, 19) else 1
            if opt_target_energy + en_gain >= 3:
                return 0.52, "CRITICAL Prior: Complete Mewtwo ex to 3 Energy (Psystrike/Psywave Ready!)"
            return 0.48, "CRITICAL Prior: Build Mewtwo ex Energy (Towards 3 Energy for 280 OHKO)"

        # Priority 3: Team Rocket Energy (ID 15, +2 Energy) STRICTLY Reserved for Mewtwo ex if Mewtwo ex is Hungry
        if energy_cid in (15, 19):
            mewtwo_in_play = (active_id == MEWTWO_EX_ID) or (MEWTWO_EX_ID in my_bench_ids)
            mewtwo_needs_en = False
            if active_id == MEWTWO_EX_ID and active_energy < 3:
                mewtwo_needs_en = True
            elif MEWTWO_EX_ID in my_bench_ids and my_ps is not None:
                for bp in _get_val(my_ps, "bench", []):
                    if bp and _get_val(bp, "cardId", _get_val(bp, "id", -1)) == MEWTWO_EX_ID:
                        if _count_effective_energy(_get_val(bp, "energyCards", [])) < 3:
                            mewtwo_needs_en = True
                            break
            if opt_target != MEWTWO_EX_ID and mewtwo_in_play and mewtwo_needs_en:
                return -0.50, "Prior Penalty: Team Rocket Energy STRICTLY Reserved for Mewtwo ex (Provides 2 Colorless Energy!)"

        # Priority 4: Power Benched Backup Spidops / Tarountula (up to 2 Energy Max)
        if not target_is_active and opt_target in (SPIDOPS_ID, TAROUNTULA_ID, 401, 400) and opt_target_energy < 2:
            mewtwo_in_play = (active_id == MEWTWO_EX_ID) or (MEWTWO_EX_ID in my_bench_ids)
            mewtwo_needs_en = False
            if active_id == MEWTWO_EX_ID and active_energy < 3:
                mewtwo_needs_en = True
            elif MEWTWO_EX_ID in my_bench_ids and my_ps is not None:
                for bp in _get_val(my_ps, "bench", []):
                    if bp and _get_val(bp, "cardId", _get_val(bp, "id", -1)) == MEWTWO_EX_ID:
                        if _count_effective_energy(_get_val(bp, "energyCards", [])) < 3:
                            mewtwo_needs_en = True
                            break
            if mewtwo_in_play and mewtwo_needs_en:
                return 0.15, "Prior: Secondary Backup Spidops Energy (Mewtwo ex Still Primary)"
            return 0.38, "Prior: Power Benched Backup Spidops/Tarountula (Max 2 Energy)"

        # --- Generic evaluation fallback via energy_evaluator ---
        try:
            from src.expert_system.energy_evaluator import evaluate_energy_target
        except ImportError:
            from expert_system.energy_evaluator import evaluate_energy_target

        score, reason = evaluate_energy_target(
            target_card_id=opt_target,
            target_current_energy=opt_target_energy,
            energy_card_id=energy_cid,
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
    elif opt_type in (getattr(OptionType, "RETREAT", 12), 12, "Retreat", "retreat"):
        opp_active_id = ctx.get("opp_active_id", -1)
        is_crustle = (opp_active_id in (344, 345, 407, 408)) or ("crustle" in opponent_name.lower())

        # CRUSTLE EX IMMUNITY RETREAT OVERRIDE:
        # If Active Pokemon is an EX Pokemon (Mewtwo ex / 431) facing Crustle, Mewtwo ex deals 0 damage!
        # MUST retreat or switch out immediately to promote non-EX attackers (Spidops / Tarountula / Articuno / Mimikyu).
        if active_id == MEWTWO_EX_ID and is_crustle:
            has_non_ex_bench = any(cid in (400, 401, 414, 434) for cid in ctx.get("my_bench_ids", []))
            if has_non_ex_bench:
                return 0.45, "CRITICAL Prior: Retreat Mewtwo ex Out of Crustle EX Immunity Trap! (Promote Non-EX Attacker)"
            return 0.20, "Prior: Retreat Mewtwo ex Out of Crustle (EX Immunity Trap)"
        active_hp = ctx.get("my_active_hp", 280)
        active_energy = ctx.get("my_active_energy", 0)
        opp_active_id = ctx.get("opp_active_id", -1)

        # HIGHEST PRIORITY: Check if ATTACK is also legally available in this exact observation.
        # If so, retreating is the single worst possible action — the active can attack right now!
        # This check is needed because MCTS with low search count cannot overcome a strong NN policy
        # prior for retreat unless the expert veto is extreme enough.
        select_obj_r = _get_val(obs, "select", {})
        all_opts_r = _get_val(select_obj_r, "option", [])
        attack_is_available = any(_get_val(o, "type", -1) == 13 for o in (all_opts_r if isinstance(all_opts_r, (list, tuple)) else []))
        req_energy_r = ATTACK_ENERGY_COSTS.get(active_id, 2)

        if attack_is_available and active_energy >= req_energy_r and active_hp > 30:
            # ABSOLUTE VETO: Attack is available and active has enough energy — retreat is FORBIDDEN.
            # Score -0.90 cannot be overcome by any realistic neural network policy prior.
            if active_id in (SPIDOPS_ID, 401):
                return -0.90, "ABSOLUTE VETO: FORBIDDEN - ATTACK IS AVAILABLE! Do NOT Retreat Spidops (Venomous Whip CAN fire this turn!)"
            elif active_id == MEWTWO_EX_ID:
                return -0.90, "ABSOLUTE VETO: FORBIDDEN - ATTACK IS AVAILABLE! Do NOT Retreat Mewtwo ex (Psywave CAN fire this turn!)"
            else:
                return -0.80, f"ABSOLUTE VETO: FORBIDDEN - ATTACK IS AVAILABLE! Do NOT Retreat Active (ID {active_id})"

        # Crustle EX Immunity Rule: Retreat active Mewtwo ex if facing Crustle (344, 345, 407, 408)
        if active_id == MEWTWO_EX_ID and (opp_active_id in (344, 345, 407, 408) or "crustle" in opponent_name.lower()):
            return 0.22, "Prior: Retreat Mewtwo ex vs Crustle (EX Immunity Counter - Bring in Spidops)"

        # Check if any benched Pokemon is ACTUALLY fully powered and ready to attack
        has_ready_bench_attacker = False
        my_ps = ctx.get("my_ps")
        if my_ps is not None:
            bench_list_raw = _get_val(my_ps, "bench", [])
            for bp in bench_list_raw:
                if bp is not None:
                    bid = _get_val(bp, "cardId", _get_val(bp, "id", -1))
                    ben_en = _count_effective_energy(_get_val(bp, "energyCards", []))
                    req_b = 3 if bid == MEWTWO_EX_ID else (2 if bid in (SPIDOPS_ID, 401) else 1)
                    if bid == MEWTWO_EX_ID and tr_count < 4:
                        continue  # Power-Saver-locked Mewtwo cannot attack
                    if ben_en >= req_b:
                        has_ready_bench_attacker = True
                        break

        # Strict Spidops / Tarountula Retreat Prohibition:
        # Never retreat Spidops or Tarountula when energy is attached (>= 1).
        if active_id in (SPIDOPS_ID, 400, 401) and active_energy >= 1:
            return -0.50, "Prior Penalty: FORBIDDEN - Do NOT Retreat Spidops/Tarountula (Wastes Attached Energy!)"

        # Healthy Powered Mewtwo ex Retreat Prohibition:
        if active_id == MEWTWO_EX_ID and active_energy >= 2 and active_hp > 120 and tr_count >= 4:
            return -0.55, "Prior Penalty: FORBIDDEN - Do NOT Retreat Healthy Powered Mewtwo ex! Stay Active & Attack with Erasure Ball!"

        # Power Saver Lockout Retreat Prohibition (Match Loss Fix):
        # Never retreat Mewtwo ex with energy when Power Saver is unmet (TR < 4).
        # Retreating discards all energy and promotes a weak basic, giving opponent a free Prize.
        # Passing/End Turn is strictly superior: keeps energy on Mewtwo ex so playing 1 TR Pokemon next turn unlocks 270 damage!
        if active_id == MEWTWO_EX_ID and active_energy >= 1 and tr_count < 4:
            return -0.50, "Prior Penalty: FORBIDDEN - Do NOT Retreat Power-Saver-Locked Mewtwo ex (Keep Energy, Pass/Bench TR Pokemon Next Turn!)"

        # E1 (LATEST evidence): 12 Articuno waste retreats; 7/12 occur when TR<4.
        # When TR<4, Articuno active IS building Power Saver — retreat is partly justified.
        # When TR>=4, Articuno active has zero value — penalty is justified at full strength.
        if active_id == ARTICUNO_ID and active_energy >= 1 and active_hp > 30:
            if tr_count >= 4:
                # Full board: Articuno energy waste with no strategic offset
                energy_pen = -0.40 if active_energy >= 2 else -0.35
                return energy_pen, f"Prior Penalty: Articuno Retreat Energy Waste (TR={tr_count}>=4, en={active_energy})"
            elif tr_count == 3:
                # One short of Power Saver: partial strategic tradeoff — softer penalty
                return -0.20, f"Prior Penalty: Articuno Retreat With Energy (TR={tr_count}, partial tradeoff)"
            else:
                # TR<=2: Articuno active may be necessary — minimal guidance
                return -0.10, f"Prior: Soft Penalty Articuno Retreat With Energy (TR={tr_count} low)"

        # Fast Aggro Suicidal Retreat Prohibition (Starmie / Lopunny):
        # Never retreat active into a benched Pokemon with < 50 HP when facing fast attackers (Starmie / Lopunny)
        is_fast_aggro_r = any(k in opponent_name.lower() for k in ("starmie", "lopunny"))
        if is_fast_aggro_r:
            my_ps_r = ctx.get("my_ps")
            bench_list_r = _get_val(my_ps_r, "bench", []) if my_ps_r else []
            has_low_hp_bench = any(bp and _get_val(bp, "hp", 100) < 50 for bp in bench_list_r)
            if has_low_hp_bench and not has_ready_bench_attacker:
                return -0.50, "Prior Penalty: FORBIDDEN - Do NOT Retreat into Fragile <50 HP Bench Target vs Fast Aggro (Starmie/Lopunny)"

        # Pointless Unpowered Active Retreat Prohibition (Loop Prevention):
        # Never retreat a healthy active Pokemon (HP > 30) when energy is 0 and no ready attacker exists on bench
        if active_energy == 0 and active_hp > 30 and not has_ready_bench_attacker:
            return -0.45, "Prior Penalty: FORBIDDEN - Pointless Retreat of Unpowered Active (Loop Prevention)"

        if active_energy >= 1 and active_hp > 30:
            return -0.35, f"Prior Penalty: Retreat Forbidden for Active with Energy (ID {active_id}) - ATTACK INSTEAD"

        # Wasted retreat prevention: never discard energy to promote a 0-energy bench Pokemon unless active is dying (hp <= 30)
        if active_energy >= 1 and not has_ready_bench_attacker:
            return -0.35, "Prior Penalty: Wasted Retreat - No Benched Attacker Ready to Attack"

        if active_id == MEWTWO_EX_ID and active_energy >= 1 and active_hp >= 40:
            return -0.35, "Prior Penalty: Retreat healthy/powered Mewtwo ex (energy waste, forbidden)"
        elif active_energy >= 1 and active_hp >= 60:
            return -0.25, "Prior Penalty: Retreat healthy active with energy (wasteful)"
        elif active_hp <= 30 and has_ready_bench_attacker:
            return 0.15, "Prior: Tactical retreat of low-HP active to ready benched attacker"
        return -0.20, "Prior Penalty: General wasteful retreat"

    # 5. ATTACK ACTIONS (OptionType.ATTACK / 13)
    elif opt_type in (getattr(OptionType, "ATTACK", 13), 13):
        req_energy = ATTACK_ENERGY_COSTS.get(active_id, 2)
        current_energy = ctx["my_active_energy"]
        opp_active_id = ctx.get("opp_active_id", -1)

        # Crustle EX Immunity Check: Mewtwo ex deals 0 damage to Crustle
        if active_id == MEWTWO_EX_ID and (opp_active_id in (344, 345, 407, 408) or "crustle" in opponent_name.lower()):
            return -0.90, "ABSOLUTE VETO: FORBIDDEN - Mewtwo ex Attack Deals 0 Damage to Crustle (EX Immunity Trap!)"

        if active_id == MEWTWO_EX_ID and tr_count < 4:
            return PENALTY_MEWTWO_POWER_SAVER_UNMET, "Prior Penalty: Mewtwo ex Power Saver Unmet (<4 TR Pokemon)"

        if current_energy >= req_energy:
            opp_hp = ctx.get("opp_active_hp", 9999)
            # Universal Lethal Window KO check: if opponent active HP is low enough for a KO, execute immediately!
            if active_id == MEWTWO_EX_ID and opp_hp <= 260:
                return 0.48, "CRITICAL Prior: Mewtwo ex Fully Powered - LETHAL KO WINDOW - ATTACK NOW"
            elif active_id == SPIDOPS_ID and opp_hp <= 180:
                return 0.48, "CRITICAL Prior: Spidops Powered - LETHAL KO WINDOW - ATTACK NOW"
            elif active_id == ARTICUNO_ID and opp_hp <= 60:
                return 0.45, "CRITICAL Prior: Articuno Snipe - LETHAL KO WINDOW - ATTACK NOW"

            if active_id == SPIDOPS_ID:
                bench_size = ctx.get("my_bench_count", len(ctx["my_bench_ids"]))
                is_fast_aggro = any(k in opponent_name.lower() for k in ("starmie", "lopunny", "garchomp"))
                if is_fast_aggro:
                    return 0.45, "CRITICAL Prior: Spidops Attack NOW vs Fast Aggro (prize race urgency)"
                if bench_size >= 4:
                    return 0.45, "CRITICAL Prior: Spidops Full Bench Attack - ATTACK NOW FOR KO"
                return 0.40, "CRITICAL Prior: Spidops Powered Attack (Venomous Whip) - ATTACK NOW"
            elif active_id == MEWTWO_EX_ID:
                return 0.40, "CRITICAL Prior: Mewtwo ex Fully Powered - ATTACK NOW"
            elif active_id == ARTICUNO_ID:
                return BONUS_ARTICUNO_ATTACK, "Prior: Affordable Articuno Snipe Attack"
            return 0.35, "Prior: Powered Active Attacker - ATTACK NOW"
        else:
            return BONUS_UNDERPOWERED_ATTACK, "Prior: Underpowered Active Attack"

    # 6. SEARCH CANDIDATE CARD SELECTION (OptionType.CARD / 3)
    elif opt_type in (getattr(OptionType, "CARD", 3), 3):
        target_card_id = _resolve_card_id_from_opt(obs, opt, ctx)
        if target_card_id > 0:
            select_obj_c = _get_val(obs, "select", {})
            select_reason = str(_get_val(select_obj_c, "reason", "")) + str(_get_val(select_obj_c, "context", ""))
            opt_area = _get_val(opt, "area", -1)
            is_discard_prompt = "discard" in select_reason.lower() or "cost" in select_reason.lower() or opt_area == 2

            if is_discard_prompt:
                if target_card_id in SUPPORTER_IDS:
                    supporter_count = sum(1 for cid in ctx["my_hand_ids"] if cid in SUPPORTER_IDS)
                    if supporter_count <= 1:
                        return -0.65, "Prior Penalty: FORBIDDEN - Do NOT Discard Sole Supporter Card (Ariana/Lillie/Giovanni)!"
                if target_card_id in (1, 15, 19):
                    energy_count = sum(1 for cid in ctx["my_hand_ids"] if cid in (1, 15, 19))
                    if energy_count <= 1:
                        return -0.55, "Prior Penalty: FORBIDDEN - Do NOT Discard Sole Energy Card in Hand!"

            try:
                from src.expert_system.search_strategy import evaluate_search_target_prior, extract_search_context
            except ImportError:
                from expert_system.search_strategy import evaluate_search_target_prior, extract_search_context

            search_ctx = extract_search_context(obs)
            prior_score, reason = evaluate_search_target_prior(target_card_id, search_ctx)
            return prior_score, f"Search Candidate Prior: {reason}"
        return 0.0, "Search Candidate: Unknown Card"

    # 7. PASS / END TURN ACTIONS (OptionType.PASS / 14)
    elif opt_type in (getattr(OptionType, "PASS", 14), 14, "Pass", "pass", "EndTurn"):
        select_obj_p = _get_val(obs, "select", {})
        all_opts_p = _get_val(select_obj_p, "option", [])
        if isinstance(all_opts_p, (list, tuple)):
            has_attach_opt = any(_get_val(o, "type", -1) == 8 for o in all_opts_p)
            has_play_opt = any(_get_val(o, "type", -1) in (7, 10) for o in all_opts_p)
            has_attack_opt = any(_get_val(o, "type", -1) == 13 for o in all_opts_p)

            if has_attach_opt:
                return -0.70, "Prior Penalty: FORBIDDEN - Do NOT Pass Turn with Unattached Energy in Hand!"
            if has_play_opt and turn > 1:
                return -0.50, "Prior Penalty: FORBIDDEN - Do NOT Pass Turn with Unplayed Cards/Abilities in Hand!"
            if has_attack_opt:
                return -0.80, "Prior Penalty: FORBIDDEN - Do NOT Pass Turn When Attack is Available!"
        return -0.15, "Prior: Standard Pass Turn"

    return 0.0, ""


# Backward-compatible API binding
get_expert_bonus = evaluate_mewtwo_expert_bonus


