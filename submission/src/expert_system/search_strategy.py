"""
Intelligent Search Strategy Engine (card_search_engine / search_strategy.py).
Provides matchup-aware, phase-aware, and combo-aware target prioritization for 
all search Supporters (TR Petrel, TR Proton) and search Items (Ultra Ball, Poffin, 
Bug Catching Set, TR Transceiver, Night Stretcher, Battle Cage, Earthen Vessel, Energy Switch).
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Iterable

# Card ID Constants
BASIC_G_ENERGY_ID    = 1    # Basic {G} Energy
TR_ENERGY_ID         = 15   # Team Rocket's Energy

TAROUNTULA_ID        = 400  # Team Rocket's Tarountula
SPIDOPS_ID           = 401  # Team Rocket's Spidops
ARTICUNO_ID          = 414  # Team Rocket's Articuno
MEWTWO_EX_ID         = 431  # Team Rocket's Mewtwo ex
MIMIKYU_ID           = 434  # Team Rocket's Mimikyu
LILLIES_CLEFAIRY_EX_ID = 272 # Lillie's Clefairy ex (Cross-deck Dragon counter)

POFFIN_ID            = 1086 # Buddy-Buddy Poffin
BUG_CATCHING_SET_ID  = 1094 # Bug Catching Set
ENERGY_SWITCH_ID      = 1095 # Energy Switch (re-attach energy)
SWITCH_ID            = 1097 # Switch
EARTHEN_VESSEL_ID     = 1106 # Earthen Vessel (search basics, discard hand)
ULTRA_BALL_ID        = 1121 # Ultra Ball
TRANSCEIVER_ID       = 1134 # Team Rocket's Transceiver
POKE_PAD_ID          = 1152 # Poké Pad
MAXIMUM_BELT_ID      = 1158 # Maximum Belt (ACE SPEC Tool)
HERO_CAPE_ID         = 1159 # Hero's Cape (ACE SPEC Tool)
NIGHT_STRETCHER_ID   = 1159 # Night Stretcher (Recovery Item)
BRAVE_BANGLE_ID      = 1175 # Brave Bangle
PRISM_TOWER_ID        = 1180 # Prism Tower (Discard 2 -> Draw 1 Stadium)

ARIANA_ID            = 1216 # Team Rocket's Ariana
GIOVANNI_ID          = 1218 # Team Rocket's Giovanni
PETREL_ID            = 1219 # Team Rocket's Petrel
PROTON_ID            = 1220 # Team Rocket's Proton
LILLIES_DETERM_ID    = 1227 # Lillie's Determination

TR_FACTORY_ID        = 1257 # Team Rocket's Factory
BATTLE_CAGE_ID       = 1264 # Battle Cage (Anti-Dragapult / Bench Protection)

TR_POKEMON_SET = frozenset({TAROUNTULA_ID, SPIDOPS_ID, ARTICUNO_ID, MEWTWO_EX_ID, MIMIKYU_ID, LILLIES_CLEFAIRY_EX_ID})
BENCH_COUNTER_ARCHETYPES = ("dragapult", "froslass", "dusknoir", "dusclops", "munkidori")
EX_META_ARCHETYPES       = ("mewtwo", "grimmsnarl", "charizard", "archaludon", "gardevoir")

DISCARD_ENGINE_CARDS = frozenset({
    PRISM_TOWER_ID, NIGHT_STRETCHER_ID, ENERGY_SWITCH_ID,
    ULTRA_BALL_ID, EARTHEN_VESSEL_ID, POKE_PAD_ID, POFFIN_ID
})

# Discard Engine Tuning Constants
REWARD_DISCARD_ENGINE_EARLY          = 0.20
REWARD_DISCARD_ENGINE_MID            = 0.12
REWARD_RECOVERY_FROM_DISCARD         = 0.18
REWARD_ENERGY_REDEPLOY_AFTER_DISCARD = 0.16


def _get_val(obj, key: str, default=None):
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



def _count_tr_pokemon_in_play(ctx: dict) -> int:
    """Counts Team Rocket Pokémon currently in play (Active + Bench)."""
    active_id = ctx.get("my_active_id", -1)
    bench_ids = ctx.get("my_bench_ids", [])
    count = 0
    for cid in [active_id] + list(bench_ids):
        if cid in TR_POKEMON_SET:
            count += 1
    return count


def _is_discard_engine_card(card_id: int) -> bool:
    """Returns True if card_id is part of the brilliant discard / recovery engine."""
    return card_id in DISCARD_ENGINE_CARDS


def extract_search_context(obs: dict) -> dict:
    """Extracts state board context for search target scoring."""
    ctx = {
        "turn": 1,
        "my_prizes": 6,
        "opp_prizes": 6,
        "my_active_id": -1,
        "my_bench_ids": [],
        "my_hand_ids": [],
        "my_hand_len": 0,
        "my_bench_count": 0,
        "my_discard_ids": [],
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

        prizes_my = _get_val(my_ps, "prize", [])
        prizes_opp = _get_val(opp_ps, "prize", [])
        ctx["my_prizes"] = len(prizes_my) if isinstance(prizes_my, (list, tuple, set)) else 6
        ctx["opp_prizes"] = len(prizes_opp) if isinstance(prizes_opp, (list, tuple, set)) else 6

        my_active_list = _get_val(my_ps, "active", [])
        if my_active_list and len(my_active_list) > 0 and my_active_list[0] is not None:
            ctx["my_active_id"] = _get_val(my_active_list[0], "cardId", -1)

        my_bench = _get_val(my_ps, "bench", [])
        if isinstance(my_bench, (list, tuple)):
            ctx["my_bench_ids"] = [_get_val(b, "cardId", -1) for b in my_bench if b is not None]
            ctx["my_bench_count"] = len(ctx["my_bench_ids"])

        my_hand = _get_val(my_ps, "hand", [])
        if isinstance(my_hand, (list, tuple)):
            ctx["my_hand_ids"] = [_get_val(h, "cardId", -1) for h in my_hand if h is not None]
            ctx["my_hand_len"] = len(ctx["my_hand_ids"])

        my_discard = _get_val(my_ps, "discard", [])
        if isinstance(my_discard, (list, tuple)):
            ctx["my_discard_ids"] = [_get_val(d, "cardId", -1) for d in my_discard if d is not None]

    return ctx


def score_search_target(card_id: int, ctx: dict, search_card_id: int = -1, opponent_name: str = "") -> float:
    """
    Computes multi-factor priority score (0.0 to 1.0) based on:
    Score = Base(0.50) + Combo + Counter + Immediate Attack + Draw + Discard Engine + Recovery + Gust - Duplicates
    """
    base_score = 0.50
    turn = ctx["turn"]
    hand_ids = ctx["my_hand_ids"]
    hand_len = ctx["my_hand_len"]
    bench_ids = ctx["my_bench_ids"]
    active_id = ctx["my_active_id"]
    opp_lower = opponent_name.lower()

    is_bench_counter_matchup = any(k in opp_lower for k in BENCH_COUNTER_ARCHETYPES)
    is_ex_matchup = any(k in opp_lower for k in EX_META_ARCHETYPES)
    early_game = turn <= 2 or ctx["my_prizes"] >= 5
    close_game = ctx["my_prizes"] <= 2 or ctx["opp_prizes"] <= 2
    tr_count = _count_tr_pokemon_in_play(ctx)

    score_components = 0.0

    # 1a. PRISM TOWER (Recurring discard & draw engine)
    if card_id == PRISM_TOWER_ID:
        if early_game or hand_len <= 3:
            score_components += 0.28
        else:
            score_components += 0.14
        if TR_FACTORY_ID in hand_ids or TR_FACTORY_ID in bench_ids:
            score_components += 0.06  # Prism Tower + Factory thinning loop

    # 1b. STADIUM SEQUENCE (TR Factory first for early draw engine & deck thinning)
    elif card_id == TR_FACTORY_ID:
        if early_game or hand_len <= 3:
            score_components += 0.30
        else:
            score_components += 0.12
        if PRISM_TOWER_ID in hand_ids:
            score_components += 0.06  # Factory + Prism Tower loop

    # 1c. COUNTER COMPONENT (Battle Cage vs Bench-Sniper)
    elif card_id == BATTLE_CAGE_ID:
        if is_bench_counter_matchup:
            has_factory_in_hand = TR_FACTORY_ID in hand_ids
            has_cage_in_hand = BATTLE_CAGE_ID in hand_ids
            if not has_factory_in_hand and not has_cage_in_hand:
                score_components += 0.45
            else:
                score_components += 0.25
        else:
            score_components += 0.05
            if BATTLE_CAGE_ID in hand_ids:
                score_components -= 0.10  # Duplicate penalty for 2nd Cage if not heavy spread matchup

    # 2. COMBO EVOLUTION + POWER SAVER (Spidops / Tarountula)
    elif card_id == SPIDOPS_ID:
        has_tarountula = (active_id == TAROUNTULA_ID or TAROUNTULA_ID in bench_ids)
        has_spidops_in_hand = SPIDOPS_ID in hand_ids
        if has_tarountula and not has_spidops_in_hand:
            bonus = 0.42
            if tr_count <= 3:
                bonus += 0.08  # Push toward 4 TR Pokémon
            score_components += bonus
        else:
            score_components += 0.20

    # 3. MAIN HEAVY ATTACKER TUTOR (Mewtwo ex)
    elif card_id == MEWTWO_EX_ID:
        has_mewtwo_in_play = (active_id == MEWTWO_EX_ID or MEWTWO_EX_ID in bench_ids)
        if not has_mewtwo_in_play:
            base_bonus = 0.40
            if tr_count == 3:
                base_bonus += 0.10  # Extra bonus if grabbing Mewtwo ex completes 4-TR requirement
            score_components += base_bonus
        else:
            score_components += 0.15

    # 4. GUST / WIN CLOSE SUPPORTER (TR Giovanni)
    elif card_id == GIOVANNI_ID:
        if close_game or is_ex_matchup:
            score_components += 0.38
            if close_game:
                score_components += 0.15  # Extra game-ending KO gust priority
        else:
            score_components += 0.22

    # 5. DRAW ENGINE SUPPORTERS (Ariana / Lillie's Determination)
    elif card_id in (ARIANA_ID, LILLIES_DETERM_ID):
        if hand_len <= 3:
            score_components += 0.36
        else:
            score_components += 0.18

    # 6. BASIC BENCH SETUP (Tarountula / Mimikyu)
    elif card_id in (TAROUNTULA_ID, MIMIKYU_ID):
        if early_game or ctx["my_bench_count"] <= 1 or tr_count < 4:
            score_components += 0.34
        else:
            score_components += 0.05

    # 7. EQUIPMENT / TOOLS (Brave Bangle / Maximum Belt / Hero's Cape)
    elif card_id in (BRAVE_BANGLE_ID, MAXIMUM_BELT_ID, HERO_CAPE_ID):
        has_main_attacker = (active_id in (MEWTWO_EX_ID, SPIDOPS_ID) or any(b in (MEWTWO_EX_ID, SPIDOPS_ID) for b in bench_ids))
        if card_id == MAXIMUM_BELT_ID:
            score_components += 0.35  # High priority ACE SPEC snipe tool
        elif has_main_attacker and is_ex_matchup:
            score_components += 0.32
        else:
            score_components += 0.10

    # 8. ENERGY ATTACHMENT / ACCELERATION
    elif card_id in (TR_ENERGY_ID, BASIC_G_ENERGY_ID):
        if early_game or active_id in (MEWTWO_EX_ID, SPIDOPS_ID):
            score_components += 0.28
        else:
            score_components += 0.00

    # 9. RECOVERY & RECYCLING (Night Stretcher / Poké Pad)
    elif card_id == NIGHT_STRETCHER_ID:
        discard_ids = ctx.get("my_discard_ids", [])
        tr_in_discard = sum(1 for cid in discard_ids if cid in TR_POKEMON_SET)
        if tr_in_discard >= 2:
            score_components += 0.26
        elif tr_in_discard == 1:
            score_components += 0.14
        else:
            score_components += 0.06

    elif card_id == POKE_PAD_ID:
        if close_game or hand_len <= 3:
            score_components += 0.30
        else:
            score_components += 0.10

    # 10. BRILLIANT DISCARD ENGINE COMPONENT
    if _is_discard_engine_card(card_id):
        if early_game:
            score_components += REWARD_DISCARD_ENGINE_EARLY
        elif close_game:
            score_components += REWARD_RECOVERY_FROM_DISCARD
        else:
            score_components += REWARD_DISCARD_ENGINE_MID

        if card_id == ENERGY_SWITCH_ID:
            has_mewtwo = (active_id == MEWTWO_EX_ID or MEWTWO_EX_ID in bench_ids)
            if has_mewtwo:
                score_components += REWARD_ENERGY_REDEPLOY_AFTER_DISCARD

    # =========================================================================
    # SEARCH-CARD SPECIFIC TARGET BIASES
    # =========================================================================
    is_petrel = (search_card_id == PETREL_ID)
    is_proton = (search_card_id == PROTON_ID)
    is_ultra_ball = (search_card_id == ULTRA_BALL_ID)
    is_transceiver = (search_card_id == TRANSCEIVER_ID)
    is_earthen_vessel = (search_card_id == EARTHEN_VESSEL_ID)

    # Petrel (Supporter search) biases toward Supporters & Stadiums
    if is_petrel:
        if card_id in (ARIANA_ID, LILLIES_DETERM_ID, GIOVANNI_ID, PROTON_ID, PETREL_ID):
            score_components += 0.08
        if card_id in (TR_FACTORY_ID, PRISM_TOWER_ID):
            score_components += 0.06

    # Proton (TR Pokémon search) biases toward TR Pokémon completing Power Saver
    if is_proton:
        if card_id in (TAROUNTULA_ID, SPIDOPS_ID, MEWTWO_EX_ID):
            if tr_count < 4:
                score_components += 0.10
        if card_id == LILLIES_CLEFAIRY_EX_ID and any(k in opp_lower for k in ("dragon", "regidrago", "giratina", "roaring")):
            score_components += 0.12

    # Ultra Ball / Earthen Vessel / Transceiver biases
    if is_ultra_ball or is_transceiver or is_earthen_vessel:
        if card_id in (TAROUNTULA_ID, SPIDOPS_ID, MEWTWO_EX_ID):
            if tr_count < 4:
                score_components += 0.10
        if card_id in (BASIC_G_ENERGY_ID, TR_ENERGY_ID):
            has_spidops_or_mewtwo = (
                active_id in (SPIDOPS_ID, MEWTWO_EX_ID) or
                any(b in (SPIDOPS_ID, MEWTWO_EX_ID) for b in bench_ids)
            )
            if has_spidops_or_mewtwo:
                score_components += 0.08
        if card_id in (ARIANA_ID, LILLIES_DETERM_ID, GIOVANNI_ID, TR_FACTORY_ID, PRISM_TOWER_ID, BATTLE_CAGE_ID, BRAVE_BANGLE_ID, MAXIMUM_BELT_ID):
            score_components += 0.06

    # SUPPORTER DEDUPLICATION PENALTY (-0.12 if 2+ Supporters already in hand)
    if card_id in (ARIANA_ID, LILLIES_DETERM_ID, GIOVANNI_ID, PROTON_ID, PETREL_ID):
        supporter_count_in_hand = sum(1 for cid in hand_ids if cid in (ARIANA_ID, LILLIES_DETERM_ID, GIOVANNI_ID, PROTON_ID, PETREL_ID))
        if supporter_count_in_hand >= 2:
            score_components -= 0.12

    # GENERAL DEDUPLICATION PENALTY (-0.15 if 2+ copies already in hand)
    if hand_ids.count(card_id) >= 2:
        score_components -= 0.15

    final_score = base_score + score_components
    return max(0.10, min(0.99, final_score))


def choose_best_search_target(
    obs: dict,
    candidate_card_ids: List[int],
    search_card_id: int = -1,
    opponent_name: str = ""
) -> Tuple[int, float]:
    """
    Evaluates candidate card IDs and returns the single highest scoring target.
    Returns: (best_card_id: int, top_score: float)
    """
    if not candidate_card_ids:
        return -1, 0.0

    ctx = extract_search_context(obs)
    scored_candidates = []

    for cid in candidate_card_ids:
        sc = score_search_target(cid, ctx, search_card_id, opponent_name)
        scored_candidates.append((sc, cid))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_cid = scored_candidates[0]
    return best_cid, best_score
