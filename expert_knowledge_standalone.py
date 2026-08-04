from __future__ import annotations
# Auto-generated standalone expert_knowledge.py
# Merged from src/configs/active_deck.py + src/expert_system/*.py
# No src/ folder needed on Kaggle — all logic is self-contained here


# ===== active_deck.py =====
import os


def _detect_deck_from_csv() -> str:
    paths = ["deck.csv", "/kaggle_simulations/agent/deck.csv"]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "648" in content or "646" in content:
                        return "GRIMMSNARL"
                    if "431" in content or "150" in content or "151" in content:
                        return "MEWTWO"
                    if "400" in content:
                        return "LILLIE"
            except Exception:
                pass
    return "GRIMMSNARL"


ACTIVE_DECK = os.getenv("ACTIVE_DECK", _detect_deck_from_csv()).upper()


def get_active_deck_name() -> str:
    """Returns the current active deck name (e.g. 'MEWTWO', 'LILLIE', 'GRIMMSNARL')."""
    return ACTIVE_DECK


def set_active_deck(deck_name: str) -> None:
    """Programmatically updates the active deck setting."""
    global ACTIVE_DECK
    ACTIVE_DECK = deck_name.upper()


# ===== base_expert.py =====
"""
Base Expert System & Context Extractor for Pokémon TCG AI Agent.
"""
from cg.api import CardType, OptionType, SelectContext

USE_EXPERT_GUIDANCE = True
EXPERT_WEIGHT = 0.10


def extract_board_context(obs) -> dict:
    """
    Extracts essential board state parameters for heuristic rule evaluation.
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
        "turn": 0,
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
                ctx["opp_bench_count"] = len(bench_ids)

                if opp_active_list and opp_active_list[0] is not None:
                    a = opp_active_list[0]
                    if isinstance(a, dict):
                        ctx["opp_active_id"] = a.get("cardId", a.get("id", -1))
                        ctx["opp_active_hp"] = a.get("hp", 999)
                    else:
                        ctx["opp_active_id"] = getattr(a, "cardId", getattr(a, "id", -1))
                        ctx["opp_active_hp"] = getattr(a, "hp", 999)

                prizes = ctx["my_prizes"]
                if prizes <= 2:
                    ctx["game_phase"] = "late"
                elif prizes <= 4:
                    ctx["game_phase"] = "mid"
                else:
                    ctx["game_phase"] = "early"

                my_hand = getattr(my_ps, "hand", []) if not isinstance(my_ps, dict) else my_ps.get("hand", [])
                hand_ids = []
                for card in my_hand:
                    if card is not None:
                        hand_ids.append(card.get("cardId", card.get("id", -1)) if isinstance(card, dict) else getattr(card, "cardId", getattr(card, "id", -1)))
                ctx["my_hand_ids"] = hand_ids

                my_bench = getattr(my_ps, "bench", []) if not isinstance(my_ps, dict) else my_ps.get("bench", [])
                my_bench_ids = []
                for p in my_bench:
                    if p is not None:
                        my_bench_ids.append(p.get("cardId", p.get("id", -1)) if isinstance(p, dict) else getattr(p, "cardId", getattr(p, "id", -1)))
                ctx["my_bench_ids"] = my_bench_ids
                ctx["my_bench_count"] = len(my_bench_ids)
    except Exception:
        pass
    return ctx

# ===== search_strategy.py =====
"""
Intelligent Search Strategy Engine (card_search_engine / search_strategy.py).
Provides matchup-aware, phase-aware, and combo-aware target prioritization for
all search Supporters (TR Petrel, TR Proton) and search Items (Ultra Ball, Poffin,
Bug Catching Set, TR Transceiver, Night Stretcher, Battle Cage, Earthen Vessel, Energy Switch).
"""
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

# ===== mewtwo_expert.py =====
"""
Team Rocket Mewtwo ex Heuristic Expert Engine (v3 Refined).
Ground Truth: Matches 60-card CSV list (deck lama.csv / deck_mewtwo.csv).
Provides phase-aware strategic action prior guidance (range [-0.20, +0.20]) for MCTS exploration.
"""

from cg.api import OptionType

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
BONUS_EARLY_POFFIN                   = 0.22
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
            if active_id in MAIN_ATTACKER_IDS:
                return BONUS_BRAVE_BANGLE_ACTIVE, "Prior: Equip Brave Bangle to Active Main Attacker"
            if any(b in MAIN_ATTACKER_IDS for b in ctx["my_bench_ids"]):
                return BONUS_BRAVE_BANGLE_BENCH, "Prior: Equip Brave Bangle to Bench Main Attacker"
            return 0.06, "Prior: Equip Brave Bangle General"

        if card_id == POKE_PAD_ID:
            if hand_len <= 3 or close_game:
                return BONUS_POKE_PAD_CLOSE if close_game else BONUS_POKE_PAD, "Prior: Poké Pad Recycle Key Supporter"
            return 0.08, "Prior: Poké Pad Supporter Recycling"

        if card_id == NIGHT_STRETCHER_ID:
            if close_game:
                return BONUS_NIGHT_STRETCHER_CLOSE, "Prior: Night Stretcher Key Recovery Close Game"
            return BONUS_NIGHT_STRETCHER, "Prior: Night Stretcher Recovery"

        if card_id == TR_FACTORY_ID:
            if early_game:
                return BONUS_TR_FACTORY_EARLY, "Prior: Play TR Factory Stadium Early Draw Engine"
            return BONUS_TR_FACTORY_LATE, "Prior: Play TR Factory Stadium Late Game"

        if card_id == PRISM_TOWER_ID:
            return 0.14, "Prior: Play Prism Tower Recurring Discard & Draw"

        if card_id == BATTLE_CAGE_ID:
            bench_counter_keywords = ("dragapult", "froslass", "dusknoir", "dusclops", "munkidori")
            is_bench_counter_meta = any(k in opponent_name.lower() for k in bench_counter_keywords)
            if is_bench_counter_meta:
                return 0.18, "Prior: Play Battle Cage Counter Bench Damage Counter Archetype"
            return 0.12, "Prior: Play Battle Cage Bench Protection"

        # D. Positioning & Tempo Items
        if card_id == SWITCH_ID:
            has_bench_attacker = any(b in MAIN_ATTACKER_IDS for b in ctx["my_bench_ids"])
            active_is_main = active_id in MAIN_ATTACKER_IDS

            # FIX 1 (expert): Penalize switching if neither active nor bench has energy
            active_e = ctx.get("my_active_energy", 0)
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
        if card_id in (BASIC_G_ENERGY_ID, TR_ENERGY_ID) or card_id > 0:
            req_energy_active = ATTACK_ENERGY_COSTS.get(active_id, 2)
            active_energy = ctx["my_active_energy"]
            active_is_maxed = (active_energy >= req_energy_active)

            # Check target Pokemon
            opt_target = getattr(opt, "targetCardId", active_id) if not isinstance(opt, dict) else opt.get("targetCardId", active_id)

            # FIX 6 (expert): Spidops energy hard cap at 2
            opt_target_energy = ctx.get("target_energy", 0)
            ENERGY_HARD_CAP_EX = {MEWTWO_EX_ID: 3, SPIDOPS_ID: 2, ARTICUNO_ID: 2, MIMIKYU_ID: 1, TAROUNTULA_ID: 1}
            if opt_target in ENERGY_HARD_CAP_EX and opt_target_energy >= ENERGY_HARD_CAP_EX[opt_target]:
                return -0.20, f"Prior Penalty: Energy Hard Cap Exceeded on {opt_target} (already at max)"

            # Strict Guard: Articuno must NEVER use Team Rocket Energy!
            if opt_target == ARTICUNO_ID or active_id == ARTICUNO_ID:
                if card_id == TR_ENERGY_ID:
                    return -0.18, "Prior Penalty: Do NOT attach Team Rocket Energy to Articuno!"
                if card_id == BASIC_G_ENERGY_ID and active_energy < 1:
                    return 0.12, "Prior: Attach Basic Grass Energy to Articuno for 1-Energy Retreat"

            if opt_target == active_id:
                if active_is_maxed:
                    # Penalize over-attaching energy to active when already maxed out
                    return -0.05, "Prior Penalty: Active Pokemon Energy Maxed (Over-Attaching)"
                if active_id == MEWTWO_EX_ID:
                    return BONUS_ENERGY_ACTIVE_MEWTWO, "Prior: Energy Attachment Readying Active Mewtwo ex"
                elif active_id in SPIDOPS_LINE_IDS:
                    return BONUS_ENERGY_SPIDOPS, "Prior: Energy Attachment Readying Spidops Line"
                return 0.10, "Prior: Readying Active Attacker"
            else:
                # Energy attached to BENCH Pokemon (Mewtwo ex / Spidops / Tarountula)
                bench_prior = 0.18 if active_is_maxed else BONUS_ENERGY_BENCH_MEWTWO
                if MEWTWO_EX_ID in ctx["my_bench_ids"]:
                    return bench_prior, "Prior: Bench Energy Attachment to Mewtwo ex"
                elif any(b in SPIDOPS_LINE_IDS for b in ctx["my_bench_ids"]):
                    return bench_prior - 0.02, "Prior: Bench Energy Attachment to Spidops Line"
                return 0.08, "Prior: General Bench Energy Attachment"



    # 3. EVOLVE ACTIONS (OptionType.EVOLVE / 9)
    elif opt_type in (getattr(OptionType, "EVOLVE", 9), 9):
        if card_id == SPIDOPS_ID or card_id > 0:
            if active_id == TAROUNTULA_ID or TAROUNTULA_ID in ctx["my_bench_ids"]:
                return BONUS_EVOLVE_SPIDOPS, "Prior: Evolve Tarountula to Spidops"
            return 0.10, "Prior: Evolve to Spidops"

    # 4. ATTACK ACTIONS (OptionType.ATTACK / 13)
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
                    # Thin bench — low damage, prefer to fill bench first
                    return -0.08, f"Prior Penalty: Spidops Thin Bench ({bench_size}) Only {spidops_damage} Damage"
                mewtwo_bench_present = MEWTWO_EX_ID in ctx["my_bench_ids"]
                if mewtwo_bench_present:
                    return BONUS_SPIDOPS_ATTACK_MEWTWO_PRIORITY, "Prior: Spidops Attack (Mewtwo On Bench)"
                return BONUS_SPIDOPS_ATTACK, "Prior: Affordable Spidops Attack"


            if active_id == ARTICUNO_ID:
                return BONUS_ARTICUNO_ATTACK, "Prior: Affordable Articuno Snipe Attack"
            return BONUS_GENERIC_ATTACK, "Prior: Affordable Active Attack"
        else:
            return BONUS_UNDERPOWERED_ATTACK, "Prior: Underpowered Active Attack"

    # 5. SEARCH TARGET SELECTION (OptionType.CARD / 3 / 4 / 12)
    elif opt_type in (getattr(OptionType, "CARD", 3), 3, 4, 12) and card_id > 0:
        target_score = score_search_target(card_id, ctx, opponent_name=opponent_name)
        bonus = (target_score - 0.50) * 0.35  # Maps score range [0.1..1.0] -> [-0.14..+0.175]
        return max(-0.20, min(0.20, bonus)), f"Prior: Intelligent Search Target Scoring ({target_score:.2f})"

    return 0.0, ""


# Backward-compatible API binding
get_expert_bonus = evaluate_mewtwo_expert_bonus

# ===== grimmsnarl_expert.py =====
"""
Marnie's Grimmsnarl ex + Munkidori Strategic Expert Engine.
Ground Truth: Matches 60-card CSV list (deckgrimnai.csv).
Acts purely as a lightweight action prior provider in range [-0.15, +0.20].
Centralized configuration parameters eliminate magic numbers and prevent MCTS over-dominance.
"""
from cg.api import OptionType

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

# -------------------------------------------------------------------------
# CENTRALIZED EXPERT PRIOR CONFIGURATION (NO MAGIC NUMBERS)
# -------------------------------------------------------------------------
EXPERT_PRIOR_CONFIG = {
    "POFFIN_MULTI_BASIC": 0.18,
    "POFFIN_SINGLE_BASIC": 0.12,
    "POFFIN_LOW_VAL": 0.04,
    "POKEGEAR_EARLY": 0.15,
    "RARE_CANDY_READY": 0.20,
    "RARE_CANDY_SHORTCUT": 0.15,
    "RARE_CANDY_EARLY": 0.10,
    "SPIKEMUTH_HIGH_VAL": 0.18,
    "SPIKEMUTH_STANDARD": 0.10,
    "UNFAIR_STAMP_HUNT": 0.18,
    "UNFAIR_STAMP_DIG": 0.14,
    "UNFAIR_STAMP_STD": 0.08,
    "MUNKIDORI_1ST": 0.15,
    "MUNKIDORI_2ND": 0.14,
    "MUNKIDORI_3RD_CRITICAL": 0.20,
    "MUNKIDORI_4TH_LOW": 0.04,
    "IMPIDIMP_BASIC": 0.12,
    "EARLY_SUPPORTER_SETUP": 0.15,
    "SUPPORTER_REFRESH": 0.10,
    "BOSS_MUNKIDORI_WAR": 0.18,
    "BOSS_GUST_DISRUPTION": 0.14,
    "BOSS_GUST_SINGLE": 0.10,
    "POKEPAD_RECYCLE": 0.08,
    "NIGHT_STRETCHER_COMBO": 0.16,
    "NIGHT_STRETCHER_KEY_POKEMON": 0.12,
    "NIGHT_STRETCHER_ENERGY": 0.10,
    "ATTACH_ACTIVE_DARK": 0.12,
    "ATTACH_MUNKIDORI_DARK": 0.10,
    "STAGE2_GRIMMSNARL_COMEBACK": 0.20,
    "STAGE2_GRIMMSNARL_AHEAD": 0.12,
    "STAGE1_MORGREM": 0.10,
    "FROSLASS_EARLY": 0.14,
    "FROSLASS_STD": 0.10,
    "MUNKIDORI_ABILITY": 0.15,
    "ATTACK_HIGH_VALUE_KO": 0.20,
    "ATTACK_LOW_VALUE_KO": 0.14,
    "ATTACK_POSTPONE_SUPPORTER": 0.08,
    "ATTACK_EXECUTION": 0.15,
}

# Matchup archetype keywords
DRAGAPULT_KEYWORDS = ("dragapult", "drakloak", "dreepy")
CHARIZARD_KEYWORDS = ("charizard", "charmeleon", "charmander")
GARDEVOIR_KEYWORDS = ("gardevoir", "kirlia", "ralts")
CRUSTLE_KEYWORDS   = ("crustle", "dwebble", "abomasnow")


def evaluate_grimmsnarl_expert_bonus(obs, option, opponent_name: str) -> tuple[float, str]:
    """
    Evaluates context-aware heuristic action prior bonus for Marnie's Grimmsnarl ex deck.
    All bonus values are strictly bounded in [-0.15, +0.20] to prevent MCTS over-dominance.
    """
    ctx = extract_board_context(obs)
    my_active_id = ctx["my_active_id"]
    opp_active_id = ctx["opp_active_id"]
    my_bench_ids = ctx.get("my_bench_ids", [])
    opp_bench_ids = ctx.get("opp_bench_ids", [])
    my_bench_count = ctx["my_bench_count"]
    turn = ctx["turn"]

    my_prizes_taken = 6 - ctx.get("my_prizes", 6)
    opp_prizes_taken = 6 - ctx.get("opp_prizes", 6)

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
    opp_lower = (opponent_name or "").lower()

    # Matchup context flags
    is_dragapult_matchup = any(kw in opp_lower for kw in DRAGAPULT_KEYWORDS)
    is_charizard_matchup = any(kw in opp_lower for kw in CHARIZARD_KEYWORDS)
    is_gardevoir_matchup = any(kw in opp_lower for kw in GARDEVOIR_KEYWORDS)
    is_crustle_matchup   = any(kw in opp_lower for kw in CRUSTLE_KEYWORDS)

    # -------------------------------------------------------------------------
    # 1. CORE PLAY ACTIONS
    # -------------------------------------------------------------------------
    if IS_PLAY:
        if card_id == POFFIN_ID and turn <= 3:
            my_hand_ids = set(ctx.get("my_hand_ids", []))
            field_ids = set(my_bench_ids + [my_active_id] + ctx.get("my_discard_ids", []))
            targets = {IMPIDIMP_ID, MUNKIDORI_ID, SNORUNT_ID}
            missing_targets = [tid for tid in targets if tid not in field_ids and tid not in my_hand_ids]

            if len(missing_targets) >= 2:
                bonus += EXPERT_PRIOR_CONFIG["POFFIN_MULTI_BASIC"]
                triggered = "Poffin_Search_Multiple_Basics_High_Value"
            elif len(missing_targets) == 1:
                bonus += EXPERT_PRIOR_CONFIG["POFFIN_SINGLE_BASIC"]
                triggered = "Poffin_Search_Single_Basic"
            else:
                bonus += EXPERT_PRIOR_CONFIG["POFFIN_LOW_VAL"]
                triggered = "Poffin_Low_Value_Use"

        elif card_id == POKEGEAR_ID and turn <= 2:
            bonus += EXPERT_PRIOR_CONFIG["POKEGEAR_EARLY"]
            triggered = "PokeGear_Early_Consistency_Engine"

        elif card_id == RARE_CANDY_ID and IMPIDIMP_ID in my_bench_ids:
            my_hand_ids = ctx.get("my_hand_ids", [])
            has_dark_energy = BASIC_DARK_ENERGY_ID in my_hand_ids
            has_attack_enablement = has_dark_energy or (my_active_id == IMPIDIMP_ID and turn >= 2)
            if has_attack_enablement:
                bonus += EXPERT_PRIOR_CONFIG["RARE_CANDY_READY"]
                triggered = "Rare_Candy_Grimmsnarl_Ready_To_Attack"
            elif turn >= 3:
                bonus += EXPERT_PRIOR_CONFIG["RARE_CANDY_SHORTCUT"]
                triggered = "Rare_Candy_Grimmsnarl_Shortcut"
            else:
                bonus += EXPERT_PRIOR_CONFIG["RARE_CANDY_EARLY"]
                triggered = "Rare_Candy_Grimmsnarl_Early_Evolution"

        elif card_id == SPIKEMUTH_GYM_ID:
            opp_bench_count = len(opp_bench_ids)
            if opp_bench_count >= 2 or (turn > 3 and opp_bench_count > 0):
                bonus += EXPERT_PRIOR_CONFIG["SPIKEMUTH_HIGH_VAL"]
                triggered = "Spikemuth_Gym_High_Value_Target"
            else:
                bonus += EXPERT_PRIOR_CONFIG["SPIKEMUTH_STANDARD"]
                triggered = "Spikemuth_Gym_Standard_Play"

        elif card_id == UNFAIR_STAMP_ID:
            my_hand_ids = ctx.get("my_hand_ids", [])
            has_spikemuth = (SPIKEMUTH_GYM_ID in my_bench_ids or SPIKEMUTH_GYM_ID in my_hand_ids)
            needs_spikemuth = not has_spikemuth
            needs_night_stretcher = NIGHT_STRETCHER_ID not in my_hand_ids
            needs_supporter = not any(cid in SUPPORTER_IDS for cid in my_hand_ids)

            if is_gardevoir_matchup and turn <= 3:
                bonus += EXPERT_PRIOR_CONFIG["UNFAIR_STAMP_HUNT"]
                triggered = "Unfair_Stamp_Gardevoir_Kirlia_Disruption"
            elif needs_spikemuth and turn <= 3:
                bonus += EXPERT_PRIOR_CONFIG["UNFAIR_STAMP_HUNT"]
                triggered = "Unfair_Stamp_Spikemuth_Gym_Hunt"
            elif needs_night_stretcher or needs_supporter:
                bonus += EXPERT_PRIOR_CONFIG["UNFAIR_STAMP_DIG"]
                triggered = "Unfair_Stamp_Resource_Dig"
            else:
                bonus += EXPERT_PRIOR_CONFIG["UNFAIR_STAMP_STD"]
                triggered = "Unfair_Stamp_Standard_Play"

        elif card_id == MUNKIDORI_ID and my_bench_count < 5:
            munkidori_count = my_bench_ids.count(MUNKIDORI_ID)
            if munkidori_count == 0:
                bonus += EXPERT_PRIOR_CONFIG["MUNKIDORI_1ST"]
                triggered = "Munkidori_First_Bench_Priority"
            elif munkidori_count == 1:
                bonus += EXPERT_PRIOR_CONFIG["MUNKIDORI_2ND"]
                triggered = "Munkidori_Second_Bench_Priority"
            elif munkidori_count == 2:
                bonus += EXPERT_PRIOR_CONFIG["MUNKIDORI_3RD_CRITICAL"]
                triggered = "Munkidori_Third_Bench_Critical"
            else:
                bonus += EXPERT_PRIOR_CONFIG["MUNKIDORI_4TH_LOW"]
                triggered = "Munkidori_Fourth_Bench_Low_Priority"

        elif card_id == IMPIDIMP_ID and my_bench_count < 5:
            bonus += EXPERT_PRIOR_CONFIG["IMPIDIMP_BASIC"]
            triggered = "Grimmsnarl_Bench_Basic_Impidimp"

        elif card_id in (PETREL_ID, LILLIES_DETERM_ID):
            my_hand_ids = ctx.get("my_hand_ids", [])
            if turn <= 2 and (SPIKEMUTH_GYM_ID not in my_bench_ids and RARE_CANDY_ID not in my_hand_ids):
                bonus += EXPERT_PRIOR_CONFIG["EARLY_SUPPORTER_SETUP"]
                triggered = "Early_Supporter_Setup_Priority"
            else:
                bonus += EXPERT_PRIOR_CONFIG["SUPPORTER_REFRESH"]
                triggered = "Grimmsnarl_Supporter_Draw_Refresh"

        elif card_id == BOSS_ORDERS_ID and len(opp_bench_ids) > 0:
            opp_munkidori_count = sum(1 for pid in opp_bench_ids if pid == MUNKIDORI_ID)
            if opp_munkidori_count > 0:
                bonus += EXPERT_PRIOR_CONFIG["BOSS_MUNKIDORI_WAR"]
                triggered = "Boss_Gust_Munkidori_War_Priority"
            elif is_charizard_matchup:
                bonus += EXPERT_PRIOR_CONFIG["BOSS_MUNKIDORI_WAR"]
                triggered = "Boss_Gust_Charizard_Pre_Evo_Target"
            elif len(opp_bench_ids) >= 2:
                bonus += EXPERT_PRIOR_CONFIG["BOSS_GUST_DISRUPTION"]
                triggered = "Grimmsnarl_Boss_Gust_Disruption"
            else:
                bonus += EXPERT_PRIOR_CONFIG["BOSS_GUST_SINGLE"]
                triggered = "Boss_Gust_Single_Target"

        elif card_id == POKE_PAD_ID:
            bonus += EXPERT_PRIOR_CONFIG["POKEPAD_RECYCLE"]
            triggered = "Grimmsnarl_PokePad_Recycle_Supporter"

        elif card_id == NIGHT_STRETCHER_ID:
            my_discard_ids = ctx.get("my_discard_ids", [])
            has_key_bench_in_discard = any(cid in (MUNKIDORI_ID, FROSLASS_ID, IMPIDIMP_ID) for cid in my_discard_ids)
            has_dark_energy_in_discard = BASIC_DARK_ENERGY_ID in my_discard_ids

            if has_key_bench_in_discard and has_dark_energy_in_discard:
                bonus += EXPERT_PRIOR_CONFIG["NIGHT_STRETCHER_COMBO"]
                triggered = "Night_Stretcher_Recovery_Pokemon_And_Energy"
            elif has_key_bench_in_discard:
                bonus += EXPERT_PRIOR_CONFIG["NIGHT_STRETCHER_KEY_POKEMON"]
                triggered = "Night_Stretcher_Recovery_Key_Pokemon"
            elif has_dark_energy_in_discard:
                bonus += EXPERT_PRIOR_CONFIG["NIGHT_STRETCHER_ENERGY"]
                triggered = "Night_Stretcher_Recovery_Energy"
            else:
                bonus += EXPERT_PRIOR_CONFIG["POFFIN_LOW_VAL"]
                triggered = "Night_Stretcher_Standard_Recovery"

    # -------------------------------------------------------------------------
    # 2. ENERGY ATTACHMENT
    # -------------------------------------------------------------------------
    elif IS_ATTACH:
        if card_id == BASIC_DARK_ENERGY_ID:
            if my_active_id in (IMPIDIMP_ID, MORGREM_ID, GRIMMSNARL_EX_ID):
                bonus += EXPERT_PRIOR_CONFIG["ATTACH_ACTIVE_DARK"]
                triggered = "Grimmsnarl_Dark_Energy_Attach_Active"
            elif MUNKIDORI_ID in my_bench_ids:
                bonus += EXPERT_PRIOR_CONFIG["ATTACH_MUNKIDORI_DARK"]
                triggered = "Grimmsnarl_Dark_Energy_Attach_Munkidori"

    # -------------------------------------------------------------------------
    # 3. EVOLUTION
    # -------------------------------------------------------------------------
    elif IS_EVOLVE:
        if card_id == GRIMMSNARL_EX_ID:
            if my_prizes_taken <= opp_prizes_taken:
                bonus += EXPERT_PRIOR_CONFIG["STAGE2_GRIMMSNARL_COMEBACK"]
                triggered = "Grimmsnarl_ex_Evolution_Comeback_Setup"
            else:
                bonus += EXPERT_PRIOR_CONFIG["STAGE2_GRIMMSNARL_AHEAD"]
                triggered = "Grimmsnarl_ex_Evolution_While_Ahead"
        elif card_id == MORGREM_ID:
            bonus += EXPERT_PRIOR_CONFIG["STAGE1_MORGREM"]
            triggered = "Morgrem_Stage1_Evolution"
        elif card_id == FROSLASS_ID:
            if turn <= 3 or is_dragapult_matchup:
                bonus += EXPERT_PRIOR_CONFIG["FROSLASS_EARLY"]
                triggered = "Froslass_Early_Evolution_Priority"
            else:
                bonus += EXPERT_PRIOR_CONFIG["FROSLASS_STD"]
                triggered = "Froslass_Stage1_Evolution"

    # -------------------------------------------------------------------------
    # 4. ABILITY & ATTACK EXECUTION
    # -------------------------------------------------------------------------
    elif IS_ABILITY:
        bonus += EXPERT_PRIOR_CONFIG["MUNKIDORI_ABILITY"]
        if is_crustle_matchup:
            bonus += 0.05
            triggered = "Munkidori_Adrenaline_Brain_Crustle_Bypass"
        else:
            triggered = "Munkidori_Adrenaline_Brain_Move_Damage"

    elif IS_ATTACK:
        my_hand_ids = ctx.get("my_hand_ids", [])
        has_unplayed_supporter = any(cid in SUPPORTER_IDS for cid in my_hand_ids)
        has_unplayed_poffin = POFFIN_ID in my_hand_ids
        can_ko = ctx.get("can_ko_active", False)
        opp_active_is_ex = ctx.get("opp_active_is_ex", False)

        if can_ko:
            if opp_active_is_ex or my_prizes_taken <= opp_prizes_taken:
                bonus += EXPERT_PRIOR_CONFIG["ATTACK_HIGH_VALUE_KO"]
                triggered = "Grimmsnarl_Attack_For_High_Value_KO"
            else:
                bonus += EXPERT_PRIOR_CONFIG["ATTACK_LOW_VALUE_KO"]
                triggered = "Grimmsnarl_Attack_For_Low_Value_KO"
        elif has_unplayed_supporter or has_unplayed_poffin:
            bonus += EXPERT_PRIOR_CONFIG["ATTACK_POSTPONE_SUPPORTER"]
            triggered = "Grimmsnarl_Attack_Postpone_For_Supporter"
        else:
            bonus += EXPERT_PRIOR_CONFIG["ATTACK_EXECUTION"]
            triggered = "Grimmsnarl_Attack_Execution"

    # Bound bonus range strictly in [-0.15, +0.20]
    bonus = max(-0.15, min(0.20, bonus))
    return bonus, triggered

# ===== lillie_expert.py =====
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
        # A. Avoid premature 2-prize ex (Clefairy ex, Latias ex) exposure early against fast physical KOs
        if IS_PLAY and card_id in (CLEFAIRY_EX_ID, LATIAS_EX_ID):
            if turn <= 5 or my_active_energy == 0:
                bonus -= 0.35
                triggered = "Lucario_Matchup_Avoid_Premature_2Prize_ex_Risk"

        # B. Prioritize 1-prize basic buffer Pokemon (Togepi, Smoochum, Mimikyu, Psyduck)
        if IS_PLAY and card_id in (TOGEPI_ID, SMOOCHUM_ID, MIMIKYU_ID, PSYDUCK_ID, SHAYMIN_ID):
            bonus += 0.35
            triggered = "Lucario_Matchup_Single_Prize_Buffer"

        # C. Priority Togekiss setup to out-scale Lucario early
        if IS_EVOLVE and card_id == TOGEKISS_ID:
            bonus += 0.45
            triggered = "Lucario_Matchup_Fast_Togekiss_Evolution"

        # D. Target Riolu (447) before evolving to Lucario
        if IS_PLAY and card_id in (BOSS_ORDERS_ID, POKEMON_CATCHER_ID):
            if 447 in opp_bench_ids:
                bonus += 0.35
                triggered = "Lucario_Matchup_Gust_Riolu_Target"

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

# ===== expert_router.py =====
"""
Expert System Router for Pokémon TCG RL Agent.
Dynamically evaluates action prior bonuses based on ACTIVE_DECK and applies
training epoch confidence decay so neural network policies take over cleanly.
"""


def get_expert_bonus(obs, option, opponent_name: str = "", epoch: int = 0) -> tuple[float, str]:
    """
    Unified entry point for retrieving expert action prior bonus and trigger label.
    Includes epoch-based confidence decay so heuristics smoothly yield to NN policy.
    Returns: (weighted_bonus: float, trigger_name: str)
    """
    if not USE_EXPERT_GUIDANCE:
        return 0.0, "Disabled"

    # Confidence decay over 30 training epochs (from 1.0 down to 0.1)
    confidence = max(0.1, 1.0 - (epoch / 30.0))

    deck = get_active_deck_name()
    if deck == "GRIMMSNARL":
        bonus, triggered = evaluate_grimmsnarl_expert_bonus(obs, option, opponent_name)
    elif deck == "LILLIE":
        bonus, triggered = evaluate_lillie_expert_bonus(obs, option, opponent_name)
    else:  # Default MEWTWO
        bonus, triggered = evaluate_mewtwo_expert_bonus(obs, option, opponent_name)

    return bonus * EXPERT_WEIGHT * confidence, triggered
