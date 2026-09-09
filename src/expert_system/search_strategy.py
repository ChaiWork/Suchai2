"""
State-Aware Strategic Search Target Evaluator (AlphaZero Paradigm).

Extracts board state features and computes dynamic, state-dependent prior bonuses [-0.10, +0.20]
for candidate search target cards (e.g. via Ultra Ball, Poffin, Transceiver, Bug Catching Set).

Card values dynamically depend on the game state (active Pokemon, energy needs, bench space,
evolution readiness, prize situation, and opponent threat level) instead of static rankings.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple


# Ground Truth Card IDs for Team Rocket Mewtwo ex Deck
BASIC_G_ENERGY_ID    = 1    # Basic {G} Energy
TR_ENERGY_ID         = 15   # Team Rocket's Energy
TAROUNTULA_ID        = 400  # Team Rocket's Tarountula
SPIDOPS_ID           = 401  # Team Rocket's Spidops
ARTICUNO_ID          = 414  # Team Rocket's Articuno
MEWTWO_EX_ID         = 431  # Team Rocket's Mewtwo ex
MIMIKYU_ID           = 434  # Team Rocket's Mimikyu

POFFIN_ID            = 1086 # Buddy-Buddy Poffin
BUG_CATCHING_SET_ID  = 1094 # Bug Catching Set
ULTRA_BALL_ID        = 1121 # Ultra Ball
TRANSCEIVER_ID       = 1134 # Team Rocket's Transceiver

ARIANA_ID            = 1216 # Team Rocket's Ariana
GIOVANNI_ID          = 1218 # Team Rocket's Giovanni
PETREL_ID            = 1219 # Team Rocket's Petrel
PROTON_ID            = 1220 # Team Rocket's Proton
LILLIES_DETERM_ID    = 1227 # Lillie's Determination
BATTLE_CAGE_ID       = 1264 # Battle Cage (Anti-Dragapult Bench Protection)

TR_POKEMON_IDS       = frozenset({400, 401, 414, 431, 434})
SUPPORTER_IDS        = frozenset({1216, 1218, 1219, 1220, 1227})


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


def extract_search_context(obs: dict) -> dict:
    """Extracts complete state-aware board context features for search target valuation."""
    ctx = {
        "turn": 1,
        "my_prizes": 6,
        "opp_prizes": 6,
        "my_active_id": -1,
        "my_active_hp": 100,
        "my_active_energy": 0,
        "opp_active_id": -1,
        "opp_active_hp": 100,
        "opp_active_energy": 0,
        "my_bench_ids": [],
        "opp_bench_ids": [],
        "my_hand_ids": [],
        "my_hand_len": 0,
        "my_bench_count": 0,
        "opp_bench_count": 0,
        "my_discard_ids": [],
        "tr_count": 0,
        "has_mewtwo_in_play": False,
        "has_mewtwo_in_hand": False,
        "has_spidops_in_play": False,
        "has_tarountula_in_play": False,
        "has_supporter_in_hand": False,
        "active_attack_ready": False,
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
            act_card = my_active_list[0]
            ctx["my_active_id"] = _get_val(act_card, "cardId", -1)
            ctx["my_active_hp"] = _get_val(act_card, "hp", 100)
            energies = _get_val(act_card, "energyCards", [])
            ctx["my_active_energy"] = len(energies) if isinstance(energies, (list, tuple)) else 0

        opp_active_list = _get_val(opp_ps, "active", [])
        if opp_active_list and len(opp_active_list) > 0 and opp_active_list[0] is not None:
            act_card = opp_active_list[0]
            ctx["opp_active_id"] = _get_val(act_card, "cardId", -1)
            ctx["opp_active_hp"] = _get_val(act_card, "hp", 100)
            energies = _get_val(act_card, "energyCards", [])
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

        my_discard = _get_val(my_ps, "discard", [])
        if isinstance(my_discard, (list, tuple)):
            ctx["my_discard_ids"] = [_get_val(d, "cardId", -1) for d in my_discard if d is not None]

    # Derived state indicators
    in_play = [ctx["my_active_id"]] + ctx["my_bench_ids"]
    ctx["tr_count"] = sum(1 for cid in in_play if cid in TR_POKEMON_IDS)
    ctx["has_mewtwo_in_play"] = (MEWTWO_EX_ID in in_play)
    ctx["has_mewtwo_in_hand"] = (MEWTWO_EX_ID in ctx["my_hand_ids"])
    ctx["has_spidops_in_play"] = (SPIDOPS_ID in in_play)
    ctx["has_tarountula_in_play"] = (TAROUNTULA_ID in in_play)
    ctx["has_supporter_in_hand"] = any(cid in SUPPORTER_IDS for cid in ctx["my_hand_ids"])

    try:
        from src.expert_system.energy_evaluator import get_required_energy
    except ImportError:
        from expert_system.energy_evaluator import get_required_energy
    # Use API-driven energy cost lookup instead of hardcoded per-card values
    req_e = get_required_energy(ctx["my_active_id"])
    ctx["active_attack_ready"] = (ctx["my_active_energy"] >= req_e)

    return ctx


def get_search_features(card_id: int, ctx: dict) -> Dict[str, float]:
    """Extracts normalized vector features for a candidate search target card."""
    return {
        "is_basic": 1.0 if card_id in (400, 414, 431, 434) else 0.0,
        "is_evolution": 1.0 if card_id in (401,) else 0.0,
        "is_supporter": 1.0 if card_id in SUPPORTER_IDS else 0.0,
        "is_item": 1.0 if card_id in (1086, 1094, 1095, 1097, 1106, 1121, 1134, 1152, 1158, 1159, 1175) else 0.0,
        "is_stadium": 1.0 if card_id in (1180, 1257, 1264) else 0.0,
        "is_energy": 1.0 if card_id in (BASIC_G_ENERGY_ID, TR_ENERGY_ID) else 0.0,
        "already_in_hand_count": float(ctx["my_hand_ids"].count(card_id)),
        "already_in_bench_count": float(ctx["my_bench_ids"].count(card_id)),
    }


def evaluate_search_target_prior(card_id: int, ctx: dict) -> Tuple[float, str]:
    """
    Evaluates state-dependent strategic prior bonus [-0.10, +0.20] for a candidate search target.
    
    Scenarios handled dynamically:
      1. Early Game / Missing Main Win Condition (Mewtwo ex): +0.20 when absent.
      2. Missing Evolution Line / Setup (Tarountula / Spidops): +0.16 to +0.18 when line incomplete.
      3. Power Saver Ability (<4 TR Pokemon in play): +0.16 for any TR Pokemon.
      4. Active Energy Needs: +0.15 for Energy when active Mewtwo ex is building.
      5. Lethal Window / Prize Swing: +0.20 for Giovanni gust when opponent active HP <= 160.
      6. Low Hand Emergency: +0.18 for Ariana / Lillie when hand_len <= 3.
    """
    if card_id <= 0:
        return 0.0, "Invalid card"

    turn = ctx.get("turn", 1)
    my_prizes = ctx.get("my_prizes", 6)
    opp_prizes = ctx.get("opp_prizes", 6)
    hand_len = ctx.get("my_hand_len", 0)
    tr_count = ctx.get("tr_count", 0)
    active_id = ctx.get("my_active_id", -1)
    active_e = ctx.get("my_active_energy", 0)
    opp_hp = ctx.get("opp_active_hp", 9999)
    close_game = (my_prizes <= 2 or opp_prizes <= 2)

    # -------------------------------------------------------------------------
    # A. MEWTWO EX (Main Win Condition)
    # -------------------------------------------------------------------------
    if card_id == MEWTWO_EX_ID:
        if not ctx.get("has_mewtwo_in_play", False) and not ctx.get("has_mewtwo_in_hand", False):
            return 0.20, "Search Priority: Deploy Main Attacker Mewtwo ex"
        elif ctx.get("has_mewtwo_in_play", False) and not ctx.get("has_mewtwo_in_hand", False) and (turn <= 3 or my_prizes >= 5):
            return 0.12, "Search Priority: Second Mewtwo ex Backup Attacker Setup"
        return 0.05, "Search: Additional Mewtwo ex"

    # -------------------------------------------------------------------------
    # B. SPIDOPS LINE (Evolution & Territory Control)
    # -------------------------------------------------------------------------
    elif card_id == SPIDOPS_ID:
        opp_act_id = ctx.get("opp_active_id", -1)
        opp_name = ctx.get("opponent_name", "")
        is_crustle = (opp_act_id in (344, 345, 407, 408)) or any(k in opp_name.lower() for k in ("crustle", "dwebble"))
        if is_crustle:
            return 0.55, "CRITICAL Search Priority: Fetch Spidops vs Crustle (Single-Prize EX Immunity Hard Counter!)"
        is_darkness = (opp_act_id in (646, 647, 648, 112, 104)) or any(k in opp_name.lower() for k in ("grimmsnarl", "marnie", "darkness"))
        if is_darkness:
            return 0.50, "CRITICAL Search Priority: Fetch Spidops vs Darkness EX (2x Grass Weakness OHKO!)"
        if ctx.get("has_tarountula_in_play", False) and not ctx.get("has_spidops_in_play", False):
            return 0.25, "Search Priority: Spidops Evolution Ready Target"
        elif not ctx.get("has_spidops_in_play", False):
            return 0.18, "Search Priority: Spidops Line Setup Target"
        return 0.05, "Search: Additional Spidops"

    elif card_id == TAROUNTULA_ID:
        opp_act_id = ctx.get("opp_active_id", -1)
        opp_name = ctx.get("opponent_name", "")
        is_crustle = (opp_act_id in (344, 345, 407, 408)) or any(k in opp_name.lower() for k in ("crustle", "dwebble"))
        if is_crustle and not ctx.get("has_tarountula_in_play", False):
            return 0.40, "CRITICAL Search Priority: Fetch Tarountula vs Crustle Base Line"
        if not ctx.get("has_tarountula_in_play", False) and not ctx.get("has_spidops_in_play", False):
            return 0.16, "Search Priority: Deploy Tarountula Base Line"
        elif ctx.get("my_bench_count", 0) <= 1:
            return 0.14, "Search Priority: Bench Fill Tarountula Guard"
        return 0.06, "Search: Additional Tarountula"

    # -------------------------------------------------------------------------
    # C. SUPPORTERS (Giovanni, Ariana, Lillie)
    # -------------------------------------------------------------------------
    elif card_id == GIOVANNI_ID:
        if opp_hp <= 160 or close_game:
            return 0.20, "Search Priority: Giovanni Gust Lethal Window / Prize Swing"
        elif ctx.get("opp_bench_count", 0) > 0:
            return 0.12, "Search: Giovanni Bench Gust Target"
        return 0.05, "Search: Giovanni Supporter"

    elif card_id == ARIANA_ID:
        if hand_len <= 3:
            return 0.18, "Search Priority: Ariana Draw Refresh (Low Hand)"
        return 0.08, "Search: Ariana Supporter"

    elif card_id == LILLIES_DETERM_ID:
        if hand_len <= 3:
            return 0.18, "Search Priority: Lillie Emergency Draw (Low Hand)"
        return 0.06, "Search: Lillie Supporter"

    elif card_id in (PETREL_ID, PROTON_ID):
        return 0.12, "Search: Tactical TR Supporter"

    # -------------------------------------------------------------------------
    # D. ENERGY (Basic {G} or TR Energy)
    # -------------------------------------------------------------------------
    elif card_id in (BASIC_G_ENERGY_ID, TR_ENERGY_ID):
        my_hand_ids = ctx.get("my_hand_ids", [])
        if active_id == MEWTWO_EX_ID and active_e < 3:
            return 0.15, "Search Priority: Energy for Active Mewtwo ex (Psywave Build)"
        elif active_id == SPIDOPS_ID and active_e < 2:
            return 0.12, "Search Priority: Energy for Active Spidops"
        elif not ctx.get("active_attack_ready", False) and my_hand_ids.count(card_id) == 0:
            return 0.10, "Search: Energy for Attack Readiness"
        return 0.02, "Search: Energy Candidate"

    # -------------------------------------------------------------------------
    # E. UTILITY POKEMON (Articuno, Mimikyu)
    # -------------------------------------------------------------------------
    elif card_id == ARTICUNO_ID:
        if turn <= 3 or ctx.get("my_bench_count", 0) <= 1:
            return 0.12, "Search: Articuno Pivot Utility"
        return 0.06, "Search: Articuno Utility"

    elif card_id == MIMIKYU_ID:
        if ctx.get("my_active_hp", 100) <= 40:
            return 0.14, "Search: Mimikyu Stall Pivot Guard"
        return 0.06, "Search: Mimikyu Stall Candidate"

    elif card_id == BATTLE_CAGE_ID:
        # Generic state-transition threat evaluation: prevent expected bench damage/snipe
        opp_name = ctx.get("opponent_name", "")
        opp_act_id = ctx.get("opp_active_id", -1)
        my_bench_cnt = ctx.get("my_bench_count", 0)
        # Check if opponent has bench-damaging capability (e.g. Dragapult ID 121, Alakazam ID 150) or bench is populated
        _BENCH_SNIPER_IDS = frozenset({121, 150, 414})
        has_bench_threat = (opp_act_id in _BENCH_SNIPER_IDS) or ("Dragapult" in opp_name or "Alakazam" in opp_name) or (my_bench_cnt >= 2)
        if has_bench_threat:
            return 0.20, "Search Priority: Battle Cage Expected Bench Damage Protection"
        return 0.08, "Search: Stadium Battle Cage"

    # -------------------------------------------------------------------------
    # F. POWER SAVER TR POKEMON UNLOCK GUARD (<4 TR Pokemon in play)
    # -------------------------------------------------------------------------
    elif card_id in TR_POKEMON_IDS and tr_count < 4:
        return 0.16, "Search Priority: TR Pokemon to Unlock Power Saver (<4 in play)"

    return 0.0, "Search: Generic Candidate"


def score_search_target(card_id: int, ctx: dict, search_card_id: int = -1, opponent_name: str = "") -> float:
    """Computes state-aware strategic prior score for candidate target card_id."""
    if opponent_name and "opponent_name" not in ctx:
        ctx["opponent_name"] = opponent_name
    prior_bonus, _ = evaluate_search_target_prior(card_id, ctx)
    # Map prior bonus [-0.10, +0.20] to logit score space
    return prior_bonus


def choose_best_search_target(
    obs: dict,
    candidate_card_ids: List[int],
    search_card_id: int = -1,
    opponent_name: str = ""
) -> Tuple[int, float]:
    """
    Evaluates all candidate search target card IDs against current state context
    and returns the best candidate ID and its strategic prior score.
    """
    if not candidate_card_ids:
        return -1, 0.0

    ctx = extract_search_context(obs)
    if opponent_name:
        ctx["opponent_name"] = opponent_name
        
    best_cid = candidate_card_ids[0]
    best_score = -1.0
    best_reason = "Default"

    for cid in candidate_card_ids:
        score, reason = evaluate_search_target_prior(cid, ctx)
        if score > best_score:
            best_score = score
            best_cid = cid
            best_reason = reason

    return best_cid, best_score


def format_search_decision_log(obs: dict, candidate_card_ids: List[int]) -> str:
    """Formats human-readable diagnostic log for a search decision."""
    ctx = extract_search_context(obs)
    lines = [
        "Search Decision:",
        f"  Turn: {ctx.get('turn', 1)}",
        f"  Current board: Active={ctx.get('my_active_id')} (HP {ctx.get('my_active_hp')}, Energy {ctx.get('my_active_energy')}), Bench={ctx.get('my_bench_ids')}",
        "  Candidate cards:"
    ]
    best_cid = candidate_card_ids[0] if candidate_card_ids else -1
    best_score = -1.0
    for cid in candidate_card_ids:
        score, reason = evaluate_search_target_prior(cid, ctx)
        lines.append(f"    Card ID {cid:4d} | Prior: {score:+.2f} | Reason: \"{reason}\"")
        if score > best_score:
            best_score = score
            best_cid = cid

    lines.append(f"  Selected: Card ID {best_cid}")
    return "\n".join(lines)


