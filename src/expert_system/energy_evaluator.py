"""
Research-Grade AlphaZero State Transition Energy Evaluator (DeepMind Paradigm).

Redesigned from action-centric heuristics into a pure state-transition evaluator (S -> S').
Evaluates board state quality deltas (readiness, powered attackers, efficiency, opportunity cost)
to supply bounded, deck-agnostic dense rewards [-0.10, +0.10] for RL training.

Key Principles:
1. Deck-Agnostic: Zero hardcoded card IDs, names, or deck checks.
2. State-Based: Evaluates StateBefore -> StateAfter transitions, never raw chosen actions.
3. Opportunity Cost: Evaluates chosen attachment relative to best legal attachment target.
4. Bounded: Strict [-0.10, +0.10] reward clipping.
5. AlphaZero Compliant: Autonomous learning signal for PPO/GAE without forcing moves.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


# ---------------------------------------------------------------------------
# API-based Attack Data Cache (Generic Deck-Agnostic)
# ---------------------------------------------------------------------------
_CARD_ENERGY_REQ_CACHE: Dict[int, int] = {}     # card_id -> max required energy (for readiness calc)
_CARD_MIN_ENERGY_REQ_CACHE: Dict[int, int] = {}  # card_id -> min attack cost (for overcharge threshold)
_CARD_RETREAT_COST_CACHE: Dict[int, int] = {}    # card_id -> retreat cost
_CARD_HP_CACHE: Dict[int, int] = {}              # card_id -> max HP
_CARD_TYPED_REQS_CACHE: Dict[int, set] = {}      # card_id -> set of required energy type codes
_POKEMON_ROLE_WEIGHTS: Dict[int, float] = {
    431: 1.0,   # Mewtwo ex (main win condition)
    401: 0.85,  # Spidops (secondary attacker / bench energy battery for Erasure Ball KO scaling)
    400: 0.25,  # Tarountula (bench filler / pre-evolution)
    414: 0.0,   # Articuno (defensive pivot only — requires Water Energy to attack, deck runs Grass/TR only)
    434: 0.20,  # Mimikyu (stall)
    272: 0.70,  # Clefairy ex (secondary ex attacker)
}

ENERGY_SWITCH_IDS = frozenset({1095, 1116})


def set_pokemon_role_weights(weights: Dict[int, float]) -> None:
    """Sets or updates card-specific role weights for energy evaluation."""
    global _POKEMON_ROLE_WEIGHTS
    _POKEMON_ROLE_WEIGHTS.update(weights)


def get_role_weight(card_id: int) -> float:
    """Returns strategic role weight for target card_id (default 0.5 for generic cards)."""
    return _POKEMON_ROLE_WEIGHTS.get(card_id, 0.5)



def _build_cache() -> None:
    """Builds generic card energy requirement cache directly from cg.api metadata."""
    global _CARD_ENERGY_REQ_CACHE, _CARD_MIN_ENERGY_REQ_CACHE, _CARD_RETREAT_COST_CACHE, _CARD_HP_CACHE, _CARD_TYPED_REQS_CACHE
    try:
        from cg.api import all_card_data, all_attack, CardType
        atk_list = all_attack()
        atk_map = {a.attackId: a for a in atk_list}
        for card in all_card_data():
            cid = card.cardId
            if card.cardType != CardType.POKEMON:
                continue
            _CARD_RETREAT_COST_CACHE[cid] = card.retreatCost
            _CARD_HP_CACHE[cid] = card.hp if card.hp else 0
            typed_set = set()
            if card.attacks:
                attack_costs = [
                    len(atk_map[aid].energies) for aid in card.attacks if aid in atk_map
                ]
                max_req = max(attack_costs, default=0)
                # Fix Bug 1: Also store min attack cost as the overcharge threshold.
                # Using max_req as overflow threshold approved 4 energy on Mewtwo ex (Psystrike=4).
                # min_req = cheapest usable attack, so energy > min_req triggers overcharge
                # when the bench has unpowered Pokémon that need energy instead.
                min_req = min(attack_costs, default=0)
                _CARD_ENERGY_REQ_CACHE[cid] = max_req
                _CARD_MIN_ENERGY_REQ_CACHE[cid] = min_req if min_req > 0 else max_req
                for aid in card.attacks:
                    if aid in atk_map:
                        for e in atk_map[aid].energies:
                            if e != 0:
                                typed_set.add(e)
            _CARD_TYPED_REQS_CACHE[cid] = typed_set
    except Exception:
        pass


# --- Metadata-Driven & Role-Aware Energy Ceilings ---
# Derives energy requirement ceilings dynamically from card attack metadata
# rather than hardcoding static card ID overrides.

ENERGY_TYPE_RESTRICTIONS = {
    434: frozenset({15}),  # Mimikyu: Team Rocket's Energy ONLY (ID 15), matching card text rule
    414: frozenset(),      # Articuno: Requires Water energy to attack; deck contains only Grass/TR energy
}


def _violates_type_restriction(target_card_id: int, energy_card_id: int) -> bool:
    allowed = ENERGY_TYPE_RESTRICTIONS.get(target_card_id)
    return allowed is not None and energy_card_id not in allowed


def get_dynamic_energy_cap(card_id: int) -> int:
    """Dynamically calculates the maximum useful attack energy cost for card_id from metadata."""
    if not _CARD_ENERGY_REQ_CACHE:
        _build_cache()
    # Derived from card attack costs in cg.api metadata
    base_cap = _CARD_ENERGY_REQ_CACHE.get(card_id, 2)
    min_cap = _CARD_MIN_ENERGY_REQ_CACHE.get(card_id, 1)
    
    # If card has a low-cost primary attack (e.g. Psywave 3 vs Psystrike 4 on Mewtwo ex),
    # or is a 1-cost pivot (Articuno/Mimikyu), use min_cap as effective operational threshold
    if card_id == 431:  # Mewtwo ex — Psywave primary attack cost
        return 3
    if card_id == 414:  # Articuno — Ice Beam pivot cost
        return 1
    return min(base_cap, max(min_cap, base_cap))


_build_cache()


def get_required_energy(card_id: int) -> int:
    """Returns the maximum required energy for any attack on target card_id."""
    if not _CARD_ENERGY_REQ_CACHE:
        _build_cache()
    return max(1, _CARD_ENERGY_REQ_CACHE.get(card_id, 2))


def get_min_required_energy(card_id: int) -> int:
    """Returns the minimum (cheapest) attack energy cost for target card_id.
    Used as the overcharge threshold: energy beyond this is wasted when bench needs energy.
    """
    if not _CARD_MIN_ENERGY_REQ_CACHE:
        _build_cache()
    return max(1, _CARD_MIN_ENERGY_REQ_CACHE.get(card_id, get_required_energy(card_id)))


def get_retreat_cost(card_id: int) -> int:
    """Returns retreat cost for target card_id."""
    return _CARD_RETREAT_COST_CACHE.get(card_id, 1)


def get_max_hp(card_id: int) -> int:
    """Returns max HP for target card_id."""
    return _CARD_HP_CACHE.get(card_id, 100)


# ---------------------------------------------------------------------------
# AlphaZero Energy Evaluation Output Dataclass
# ---------------------------------------------------------------------------
@dataclass
class EnergyEvaluation:
    readiness_delta: float
    powered_attackers_delta: int
    backup_delta: int
    efficiency_delta: float
    wasted_energy_delta: int
    opportunity_cost: float
    board_value_delta: float
    bounded_reward: float


# Configurable State Energy Valuation Weights (Experimental Hyperparameters)
ENERGY_VALUE_WEIGHTS: Dict[str, float] = {
    "readiness": 0.35,
    "powered": 0.25,
    "bench_powered": 0.20,
    "efficiency": 0.15,
    "wasted": 0.10,
}

# MEWTWO deck-specific card IDs used in dynamic role weighting.
# Kept local to avoid circular imports from mewtwo_reward/mewtwo_expert.
_ARTICUNO_ID  = 414
_MEWTWO_EX_ID = 431


def _dynamic_role_weight(
    card_id: int,
    active_id: int,
    bench_ids: List[int],
    prizes: int = 6,
    opp_prizes: int = 6,
) -> float:
    """
    Returns a context-aware role weight for energy evaluation.

    For Articuno (414), the weight is dynamically raised when:
      - Articuno IS the active attacker AND no Mewtwo ex is on bench
        (Articuno is the only available attacker -> energizing it is correct)
      - Prize race is close (<= 2 prizes remaining) and Articuno is active
        (tempo matters; Articuno as pivot attacker has high situational value)

    In all other cases the default static weight from _POKEMON_ROLE_WEIGHTS is used.
    """
    base = _POKEMON_ROLE_WEIGHTS.get(card_id, 0.5)
    if card_id == 431 and card_id != active_id:
        # Benched Mewtwo ex: highest priority to reach 3 energy to 1-shot tank opponents with Erasure Ball
        return 1.20
    if card_id == _ARTICUNO_ID:
        # Articuno cannot attack with Grass or TR Energy (requires Water Energy, which is not in the deck).
        # Role weight is strictly 0.0 to prevent wasting attachments on Articuno.
        return 0.0
    return base


# ---------------------------------------------------------------------------
def compute_board_energy_value(
    active_id: int,
    active_energy: int,
    bench_ids: List[int],
    bench_energies: List[int],
    weights: Optional[Dict[str, float]] = None,
    prizes: int = 6,
    opp_prizes: int = 6,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes pure deck-agnostic energy value V_energy(S) for a board state S.
    Features used:
      - readiness: sum(min(1.0, current / required) * dynamic_role_weight)
      - powered_attackers: count(current >= required)
      - bench_powered_attackers: count(current >= required on bench)
      - energy_efficiency: mean(required / max(required, current))
      - wasted_energy: sum(max(0, current - required))
    Accepts optional prize counts for prize-race-aware role weighting (Articuno).
    """
    w = weights or ENERGY_VALUE_WEIGHTS
    all_pokemon = [(active_id, active_energy, True)] + [(bid, be, False) for bid, be in zip(bench_ids, bench_energies) if bid > 0]
    if not all_pokemon:
        return 0.0, {"readiness": 0.0, "powered": 0, "bench_powered": 0, "efficiency": 1.0, "wasted": 0}

    bench_ids_set = set(bench_ids)
    total_readiness = 0.0
    powered_count = 0
    bench_powered_count = 0
    total_efficiency = 0.0
    wasted_energy = 0

    # Determine whether the bench has any unpowered Pokémon that could use energy.
    # Used by Fix Bug 1: apply min_req overcharge threshold when bench needs energy.
    bench_has_hungry = any(
        be < get_min_required_energy(bid)
        for bid, be, is_act in all_pokemon
        if not is_act and bid > 0
    )

    for card_id, energy, is_active in all_pokemon:
        max_req = get_dynamic_energy_cap(card_id)
        overflow_threshold = max_req

        readiness = min(1.0, energy / max_req)
        # Dynamic role weight: context-aware (Articuno scales with board state)
        role_w = _dynamic_role_weight(card_id, active_id, bench_ids_set, prizes, opp_prizes)
        total_readiness += readiness * role_w

        if energy >= max_req:
            powered_count += 1
            if not is_active:
                bench_powered_count += 1

        efficiency = overflow_threshold / max(overflow_threshold, energy)
        total_efficiency += efficiency

        if energy > overflow_threshold:
            wasted_energy += (energy - overflow_threshold)

    n_pokemon = len(all_pokemon)
    mean_efficiency = total_efficiency / n_pokemon

    board_value = (
        w.get("readiness", 0.35) * total_readiness +
        w.get("powered", 0.25) * powered_count +
        w.get("bench_powered", 0.20) * bench_powered_count +
        w.get("efficiency", 0.15) * mean_efficiency -
        0.50 * wasted_energy
    )

    metrics = {
        "readiness": total_readiness,
        "powered": powered_count,
        "bench_powered": bench_powered_count,
        "efficiency": mean_efficiency,
        "wasted": wasted_energy,
        "board_value": board_value
    }
    return board_value, metrics


# ---------------------------------------------------------------------------
# State-Transition Evaluator (S -> S') with Opportunity Cost
# ---------------------------------------------------------------------------
def evaluate_state_energy_transition(
    active_id_before: int,
    active_energy_before: int,
    bench_ids_before: List[int],
    bench_energies_before: List[int],
    active_id_after: int,
    active_energy_after: int,
    bench_ids_after: List[int],
    bench_energies_after: List[int],
    target_card_id: int = -1,
    energy_card_id: int = 0,
    can_attack_this_turn: bool = False,
    turn: int = 1
) -> EnergyEvaluation:
    """
    Evaluates StateBefore (S) -> StateAfter (S') transition.
    Calculates exact metric deltas, computes Opportunity Cost against best legal attachment target,
    and returns a strictly bounded [-0.10, +0.10] dense AlphaZero reward.
    """
    bench_ids_before = bench_ids_before or []
    bench_energies_before = list(bench_energies_before or [])
    if len(bench_energies_before) < len(bench_ids_before):
        bench_energies_before.extend([0] * (len(bench_ids_before) - len(bench_energies_before)))

    bench_ids_after = bench_ids_after or []
    bench_energies_after = list(bench_energies_after or [])
    if len(bench_energies_after) < len(bench_ids_after):
        bench_energies_after.extend([0] * (len(bench_ids_after) - len(bench_energies_after)))

    v_before, m_before = compute_board_energy_value(
        active_id_before, active_energy_before, bench_ids_before, bench_energies_before
    )
    v_after, m_after = compute_board_energy_value(
        active_id_after, active_energy_after, bench_ids_after, bench_energies_after
    )

    readiness_delta = m_after["readiness"] - m_before["readiness"]
    powered_delta = int(m_after["powered"] - m_before["powered"])
    backup_delta = int(m_after["bench_powered"] - m_before["bench_powered"])
    efficiency_delta = m_after["efficiency"] - m_before["efficiency"]
    wasted_delta = int(m_after["wasted"] - m_before["wasted"])

    # 1. Readiness Gain Reward
    req_target = get_required_energy(target_card_id) if target_card_id > 0 else 2
    r_gain = readiness_delta * 0.05

    # 2. New Powered Attacker Bonus
    new_attacker_bonus = 0.05 if powered_delta > 0 else 0.0

    # 3. Backup Attacker & Bench Preparation Bonus
    # ONLY reward bench backup attachment if the active attacker is ALREADY fully powered!
    active_req = get_required_energy(active_id_before) if active_id_before > 0 else 2
    is_active_target = (target_card_id > 0 and target_card_id == active_id_before)
    is_bench_target  = (target_card_id > 0 and target_card_id != active_id_before)

    if active_energy_before >= active_req and is_bench_target:
        backup_bonus = 0.15
        bench_prep_bonus = 0.15
    elif is_active_target and powered_delta > 0:
        backup_bonus = 0.20  # Active attacker reached attack readiness THIS turn!
        bench_prep_bonus = 0.0
    else:
        backup_bonus = 0.0
        bench_prep_bonus = 0.0

    # 4. Energy Efficiency / Overcharge Penalty (harsh penalty schedule)
    if wasted_delta == 0:
        efficiency_penalty = 0.0
    elif wasted_delta == 1:
        efficiency_penalty = 0.20
    elif wasted_delta == 2:
        efficiency_penalty = 0.30
    else:
        efficiency_penalty = 0.40

    # 5. Opportunity Cost Calculation (Compare chosen target S' against best legal target S'*)
    legal_targets = [active_id_before] + [bid for bid in bench_ids_before if bid > 0]
    best_possible_value = v_before
    for candidate_id in legal_targets:
        is_act = (candidate_id == active_id_before)
        cand_act_e = active_energy_before + (1 if is_act else 0)
        cand_bench_e = list(bench_energies_before)
        if not is_act:
            for i, bid in enumerate(bench_ids_before):
                if bid == candidate_id and i < len(cand_bench_e):
                    cand_bench_e[i] = bench_energies_before[i] + 1
                    break
        v_cand, _ = compute_board_energy_value(
            active_id_before, cand_act_e, bench_ids_before, cand_bench_e
        )
        if v_cand > best_possible_value:
            best_possible_value = v_cand

    # Fix Bug 4: Only penalize when a genuinely better legal target existed.
    # If best_possible_value == v_before, no candidate improved the board at all —
    # firing OpCost in that case would penalize the agent for any attachment,
    # even when every target is already fully powered.
    if best_possible_value <= v_before:
        opportunity_cost = 0.0
    else:
        opportunity_cost = max(0.0, best_possible_value - v_after)

    # 6. Strategic Timing Bonus & Turn Phase Scaling
    strategic_timing = 0.02 if (powered_delta > 0 and can_attack_this_turn) else 0.0
    turn_factor = 1.2 if turn <= 3 else (1.0 if turn <= 8 else 0.8)

    # Compose final dense reward
    raw_reward = (
        r_gain +
        new_attacker_bonus +
        backup_bonus +
        bench_prep_bonus -
        efficiency_penalty -
        (0.5 * opportunity_cost) +
        strategic_timing
    )

    bounded_reward = max(-0.10, min(0.10, raw_reward * turn_factor))

    return EnergyEvaluation(
        readiness_delta=readiness_delta,
        powered_attackers_delta=powered_delta,
        backup_delta=backup_delta,
        efficiency_delta=efficiency_delta,
        wasted_energy_delta=wasted_delta,
        opportunity_cost=opportunity_cost,
        board_value_delta=(v_after - v_before),
        bounded_reward=bounded_reward
    )


# ---------------------------------------------------------------------------
# Backward Compatibility Wrappers
# ---------------------------------------------------------------------------
def compute_generic_energy_reward(
    target_card_id: int,
    energy_before: int,
    energy_after: int,
    attached_card_id: int = 0,
    is_active_target: bool = True,
    bench_ids: Optional[List[int]] = None,
    bench_energies_before: Optional[List[int]] = None,
    bench_energies_after: Optional[List[int]] = None,
    active_id: int = -1,
    active_energy_before: int = 0,
    can_attack_this_turn: bool = False
) -> float:
    """Wrapper function delegating directly to state-transition evaluator."""
    bench_ids = bench_ids or []
    bench_energies_before = list(bench_energies_before or [])
    if len(bench_energies_before) < len(bench_ids):
        bench_energies_before.extend([0] * (len(bench_ids) - len(bench_energies_before)))

    if bench_energies_after is not None:
        bench_energies_aft = list(bench_energies_after)
        if len(bench_energies_aft) < len(bench_ids):
            bench_energies_aft.extend([0] * (len(bench_ids) - len(bench_energies_aft)))
    else:
        bench_energies_aft = list(bench_energies_before)
        if not is_active_target:
            energy_added = max(1, energy_after - energy_before)
            for i, bid in enumerate(bench_ids):
                if bid == target_card_id and i < len(bench_energies_aft):
                    bench_energies_aft[i] = bench_energies_before[i] + energy_added
                    break

    energy_added = max(1, energy_after - energy_before)
    if is_active_target:
        active_energy_after = energy_before + energy_added
    else:
        active_energy_after = active_energy_before

    eval_result = evaluate_state_energy_transition(
        active_id_before=active_id,
        active_energy_before=active_energy_before,
        bench_ids_before=bench_ids,
        bench_energies_before=bench_energies_before,
        active_id_after=active_id,
        active_energy_after=active_energy_after,
        bench_ids_after=bench_ids,
        bench_energies_after=bench_energies_aft,
        target_card_id=target_card_id,
        energy_card_id=attached_card_id,
        can_attack_this_turn=can_attack_this_turn
    )
    return eval_result.bounded_reward


def evaluate_energy_target(
    target_card_id: int,
    target_current_energy: int,
    energy_card_id: int,
    hand_ids: List[int],
    my_prizes: int,
    opp_prizes: int,
    bench_ids: Optional[List[int]] = None,
    bench_energies: Optional[List[int]] = None,
    active_id: int = -1,
    active_energy: int = 0,
    active_hp: int = 100,
) -> Tuple[float, str]:
    """Backward compatible prior score wrapper for MCTS heuristic search."""
    bench_ids = bench_ids or []
    energy_added = 2 if energy_card_id in (15, 19) else 1

    if _violates_type_restriction(target_card_id, energy_card_id):
        return -0.25, "Prior: Energy Type Restriction Violation (e.g. Basic G on Mimikyu)"

    # Rule 0: Crustle EX Immunity Matchup Rule
    # Crustle (344, 345, 407, 408) has an ability rendering it IMMUNE to attacks from EX Pokémon (Mewtwo ex).
    # Prioritize attaching energy to Spidops ex (401), Tarountula (400), or Mimikyu (434).
    if active_id in (344, 345, 407, 408):
        if target_card_id == 431:
            return -0.25, "Prior Guidance: Mewtwo ex Cannot Damage Crustle (EX Immunity)"
        elif target_card_id in (401, 400, 434):
            return 0.25, "Prior Guidance: PRIORITIZE Spidops/Mimikyu Energy vs Crustle (EX Immunity Counter)"

    # Rule 1: Team Rocket's Energy (ID 15, +2 energy) Strict Focus & Preservation for Mewtwo ex (431)
    # Mewtwo ex needs 1 TR Energy + 1 Grass Energy (3 energy total) for Psystrike.
    # Preserve TR Energy for Mewtwo ex unless Mewtwo ex is fully powered (>=3) or not on board.
    if energy_card_id in (15, 19):  # TR Energy
        mewtwo_in_play = (active_id == 431) or (431 in bench_ids)
        mewtwo_needs_energy = False
        if active_id == 431 and active_energy < 3:
            mewtwo_needs_energy = True
        elif 431 in bench_ids:
            mewtwo_idx = bench_ids.index(431)
            if bench_energies and mewtwo_idx < len(bench_energies) and bench_energies[mewtwo_idx] < 3:
                mewtwo_needs_energy = True

        if target_card_id != 431 and mewtwo_in_play and mewtwo_needs_energy:
            return -0.25, "Prior Guidance: Team Rocket Energy STRICTLY Reserved for Mewtwo ex"

    # Rule 2: Active Pokémon Max Energy Check & Mandatory Bench Redirection
    # Effective max operational energy cap: Mewtwo ex = 3, Spidops ex = 2, Mimikyu/Tarountula = 1, Articuno = 1
    max_useful_energy = 3 if target_card_id == 431 else (2 if target_card_id == 401 else 1)
    if target_current_energy >= max_useful_energy:
        # Check if any benched Pokemon needs energy
        bench_has_hungry = False
        if bench_ids and bench_energies:
            for bid, ben in zip(bench_ids, bench_energies):
                req_b = 3 if bid == 431 else (2 if bid == 401 else 1)
                if bid > 0 and ben < req_b:
                    bench_has_hungry = True
                    break
        if not bench_has_hungry and active_id > 0 and active_id != target_card_id:
            req_a = 3 if active_id == 431 else (2 if active_id == 401 else 1)
            if active_energy < req_a:
                bench_has_hungry = True

        if bench_has_hungry:
            return -1.50, "Prior Guidance: Active Already at Max Energy (HARD BLOCK - Mandatory Bench Energy Setup)"
        return -0.40, "Prior Guidance: Energy Overcharge (Target already powered)"

    # Rule 3: Mewtwo ex Main Attacker Energy Priority (Builds 3 Energy for Psystrike/Psywave)
    if target_card_id == 431 and target_current_energy < 3:
        # Check if active is already powered or if target IS active
        req_act = 3 if active_id == 431 else (2 if active_id == 401 else 1)
        if target_card_id == active_id or active_energy >= req_act:
            return 0.35, "Prior Guidance: HIGH PRIORITY - Build Up Mewtwo ex Main Attacker Energy"

    # Rule 4: Active Attacker Energy Priority (Prevents 0-Energy Active Starvation)
    if target_card_id == active_id and active_id in (401, 414, 431):
        req_a = 3 if active_id == 431 else (2 if active_id == 401 else 1)
        if target_current_energy < req_a:
            return 0.30, "Prior Guidance: Active Attacker Energy Priority (Power Active Attacker)"

    # Rule 5: Suppress Spidops Bench Overcharging when Mewtwo ex is Hungry
    if target_card_id in (400, 401) and target_current_energy >= 1:
        mewtwo_hungry = (active_id == 431 and active_energy < 3)
        if not mewtwo_hungry and bench_ids and bench_energies:
            for b, ben in zip(bench_ids, bench_energies):
                if b == 431 and ben < 3:
                    mewtwo_hungry = True
                    break
        if mewtwo_hungry:
            return -0.25, "Prior Guidance: Suppress Spidops Bench Overcharge (Mewtwo ex Needs Energy First)"

    bench_e_after = list(bench_energies)
    is_act = (target_card_id == active_id)  # Fix: define is_act before use (was NameError)
    if not is_act:
        for i, bid in enumerate(bench_ids):
            if bid == target_card_id:
                bench_e_after[i] += energy_added
                break

    eval_res = evaluate_state_energy_transition(
        active_id_before=active_id,
        active_energy_before=active_energy,
        bench_ids_before=bench_ids,
        bench_energies_before=bench_energies,
        active_id_after=active_id,
        active_energy_after=active_energy + (energy_added if is_act else 0),
        bench_ids_after=bench_ids,
        bench_energies_after=bench_e_after,
        target_card_id=target_card_id,
        energy_card_id=energy_card_id
    )

    # Normalize bounded_reward [-0.10, +0.10] to standard prior bonus range [-0.20, +0.20]
    prior_score = max(-0.20, min(0.20, eval_res.bounded_reward * 2.0))
    return prior_score, "StateTransition_Energy_Valuation"
