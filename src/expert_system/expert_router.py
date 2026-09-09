"""
Expert System Router for Pokémon TCG RL Agent.
Architecture: NN = Player, Expert = Coach, MCTS = Decision Mechanism.

Expert guidance is a soft prior bonus on MCTS action probabilities.
It never directly selects an action — the NN and MCTS always decide.

Confidence formula:
    confidence = epoch_scale(epoch) × uncertainty_scale(nn_entropy) × risk_scale(obs, option)

This ensures expert activates STRONGEST when:
  1. Training is early (epoch_scale high)
  2. NN is genuinely uncertain (uncertainty_scale high)
  3. The game state is strategically risky (risk_scale high)

And weakens naturally when the NN becomes confident and epoch advances.
"""
try:
    from expert_system.base_expert import USE_EXPERT_GUIDANCE, EXPERT_WEIGHT
    from expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
except ImportError:
    from src.expert_system.base_expert import USE_EXPERT_GUIDANCE, EXPERT_WEIGHT
    from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus


# High-risk Mewtwo state conditions that warrant stronger expert guidance
# regardless of NN confidence (because a confident-but-wrong NN is dangerous)
_MEWTWO_EX_ID      = 431
_SPIDOPS_ID        = 401
_ARTICUNO_ID       = 414
_MIMIKYU_ID        = 434
_BATTLE_CAGE_ID    = 1264
_DRAGAPULT_EX_ID   = 121  # Opponent threat ID — kept for risk detection logic
_DREEPY_ID         = 119
_DRAKLOAK_ID       = 120
_RETREAT_TYPE      = 5


def _compute_risk_scale(obs, option) -> float:
    """
    Returns a strategic risk multiplier in range [1.0, 1.60].
    Expert guidance is amplified selectively only in high-risk game states.

    High-risk situations for Mewtwo ex:
      - Powered Mewtwo / Spidops being retreated (retreat waste)
      - Energy being attached to Articuno when attackers need it
      - Opponent has Dragapult bench snipe threat and Battle Cage not deployed
      - Deck-out danger (deck count <= 5)
      - Bench has fewer than 3 TR Pokémon (setup incomplete)
    """
    try:
        opt_type = option.get("type", -1) if isinstance(option, dict) else getattr(option, "type", -1)
        card_id  = option.get("cardId", -1) if isinstance(option, dict) else getattr(option, "cardId", -1)

        current = obs.get("current") if isinstance(obs, dict) else getattr(obs, "current", None)
        if current is None:
            return 1.0

        yi = current.get("yourIndex", 0) if isinstance(current, dict) else getattr(current, "yourIndex", 0)
        players = current.get("players", []) if isinstance(current, dict) else getattr(current, "players", [])
        if not players or len(players) < 2:
            return 1.0

        me  = players[yi]
        opp = players[1 - yi]

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        my_active_list = _get(me, "active", [])
        active_id = -1
        active_energy = 0
        if my_active_list:
            ac = my_active_list[0]
            if ac:
                active_id     = _get(ac, "cardId", -1)
                energies      = _get(ac, "energyCards", [])
                active_energy = len(energies) if isinstance(energies, list) else 0

        my_bench  = _get(me, "bench", [])
        opp_bench = _get(opp, "bench", [])
        bench_ids     = [_get(b, "cardId", -1) for b in my_bench  if b] if isinstance(my_bench,  list) else []
        opp_bench_ids = [_get(b, "cardId", -1) for b in opp_bench if b] if isinstance(opp_bench, list) else []

        deck_count = _get(me, "deckCount", 20)
        tr_pokemon_ids = {400, 401, 414, 431, 434}
        tr_count = sum(1 for b in bench_ids if b in tr_pokemon_ids) + (1 if active_id in tr_pokemon_ids else 0)

        risk = 1.0

        # Risk 1: Powered attacker about to retreat — very high risk of energy waste
        if opt_type in (_RETREAT_TYPE, "Retreat"):
            if active_id in (_MEWTWO_EX_ID, _SPIDOPS_ID) and active_energy >= 2:
                risk = max(risk, 1.60)  # Maximum amplification: retreat veto is critical

        # Risk 2: Energy being attached to Articuno when main attackers not powered
        if opt_type in (8, "Attach"):
            target_id = card_id
            main_needs_energy = (
                _MEWTWO_EX_ID in bench_ids and active_id != _MEWTWO_EX_ID or
                _SPIDOPS_ID in bench_ids
            )
            if target_id == _ARTICUNO_ID and main_needs_energy:
                risk = max(risk, 1.40)

        # Risk 3: Opponent has Dragapult bench snipe threat, Battle Cage not deployed
        opp_has_dragapult = any(b in (_DRAGAPULT_EX_ID, _DREEPY_ID, _DRAKLOAK_ID) for b in opp_bench_ids)
        if opp_has_dragapult:
            risk = max(risk, 1.30)  # Battle Cage timing is critical vs Dragapult

        # Risk 4: Deck-out danger — avoid unnecessary draw/search
        if deck_count <= 5:
            risk = max(risk, 1.25)

        # Risk 5: TR Pokémon count below 3 (bench setup incomplete)
        if tr_count < 3:
            risk = max(risk, 1.20)

        return min(risk, 1.60)  # Hard cap at 1.60x to prevent expert domination

    except Exception:
        return 1.0  # Safe default: no amplification on error


def get_expert_bonus(obs, option, opponent_name: str = "", epoch: int = 0,
                     nn_entropy: float = 0.5) -> tuple[float, str]:
    """
    Unified entry point for retrieving expert action prior bonus and trigger label.

    Confidence = epoch_scale(epoch) × uncertainty_scale(nn_entropy) × risk_scale(obs, option)

    epoch_scale:       decays as NN trains and learns (no hard floor — NN can become dominant)
    uncertainty_scale: amplifies expert when NN is genuinely uncertain
    risk_scale:        amplifies expert selectively in strategic high-risk game states

    Returns: (weighted_bonus: float, trigger_name: str)
    """
    if not USE_EXPERT_GUIDANCE:
        return 0.0, "Disabled"

    try:
        from expert_system.base_expert import expert_scale
    except ImportError:
        from src.expert_system.base_expert import expert_scale

    # 1. Epoch-based decay (NN becomes more dominant over time)
    epoch_confidence = expert_scale(epoch)

    # 2. Uncertainty scale: uncertain NN → stronger expert; confident NN → weaker expert
    # Range [0.70, 1.30] to avoid expert dominating even a confident NN
    entropy_clamped   = max(0.0, min(1.0, float(nn_entropy)))
    uncertainty_scale = 0.70 + 0.60 * entropy_clamped

    # 3. Strategic risk scale: amplify only in genuinely dangerous game states
    risk_scale = _compute_risk_scale(obs, option)

    # Final combined confidence (expert is Coach, MCTS is the decision mechanism)
    confidence = epoch_confidence * uncertainty_scale * risk_scale

    bonus, triggered = evaluate_mewtwo_expert_bonus(obs, option, opponent_name)

    # Negative expert vetoes (bonus < 0) are strict strategic constraints.
    # They must NEVER be diluted by epoch decay or low entropy scaling so MCTS never ignores forbidden actions.
    final_multiplier = EXPERT_WEIGHT if bonus < 0 else (EXPERT_WEIGHT * confidence)
    return bonus * final_multiplier, triggered
