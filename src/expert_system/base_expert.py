"""
Base Expert System & Context Extractor for Pokémon TCG AI Agent.
"""
from cg.api import CardType, OptionType, SelectContext

USE_EXPERT_GUIDANCE = True
EXPERT_WEIGHT = 0.10

# Number of training epochs over which expert guidance fully decays to zero.
# Phase 1:  epoch 1-5   → scale 1.00 (full curriculum guidance)
# Phase 2:  epoch 6-10  → scale 0.70
# Phase 3:  epoch 11-20 → scale ~0.40
# Phase 4:  epoch 21-30 → scale ~0.15
# Phase 5:  epoch 31+   → scale 0.00 (pure RL — NN decides everything)
EXPERT_DECAY_EPOCHS = 40


def expert_scale(epoch: int) -> float:
    """
    Returns a [1.0 → 0.0] multiplier that linearly decays the expert system
    influence to zero over EXPERT_DECAY_EPOCHS epochs.

    Used as a universal decay gate for:
    - Expert logit biases in MCTS node priors (agent.py)
    - Hard logit blocks (overcharge, type mismatch, turn-pass guard)
    - Reward shaping constants in mewtwo_reward.py

    At epoch >= EXPERT_DECAY_EPOCHS the expert system has zero authority;
    the neural network is the sole decision-maker.
    """
    if not USE_EXPERT_GUIDANCE:
        return 0.0
    return max(0.0, 1.0 - (max(0, epoch - 1) / EXPERT_DECAY_EPOCHS))


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
                        ctx["my_active_id"]     = a.get("cardId", a.get("id", -1))
                        ctx["my_active_energy"] = len(a.get("energyCards", []))
                        ctx["my_active_hp"]     = a.get("hp", 100)
                    else:
                        ctx["my_active_id"]     = getattr(a, "cardId", getattr(a, "id", -1))
                        ctx["my_active_energy"] = len(getattr(a, "energyCards", []))
                        ctx["my_active_hp"]     = getattr(a, "hp", 100)

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
