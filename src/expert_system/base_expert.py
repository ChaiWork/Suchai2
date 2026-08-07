"""
Base Expert System & Context Extractor for Pokémon TCG AI Agent.
"""
from cg.api import CardType, OptionType, SelectContext

USE_EXPERT_GUIDANCE = True
EXPERT_WEIGHT = 0.25

# Expert influence schedule — stepped plateaus rather than linear-to-zero decay.
# AlphaZero principle: expert priors are never fully removed; the NN learns ON TOP of them.
# A permanent 10% floor means late-training priors still guide MCTS away from rule violations.
#
# Epoch   1-20  → 1.00  (full curriculum: expert leads, NN follows)
# Epoch  21-50  → 0.50  (mid training:  expert and NN contribute equally)
# Epoch  51-100 → 0.25  (late training: NN dominates, expert guards boundaries)
# Epoch  100+   → 0.10  (floor:         permanent strategic prior, no complete removal)
# Legacy compatibility constant
EXPERT_DECAY_EPOCHS = 100

def expert_scale(epoch: int) -> float:
    """
    Returns a stepped-plateau multiplier controlling expert system influence.

    Design: AlphaZero systems retain expert priors permanently — the NN learns
    better strategies on top of the prior, not instead of it. A hard floor of 0.10
    ensures rule-safety guards (Turn 1 supporter block, Power Saver gate) remain
    active even at epoch 500+.

    Used as a universal gate for:
    - Expert logit biases in MCTS node priors (agent.py)
    - Reward shaping constants in mewtwo_reward.py

    Schedule:
      Epoch   1-20  → 1.00
      Epoch  21-50  → 0.50
      Epoch  51-100 → 0.25
      Epoch  100+   → 0.10 (permanent floor)
    """
    if not USE_EXPERT_GUIDANCE:
        return 0.0
    if epoch <= 30:
        return 1.00
    if epoch <= 60:
        return 0.80
    if epoch <= 100:
        return 0.70
    return 0.60  # Permanent strong floor — preserves expert priors during late fine-tuning (Epoch 30+)


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
