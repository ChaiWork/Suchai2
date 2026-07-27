import os
import random

from cg.api import (
    AreaType,
    CardType,
    Observation,
    SelectContext,
    OptionType,
    Card,
    Pokemon,
    all_card_data,
    to_observation_class
)

"""
======================================================================
MARNIE'S GRIMMSNARL EX / MUNKIDORI SPREAD DECK EXPERT RULE-BASED AGENT
======================================================================
Archetype: Impidimp (646) -> Morgrem (647) -> Marnie's Grimmsnarl ex (648)
Support:   Munkidori (112) + Snorunt (860) -> Froslass (104)
Stadium:   Spikemuth Gym (1259) (+30 Dark damage)
Engine:    Rare Candy (1079), Dawn (1231), Unfair Stamp (1080), Handheld Fan (1161)
======================================================================
"""

# Path resolution following Kaggle-compatible standards
file_path = os.path.join("decks", "Rulebasedmodel_Grimmsnarl_ex", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("decks", "hard", "Rulebasedmodel_Grimmsnarl_ex", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "Rulebasedmodel_Grimmsnarl_ex", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "hard", "Rulebasedmodel_Grimmsnarl_ex", "deck.csv")
if not os.path.exists(file_path):
    file_path = "deck.csv"

my_deck = []
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
        for line in lines[:60]:
            try:
                my_deck.append(int(line))
            except ValueError:
                pass

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# ======================================================================
# CARD CONSTANTS
# ======================================================================
# Pokémon
Marnie_Impidimp = 646
Marnie_Morgrem = 647
Marnie_Grimmsnarl_ex = 648
Munkidori = 112
Snorunt = 860
Froslass = 104

# Energy
Basic_Dark_Energy = 7

# Trainer Cards — Items & Search
Buddy_Buddy_Poffin = 1086
Rare_Candy = 1079
Night_Stretcher = 1097
Pokegear = 1122
Poke_Pad = 1152
Unfair_Stamp = 1080

# Trainer Cards — Supporters
Boss_Orders = 1182
Dawn = 1231
Lillie_Determination = 1227
Team_Rocket_Petrel = 1219

# Trainer Cards — Tools & Stadiums
Handheld_Fan = 1161
Spikemuth_Gym = 1259


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Safely retrieves a Card or Pokemon object from the observation given area and index."""
    try:
        ps = obs.current.players[player_index]
        match area:
            case AreaType.DECK:
                return obs.select.deck[index] if obs.select and obs.select.deck and index < len(obs.select.deck) else None
            case AreaType.HAND:
                return ps.hand[index] if ps.hand and index < len(ps.hand) else None
            case AreaType.DISCARD:
                return ps.discard[index] if ps.discard and index < len(ps.discard) else None
            case AreaType.ACTIVE:
                return ps.active[index] if ps.active and index < len(ps.active) else None
            case AreaType.BENCH:
                return ps.bench[index] if ps.bench and index < len(ps.bench) else None
            case AreaType.PRIZE:
                return ps.prize[index] if ps.prize and index < len(ps.prize) else None
            case AreaType.STADIUM:
                return obs.current.stadium[index] if obs.current.stadium and index < len(obs.current.stadium) else None
            case AreaType.LOOKING:
                return obs.current.looking[index] if obs.current.looking and index < len(obs.current.looking) else None
            case _:
                return None
    except Exception:
        return None


def agent(obs_raw) -> list[int]:
    """
    Production-Quality Deterministic Rule-Based Agent for Grimmsnarl-ex / Munkidori Spread.
    Score-based option selection with strict hierarchy and strategic card knowledge.
    """
    obs = to_observation_class(obs_raw)

    if obs.select is None or not obs.select.option:
        return my_deck if my_deck else []

    options = obs.select.option
    ctx = obs.select.context
    your_index = obs.current.yourIndex
    my_player = obs.current.players[your_index]
    opp_player = obs.current.players[1 - your_index]

    max_count = getattr(obs.select, "maxCount", 1) or 1
    min_count = getattr(obs.select, "minCount", 1) or 1

    # 1. Handle single option case
    if len(options) == 1:
        return [0]

    # 2. Handle YES/NO prompts safely
    if options[0].type in (OptionType.YES, OptionType.NO):
        for idx, opt in enumerate(options):
            if opt.type == OptionType.YES:
                return [idx]
        return [0]

    # 3. Handle Active Promotion / Setup Prompts (TO_ACTIVE / SETUP_ACTIVE_POKEMON / SWITCH)
    if ctx in (SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SWITCH):
        candidate_scores = []
        for idx, opt in enumerate(options):
            cand_id = -1
            if hasattr(opt, "cardId") and opt.cardId != -1:
                cand_id = opt.cardId
            elif hasattr(opt, "index") and 0 <= opt.index < len(my_player.bench) and my_player.bench[opt.index]:
                cand_id = my_player.bench[opt.index].id

            score = 10.0
            if cand_id == Marnie_Grimmsnarl_ex:
                score += 100.0  # Primary 320 HP heavy attacker
            elif cand_id == Marnie_Morgrem:
                score += 70.0
            elif cand_id == Marnie_Impidimp:
                score += 50.0   # Basic evolution target
            elif cand_id == Munkidori:
                score += 40.0   # Poison / Adrena-Brain pivot
            elif cand_id == Snorunt or cand_id == Froslass:
                score += 30.0

            candidate_scores.append((idx, score))

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return [candidate_scores[0][0]]

    # 4. Score-Based Option Selection for Main Game Decisions
    option_scores = []

    # Pre-calculate game context metrics
    opp_hand_count = getattr(opp_player, "handCount", len(opp_player.hand) if opp_player.hand else 0)
    my_bench_count = len([b for b in my_player.bench if b is not None])
    opp_bench_count = len([b for b in opp_player.bench if b is not None])
    active_energy_count = len(my_player.active[0].energies) if my_player.active and my_player.active[0] else 0

    for idx, opt in enumerate(options):
        score = 0.0

        # ======================================================================
        # PRIORITY 1: ATTACK (+1000 Base)
        # ======================================================================
        if opt.type == OptionType.ATTACK:
            score += 1000.0
            # Active Grimmsnarl ex with energy ready to deal massive damage
            if my_player.active and my_player.active[0] and my_player.active[0].id == Marnie_Grimmsnarl_ex:
                score += 200.0
                if active_energy_count >= 2:
                    score += 100.0  # Main attack ready

        # ======================================================================
        # PRIORITY 2: EVOLUTION (+850 Base)
        # ======================================================================
        elif opt.type == OptionType.EVOLVE:
            score += 850.0
            evolved_cid = getattr(opt, "cardId", -1)

            if evolved_cid == Marnie_Grimmsnarl_ex:
                score += 150.0  # Critical Stage 2 evolution (320 HP raid boss)
            elif evolved_cid == Marnie_Morgrem:
                score += 80.0
            elif evolved_cid == Froslass:
                score += 60.0   # Status / paralysis support

        # ======================================================================
        # PRIORITY 3: ABILITY (+700 Base)
        # ======================================================================
        elif opt.type == OptionType.ABILITY:
            score += 700.0
            # Munkidori Adrena-Brain damage counter manipulation
            score += 50.0

        # ======================================================================
        # PRIORITY 4: ENERGY ATTACHMENT (+500 Base)
        # ======================================================================
        elif opt.type == OptionType.ATTACH:
            score += 500.0
            target_id = -1
            target_energies = 0

            # Detect attachment target
            if hasattr(opt, "inPlayArea") and hasattr(opt, "inPlayIndex"):
                if opt.inPlayArea == AreaType.ACTIVE and len(my_player.active) > 0 and my_player.active[0]:
                    target_id = my_player.active[0].id
                    target_energies = len(my_player.active[0].energies)
                elif opt.inPlayArea == AreaType.BENCH and 0 <= opt.inPlayIndex < len(my_player.bench) and my_player.bench[opt.inPlayIndex]:
                    target_id = my_player.bench[opt.inPlayIndex].id
                    target_energies = len(my_player.bench[opt.inPlayIndex].energies)

            if target_id == Marnie_Grimmsnarl_ex:
                if target_energies < 2:
                    score += 200.0  # Top priority: charge Grimmsnarl ex to 2 Dark Energies
                else:
                    score += 50.0
            elif target_id in (Marnie_Morgrem, Marnie_Impidimp):
                score += 150.0  # Charge pre-evolution line
            elif target_id == Munkidori:
                score += 100.0  # Charge Munkidori for Poison chip attack

        # ======================================================================
        # PRIORITY 5-9: PLAY / CARD (SUPPORTER, ITEM, TOOL, STADIUM, BENCH)
        # ======================================================================
        elif opt.type in (OptionType.PLAY, OptionType.CARD):
            score += 300.0
            played_cid = getattr(opt, "cardId", -1)

            if played_cid == -1 and hasattr(opt, "index") and my_player.hand and 0 <= opt.index < len(my_player.hand) and my_player.hand[opt.index]:
                played_cid = my_player.hand[opt.index].id

            # Supporter Cards (+350 Priority)
            if played_cid == Boss_Orders:
                score += 150.0  # Gust target: priority Boss's Orders
            elif played_cid == Dawn:
                score += 120.0  # Energy transfer to active Grimmsnarl ex
            elif played_cid == Unfair_Stamp and opp_hand_count >= 5:
                score += 110.0  # Hand disruption when opponent has full hand
            elif played_cid in (Lillie_Determination, Team_Rocket_Petrel):
                score += 90.0   # Search / draw support

            # Item Cards (+300 Priority)
            elif played_cid == Buddy_Buddy_Poffin:
                if my_bench_count < 4:
                    score += 120.0  # Early bench flooding with Impidimp & Munkidori
                else:
                    score += 50.0
            elif played_cid == Rare_Candy:
                score += 110.0  # Fast-track Impidimp -> Grimmsnarl ex
            elif played_cid in (Poke_Pad, Night_Stretcher, Pokegear):
                score += 60.0

            # Tool Cards (+250 Priority)
            elif played_cid == Handheld_Fan:
                score += 50.0   # Attach tool to Grimmsnarl ex to clear status

            # Stadium Cards (+250 Priority)
            elif played_cid == Spikemuth_Gym:
                score += 80.0   # Boost Dark attacks by +30 damage

            # Bench Pokémon (Basic) (+200 Priority)
            elif played_cid == Marnie_Impidimp:
                score += 100.0  # High priority basic for evolution
            elif played_cid == Munkidori:
                score += 90.0   # High priority support basic
            elif played_cid == Snorunt:
                score += 60.0

        # ======================================================================
        # PRIORITY 10: RETREAT (+150 Base)
        # ======================================================================
        elif opt.type == OptionType.RETREAT:
            score += 150.0
            # Only retreat if active is NOT Grimmsnarl ex and bench has Grimmsnarl ex ready
            if my_player.active and my_player.active[0] and my_player.active[0].id != Marnie_Grimmsnarl_ex:
                has_ready_grimmsnarl = any(b and b.id == Marnie_Grimmsnarl_ex and len(b.energies) >= 2 for b in my_player.bench)
                if has_ready_grimmsnarl:
                    score += 200.0  # Strategic retreat to ready Grimmsnarl ex

        option_scores.append((idx, score))

    # Sort options by score descending
    option_scores.sort(key=lambda x: x[1], reverse=True)

    # Return top options up to maxCount
    selected_indices = [x[0] for x in option_scores[:max_count]]
    return selected_indices if selected_indices else [0]
