import os
import random
from cg.api import (
    AreaType, CardType, Observation, SelectContext, OptionType,
    Card, Pokemon, all_card_data, to_observation_class
)

"""
======================================================================
MEGA TYRANITAR EX / SANDACONDA RAMP RULE-BASED AGENT
Archetype: Larvitar -> Pupitar -> Mega Tyranitar ex (340 HP) + Sandaconda Ramp
Cloned from loss replays 88475970 & 88476494
======================================================================
"""

file_path = os.path.join("decks", "hard", "Rulebasedmodel_Tyranitar", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "hard", "Rulebasedmodel_Tyranitar", "deck.csv")
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

Larvitar = 246
Pupitar = 247
Mega_Tyranitar_ex = 248
Silicobra = 843
Sandaconda = 844
Fighting_Gong = 1085
Basic_Fighting_Energy = 6
Rock_Fighting_Energy = 20

def agent(obs_raw) -> list[int]:
    obs = to_observation_class(obs_raw)
    if obs.select is None or not obs.select.option:
        return []

    your_index = obs.current.yourIndex
    my_player = obs.current.players[your_index]
    options = obs.select.option
    ctx = obs.select.context

    if len(options) == 1:
        return [0]

    if options[0].type == OptionType.YES or options[0].type == OptionType.NO:
        for idx, opt in enumerate(options):
            if opt.type == OptionType.YES:
                return [idx]
        return [0]

    # Active Promotion Logic
    if ctx in (SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON):
        candidate_scores = []
        for idx, opt in enumerate(options):
            cand_id = -1
            if hasattr(opt, "cardId") and opt.cardId != -1:
                cand_id = opt.cardId
            elif hasattr(opt, "index") and 0 <= opt.index < len(my_player.bench) and my_player.bench[opt.index]:
                cand_id = my_player.bench[opt.index].id
            score = 10.0
            if cand_id == Mega_Tyranitar_ex:
                score += 100.0
            elif cand_id == Sandaconda:
                score += 60.0
            elif cand_id in (Larvitar, Silicobra):
                score += 30.0
            candidate_scores.append((idx, score))
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return [candidate_scores[0][0]]

    best_idx = 0
    best_score = -9999.0

    for idx, opt in enumerate(options):
        score = 0.0

        if opt.type == OptionType.ATTACK:
            score += 1000.0

        elif opt.type == OptionType.EVOLVE:
            score += 800.0

        elif opt.type == OptionType.ATTACH:
            score += 500.0
            target_id = -1
            if hasattr(opt, "inPlayArea") and hasattr(opt, "inPlayIndex"):
                if opt.inPlayArea == AreaType.ACTIVE and len(my_player.active) > 0 and my_player.active[0]:
                    target_id = my_player.active[0].id
                elif opt.inPlayArea == AreaType.BENCH and 0 <= opt.inPlayIndex < len(my_player.bench) and my_player.bench[opt.inPlayIndex]:
                    target_id = my_player.bench[opt.inPlayIndex].id
            if target_id == Mega_Tyranitar_ex:
                score += 250.0
            elif target_id == Sandaconda:
                score += 150.0

        elif opt.type in (OptionType.PLAY, OptionType.CARD, OptionType.ABILITY):
            score += 300.0
            played_cid = getattr(opt, "cardId", -1)
            if played_cid == Fighting_Gong:
                score += 350.0

        if score > best_score:
            best_score = score
            best_idx = idx

    return [best_idx]
