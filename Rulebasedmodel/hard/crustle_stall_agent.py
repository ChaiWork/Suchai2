import os
import random
from cg.api import (
    AreaType, CardType, Observation, SelectContext, OptionType,
    Card, Pokemon, all_card_data, to_observation_class
)

"""
======================================================================
BANCHO CRUSTLE STALL & SPIKY ENERGY HEALING RULE-BASED AGENT
Archetype: Crustle (345) + Spiky Energy (14) + Cook (70 HP Heal) + Waitress
Cloned from loss replay 88505471
======================================================================
"""

file_path = os.path.join("decks", "hard", "Rulebasedmodel_Crustle_Stall", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "hard", "Rulebasedmodel_Crustle_Stall", "deck.csv")
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

Dwebble = 344
Crustle = 345
Spiky_Energy = 14
Cook = 1140
Waitress = 1141
Jumbo_Ice_Cream = 1147
Heros_Cape = 1159
Lillies_Determination = 1227

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

    # Active Promotion Logic: Wall behind Crustle
    if ctx in (SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON):
        candidate_scores = []
        for idx, opt in enumerate(options):
            cand_id = -1
            if hasattr(opt, "cardId") and opt.cardId != -1:
                cand_id = opt.cardId
            elif hasattr(opt, "index") and 0 <= opt.index < len(my_player.bench) and my_player.bench[opt.index]:
                cand_id = my_player.bench[opt.index].id
            score = 10.0
            if cand_id == Crustle:
                score += 100.0
            elif cand_id == Dwebble:
                score += 50.0
            candidate_scores.append((idx, score))
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return [candidate_scores[0][0]]

    best_idx = 0
    best_score = -9999.0

    active_hp_lost = 0
    if len(my_player.active) > 0 and my_player.active[0]:
        act = my_player.active[0]
        active_hp_lost = getattr(act, "maxHp", 120) - getattr(act, "hp", 120)

    for idx, opt in enumerate(options):
        score = 0.0

        if opt.type == OptionType.ATTACK:
            score += 900.0

        elif opt.type == OptionType.EVOLVE:
            score += 850.0

        elif opt.type == OptionType.ATTACH:
            score += 500.0
            attached_cid = getattr(opt, "cardId", -1)
            if attached_cid == Spiky_Energy:
                score += 300.0  # Spiky energy damage reflection

        elif opt.type in (OptionType.PLAY, OptionType.CARD, OptionType.ABILITY):
            score += 300.0
            played_cid = getattr(opt, "cardId", -1)

            # Heal when damaged
            if played_cid in (Cook, Jumbo_Ice_Cream, Waitress) and active_hp_lost >= 40:
                score += 600.0  # Priority heal to maintain wall
            elif played_cid == Heros_Cape:
                score += 400.0
            elif played_cid == Lillies_Determination:
                score += 250.0

        if score > best_score:
            best_score = score
            best_idx = idx

    return [best_idx]
