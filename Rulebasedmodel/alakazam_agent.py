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
ALAKAZAM DECK RULE-BASED EXPERT AGENT
Archetype: Abra (741) -> Kadabra (742) -> Alakazam (743) / Rare Candy (1079)
======================================================================
"""

file_path = os.path.join("decks", "hard", "Rulebasedmodel_Alakazam", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "hard", "Rulebasedmodel_Alakazam", "deck.csv")
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

Abra = 741
Kadabra = 742
Alakazam = 743
Dunsparce = 305
Dudunsparce = 66

Psychic_Energy = 5
Enriching_Energy = 13
Telepath_Energy = 19

Rare_Candy = 1079
Enhanced_Hammer = 1081
Buddy_Poffin = 1086
Night_Stretcher = 1097
Sacred_Ash = 1129
Poke_Pad = 1152
Boss_Orders = 1182
Lanas_Aid = 1184
Hilda = 1225
Dawn = 1231
Battle_Cage = 1264


def agent(obs_raw) -> list[int]:
    """Alakazam Rule-Based Agent decision function."""
    obs = to_observation_class(obs_raw)
    
    if obs.select is None or not obs.select.option:
        return []
        
    your_index = obs.current.yourIndex
    my_player = obs.current.players[your_index]
    opp_player = obs.current.players[1 - your_index]
    
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
    if ctx == SelectContext.TO_ACTIVE:
        candidate_scores = []
        for idx, opt in enumerate(options):
            cand_id = -1
            if hasattr(opt, "cardId") and opt.cardId != -1:
                cand_id = opt.cardId
            elif hasattr(opt, "index") and 0 <= opt.index < len(my_player.bench) and my_player.bench[opt.index]:
                cand_id = my_player.bench[opt.index].id
                
            score = 10.0
            if cand_id == Alakazam:
                score += 100.0
            elif cand_id in (Kadabra, Dudunsparce):
                score += 60.0
            elif cand_id in (Abra, Dunsparce):
                score += 30.0
            candidate_scores.append((idx, score))
            
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return [candidate_scores[0][0]]

    best_idx = 0
    best_score = -9999.0

    for idx, opt in enumerate(options):
        score = 0.0
        
        # 1. ATTACK (Highest priority)
        if opt.type == OptionType.ATTACK:
            score += 1000.0
            
        # 2. EVOLVE (Abra -> Kadabra / Alakazam, Dunsparce -> Dudunsparce)
        elif opt.type == OptionType.EVOLVE:
            score += 700.0
            
        # 3. ATTACH ENERGY (Prioritize Alakazam > Kadabra > Abra)
        elif opt.type == OptionType.ATTACH:
            score += 500.0
            target_id = -1
            if hasattr(opt, "inPlayArea") and hasattr(opt, "inPlayIndex"):
                if opt.inPlayArea == AreaType.ACTIVE and len(my_player.active) > 0 and my_player.active[0]:
                    target_id = my_player.active[0].id
                elif opt.inPlayArea == AreaType.BENCH and 0 <= opt.inPlayIndex < len(my_player.bench) and my_player.bench[opt.inPlayIndex]:
                    target_id = my_player.bench[opt.inPlayIndex].id
            if target_id == Alakazam:
                score += 200.0
            elif target_id in (Kadabra, Abra):
                score += 100.0

        # 4. PLAY / TRAINER / ABILITY
        elif opt.type in (OptionType.PLAY, OptionType.CARD, OptionType.ABILITY):
            score += 300.0
            played_cid = getattr(opt, "cardId", -1)
            if played_cid == -1 and hasattr(opt, "index") and my_player.hand and 0 <= opt.index < len(my_player.hand) and my_player.hand[opt.index]:
                played_cid = my_player.hand[opt.index].id
                
            if played_cid in (Rare_Candy, Enhanced_Hammer, Buddy_Poffin, Hilda, Dawn):
                score += 250.0
            elif played_cid in (Night_Stretcher, Sacred_Ash, Boss_Orders):
                score += 200.0

        if score > best_score:
            best_score = score
            best_idx = idx

    k = obs.select.maxCount
    if k > 1 and len(options) >= k:
        scored_all = [(idx, random.random()) for idx in range(len(options))]
        scored_all.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored_all[:k]]

    return [best_idx]
