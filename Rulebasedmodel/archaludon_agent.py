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
ARCHALUDON EX DECK RULE-BASED EXPERT AGENT
Archetype: Duraludon (169) -> Archaludon ex (190) + Full Metal Lab (1244)
======================================================================
"""

file_path = os.path.join("decks", "hard", "Rulebasedmodel_Archaludon", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "hard", "Rulebasedmodel_Archaludon", "deck.csv")
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

Duraludon = 169
Archaludon_ex = 190
Cinderace_Line = 666
Metal_Energy = 8

Night_Stretcher = 1097
Ultra_Ball = 1121
Pokegear = 1122
Jumbo_Ice_Cream = 1147
Poke_Pad = 1152
Heros_Cape = 1159
Boss_Orders = 1182
Explorers_Guidance = 1185
Xerosics_Machinations = 1197
Lillies_Determination = 1227
Full_Metal_Lab = 1244


def agent(obs_raw) -> list[int]:
    """Archaludon ex Rule-Based Agent decision function."""
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
            if cand_id == Archaludon_ex:
                score += 100.0  # Top 300 HP Metal tank
            elif cand_id == Duraludon:
                score += 50.0
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
            
        # 2. EVOLVE (Duraludon -> Archaludon ex)
        elif opt.type == OptionType.EVOLVE:
            score += 800.0  # Evolve for Aggregate Alloy energy acceleration
            
        # 3. ATTACH ENERGY (Prioritize Archaludon ex > Duraludon)
        elif opt.type == OptionType.ATTACH:
            score += 500.0
            target_id = -1
            if hasattr(opt, "inPlayArea") and hasattr(opt, "inPlayIndex"):
                if opt.inPlayArea == AreaType.ACTIVE and len(my_player.active) > 0 and my_player.active[0]:
                    target_id = my_player.active[0].id
                elif opt.inPlayArea == AreaType.BENCH and 0 <= opt.inPlayIndex < len(my_player.bench) and my_player.bench[opt.inPlayIndex]:
                    target_id = my_player.bench[opt.inPlayIndex].id
            if target_id == Archaludon_ex:
                score += 200.0
            elif target_id == Duraludon:
                score += 100.0

        # 4. PLAY / TRAINER / STADIUM
        elif opt.type in (OptionType.PLAY, OptionType.CARD, OptionType.ABILITY):
            score += 300.0
            played_cid = getattr(opt, "cardId", -1)
            if played_cid == -1 and hasattr(opt, "index") and my_player.hand and 0 <= opt.index < len(my_player.hand) and my_player.hand[opt.index]:
                played_cid = my_player.hand[opt.index].id
                
            # Full Metal Lab Stadium (Reduce metal damage by 30)
            if played_cid == Full_Metal_Lab:
                score += 350.0
            # Hero's Cape (+100 HP tool)
            elif played_cid == Heros_Cape:
                score += 300.0
            # Search & Draw Trainers
            elif played_cid in (Ultra_Ball, Explorers_Guidance, Lillies_Determination, Pokegear):
                score += 250.0
            # Xerosic's Disruption
            elif played_cid == Xerosics_Machinations:
                score += 220.0

        if score > best_score:
            best_score = score
            best_idx = idx

    k = obs.select.maxCount
    if k > 1 and len(options) >= k:
        scored_all = [(idx, random.random()) for idx in range(len(options))]
        scored_all.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored_all[:k]]

    return [best_idx]
