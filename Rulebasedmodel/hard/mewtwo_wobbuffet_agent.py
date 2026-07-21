import os
import sys
from collections import defaultdict

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
Team Rocket's Mewtwo ex + Wobbuffet Deck
Tactical, Rule-based Agent
"""

# Load deck.csv in the dataset
file_path = "decks/Rulebasedmodel_Mewtwo_Wobbuffet/deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/decks/Rulebasedmodel_Mewtwo_Wobbuffet/deck.csv"

# Fallback to local if not found
if not os.path.exists(file_path):
    file_path = "deck.csv"

my_deck = []
if os.path.exists(file_path):
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    for i in range(min(60, len(csv))):
        if csv[i].strip():
            my_deck.append(int(csv[i].strip()))
else:
    my_deck = [431]*60 # safe fallback

# Fetch card metadata database
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist constants
Mewtwo_ex = 431
Spidops = 401
Tarountula = 400
Articuno = 414
Mimikyu = 434
Wobbuffet = 432
Murkrow = 463

Team_Rockets_Energy = 15
Basic_Psychic_Energy = 5
Basic_Grass_Energy = 1

Bug_Catching_Set = 1094
Night_Stretcher = 1097
Energy_Search = 1119
Ultra_Ball = 1121
Sacred_Ash = 1129
Rockets_Transceiver = 1134
Poke_Pad = 1152
Heros_Cape = 1159
Brave_Bangle = 1175

Rockets_Ariana = 1216
Rockets_Archer = 1217
Rockets_Giovanni = 1218
Rockets_Petrel = 1219
Rockets_Proton = 1220
Lillies_Determination = 1227
Rockets_Factory = 1257

def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None

def is_team_rocket_pokemon(card_id: int) -> bool:
    return card_id in [Tarountula, Spidops, Mewtwo_ex, Articuno, Mimikyu, Wobbuffet, Murkrow]

def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    # Gather board state counts
    field_counts = defaultdict(int)
    field_hand_counts = defaultdict(int)
    
    active_mewtwo = False
    active_wobbuffet = False
    active_murkrow = False
    
    benched_tarountula_count = 0
    benched_spidops_count = 0
    benched_mewtwo_count = 0
    benched_wobbuffet_count = 0
    
    total_rocket_pokemon_in_play = 0
    
    active_energies_count = 0
    benched_energies_count = 0

    for p in my_state.active:
        if p is None:
            continue
        field_counts[p.id] += 1
        field_hand_counts[p.id] += 1
        active_energies_count += len(p.energyCards)
        if is_team_rocket_pokemon(p.id):
            total_rocket_pokemon_in_play += 1
        if p.id == Mewtwo_ex:
            active_mewtwo = True
        elif p.id == Wobbuffet:
            active_wobbuffet = True
        elif p.id == Murkrow:
            active_murkrow = True

    max_bench_damage = 0
    for p in my_state.bench:
        if p is None:
            continue
        field_counts[p.id] += 1
        field_hand_counts[p.id] += 1
        benched_energies_count += len(p.energyCards)
        if is_team_rocket_pokemon(p.id):
            total_rocket_pokemon_in_play += 1
        if p.id == Tarountula:
            benched_tarountula_count += 1
        elif p.id == Spidops:
            benched_spidops_count += 1
        elif p.id == Mewtwo_ex:
            benched_mewtwo_count += 1
        elif p.id == Wobbuffet:
            benched_wobbuffet_count += 1
            
        damage = p.maxHp - p.hp
        if damage > max_bench_damage:
            max_bench_damage = damage

    hand_counts = defaultdict(int)
    for c in my_state.hand:
        hand_counts[c.id] += 1
        field_hand_counts[c.id] += 1

    discard_counts = defaultdict(int)
    for c in my_state.discard:
        discard_counts[c.id] += 1

    stadium_id = 0
    for c in state.stadium:
        stadium_id = c.id

    near_deckout = (my_state.deckCount <= 3)

    active_is_ex = False
    if len(my_state.active) >= 1 and my_state.active[0] is not None:
        active_id = my_state.active[0].id
        if active_id == Mewtwo_ex:
            active_is_ex = True

    op_active_id = 0
    op_active_hp = 9999
    op_active_is_ex = False
    if len(op_state.active) >= 1 and op_state.active[0] is not None:
        op_active = op_state.active[0]
        op_active_id = op_active.id
        op_active_hp = op_active.hp
        op_active_card = card_table.get(op_active.id)
        if op_active_card and getattr(op_active_card, 'ex', False):
            op_active_is_ex = True

    op_active_is_immune = (op_active_id == Mimikyu)

    DECKLIST_TOTALS = {
        Mewtwo_ex: 2,
        Spidops: 4,
        Tarountula: 4,
        Wobbuffet: 1,
        Murkrow: 2,
        Team_Rockets_Energy: 4,
        Night_Stretcher: 1,
        Ultra_Ball: 0
    }

    scores = []
    for idx, o in enumerate(select.option):
        score = 0
        
        if o.type == OptionType.END:
            if op_state.deckCount == 0:
                score = 200000
            else:
                score = 0
                
        elif o.type == OptionType.NUMBER:
            score = o.number
            
        elif o.type == OptionType.YES:
            score = 100
            if hasattr(select, 'message') and "discard" in str(select.message).lower():
                needed_damage = op_active_hp - 160
                if needed_damage <= 0:
                    score = 1
                else:
                    score = 200
                    
        elif o.type == OptionType.NO:
            score = 50
            if hasattr(select, 'message') and "discard" in str(select.message).lower():
                needed_damage = op_active_hp - 160
                if needed_damage <= 0:
                    score = 300
                    
        elif o.type == OptionType.EVOLVE:
            score = 120000
            
        elif o.type == OptionType.RETREAT:
            if op_active_is_immune and active_is_ex:
                score = 180000
            elif active_mewtwo:
                score = -100
            else:
                score = 1000
                    
        elif o.type == OptionType.ABILITY:
            if near_deckout:
                score = -100
            else:
                score = 89000
            
        elif o.type == OptionType.ATTACK:
            if op_active_is_immune and active_is_ex:
                score = -100
            else:
                score = 60000
                if o.attackId == 608:  # Erasure Ball
                    score += 10000
                elif o.attackId == 609:  # Rocket Mirror (Tech move)
                    if max_bench_damage >= 50:
                        score += 15000
                    else:
                        score -= 50000
                elif o.attackId == 560:  # Rocket Rush
                    score += 5000
                elif o.attackId == 652:  # Deceit (Search Supporter)
                    score += 4000
                
        elif o.type == OptionType.PLAY:
            c = get_card(obs, AreaType.HAND, o.index, my_index)
            if c is not None:
                card_data = card_table.get(c.id)
                if card_data is not None:
                    if card_data.cardType == CardType.SUPPORTER:
                        score = 75000
                        if c.id == Rockets_Proton:
                            if near_deckout:
                                score = -100
                            else:
                                score = 95000
                        elif c.id == Rockets_Ariana:
                            if near_deckout:
                                score = -100
                            else:
                                score = 88000
                        elif c.id == Rockets_Giovanni:
                            if benched_mewtwo_count > 0 and benched_energies_count >= 2:
                                score = 90000
                            else:
                                score = 1000
                        elif c.id == Rockets_Petrel:
                            score = 75000
                        elif c.id == Rockets_Archer:
                            if near_deckout:
                                score = -100
                            else:
                                score = 70000
                        elif c.id == Lillies_Determination:
                            if near_deckout:
                                score = -100
                            else:
                                score = 75000
                            
                    elif card_data.cardType == CardType.STADIUM:
                        if c.id == Rockets_Factory:
                            if stadium_id != Rockets_Factory:
                                score = 95000
                            else:
                                score = -1
                                
                    elif card_data.cardType == CardType.POKEMON:
                        score = 30000
                        is_bench_empty = (len(my_state.bench) == 0)
                        
                        if c.id == Tarountula:
                            if is_bench_empty:
                                score = 150000
                            elif field_counts[Tarountula] == 0:
                                score = 96000
                            elif field_counts[Tarountula] == 1:
                                score = 90000
                            else:
                                score = 65000
                        elif c.id == Mewtwo_ex:
                            if is_bench_empty:
                                score = 149000
                            elif field_counts[Mewtwo_ex] < 2:
                                score = 85000
                            else:
                                score = 50000
                        elif c.id == Murkrow:
                            if is_bench_empty:
                                score = 148500
                            elif field_counts[Murkrow] < 1:
                                score = 80000
                            else:
                                score = 40000
                        elif c.id == Wobbuffet:
                            if is_bench_empty:
                                score = 148000
                            elif field_counts[Wobbuffet] < 1:
                                score = 78000
                            else:
                                score = 30000
                        elif c.id == Articuno:
                            if is_bench_empty:
                                score = 147000
                            else:
                                score = 76000
                        elif c.id == Mimikyu:
                            if is_bench_empty:
                                score = 146000
                            else:
                                score = 75000
                            
                        if len(my_state.bench) >= 5:
                            score = -1
                            
                    else:  # Items
                        score = 72000
                        if c.id == Rockets_Transceiver:
                            if near_deckout:
                                score = -100
                            else:
                                score = 92000
                        elif c.id == Energy_Search:
                            score = 91000
                        elif c.id == Bug_Catching_Set:
                            if near_deckout:
                                score = -100
                            else:
                                score = 78000
                        elif c.id == Heros_Cape:
                            if active_mewtwo or benched_mewtwo_count > 0:
                                score = 93000
                            else:
                                score = 500
                        elif c.id == Brave_Bangle:
                            if op_active_is_ex:
                                score = 84000
                                
        elif o.type == OptionType.ATTACH or context == SelectContext.ATTACH_FROM:
            card = None
            if o.type == OptionType.ATTACH:
                card = get_card(obs, o.index, my_index)
                target_pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                is_active_target = (o.inPlayArea == AreaType.ACTIVE)
            else:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                target_pokemon = None
                is_active_target = False
                
            if card is not None:
                score = 50000
                is_vulnerable = False
                if target_pokemon is not None and hasattr(target_pokemon, "hp") and target_pokemon.hp is not None:
                    if target_pokemon.hp <= 60:
                        is_vulnerable = True
                
                if card.id == Team_Rockets_Energy:
                    score = 85000
                    if target_pokemon is not None:
                        if target_pokemon.id == Mewtwo_ex:
                            score += 15000
                            if not is_active_target:
                                score += 5000
                        elif target_pokemon.id == Spidops:
                            score += 5000
                    if is_vulnerable:
                        score -= 20000
                            
                elif card.id == Basic_Psychic_Energy:
                    score = 80000
                    if target_pokemon is not None:
                        if target_pokemon.id == Mewtwo_ex:
                            score += 10000
                            if not is_active_target:
                                score += 5000
                        elif target_pokemon.id == Wobbuffet:
                            score += 5000
                    if is_vulnerable:
                        score -= 10000
                                
                elif card.id == Basic_Grass_Energy:
                    score = 75000
                    if target_pokemon is not None:
                        if target_pokemon.id in [Tarountula, Spidops]:
                            score += 15000
                            
                elif card.id == Heros_Cape:
                    score = 500
                    if target_pokemon is not None and target_pokemon.id == Mewtwo_ex:
                        score = 88000
                        if is_active_target:
                            score += 5000
                            
                elif card.id == Brave_Bangle:
                    score = 500
                    if target_pokemon is not None and target_pokemon.id == Spidops:
                        if op_active_is_ex:
                            score = 84000
                            
        elif o.type == OptionType.CARD:
            c = get_card(obs, o.area, o.index, o.playerIndex)
            if c is not None:
                card_data = card_table.get(c.id)
                if (context == SelectContext.SWITCH or 
                    context == SelectContext.TO_ACTIVE or 
                    context == SelectContext.SETUP_ACTIVE_POKEMON):
                    if context == SelectContext.SETUP_ACTIVE_POKEMON:
                        if c.id == Murkrow:
                            score = 28000
                        elif c.id == Tarountula:
                            score = 27000
                        elif c.id == Articuno:
                            score = 26000
                        elif c.id == Mimikyu:
                            score = 25000
                        elif c.id in [Mewtwo_ex, Wobbuffet]:
                            score = 1000
                    else:
                        if op_active_is_immune and c.id in [Mewtwo_ex]:
                            score = -1000
                        elif c.id == Spidops:
                            score = 28000
                        elif c.id == Tarountula:
                            score = 27000
                        elif c.id == Articuno:
                            score = 26000
                        elif c.id == Mimikyu:
                            score = 25000
                        elif c.id == Mewtwo_ex:
                            mewtwo_energies = len(c.energies) if isinstance(c, Pokemon) else 0
                            if mewtwo_energies >= 2:
                                score = 30000
                            else:
                                score = 1000
                            
                elif context == SelectContext.TO_HAND or context == SelectContext.TO_BENCH:
                    if c.id == Tarountula:
                        if field_counts[Tarountula] == 0:
                            score = 60000
                        elif field_counts[Tarountula] < 2:
                            score = 45000
                        else:
                            score = 12000
                    elif c.id == Spidops:
                        if field_counts[Tarountula] > 0 and field_counts[Spidops] == 0:
                            score = 58000
                        elif field_counts[Tarountula] > field_counts[Spidops]:
                            score = 48000
                        else:
                            score = 10000
                    elif c.id == Mewtwo_ex:
                        if field_counts[Tarountula] > 0 and field_counts[Spidops] > 0:
                            score = 55000
                        else:
                            score = 35000
                    elif c.id == Wobbuffet:
                        score = 40000
                    elif c.id == Murkrow:
                        score = 38000
                    elif c.id == Team_Rockets_Energy:
                        score = 40000
                    elif c.id == Basic_Psychic_Energy:
                        score = 30000
                    elif c.id == Basic_Grass_Energy:
                        score = 25000
                        
                elif context == SelectContext.DISCARD:
                    is_last_critical = (c.id in DECKLIST_TOTALS and (DECKLIST_TOTALS[c.id] - discard_counts[c.id] <= 1))
                    if is_last_critical or c.id == Night_Stretcher:
                        score = -10000
                    elif c.id == Rockets_Factory and (stadium_id == Rockets_Factory or hand_counts[Rockets_Factory] > 1):
                        score = 8000
                    elif card_data is not None and card_data.cardType == CardType.SUPPORTER and hand_counts[c.id] > 1:
                        score = 7000
                    elif c.id in [Brave_Bangle, Heros_Cape] and hand_counts[c.id] > 1:
                        score = 6000
                    elif c.id in [Basic_Psychic_Energy, Basic_Grass_Energy]:
                        score = 5000
                    else:
                        score = 100
                
                elif hasattr(select, 'message') and "mirror" in str(select.message).lower():
                    if o.area == AreaType.BENCH:
                        bench_pk = get_card(obs, AreaType.BENCH, o.index, my_index)
                        if bench_pk is not None:
                            score = (bench_pk.maxHp - bench_pk.hp) * 1000
                        else:
                            score = -1000
                    else:
                        score = -1000

        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc_indices[:select.maxCount]
