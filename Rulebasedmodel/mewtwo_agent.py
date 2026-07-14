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
Team Rocket's Mewtwo ex Deck
Tactical, Rule-based Agent
"""

# Load deck.csv in the dataset
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path

with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

# Fetch card metadata database
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist constants
Mewtwo_ex = 431
Spidops = 401
Tarountula = 400
Articuno = 414
Mimikyu = 434
Clefairy_ex = 272

Team_Rockets_Energy = 15
Basic_Psychic_Energy = 5
Basic_Grass_Energy = 1

Bug_Catching_Set = 1094
Night_Stretcher = 1097
Energy_Switch = 1116
Ultra_Ball = 1121
Sacred_Ash = 1129
Rockets_Transceiver = 1134
Poke_Pad = 1152
Maximum_Belt = 1158
Brave_Bangle = 1175

Rockets_Ariana = 1216
Rockets_Archer = 1217
Rockets_Giovanni = 1218
Rockets_Petrel = 1219
Rockets_Proton = 1220
Lillies_Determination = 1227
Rockets_Factory = 1257

def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
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
    """Check if the card ID is a Team Rocket's Pokémon."""
    return card_id in [Tarountula, Spidops, Mewtwo_ex, Articuno, Mimikyu]

def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function."""
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
    active_tarountula = False
    active_spidops = False
    
    benched_tarountula_count = 0
    benched_spidops_count = 0
    benched_mewtwo_count = 0
    
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
        elif p.id == Tarountula:
            active_tarountula = True
        elif p.id == Spidops:
            active_spidops = True

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

    op_active_hp = 9999
    op_active_is_ex = False
    if len(op_state.active) >= 1 and op_state.active[0] is not None:
        op_active = op_state.active[0]
        op_active_hp = op_active.hp
        op_active_card = card_table.get(op_active.id)
        if op_active_card and getattr(op_active_card, 'ex', False):
            op_active_is_ex = True

    # Check if we can attack this turn
    can_attack = False
    if context == SelectContext.MAIN:
        for o in select.option:
            if o.type == OptionType.ATTACK:
                can_attack = True

    # Assign scores
    scores = []
    for idx, o in enumerate(select.option):
        score = 0
        
        if o.type == OptionType.NUMBER:
            score = o.number
            
        elif o.type == OptionType.YES:
            score = 100
            # Context specific for Erasure Ball energy discard selection:
            # If we need extra damage, prioritize YES, otherwise prefer NO/don't discard.
            if hasattr(select, 'message') and "discard" in str(select.message).lower():
                needed_damage = op_active_hp - 160
                if needed_damage <= 0:
                    score = 1  # Low priority, prefer NO
                else:
                    score = 200  # High priority, we need extra damage
                    
        elif o.type == OptionType.NO:
            score = 50
            if hasattr(select, 'message') and "discard" in str(select.message).lower():
                needed_damage = op_active_hp - 160
                if needed_damage <= 0:
                    score = 300  # Prefer not discarding energy if 160 is enough
                    
        elif o.type == OptionType.EVOLVE:
            score = 120000  # Evolving Spidops is extremely high priority
            
        elif o.type == OptionType.RETREAT:
            # Retreat only if active is NOT our primary charged attacker, or if active is low HP
            if active_mewtwo:
                score = -100
            else:
                if benched_mewtwo_count > 0 and benched_energies_count >= 2:
                    score = 80000
                else:
                    score = -10
                    
        elif o.type == OptionType.ABILITY:
            score = 50000
            
        elif o.type == OptionType.ATTACK:
            score = 150000
            if o.attackId == 608:  # Erasure Ball
                score += 10000
            elif o.attackId == 560:  # Rocket Rush
                score += 5000
                
        elif o.type == OptionType.PLAY:
            c = get_card(obs, AreaType.HAND, o.index, my_index)
            if c is not None:
                card_data = card_table.get(c.id)
                if card_data is not None:
                    # Supporters
                    if card_data.cardType == CardType.SUPPORTER:
                        score = 25000
                        if c.id == Rockets_Proton:
                            if field_counts[Tarountula] + field_counts[Mewtwo_ex] < 3:
                                score = 95000
                        elif c.id == Rockets_Ariana:
                            score = 80000
                        elif c.id == Rockets_Giovanni:
                            if benched_mewtwo_count > 0 and benched_energies_count >= 2:
                                score = 90000
                            else:
                                score = 1000
                        elif c.id == Rockets_Petrel:
                            score = 75000
                        elif c.id == Rockets_Archer:
                            score = 70000
                            
                    # Stadiums
                    elif card_data.cardType == CardType.STADIUM:
                        if c.id == Rockets_Factory:
                            if stadium_id != Rockets_Factory:
                                score = 85000
                            else:
                                score = -1
                                
                    # Pokémon placement
                    elif card_data.cardType == CardType.POKEMON:
                        score = 30000
                        if c.id == Tarountula:
                            if field_counts[Tarountula] < 2:
                                score = 96000
                            elif len(my_state.bench) >= 5:
                                score = -1
                        elif c.id == Mewtwo_ex:
                            if field_counts[Mewtwo_ex] < 2:
                                score = 95000
                            elif len(my_state.bench) >= 5:
                                score = -1
                        elif len(my_state.bench) >= 5:
                            score = -1
                            
                    # Items
                    else:
                        score = 15000
                        if c.id == Rockets_Transceiver:
                            score = 92000
                        elif c.id == Ultra_Ball:
                            if field_counts[Tarountula] < 1 or field_counts[Mewtwo_ex] < 1:
                                score = 91000
                            elif field_counts[Spidops] < field_counts[Tarountula]:
                                score = 88000
                            else:
                                score = 5000
                        elif c.id == Bug_Catching_Set:
                            score = 78000
                        elif c.id == Maximum_Belt:
                            if op_active_is_ex and (active_mewtwo or (benched_mewtwo_count > 0 and benched_energies_count >= 2)):
                                score = 93000
                            else:
                                score = 500
                        elif c.id == Energy_Switch:
                            if benched_mewtwo_count > 0 and active_mewtwo is False:
                                score = 85000
                            else:
                                score = 500
                        elif c.id == Night_Stretcher:
                            if discard_counts[Mewtwo_ex] > 0 or discard_counts[Spidops] > 0:
                                score = 79000
                            else:
                                score = 2000
                                
        elif o.type == OptionType.ATTACH or context == SelectContext.ATTACH_FROM:
            card = None
            if o.type == OptionType.ATTACH:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                target_pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                is_active_target = (o.inPlayArea == AreaType.ACTIVE)
            else:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                target_pokemon = None
                is_active_target = False
                
            if card is not None:
                if card.id == Team_Rockets_Energy:
                    score = 45000
                    if target_pokemon is not None:
                        if target_pokemon.id == Mewtwo_ex:
                            score += 15000
                            if not is_active_target:
                                score += 5000
                        elif target_pokemon.id == Spidops:
                            score += 5000
                            
                elif card.id == Basic_Psychic_Energy:
                    score = 40000
                    if target_pokemon is not None:
                        if target_pokemon.id == Mewtwo_ex:
                            score += 10000
                            if not is_active_target:
                                score += 5000
                                
                elif card.id == Basic_Grass_Energy:
                    score = 35000
                    if target_pokemon is not None:
                        if target_pokemon.id == Tarountula or target_pokemon.id == Spidops:
                            score += 15000
                            
                elif card.id == Maximum_Belt:
                    score = 500
                    if target_pokemon is not None and target_pokemon.id == Mewtwo_ex:
                        if op_active_is_ex:
                            score = 86000
                            
                elif card.id == Brave_Bangle:
                    score = 500
                    if target_pokemon is not None and target_pokemon.id == Spidops:
                        if op_active_is_ex:
                            score = 84000
                            
        elif o.type == OptionType.CARD:
            c = get_card(obs, o.area, o.index, o.playerIndex)
            if c is not None:
                if (context == SelectContext.SWITCH or 
                    context == SelectContext.TO_ACTIVE or 
                    context == SelectContext.SETUP_ACTIVE_POKEMON):
                    if c.id == Tarountula:
                        score = 25000
                    elif c.id == Mimikyu:
                        score = 26000
                    elif c.id == Articuno:
                        score = 27000
                    elif c.id == Mewtwo_ex:
                        mewtwo_energies = len(c.energies) if isinstance(c, Pokemon) else 0
                        if mewtwo_energies >= 2:
                            score = 30000
                        else:
                            score = 1000
                            
                elif context == SelectContext.TO_HAND or context == SelectContext.TO_BENCH:
                    if c.id == Mewtwo_ex:
                        score = 50000
                    elif c.id == Spidops:
                        if field_counts[Tarountula] > field_counts[Spidops]:
                            score = 48000
                        else:
                            score = 10000
                    elif c.id == Tarountula:
                        if field_counts[Tarountula] < 2:
                            score = 45000
                        else:
                            score = 12000
                    elif c.id == Team_Rockets_Energy:
                        score = 40000
                    elif c.id == Basic_Psychic_Energy:
                        score = 30000
                    elif c.id == Basic_Grass_Energy:
                        score = 25000
                        
                elif context == SelectContext.DISCARD:
                    if c.id in [Poke_Pad, Sacred_Ash, Basic_Psychic_Energy, Basic_Grass_Energy]:
                        score = 10000
                    else:
                        score = 100

        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc_indices[:select.maxCount]
