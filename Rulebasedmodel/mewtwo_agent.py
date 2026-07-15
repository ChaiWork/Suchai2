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

    # Identify if we are close to decking ourselves out (stop drawing)
    near_deckout = (my_state.deckCount <= 3)

    # Identify Game Plan (SETUP, AGGRESSIVE, CONSERVE, TEMPO)
    engine_complete = (field_counts[Spidops] > 0)
    prizes_me = len(my_state.prize)
    prizes_op = len(op_state.prize)
    
    if not engine_complete:
        game_plan = "SETUP"
    elif prizes_me > prizes_op:
        game_plan = "AGGRESSIVE"
    elif prizes_me < prizes_op:
        game_plan = "CONSERVE"
    else:
        game_plan = "TEMPO"

    # Resource tracking: count known copies of critical cards
    DECKLIST_TOTALS = {
        Mewtwo_ex: 2,
        Spidops: 4,
        Tarountula: 4,
        Team_Rockets_Energy: 3,
        Night_Stretcher: 2,
        Ultra_Ball: 4
    }

    active_is_ex = False
    if len(my_state.active) >= 1 and my_state.active[0] is not None:
        active_id = my_state.active[0].id
        if active_id in [Mewtwo_ex, Clefairy_ex]:
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
        
        if o.type == OptionType.END:
            if op_state.deckCount == 0:
                score = 200000  # Instantly end turn to win by opponent deckout
            else:
                score = 0
                
        elif o.type == OptionType.NUMBER:
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
            if op_active_is_immune and active_is_ex:
                score = 180000  # High priority to retreat from immune opponent to Spidops/non-ex
            elif active_mewtwo:
                score = -100
            else:
                if benched_mewtwo_count > 0 and benched_energies_count >= 2:
                    score = 80000
                else:
                    score = -10
                    
        elif o.type == OptionType.ABILITY:
            if near_deckout:
                score = -100  # Avoid drawing when near deck-out
            else:
                score = 89000
            
        elif o.type == OptionType.ATTACK:
            if op_active_is_immune and active_is_ex:
                score = -100  # Wastes turn to attack immune opponent with ex
            else:
                score = 60000  # Base attack score (lower than play/attach/ability to enforce sequencing)
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
                        score = 75000
                        if c.id == Rockets_Proton:
                            if near_deckout:
                                score = -100
                            elif field_counts[Tarountula] + field_counts[Mewtwo_ex] < 3:
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
                            
                    # Stadiums
                    elif card_data.cardType == CardType.STADIUM:
                        if c.id == Rockets_Factory:
                            if stadium_id != Rockets_Factory:
                                score = 95000  # Prioritize Factory above transceiver (95000 > 92000)
                            else:
                                score = -1
                                
                    # Pokémon placement
                    elif card_data.cardType == CardType.POKEMON:
                        score = 30000  # Default basic play score
                        
                        # Bench-Out Safeguard: If bench is empty, prioritize benching any basic
                        is_bench_empty = (len(my_state.bench) == 0)
                        
                        # Heuristic: Bench Order / Setup Priority
                        if c.id == Tarountula:
                            if is_bench_empty:
                                score = 150000  # Absolute priority to prevent bench-out
                            elif field_counts[Tarountula] == 0:
                                score = 96000  # 1st Tarountula is highest priority
                            elif field_counts[Tarountula] == 1:
                                score = 90000  # 2nd Tarountula
                            else:
                                score = 65000  # 3rd+ Tarountula
                        elif c.id == Mewtwo_ex:
                            if is_bench_empty:
                                score = 149000
                            elif field_counts[Mewtwo_ex] < 2:
                                score = 85000
                            else:
                                score = 50000
                        elif c.id == Articuno:
                            if is_bench_empty:
                                score = 148000
                            else:
                                score = 78000
                        elif c.id == Mimikyu:
                            if is_bench_empty:
                                score = 147000
                            else:
                                score = 77000
                        elif c.id == Clefairy_ex:
                            if is_bench_empty:
                                score = 146000
                            else:
                                score = 1000  # Avoid benching Clefairy unless utility is relevant
                            
                        # Bench Capacity Management (Keep at least 1 slot open if possible)
                        if len(my_state.bench) >= 4:
                            if c.id in [Articuno, Mimikyu, Clefairy_ex]:
                                score = 500  # Keep slot open
                                
                        if len(my_state.bench) >= 5:
                            score = -1
                            
                    # Items
                    else:
                        score = 72000
                        if c.id == Rockets_Transceiver:
                            if near_deckout:
                                score = -100
                            else:
                                score = 92000
                        elif c.id == Ultra_Ball:
                            if near_deckout:
                                score = -100
                            # Avoid using Ultra Ball until free search options are exhausted
                            elif field_counts[Tarountula] < 1 or field_counts[Mewtwo_ex] < 1:
                                score = 72000
                            elif field_counts[Spidops] < field_counts[Tarountula]:
                                score = 71000
                            else:
                                score = 5000
                        elif c.id == Bug_Catching_Set:
                            if near_deckout:
                                score = -100
                            else:
                                score = 78000
                        elif c.id == Maximum_Belt:
                            if op_active_is_ex and (active_mewtwo or (benched_mewtwo_count > 0 and benched_energies_count >= 2)):
                                score = 93000
                            else:
                                score = 500
                        elif c.id == Energy_Switch:
                            # Use Energy Switch only if we have Mewtwo ex active and it enables a high-damage attack (reaches 3 energy)
                            mewtwo_active_energy = 0
                            if len(my_state.active) >= 1 and my_state.active[0] is not None and my_state.active[0].id == Mewtwo_ex:
                                mewtwo_active_energy = len(my_state.active[0].energyCards)
                            
                            # If active Mewtwo has 2 energy, Energy Switch can enable the 3rd energy attack!
                            if mewtwo_active_energy == 2 and benched_energies_count >= 1:
                                score = 90000
                            else:
                                score = 1000  # Save for prize-critical swing turn
                        elif c.id == Sacred_Ash:
                            # Boost recovery item if we have targets in discard
                            if discard_counts[Mewtwo_ex] > 0 or discard_counts[Spidops] > 0:
                                score = 70000
                            else:
                                score = 2000
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
                score = 50000  # Default base score
                
                # Heuristic: Avoid attaching energy/resources to low HP/vulnerable targets
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
                        score -= 20000  # Avoid wasting Rocket Energy
                            
                elif card.id == Basic_Psychic_Energy:
                    score = 80000
                    if target_pokemon is not None:
                        if target_pokemon.id == Mewtwo_ex:
                            score += 10000
                            if not is_active_target:
                                score += 5000
                    if is_vulnerable:
                        score -= 10000
                                
                elif card.id == Basic_Grass_Energy:
                    score = 75000
                    if target_pokemon is not None:
                        if target_pokemon.id in [Tarountula, Spidops]:
                            score += 15000
                            # Boost grass energy early during SETUP to get Spidops online
                            if game_plan == "SETUP":
                                score += 10000
                            
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
                card_data = card_table.get(c.id)
                if (context == SelectContext.SWITCH or 
                    context == SelectContext.TO_ACTIVE or 
                    context == SelectContext.SETUP_ACTIVE_POKEMON):
                    if context == SelectContext.SETUP_ACTIVE_POKEMON:
                        # Turn 0 Setup Active choice: prioritize opening basics, ensure all get positive scores
                        if c.id == Tarountula:
                            score = 27000
                        elif c.id == Articuno:
                            score = 26000
                        elif c.id == Mimikyu:
                            score = 25000
                        elif c.id in [Mewtwo_ex, Clefairy_ex]:
                            score = 1000
                    else:
                        # Mid-game promotions and switches
                        if op_active_is_immune and c.id in [Mewtwo_ex, Clefairy_ex]:
                            score = -1000  # Avoid putting ex Pokemon active against Mimikyu
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
                    # Heuristic: Setup Tarountula/Spidops engine first before seeking Mewtwo ex
                    if c.id == Tarountula:
                        if field_counts[Tarountula] == 0:
                            score = 60000  # Highest priority if empty field
                        elif field_counts[Tarountula] < 2:
                            score = 45000
                        else:
                            score = 12000
                    elif c.id == Spidops:
                        if field_counts[Tarountula] > 0 and field_counts[Spidops] == 0:
                            score = 58000  # High priority to get Spidops online
                        elif field_counts[Tarountula] > field_counts[Spidops]:
                            score = 48000
                        else:
                            score = 10000
                    elif c.id == Mewtwo_ex:
                        if field_counts[Tarountula] > 0 and field_counts[Spidops] > 0:
                            score = 55000  # Prioritize Mewtwo ex only when engine is online
                        else:
                            score = 35000  # Lower priority if engine is missing
                    elif c.id == Team_Rockets_Energy:
                        score = 40000
                    elif c.id == Basic_Psychic_Energy:
                        score = 30000
                    elif c.id == Basic_Grass_Energy:
                        score = 25000
                        
                elif context == SelectContext.DISCARD:
                    # Heuristic: Ultra Ball Discard Priorities (Avoid discarding last copies)
                    is_last_critical = (c.id in DECKLIST_TOTALS and (DECKLIST_TOTALS[c.id] - discard_counts[c.id] <= 1))
                    
                    if is_last_critical or c.id == Night_Stretcher:
                        score = -10000  # Never discard critical singletons or recovery
                    elif c.id == Rockets_Factory and (stadium_id == Rockets_Factory or hand_counts[Rockets_Factory] > 1):
                        score = 8000  # Safe to discard duplicate Factory
                    elif card_data is not None and card_data.cardType == CardType.SUPPORTER and hand_counts[c.id] > 1:
                        score = 7000  # Safe to discard duplicate Supporter
                    elif c.id in [Brave_Bangle, Maximum_Belt] and hand_counts[c.id] > 1:
                        score = 6000  # Safe to discard duplicate Tool
                    elif c.id in [Basic_Psychic_Energy, Basic_Grass_Energy]:
                        score = 5000  # Prefer basic energy discard over basics
                    else:
                        score = 100

        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc_indices[:select.maxCount]
