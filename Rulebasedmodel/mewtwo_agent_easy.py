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
Team Rocket's Mewtwo ex Deck - Easy Version (Punching Bag)
For MCTS AI Agent to learn basic rules of attacking and winning.
"""

# Load deck.csv in the dataset
file_path = os.path.join("decks", "Rulebasedmodel_Mewtwo_Easy", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "Rulebasedmodel_Mewtwo_Easy", "deck.csv")
if not os.path.exists(file_path):
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

def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object."""
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

    scores = []
    for idx, o in enumerate(select.option):
        score = 0
        
        if o.type == OptionType.END:
            score = 1000  # End turn has moderate priority (allows attach/play first, but prevents attack)
            
        elif o.type == OptionType.NUMBER:
            score = o.number
            
        elif o.type == OptionType.YES:
            score = 100
            
        elif o.type == OptionType.NO:
            score = 50
            
        elif o.type == OptionType.EVOLVE:
            score = 1500  # Evolving is simple and high priority
            
        elif o.type == OptionType.RETREAT:
            score = -100  # Never retreat
            
        elif o.type == OptionType.ABILITY:
            score = -100  # Never use abilities
            
        elif o.type == OptionType.ATTACK:
            score = -100  # Never attack (so END at 1000 will be chosen first)
            
        elif o.type == OptionType.PLAY:
            c = get_card(obs, AreaType.HAND, o.index, my_index)
            if c is not None:
                card_data = card_table.get(c.id)
                if card_data is not None:
                    if card_data.cardType == CardType.POKEMON:
                        # Bench basics so we don't bench out
                        score = 2000
                    else:
                        # Don't play items or supporters to keep game state simple
                        score = -10
                else:
                    score = -10
            else:
                score = -10
                
        elif o.type == OptionType.ATTACH or context == SelectContext.ATTACH_FROM:
            score = 1200  # Prioritize attaching energy before ending turn
            
        elif o.type == OptionType.CARD:
            if context in [SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON]:
                # Need to select active pokemon at game start
                score = 2000
            else:
                score = 100
                
        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc_indices[:select.maxCount]
