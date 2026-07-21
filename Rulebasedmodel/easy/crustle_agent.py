import os
import sys
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
Crustle / Drednaw Deck Rule-based Agent
Opponent deck from Kaggle match 87285610.
"""

file_path = os.path.join("decks", "Rulebasedmodel_Crustle", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "Rulebasedmodel_Crustle", "deck.csv")
if not os.path.exists(file_path):
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path

with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    if csv[i].strip():
        my_deck.append(int(csv[i].strip()))

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

Dwebble = 344
Crustle = 345
Chewtle = 157
Drednaw = 158
Shaymin = 343
Basic_Grass_Energy = 1
Boss_Orders = 1182
Carmine = 1192
Lillie_Determination = 1227
Buddy_Buddy_Poffin = 1086
Switch_Card = 1123
Poke_Pad = 1152
Hero_Cape = 1159

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

def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                if context in [SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON]:
                    if card.id in [Crustle, Drednaw]:
                        score += 50
                    elif card.id in [Dwebble, Chewtle]:
                        score += 20
                elif context in [SelectContext.TO_BENCH, SelectContext.TO_HAND]:
                    if card.id in [Dwebble, Crustle]:
                        score += 40
                    elif card.id in [Chewtle, Drednaw]:
                        score += 30
                    elif card.id == Shaymin:
                        score += 10
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is not None:
                score = 100
                if card.id in [Carmine, Lillie_Determination]:
                    score += 500
                elif card.id == Buddy_Buddy_Poffin:
                    score += 400
                elif card.id in [Crustle, Drednaw]:
                    score += 300
        elif o.type == OptionType.ATTACH:
            score = 1000
        elif o.type == OptionType.EVOLVE:
            score = 5000
        elif o.type == OptionType.ATTACK:
            score = 100000
        elif o.type == OptionType.END:
            score = -100

        scores.append(score)

    if not scores:
        return []

    max_score = max(scores)
    best_indices = [i for i, s in enumerate(scores) if s == max_score]
    return best_indices[:select.maxCount]
