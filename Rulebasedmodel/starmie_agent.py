import os
import sys
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
Mega Starmie ex / Mega Froslass ex / Munkidori Deck Rule-based Agent
Opponent deck from Kaggle match 87286085.
"""

file_path = os.path.join("decks", "Rulebasedmodel_Starmie", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "Rulebasedmodel_Starmie", "deck.csv")
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

Staryu = 1030
Mega_Starmie_ex = 1031
Snorunt = 860
Froslass = 104
Mega_Froslass_ex = 861
Munkidori = 112
Basic_Water_Energy = 3
Basic_Dark_Energy = 7
Buddy_Buddy_Poffin = 1086
Mega_Signal = 1145
Hilda = 1225
Lillie_Determination = 1227
Risky_Ruins = 1260

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
                    if card.id in [Mega_Starmie_ex, Mega_Froslass_ex]:
                        score += 50
                    elif card.id in [Staryu, Snorunt, Munkidori]:
                        score += 20
                elif context in [SelectContext.TO_BENCH, SelectContext.TO_HAND]:
                    if card.id in [Staryu, Mega_Starmie_ex]:
                        score += 40
                    elif card.id in [Snorunt, Mega_Froslass_ex, Froslass]:
                        score += 30
                    elif card.id == Munkidori:
                        score += 25
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is not None:
                score = 100
                if card.id in [Hilda, Lillie_Determination]:
                    score += 500
                elif card.id in [Buddy_Buddy_Poffin, Mega_Signal]:
                    score += 400
                elif card.id in [Mega_Starmie_ex, Mega_Froslass_ex, Froslass]:
                    score += 350
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
