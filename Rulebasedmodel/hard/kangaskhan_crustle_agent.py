import os
import sys
from collections import defaultdict

from cg.api import (
    AreaType, CardType, EnergyType, Observation, SelectContext,
    OptionType, Card, Pokemon, all_card_data, to_observation_class
)

"""
Mega Kangaskhan ex / Crustle Deck Rule-based Agent
Extracted from loss replay 87482395.
"""

file_path = os.path.join("decks", "Rulebasedmodel_Kangaskhan_Crustle", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "Rulebasedmodel_Kangaskhan_Crustle", "deck.csv")
if not os.path.exists(file_path):
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path

with open(file_path, "r") as file:
    csv_lines = file.read().split("\n")
my_deck = []
for i in range(min(60, len(csv_lines))):
    if csv_lines[i].strip():
        my_deck.append(int(csv_lines[i].strip()))

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Card Constants
Basic_Grass_Energy = 1
Mist_Energy = 11
Spiky_Energy = 14
Grow_Grass_Energy = 18
Shaymin = 343
Dwebble = 344
Crustle = 345
Mega_Kangaskhan_ex = 756
Buddy_Buddy_Poffin = 1086
Hand_Trimmer = 1087
Pokegear_30 = 1122
Switch_Card = 1123
Jumbo_Ice_Cream = 1147
Hero_Cape = 1159
Boss_Orders = 1182
Xerosic_Machinations = 1197
Hilda = 1225
Lillie_Determination = 1227
Battle_Cage = 1264


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
                    if card.id == Mega_Kangaskhan_ex:
                        score += 60
                    elif card.id == Crustle:
                        score += 50
                    elif card.id == Dwebble:
                        score += 30
                    elif card.id == Shaymin:
                        score += 10
                elif context in [SelectContext.TO_BENCH, SelectContext.TO_HAND]:
                    if card.id == Mega_Kangaskhan_ex:
                        score += 60
                    elif card.id == Crustle:
                        score += 50
                    elif card.id == Dwebble:
                        score += 40
                    elif card.id in [Grow_Grass_Energy, Spiky_Energy, Mist_Energy]:
                        score += 35
                    elif card.id in [Hilda, Lillie_Determination, Xerosic_Machinations]:
                        score += 30
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is not None:
                if card.id == Dwebble:
                    score += 45
                elif card.id == Crustle:
                    score += 50
                elif card.id == Mega_Kangaskhan_ex:
                    score += 60
                elif card.id in [Grow_Grass_Energy, Spiky_Energy, Mist_Energy, Basic_Grass_Energy]:
                    score += 40
                elif card.id == Hand_Trimmer:
                    opp_hand_len = len(state.players[1 - my_index].hand) if (state.players[1 - my_index] and state.players[1 - my_index].hand is not None) else 0
                    if opp_hand_len >= 6:
                        score += 55
                    else:
                        score += 20
                elif card.id == Xerosic_Machinations:
                    score += 50
                elif card.id in [Hilda, Lillie_Determination]:
                    score += 35
                elif card.id == Buddy_Buddy_Poffin:
                    score += 38
                elif card.id == Battle_Cage:
                    score += 25
                elif card.id == Jumbo_Ice_Cream:
                    score += 20
        elif o.type == OptionType.ATTACK:
            score += 100
        elif o.type == OptionType.RETREAT:
            score += 15

        scores.append(score)

    max_score = max(scores) if scores else 0
    best_indices = [i for i, s in enumerate(scores) if s == max_score]
    max_cnt = getattr(select, "maxCount", 1) if select else 1
    if not best_indices:
        return [0]
    return best_indices[:max_cnt]
