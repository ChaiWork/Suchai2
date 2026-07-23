import os
import sys
from collections import defaultdict

from cg.api import (
    AreaType, CardType, EnergyType, Observation, SelectContext,
    OptionType, Card, Pokemon, all_card_data, to_observation_class
)

"""
Hop's Trevenant / Hop's Zacian ex Deck Rule-based Agent
Extracted from loss replay 87566810.
"""

file_path = os.path.join("decks", "Rulebasedmodel_Trevenant", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "Rulebasedmodel_Trevenant", "deck.csv")
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
Mist_Energy = 11
Telepath_Psychic_Energy = 19
Bloodmoon_Ursaluna_ex = 44
Fezandipiti_ex = 140
Latias_ex = 184
Lillies_Clefairy_ex = 272
Hop_Zacian_ex = 299
Hop_Snorlax = 304
Hop_Cramorant = 311
Shaymin = 343
Psyduck = 858
Hop_Phantump = 878
Hop_Trevenant = 879
Unfair_Stamp = 1080
Night_Stretcher = 1097
Hop_Bag = 1115
Ultra_Ball = 1121
Pokegear_30 = 1122
Switch_Card = 1123
Poke_Pad = 1152
Hop_Choice_Band = 1171
Boss_Orders = 1182
Hassel = 1193
Colress_Tenacity = 1194
Judge = 1213
Hilda = 1225
Lillie_Determination = 1227
Postwick = 1255


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
                    if card.id in [Hop_Trevenant, Hop_Zacian_ex]:
                        score += 65
                    elif card.id == Hop_Phantump:
                        score += 45
                    elif card.id in [Bloodmoon_Ursaluna_ex, Fezandipiti_ex, Latias_ex]:
                        score += 35
                elif context in [SelectContext.TO_BENCH, SelectContext.TO_HAND]:
                    if card.id in [Hop_Phantump, Hop_Trevenant, Hop_Zacian_ex]:
                        score += 60
                    elif card.id in [Telepath_Psychic_Energy, Mist_Energy]:
                        score += 50
                    elif card.id in [Hop_Bag, Hop_Choice_Band, Postwick]:
                        score += 45
                    elif card.id in [Lillie_Determination, Hilda, Boss_Orders]:
                        score += 35
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is not None:
                if card.id == Hop_Phantump:
                    score += 50
                elif card.id in [Hop_Trevenant, Hop_Zacian_ex]:
                    score += 65
                elif card.id in [Telepath_Psychic_Energy, Mist_Energy]:
                    score += 45
                elif card.id == Hop_Bag:
                    score += 50
                elif card.id == Hop_Choice_Band:
                    score += 48
                elif card.id in [Lillie_Determination, Hilda]:
                    score += 38
                elif card.id == Postwick:
                    score += 30
                elif card.id in [Ultra_Ball, Poke_Pad, Pokegear_30]:
                    score += 25
        elif o.type == OptionType.ABILITY:
            score += 70
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
