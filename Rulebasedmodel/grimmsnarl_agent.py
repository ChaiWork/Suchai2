import os
import sys
from collections import defaultdict

from cg.api import (
    AreaType, CardType, EnergyType, Observation, SelectContext,
    OptionType, Card, Pokemon, all_card_data, to_observation_class
)

"""
Marnie's Grimmsnarl ex / Munkidori / Froslass Deck Rule-based Agent
Extracted from loss replay 87552746.
"""

file_path = os.path.join("decks", "Rulebasedmodel_Grimmsnarl", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "Rulebasedmodel_Grimmsnarl", "deck.csv")
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
Basic_Dark_Energy = 7
Froslass = 104
Munkidori = 112
Marnie_Impidimp = 646
Marnie_Morgrem = 647
Marnie_Grimmsnarl_ex = 648
Snorunt = 860
Rare_Candy = 1079
Unfair_Stamp = 1080
Buddy_Buddy_Poffin = 1086
Night_Stretcher = 1097
Pokegear_30 = 1122
Tool_Scrapper = 1137
Poke_Pad = 1152
Boss_Orders = 1182
Team_Rocket_Petrel = 1219
Lillie_Determination = 1227
Dawn = 1231
Spikemuth_Gym = 1259


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
                    if card.id == Marnie_Grimmsnarl_ex:
                        score += 65
                    elif card.id in [Marnie_Morgrem, Marnie_Impidimp]:
                        score += 45
                    elif card.id == Munkidori:
                        score += 35
                    elif card.id in [Snorunt, Froslass]:
                        score += 25
                elif context in [SelectContext.TO_BENCH, SelectContext.TO_HAND]:
                    if card.id == Marnie_Grimmsnarl_ex:
                        score += 60
                    elif card.id in [Marnie_Impidimp, Munkidori]:
                        score += 50
                    elif card.id in [Marnie_Morgrem, Rare_Candy]:
                        score += 45
                    elif card.id == Basic_Dark_Energy:
                        score += 40
                    elif card.id in [Lillie_Determination, Team_Rocket_Petrel]:
                        score += 35
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is not None:
                if card.id == Marnie_Impidimp:
                    score += 50
                elif card.id in [Marnie_Morgrem, Marnie_Grimmsnarl_ex]:
                    score += 60
                elif card.id == Rare_Candy:
                    score += 55
                elif card.id == Munkidori:
                    score += 45
                elif card.id == Basic_Dark_Energy:
                    score += 40
                elif card.id in [Lillie_Determination, Team_Rocket_Petrel]:
                    score += 35
                elif card.id == Buddy_Buddy_Poffin:
                    score += 42
                elif card.id == Spikemuth_Gym:
                    score += 30
                elif card.id in [Night_Stretcher, Poke_Pad]:
                    score += 25
        elif o.type == OptionType.ABILITY:
            score += 80  # Munkidori Adrena-Brain ability priority
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
