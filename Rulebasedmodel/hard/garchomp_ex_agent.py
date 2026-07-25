import os
import sys

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
Rulebasedmodel_Garchomp_ex Rule-based Opponent Agent
Auto-generated from tournament replay 87889092garchomp.json
"""

file_path = os.path.join("decks", "Rulebasedmodel_Garchomp_ex", "deck.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("/kaggle_simulations", "agent", "decks", "Rulebasedmodel_Garchomp_ex", "deck.csv")
if not os.path.exists(file_path):
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path

with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(min(60, len(csv))):
    if csv[i].strip():
        my_deck.append(int(csv[i].strip()))

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

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

    max_count = select.maxCount if hasattr(select, "maxCount") and select.maxCount is not None else 1
    min_count = select.minCount if hasattr(select, "minCount") and select.minCount is not None else 1
    if max_count <= 0:
        max_count = 1

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
                    if getattr(card, 'ex', False) or getattr(card, 'megaEx', False):
                        score += 50
                    elif card.cardType == CardType.POKEMON:
                        score += 20
                elif context in [SelectContext.TO_BENCH, SelectContext.TO_HAND]:
                    if card.cardType == CardType.POKEMON:
                        score += 30
                    elif card.cardType == CardType.SUPPORTER:
                        score += 25
                    elif card.cardType in [CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY]:
                        score += 20
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is not None:
                if card.cardType == CardType.POKEMON:
                    score += 30
                elif card.cardType == CardType.SUPPORTER:
                    score += 25
                elif card.cardType in [CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY]:
                    score += 20
        elif o.type == OptionType.ATTACH:
            score += 40
        elif o.type == OptionType.EVOLVE:
            score += 50
        elif o.type == OptionType.ATTACK:
            score += 100
        elif o.type == OptionType.ABILITY:
            score += 35

        scores.append(score)

    # Sort indices by score descending and return at most max_count options
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n_return = max(min_count, min(max_count, len(scores)))
    return sorted_indices[:n_return]
