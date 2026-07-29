from cg.api import all_card_data, all_attack, CardType

# Global Card Database and Helpers
card_table = {c.cardId: c for c in all_card_data()}
attack_table = {a.attackId: a for a in all_attack()}
evolves_from_set = {c.evolvesFrom for c in all_card_data() if c.evolvesFrom}


def get_card_data(card_id: int):
    return card_table.get(card_id)


def is_defensive_blocker(c):
    if c is None or c.cardType != CardType.POKEMON:
        return False
    # Exclude Mimikyu (434) — Team Rocket's Mimikyu has 60 HP and NO Safeguard ability!
    return c.basic and not c.ex and (c.name not in evolves_from_set) and c.cardId != 434


def can_attack(card_id: int, energy_count: int):
    c = get_card_data(card_id)
    if c is None or c.cardType != CardType.POKEMON:
        return False
    for aid in c.attacks:
        att = attack_table.get(aid)
        if att is not None and energy_count >= len(att.energies):
            return True
    return False


def count_effective_energy_cards(cards):
    if not cards:
        return 0
    total = 0
    for c in cards:
        if c is not None:
            cid = getattr(c, "cardId", getattr(c, "id", -1))
            if cid == 15:  # Team Rocket's Energy provides 2 energies
                total += 2
            else:
                total += 1
    return total


def count_attached_energy(ps):
    count = 0
    if len(ps.active) > 0 and ps.active[0] is not None:
        count += count_effective_energy_cards(ps.active[0].energyCards)
    for poke in ps.bench:
        if poke is not None:
            count += count_effective_energy_cards(poke.energyCards)
    return count


def count_active_energy(ps):
    if len(ps.active) > 0 and ps.active[0] is not None:
        return count_effective_energy_cards(ps.active[0].energyCards)
    return 0


def count_pokemon(ps):
    count = 0
    if len(ps.active) > 0 and ps.active[0] is not None:
        count += 1
    for poke in ps.bench:
        if poke is not None:
            count += 1
    return count
