"""
Active Deck Configuration for Pokémon TCG RL Agent.
Dedicated strictly to Team Rocket Mewtwo ex + Battle Cage.
"""

ACTIVE_DECK = "MEWTWO"


def get_active_deck_name() -> str:
    """Returns the current active deck name ('MEWTWO')."""
    return ACTIVE_DECK


def set_active_deck(deck_name: str) -> None:
    """No-op setter kept for interface compatibility."""
    global ACTIVE_DECK
    ACTIVE_DECK = "MEWTWO"


