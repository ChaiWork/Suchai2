"""
Active Deck Configuration for Pokémon TCG RL Agent.
Select active deck archetype via environment variable or default configuration.
Supported values: "MEWTWO", "LILLIE", "GRIMMSNARL"
"""
import os

ACTIVE_DECK = os.getenv("ACTIVE_DECK", "LILLIE").upper()


def get_active_deck_name() -> str:
    """Returns the current active deck name (e.g. 'MEWTWO', 'LILLIE')."""
    return ACTIVE_DECK


def set_active_deck(deck_name: str) -> None:
    """Programmatically updates the active deck setting."""
    global ACTIVE_DECK
    ACTIVE_DECK = deck_name.upper()
