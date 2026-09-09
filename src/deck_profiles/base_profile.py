"""
Base class for Pokémon TCG Deck Profiles.
Defines card IDs, CSV paths, and key card groupings for deck archetypes.
"""
from abc import ABC, abstractmethod
from typing import Dict, Set, Any


class BaseDeckProfile(ABC):
    """Abstract Base Class for all deck profiles."""

    @property
    @abstractmethod
    def deck_name(self) -> str:
        """Name identifier of the deck."""
        pass

    @property
    @abstractmethod
    def key_card_ids(self) -> Dict[str, Any]:
        """Dictionary mapping semantic names to card IDs or sets of IDs."""
        pass

    @property
    @abstractmethod
    def default_deck_csv_path(self) -> str:
        """Path to default CSV definition for this deck."""
        pass
