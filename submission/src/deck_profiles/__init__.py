from src.deck_profiles.base_profile import BaseDeckProfile
from src.deck_profiles.mewtwo_profile import MewtwoDeckProfile


def get_deck_profile(deck_name: str = "MEWTWO") -> BaseDeckProfile:
    """Returns the Mewtwo deck profile."""
    return MewtwoDeckProfile()

