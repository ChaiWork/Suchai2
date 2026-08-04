from src.deck_profiles.base_profile import BaseDeckProfile
from src.deck_profiles.mewtwo_profile import MewtwoDeckProfile
from src.deck_profiles.lillie_profile import LillieDeckProfile


def get_deck_profile(deck_name: str) -> BaseDeckProfile:
    """Factory function to instantiate deck profile by name."""
    name = deck_name.upper()
    if name == "LILLIE":
        return LillieDeckProfile()
    return MewtwoDeckProfile()
