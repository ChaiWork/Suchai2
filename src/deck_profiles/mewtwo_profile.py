"""
Deck Profile for Team Rocket Mewtwo ex + Spidops Archetype.
"""
from typing import Dict, Any
from src.deck_profiles.base_profile import BaseDeckProfile


class MewtwoDeckProfile(BaseDeckProfile):
    """Profile for Team Rocket Mewtwo ex deck."""

    @property
    def deck_name(self) -> str:
        return "MEWTWO"

    @property
    def default_deck_csv_path(self) -> str:
        return "deck.csv"

    @property
    def key_card_ids(self) -> Dict[str, Any]:
        return {
            "MEWTWO_EX_ID": 431,
            "SPIDOPS_ID": 401,
            "MIMIKYU_ID": 434,
            "ARTICUNO_ID": 414,
            "TAROUNTULA_ID": 400,
            "TR_FACTORY_ID": 1257,
            "TR_POKEMON_IDS": {400, 401, 414, 431, 434},
            "TR_SUPPORTER_IDS": {1216, 1218, 1219, 1220},
            "HEROS_CAPE_ID": 1159,
            "MAXIMUM_BELT_ID": 1158,
            "BRAVE_BANGLE_ID": 1175,
            "NIGHT_STRETCHER_ID": 1097,
        }
