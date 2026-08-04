"""
Deck Profile for Lillie's Clefairy ex + Togekiss Archetype.
"""
from typing import Dict, Any
from src.deck_profiles.base_profile import BaseDeckProfile


class LillieDeckProfile(BaseDeckProfile):
    """Profile for Lillie's Clefairy ex + Togekiss deck."""

    @property
    def deck_name(self) -> str:
        return "LILLIE"

    @property
    def default_deck_csv_path(self) -> str:
        return "decks/lillie_clefairy.csv"

    @property
    def key_card_ids(self) -> Dict[str, Any]:
        return {
            "CLEFAIRY_EX_ID": 272,      # Lillie's Clefairy ex (JTG #56)
            "TOGEKISS_ID": 214,         # Togekiss (SSP #72)
            "TOGEPI_ID": 959,           # Togepi (ASC #80 / SSP #70)
            "TOGETIC_ID": 960,          # Togetic (ASC #81 / SSP #71)
            "SMOOCHUM_ID": 183,         # Smoochum (SSP #75)
            "LATIAS_EX_ID": 184,        # Latias ex (SSP #76)
            "TELEPATHIC_ENERGY_ID": 19, # Telepath Psychic Energy (POR #87)
            "MYSTERY_GARDEN_ID": 1263,  # Mystery Garden (MEG #122)
            "WONDROUS_PATCH_ID": 1146,  # Wondrous Patch (PFL #94)
            "LILLIES_PEARL_ID": 1172,   # Lillie's Pearl (JTG #151)
            "RARE_CANDY_ID": 1079,      # Rare Candy (SVI #191)
            "ULTRA_BALL_ID": 1121,      # Ultra Ball (SVI #196)
            "POKE_PAD_ID": 1152,        # Poké Pad (POR #81)
            "COLRESS_TENACITY_ID": 1194,# Colress's Tenacity (SFA #57)
            "HILDA_ID": 1225,           # Hilda (WHT #84)
            "NIGHT_STRETCHER_ID": 1097, # Night Stretcher (SFA #61)
            "BOSS_ORDERS_ID": 1182,     # Boss's Orders (PAL #172)
            "AIR_BALLOON_ID": 1174,     # Air Balloon (BLK #79)
            "UNFAIR_STAMP_ID": 1080,    # Unfair Stamp (TWM #165)
        }
